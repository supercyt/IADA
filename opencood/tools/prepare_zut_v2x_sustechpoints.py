#!/usr/bin/env python3
"""Prepare fused cooperative ZUT point clouds for SUSTechPOINTS annotation.

Input is the synchronized, per-CAV staging dataset created by
``convert_zut_v2x_to_opv2v.py``.  For every paired frame, the rear CAV cloud is
transformed into the front CAV LiDAR frame and merged with the front cloud.

The generated root contains only SUSTechPOINTS scene directories::

    experiment_1/
      pcd/000000.pcd
      label/                 # annotations are saved here by SUSTechPOINTS
      frame_meta.json        # required later for OPV2V mapping

No pseudo-labels from ``fusion_data`` are exported.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

try:
    from opencood.tools.convert_zut_v2x_to_opv2v import write_binary_pcd
except ModuleNotFoundError:  # direct: python opencood/tools/prepare_*.py
    from convert_zut_v2x_to_opv2v import write_binary_pcd


DEFAULT_SOURCE = Path(
    "/home/caoyitong/DataProjects/v2x_datasets/zut_v2x_real/opv2v"
)
DEFAULT_OUTPUT = Path(
    "/home/caoyitong/DataProjects/v2x_datasets/zut_v2x_real/sustechpoints"
)


def pose_to_world(pose: Sequence[float]) -> np.ndarray:
    """OpenCOOD [x,y,z,roll,yaw,pitch] pose to a 4x4 transform."""
    x, y, z, roll, yaw, pitch = [float(value) for value in pose]
    cr, sr = math.cos(math.radians(roll)), math.sin(math.radians(roll))
    cy, sy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    cp, sp = math.cos(math.radians(pitch)), math.sin(math.radians(pitch))
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.array(
        [
            [cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr],
            [sp, -cp * sr, cp * cr],
        ]
    )
    transform[:3, 3] = [x, y, z]
    return transform


def read_binary_pcd(path: Path) -> np.ndarray:
    """Read the binary x/y/z/rgb PCD emitted by the staging converter."""
    with path.open("rb") as file:
        header: dict[str, list[str]] = {}
        while True:
            line = file.readline()
            if not line:
                raise ValueError(f"{path}: incomplete PCD header")
            decoded = line.decode("ascii").strip()
            if decoded and not decoded.startswith("#"):
                parts = decoded.split()
                header[parts[0]] = parts[1:]
            if decoded.startswith("DATA "):
                break
        if header.get("DATA") != ["binary"]:
            raise ValueError(f"{path}: only binary PCD is supported")
        fields = header.get("FIELDS", [])
        if fields != ["x", "y", "z", "rgb"]:
            raise ValueError(f"{path}: expected FIELDS x y z rgb, got {fields}")
        point_count = int(header["POINTS"][0])
        payload = file.read()
    if len(payload) != point_count * 16:
        raise ValueError(f"{path}: payload size does not match POINTS")
    values = np.frombuffer(
        payload,
        dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<f4")],
        count=point_count,
    )
    points = np.empty((point_count, 4), dtype=np.float32)
    points[:, 0] = values["x"]
    points[:, 1] = values["y"]
    points[:, 2] = values["z"]
    rgb = values["rgb"].view(np.uint32)
    points[:, 3] = ((rgb >> 16) & 0xFF).astype(np.float32)
    return points


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    output = points.copy()
    output[:, :3] = (
        transform[:3, :3] @ points[:, :3].astype(np.float64).T
    ).T + transform[:3, 3]
    return output


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        value = yaml.safe_load(file)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected YAML mapping")
    return value


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")
    os.replace(temporary, path)


def prepare_scene(
    source_scene: Path, output_scene: Path, limit: int | None = None
) -> dict[str, Any]:
    front_dir = source_scene / "0"
    rear_dir = source_scene / "1"
    stems = sorted(path.stem for path in front_dir.glob("*.pcd"))
    if not stems or stems != sorted(path.stem for path in rear_dir.glob("*.pcd")):
        raise ValueError(f"{source_scene}: front/rear PCD frames do not match")
    for cav_dir in (front_dir, rear_dir):
        yaml_stems = sorted(path.stem for path in cav_dir.glob("*.yaml"))
        if yaml_stems != stems:
            raise ValueError(f"{cav_dir}: PCD/YAML frames do not match")
    if limit is not None:
        stems = stems[:limit]

    # A moving CAV with a constant pose cannot safely be transformed into the
    # other CAV's coordinate frame.  Refuse to create plausible-looking but
    # geometrically false cooperative annotation data in that case.
    per_cav_metadata = {
        cav: [load_yaml(source_scene / cav / f"{stem}.yaml") for stem in stems]
        for cav in ("0", "1")
    }
    for cav, metadata in per_cav_metadata.items():
        poses = np.asarray([item["lidar_pose"] for item in metadata], dtype=np.float64)
        speeds = np.asarray(
            [float(item.get("ego_speed", 0.0)) for item in metadata],
            dtype=np.float64,
        )
        translation_span = np.ptp(poses[:, :3], axis=0)
        yaw_span = float(np.ptp(np.unwrap(np.radians(poses[:, 4]))))
        if (
            len(metadata) >= 2
            and float(np.max(np.abs(speeds))) > 1.0
            and float(np.linalg.norm(translation_span)) < 0.05
            and yaw_span < math.radians(0.1)
        ):
            raise ValueError(
                f"{source_scene}: CAV {cav} moves (max ego_speed "
                f"{float(np.max(np.abs(speeds))):.3f}) but its fusion_data pose "
                "is frozen; cooperative point-cloud fusion is unsafe"
            )

    if output_scene.exists() and any(output_scene.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty scene: {output_scene}")
    pcd_dir = output_scene / "pcd"
    label_dir = output_scene / "label"
    pcd_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    frame_meta: dict[str, Any] = {
        "coordinate_frame": "front_cav_lidar",
        "rotation_unit_in_sustech_labels": "radian",
        "frames": {},
    }
    point_counts = []
    for index, stem in enumerate(stems):
        front_yaml = per_cav_metadata["0"][index]
        rear_yaml = per_cav_metadata["1"][index]
        front_pose = [float(value) for value in front_yaml["lidar_pose"]]
        rear_pose = [float(value) for value in rear_yaml["lidar_pose"]]
        rear_to_front = np.linalg.solve(
            pose_to_world(front_pose), pose_to_world(rear_pose)
        )

        front_points = read_binary_pcd(front_dir / f"{stem}.pcd")
        rear_points = read_binary_pcd(rear_dir / f"{stem}.pcd")
        fused = np.concatenate(
            [front_points, transform_points(rear_points, rear_to_front)], axis=0
        )
        write_binary_pcd(pcd_dir / f"{stem}.pcd", fused)
        point_counts.append(len(fused))
        frame_meta["frames"][stem] = {
            "front_pose": front_pose,
            "rear_pose": rear_pose,
            "rear_to_front": rear_to_front.tolist(),
            "front_ego_speed": float(front_yaml.get("ego_speed", 0.0)),
            "rear_ego_speed": float(rear_yaml.get("ego_speed", 0.0)),
        }
        if (index + 1) % 25 == 0 or index + 1 == len(stems):
            print(
                f"  {source_scene.name}: {index + 1}/{len(stems)} frames",
                flush=True,
            )

    write_json(output_scene / "frame_meta.json", frame_meta)
    return {
        "frames": len(stems),
        "min_points": min(point_counts),
        "max_points": max(point_counts),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--scene", action="append", help="scene name to export (repeatable)"
    )
    parser.add_argument(
        "--limit", type=int, help="export at most this many frames per scene"
    )
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    source_split = args.source_root / args.split
    scenes = args.scene or sorted(
        path.name for path in source_split.iterdir() if path.is_dir()
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    # SUSTechPOINTS treats every root entry as a scene, so keep the root free
    # of manifests/readmes and store mapping metadata inside each scene.
    for name in scenes:
        print(f"Preparing {name} ...", flush=True)
        summary = prepare_scene(
            source_split / name, args.output_root / name, limit=args.limit
        )
        print(f"  {summary}", flush=True)
    print(f"Done: {args.output_root}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
