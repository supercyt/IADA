#!/usr/bin/env python3
"""Map completed SUSTechPOINTS labels to a clean OPV2V dataset.

Only standard OPV2V metadata is written: pose, speed and ``vehicles``.  Raw
``fusion_data`` detector output and other custom fields are intentionally not
copied.  SUSTechPOINTS boxes are expected in the fused front-CAV frame produced
by ``prepare_zut_v2x_sustechpoints.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

try:
    from opencood.tools.prepare_zut_v2x_sustechpoints import pose_to_world
except ModuleNotFoundError:  # direct: python opencood/tools/convert_*.py
    from prepare_zut_v2x_sustechpoints import pose_to_world


DEFAULT_ANNOTATION = Path(
    "/home/caoyitong/DataProjects/v2x_datasets/zut_v2x_real/sustechpoints"
)
DEFAULT_SOURCE = Path(
    "/home/caoyitong/DataProjects/v2x_datasets/zut_v2x_real/opv2v"
)
DEFAULT_OUTPUT = Path(
    "/home/caoyitong/DataProjects/v2x_datasets/zut_v2x_real/opv2v_labeled"
)
DEFAULT_VEHICLE_TYPES = ("Car", "Van", "Truck", "Bus")


class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data: Any) -> bool:
        return True


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        value = yaml.safe_load(file)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected YAML mapping")
    return value


def write_yaml(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        yaml.dump(
            value,
            file,
            Dumper=NoAliasDumper,
            allow_unicode=True,
            sort_keys=False,
        )
    os.replace(temporary, path)


def link_or_copy(source: Path, target: Path) -> None:
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def sustech_rotation(rotation: dict[str, Any]) -> np.ndarray:
    """SUSTechPOINTS XYZ Euler radians to a 3x3 rotation matrix."""
    rx = float(rotation.get("x", 0.0))
    ry = float(rotation.get("y", 0.0))
    rz = float(rotation.get("z", 0.0))
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    rotation_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    rotation_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rotation_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rotation_z @ rotation_y @ rotation_x


def transform_to_opv2v_angles(transform: np.ndarray) -> list[float]:
    rotation = transform[:3, :3]
    yaw = math.degrees(math.atan2(rotation[1, 0], rotation[0, 0]))
    roll = math.degrees(math.atan2(-rotation[2, 1], rotation[2, 2]))
    pitch = math.degrees(
        math.atan2(rotation[2, 0], math.hypot(rotation[2, 1], rotation[2, 2]))
    )
    return [roll, yaw, pitch]


def annotation_to_vehicle(
    annotation: dict[str, Any], front_to_world: np.ndarray
) -> tuple[str, dict[str, Any]]:
    try:
        psr = annotation["psr"]
        position = psr["position"]
        scale = psr["scale"]
        rotation = psr["rotation"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"invalid SUSTechPOINTS object: {annotation}") from error

    local = np.eye(4, dtype=np.float64)
    local[:3, :3] = sustech_rotation(rotation)
    local[:3, 3] = [
        float(position["x"]),
        float(position["y"]),
        float(position["z"]),
    ]
    world = front_to_world @ local
    dimensions = [float(scale[name]) for name in ("x", "y", "z")]
    if any(not math.isfinite(value) or value <= 0 for value in dimensions):
        raise ValueError(f"invalid box scale: {scale}")
    object_id = str(annotation.get("obj_id", "")).strip()
    if not object_id:
        raise ValueError("every annotation must have a stable obj_id")
    vehicle = {
        "angle": transform_to_opv2v_angles(world),
        "center": [0.0, 0.0, 0.0],
        "extent": [value / 2.0 for value in dimensions],
        "location": [float(value) for value in world[:3, 3]],
        "speed": 0.0,
    }
    return object_id, vehicle


def convert_scene(
    annotation_scene: Path,
    source_scene: Path,
    output_scene: Path,
    vehicle_types: set[str],
    allow_missing_labels: bool,
) -> dict[str, int]:
    frame_meta = load_json(annotation_scene / "frame_meta.json")
    frames: dict[str, Any] = frame_meta["frames"]
    if output_scene.exists() and any(output_scene.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty scene: {output_scene}")
    cav_dirs = {cav: output_scene / str(cav) for cav in (0, 1)}
    for directory in cav_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    annotated_frames = 0
    object_count = 0
    for index, (stem, meta) in enumerate(sorted(frames.items())):
        label_path = annotation_scene / "label" / f"{stem}.json"
        if not label_path.exists():
            if not allow_missing_labels:
                raise FileNotFoundError(
                    f"missing annotation {label_path}; save [] for a reviewed empty frame"
                )
            annotations = []
        else:
            annotations = load_json(label_path)
            annotated_frames += 1
        if not isinstance(annotations, list):
            raise ValueError(f"{label_path}: annotation root must be a JSON array")

        front_to_world = pose_to_world(meta["front_pose"])
        vehicles: dict[str, Any] = {}
        for annotation in annotations:
            object_type = str(annotation.get("obj_type", ""))
            if object_type not in vehicle_types:
                continue
            object_id, vehicle = annotation_to_vehicle(annotation, front_to_world)
            if object_id in vehicles:
                raise ValueError(f"{label_path}: duplicate obj_id {object_id!r}")
            vehicles[object_id] = vehicle
        object_count += len(vehicles)

        for cav, role in ((0, "front"), (1, "rear")):
            source_cav = source_scene / str(cav)
            source_yaml = load_yaml(source_cav / f"{stem}.yaml")
            pose = [float(value) for value in source_yaml["lidar_pose"]]
            clean = {
                "lidar_pose": pose,
                "lidar_pose_clean": pose.copy(),
                "true_ego_pos": pose.copy(),
                "ego_speed": float(meta[f"{role}_ego_speed"]),
                "vehicles": vehicles,
            }
            write_yaml(cav_dirs[cav] / f"{stem}.yaml", clean)
            link_or_copy(
                source_cav / f"{stem}.pcd", cav_dirs[cav] / f"{stem}.pcd"
            )
        if (index + 1) % 25 == 0 or index + 1 == len(frames):
            print(
                f"  {annotation_scene.name}: {index + 1}/{len(frames)} frames",
                flush=True,
            )

    protocol = source_scene / "data_protocol.yaml"
    if protocol.exists():
        shutil.copy2(protocol, output_scene / protocol.name)
    return {
        "frames": len(frames),
        "annotated_frames": annotated_frames,
        "vehicle_boxes": object_count,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-root", type=Path, default=DEFAULT_ANNOTATION)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--vehicle-types",
        default=",".join(DEFAULT_VEHICLE_TYPES),
        help="comma-separated SUSTechPOINTS types to map into OPV2V vehicles",
    )
    parser.add_argument(
        "--allow-missing-labels",
        action="store_true",
        help="treat missing label JSON files as reviewed empty frames",
    )
    parser.add_argument("--scene", action="append", help="scene to convert")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    vehicle_types = {
        value.strip() for value in args.vehicle_types.split(",") if value.strip()
    }
    if not vehicle_types:
        raise ValueError("--vehicle-types cannot be empty")
    scenes = args.scene or sorted(
        path.name for path in args.annotation_root.iterdir() if path.is_dir()
    )
    output_split = args.output_root / args.split
    output_split.mkdir(parents=True, exist_ok=True)
    for name in scenes:
        print(f"Converting {name} ...", flush=True)
        summary = convert_scene(
            args.annotation_root / name,
            args.source_root / args.split / name,
            output_split / name,
            vehicle_types,
            args.allow_missing_labels,
        )
        print(f"  {summary}", flush=True)
    print(f"Done: {args.output_root}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
