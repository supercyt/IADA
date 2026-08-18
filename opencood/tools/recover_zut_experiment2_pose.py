#!/usr/bin/env python3
"""Recover the frozen rear-CAV pose in ZUT experiment 2.

The rear CAV's GPS/attitude fields in experiment 2 are frozen, while its CAN
speed remains live.  This tool builds a one-lap route from the healthy front
CAV, advances the rear CAV along that route using its own speed, and uses
experiment 1 to calibrate the speed-to-route-distance scale.  The first frozen
GPS/attitude sample is used only as an initial route anchor and attitude bias.

The output is deliberately named ``experiment_2_estimated``.  It is suitable
as an annotation intermediate, but it must not be represented as measured
ground-truth localization.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    from opencood.tools import convert_zut_v2x_to_opv2v as converter
    from opencood.tools.prepare_zut_v2x_sustechpoints import prepare_scene
except ModuleNotFoundError:  # direct: python opencood/tools/recover_*.py
    import convert_zut_v2x_to_opv2v as converter
    from prepare_zut_v2x_sustechpoints import prepare_scene


DEFAULT_INPUT = Path("/home/caoyitong/DataProjects/v2x_datasets/zut_v2x_real")
SCENE_NAME = "experiment_2_estimated"


@dataclass
class TimedTrajectory:
    timestamp: np.ndarray
    xyz: np.ndarray
    roll: np.ndarray
    yaw_unwrapped: np.ndarray
    pitch: np.ndarray
    speed: np.ndarray


@dataclass
class PeriodicRoute:
    phase: np.ndarray
    xyz: np.ndarray
    roll: np.ndarray
    yaw_unwrapped: np.ndarray
    pitch: np.ndarray
    length: float
    lap_duration: float
    closure_position_error: float
    closure_yaw_error: float


def normalize_angle(degrees: float | np.ndarray) -> float | np.ndarray:
    return (degrees + 180.0) % 360.0 - 180.0


def percentile_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "median": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def load_fusions(
    bag: converter.Bag,
) -> tuple[np.ndarray, list[converter.FusionData]]:
    refs = bag.fusion_refs()
    return (
        np.asarray([ref.timestamp for ref in refs], dtype=np.float64),
        [bag.fusion(ref.message_id) for ref in refs],
    )


def common_origin(
    streams: Sequence[tuple[np.ndarray, list[converter.FusionData]]],
) -> tuple[float, float, float]:
    fixes = [
        (item.longitude, item.latitude, item.height)
        for _, messages in streams
        for item in messages
        if converter._valid_geodetic(item)
    ]
    if not fixes:
        raise ValueError("no valid geodetic fixes")
    median = np.median(np.asarray(fixes, dtype=np.float64), axis=0)
    return tuple(float(value) for value in median)


def regularize(
    timestamps: np.ndarray,
    messages: Sequence[converter.FusionData],
    origin: tuple[float, float, float],
    period: float = 0.05,
    reconstruct_xy: bool = False,
) -> TimedTrajectory:
    poses = np.asarray(
        [converter.fusion_pose(message, origin) for message in messages],
        dtype=np.float64,
    )
    grid = np.arange(timestamps[0], timestamps[-1] + period / 2.0, period)
    raw_yaw_unwrapped = np.unwrap(np.radians(poses[:, 4]))
    yaw_unwrapped = np.interp(grid, timestamps, raw_yaw_unwrapped)
    speed = np.interp(
        grid,
        timestamps,
        np.asarray([max(0.0, item.carspeed) for item in messages]),
    )

    xyz = np.column_stack(
        [np.interp(grid, timestamps, poses[:, axis]) for axis in range(3)]
    )
    if reconstruct_xy:
        # Fill multi-second missing-fusion intervals with heading/speed
        # dead-reckoning, while exactly anchoring the result back to every
        # available healthy GNSS position.
        dt = np.diff(grid)
        dx = 0.5 * (
            speed[:-1] * np.cos(yaw_unwrapped[:-1])
            + speed[1:] * np.cos(yaw_unwrapped[1:])
        ) * dt
        dy = 0.5 * (
            speed[:-1] * np.sin(yaw_unwrapped[:-1])
            + speed[1:] * np.sin(yaw_unwrapped[1:])
        ) * dt
        odometry = np.column_stack(
            [np.r_[0.0, np.cumsum(dx)], np.r_[0.0, np.cumsum(dy)]]
        )
        raw_odometry = np.column_stack(
            [
                np.interp(timestamps, grid, odometry[:, 0]),
                np.interp(timestamps, grid, odometry[:, 1]),
            ]
        )
        correction = poses[:, :2] - raw_odometry
        xyz[:, :2] = odometry + np.column_stack(
            [
                np.interp(grid, timestamps, correction[:, 0]),
                np.interp(grid, timestamps, correction[:, 1]),
            ]
        )

    return TimedTrajectory(
        timestamp=grid,
        xyz=xyz,
        roll=np.interp(grid, timestamps, poses[:, 3]),
        yaw_unwrapped=yaw_unwrapped,
        pitch=np.interp(grid, timestamps, poses[:, 5]),
        speed=speed,
    )


def integrated_speed(trajectory: TimedTrajectory) -> np.ndarray:
    dt = np.diff(trajectory.timestamp)
    return np.r_[
        0.0,
        np.cumsum(0.5 * (trajectory.speed[:-1] + trajectory.speed[1:]) * dt),
    ]


def build_one_lap_route(trajectory: TimedTrajectory) -> PeriodicRoute:
    speed_distance = integrated_speed(trajectory)
    displacement = np.linalg.norm(
        trajectory.xyz[:, :2] - trajectory.xyz[0, :2], axis=1
    )
    yaw_error = np.abs(
        normalize_angle(
            np.degrees(trajectory.yaw_unwrapped - trajectory.yaw_unwrapped[0])
        )
    )
    candidates = np.flatnonzero(
        (speed_distance > 0.35 * speed_distance[-1])
        & (speed_distance < 0.65 * speed_distance[-1])
    )
    if candidates.size == 0:
        raise ValueError("cannot find a first-lap closure candidate")
    closure = candidates[
        np.argmin(displacement[candidates] + 0.08 * yaw_error[candidates])
    ]

    geometric_distance = np.r_[
        0.0,
        np.cumsum(
            np.linalg.norm(np.diff(trajectory.xyz[: closure + 1, :2], axis=0), axis=1)
        ),
    ]
    if geometric_distance[-1] < 100.0:
        raise ValueError("detected route is implausibly short")
    phase = np.arange(0.0, geometric_distance[-1], 0.05)

    def interpolate(values: np.ndarray) -> np.ndarray:
        return np.interp(phase, geometric_distance, values[: closure + 1])

    return PeriodicRoute(
        phase=phase,
        xyz=np.column_stack(
            [interpolate(trajectory.xyz[:, axis]) for axis in range(3)]
        ),
        roll=interpolate(trajectory.roll),
        yaw_unwrapped=interpolate(trajectory.yaw_unwrapped),
        pitch=interpolate(trajectory.pitch),
        length=float(geometric_distance[-1]),
        lap_duration=float(
            trajectory.timestamp[closure] - trajectory.timestamp[0]
        ),
        closure_position_error=float(displacement[closure]),
        closure_yaw_error=float(yaw_error[closure]),
    )


def interpolate_periodic(route: PeriodicRoute, phase: np.ndarray) -> np.ndarray:
    query = np.mod(phase, route.length)
    route_phase = np.r_[route.phase, route.length]
    xyz = np.vstack([route.xyz, route.xyz[0]])
    roll = np.r_[route.roll, route.roll[0]]
    pitch = np.r_[route.pitch, route.pitch[0]]

    direction = 1.0 if route.yaw_unwrapped[-1] >= route.yaw_unwrapped[0] else -1.0
    yaw = np.r_[route.yaw_unwrapped, route.yaw_unwrapped[0] + direction * 2.0 * math.pi]
    return np.column_stack(
        [
            np.interp(query, route_phase, xyz[:, 0]),
            np.interp(query, route_phase, xyz[:, 1]),
            np.interp(query, route_phase, xyz[:, 2]),
            np.interp(query, route_phase, roll),
            np.degrees(np.interp(query, route_phase, yaw)),
            np.interp(query, route_phase, pitch),
        ]
    )


def project_pose_to_route(route: PeriodicRoute, pose: np.ndarray) -> tuple[float, float, float]:
    position_error = np.linalg.norm(route.xyz[:, :2] - pose[:2], axis=1)
    yaw_error = np.abs(
        normalize_angle(np.degrees(route.yaw_unwrapped) - float(pose[4]))
    )
    index = int(np.argmin(position_error + 0.02 * yaw_error))
    return (
        float(route.phase[index]),
        float(position_error[index]),
        float(yaw_error[index]),
    )


def fit_speed_scale(
    route: PeriodicRoute,
    rear: TimedTrajectory,
    initial_phase: float,
    mask: np.ndarray | None = None,
) -> float:
    distance = integrated_speed(rear)
    if mask is None:
        mask = np.ones(len(distance), dtype=bool)
    best = (float("inf"), 1.0)
    for scale in np.arange(0.94, 1.061, 0.001):
        estimate = interpolate_periodic(route, initial_phase + scale * distance)
        error = np.linalg.norm(estimate[mask, :2] - rear.xyz[mask, :2], axis=1)
        keep = max(1, int(0.8 * len(error)))
        score = float(np.mean(np.sort(error)[:keep]))
        if score < best[0]:
            best = (score, float(scale))
    return best[1]


def evaluate_model(
    route: PeriodicRoute,
    rear: TimedTrajectory,
    initial_phase: float,
    speed_scale: float,
    mask: np.ndarray | None = None,
) -> dict[str, Any]:
    if mask is None:
        mask = np.ones(len(rear.timestamp), dtype=bool)
    estimate = interpolate_periodic(
        route, initial_phase + speed_scale * integrated_speed(rear)
    )
    position_error = np.linalg.norm(estimate[mask, :2] - rear.xyz[mask, :2], axis=1)
    yaw_error = np.abs(normalize_angle(estimate[mask, 4] - np.degrees(rear.yaw_unwrapped[mask])))
    return {
        "samples": int(np.sum(mask)),
        "position_error_m": percentile_summary(position_error),
        "yaw_error_deg": percentile_summary(yaw_error),
    }


def learn_from_experiment_1(input_root: Path) -> tuple[float, dict[str, Any]]:
    bags = {
        role: converter.Bag(input_root / path)
        for role, path in converter.EXPERIMENTS["experiment_1"].items()
    }
    try:
        streams = {role: load_fusions(bag) for role, bag in bags.items()}
        origin = common_origin(list(streams.values()))
        front = regularize(*streams["front"], origin, reconstruct_xy=True)
        rear = regularize(*streams["rear"], origin)
        route = build_one_lap_route(front)
        rear_initial_pose = np.array(
            [
                rear.xyz[0, 0], rear.xyz[0, 1], rear.xyz[0, 2], rear.roll[0],
                np.degrees(rear.yaw_unwrapped[0]), rear.pitch[0],
            ]
        )
        initial_phase, projection_error, projection_yaw_error = project_pose_to_route(
            route, rear_initial_pose
        )
        elapsed = rear.timestamp - rear.timestamp[0]
        first_lap = elapsed <= route.lap_duration
        holdout = ~first_lap
        first_lap_scale = fit_speed_scale(route, rear, initial_phase, first_lap)
        final_scale = fit_speed_scale(route, rear, initial_phase)
        report = {
            "route_length_m": route.length,
            "lap_duration_s": route.lap_duration,
            "route_closure_position_error_m": route.closure_position_error,
            "route_closure_yaw_error_deg": route.closure_yaw_error,
            "initial_projection_error_m": projection_error,
            "initial_projection_yaw_error_deg": projection_yaw_error,
            "speed_scale_fit_first_lap": first_lap_scale,
            "speed_scale_fit_all": final_scale,
            "first_lap_fit": evaluate_model(
                route, rear, initial_phase, first_lap_scale, first_lap
            ),
            "second_lap_holdout": evaluate_model(
                route, rear, initial_phase, first_lap_scale, holdout
            ),
            "all_data_final_fit": evaluate_model(
                route, rear, initial_phase, final_scale
            ),
        }
        return final_scale, report
    finally:
        for bag in bags.values():
            bag.close()


def build_experiment_2_estimator(
    state: dict[str, Any], speed_scale: float, expected_laps: float = 2.0
) -> tuple[Any, dict[str, Any]]:
    bags: dict[str, converter.Bag] = state["bags"]
    streams = {role: load_fusions(bag) for role, bag in bags.items()}
    origin = state["origin"]
    front = regularize(*streams["front"], origin, reconstruct_xy=True)
    rear = regularize(*streams["rear"], origin)
    route = build_one_lap_route(front)

    rear_initial_pose = np.array(
        [
            rear.xyz[0, 0], rear.xyz[0, 1], rear.xyz[0, 2], rear.roll[0],
            np.degrees(rear.yaw_unwrapped[0]), rear.pitch[0],
        ]
    )
    initial_phase, projection_error, projection_yaw_error = project_pose_to_route(
        route, rear_initial_pose
    )
    route_initial = interpolate_periodic(route, np.asarray([initial_phase]))[0]
    attitude_bias = rear_initial_pose[[3, 4, 5]] - route_initial[[3, 4, 5]]
    height_bias = rear_initial_pose[2] - route_initial[2]
    attitude_bias[1] = normalize_angle(attitude_bias[1])
    rear_distance = integrated_speed(rear)
    closure_scale = expected_laps * route.length / float(rear_distance[-1])
    # Both experiments consist of two closed laps.  The experiment-1 scale
    # independently validates the CAN odometry scale; the experiment-2 loop
    # closure supplies the more precise scale for this particular run.
    applied_speed_scale = closure_scale

    def estimate(timestamp: float) -> list[float]:
        distance = float(np.interp(timestamp, rear.timestamp, rear_distance))
        pose = interpolate_periodic(
            route, np.asarray([initial_phase + applied_speed_scale * distance])
        )[0]
        pose[2] += height_bias
        pose[3] += attitude_bias[0]
        pose[4] = normalize_angle(pose[4] + attitude_bias[1])
        pose[5] += attitude_bias[2]
        return [float(value) for value in pose]

    report = {
        "route_length_m": route.length,
        "lap_duration_s": route.lap_duration,
        "route_closure_position_error_m": route.closure_position_error,
        "route_closure_yaw_error_deg": route.closure_yaw_error,
        "initial_projection_error_m": projection_error,
        "initial_projection_yaw_error_deg": projection_yaw_error,
        "speed_scale_from_experiment_1": speed_scale,
        "speed_scale_from_experiment_2_loop_closure": closure_scale,
        "applied_speed_scale": applied_speed_scale,
        "expected_laps": expected_laps,
        "rear_integrated_distance_m": float(rear_distance[-1]),
        "estimated_laps": float(
            applied_speed_scale * rear_distance[-1] / route.length
        ),
        "initial_attitude_bias_deg": {
            "roll": float(attitude_bias[0]),
            "yaw": float(attitude_bias[1]),
            "pitch": float(attitude_bias[2]),
        },
    }
    return estimate, report


def export_estimated_scene(
    state: dict[str, Any], estimate_pose: Any, scenario_dir: Path
) -> list[dict[str, Any]]:
    if scenario_dir.exists() and any(scenario_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty scene: {scenario_dir}")
    cav_dirs = {"front": scenario_dir / "0", "rear": scenario_dir / "1"}
    for directory in cav_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    bags: dict[str, converter.Bag] = state["bags"]
    manifest = []
    for frame_index, record in enumerate(state["frames"]):
        stem = f"{frame_index:06d}"
        frame_info: dict[str, Any] = {
            "frame": stem,
            "sync_delta_ms": record["match"].delta_seconds * 1000.0,
            "cavs": {},
        }
        for role in ("front", "rear"):
            forward_ref = record["forward"][role]
            associations = record["associated"][role]
            fusion_ref = associations["fusion"][0]
            fusion = bags[role].fusion(fusion_ref.message_id)
            if role == "front":
                pose = converter.fusion_pose(fusion, state["origin"])
                pose_source = "measured_fusion_data"
            else:
                pose = estimate_pose(fusion.timestamp)
                pose_source = "estimated_front_route_plus_rear_speed"

            pointclouds = [bags[role].pointcloud(forward_ref.message_id)]
            used_topics = [converter.FORWARD_TOPIC]
            for name, topic in (
                ("left", converter.LEFT_TOPIC), ("right", converter.RIGHT_TOPIC)
            ):
                item = associations[name]
                if item is not None:
                    pointclouds.append(bags[role].pointcloud(item[0].message_id))
                    used_topics.append(topic)
            merged = np.concatenate(pointclouds, axis=0)
            converter.write_binary_pcd(cav_dirs[role] / f"{stem}.pcd", merged)
            converter.write_yaml(
                cav_dirs[role] / f"{stem}.yaml",
                converter.fusion_metadata(fusion, pose),
            )
            frame_info["cavs"][role] = {
                "cav_id": 0 if role == "front" else 1,
                "frame_timestamp": forward_ref.timestamp,
                "fusion_timestamp": fusion.timestamp,
                "point_count": int(len(merged)),
                "included_lidar_topics": used_topics,
                "pose_source": pose_source,
            }
        manifest.append(frame_info)
        if (frame_index + 1) % 25 == 0 or frame_index + 1 == len(state["frames"]):
            print(f"  export: {frame_index + 1}/{len(state['frames'])}", flush=True)

    converter.write_yaml(
        scenario_dir / "data_protocol.yaml",
        {
            "dataset": "ZUT V2X real",
            "format": "OPV2V-compatible annotation intermediate",
            "scenario": SCENE_NAME,
            "cavs": {0: "front", 1: "rear"},
            "rear_pose": "estimated from front route and rear CAN speed; not ground truth",
            "labels": "unlabelled; vehicles is empty",
        },
    )
    return manifest


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")
    os.replace(temporary, path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--output-root", type=Path,
        help="staging root (default: INPUT_ROOT/opv2v_pose_estimated)",
    )
    parser.add_argument(
        "--sustech-output-root", type=Path,
        help="SUSTechPOINTS root (default: INPUT_ROOT/sustechpoints_pose_estimated)",
    )
    parser.add_argument("--sync-ms", type=float, default=10.0)
    parser.add_argument("--max-association-ms", type=float, default=100.0)
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args(argv)
    if args.output_root is None:
        args.output_root = args.input_root / "opv2v_pose_estimated"
    if args.sustech_output_root is None:
        args.sustech_output_root = args.input_root / "sustechpoints_pose_estimated"
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    speed_scale, calibration = learn_from_experiment_1(args.input_root)
    report, state = converter.inspect_experiment(
        args.input_root,
        "experiment_2",
        converter.EXPERIMENTS["experiment_2"],
        args.sync_ms / 1000.0,
        args.max_association_ms / 1000.0,
    )
    try:
        estimate_pose, recovery = build_experiment_2_estimator(state, speed_scale)
        document = {
            "method": "front periodic route + rear CAN speed integration",
            "warning": "rear poses are estimates, not measured ground truth",
            "experiment_1_calibration": calibration,
            "experiment_2_recovery": recovery,
            "experiment_2_extraction": report,
        }
        if args.analyze_only:
            print(json.dumps(document, ensure_ascii=False, indent=2))
            return 0

        scenario_dir = args.output_root / "train" / SCENE_NAME
        frames = export_estimated_scene(state, estimate_pose, scenario_dir)
        document["frames"] = frames
        document["output_scene"] = str(scenario_dir)
        write_json(args.output_root / "recovery_report.json", document)
        print("Preparing SUSTechPOINTS scene ...", flush=True)
        sustech_scene = args.sustech_output_root / SCENE_NAME
        summary = prepare_scene(scenario_dir, sustech_scene)
        document["sustech_scene"] = str(sustech_scene)
        document["sustech_summary"] = summary
        write_json(scenario_dir / "recovery_report.json", document)
        print(json.dumps({"recovery": recovery, "sustech": summary}, indent=2))
        print(f"Done: {sustech_scene}")
        return 0
    finally:
        for bag in state["bags"].values():
            bag.close()


if __name__ == "__main__":
    sys.exit(main())
