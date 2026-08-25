#!/usr/bin/env python3
"""Run OpenCOOD on unlabeled ZUT test data and export SUSTechPOINTS labels."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import inference_utils, train_utils
from opencood.utils import box_utils
from opencood.utils.transformation_utils import x1_to_x2


DEFAULT_TEST = Path("/home/caoyitong/DataProjects/v2x_datasets/ZUT_OPV2V/test")
DEFAULT_SOURCE = Path(
    "/home/caoyitong/DataProjects/v2x_datasets/zut_v2x_real/"
    "sustechpoints_resynced/experiment_2")
DEFAULT_OUTPUT = Path(
    "/home/caoyitong/DataProjects/v2x_datasets/zut_v2x_real/"
    "sustechpoints_prelabel/experiment_2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir", required=True, type=Path)
    parser.add_argument("--test_dir", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--source_scene", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output_scene", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fusion_method", default="intermediate",
                        choices=("intermediate", "early", "late", "no", "single"))
    parser.add_argument("--checkpoint_epoch", type=int)
    parser.add_argument("--score_threshold", type=float, default=0.30)
    parser.add_argument("--tracking_distance", type=float, default=3.5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def prepare_output(output: Path, overwrite: bool) -> None:
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"output exists: {output}; use --overwrite")
        resolved = output.resolve()
        if resolved == Path("/") or len(resolved.parts) < 4:
            raise ValueError(f"refusing unsafe output deletion: {resolved}")
        shutil.rmtree(output)
    (output / "pcd").mkdir(parents=True)
    (output / "label").mkdir()


def sample_identity(dataset, index: int) -> tuple[str, str]:
    scenario_index = 0
    for candidate, end in enumerate(dataset.len_record):
        if index < end:
            scenario_index = candidate
            break
    previous_end = 0 if scenario_index == 0 else dataset.len_record[scenario_index - 1]
    timestamp_index = index - previous_end
    database = dataset.scenario_database[scenario_index]
    timestamp = dataset.return_timestamp_key(database, timestamp_index)
    scenario = Path(dataset.scenario_folders[scenario_index]).name
    return scenario, timestamp


def run_inference(method: str, batch_data, model, dataset):
    if method in {"intermediate", "early"}:
        # Experiment 2 is intentionally unlabeled.  The generic OpenCOOD
        # helper always calls generate_gt_bbx(), which assumes at least one
        # GT box and crashes on empty labels.  Prelabeling only needs decoded
        # predictions, so bypass GT generation entirely.
        output_dict = {"ego": model(batch_data["ego"])}
        pred_box, pred_score = dataset.post_processor.post_process(
            batch_data, output_dict
        )
        return {"pred_box_tensor": pred_box, "pred_score": pred_score}
    if method == "late":
        return inference_utils.inference_late_fusion(batch_data, model, dataset)
    if method == "single":
        return inference_utils.inference_no_fusion(
            batch_data, model, dataset, single_gt=True)
    return inference_utils.inference_no_fusion(batch_data, model, dataset)


def tensor_numpy(value) -> np.ndarray:
    if value is None:
        return np.empty((0,), dtype=np.float32)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def prediction_to_front_transform(batch_data) -> tuple[np.ndarray, str]:
    """Return the transform from the selected ego to front-CAV coordinates."""
    cav_ids = [str(value) for value in batch_data["ego"]["cav_id_list"]]
    if not cav_ids or cav_ids[0] not in {"0", "1"}:
        raise ValueError(f"unexpected CAV order: {cav_ids}")
    if "0" not in cav_ids:
        raise ValueError("front CAV 0 is required for SUSTechPOINTS export")
    poses = tensor_numpy(batch_data["ego"]["lidar_pose_clean"]).reshape(-1, 6)
    ego_id = cav_ids[0]
    transform = x1_to_x2(poses[0], poses[cav_ids.index("0")])
    return transform, ego_id


def angle_difference(first: float, second: float) -> float:
    # Cars are 180-degree symmetric for association.
    delta = abs((first - second + math.pi / 2) % math.pi - math.pi / 2)
    return float(delta)


def associate_tracks(frames: list[dict], next_track_id: int,
                     maximum_distance: float) -> int:
    active: dict[int, dict] = {}
    for frame_index, frame in enumerate(frames):
        boxes = frame["boxes"]
        count = len(boxes)
        assigned = [-1] * count
        track_ids = [track_id for track_id, value in active.items()
                     if frame_index - value["last"] <= 2]
        if count and track_ids:
            cost = np.full((len(track_ids), count), 1e6, dtype=np.float64)
            for row, track_id in enumerate(track_ids):
                previous = active[track_id]
                for column, box in enumerate(boxes):
                    distance = float(np.linalg.norm(box[:2] - previous["box"][:2]))
                    if distance > maximum_distance:
                        continue
                    size_delta = float(np.mean(np.abs(
                        np.log(np.maximum(box[3:6], 0.1) /
                               np.maximum(previous["box"][3:6], 0.1)))))
                    heading_delta = angle_difference(float(box[6]),
                                                     float(previous["box"][6]))
                    cost[row, column] = distance + 0.8 * size_delta + 0.3 * heading_delta
            rows, columns = linear_sum_assignment(cost)
            for row, column in zip(rows, columns):
                if cost[row, column] < 1e5:
                    assigned[column] = track_ids[row]
        for box_index, box in enumerate(boxes):
            if assigned[box_index] < 0:
                assigned[box_index] = next_track_id
                next_track_id += 1
            active[assigned[box_index]] = {"box": box.copy(), "last": frame_index}
        frame["track_ids"] = assigned
    return next_track_id


def prediction_to_sustech(box: np.ndarray, score: float, track_id: int) -> dict:
    return {
        # SUSTechPOINTS parses IDs numerically when calculating the next
        # automatic object number.  Non-numeric prefixes poison maxId as NaN.
        "obj_id": str(track_id),
        "obj_type": "Car",
        "psr": {
            "position": {"x": float(box[0]), "y": float(box[1]),
                         "z": float(box[2])},
            "rotation": {"x": 0.0, "y": 0.0, "z": float(box[6])},
            "scale": {"x": float(box[3]), "y": float(box[4]),
                      "z": float(box[5])},
        },
        "auto_score": float(score),
    }


def hardlink_or_copy(source: Path, target: Path) -> None:
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def main() -> int:
    args = parse_args()
    prepare_output(args.output_scene, args.overwrite)

    # load_yaml(None, opt) expects model_dir on the argparse namespace.
    hypes = yaml_utils.load_yaml(None, args)
    hypes["validate_dir"] = str(args.test_dir)
    hypes["test_dir"] = str(args.test_dir)
    if "noise_setting" in hypes:
        hypes["noise_setting"]["add_noise"] = False

    dataset = build_dataset(hypes, visualize=True, train=False)
    loader = DataLoader(dataset, batch_size=1, num_workers=args.num_workers,
                        collate_fn=dataset.collate_batch_test, shuffle=False,
                        pin_memory=False, drop_last=False)
    model = train_utils.create_model(hypes)
    _, model = train_utils.load_saved_model(
        str(args.model_dir), model, checkpoint_epoch=args.checkpoint_epoch)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    predictions: dict[str, list[dict]] = defaultdict(list)
    with torch.no_grad():
        for index, batch_data in enumerate(loader):
            scenario, timestamp = sample_identity(dataset, index)
            if batch_data is None:
                predictions[scenario].append(
                    {"timestamp": timestamp, "boxes": np.empty((0, 7)),
                     "scores": np.empty((0,))})
                continue
            batch_data = train_utils.to_device(batch_data, device)
            ego_to_front, ego_id = prediction_to_front_transform(batch_data)
            result = run_inference(args.fusion_method, batch_data, model, dataset)
            corners = tensor_numpy(result["pred_box_tensor"])
            scores = tensor_numpy(result["pred_score"]).reshape(-1)
            if corners.size:
                corners = box_utils.project_box3d(corners, ego_to_front)
                boxes = box_utils.corner_to_center(corners, order="lwh")
                mask = scores >= args.score_threshold
                boxes, scores = boxes[mask], scores[mask]
            else:
                boxes = np.empty((0, 7), dtype=np.float32)
                scores = np.empty((0,), dtype=np.float32)
            predictions[scenario].append(
                {"timestamp": timestamp, "boxes": boxes, "scores": scores,
                 "ego_id": ego_id})
            if (index + 1) % 25 == 0 or index + 1 == len(dataset):
                print(f"inference {index + 1}/{len(dataset)}", flush=True)

    next_track_id = 1
    frame_count = 0
    box_count = 0
    exported_frames: set[str] = set()
    report_frames = []
    for scenario in sorted(predictions):
        frames = predictions[scenario]
        next_track_id = associate_tracks(frames, next_track_id,
                                         args.tracking_distance)
        for frame in frames:
            timestamp = frame["timestamp"]
            if timestamp in exported_frames:
                raise ValueError(f"duplicate source frame in test split: {timestamp}")
            exported_frames.add(timestamp)
            annotations = [prediction_to_sustech(box, score, track_id)
                           for box, score, track_id in zip(
                               frame["boxes"], frame["scores"], frame["track_ids"])]
            label_path = args.output_scene / "label" / f"{timestamp}.json"
            label_path.write_text(json.dumps(annotations, ensure_ascii=False,
                                             indent=2) + "\n", encoding="utf-8")
            source_pcd = args.source_scene / "pcd" / f"{timestamp}.pcd"
            if not source_pcd.exists():
                raise FileNotFoundError(source_pcd)
            hardlink_or_copy(source_pcd,
                             args.output_scene / "pcd" / source_pcd.name)
            frame_count += 1
            box_count += len(annotations)
            report_frames.append({"scenario": scenario, "frame": timestamp,
                                  "predictions": len(annotations)})

    for name in ("frame_meta.json", "sync_report.json"):
        source = args.source_scene / name
        if source.exists():
            shutil.copy2(source, args.output_scene / name)
    report = {
        "model_dir": str(args.model_dir.resolve()),
        "test_dir": str(args.test_dir.resolve()),
        "score_threshold": args.score_threshold,
        "tracking_distance": args.tracking_distance,
        "inference_ego": "1 (rear)",
        "export_coordinate": "0 (front/SUSTechPOINTS)",
        "frames": frame_count,
        "car_boxes": box_count,
        "frame_results": report_frames,
    }
    (args.output_scene / "prelabel_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"exported {box_count} Car boxes in {frame_count} frames")
    print(f"output: {args.output_scene}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
