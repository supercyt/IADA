"""Compute CUDA-X RCE bounds from source-train positive anchors only."""

import argparse
import random

import numpy as np
import torch

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools.sim2real_utils import build_source_config


def _parser():
    parser = argparse.ArgumentParser(
        description=(
            "Compute CUDA-X encoded residual bounds from the configured "
            "source training split"
        )
    )
    parser.add_argument("--hypes_yaml", "-y", required=True)
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional prefix length for a quick diagnostic run.",
    )
    return parser.parse_args()


def compute_source_residual_bounds(dataset, max_samples=None):
    """Return max-absolute bounds in encoded [x,y,z,h,w,l] order."""

    sample_count = len(dataset)
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("max_samples must be positive")
        sample_count = min(sample_count, max_samples)

    bounds = np.zeros(6, dtype=np.float64)
    positive_count = 0
    for sample_index in range(sample_count):
        sample = dataset[sample_index]
        if sample is None:
            continue
        labels = sample["ego"]["label_dict"]
        positives = np.asarray(labels["pos_equal_one"], dtype=bool)
        targets = np.asarray(labels["targets"])
        if positives.ndim != 3 or targets.shape != (
            positives.shape[0],
            positives.shape[1],
            positives.shape[2] * 7,
        ):
            raise ValueError(
                f"sample {sample_index} has an unexpected anchor-label shape"
            )
        encoded = targets.reshape(*positives.shape, 7)[positives, :6]
        if encoded.size == 0:
            continue
        if not np.isfinite(encoded).all():
            raise ValueError(
                f"sample {sample_index} contains non-finite source residuals"
            )
        bounds = np.maximum(bounds, np.abs(encoded).max(axis=0))
        positive_count += encoded.shape[0]

    if positive_count == 0 or np.any(bounds <= 0):
        raise RuntimeError(
            "No usable positive source anchors were found for all six "
            "encoded residual coordinates"
        )
    return bounds, positive_count, sample_count


def main():
    from opencood.data_utils.datasets import build_dataset

    options = _parser()
    target_hypes = yaml_utils.load_yaml(options.hypes_yaml, None)
    seed = int(target_hypes["domain_adaptation"].get("seed", 303))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    source_hypes = build_source_config(target_hypes)
    source_dataset = build_dataset(
        source_hypes, visualize=False, train=True
    )
    # RCE bounds depend only on source GT boxes and anchors. Avoid loading and
    # voxelizing LiDAR here; on OPV2V this reduces a full scan from tens of
    # minutes to a small label-only preprocessing job.
    source_dataset.load_lidar_file = False
    bounds, positive_count, sample_count = compute_source_residual_bounds(
        source_dataset, options.max_samples
    )
    formatted = ", ".join(f"{value:.9g}" for value in bounds)
    print(
        f"Scanned {sample_count} source samples and {positive_count} "
        "positive anchors."
    )
    print("Encoded order: [dx, dy, dz, dlog_h, dlog_w, dlog_l]")
    print(f"cudax_residual_bounds: [{formatted}]")


if __name__ == "__main__":
    main()
