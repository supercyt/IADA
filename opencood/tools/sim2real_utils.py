"""Utilities shared by the OPV2V-to-DAIR Sim2Real training path."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from copy import deepcopy
from typing import Any, Dict, Iterator, Tuple, TypeVar

import torch


_Batch = Dict[str, Any]
_T = TypeVar("_T")


def _deep_merge_in_place(
    destination: MutableMapping[str, Any],
    overrides: Mapping[str, Any],
) -> None:
    """Recursively merge ``overrides`` into ``destination``.

    Mapping values are merged recursively. All other values, including lists,
    replace the target value and are deep-copied so that the resulting source
    configuration cannot alias the target configuration.
    """

    for key, override_value in overrides.items():
        destination_value = destination.get(key)
        if isinstance(destination_value, Mapping) and isinstance(
            override_value, Mapping
        ):
            merged_child = deepcopy(dict(destination_value))
            _deep_merge_in_place(merged_child, override_value)
            destination[key] = merged_child
        else:
            destination[key] = deepcopy(override_value)


def build_source_config(
    target_config: Mapping[str, Any],
    *,
    keep_domain_adaptation: bool = False,
) -> Dict[str, Any]:
    """Derive the source dataset config from a target-centric top-level config.

    ``domain_adaptation.source`` contains source-only overrides such as the
    dataset type and data paths. The rest of the target-centric configuration
    (most importantly the shared voxel grid, model, and postprocessor settings)
    is retained.

    Parameters
    ----------
    target_config
        Fully parsed top-level target configuration.
    keep_domain_adaptation
        Keep the harmless domain-adaptation section in the returned config.
        It is removed by default to make the result a plain dataset config.
    """

    if not isinstance(target_config, Mapping):
        raise TypeError("target_config must be a mapping")

    domain_adaptation = target_config.get("domain_adaptation")
    if not isinstance(domain_adaptation, Mapping):
        raise KeyError(
            "target_config must contain a 'domain_adaptation' mapping"
        )

    source_overrides = domain_adaptation.get("source")
    if not isinstance(source_overrides, Mapping):
        raise KeyError(
            "target_config['domain_adaptation'] must contain a 'source' "
            "mapping"
        )

    source_config = deepcopy(dict(target_config))
    _deep_merge_in_place(source_config, source_overrides)
    if not keep_domain_adaptation:
        source_config.pop("domain_adaptation", None)

    return source_config


class ForeverDataIterator(Iterator[_T]):
    """Iterate over a re-iterable data loader, rebuilding it at exhaustion."""

    def __init__(self, data_loader: Any) -> None:
        self.data_loader = data_loader
        self._iterator = iter(data_loader)

    def __iter__(self) -> "ForeverDataIterator[_T]":
        return self

    def __next__(self) -> _T:
        try:
            return next(self._iterator)
        except StopIteration:
            self._iterator = iter(self.data_loader)
            # If the data loader is empty, the second StopIteration is allowed
            # to propagate rather than spinning forever.
            return next(self._iterator)

    def reset(self) -> None:
        """Explicitly rebuild the underlying iterator."""

        self._iterator = iter(self.data_loader)


def _require_tensor(
    mapping: Mapping[str, Any],
    key: str,
    location: str,
) -> torch.Tensor:
    if key not in mapping:
        raise KeyError(f"{location} is missing required key '{key}'")
    value = mapping[key]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{location}['{key}'] must be a torch.Tensor")
    return value


def _assert_common_grid(
    source_ego: Mapping[str, Any],
    target_ego: Mapping[str, Any],
) -> None:
    """Assert that both collated batches use the same physical anchor grid."""

    source_anchor = _require_tensor(source_ego, "anchor_box", "source_ego")
    target_anchor = _require_tensor(target_ego, "anchor_box", "target_ego")

    if source_anchor.ndim != 4:
        raise ValueError(
            "source_ego['anchor_box'] must have shape [H, W, A, 7]"
        )
    if target_anchor.ndim != 4:
        raise ValueError(
            "target_ego['anchor_box'] must have shape [H, W, A, 7]"
        )
    if source_anchor.shape != target_anchor.shape:
        raise ValueError(
            "source and target anchor grids have different shapes: "
            f"{tuple(source_anchor.shape)} != {tuple(target_anchor.shape)}"
        )
    if not torch.equal(source_anchor, target_anchor):
        raise ValueError(
            "source and target anchor grids differ; use one target-centric "
            "preprocess/postprocess grid for both domains"
        )


def _validate_domain_batch(
    ego: Mapping[str, Any],
    domain: str,
) -> Tuple[_Batch, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    if not isinstance(ego, Mapping):
        raise TypeError(f"{domain}_ego must be a mapping")

    if "processed_lidar" not in ego:
        raise KeyError(
            f"{domain}_ego is missing required key 'processed_lidar'"
        )
    processed_lidar = ego["processed_lidar"]
    if not isinstance(processed_lidar, Mapping):
        raise TypeError(
            f"{domain}_ego['processed_lidar'] must be a mapping"
        )

    voxel_features = _require_tensor(
        processed_lidar, "voxel_features", f"{domain}_ego['processed_lidar']"
    )
    voxel_coords = _require_tensor(
        processed_lidar, "voxel_coords", f"{domain}_ego['processed_lidar']"
    )
    voxel_num_points = _require_tensor(
        processed_lidar,
        "voxel_num_points",
        f"{domain}_ego['processed_lidar']",
    )
    record_len = _require_tensor(ego, "record_len", f"{domain}_ego")
    pairwise_t_matrix = _require_tensor(
        ego, "pairwise_t_matrix", f"{domain}_ego"
    )
    lidar_pose = _require_tensor(ego, "lidar_pose", f"{domain}_ego")

    if voxel_features.ndim < 2:
        raise ValueError(
            f"{domain} voxel_features must have at least two dimensions"
        )
    if voxel_coords.ndim != 2 or voxel_coords.shape[1] != 4:
        raise ValueError(f"{domain} voxel_coords must have shape [M, 4]")
    if voxel_num_points.ndim != 1:
        raise ValueError(
            f"{domain} voxel_num_points must have shape [M]"
        )
    if not (
        voxel_features.shape[0]
        == voxel_coords.shape[0]
        == voxel_num_points.shape[0]
    ):
        raise ValueError(f"{domain} voxel tensors disagree on voxel count")
    if voxel_coords.is_floating_point():
        raise TypeError(f"{domain} voxel_coords must use an integer dtype")

    if record_len.ndim != 1:
        raise ValueError(f"{domain} record_len must have shape [B]")
    if record_len.is_floating_point():
        raise TypeError(f"{domain} record_len must use an integer dtype")
    if record_len.numel() == 0:
        raise ValueError(f"{domain} batch must contain a scene")
    if not bool(torch.all(record_len > 0).item()):
        raise ValueError(f"{domain} record_len entries must be positive")

    scene_count = int(record_len.numel())
    agent_count = int(record_len.sum().item())

    if pairwise_t_matrix.ndim != 5:
        raise ValueError(
            f"{domain} pairwise_t_matrix must have shape [B, L, L, 4, 4]"
        )
    if pairwise_t_matrix.shape[0] != scene_count:
        raise ValueError(
            f"{domain} pairwise batch dimension does not match record_len"
        )
    if pairwise_t_matrix.shape[1] != pairwise_t_matrix.shape[2]:
        raise ValueError(
            f"{domain} pairwise agent dimensions must be square"
        )
    if pairwise_t_matrix.shape[-2:] != (4, 4):
        raise ValueError(f"{domain} pairwise transforms must be 4x4")
    max_agents = int(pairwise_t_matrix.shape[1])
    if not bool(torch.all(record_len <= max_agents).item()):
        raise ValueError(
            f"{domain} record_len exceeds pairwise padding size L={max_agents}"
        )

    if lidar_pose.ndim != 2:
        raise ValueError(
            f"{domain} lidar_pose must have shape "
            "[sum(record_len), pose_dim]"
        )
    if lidar_pose.shape[0] != agent_count:
        raise ValueError(
            f"{domain} lidar_pose count does not match sum(record_len)"
        )

    if voxel_coords.shape[0] > 0:
        agent_indices = voxel_coords[:, 0]
        if int(agent_indices.min().item()) < 0:
            raise ValueError(
                f"{domain} voxel agent indices must be non-negative"
            )
        if int(agent_indices.max().item()) >= agent_count:
            raise ValueError(
                f"{domain} voxel agent index exceeds sum(record_len)"
            )

    processed = {
        "voxel_features": voxel_features,
        "voxel_coords": voxel_coords,
        "voxel_num_points": voxel_num_points,
    }
    return (
        processed,
        record_len,
        pairwise_t_matrix,
        lidar_pose,
        agent_count,
    )


def _assert_cat_compatible(
    source: torch.Tensor,
    target: torch.Tensor,
    name: str,
) -> None:
    if source.ndim != target.ndim:
        raise ValueError(f"source and target {name} ranks differ")
    if source.shape[1:] != target.shape[1:]:
        raise ValueError(
            f"source and target {name} trailing shapes differ: "
            f"{tuple(source.shape[1:])} != {tuple(target.shape[1:])}"
        )
    if source.dtype != target.dtype:
        raise TypeError(
            f"source and target {name} dtypes differ: "
            f"{source.dtype} != {target.dtype}"
        )
    if source.device != target.device:
        raise ValueError(
            f"source and target {name} devices differ: "
            f"{source.device} != {target.device}"
        )


def _canonical_prior_encoding(
    ego: Mapping[str, Any],
    record_len: torch.Tensor,
    pairwise_t_matrix: torch.Tensor,
    domain: str,
) -> torch.Tensor:
    """Build and validate the V2X-ViT ``[velocity, delay, type]`` prior.

    OPV2V contains vehicles only, so every source value is zero. DAIR-V2X
    collates the vehicle first and roadside infrastructure second; therefore
    only local agent index 1 in a two-or-more-agent target scene receives type
    1. Single-agent scenes and padded slots remain zero.

    A batch-provided prior is treated as an assertion, not as an override. It
    must exactly match this domain-derived value. This prevents stale dataset
    metadata from silently marking a source collaborator as infrastructure or
    omitting the target roadside agent. Detection labels are never inspected.
    """

    if domain not in ("source", "target"):
        raise ValueError("prior domain must be 'source' or 'target'")
    batch_size = int(record_len.numel())
    max_agents = int(pairwise_t_matrix.shape[1])
    canonical = torch.zeros(
        batch_size,
        max_agents,
        3,
        dtype=torch.float32,
        device=record_len.device,
    )
    if domain == "target":
        has_infrastructure = record_len >= 2
        canonical[has_infrastructure, 1, 2] = 1.0

    provided = ego.get("prior_encoding")
    if provided is None:
        return canonical
    if not isinstance(provided, torch.Tensor):
        raise TypeError(
            f"{domain}_ego['prior_encoding'] must be a torch.Tensor"
        )
    if provided.shape != canonical.shape:
        raise ValueError(
            f"{domain} prior_encoding must have shape "
            f"{tuple(canonical.shape)}, got {tuple(provided.shape)}"
        )
    if not provided.is_floating_point():
        raise TypeError(f"{domain} prior_encoding must use a floating dtype")
    if provided.device != canonical.device:
        raise ValueError(
            f"{domain} prior_encoding and record_len devices differ"
        )
    if not bool(torch.isfinite(provided).all().item()):
        raise ValueError(f"{domain} prior_encoding must contain finite values")
    if not torch.equal(provided, canonical.to(dtype=provided.dtype)):
        raise ValueError(
            f"{domain} prior_encoding conflicts with the canonical Sim2Real "
            "agent-type policy"
        )
    return canonical


def build_prior_encoding(
    ego: Mapping[str, Any],
    domain: str,
) -> torch.Tensor:
    """Return the canonical padded prior for one collated domain batch."""

    if not isinstance(ego, Mapping):
        raise TypeError(f"{domain}_ego must be a mapping")
    record_len = _require_tensor(ego, "record_len", f"{domain}_ego")
    pairwise_t_matrix = _require_tensor(
        ego, "pairwise_t_matrix", f"{domain}_ego"
    )
    if record_len.ndim != 1 or record_len.is_floating_point():
        raise ValueError(
            f"{domain} record_len must be a one-dimensional integer tensor"
        )
    if record_len.numel() == 0 or not bool((record_len > 0).all().item()):
        raise ValueError(f"{domain} record_len entries must be positive")
    if pairwise_t_matrix.ndim != 5 or pairwise_t_matrix.shape[-2:] != (4, 4):
        raise ValueError(
            f"{domain} pairwise_t_matrix must have shape [B, L, L, 4, 4]"
        )
    if pairwise_t_matrix.shape[0] != record_len.numel() or (
        pairwise_t_matrix.shape[1] != pairwise_t_matrix.shape[2]
    ):
        raise ValueError(
            f"{domain} pairwise_t_matrix does not match record_len/padding"
        )
    if not bool((record_len <= pairwise_t_matrix.shape[1]).all().item()):
        raise ValueError(f"{domain} record_len exceeds pairwise padding size")
    return _canonical_prior_encoding(
        ego, record_len, pairwise_t_matrix, domain
    )


def merge_source_target_batches(
    source_ego: Mapping[str, Any],
    target_ego: Mapping[str, Any],
) -> Tuple[_Batch, int, torch.Tensor]:
    """Merge collated source/target ego batches without target labels.

    The returned dictionary intentionally contains only inputs consumed by the
    PointPillar model. Scene-level domain labels are zero for source scenes and
    one for target scenes.
    """

    _assert_common_grid(source_ego, target_ego)
    (
        source_lidar,
        source_record_len,
        source_pairwise,
        source_pose,
        source_agent_count,
    ) = _validate_domain_batch(source_ego, "source")
    (
        target_lidar,
        target_record_len,
        target_pairwise,
        target_pose,
        _,
    ) = _validate_domain_batch(target_ego, "target")

    for key in ("voxel_features", "voxel_coords", "voxel_num_points"):
        _assert_cat_compatible(
            source_lidar[key],
            target_lidar[key],
            f"processed_lidar.{key}",
        )
    _assert_cat_compatible(
        source_record_len, target_record_len, "record_len"
    )
    _assert_cat_compatible(
        source_pairwise, target_pairwise, "pairwise_t_matrix"
    )
    _assert_cat_compatible(source_pose, target_pose, "lidar_pose")

    source_l = int(source_pairwise.shape[1])
    target_l = int(target_pairwise.shape[1])
    if source_l != target_l:
        raise ValueError(
            "source and target pairwise padding sizes differ: "
            f"L={source_l} != L={target_l}"
        )

    source_prior = _canonical_prior_encoding(
        source_ego, source_record_len, source_pairwise, "source"
    )
    target_prior = _canonical_prior_encoding(
        target_ego, target_record_len, target_pairwise, "target"
    )

    target_voxel_coords = target_lidar["voxel_coords"].clone()
    if target_voxel_coords.shape[0] > 0:
        target_voxel_coords[:, 0] += source_agent_count

    merged = {
        "processed_lidar": {
            "voxel_features": torch.cat(
                [
                    source_lidar["voxel_features"],
                    target_lidar["voxel_features"],
                ],
                dim=0,
            ),
            "voxel_coords": torch.cat(
                [source_lidar["voxel_coords"], target_voxel_coords],
                dim=0,
            ),
            "voxel_num_points": torch.cat(
                [
                    source_lidar["voxel_num_points"],
                    target_lidar["voxel_num_points"],
                ],
                dim=0,
            ),
        },
        "record_len": torch.cat(
            [source_record_len, target_record_len], dim=0
        ),
        "pairwise_t_matrix": torch.cat(
            [source_pairwise, target_pairwise], dim=0
        ),
        "lidar_pose": torch.cat([source_pose, target_pose], dim=0),
        "prior_encoding": torch.cat(
            [source_prior, target_prior], dim=0
        ),
    }

    source_scene_count = int(source_record_len.numel())
    target_scene_count = int(target_record_len.numel())
    domain_labels = torch.cat(
        [
            torch.zeros(
                source_scene_count,
                dtype=torch.float32,
                device=source_record_len.device,
            ),
            torch.ones(
                target_scene_count,
                dtype=torch.float32,
                device=source_record_len.device,
            ),
        ],
        dim=0,
    )

    return merged, source_scene_count, domain_labels


__all__ = [
    "ForeverDataIterator",
    "build_prior_encoding",
    "build_source_config",
    "merge_source_target_batches",
]
