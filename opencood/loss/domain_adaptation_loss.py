"""Losses for fusion-agnostic collaborative perception domain adapters."""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


def balanced_domain_loss(
    domain_logits: torch.Tensor,
    domain_labels: torch.Tensor,
    scene_indices: Optional[torch.Tensor] = None,
    valid_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Binary domain loss with equal source/target contribution.

    ``scene_indices`` maps the first logit dimension to scene-level domain
    labels. Remaining logit dimensions can represent pixels or proposals and
    are broadcast automatically. A differentiable zero is returned when
    either domain has no valid element.
    """

    if domain_logits.ndim == 0 or domain_logits.shape[0] == 0:
        raise ValueError("domain_logits must have a non-empty batch dimension")
    labels = domain_labels.reshape(-1).to(
        device=domain_logits.device, dtype=domain_logits.dtype
    )
    if scene_indices is None:
        if domain_logits.shape[0] != labels.numel():
            raise ValueError(
                "scene_indices are required when logits are not scene-aligned"
            )
        scene_indices = torch.arange(
            labels.numel(), device=domain_logits.device
        )
    else:
        scene_indices = scene_indices.reshape(-1).to(
            device=domain_logits.device, dtype=torch.long
        )
    if scene_indices.numel() != domain_logits.shape[0]:
        raise ValueError("scene_indices must match the first logit dimension")
    if bool((scene_indices < 0).any().item()) or bool(
        (scene_indices >= labels.numel()).any().item()
    ):
        raise ValueError("scene_indices contain an out-of-range scene")

    element_labels = labels[scene_indices]
    label_shape = (domain_logits.shape[0],) + (1,) * (
        domain_logits.ndim - 1
    )
    element_labels = element_labels.reshape(label_shape).expand_as(
        domain_logits
    )

    if valid_mask is None:
        valid = torch.ones_like(domain_logits, dtype=torch.bool)
    else:
        valid = valid_mask.to(
            device=domain_logits.device, dtype=torch.bool
        )
        if valid.ndim == 1 and domain_logits.ndim > 1:
            valid = valid.reshape(label_shape)
        try:
            valid = valid.expand_as(domain_logits)
        except RuntimeError as error:
            raise ValueError(
                "valid_mask is not broadcastable to domain_logits"
            ) from error

    per_element = F.binary_cross_entropy_with_logits(
        domain_logits, element_labels, reduction="none"
    )
    predictions = (domain_logits >= 0).to(element_labels.dtype)
    domain_losses = []
    domain_accuracies = []
    valid_count = 0
    for domain_value in (0.0, 1.0):
        selected = valid & (element_labels == domain_value)
        count = int(selected.sum().item())
        valid_count += count
        if count:
            domain_losses.append(per_element[selected].mean())
            domain_accuracies.append(
                (predictions[selected] == element_labels[selected])
                .float()
                .mean()
            )

    if len(domain_losses) != 2:
        zero = domain_logits.sum() * 0.0
        return zero, domain_logits.new_tensor(float("nan")), valid_count
    return (
        torch.stack(domain_losses).mean(),
        torch.stack(domain_accuracies).mean(),
        valid_count,
    )


def graph_variance_floor_loss(
    graph_embedding: torch.Tensor,
    valid_graph_mask: torch.Tensor,
    source_scene_count: int,
    target_std: float,
    epsilon: float = 1.0e-8,
) -> Tuple[torch.Tensor, int]:
    """Keep IADA graph representations from collapsing to a constant."""

    if graph_embedding.ndim != 2:
        raise ValueError("graph_embedding must have shape [B, D]")
    if target_std < 0:
        raise ValueError("target_std must be non-negative")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    valid = valid_graph_mask.reshape(-1).to(
        device=graph_embedding.device, dtype=torch.bool
    )
    if valid.numel() != graph_embedding.shape[0]:
        raise ValueError("graph embedding and valid mask batch sizes differ")
    if source_scene_count < 0 or source_scene_count > graph_embedding.shape[0]:
        raise ValueError("source_scene_count is outside the graph batch")

    losses = []
    for domain_slice in (
        slice(0, source_scene_count),
        slice(source_scene_count, None),
    ):
        selected = graph_embedding[domain_slice][valid[domain_slice]]
        if selected.shape[0] >= 2:
            standard_deviation = torch.sqrt(
                selected.var(dim=0, unbiased=False).mean() + epsilon
            )
            losses.append(F.relu(target_std - standard_deviation))
    if len(losses) == 2 and target_std > 0:
        return torch.stack(losses).mean(), 1
    return graph_embedding.sum() * 0.0, 0


def dusa_agent_loss(
    agent_logits: torch.Tensor,
    confidence_weights: torch.Tensor,
    scene_indices: torch.Tensor,
    local_indices: torch.Tensor,
    domain_labels: torch.Tensor,
    record_len: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Confidence-aware target-only vehicle/infrastructure alignment.

    DAIR-V2X uses local agent index 0 for the vehicle ego and 1 for roadside
    infrastructure. Scenes without exactly this pair are skipped so flattened
    agents from different scenes can never be paired accidentally.
    """

    if agent_logits.shape != confidence_weights.shape:
        raise ValueError("DUSA agent logits and confidence weights must match")
    scene_indices = scene_indices.reshape(-1).to(
        device=agent_logits.device, dtype=torch.long
    )
    local_indices = local_indices.reshape(-1).to(
        device=agent_logits.device, dtype=torch.long
    )
    if scene_indices.numel() != agent_logits.shape[0] or (
        local_indices.numel() != agent_logits.shape[0]
    ):
        raise ValueError("DUSA agent index metadata has the wrong length")
    labels = domain_labels.reshape(-1).to(agent_logits.device)
    counts = torch.as_tensor(
        record_len, device=agent_logits.device, dtype=torch.long
    ).reshape(-1)
    if counts.numel() != labels.numel():
        raise ValueError("DUSA record_len and domain labels must be scene-aligned")

    selected_logits = []
    selected_weights = []
    selected_labels = []
    valid_count = 0
    for scene_index in range(labels.numel()):
        if labels[scene_index].item() != 1 or counts[scene_index].item() != 2:
            continue
        selected_agents = scene_indices == scene_index
        roles = local_indices[selected_agents]
        if roles.numel() != 2 or set(roles.tolist()) != {0, 1}:
            continue
        logits = agent_logits[selected_agents]
        weights = confidence_weights[selected_agents]
        role_shape = (roles.numel(),) + (1,) * (logits.ndim - 1)
        role_labels = roles.to(logits.dtype).reshape(role_shape).expand_as(
            logits
        )
        selected_logits.append(logits)
        selected_weights.append(weights)
        selected_labels.append(role_labels)
        valid_count += int(logits.numel())

    if not selected_logits:
        zero = agent_logits.sum() * 0.0
        return zero, agent_logits.new_tensor(float("nan")), 0
    logits = torch.cat(selected_logits, dim=0)
    weights = torch.cat(selected_weights, dim=0)
    role_labels = torch.cat(selected_labels, dim=0)
    # The official DUSA implementation normalizes the shared confidence map
    # once over the complete target mini-batch, then relies on BCE's default
    # element-wise mean (it does not divide by the sum of weights).
    weights = weights / (weights.amax() + 1.0e-6)
    per_element = F.binary_cross_entropy_with_logits(
        logits, role_labels, reduction="none"
    )
    loss = (per_element * weights).mean()
    predictions = (logits >= 0).to(role_labels.dtype)
    accuracy = (
        (predictions == role_labels).to(weights.dtype) * weights
    ).sum() / weights.sum().clamp_min(1.0e-6)
    return loss, accuracy, valid_count


def cudax_bin_loss(
    bin_logits: torch.Tensor,
    source_targets: Mapping[str, torch.Tensor],
    bin_count: int,
    residual_bounds: Sequence[float],
) -> torch.Tensor:
    """Source-only CUDA-X residual coordinate encoding loss."""

    if bin_count <= 1:
        raise ValueError("CUDA-X bin_count must be greater than one")
    if len(residual_bounds) != 6 or any(
        float(bound) <= 0 for bound in residual_bounds
    ):
        raise ValueError("CUDA-X residual_bounds must contain six positives")
    targets = source_targets["targets"]
    positives = source_targets["pos_equal_one"]
    batch_size = targets.shape[0]
    if positives.shape[0] != batch_size:
        raise ValueError("CUDA-X source target tensors have different batches")
    if positives.ndim != 4 or targets.ndim != 4:
        raise ValueError(
            "CUDA-X targets and pos_equal_one must be four-dimensional"
        )
    height, width, anchor_number = positives.shape[1:4]
    if targets.shape[1:3] != (height, width) or (
        targets.shape[3] != anchor_number * 7
    ):
        raise ValueError(
            "CUDA-X regression targets do not match the positive-anchor grid"
        )
    expected_channels = anchor_number * 6 * bin_count
    if bin_logits.shape[0] != batch_size or (
        bin_logits.shape[1] != expected_channels
    ):
        raise ValueError(
            "CUDA-X bin logits do not match source batch/anchor dimensions"
        )
    if bin_logits.shape[-2:] != (height, width):
        raise ValueError(
            "CUDA-X bin logits must use the same H/W anchor grid as the "
            "source labels"
        )
    logits = bin_logits.reshape(
        batch_size, anchor_number, 6, bin_count, height, width
    ).permute(0, 4, 5, 1, 2, 3)
    regression_targets = targets.reshape(
        batch_size, height, width, anchor_number, 7
    )[..., :6]
    positive_mask = positives.to(dtype=torch.bool)
    if not bool(positive_mask.any().item()):
        return bin_logits.sum() * 0.0
    logits = logits[positive_mask]
    regression_targets = regression_targets[positive_mask]
    bounds = regression_targets.new_tensor(residual_bounds).reshape(
        1, 6
    )
    normalized = (regression_targets + bounds) * (
        float(bin_count) / (2.0 * bounds)
    )
    classes = normalized.floor().clamp(0, bin_count - 1).long()
    per_coordinate = F.cross_entropy(
        logits.reshape(-1, bin_count),
        classes.reshape(-1),
        reduction="none",
    ).reshape_as(classes)
    return per_coordinate.mean()


def compute_adaptation_loss(
    method: str,
    output_dict: Mapping[str, torch.Tensor],
    domain_labels: torch.Tensor,
    source_scene_count: int,
    record_len: torch.Tensor,
    source_label_dict: Mapping[str, torch.Tensor],
    config: Mapping[str, object],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Dispatch the complete, already-weighted loss for one DA method."""

    method = method.lower().replace("-", "").replace("_", "")
    metrics: Dict[str, torch.Tensor] = {}
    if method in ("grl", "discriminator", "naive"):
        loss, accuracy, valid_count = balanced_domain_loss(
            output_dict["domain_logits"],
            domain_labels,
            output_dict.get("domain_scene_index"),
            output_dict.get("domain_valid_mask"),
        )
        metrics.update(
            domain_loss=loss,
            domain_accuracy=accuracy,
            valid_domain_count=loss.new_tensor(valid_count),
        )
        return float(config.get("domain_loss_weight", 0.1)) * loss, metrics

    if method == "dusa":
        lsa_loss, lsa_accuracy, lsa_count = balanced_domain_loss(
            output_dict["domain_logits"],
            domain_labels,
            output_dict.get("domain_scene_index"),
            output_dict.get("domain_valid_mask"),
        )
        cia_loss, cia_accuracy, cia_count = dusa_agent_loss(
            output_dict["agent_domain_logits"],
            output_dict["agent_domain_weights"],
            output_dict["agent_scene_index"],
            output_dict["agent_local_index"],
            domain_labels,
            record_len,
        )
        metrics.update(
            domain_loss=lsa_loss + cia_loss,
            domain_accuracy=lsa_accuracy,
            lsa_loss=lsa_loss,
            lsa_accuracy=lsa_accuracy,
            lsa_valid_count=lsa_loss.new_tensor(lsa_count),
            cia_loss=cia_loss,
            cia_accuracy=cia_accuracy,
            cia_valid_count=cia_loss.new_tensor(cia_count),
        )
        return (
            float(config.get("dusa_lsa_weight", 1.0)) * lsa_loss
            + float(config.get("dusa_cia_weight", 1.0)) * cia_loss,
            metrics,
        )

    if method == "cudax":
        domain_losses = []
        domain_accuracies = []
        for name in ("ckt", "blc", "cpa"):
            loss, accuracy, valid_count = balanced_domain_loss(
                output_dict[f"{name}_domain_logits"],
                domain_labels,
                output_dict.get("domain_scene_index"),
                output_dict.get("domain_valid_mask"),
            )
            metrics[f"{name}_domain_loss"] = loss
            metrics[f"{name}_domain_accuracy"] = accuracy
            metrics[f"{name}_valid_count"] = loss.new_tensor(valid_count)
            domain_losses.append(loss)
            domain_accuracies.append(accuracy)
        source_bin_logits = output_dict["bin_logits"][:source_scene_count]
        residual_bounds = config.get("cudax_residual_bounds")
        if residual_bounds is None or len(residual_bounds) != 6:
            raise ValueError(
                "CUDA-X requires six source-only cudax_residual_bounds in "
                "encoded [x, y, z, h, w, l] order"
            )
        bin_loss = cudax_bin_loss(
            source_bin_logits,
            source_label_dict,
            int(config.get("cudax_bin_count", 5)),
            residual_bounds,
        )
        domain_loss = torch.stack(domain_losses).sum()
        finite_accuracies = [
            value for value in domain_accuracies if not torch.isnan(value)
        ]
        domain_accuracy = (
            torch.stack(finite_accuracies).mean()
            if finite_accuracies
            else bin_loss.new_tensor(float("nan"))
        )
        metrics.update(
            domain_loss=domain_loss,
            domain_accuracy=domain_accuracy,
            bin_loss=bin_loss,
        )
        return (
            float(config.get("cudax_bin_loss_weight", 0.1)) * bin_loss
            + float(config.get("cudax_domain_loss_weight", 0.1))
            * domain_loss,
            metrics,
        )

    if method == "iada":
        domain_loss, domain_accuracy, valid_count = balanced_domain_loss(
            output_dict["domain_logits"],
            domain_labels,
            output_dict.get("domain_scene_index"),
            output_dict.get("domain_valid_mask"),
        )
        variance_loss, variance_applied = graph_variance_floor_loss(
            output_dict["graph_embedding"],
            output_dict["valid_graph_mask"],
            source_scene_count,
            float(config.get("graph_variance_target_std", 0.0)),
        )
        metrics.update(
            domain_loss=domain_loss,
            domain_accuracy=domain_accuracy,
            valid_domain_count=domain_loss.new_tensor(valid_count),
            graph_variance_loss=variance_loss,
            graph_variance_update_applied=domain_loss.new_tensor(
                variance_applied
            ),
        )
        return (
            float(config.get("domain_loss_weight", 0.1)) * domain_loss
            + float(config.get("graph_variance_weight", 0.0))
            * variance_loss,
            metrics,
        )

    raise ValueError(f"Unsupported domain adaptation method: {method!r}")


__all__ = [
    "balanced_domain_loss",
    "compute_adaptation_loss",
    "cudax_bin_loss",
    "dusa_agent_loss",
    "graph_variance_floor_loss",
]
