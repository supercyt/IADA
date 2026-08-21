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


def entropy_weighted_domain_loss(
    domain_logits: torch.Tensor,
    attention: torch.Tensor,
    domain_labels: torch.Tensor,
    scene_indices: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Domain-balanced BCE weighted by an SSDA entropy attention map."""

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

    label_shape = (domain_logits.shape[0],) + (1,) * (
        domain_logits.ndim - 1
    )
    element_labels = labels[scene_indices].reshape(label_shape).expand_as(
        domain_logits
    )
    weights = attention.to(
        device=domain_logits.device, dtype=domain_logits.dtype
    )
    try:
        weights = weights.expand_as(domain_logits)
    except RuntimeError as error:
        raise ValueError(
            "attention is not broadcastable to domain_logits"
        ) from error
    if bool((weights < 0).any().item()):
        raise ValueError("SSDA attention weights must be non-negative")

    per_element = F.binary_cross_entropy_with_logits(
        domain_logits, element_labels, reduction="none"
    )
    predictions = (domain_logits >= 0).to(element_labels.dtype)
    losses = []
    accuracies = []
    valid_count = 0
    for domain_value in (0.0, 1.0):
        selected = element_labels == domain_value
        selected_count = int(selected.sum().item())
        valid_count += selected_count
        if not selected_count:
            continue
        selected_weights = weights[selected]
        losses.append(
            (per_element[selected] * selected_weights).mean()
        )
        accuracy_denominator = selected_weights.sum()
        if float(accuracy_denominator.detach().item()) > 0:
            accuracies.append(
                (
                    (predictions[selected] == element_labels[selected]).to(
                        selected_weights.dtype
                    )
                    * selected_weights
                ).sum()
                / accuracy_denominator
            )

    if len(losses) != 2:
        zero = domain_logits.sum() * 0.0
        return zero, domain_logits.new_tensor(float("nan")), valid_count
    accuracy = (
        torch.stack(accuracies).mean()
        if accuracies
        else domain_logits.new_tensor(float("nan"))
    )
    return torch.stack(losses).mean(), accuracy, valid_count


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


def single_domain_loss(
    domain_logits: torch.Tensor,
    domain_label: float,
    scene_indices: Optional[torch.Tensor] = None,
    valid_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Return the mean adversarial loss for one domain mini-batch.

    This is the per-domain half of :func:`balanced_domain_loss`.  Keeping it
    separate lets training backpropagate through the source graph before the
    target forward is constructed, while preserving the original balanced
    source/target objective.
    """

    if domain_label not in (0.0, 1.0):
        raise ValueError("domain_label must be either 0 or 1")
    if domain_logits.ndim == 0 or domain_logits.shape[0] == 0:
        raise ValueError("domain_logits must have a non-empty batch dimension")

    if scene_indices is not None:
        scene_indices = scene_indices.reshape(-1)
        if scene_indices.numel() != domain_logits.shape[0]:
            raise ValueError("scene_indices must match the first logit dimension")

    labels = torch.full_like(domain_logits, float(domain_label))
    if valid_mask is None:
        valid = torch.ones_like(domain_logits, dtype=torch.bool)
    else:
        valid = valid_mask.to(device=domain_logits.device, dtype=torch.bool)
        if valid.ndim == 1 and domain_logits.ndim > 1:
            valid = valid.reshape(
                (domain_logits.shape[0],)
                + (1,) * (domain_logits.ndim - 1)
            )
        try:
            valid = valid.expand_as(domain_logits)
        except RuntimeError as error:
            raise ValueError(
                "valid_mask is not broadcastable to domain_logits"
            ) from error

    valid_count = int(valid.sum().item())
    if not valid_count:
        zero = domain_logits.sum() * 0.0
        return zero, domain_logits.new_tensor(float("nan")), 0

    loss = F.binary_cross_entropy_with_logits(
        domain_logits[valid], labels[valid]
    )
    predictions = (domain_logits[valid] >= 0).to(labels.dtype)
    accuracy = (predictions == labels[valid]).float().mean()
    return loss, accuracy, valid_count


def single_domain_entropy_weighted_loss(
    domain_logits: torch.Tensor,
    attention: torch.Tensor,
    domain_label: float,
    scene_indices: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Return one domain's half of the SSDA alignment objective."""

    if domain_label not in (0.0, 1.0):
        raise ValueError("domain_label must be either 0 or 1")
    if domain_logits.ndim == 0 or domain_logits.shape[0] == 0:
        raise ValueError("domain_logits must have a non-empty batch dimension")
    if scene_indices is not None:
        scene_indices = scene_indices.reshape(-1)
        if scene_indices.numel() != domain_logits.shape[0]:
            raise ValueError("scene_indices must match the first logit dimension")

    labels = torch.full_like(domain_logits, float(domain_label))
    weights = attention.to(
        device=domain_logits.device, dtype=domain_logits.dtype
    )
    try:
        weights = weights.expand_as(domain_logits)
    except RuntimeError as error:
        raise ValueError(
            "attention is not broadcastable to domain_logits"
        ) from error
    if bool((weights < 0).any().item()):
        raise ValueError("SSDA attention weights must be non-negative")

    valid_count = int(domain_logits.numel())
    per_element = F.binary_cross_entropy_with_logits(
        domain_logits, labels, reduction="none"
    )
    loss = (per_element * weights).mean()
    predictions = (domain_logits >= 0).to(labels.dtype)
    accuracy_denominator = weights.sum()
    if float(accuracy_denominator.detach().item()) > 0:
        accuracy = (
            (predictions == labels).to(weights.dtype) * weights
        ).sum() / accuracy_denominator
    else:
        accuracy = domain_logits.new_tensor(float("nan"))
    return loss, accuracy, valid_count


def single_domain_graph_variance_loss(
    graph_embedding: torch.Tensor,
    valid_graph_mask: torch.Tensor,
    target_std: float,
    enabled: bool,
    epsilon: float = 1.0e-8,
) -> torch.Tensor:
    """Return one domain's contribution to the IADA variance floor."""

    valid = valid_graph_mask.reshape(-1).to(
        device=graph_embedding.device, dtype=torch.bool
    )
    selected = graph_embedding[valid]
    if not enabled or target_std <= 0 or selected.shape[0] < 2:
        return graph_embedding.sum() * 0.0
    standard_deviation = torch.sqrt(
        selected.var(dim=0, unbiased=False).mean() + epsilon
    )
    return 0.5 * F.relu(target_std - standard_deviation)


def single_domain_conditioned_loss(
    domain_logits: torch.Tensor,
    condition_weights: torch.Tensor,
    domain_label: float,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Normalize every active IADA condition independently.

    Normalizing by the sum of soft weights is important here: averaging over
    the entire BEV map would make the adversarial signal vanish as object
    evidence becomes sparse.
    """

    if domain_label not in (0.0, 1.0):
        raise ValueError("domain_label must be either 0 or 1")
    if domain_logits.shape != condition_weights.shape:
        raise ValueError("IADA local logits and condition weights must match")
    if domain_logits.ndim != 4:
        raise ValueError("IADA local domain tensors must have shape [B,K,H,W]")
    weights = condition_weights.to(
        device=domain_logits.device, dtype=domain_logits.dtype
    )
    if bool((weights < 0).any().item()):
        raise ValueError("IADA condition weights must be non-negative")

    labels = torch.full_like(domain_logits, float(domain_label))
    element_loss = F.binary_cross_entropy_with_logits(
        domain_logits, labels, reduction="none"
    )
    correct = ((domain_logits >= 0) == (labels > 0.5)).to(weights.dtype)
    losses = []
    accuracies = []
    for condition in range(domain_logits.shape[1]):
        selected_weights = weights[:, condition]
        denominator = selected_weights.sum()
        if float(denominator.detach().item()) <= 1.0e-8:
            continue
        losses.append(
            (element_loss[:, condition] * selected_weights).sum()
            / denominator
        )
        accuracies.append(
            (correct[:, condition] * selected_weights).sum() / denominator
        )
    if not losses:
        zero = domain_logits.sum() * 0.0
        return zero, domain_logits.new_tensor(float("nan")), 0
    return (
        torch.stack(losses).mean(),
        torch.stack(accuracies).mean(),
        int((weights > 0).sum().item()),
    )


def iada_effect_domain_loss(
    output_dict: Mapping[str, torch.Tensor],
    domain_label: float,
    config: Mapping[str, object],
    *,
    enabled: bool,
    variance_enabled: bool,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """One domain's balanced conditional collaboration-effect objective."""

    global_logits = output_dict["iada_global_domain_logits"]
    local_logits = output_dict["iada_local_domain_logits"]
    valid = output_dict["iada_global_domain_valid"]
    if enabled:
        global_loss, global_accuracy, global_count = single_domain_loss(
            global_logits, domain_label, valid_mask=valid
        )
        local_loss, local_accuracy, local_count = (
            single_domain_conditioned_loss(
                local_logits,
                output_dict["iada_local_domain_weights"],
                domain_label,
            )
        )
    else:
        global_loss = global_logits.sum() * 0.0
        local_loss = local_logits.sum() * 0.0
        global_accuracy = global_logits.new_tensor(float("nan"))
        local_accuracy = local_logits.new_tensor(float("nan"))
        global_count = local_count = 0

    half_global = 0.5 * global_loss
    half_local = 0.5 * local_loss
    identity = output_dict.get(
        "iada_gate_identity_loss", global_logits.new_zeros(())
    ).mean()
    half_identity = 0.5 * identity

    effect = output_dict["iada_effect_features"]
    selected_effect = effect[valid.to(device=effect.device, dtype=torch.bool)]
    target_std = float(config.get("iada_effect_target_std", 0.05))
    if variance_enabled and target_std > 0 and selected_effect.shape[0] >= 2:
        samples = selected_effect.permute(0, 2, 3, 1).reshape(
            -1, selected_effect.shape[1]
        )
        effect_std = torch.sqrt(
            samples.var(dim=0, unbiased=False).mean() + 1.0e-8
        )
        half_variance = 0.5 * F.relu(target_std - effect_std)
    else:
        effect_std = effect.new_zeros(())
        half_variance = effect.sum() * 0.0

    total = (
        float(config.get("iada_global_domain_weight", 0.05)) * half_global
        + float(config.get("iada_local_domain_weight", 0.05)) * half_local
        + float(config.get("iada_gate_identity_weight", 0.1)) * half_identity
        + float(config.get("iada_effect_variance_weight", 0.01))
        * half_variance
    )
    domain_accuracy = torch.stack(
        (global_accuracy, local_accuracy)
    ).nanmean()
    metrics = {
        "domain_loss": half_global + half_local,
        "domain_accuracy": domain_accuracy,
        "iada_global_domain_loss": half_global,
        "iada_global_domain_accuracy": global_accuracy,
        "iada_global_valid_count": global_logits.new_tensor(global_count),
        "iada_local_domain_loss": half_local,
        "iada_local_domain_accuracy": local_accuracy,
        "iada_local_valid_count": global_logits.new_tensor(local_count),
        "iada_gate_identity_loss": half_identity,
        "iada_effect_variance_loss": half_variance,
        "iada_effect_std": effect_std,
        "iada_local_active_fraction": output_dict.get(
            "iada_local_active_fraction", global_logits.new_zeros(())
        ).mean(),
    }
    return total, metrics


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


def _iada_anchor_last(
    tensor: torch.Tensor,
    anchor_number: int,
    values_per_anchor: int = 1,
) -> torch.Tensor:
    """Convert PointPillar channel-first predictions to [B,H,W,A,V]."""

    if tensor.ndim != 4 or tensor.shape[1] != (
        anchor_number * values_per_anchor
    ):
        raise ValueError("IADA prediction channels do not match anchors")
    batch_size, _, height, width = tensor.shape
    return tensor.reshape(
        batch_size,
        anchor_number,
        values_per_anchor,
        height,
        width,
    ).permute(0, 3, 4, 1, 2)


def _iada_source_effect_memory_update(
    output_dict: Mapping[str, torch.Tensor],
    source_targets: Mapping[str, torch.Tensor],
    ego_cls: torch.Tensor,
    fused_cls: torch.Tensor,
    ego_reg_error: torch.Tensor,
    fused_reg_error: torch.Tensor,
    config: Mapping[str, object],
) -> int:
    """Update source-only discovery/suppression/refinement prototypes."""

    required = {
        "iada_effect_features",
        "iada_effect_prototypes",
        "iada_effect_counts",
        "iada_range_index",
    }
    if not required.issubset(output_dict):
        return 0
    prototypes = output_dict["iada_effect_prototypes"]
    counts = output_dict["iada_effect_counts"]
    features = output_dict["iada_effect_features"].permute(0, 2, 3, 1)
    positive = source_targets["pos_equal_one"].to(dtype=torch.bool)
    negative = source_targets.get("neg_equal_one")
    negative = (
        negative.to(dtype=torch.bool)
        if negative is not None
        else ~positive
    )
    ego_probability = ego_cls.sigmoid()
    fused_probability = fused_cls.sigmoid()
    ego_max = ego_probability.amax(dim=-1)
    fused_max = fused_probability.amax(dim=-1)
    positive_spatial = positive.any(dim=-1)
    negative_spatial = negative.any(dim=-1)
    positive_count = positive.sum(dim=-1).clamp_min(1)
    ego_refinement_error = (
        ego_reg_error * positive.to(ego_reg_error.dtype)
    ).sum(dim=-1) / positive_count
    fused_refinement_error = (
        fused_reg_error.detach() * positive.to(fused_reg_error.dtype)
    ).sum(dim=-1) / positive_count
    discovery_threshold = float(
        config.get("iada_discovery_ego_threshold", 0.5)
    )
    refinement_threshold = float(
        config.get("iada_refinement_error_threshold", 0.1)
    )
    advantage_threshold = float(
        config.get("iada_target_advantage_threshold", 0.03)
    )
    discovery = (
        positive_spatial
        & (ego_max < discovery_threshold)
        & (fused_max - ego_max >= advantage_threshold)
    )
    suppression = (
        negative_spatial
        & (ego_max > discovery_threshold)
        & (ego_max - fused_max >= advantage_threshold)
    )
    refinement = (
        positive_spatial
        & ~discovery
        & (ego_refinement_error > refinement_threshold)
        & (ego_refinement_error > fused_refinement_error)
    )
    masks = (discovery, suppression, refinement)
    range_index = output_dict["iada_range_index"].long()
    momentum = float(
        output_dict.get(
            "iada_prototype_momentum",
            features.new_tensor(config.get("iada_prototype_momentum", 0.99)),
        ).item()
    )
    update_count = 0
    with torch.no_grad():
        for batch_index in range(features.shape[0]):
            range_value = int(range_index[batch_index].item())
            for mode_index, mask in enumerate(masks):
                selected = features[batch_index][mask[batch_index]]
                if selected.numel() == 0:
                    continue
                batch_prototype = F.normalize(
                    selected.mean(dim=0), dim=0, eps=1.0e-6
                )
                if counts[range_value, mode_index] == 0:
                    prototypes[range_value, mode_index].copy_(batch_prototype)
                else:
                    prototypes[range_value, mode_index].mul_(momentum).add_(
                        batch_prototype, alpha=1.0 - momentum
                    )
                    prototypes[range_value, mode_index].copy_(
                        F.normalize(
                            prototypes[range_value, mode_index],
                            dim=0,
                            eps=1.0e-6,
                        )
                    )
                counts[range_value, mode_index].add_(selected.shape[0])
                update_count += int(selected.shape[0])
    return update_count


def iada_source_adaptation_loss(
    output_dict: Mapping[str, torch.Tensor],
    source_targets: Mapping[str, torch.Tensor],
    config: Mapping[str, object],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Source supervision for safe, useful collaborative corrections."""

    fused_cls_raw = output_dict["cls_preds"]
    fused_reg_raw = output_dict["reg_preds"]
    ego_cls_raw = output_dict["iada_ego_cls_preds"]
    ego_reg_raw = output_dict["iada_ego_reg_preds"]
    anchor_number = int(source_targets["pos_equal_one"].shape[-1])
    fused_cls = _iada_anchor_last(fused_cls_raw, anchor_number).squeeze(-1)
    ego_cls = _iada_anchor_last(ego_cls_raw, anchor_number).squeeze(-1)
    fused_reg = _iada_anchor_last(fused_reg_raw, anchor_number, 7)
    ego_reg = _iada_anchor_last(ego_reg_raw, anchor_number, 7)
    targets = source_targets["targets"].reshape_as(fused_reg)
    positive = source_targets["pos_equal_one"].to(dtype=torch.bool)
    negative = source_targets.get("neg_equal_one")
    valid = positive | (
        negative.to(dtype=torch.bool)
        if negative is not None
        else ~positive
    )
    class_targets = positive.to(fused_cls.dtype)

    fused_cls_error = F.binary_cross_entropy_with_logits(
        fused_cls, class_targets, reduction="none"
    )
    ego_cls_error = F.binary_cross_entropy_with_logits(
        ego_cls, class_targets, reduction="none"
    ).detach()
    fused_reg_error = F.smooth_l1_loss(
        fused_reg, targets, reduction="none"
    ).mean(dim=-1)
    ego_reg_error = F.smooth_l1_loss(
        ego_reg, targets, reduction="none"
    ).mean(dim=-1).detach()
    margin = float(config.get("iada_safe_margin", 0.0))
    safe_cls_terms = []
    for mask in (positive, valid & ~positive):
        if bool(mask.any().item()):
            safe_cls_terms.append(
                F.relu(fused_cls_error - ego_cls_error + margin)[mask].mean()
            )
    safe_cls = (
        torch.stack(safe_cls_terms).mean()
        if safe_cls_terms
        else fused_cls_raw.sum() * 0.0
    )
    if bool(positive.any().item()):
        safe_reg = F.relu(
            fused_reg_error - ego_reg_error + margin
        )[positive].mean()
        # The ego prediction is a fixed counterfactual reference.  Detaching it
        # on only the target side leaves a spurious ``-grad(ego_reg)`` term even
        # though ego_reg cancels algebraically in the forward value.
        ego_reg_reference = ego_reg.detach()
        correction_target = (targets - ego_reg_reference)[positive]
        correction_loss = F.smooth_l1_loss(
            (fused_reg - ego_reg_reference)[positive], correction_target
        )
    else:
        safe_reg = fused_reg_raw.sum() * 0.0
        correction_loss = fused_reg_raw.sum() * 0.0
    safe_loss = safe_cls + safe_reg

    utility_target = (ego_cls_error - fused_cls_error).detach()
    utility_target = utility_target + positive.to(utility_target.dtype) * (
        ego_reg_error - fused_reg_error
    ).detach()
    utility_target = utility_target.tanh()
    utility_logits = output_dict["iada_utility_logits"]
    utility_prediction = _iada_anchor_last(
        utility_logits, anchor_number
    ).squeeze(-1).tanh()
    utility_loss = F.smooth_l1_loss(
        utility_prediction[valid], utility_target[valid]
    )

    memory_updates = _iada_source_effect_memory_update(
        output_dict,
        source_targets,
        ego_cls,
        fused_cls,
        ego_reg_error,
        fused_reg_error,
        config,
    )
    total = (
        float(config.get("iada_safe_weight", 0.0)) * safe_loss
        + float(config.get("iada_correction_weight", 0.0))
        * correction_loss
        + float(config.get("iada_utility_weight", 0.0)) * utility_loss
    )
    metrics = {
        "iada_safe_loss": safe_loss,
        "iada_correction_loss": correction_loss,
        "iada_utility_loss": utility_loss,
        "iada_effect_memory_updates": total.new_tensor(memory_updates),
        "iada_gate_mean": output_dict["iada_gate_mean"].mean(),
        "iada_gate_deviation": output_dict[
            "iada_gate_deviation"
        ].mean(),
        "iada_gate_saturation": output_dict.get(
            "iada_gate_saturation", total.new_zeros(())
        ).mean(),
        "domain_loss": total,
        "domain_accuracy": total.new_tensor(float("nan")),
    }
    return total, metrics


def iada_target_adaptation_loss(
    output_dict: Mapping[str, torch.Tensor],
    config: Mapping[str, object],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Target intervention consistency and conditional effect alignment."""

    zero = output_dict["cls_preds"].sum() * 0.0
    consistency_loss = zero
    effect_loss = zero
    selected_count = 0
    teacher_keys = {
        "iada_consistency_cls_preds",
        "iada_consistency_reg_preds",
        "iada_teacher_cls_preds",
        "iada_teacher_reg_preds",
        "iada_ego_cls_preds",
        "iada_ego_reg_preds",
    }
    if teacher_keys.issubset(output_dict):
        student_cls = output_dict["iada_consistency_cls_preds"]
        student_reg = output_dict["iada_consistency_reg_preds"]
        teacher_cls = output_dict["iada_teacher_cls_preds"].detach()
        teacher_reg = output_dict["iada_teacher_reg_preds"].detach()
        ego_cls = output_dict["iada_ego_cls_preds"]
        ego_reg = output_dict["iada_ego_reg_preds"]
        teacher_probability = teacher_cls.sigmoid()
        ego_probability = ego_cls.detach().sigmoid()
        confidence_threshold = float(
            config.get("iada_target_confidence_threshold", 0.6)
        )
        advantage_threshold = float(
            config.get("iada_target_advantage_threshold", 0.03)
        )
        teacher_advantage = teacher_probability - ego_probability
        teacher_reg_advantage = teacher_reg - ego_reg.detach()
        reliable = (teacher_probability >= confidence_threshold) & (
            (teacher_advantage.abs() >= advantage_threshold)
            | (
                teacher_reg_advantage.reshape(
                    teacher_reg_advantage.shape[0], -1, 7,
                    *teacher_reg_advantage.shape[-2:]
                ).abs().mean(dim=2).amax(dim=1, keepdim=True)
                >= advantage_threshold
            )
        )
        selected_count = int(reliable.sum().item())
        if selected_count:
            # Keep the ego-only counterfactual fixed on both sides.  Otherwise
            # the numerically cancelling ego terms inject an unintended
            # negative gradient into the shared detector on unlabeled target
            # data.
            ego_cls_reference = ego_cls.detach()
            ego_reg_reference = ego_reg.detach()
            student_advantage = (
                student_cls.sigmoid() - ego_cls_reference.sigmoid()
            )
            consistency_cls = F.smooth_l1_loss(
                student_advantage[reliable], teacher_advantage[reliable]
            )
            anchor_mask = reliable.repeat_interleave(7, dim=1)
            consistency_reg = F.smooth_l1_loss(
                (student_reg - ego_reg_reference)[anchor_mask],
                teacher_reg_advantage[anchor_mask],
            )
            consistency_loss = consistency_cls + consistency_reg

        memory_keys = {
            "iada_effect_features",
            "iada_effect_prototypes",
            "iada_effect_counts",
            "iada_range_index",
        }
        if memory_keys.issubset(output_dict):
            spatial_confidence = teacher_probability.amax(dim=1)
            spatial_ego = ego_probability.amax(dim=1)
            spatial_reliable = reliable.any(dim=1)
            discovery = spatial_reliable & (
                spatial_confidence - spatial_ego >= advantage_threshold
            )
            suppression = spatial_reliable & (
                spatial_ego - spatial_confidence >= advantage_threshold
            )
            refinement = spatial_reliable & ~(discovery | suppression)
            effect = output_dict["iada_effect_features"].permute(0, 2, 3, 1)
            prototypes = output_dict["iada_effect_prototypes"].detach()
            counts = output_dict["iada_effect_counts"]
            ranges = output_dict["iada_range_index"].long()
            effect_terms = []
            for batch_index in range(effect.shape[0]):
                range_value = int(ranges[batch_index].item())
                for mode_index, mask in enumerate(
                    (discovery, suppression, refinement)
                ):
                    if counts[range_value, mode_index] <= 0:
                        continue
                    selected = effect[batch_index][mask[batch_index]]
                    if selected.numel() == 0:
                        continue
                    prototype = F.normalize(
                        prototypes[range_value, mode_index],
                        dim=0,
                        eps=1.0e-6,
                    )
                    effect_terms.append(
                        1.0 - (selected * prototype).sum(dim=-1).mean()
                    )
            if effect_terms:
                effect_loss = torch.stack(effect_terms).mean()

    total = (
        float(config.get("iada_consistency_weight", 0.0))
        * consistency_loss
        + float(config.get("iada_effect_weight", 0.0)) * effect_loss
    )
    metrics = {
        "iada_consistency_loss": consistency_loss,
        "iada_effect_loss": effect_loss,
        "iada_target_selected": total.new_tensor(selected_count),
        "iada_target_selected_fraction": reliable.to(total.dtype).mean()
        if teacher_keys.issubset(output_dict)
        else total.new_zeros(()),
        "iada_gate_mean": output_dict["iada_gate_mean"].mean(),
        "iada_gate_deviation": output_dict[
            "iada_gate_deviation"
        ].mean(),
        "iada_gate_saturation": output_dict.get(
            "iada_gate_saturation", total.new_zeros(())
        ).mean(),
        "domain_loss": total,
        "domain_accuracy": total.new_tensor(float("nan")),
    }
    return total, metrics


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
        total_scenes = int(domain_labels.numel())
        if source_scene_count <= 0 or source_scene_count > total_scenes:
            raise ValueError("IADA source scene count is invalid")
        if source_scene_count == total_scenes:
            return iada_source_adaptation_loss(
                output_dict, source_label_dict, config
            )

        def slice_scenes(start: int, end: int):
            sliced = {}
            for key, value in output_dict.items():
                if (
                    torch.is_tensor(value)
                    and value.ndim > 0
                    and value.shape[0] == total_scenes
                    and key
                    not in ("iada_effect_prototypes", "iada_effect_counts")
                ):
                    sliced[key] = value[start:end]
                else:
                    sliced[key] = value
            return sliced

        source_record_len = record_len[:source_scene_count]
        target_record_len = record_len[source_scene_count:]
        domain_enabled = bool((source_record_len > 1).any().item()) and bool(
            (target_record_len > 1).any().item()
        )
        target_std = float(config.get("iada_effect_target_std", 0.05))
        variance_enabled = (
            target_std > 0
            and int((source_record_len > 1).sum().item()) >= 2
            and int((target_record_len > 1).sum().item()) >= 2
        )
        source_loss, source_metrics = compute_single_domain_adaptation_loss(
            "iada",
            "source",
            slice_scenes(0, source_scene_count),
            source_record_len,
            source_label_dict,
            config,
            iada_domain_enabled=domain_enabled,
            iada_variance_enabled=variance_enabled,
        )
        target_loss, target_metrics = compute_single_domain_adaptation_loss(
            "iada",
            "target",
            slice_scenes(source_scene_count, total_scenes),
            target_record_len,
            None,
            config,
            iada_domain_enabled=domain_enabled,
            iada_variance_enabled=variance_enabled,
        )
        for key in source_metrics.keys() | target_metrics.keys():
            values = [
                item[key]
                for item in (source_metrics, target_metrics)
                if key in item
            ]
            if key.endswith(
                (
                    "accuracy",
                    "_mean",
                    "_deviation",
                    "_saturation",
                    "_fraction",
                    "_std",
                )
            ):
                metrics[key] = torch.stack(values).nanmean()
            else:
                metrics[key] = torch.stack(values).sum()
        return source_loss + target_loss, metrics

    if method == "ssda":
        global_loss, global_accuracy, global_count = (
            entropy_weighted_domain_loss(
                output_dict["ssda_global_logits"],
                output_dict["ssda_global_attention"],
                domain_labels,
                output_dict.get("ssda_global_scene_index"),
            )
        )
        local_loss, local_accuracy, local_count = (
            entropy_weighted_domain_loss(
                output_dict["ssda_local_logits"],
                output_dict["ssda_local_attention"],
                domain_labels,
                output_dict.get("ssda_local_scene_index"),
            )
        )
        metrics.update(
            domain_loss=global_loss + local_loss,
            domain_accuracy=torch.stack(
                (global_accuracy, local_accuracy)
            ).nanmean(),
            ssda_global_loss=global_loss,
            ssda_global_accuracy=global_accuracy,
            ssda_global_valid_count=global_loss.new_tensor(global_count),
            ssda_local_loss=local_loss,
            ssda_local_accuracy=local_accuracy,
            ssda_local_valid_count=local_loss.new_tensor(local_count),
        )
        return (
            float(config.get("ssda_global_weight", 0.5)) * global_loss
            + float(config.get("ssda_local_weight", 1.0)) * local_loss,
            metrics,
        )

    raise ValueError(f"Unsupported domain adaptation method: {method!r}")


def compute_single_domain_adaptation_loss(
    method: str,
    domain: str,
    output_dict: Mapping[str, torch.Tensor],
    record_len: torch.Tensor,
    source_label_dict: Optional[Mapping[str, torch.Tensor]],
    config: Mapping[str, object],
    *,
    iada_domain_enabled: bool = True,
    iada_variance_enabled: bool = False,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute one domain's contribution to a balanced DA objective.

    Source and target contributions include their ``0.5`` balance factor, so
    summing the two returned losses is equivalent to
    :func:`compute_adaptation_loss`.  Source-only and target-only auxiliary
    terms (CUDA-X BLC bins and DUSA CIA respectively) retain their full weight.
    """

    if domain not in ("source", "target"):
        raise ValueError("domain must be either 'source' or 'target'")
    domain_label = 0.0 if domain == "source" else 1.0
    method = method.lower().replace("-", "").replace("_", "")
    metrics: Dict[str, torch.Tensor] = {}

    if method in ("grl", "discriminator", "naive"):
        loss, accuracy, valid_count = single_domain_loss(
            output_dict["domain_logits"],
            domain_label,
            output_dict.get("domain_scene_index"),
            output_dict.get("domain_valid_mask"),
        )
        contribution = 0.5 * loss
        metrics.update(
            domain_loss=contribution,
            domain_accuracy=accuracy,
            valid_domain_count=loss.new_tensor(valid_count),
        )
        return (
            float(config.get("domain_loss_weight", 0.1)) * contribution,
            metrics,
        )

    if method == "dusa":
        lsa_loss, lsa_accuracy, lsa_count = single_domain_loss(
            output_dict["domain_logits"],
            domain_label,
            output_dict.get("domain_scene_index"),
            output_dict.get("domain_valid_mask"),
        )
        lsa_contribution = 0.5 * lsa_loss
        if domain == "target":
            target_labels = torch.ones(
                len(record_len), device=lsa_loss.device
            )
            cia_loss, cia_accuracy, cia_count = dusa_agent_loss(
                output_dict["agent_domain_logits"],
                output_dict["agent_domain_weights"],
                output_dict["agent_scene_index"],
                output_dict["agent_local_index"],
                target_labels,
                record_len,
            )
        else:
            cia_loss = lsa_loss.new_zeros(())
            cia_accuracy = lsa_loss.new_tensor(float("nan"))
            cia_count = 0
        metrics.update(
            domain_loss=lsa_contribution + cia_loss,
            domain_accuracy=lsa_accuracy,
            lsa_loss=lsa_contribution,
            lsa_accuracy=lsa_accuracy,
            lsa_valid_count=lsa_loss.new_tensor(lsa_count),
            cia_loss=cia_loss,
            cia_accuracy=cia_accuracy,
            cia_valid_count=lsa_loss.new_tensor(cia_count),
        )
        return (
            float(config.get("dusa_lsa_weight", 1.0)) * lsa_contribution
            + float(config.get("dusa_cia_weight", 1.0)) * cia_loss,
            metrics,
        )

    if method == "cudax":
        domain_losses = []
        domain_accuracies = []
        for name in ("ckt", "blc", "cpa"):
            loss, accuracy, valid_count = single_domain_loss(
                output_dict[f"{name}_domain_logits"],
                domain_label,
                output_dict.get("domain_scene_index"),
                output_dict.get("domain_valid_mask"),
            )
            contribution = 0.5 * loss
            metrics[f"{name}_domain_loss"] = contribution
            metrics[f"{name}_domain_accuracy"] = accuracy
            metrics[f"{name}_valid_count"] = loss.new_tensor(valid_count)
            domain_losses.append(contribution)
            domain_accuracies.append(accuracy)
        domain_loss = torch.stack(domain_losses).sum()
        domain_accuracy = torch.stack(domain_accuracies).nanmean()

        if domain == "source":
            if source_label_dict is None:
                raise ValueError("CUDA-X source loss requires source labels")
            residual_bounds = config.get("cudax_residual_bounds")
            if residual_bounds is None or len(residual_bounds) != 6:
                raise ValueError(
                    "CUDA-X requires six source-only cudax_residual_bounds in "
                    "encoded [x, y, z, h, w, l] order"
                )
            bin_loss = cudax_bin_loss(
                output_dict["bin_logits"],
                source_label_dict,
                int(config.get("cudax_bin_count", 5)),
                residual_bounds,
            )
        else:
            bin_loss = domain_loss.new_zeros(())
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
        if domain == "source":
            if source_label_dict is None:
                raise ValueError("IADA source loss requires source labels")
            auxiliary_loss, auxiliary_metrics = iada_source_adaptation_loss(
                output_dict, source_label_dict, config
            )
        else:
            auxiliary_loss, auxiliary_metrics = iada_target_adaptation_loss(
                output_dict, config
            )
        adversarial_loss, adversarial_metrics = iada_effect_domain_loss(
            output_dict,
            domain_label,
            config,
            enabled=iada_domain_enabled,
            variance_enabled=iada_variance_enabled,
        )
        metrics.update(auxiliary_metrics)
        metrics.update(adversarial_metrics)
        metrics["iada_aux_loss"] = auxiliary_loss
        return auxiliary_loss + adversarial_loss, metrics

    if method == "ssda":
        global_loss, global_accuracy, global_count = (
            single_domain_entropy_weighted_loss(
                output_dict["ssda_global_logits"],
                output_dict["ssda_global_attention"],
                domain_label,
                output_dict.get("ssda_global_scene_index"),
            )
        )
        local_loss, local_accuracy, local_count = (
            single_domain_entropy_weighted_loss(
                output_dict["ssda_local_logits"],
                output_dict["ssda_local_attention"],
                domain_label,
                output_dict.get("ssda_local_scene_index"),
            )
        )
        global_contribution = 0.5 * global_loss
        local_contribution = 0.5 * local_loss
        metrics.update(
            domain_loss=global_contribution + local_contribution,
            domain_accuracy=torch.stack(
                (global_accuracy, local_accuracy)
            ).nanmean(),
            ssda_global_loss=global_contribution,
            ssda_global_accuracy=global_accuracy,
            ssda_global_valid_count=global_loss.new_tensor(global_count),
            ssda_local_loss=local_contribution,
            ssda_local_accuracy=local_accuracy,
            ssda_local_valid_count=local_loss.new_tensor(local_count),
        )
        return (
            float(config.get("ssda_global_weight", 0.5))
            * global_contribution
            + float(config.get("ssda_local_weight", 1.0))
            * local_contribution,
            metrics,
        )

    raise ValueError(f"Unsupported domain adaptation method: {method!r}")


__all__ = [
    "balanced_domain_loss",
    "compute_adaptation_loss",
    "compute_single_domain_adaptation_loss",
    "cudax_bin_loss",
    "dusa_agent_loss",
    "entropy_weighted_domain_loss",
    "graph_variance_floor_loss",
    "iada_effect_domain_loss",
    "iada_source_adaptation_loss",
    "iada_target_adaptation_loss",
    "single_domain_entropy_weighted_loss",
    "single_domain_conditioned_loss",
    "single_domain_graph_variance_loss",
    "single_domain_loss",
]
