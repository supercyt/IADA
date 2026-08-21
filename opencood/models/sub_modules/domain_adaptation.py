"""Fusion-agnostic domain adapters for collaborative PointPillar models.

Every adapter consumes the same two representations:

``agent_features``
    Per-agent BEV maps before collaboration, flattened as
    ``[sum(record_len), C, H, W]``.

``fused_features``
    Ego-centric collaborative BEV maps after an arbitrary fusion method,
    shaped ``[B, C, H, W]``.

This interface deliberately keeps domain adaptation outside individual fusion
operators.  In particular, no adapter produces attention logits or assumes an
AttFuse-specific API.
"""

from __future__ import annotations

import copy
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from opencood.models.sub_modules.interaction_da import (
    GradientReversal,
)


def _record_len_tensor(record_len, device: torch.device) -> torch.Tensor:
    counts = torch.as_tensor(record_len, dtype=torch.long, device=device)
    if counts.ndim != 1 or counts.numel() == 0:
        raise ValueError("record_len must be a non-empty one-dimensional tensor")
    if bool((counts <= 0).any().item()):
        raise ValueError("every scene must contain at least one agent")
    return counts


def scene_and_local_indices(
    record_len,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Map flattened agents to their scene and within-scene indices."""

    counts = _record_len_tensor(record_len, device)
    scene_indices = torch.repeat_interleave(
        torch.arange(counts.numel(), device=device), counts
    )
    local_indices = torch.cat(
        [torch.arange(int(count.item()), device=device) for count in counts]
    )
    return scene_indices, local_indices


def extract_ego_features(
    agent_features: torch.Tensor,
    record_len,
) -> torch.Tensor:
    """Select the first (ego) agent from every flattened scene."""

    counts = _record_len_tensor(record_len, agent_features.device)
    if int(counts.sum().item()) != agent_features.shape[0]:
        raise ValueError(
            "sum(record_len) must equal the number of agent feature maps"
        )
    starts = torch.cat(
        (
            counts.new_zeros(1),
            counts.cumsum(dim=0)[:-1],
        )
    )
    return agent_features[starts]


def _coordinate_encoding(
    reference: torch.Tensor,
    lidar_range: Optional[Tuple[float, ...]] = None,
) -> torch.Tensor:
    """Return the absolute metric y/x encoding used by official DUSA."""

    if reference.ndim != 4:
        raise ValueError("BEV features must have shape [N, C, H, W]")
    height, width = reference.shape[-2:]
    if lidar_range is None or len(lidar_range) != 6:
        raise ValueError(
            "DUSA metric position encoding requires a six-value lidar_range"
        )
    x_limits = (float(lidar_range[0]), float(lidar_range[3]))
    y_limits = (float(lidar_range[1]), float(lidar_range[4]))
    y = torch.linspace(
        y_limits[0],
        y_limits[1],
        height,
        device=reference.device,
        dtype=reference.dtype,
    )
    x = torch.linspace(
        x_limits[0],
        x_limits[1],
        width,
        device=reference.device,
        dtype=reference.dtype,
    )
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    coordinates = torch.stack((grid_y, grid_x), dim=0).unsqueeze(0)
    return coordinates.abs().expand(reference.shape[0], -1, -1, -1)


class GlobalDomainDiscriminator(nn.Module):
    """GRL followed by global pooling and a small binary discriminator."""

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or hidden_dim <= 0:
            raise ValueError("discriminator dimensions must be positive")
        if dropout < 0 or dropout >= 1:
            raise ValueError("dropout must be in [0, 1)")
        layers = [
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, 1))
        self.gradient_reversal = GradientReversal()
        self.classifier = nn.Sequential(*layers)

    def forward(
        self,
        features: torch.Tensor,
        grl_lambda: float,
    ) -> torch.Tensor:
        if features.ndim == 4:
            features = features.mean(dim=(-2, -1))
        if features.ndim != 2:
            raise ValueError("domain features must have shape [N, C] or [N,C,H,W]")
        reversed_features = self.gradient_reversal(
            features, coefficient=grl_lambda
        )
        return self.classifier(reversed_features)


class FusionAgnosticDomainAdapter(nn.Module):
    """Common two-phase API used by all domain adaptation methods."""

    method = "none"
    requires_agent_confidence = False

    def adapt_agents(
        self,
        agent_features: torch.Tensor,
        record_len: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Optionally refine per-agent features before collaboration."""

        return agent_features, {}

    def adapt_fused(
        self,
        agent_features: torch.Tensor,
        fused_features: torch.Tensor,
        record_len: torch.Tensor,
        grl_lambda: float,
        pairwise_t_matrix: Optional[torch.Tensor] = None,
        adapter_domain: Optional[str] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return fused_features, {}

    def forward(
        self,
        agent_features: torch.Tensor,
        fused_features: torch.Tensor,
        record_len: torch.Tensor,
        pairwise_t_matrix: torch.Tensor,
        grl_lambda: float,
        agent_confidence_logits: Optional[torch.Tensor] = None,
        fused_class_logits: Optional[torch.Tensor] = None,
        context: Optional[Dict[str, torch.Tensor]] = None,
        detection_features: Optional[torch.Tensor] = None,
        adapter_domain: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError


class NaiveDomainAdapter(FusionAgnosticDomainAdapter):
    """Standard agent-level global adversarial feature alignment."""

    method = "grl"

    def __init__(self, in_channels: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.domain_discriminator = GlobalDomainDiscriminator(
            in_channels, hidden_dim
        )

    def forward(
        self,
        agent_features: torch.Tensor,
        fused_features: torch.Tensor,
        record_len: torch.Tensor,
        pairwise_t_matrix: torch.Tensor,
        grl_lambda: float,
        agent_confidence_logits: Optional[torch.Tensor] = None,
        fused_class_logits: Optional[torch.Tensor] = None,
        context: Optional[Dict[str, torch.Tensor]] = None,
        detection_features: Optional[torch.Tensor] = None,
        adapter_domain: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        scene_indices, _ = scene_and_local_indices(
            record_len, agent_features.device
        )
        return {
            "domain_logits": self.domain_discriminator(
                agent_features, grl_lambda
            ),
            "domain_scene_index": scene_indices,
            "domain_valid_mask": torch.ones_like(
                scene_indices, dtype=torch.bool
            ),
        }


class DUSAAdapter(FusionAgnosticDomainAdapter):
    """Paper-faithful LSA/CIA implementation of DUSA.

    LSA aligns only ego-agent features across simulation and reality using a
    learnable location selector. CIA predicts the target-domain agent role at
    every BEV location and exposes confidence weights computed as the minimum
    confidence shared by all agents in a scene.
    """

    method = "dusa"
    requires_agent_confidence = True

    def __init__(
        self,
        in_channels: int,
        lsa_hidden_dim: int = 1024,
        cia_hidden_dim: int = 512,
        lsa_grl_scale: float = 0.05,
        cia_grl_scale: float = 0.1,
        lidar_range: Optional[Tuple[float, ...]] = None,
        feature_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        self.lsa_grl_scale = float(lsa_grl_scale)
        self.cia_grl_scale = float(cia_grl_scale)
        if lidar_range is None or len(lidar_range) != 6:
            raise ValueError("DUSA requires the detector lidar_range")
        self.lidar_range = tuple(lidar_range)
        if feature_size is None or len(feature_size) != 2:
            raise ValueError(
                "DUSA requires domain_adapter.feature_size=[H, W] for its "
                "learnable location selection map"
            )
        feature_height, feature_width = (int(value) for value in feature_size)
        if feature_height <= 0 or feature_width <= 0:
            raise ValueError("DUSA feature_size values must be positive")
        self.location_selection_map = nn.Parameter(
            torch.ones(1, 1, feature_height, feature_width)
        )
        self.gradient_reversal = GradientReversal()
        self.sim_real_discriminator = nn.Sequential(
            nn.Linear(in_channels + 2, lsa_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(lsa_hidden_dim, lsa_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(lsa_hidden_dim, 1),
        )
        self.agent_discriminator = nn.Sequential(
            nn.Conv2d(in_channels + 2, cia_hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(cia_hidden_dim, 1, kernel_size=1),
        )
        self._initialize_discriminators()

    def _initialize_discriminators(self) -> None:
        linear_layers = [
            module
            for module in self.sim_real_discriminator
            if isinstance(module, nn.Linear)
        ]
        for layer in linear_layers[:-1]:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.zeros_(layer.bias)
        nn.init.normal_(linear_layers[-1].weight, std=0.05)
        nn.init.zeros_(linear_layers[-1].bias)
        for layer in self.agent_discriminator:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, std=0.001)
                nn.init.zeros_(layer.bias)

    @staticmethod
    def _confidence_weights(
        confidence_logits: torch.Tensor,
        record_len: torch.Tensor,
    ) -> torch.Tensor:
        confidence = confidence_logits.detach().sigmoid().mean(
            dim=1, keepdim=True
        )
        counts = _record_len_tensor(record_len, confidence.device)
        split_confidence = torch.split(
            confidence, [int(count.item()) for count in counts], dim=0
        )
        weights = []
        for scene_confidence in split_confidence:
            shared = scene_confidence.amin(dim=0, keepdim=True)
            weights.append(shared.expand(scene_confidence.shape[0], -1, -1, -1))
        return torch.cat(weights, dim=0)

    def forward(
        self,
        agent_features: torch.Tensor,
        fused_features: torch.Tensor,
        record_len: torch.Tensor,
        pairwise_t_matrix: torch.Tensor,
        grl_lambda: float,
        agent_confidence_logits: Optional[torch.Tensor] = None,
        fused_class_logits: Optional[torch.Tensor] = None,
        context: Optional[Dict[str, torch.Tensor]] = None,
        detection_features: Optional[torch.Tensor] = None,
        adapter_domain: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        if adapter_domain != "source" and agent_confidence_logits is None:
            raise ValueError("DUSA requires per-agent confidence logits")
        if (
            agent_confidence_logits is not None
            and agent_confidence_logits.shape[0] != agent_features.shape[0]
        ):
            raise ValueError("DUSA confidence and feature agent counts differ")

        coordinates = _coordinate_encoding(agent_features, self.lidar_range)
        ego_features = extract_ego_features(agent_features, record_len)
        ego_coordinates = extract_ego_features(coordinates, record_len)
        if self.location_selection_map.shape[-2:] != ego_features.shape[-2:]:
            raise ValueError(
                "DUSA feature_size does not match the adapter feature map: "
                f"{tuple(self.location_selection_map.shape[-2:])} != "
                f"{tuple(ego_features.shape[-2:])}"
            )
        reversed_ego_features = self.gradient_reversal(
            torch.cat((ego_features, ego_coordinates), dim=1),
            coefficient=self.lsa_grl_scale,
        )
        selector = torch.sigmoid(self.location_selection_map)
        selected_ego = (reversed_ego_features * selector).mean(
            dim=(-2, -1)
        )

        lsa_output = {
            "domain_logits": self.sim_real_discriminator(selected_ego),
            "domain_scene_index": torch.arange(
                len(record_len), device=agent_features.device
            ),
            "domain_valid_mask": torch.ones(
                len(record_len),
                dtype=torch.bool,
                device=agent_features.device,
            ),
        }
        if adapter_domain == "source":
            return lsa_output

        reversed_agent_features = self.gradient_reversal(
            torch.cat((agent_features, coordinates), dim=1),
            coefficient=self.cia_grl_scale,
        )
        agent_logits = self.agent_discriminator(reversed_agent_features)
        scene_indices, local_indices = scene_and_local_indices(
            record_len, agent_features.device
        )
        return {
            **lsa_output,
            "agent_domain_logits": agent_logits,
            "agent_domain_weights": self._confidence_weights(
                agent_confidence_logits, record_len
            ),
            "agent_scene_index": scene_indices,
            "agent_local_index": local_indices,
        }


class IADAAdapter(FusionAgnosticDomainAdapter):
    """Interventional advantage adaptation for collaborative perception.

    IADA treats the ego-only feature as a counterfactual control and learns a
    bounded, identity-initialized modulation of the innovation introduced by
    collaboration.  The adapter deliberately avoids a domain discriminator:
    source supervision and target consistency are applied to the *change* in
    predictions between ego-only and collaborative inference.
    """

    method = "iada"

    def __init__(
        self,
        in_channels: int,
        anchor_number: int,
        hidden_dim: int = 64,
        effect_dim: int = 64,
        geometry_scale: float = 100.0,
        gate_limit: float = 1.0,
        source_supervision_enabled: bool = True,
        target_consistency_enabled: bool = True,
        effect_memory_enabled: bool = True,
        consistency_dropout: float = 0.1,
        teacher_momentum: float = 0.999,
        prototype_momentum: float = 0.99,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or anchor_number <= 0:
            raise ValueError("IADA channel and anchor counts must be positive")
        if hidden_dim <= 0 or effect_dim <= 0 or geometry_scale <= 0:
            raise ValueError("IADA hidden/effect dimensions and scale are invalid")
        if gate_limit < 0 or not 0 <= consistency_dropout < 1:
            raise ValueError("IADA gate limit/dropout is invalid")
        if not 0 <= teacher_momentum < 1 or not 0 <= prototype_momentum < 1:
            raise ValueError("IADA EMA momenta must be in [0, 1)")

        self.in_channels = int(in_channels)
        self.anchor_number = int(anchor_number)
        self.geometry_scale = float(geometry_scale)
        self.gate_limit = float(gate_limit)
        self.source_supervision_enabled = bool(source_supervision_enabled)
        self.target_consistency_enabled = bool(target_consistency_enabled)
        self.effect_memory_enabled = bool(effect_memory_enabled)
        self.consistency_dropout = float(consistency_dropout)
        self.teacher_momentum = float(teacher_momentum)
        self.prototype_momentum = float(prototype_momentum)

        group_count = min(8, hidden_dim)
        while hidden_dim % group_count:
            group_count -= 1
        self.innovation_encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
            nn.GroupNorm(group_count, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
        )
        self.geometry_encoder = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gate_head = nn.Conv2d(hidden_dim, in_channels, kernel_size=1)
        # Zero initialization makes calibrated == native fused at step zero.
        nn.init.zeros_(self.gate_head.weight)
        nn.init.zeros_(self.gate_head.bias)
        self.effect_head = nn.Sequential(
            nn.Conv2d(hidden_dim, effect_dim, kernel_size=1),
            nn.SiLU(inplace=True),
        )
        self.utility_head = nn.Conv2d(
            effect_dim, anchor_number, kernel_size=1
        )

        self.teacher_innovation_encoder = copy.deepcopy(
            self.innovation_encoder
        )
        self.teacher_geometry_encoder = copy.deepcopy(self.geometry_encoder)
        self.teacher_gate_head = copy.deepcopy(self.gate_head)
        for module in (
            self.teacher_innovation_encoder,
            self.teacher_geometry_encoder,
            self.teacher_gate_head,
        ):
            module.requires_grad_(False)

        # Three effect modes (discovery, suppression, refinement) by three
        # geometry ranges (near, middle, far).
        self.register_buffer(
            "effect_prototypes", torch.zeros(3, 3, effect_dim)
        )
        self.register_buffer("effect_counts", torch.zeros(3, 3))

    @staticmethod
    def _channel_normalize(features: torch.Tensor) -> torch.Tensor:
        statistics = features.float()
        mean = statistics.mean(dim=1, keepdim=True)
        variance = statistics.var(dim=1, keepdim=True, unbiased=False)
        return ((statistics - mean) * torch.rsqrt(variance + 1.0e-5)).to(
            features.dtype
        )

    def _geometry_context(
        self,
        record_len: torch.Tensor,
        pairwise_t_matrix: Optional[torch.Tensor],
        reference: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = int(record_len.numel())
        context = reference.new_zeros((batch_size, 5))
        if pairwise_t_matrix is not None and pairwise_t_matrix.shape[1] > 1:
            transforms = pairwise_t_matrix[:, 0, 1].to(reference)
            has_partner = (record_len > 1).to(reference.dtype)
            dx = transforms[:, 0, 3] / self.geometry_scale
            dy = transforms[:, 1, 3] / self.geometry_scale
            distance = torch.sqrt(dx.square() + dy.square())
            yaw = torch.atan2(transforms[:, 1, 0], transforms[:, 0, 0])
            context = torch.stack(
                (dx, dy, distance, torch.sin(yaw), torch.cos(yaw)), dim=1
            ) * has_partner.unsqueeze(1)
        distance_meters = context[:, 2].abs() * self.geometry_scale
        range_index = torch.bucketize(
            distance_meters,
            reference.new_tensor((30.0, 60.0)),
        )
        return context, range_index

    @staticmethod
    @torch.no_grad()
    def _ema_update(teacher: nn.Module, student: nn.Module, momentum: float):
        for teacher_parameter, student_parameter in zip(
            teacher.parameters(), student.parameters()
        ):
            teacher_parameter.mul_(momentum).add_(
                student_parameter, alpha=1.0 - momentum
            )
        for teacher_buffer, student_buffer in zip(
            teacher.buffers(), student.buffers()
        ):
            teacher_buffer.copy_(student_buffer)

    @torch.no_grad()
    def update_teacher(self) -> None:
        self._ema_update(
            self.teacher_innovation_encoder,
            self.innovation_encoder,
            self.teacher_momentum,
        )
        self._ema_update(
            self.teacher_geometry_encoder,
            self.geometry_encoder,
            self.teacher_momentum,
        )
        self._ema_update(
            self.teacher_gate_head,
            self.gate_head,
            self.teacher_momentum,
        )

    def _encode_innovation(
        self,
        innovation: torch.Tensor,
        geometry: torch.Tensor,
        *,
        teacher: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if teacher:
            hidden = self.teacher_innovation_encoder(innovation)
            geometry_hidden = self.teacher_geometry_encoder(geometry)
            gate_logits = self.teacher_gate_head(
                hidden + geometry_hidden[:, :, None, None]
            )
        else:
            hidden = self.innovation_encoder(innovation)
            geometry_hidden = self.geometry_encoder(geometry)
            hidden = hidden + geometry_hidden[:, :, None, None]
            gate_logits = self.gate_head(hidden)
        gate = 1.0 + self.gate_limit * torch.tanh(gate_logits)
        return hidden, gate

    def adapt_fused(
        self,
        agent_features: torch.Tensor,
        fused_features: torch.Tensor,
        record_len: torch.Tensor,
        grl_lambda: float,
        pairwise_t_matrix: Optional[torch.Tensor] = None,
        adapter_domain: Optional[str] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        del grl_lambda
        ego_features = extract_ego_features(agent_features, record_len)
        if ego_features.shape != fused_features.shape:
            raise ValueError("IADA ego and fused feature shapes must match")
        raw_innovation = fused_features - ego_features
        # Adapter context should not create an auxiliary shortcut into the
        # warm-started detector.  Detection gradients still reach the detector
        # through ``raw_innovation`` in ``calibrated`` below.
        normalized_innovation = (
            self._channel_normalize(fused_features)
            - self._channel_normalize(ego_features)
        ).detach()
        geometry, range_index = self._geometry_context(
            record_len, pairwise_t_matrix, fused_features
        )
        hidden, gate = self._encode_innovation(
            normalized_innovation, geometry
        )
        calibrated = ego_features + gate * raw_innovation
        effect = F.normalize(
            self.effect_head(hidden), p=2, dim=1, eps=1.0e-6
        )
        context = {
            "iada_ego_features": ego_features,
            "iada_raw_innovation": raw_innovation,
            "iada_effect_features": effect,
            "iada_utility_logits": self.utility_head(effect),
            "iada_gate_mean": gate.mean().reshape(1),
            "iada_gate_deviation": (gate - 1.0).abs().mean().reshape(1),
            "iada_gate_saturation": (
                (gate - 1.0).abs()
                >= max(0.95 * self.gate_limit, 1.0e-6)
            ).to(gate.dtype).mean().reshape(1),
            "iada_range_index": range_index,
        }

        if (
            self.training
            and adapter_domain == "target"
            and self.target_consistency_enabled
        ):
            self.update_teacher()
            dropped_innovation = F.dropout2d(
                raw_innovation,
                p=self.consistency_dropout,
                training=self.consistency_dropout > 0,
            )
            context["iada_consistency_features"] = (
                ego_features + gate * dropped_innovation
            )
            with torch.no_grad():
                _, teacher_gate = self._encode_innovation(
                    normalized_innovation.detach(),
                    geometry.detach(),
                    teacher=True,
                )
                context["iada_teacher_features"] = (
                    ego_features.detach()
                    + teacher_gate * raw_innovation.detach()
                )

        return calibrated, context

    def forward(
        self,
        agent_features: torch.Tensor,
        fused_features: torch.Tensor,
        record_len: torch.Tensor,
        pairwise_t_matrix: torch.Tensor,
        grl_lambda: float,
        agent_confidence_logits: Optional[torch.Tensor] = None,
        fused_class_logits: Optional[torch.Tensor] = None,
        context: Optional[Dict[str, torch.Tensor]] = None,
        detection_features: Optional[torch.Tensor] = None,
        adapter_domain: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        del (
            agent_features,
            fused_features,
            record_len,
            pairwise_t_matrix,
            grl_lambda,
            agent_confidence_logits,
            fused_class_logits,
            detection_features,
            adapter_domain,
        )
        if context is None or "iada_effect_features" not in context:
            raise ValueError("IADA adapt_fused must run before its heads")
        output_keys = (
            "iada_effect_features",
            "iada_utility_logits",
            "iada_gate_mean",
            "iada_gate_deviation",
            "iada_gate_saturation",
            "iada_range_index",
            "iada_ego_cls_preds",
            "iada_ego_reg_preds",
            "iada_consistency_cls_preds",
            "iada_consistency_reg_preds",
            "iada_teacher_cls_preds",
            "iada_teacher_reg_preds",
        )
        output = {key: context[key] for key in output_keys if key in context}
        if self.effect_memory_enabled:
            output["iada_effect_prototypes"] = self.effect_prototypes
            output["iada_effect_counts"] = self.effect_counts
            output["iada_prototype_momentum"] = self.effect_counts.new_tensor(
                self.prototype_momentum
            )
        return output


class HaarWaveletReconstruction(nn.Module):
    """Fixed depth-wise 2D Haar analysis and per-band reconstruction.

    The four returned channel groups are the reconstructed LL, LH, HL, and HH
    components from equations (1)-(3) of Selective Shift. Padding is removed
    after reconstruction so the module also supports odd-sized feature maps.
    """

    def __init__(self) -> None:
        super().__init__()
        filters = torch.tensor(
            (
                ((1.0, 1.0), (1.0, 1.0)),
                ((-1.0, -1.0), (1.0, 1.0)),
                ((-1.0, 1.0), (-1.0, 1.0)),
                ((1.0, -1.0), (-1.0, 1.0)),
            )
        ).unsqueeze(1) / 2.0
        self.register_buffer("filters", filters, persistent=True)

    def _analysis(
        self,
        features: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        if features.ndim != 4:
            raise ValueError("Haar input must have shape [N, C, H, W]")
        if not features.is_floating_point():
            raise TypeError("Haar input must use a floating-point dtype")

        height, width = features.shape[-2:]
        pad_bottom = height % 2
        pad_right = width % 2
        padded = F.pad(
            features,
            (0, pad_right, 0, pad_bottom),
            mode="replicate",
        )
        channels = features.shape[1]
        analysis_filters = self.filters.to(
            device=features.device, dtype=features.dtype
        ).repeat(channels, 1, 1, 1)
        coefficients = F.conv2d(
            padded,
            analysis_filters,
            stride=2,
            groups=channels,
        )
        coefficient_height, coefficient_width = coefficients.shape[-2:]
        coefficients = coefficients.reshape(
            features.shape[0],
            channels,
            4,
            coefficient_height,
            coefficient_width,
        )
        return coefficients, (height, width)

    def _reconstruct_band(
        self,
        coefficients: torch.Tensor,
        band: int,
        output_size: Tuple[int, int],
    ) -> torch.Tensor:
        channels = coefficients.shape[1]
        filters = self.filters.to(
            device=coefficients.device, dtype=coefficients.dtype
        )
        synthesis_filter = filters[band : band + 1].repeat(
            channels, 1, 1, 1
        )
        component = F.conv_transpose2d(
            coefficients[:, :, band],
            synthesis_filter,
            stride=2,
            groups=channels,
        )
        height, width = output_size
        return component[..., :height, :width]

    def band_summaries(self, features: torch.Tensor) -> torch.Tensor:
        """Return channel-mean reconstructed bands without a 4C allocation."""

        coefficients, output_size = self._analysis(features)
        summaries = []
        for band in range(4):
            component = self._reconstruct_band(
                coefficients, band, output_size
            )
            summaries.append(component.mean(dim=1, keepdim=True))
        return torch.cat(summaries, dim=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        coefficients, output_size = self._analysis(features)
        reconstructed = []
        for band in range(4):
            reconstructed.append(
                self._reconstruct_band(coefficients, band, output_size)
            )
        return torch.cat(reconstructed, dim=1)


class FrequencyShiftAdjustment(nn.Module):
    """Frequency-decoupled feature shift adjustment (FSA)."""

    def __init__(
        self,
        channels: int,
        sao_enabled: bool = True,
        sao_probability: float = 1.0,
        epsilon: float = 1.0e-5,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("FSA channels must be positive")
        if sao_probability < 0 or sao_probability > 1:
            raise ValueError("sao_probability must be in [0, 1]")
        if epsilon <= 0:
            raise ValueError("FSA epsilon must be positive")
        self.sao_enabled = bool(sao_enabled)
        self.sao_probability = float(sao_probability)
        self.epsilon = float(epsilon)
        self.wavelet = HaarWaveletReconstruction()
        # Summing four per-band convolutions is equivalent to applying one
        # localized grouped convolution to Cat(LL, LH, HL, HH), while allowing
        # checkpointed reconstruction without a [N, 4C, H, W] allocation.
        self.frequency_attention = nn.ModuleList(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
            )
            for _ in range(4)
        )
        self.statistical_affine = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            groups=channels,
        )
        # Both learned multiplicative maps start neutral; SAO normalization is
        # intentionally active from the beginning of SSDA training.
        for layer in self.frequency_attention:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.statistical_affine.weight)
        nn.init.zeros_(self.statistical_affine.bias)

    @staticmethod
    def _share_scene_maps(
        maps: torch.Tensor,
        record_len: torch.Tensor,
    ) -> torch.Tensor:
        counts = _record_len_tensor(record_len, maps.device)
        if int(counts.sum().item()) != maps.shape[0]:
            raise ValueError("FSA record_len does not match agent features")
        shared = []
        offset = 0
        for count_tensor in counts:
            count = int(count_tensor.item())
            scene_map = maps[offset : offset + count].mean(
                dim=0, keepdim=True
            )
            shared.append(scene_map.expand(count, -1, -1, -1))
            offset += count
        return torch.cat(shared, dim=0)

    def _donor_indices(
        self,
        record_len: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        counts = _record_len_tensor(record_len, device)
        indices = []
        offset = 0
        for count_tensor in counts:
            count = int(count_tensor.item())
            if count == 1:
                permutation = torch.zeros(
                    1, dtype=torch.long, device=counts.device
                )
            else:
                # A cyclic non-zero shift guarantees that every agent receives
                # another agent's statistics while remaining scene-local.
                shift = int(
                    torch.randint(1, count, (), device=counts.device).item()
                )
                permutation = (
                    torch.arange(count, device=counts.device) + shift
                ) % count
            indices.append(permutation + offset)
            offset += count
        return torch.cat(indices)

    def forward(
        self,
        features: torch.Tensor,
        record_len: torch.Tensor,
    ) -> torch.Tensor:
        coefficients, output_size = self.wavelet._analysis(features)
        agent_maps = None
        for band, refinement in enumerate(self.frequency_attention):
            def refine_band(values, band_index=band, layer=refinement):
                component = self.wavelet._reconstruct_band(
                    values, band_index, output_size
                )
                return layer(component)

            if self.training and coefficients.requires_grad:
                band_map = checkpoint(
                    refine_band,
                    coefficients,
                    use_reentrant=False,
                )
            else:
                band_map = refine_band(coefficients)
            agent_maps = (
                band_map if agent_maps is None else agent_maps + band_map
            )
        shared_maps = self._share_scene_maps(agent_maps, record_len)
        frequency_weights = 1.0 + torch.tanh(shared_maps)
        refined = frequency_weights * features

        # SAO randomly exchanges agent statistics during training.  The paper
        # does not specify stochastic obfuscation at inference, so evaluation
        # keeps the deterministic refined feature while still applying SFW.
        apply_sao = self.training and self.sao_enabled and (
            self.sao_probability > 0
        )
        obfuscated = refined
        if apply_sao:
            should_swap = self.sao_probability >= 1.0 or bool(
                torch.rand((), device=features.device).item()
                < self.sao_probability
            )
            if should_swap:
                donor_features = refined[
                    self._donor_indices(record_len, refined.device)
                ].float()
                donor_mean = donor_features.mean(
                    dim=(-2, -1), keepdim=True
                )
                donor_variance = donor_features.var(
                    dim=(-2, -1), keepdim=True, unbiased=False
                )
                # Selective Shift, equation (6): normalize the current
                # feature with the randomly selected agent's mean and
                # variance.  Statistics are accumulated in float32 under AMP.
                obfuscated = (
                    (refined.float() - donor_mean)
                    / (donor_variance + self.epsilon)
                ).to(refined.dtype)
        statistical_weights = 1.0 + torch.tanh(
            self.statistical_affine(obfuscated)
        )
        return statistical_weights * obfuscated


class StagedAdaptiveAlignment(nn.Module):
    """Entropy-driven global and local alignment heads (SAA)."""

    def __init__(
        self,
        channels: int,
        hidden_dim: int = 128,
        detach_attention: bool = False,
    ) -> None:
        super().__init__()
        if channels <= 0 or hidden_dim <= 0:
            raise ValueError("SAA dimensions must be positive")
        self.detach_attention = bool(detach_attention)
        self.gradient_reversal = GradientReversal()
        self.global_classifier = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )
        self.local_classifier = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )
        self.local_attention_fusion = nn.Conv2d(2, 1, kernel_size=1)

    @staticmethod
    def _entropy(probabilities: torch.Tensor) -> torch.Tensor:
        epsilon = torch.finfo(probabilities.dtype).eps
        probabilities = probabilities.clamp(epsilon, 1.0 - epsilon)
        return -(
            probabilities * probabilities.log()
            + (1.0 - probabilities) * (1.0 - probabilities).log()
        )

    def forward(
        self,
        agent_features: torch.Tensor,
        record_len: torch.Tensor,
        grl_lambda: float,
        agent_class_logits: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if agent_class_logits.shape[0] != agent_features.shape[0]:
            raise ValueError("SAA class logits and agent features differ")
        if agent_class_logits.shape[-2:] != agent_features.shape[-2:]:
            raise ValueError("SAA class logits and features must share H/W")
        agent_class_probabilities = agent_class_logits.sigmoid()
        agent_entropy = self._entropy(agent_class_probabilities)
        ego_class_probabilities = extract_ego_features(
            agent_class_probabilities, record_len
        )
        ego_entropy = self._entropy(ego_class_probabilities)
        global_attention = ego_entropy.amin(dim=1, keepdim=True)

        local_entropy = (
            agent_entropy.amin(dim=1, keepdim=True)
            + agent_entropy.amax(dim=1, keepdim=True)
        ) / 2.0
        volume_attention = agent_class_probabilities.amin(
            dim=1, keepdim=True
        )
        mixture = torch.sigmoid(
            self.local_attention_fusion(
                torch.cat((local_entropy, volume_attention), dim=1)
            )
        )
        local_attention = (
            mixture * volume_attention
            + (1.0 - mixture) * local_entropy
        )
        if self.detach_attention:
            global_attention = global_attention.detach()
            local_attention = local_attention.detach()

        ego_features = extract_ego_features(agent_features, record_len)
        global_logits = self.global_classifier(
            self.gradient_reversal(
                ego_features, coefficient=grl_lambda
            )
        )
        local_logits = self.local_classifier(
            self.gradient_reversal(
                agent_features, coefficient=grl_lambda
            )
        )
        scene_indices, local_indices = scene_and_local_indices(
            record_len, agent_features.device
        )
        return {
            "ssda_global_logits": global_logits,
            "ssda_global_attention": global_attention,
            "ssda_global_scene_index": torch.arange(
                len(record_len), device=agent_features.device
            ),
            "ssda_local_logits": local_logits,
            "ssda_local_attention": local_attention,
            "ssda_local_scene_index": scene_indices,
            "ssda_local_agent_index": local_indices,
        }


class SSDAAdapter(FusionAgnosticDomainAdapter):
    """Selective Shift domain adaptation with FSA and SAA."""

    method = "ssda"
    requires_agent_confidence = True

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 128,
        sao_enabled: bool = True,
        sao_probability: float = 1.0,
        fsa_epsilon: float = 1.0e-5,
        detach_attention: bool = False,
    ) -> None:
        super().__init__()
        self.fsa = FrequencyShiftAdjustment(
            in_channels,
            sao_enabled=sao_enabled,
            sao_probability=sao_probability,
            epsilon=fsa_epsilon,
        )
        self.saa = StagedAdaptiveAlignment(
            in_channels,
            hidden_dim=hidden_dim,
            detach_attention=detach_attention,
        )

    def adapt_agents(
        self,
        agent_features: torch.Tensor,
        record_len: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return self.fsa(agent_features, record_len), {}

    def forward(
        self,
        agent_features: torch.Tensor,
        fused_features: torch.Tensor,
        record_len: torch.Tensor,
        pairwise_t_matrix: torch.Tensor,
        grl_lambda: float,
        agent_confidence_logits: Optional[torch.Tensor] = None,
        fused_class_logits: Optional[torch.Tensor] = None,
        context: Optional[Dict[str, torch.Tensor]] = None,
        detection_features: Optional[torch.Tensor] = None,
        adapter_domain: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        if agent_confidence_logits is None:
            raise ValueError("SSDA requires per-agent classification logits")
        return self.saa(
            agent_features,
            record_len,
            grl_lambda,
            agent_confidence_logits,
        )


class CollaborativeKnowledgeTransfer(nn.Module):
    """Spatial/channel reconstruction used by the CUDA-X CKT branch."""

    def __init__(
        self,
        channels: int,
        groups: int = 8,
        shuffle_groups: int = 2,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("CKT channels must be positive")
        if groups <= 0:
            raise ValueError("CKT groups must be positive")
        if channels % groups != 0:
            raise ValueError(
                "CKT channels must be divisible by ckt_groups; got "
                f"{channels} channels and {groups} groups"
            )
        self.groups = groups
        self.slice_channels = channels // groups
        if shuffle_groups <= 0 or self.slice_channels % shuffle_groups != 0:
            raise ValueError(
                "each CKT slice must be divisible by ckt_shuffle_groups"
            )
        self.shuffle_groups = shuffle_groups
        self.channel_gates = nn.ModuleList(
            [
                nn.Conv2d(
                    self.slice_channels,
                    self.slice_channels,
                    kernel_size=1,
                )
                for _ in range(groups)
            ]
        )
        self.spatial_norms = nn.ModuleList(
            [nn.LayerNorm(self.slice_channels) for _ in range(groups)]
        )
        self.spatial_gates = nn.ModuleList(
            [nn.Linear(self.slice_channels, 1) for _ in range(groups)]
        )

    def forward(
        self,
        ego_features: torch.Tensor,
        fused_features: torch.Tensor,
    ) -> torch.Tensor:
        if ego_features.shape != fused_features.shape:
            raise ValueError("CKT ego and fused feature shapes must match")
        ego_slices = torch.chunk(ego_features, self.groups, dim=1)
        fused_slices = torch.chunk(fused_features, self.groups, dim=1)
        reconstructed_slices = []
        for index, (ego_slice, fused_slice) in enumerate(
            zip(ego_slices, fused_slices)
        ):
            channel_context = torch.sigmoid(
                self.channel_gates[index](
                    F.adaptive_avg_pool2d(fused_slice, output_size=1)
                )
            )
            channel_last = fused_slice.permute(0, 2, 3, 1)
            spatial_context = torch.sigmoid(
                self.spatial_gates[index](
                    self.spatial_norms[index](channel_last)
                )
            ).permute(0, 3, 1, 2)
            reconstructed_slice = (
                ego_slice * channel_context * spatial_context
            )
            batch_size, channels, height, width = (
                reconstructed_slice.shape
            )
            reconstructed_slice = (
                reconstructed_slice.reshape(
                    batch_size,
                    self.shuffle_groups,
                    channels // self.shuffle_groups,
                    height,
                    width,
                )
                .transpose(1, 2)
                .contiguous()
                .reshape(batch_size, channels, height, width)
            )
            reconstructed_slices.append(reconstructed_slice)
        reconstructed = torch.cat(reconstructed_slices, dim=1)
        return reconstructed


class CUDAXAdapter(FusionAgnosticDomainAdapter):
    """CUDA-X reproduction from the published CKT/BLC/CPA specification.

    The CUDA-X authors had not released implementation code when this module
    was written.  This implementation therefore follows the paper equations
    and reported architecture while keeping shapes dynamic for OpenCOOD.
    """

    method = "cudax"

    def __init__(
        self,
        in_channels: int,
        detection_channels: int,
        anchor_number: int,
        hidden_dim: int = 256,
        discriminator_hidden_dim: int = 128,
        ckt_groups: int = 8,
        ckt_shuffle_groups: int = 2,
        bin_count: int = 5,
        feature_size: Optional[Tuple[int, int]] = None,
        cpa_hidden_dim: int = 1024,
    ) -> None:
        super().__init__()
        if anchor_number <= 0 or bin_count <= 1:
            raise ValueError("CUDA-X anchor_number and bin_count are invalid")
        if feature_size is None or len(feature_size) != 2:
            raise ValueError(
                "CUDA-X requires domain_adapter.feature_size=[H, W] to "
                "construct its CPA classifier"
            )
        feature_height, feature_width = (int(value) for value in feature_size)
        if feature_height <= 0 or feature_width <= 0:
            raise ValueError("CUDA-X feature_size values must be positive")
        if feature_width % 2 or anchor_number % 2:
            raise ValueError(
                "CUDA-X CPA requires an even feature width and anchor_number"
            )
        self.anchor_number = int(anchor_number)
        self.bin_count = int(bin_count)
        self.feature_size = (feature_height, feature_width)
        self.cpa_input_dim = (feature_width // 2) * (anchor_number // 2)
        self.ckt = CollaborativeKnowledgeTransfer(
            in_channels,
            groups=ckt_groups,
            shuffle_groups=ckt_shuffle_groups,
        )
        self.ckt_discriminator = GlobalDomainDiscriminator(
            in_channels, discriminator_hidden_dim
        )
        self.bin_features = nn.Sequential(
            nn.Conv2d(detection_channels, hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.bin_head = nn.Conv2d(
            hidden_dim,
            self.anchor_number * 6 * self.bin_count,
            kernel_size=1,
        )
        self.blc_gradient_reversal = GradientReversal()
        self.blc_discriminator = nn.Sequential(
            nn.Conv2d(detection_channels, hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )
        self.cpa_gradient_reversal = GradientReversal()
        self.cpa_discriminator = nn.Sequential(
            nn.Linear(self.cpa_input_dim, cpa_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cpa_hidden_dim, cpa_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cpa_hidden_dim, 1),
        )

    def adapt_fused(
        self,
        agent_features: torch.Tensor,
        fused_features: torch.Tensor,
        record_len: torch.Tensor,
        grl_lambda: float,
        pairwise_t_matrix: Optional[torch.Tensor] = None,
        adapter_domain: Optional[str] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        del pairwise_t_matrix, adapter_domain
        ego_features = extract_ego_features(agent_features, record_len)
        reconstructed = self.ckt(ego_features, fused_features)
        return fused_features, {
            "ckt_features": reconstructed,
            "ckt_domain_logits": self.ckt_discriminator(
                reconstructed, grl_lambda
            ),
        }

    def forward(
        self,
        agent_features: torch.Tensor,
        fused_features: torch.Tensor,
        record_len: torch.Tensor,
        pairwise_t_matrix: torch.Tensor,
        grl_lambda: float,
        agent_confidence_logits: Optional[torch.Tensor] = None,
        fused_class_logits: Optional[torch.Tensor] = None,
        context: Optional[Dict[str, torch.Tensor]] = None,
        detection_features: Optional[torch.Tensor] = None,
        adapter_domain: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        if fused_class_logits is None:
            raise ValueError("CUDA-X CPA requires fused classification logits")
        if context is None or "ckt_domain_logits" not in context:
            raise ValueError("CUDA-X adapt_fused must run before its DA heads")

        if detection_features is None:
            detection_features = fused_features
        if tuple(fused_class_logits.shape[-2:]) != self.feature_size:
            raise ValueError(
                "CUDA-X feature_size does not match fused classification "
                f"logits: {self.feature_size} != "
                f"{tuple(fused_class_logits.shape[-2:])}"
            )
        if fused_class_logits.shape[1] != self.anchor_number:
            raise ValueError(
                "CUDA-X CPA classification channels must equal anchor_number"
            )

        reversed_detection_features = self.blc_gradient_reversal(
            detection_features, coefficient=grl_lambda
        )
        reversed_class_logits = self.cpa_gradient_reversal(
            fused_class_logits, coefficient=grl_lambda
        )
        cpa_features = reversed_class_logits.permute(0, 2, 3, 1).contiguous()
        cpa_features = F.avg_pool2d(cpa_features, kernel_size=2, stride=2)
        cpa_features = cpa_features.flatten(start_dim=2)
        if cpa_features.shape[-1] != self.cpa_input_dim:
            raise RuntimeError(
                "CUDA-X CPA input dimension differs from its configured "
                f"feature map: {cpa_features.shape[-1]} != "
                f"{self.cpa_input_dim}"
            )
        batch_size = detection_features.shape[0]
        scene_indices = torch.arange(
            batch_size, device=fused_features.device
        )
        valid = torch.ones(
            batch_size, dtype=torch.bool, device=fused_features.device
        )
        output = {
            "ckt_domain_logits": context["ckt_domain_logits"],
            "blc_domain_logits": self.blc_discriminator(
                reversed_detection_features
            ),
            "cpa_domain_logits": self.cpa_discriminator(
                cpa_features
            ).squeeze(-1),
            "domain_scene_index": scene_indices,
            "domain_valid_mask": valid,
        }
        if adapter_domain != "target":
            output["bin_logits"] = self.bin_head(
                self.bin_features(detection_features)
            )
        return output


def build_domain_adapter(
    config: Optional[dict],
    in_channels: int,
    anchor_number: int,
    detection_channels: Optional[int] = None,
    lidar_range: Optional[Tuple[float, ...]] = None,
) -> Optional[FusionAgnosticDomainAdapter]:
    """Build an optional adapter from ``model.args.domain_adapter``."""

    config = dict(config or {})
    if not config.get("enabled", False):
        return None
    method = (
        str(config.get("method", ""))
        .lower()
        .replace("-", "")
        .replace("_", "")
    )
    hidden_dim = int(config.get("hidden_dim", 128))
    if method in ("grl", "discriminator", "naive"):
        return NaiveDomainAdapter(in_channels, hidden_dim)
    if method == "dusa":
        return DUSAAdapter(
            in_channels,
            lsa_hidden_dim=int(config.get("dusa_lsa_hidden_dim", 1024)),
            cia_hidden_dim=int(config.get("dusa_cia_hidden_dim", 512)),
            lsa_grl_scale=float(config.get("lsa_grl_scale", 0.05)),
            cia_grl_scale=float(config.get("cia_grl_scale", 0.1)),
            lidar_range=(
                tuple(lidar_range) if lidar_range is not None else None
            ),
            feature_size=(
                tuple(config["feature_size"])
                if "feature_size" in config
                else None
            ),
        )
    if method == "iada":
        return IADAAdapter(
            in_channels=in_channels,
            anchor_number=anchor_number,
            hidden_dim=int(config.get("iada_hidden_dim", 64)),
            effect_dim=int(config.get("iada_effect_dim", 64)),
            geometry_scale=float(config.get("geometry_scale", 100.0)),
            gate_limit=float(config.get("iada_gate_limit", 1.0)),
            source_supervision_enabled=bool(
                config.get("iada_source_supervision_enabled", True)
            ),
            target_consistency_enabled=bool(
                config.get("iada_target_consistency_enabled", True)
            ),
            effect_memory_enabled=bool(
                config.get("iada_effect_memory_enabled", True)
            ),
            consistency_dropout=float(
                config.get("iada_consistency_dropout", 0.1)
            ),
            teacher_momentum=float(
                config.get("iada_teacher_momentum", 0.999)
            ),
            prototype_momentum=float(
                config.get("iada_prototype_momentum", 0.99)
            ),
        )
    if method == "ssda":
        return SSDAAdapter(
            in_channels,
            hidden_dim=int(config.get("ssda_hidden_dim", hidden_dim)),
            sao_enabled=bool(config.get("sao_enabled", True)),
            sao_probability=float(config.get("sao_probability", 1.0)),
            fsa_epsilon=float(config.get("fsa_epsilon", 1.0e-5)),
            detach_attention=bool(
                config.get("ssda_detach_attention", False)
            ),
        )
    if method == "cudax":
        return CUDAXAdapter(
            in_channels,
            detection_channels=(
                in_channels
                if detection_channels is None
                else detection_channels
            ),
            anchor_number=anchor_number,
            hidden_dim=int(config.get("cudax_hidden_dim", 256)),
            discriminator_hidden_dim=hidden_dim,
            ckt_groups=int(config.get("ckt_groups", 8)),
            ckt_shuffle_groups=int(config.get("ckt_shuffle_groups", 2)),
            bin_count=int(
                config.get(
                    "cudax_bin_count",
                    config.get("bin_count", 5),
                )
            ),
            feature_size=(
                tuple(config["feature_size"])
                if "feature_size" in config
                else None
            ),
            cpa_hidden_dim=int(config.get("cpa_hidden_dim", 1024)),
        )
    raise ValueError(
        "domain_adapter.method must be one of grl, dusa, cudax, iada, or "
        "ssda; "
        f"got {config.get('method')!r}"
    )


__all__ = [
    "CUDAXAdapter",
    "DUSAAdapter",
    "FusionAgnosticDomainAdapter",
    "GlobalDomainDiscriminator",
    "IADAAdapter",
    "FrequencyShiftAdjustment",
    "HaarWaveletReconstruction",
    "NaiveDomainAdapter",
    "SSDAAdapter",
    "StagedAdaptiveAlignment",
    "build_domain_adapter",
    "extract_ego_features",
    "scene_and_local_indices",
]
