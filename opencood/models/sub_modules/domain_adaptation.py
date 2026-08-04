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

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.models.sub_modules.interaction_da import (
    GradientReversal,
    InteractionDomainAdapter,
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

    def adapt_fused(
        self,
        agent_features: torch.Tensor,
        fused_features: torch.Tensor,
        record_len: torch.Tensor,
        grl_lambda: float,
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
    ) -> Dict[str, torch.Tensor]:
        if agent_confidence_logits is None:
            raise ValueError("DUSA requires per-agent confidence logits")
        if agent_confidence_logits.shape[0] != agent_features.shape[0]:
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

        reversed_agent_features = self.gradient_reversal(
            torch.cat((agent_features, coordinates), dim=1),
            coefficient=self.cia_grl_scale,
        )
        agent_logits = self.agent_discriminator(reversed_agent_features)
        scene_indices, local_indices = scene_and_local_indices(
            record_len, agent_features.device
        )
        return {
            "domain_logits": self.sim_real_discriminator(selected_ego),
            "domain_scene_index": torch.arange(
                len(record_len), device=agent_features.device
            ),
            "domain_valid_mask": torch.ones(
                len(record_len), dtype=torch.bool, device=agent_features.device
            ),
            "agent_domain_logits": agent_logits,
            "agent_domain_weights": self._confidence_weights(
                agent_confidence_logits, record_len
            ),
            "agent_scene_index": scene_indices,
            "agent_local_index": local_indices,
        }


class IADAAdapter(FusionAgnosticDomainAdapter):
    """Interaction-graph alignment without fusion-specific score injection."""

    method = "iada"

    def __init__(
        self,
        in_channels: int,
        graph_dim: int = 256,
        discriminator_hidden_dim: int = 128,
        geometry_scale: float = 100.0,
        normalize_domain_embedding: bool = True,
    ) -> None:
        super().__init__()
        self.interaction_adapter = InteractionDomainAdapter(
            in_channels=in_channels,
            hidden_dim=graph_dim,
            discriminator_hidden_dim=discriminator_hidden_dim,
            geometry_scale=geometry_scale,
            normalize_domain_embedding=normalize_domain_embedding,
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
    ) -> Dict[str, torch.Tensor]:
        graph_output = self.interaction_adapter(
            agent_features,
            record_len,
            pairwise_t_matrix,
            grl_lambda=grl_lambda,
        )
        batch_size = graph_output["domain_logits"].shape[0]
        return {
            "domain_logits": graph_output["domain_logits"],
            "domain_scene_index": torch.arange(
                batch_size, device=agent_features.device
            ),
            "domain_valid_mask": graph_output["valid_graph_mask"],
            "graph_embedding": graph_output["graph_embedding"],
            "valid_graph_mask": graph_output["valid_graph_mask"],
        }


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
            reconstructed_slices.append(
                torch.channel_shuffle(
                    ego_slice * channel_context * spatial_context,
                    self.shuffle_groups,
                )
            )
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
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
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

        bin_features = self.bin_features(detection_features)
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
        return {
            "ckt_domain_logits": context["ckt_domain_logits"],
            "blc_domain_logits": self.blc_discriminator(
                reversed_detection_features
            ),
            "cpa_domain_logits": self.cpa_discriminator(cpa_features).squeeze(-1),
            "bin_logits": self.bin_head(bin_features),
            "domain_scene_index": scene_indices,
            "domain_valid_mask": valid,
        }


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
            in_channels,
            graph_dim=int(config.get("graph_dim", 256)),
            discriminator_hidden_dim=hidden_dim,
            geometry_scale=float(config.get("geometry_scale", 100.0)),
            normalize_domain_embedding=bool(
                config.get("normalize_domain_embedding", True)
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
        "domain_adapter.method must be one of grl, dusa, cudax, or iada; "
        f"got {config.get('method')!r}"
    )


__all__ = [
    "CUDAXAdapter",
    "DUSAAdapter",
    "FusionAgnosticDomainAdapter",
    "GlobalDomainDiscriminator",
    "IADAAdapter",
    "NaiveDomainAdapter",
    "build_domain_adapter",
    "extract_ego_features",
    "scene_and_local_indices",
]
