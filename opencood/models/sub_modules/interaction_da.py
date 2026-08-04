"""Interaction-level domain adaptation modules for cooperative perception.

The graph encoder consumes the flattened per-agent feature layout used by
OpenCOOD (``[sum(record_len), C, H, W]``) together with raw metric pairwise
transforms.  It pads agents only inside this module and always returns masks
for padded nodes and edges.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class _GradientReversalFunction(Function):
    """Identity in the forward pass and gradient negation in backward."""

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, coefficient: float) -> torch.Tensor:
        ctx.coefficient = coefficient
        return inputs.view_as(inputs)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.coefficient * grad_output, None


class GradientReversal(nn.Module):
    """Gradient reversal layer with an optionally overridden coefficient.

    Parameters
    ----------
    coefficient : float
        Default gradient multiplier.  The backward gradient is multiplied by
        ``-coefficient`` while the forward value is unchanged.
    """

    def __init__(self, coefficient: float = 1.0):
        super().__init__()
        self.coefficient = float(coefficient)

    def forward(
        self,
        inputs: torch.Tensor,
        coefficient: Optional[float] = None,
    ) -> torch.Tensor:
        reverse_coefficient = (
            self.coefficient if coefficient is None else float(coefficient)
        )
        return _GradientReversalFunction.apply(inputs, reverse_coefficient)

    def extra_repr(self) -> str:
        return "coefficient={}".format(self.coefficient)


class InteractionGraphEncoder(nn.Module):
    """Encode agent features and relative geometry into an interaction graph.

    Parameters
    ----------
    in_channels : int
        Channel count of each input BEV feature map.
    hidden_dim : int
        Dimension used for node, edge, and graph embeddings.
    geometry_scale : float
        Metric scale used to normalize ``dx``, ``dy``, and distance.
    """

    EDGE_ATTRIBUTE_DIM = 5

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 256,
        geometry_scale: float = 100.0,
    ):
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if geometry_scale <= 0:
            raise ValueError("geometry_scale must be positive")

        self.in_channels = int(in_channels)
        self.hidden_dim = int(hidden_dim)
        self.geometry_scale = float(geometry_scale)

        self.node_mlp = nn.Sequential(
            nn.Linear(self.in_channels, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(
                2 * self.hidden_dim + self.EDGE_ATTRIBUTE_DIM,
                self.hidden_dim,
            ),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        agent_features: torch.Tensor,
        record_len: torch.Tensor,
        pairwise_t_matrix: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Build interaction features for every scene in a batch.

        Parameters
        ----------
        agent_features : torch.Tensor
            Flattened agent BEV features with shape
            ``[sum(record_len), C, H, W]``.
        record_len : torch.Tensor
            Valid agent count per scene, shape ``[B]``.
        pairwise_t_matrix : torch.Tensor
            Raw, unnormalized metric transforms with shape
            ``[B, L, L, 4, 4]``.

        Returns
        -------
        dict
            ``pooled_node_features``: ``[B, L, C]`` GAP features.

            ``node_embeddings``: ``[B, L, D]`` encoded, padded nodes zeroed.

            ``node_mask``: ``[B, L]`` valid-node mask.

            ``edge_attributes``: ``[B, L, L, 5]`` containing normalized
            ``dx``, ``dy``, distance, ``sin(yaw)``, and ``cos(yaw)``.

            ``edge_hidden``: ``[B, L, L, D]`` encoded pairwise features.

            ``edge_mask``: ``[B, L, L]`` valid edges including self-edges.

            ``graph_edge_mask``: valid off-diagonal interaction edges.

            ``graph_embedding``: ``[B, D]`` masked off-diagonal edge mean,
            with a masked node-mean fallback for single-agent scenes.

            ``valid_graph_mask``: ``[B]``; true iff a scene has at least one
            off-diagonal interaction edge.
        """
        record_len, batch_size, max_agents = self._validate_and_prepare_inputs(
            agent_features,
            record_len,
            pairwise_t_matrix,
        )

        pooled_flat = agent_features.mean(dim=(-2, -1))
        node_positions = torch.arange(
            max_agents,
            device=agent_features.device,
        ).unsqueeze(0)
        node_mask = node_positions < record_len.unsqueeze(1)

        pooled_nodes = agent_features.new_zeros(
            (batch_size, max_agents, self.in_channels)
        )
        pooled_nodes[node_mask] = pooled_flat

        node_embeddings = self.node_mlp(pooled_nodes)
        node_embeddings = node_embeddings * node_mask.unsqueeze(-1).to(
            node_embeddings.dtype
        )

        edge_mask = node_mask.unsqueeze(2) & node_mask.unsqueeze(1)
        edge_attributes = self._extract_edge_attributes(
            pairwise_t_matrix,
            edge_mask,
            agent_features,
        )

        node_i = node_embeddings.unsqueeze(2).expand(
            -1, -1, max_agents, -1
        )
        node_j = node_embeddings.unsqueeze(1).expand(
            -1, max_agents, -1, -1
        )
        edge_inputs = torch.cat(
            (node_i, node_j, edge_attributes),
            dim=-1,
        )
        edge_hidden = self.edge_mlp(edge_inputs)
        edge_hidden = edge_hidden * edge_mask.unsqueeze(-1).to(
            edge_hidden.dtype
        )

        identity_mask = torch.eye(
            max_agents,
            dtype=torch.bool,
            device=agent_features.device,
        ).unsqueeze(0)
        graph_edge_mask = edge_mask & ~identity_mask
        valid_graph_mask = graph_edge_mask.flatten(1).any(dim=1)

        edge_denominator = graph_edge_mask.sum(
            dim=(1, 2),
            keepdim=False,
        ).clamp_min(1).to(edge_hidden.dtype)
        edge_graph_embedding = (
            edge_hidden
            * graph_edge_mask.unsqueeze(-1).to(edge_hidden.dtype)
        ).sum(dim=(1, 2)) / edge_denominator.unsqueeze(-1)

        node_denominator = node_mask.sum(dim=1).clamp_min(1).to(
            node_embeddings.dtype
        )
        node_graph_embedding = node_embeddings.sum(dim=1) / (
            node_denominator.unsqueeze(-1)
        )
        graph_embedding = torch.where(
            valid_graph_mask.unsqueeze(-1),
            edge_graph_embedding,
            node_graph_embedding,
        )

        return {
            "pooled_node_features": pooled_nodes,
            "node_embeddings": node_embeddings,
            "node_mask": node_mask,
            "edge_attributes": edge_attributes,
            "edge_hidden": edge_hidden,
            "edge_mask": edge_mask,
            "graph_edge_mask": graph_edge_mask,
            "graph_embedding": graph_embedding,
            "valid_graph_mask": valid_graph_mask,
        }

    def _validate_and_prepare_inputs(
        self,
        agent_features: torch.Tensor,
        record_len: torch.Tensor,
        pairwise_t_matrix: torch.Tensor,
    ) -> Tuple[torch.Tensor, int, int]:
        if agent_features.ndim != 4:
            raise ValueError(
                "agent_features must have shape [sumN, C, H, W]"
            )
        if agent_features.shape[1] != self.in_channels:
            raise ValueError(
                f"agent feature channels ({agent_features.shape[1]}) do not "
                f"match in_channels ({self.in_channels})"
            )
        if not agent_features.is_floating_point():
            raise TypeError("agent_features must use a floating-point dtype")

        record_len = torch.as_tensor(
            record_len,
            dtype=torch.long,
            device=agent_features.device,
        )
        if record_len.ndim != 1:
            raise ValueError("record_len must have shape [B]")
        if record_len.numel() == 0:
            raise ValueError("record_len must contain at least one scene")
        if torch.any(record_len <= 0).item():
            raise ValueError("every scene must contain at least one agent")
        if int(record_len.sum().item()) != agent_features.shape[0]:
            raise ValueError(
                "sum(record_len) must equal the number of agent features"
            )

        if pairwise_t_matrix.ndim != 5:
            raise ValueError(
                "pairwise_t_matrix must have shape [B, L, L, 4, 4]"
            )
        if tuple(pairwise_t_matrix.shape[-2:]) != (4, 4):
            raise ValueError(
                "pairwise_t_matrix must contain raw 4x4 transforms"
            )
        batch_size = record_len.numel()
        if pairwise_t_matrix.shape[0] != batch_size:
            raise ValueError(
                "pairwise_t_matrix batch size must match record_len"
            )
        if pairwise_t_matrix.shape[1] != pairwise_t_matrix.shape[2]:
            raise ValueError("pairwise agent dimensions must be square")

        max_agents = pairwise_t_matrix.shape[1]
        if torch.any(record_len > max_agents).item():
            raise ValueError(
                "record_len contains more agents than pairwise_t_matrix"
            )
        return record_len, batch_size, max_agents

    def _extract_edge_attributes(
        self,
        pairwise_t_matrix: torch.Tensor,
        edge_mask: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        transforms = pairwise_t_matrix.to(
            device=reference.device,
            dtype=reference.dtype,
        )
        dx = transforms[..., 0, 3]
        dy = transforms[..., 1, 3]
        distance = torch.sqrt(dx.square() + dy.square())
        yaw = torch.atan2(transforms[..., 1, 0], transforms[..., 0, 0])

        edge_attributes = torch.stack(
            (
                dx / self.geometry_scale,
                dy / self.geometry_scale,
                distance / self.geometry_scale,
                torch.sin(yaw),
                torch.cos(yaw),
            ),
            dim=-1,
        )
        return edge_attributes * edge_mask.unsqueeze(-1).to(
            edge_attributes.dtype
        )


class InteractionDiscriminator(nn.Module):
    """Binary domain discriminator that returns logits, not probabilities."""

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("input_dim and hidden_dim must be positive")
        if dropout < 0 or dropout >= 1:
            raise ValueError("dropout must be in [0, 1)")

        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, graph_embedding: torch.Tensor) -> torch.Tensor:
        return self.net(graph_embedding)


class InteractionDomainAdapter(nn.Module):
    """End-to-end interaction graph encoder and adversarial domain head."""

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 256,
        discriminator_hidden_dim: int = 128,
        geometry_scale: float = 100.0,
        discriminator_dropout: float = 0.0,
        grl_coefficient: float = 1.0,
        normalize_domain_embedding: bool = False,
    ):
        super().__init__()
        self.normalize_domain_embedding = bool(normalize_domain_embedding)
        self.graph_encoder = InteractionGraphEncoder(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            geometry_scale=geometry_scale,
        )
        self.gradient_reversal = GradientReversal(grl_coefficient)
        self.domain_discriminator = InteractionDiscriminator(
            input_dim=hidden_dim,
            hidden_dim=discriminator_hidden_dim,
            dropout=discriminator_dropout,
        )

    def forward(
        self,
        agent_features: torch.Tensor,
        record_len: torch.Tensor,
        pairwise_t_matrix: torch.Tensor,
        grl_lambda: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        output = self.graph_encoder(
            agent_features,
            record_len,
            pairwise_t_matrix,
        )
        domain_embedding = output["graph_embedding"]
        if self.normalize_domain_embedding:
            domain_embedding = F.normalize(
                domain_embedding, p=2, dim=1, eps=1.0e-6
            )
        reversed_embedding = self.gradient_reversal(
            domain_embedding,
            coefficient=grl_lambda,
        )
        output["domain_embedding"] = domain_embedding
        output["reversed_graph_embedding"] = reversed_embedding
        output["domain_logits"] = self.domain_discriminator(
            reversed_embedding
        )
        return output
