"""LiDAR-only PyramidFusion adapted from the HEAL implementation."""

import torch
from torch import nn

from opencood.models.fuse_modules.fusion_in_one import regroup
from opencood.models.sub_modules.base_bev_backbone_resnet import (
    ResNetBEVBackbone,
)
from opencood.models.sub_modules.resblock import Bottleneck, ResNetModified
from opencood.models.sub_modules.torch_transformation_utils import (
    warp_affine_simple,
)


class _PyramidBottleneck(Bottleneck):
    """ResNeXt bottleneck with HEAL's unit channel expansion."""

    expansion = 1


def weighted_fuse(
    features, scores, record_len, affine_matrix, align_corners=False
):
    """Warp each scene to ego and softmax-weight agents at every pixel."""

    _, _, height, width = features.shape
    batch_size = affine_matrix.shape[0]
    split_features = regroup(features, record_len)
    split_scores = regroup(scores, record_len)
    fused = []
    for batch_index in range(batch_size):
        agent_count = int(record_len[batch_index].item())
        transforms = affine_matrix[batch_index, 0, :agent_count]
        warped_features = warp_affine_simple(
            split_features[batch_index],
            transforms,
            (height, width),
            align_corners=align_corners,
        )
        warped_scores = warp_affine_simple(
            split_scores[batch_index],
            transforms,
            (height, width),
            align_corners=align_corners,
        )
        valid_scores = warped_scores.masked_fill(
            warped_scores == 0, -torch.inf
        )
        weights = torch.softmax(valid_scores, dim=0)
        weights = torch.nan_to_num(weights, nan=0.0)
        fused.append((warped_features * weights).sum(dim=0))
    return torch.stack(fused)


class PyramidFusion(ResNetBEVBackbone):
    """Multi-scale occupancy-guided cooperative feature fusion."""

    def __init__(self, model_cfg, input_channels=64):
        model_cfg = dict(model_cfg)
        model_cfg.setdefault("inplanes", input_channels)
        super().__init__(model_cfg, input_channels)
        if model_cfg.get("resnext", False):
            self.resnet = ResNetModified(
                _PyramidBottleneck,
                self.model_cfg["layer_nums"],
                self.model_cfg["layer_strides"],
                self.model_cfg["num_filters"],
                inplanes=model_cfg.get("inplanes", input_channels),
                groups=32,
                width_per_group=4,
            )
        self.align_corners = bool(model_cfg.get("align_corners", False))
        self.single_heads = nn.ModuleList(
            nn.Conv2d(channels, 1, kernel_size=1)
            for channels in self.model_cfg["num_filters"]
        )

    def forward_single(self, spatial_features):
        multiscale = self.get_multiscale_feature(spatial_features)
        occupancy = [
            head(feature)
            for head, feature in zip(self.single_heads, multiscale)
        ]
        return self.decode_multiscale_feature(multiscale), occupancy

    def forward_collab(
        self,
        spatial_features,
        record_len,
        affine_matrix,
        return_agent_features=False,
    ):
        multiscale = self.get_multiscale_feature(spatial_features)
        occupancy = []
        fused_multiscale = []
        for head, feature in zip(self.single_heads, multiscale):
            logits = head(feature)
            occupancy.append(logits)
            scores = torch.sigmoid(logits) + 1.0e-4
            fused_multiscale.append(
                weighted_fuse(
                    feature,
                    scores,
                    record_len,
                    affine_matrix,
                    align_corners=self.align_corners,
                )
            )

        fused = self.decode_multiscale_feature(fused_multiscale)
        if not return_agent_features:
            return fused, occupancy
        agents = self.decode_multiscale_feature(multiscale)
        return fused, occupancy, agents


__all__ = ["PyramidFusion", "weighted_fuse"]
