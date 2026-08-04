"""PointPillar AttFuse with interaction-aware domain adaptation."""

import torch
import torch.nn as nn

from opencood.models.fuse_modules.self_attn import AttFusion
from opencood.models.sub_modules.att_bev_backbone import AttBEVBackbone
from opencood.models.sub_modules.interaction_da import InteractionDomainAdapter
from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.utils.transformation_utils import normalize_pairwise_tfm


class PointPillarIADA(nn.Module):
    """Multi-scale AttFuse augmented with an interaction domain branch.

    The module names of the PointPillar encoder, backbone, and detection heads
    intentionally match :class:`PointPillarIntermediate`, so an existing
    AttFuse checkpoint can be loaded with ``strict=False``. The graph branch is
    built from the deepest per-agent BEV feature and is used only for domain
    alignment; fusion remains the unmodified native AttFuse implementation.
    """

    def __init__(self, args):
        super().__init__()

        self.voxel_size = args["voxel_size"]
        self.pillar_vfe = PillarVFE(
            args["pillar_vfe"],
            num_point_features=4,
            voxel_size=args["voxel_size"],
            point_cloud_range=args["lidar_range"],
        )
        self.scatter = PointPillarScatter(args["point_pillar_scatter"])
        self.backbone = AttBEVBackbone(args["base_bev_backbone"], 64)

        feature_dims = args["base_bev_backbone"]["num_filters"]
        self.backbone.fuse_modules = nn.ModuleList(
            AttFusion(feature_dim) for feature_dim in feature_dims
        )

        output_channels = self.backbone.num_bev_features
        self.cls_head = nn.Conv2d(
            output_channels, args["anchor_number"], kernel_size=1
        )
        self.reg_head = nn.Conv2d(
            output_channels, 7 * args["anchor_number"], kernel_size=1
        )

        self.use_dir = "dir_args" in args
        if self.use_dir:
            self.dir_head = nn.Conv2d(
                output_channels,
                args["dir_args"]["num_bins"] * args["anchor_number"],
                kernel_size=1,
            )

        interaction_cfg = args.get("interaction_da", {})
        self.interaction_enabled = interaction_cfg.get("enabled", True)
        if self.interaction_enabled:
            graph_dim = interaction_cfg.get("graph_dim", 256)
            self.interaction_da = InteractionDomainAdapter(
                in_channels=feature_dims[-1],
                hidden_dim=graph_dim,
                geometry_scale=interaction_cfg.get(
                    "geometry_scale", 100.0
                ),
                normalize_domain_embedding=interaction_cfg.get(
                    "normalize_domain_embedding", False
                ),
            )

    def _encode_multiscale(self, spatial_features):
        feature_list = []
        x = spatial_features
        for level, block in enumerate(self.backbone.blocks):
            x = block(x)
            if (
                self.backbone.compress
                and level < len(self.backbone.compression_modules)
            ):
                x = self.backbone.compression_modules[level](x)
            feature_list.append(x)
        return feature_list

    def _decode_multiscale(self, fused_feature_list):
        upsampled = []
        for level, feature in enumerate(fused_feature_list):
            if len(self.backbone.deblocks) > 0:
                feature = self.backbone.deblocks[level](feature)
            upsampled.append(feature)

        if len(upsampled) > 1:
            decoded = torch.cat(upsampled, dim=1)
        else:
            decoded = upsampled[0]

        if len(self.backbone.deblocks) > len(self.backbone.blocks):
            decoded = self.backbone.deblocks[-1](decoded)
        return decoded

    def forward(self, data_dict):
        processed_lidar = data_dict["processed_lidar"]
        record_len = data_dict["record_len"]
        pairwise_t_matrix = data_dict["pairwise_t_matrix"]

        batch_dict = {
            "voxel_features": processed_lidar["voxel_features"],
            "voxel_coords": processed_lidar["voxel_coords"],
            "voxel_num_points": processed_lidar["voxel_num_points"],
            "record_len": record_len,
        }
        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        spatial_features = batch_dict["spatial_features"]

        _, _, height, width = spatial_features.shape
        normalized_pairwise = normalize_pairwise_tfm(
            pairwise_t_matrix.clone(),
            height,
            width,
            self.voxel_size[0],
        ).to(device=spatial_features.device, dtype=spatial_features.dtype)

        multiscale_features = self._encode_multiscale(spatial_features)

        graph_output = None
        if self.interaction_enabled:
            graph_output = self.interaction_da(
                multiscale_features[-1],
                record_len,
                pairwise_t_matrix,
                grl_lambda=data_dict.get("grl_lambda", 1.0),
            )

        fused_feature_list = []
        for feature, fuse_module in zip(
            multiscale_features, self.backbone.fuse_modules
        ):
            fused_feature_list.append(
                fuse_module(
                    feature,
                    record_len,
                    normalized_pairwise,
                )
            )

        fused_feature = self._decode_multiscale(fused_feature_list)
        output_dict = {
            "cls_preds": self.cls_head(fused_feature),
            "reg_preds": self.reg_head(fused_feature),
        }
        if self.use_dir:
            output_dict["dir_preds"] = self.dir_head(fused_feature)

        if graph_output is not None:
            output_dict.update(
                {
                    "domain_logits": graph_output["domain_logits"],
                    "graph_embedding": graph_output["graph_embedding"],
                    "valid_graph_mask": graph_output["valid_graph_mask"],
                }
            )

        return output_dict
