# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib
# Support F-Cooper, Self-Att, DiscoNet(wo KD), V2VNet, V2XViT, When2comm

import torch
import torch.nn as nn

from opencood.models.fuse_modules.fusion_in_one import (
    AttFusion,
    CoBEVT,
    DiscoFusion,
    MaxFusion,
    V2VNetFusion,
    V2XViTFusion,
    When2commFusion,
)
from opencood.models.fuse_modules.pyramid_fuse import PyramidFusion
from opencood.models.sub_modules.domain_adaptation import build_domain_adapter
from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.utils.transformation_utils import normalize_pairwise_tfm

class PointPillarBaseline(nn.Module):
    """
    F-Cooper implementation with point pillar backbone.
    """
    def __init__(self, args):
        super(PointPillarBaseline, self).__init__()

        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        is_resnet = args['base_bev_backbone'].get("resnet", False)
        if is_resnet:
            self.backbone = ResNetBEVBackbone(args['base_bev_backbone'], 64) # or you can use ResNetBEVBackbone, which is stronger
        else:
            self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64) # or you can use ResNetBEVBackbone, which is stronger
        self.voxel_size = args['voxel_size']

        fusion_method = args['fusion_method']
        self.fusion_method = fusion_method
        self.pyramid_fusion = fusion_method == "pyramid"
        self.pyramid_aux_enabled = bool(
            args.get("pyramid_aux_loss", {}).get("enabled", False)
        )
        if fusion_method == "max":
            self.fusion_net = MaxFusion()
        elif fusion_method == "att":
            self.fusion_net = AttFusion(args['att']['feat_dim'])
        elif fusion_method == "disconet":
            self.fusion_net = DiscoFusion(args['disconet']['feat_dim'])
        elif fusion_method == "v2vnet":
            self.fusion_net = V2VNetFusion(args['v2vnet'])
        elif fusion_method == 'v2xvit':
            self.fusion_net = V2XViTFusion(args['v2xvit'])
        elif fusion_method == 'cobevt':
            self.fusion_net = CoBEVT(args['cobevt'])
        elif fusion_method == 'pyramid':
            self.fusion_net = PyramidFusion(
                args['pyramid'],
                input_channels=self.backbone.num_bev_features,
            )
        elif fusion_method == 'when2comm':
            self.fusion_net = When2commFusion(args['when2comm'])
        else:
            raise ValueError(
                f"Unsupported PointPillar fusion_method: {fusion_method!r}"
            )

        output_backbone = (
            args['pyramid']
            if self.pyramid_fusion
            else args['base_bev_backbone']
        )
        self.out_channel = sum(output_backbone['num_upsample_filter'])

        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
            self.out_channel = args['shrink_header']['dim'][-1]

        self.compression = False
        if "compression" in args:
            self.compression = True
            self.naive_compressor = NaiveCompressor(self.out_channel, args['compression'])

        self.cls_head = nn.Conv2d(self.out_channel, args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(self.out_channel, 7 * args['anchor_number'],
                                  kernel_size=1)
        self.use_dir = False
        if 'dir_args' in args.keys():
            self.use_dir = True
            self.dir_head = nn.Conv2d(self.out_channel, args['dir_args']['num_bins'] * args['anchor_number'],
                                  kernel_size=1) # BIN_NUM = 2

        self.domain_adapter = build_domain_adapter(
            args.get('domain_adapter'),
            in_channels=self.out_channel,
            anchor_number=args['anchor_number'],
            detection_channels=self.out_channel,
            lidar_range=tuple(args['lidar_range']),
        )
        if (
            self.pyramid_fusion
            and self.domain_adapter is not None
            and self.domain_adapter.method == "ssda"
        ):
            raise ValueError(
                "PyramidFusion does not support SSDA's pre-fusion FSA hook; "
                "baseline, GRL, DUSA, CUDA-X, and IADA are supported"
            )

        if 'backbone_fix' in args.keys() and args['backbone_fix']:
            self.backbone_fix()

    def backbone_fix(self):
        """
        Fix the parameters of backbone during finetune on timedelay。
        """
        for p in self.pillar_vfe.parameters():
            p.requires_grad = False

        for p in self.scatter.parameters():
            p.requires_grad = False

        for p in self.backbone.parameters():
            p.requires_grad = False

        if self.compression:
            for p in self.naive_compressor.parameters():
                p.requires_grad = False
        if self.shrink_flag:
            for p in self.shrink_conv.parameters():
                p.requires_grad = False

        for p in self.cls_head.parameters():
            p.requires_grad = False
        for p in self.reg_head.parameters():
            p.requires_grad = False

    def forward(self, data_dict):
        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len}
        # n, 4 -> n, c
        batch_dict = self.pillar_vfe(batch_dict)
        # n, c -> N, C, H, W
        batch_dict = self.scatter(batch_dict)
        # calculate pairwise affine transformation matrix
        _, _, H0, W0 = batch_dict['spatial_features'].shape # original feature map shape H0, W0
        normalized_affine_matrix = normalize_pairwise_tfm(data_dict['pairwise_t_matrix'], H0, W0, self.voxel_size[0])
        batch_dict = self.backbone(batch_dict)

        spatial_features_2d = batch_dict['spatial_features_2d']
        adaptation_context = {}
        occupancy_outputs = None
        if self.pyramid_fusion:
            if self.compression:
                raise ValueError(
                    "PyramidFusion does not support decoded-feature compression"
                )
            needs_agent_features = self.domain_adapter is not None
            pyramid_output = self.fusion_net.forward_collab(
                spatial_features_2d,
                record_len,
                normalized_affine_matrix,
                return_agent_features=needs_agent_features,
            )
            if needs_agent_features:
                fused_feature, occupancy_outputs, agent_features = (
                    pyramid_output
                )
            else:
                fused_feature, occupancy_outputs = pyramid_output
                agent_features = None
            if self.shrink_flag:
                if agent_features is not None:
                    agent_features = self.shrink_conv(agent_features)
                fused_feature = self.shrink_conv(fused_feature)
            if self.domain_adapter is not None:
                agent_features, adaptation_context = (
                    self.domain_adapter.adapt_agents(
                        agent_features, record_len
                    )
                )
        else:
            if self.shrink_flag:
                spatial_features_2d = self.shrink_conv(spatial_features_2d)
            if self.compression:
                spatial_features_2d = self.naive_compressor(
                    spatial_features_2d
                )
            agent_features = spatial_features_2d
            if self.domain_adapter is not None:
                agent_features, adaptation_context = (
                    self.domain_adapter.adapt_agents(
                        agent_features, record_len
                    )
                )
            if self.fusion_method == 'v2xvit':
                prior_encoding = data_dict.get('prior_encoding')
                # Match the explicit prior to autocast feature dtype.
                if (torch.is_tensor(prior_encoding)
                        and prior_encoding.is_floating_point()
                        and prior_encoding.device == agent_features.device):
                    prior_encoding = prior_encoding.to(agent_features.dtype)
                fused_feature = self.fusion_net(
                    agent_features,
                    record_len,
                    normalized_affine_matrix,
                    prior_encoding=prior_encoding,
                )
            else:
                fused_feature = self.fusion_net(
                    agent_features, record_len, normalized_affine_matrix
                )

        if self.domain_adapter is not None:
            fused_feature, fused_context = \
                self.domain_adapter.adapt_fused(
                    agent_features,
                    fused_feature,
                    record_len,
                    data_dict.get('grl_lambda', 1.0),
                    pairwise_t_matrix=data_dict['pairwise_t_matrix'],
                    adapter_domain=data_dict.get('adapter_domain'),
                )
            adaptation_context.update(fused_context)

        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)

        output_dict = {'cls_preds': psm,
                       'reg_preds': rm}

        if self.pyramid_aux_enabled:
            output_dict['occ_single_list'] = occupancy_outputs

        if self.use_dir:
            output_dict.update({'dir_preds': self.dir_head(fused_feature)})

        if self.domain_adapter is not None:
            adapter_domain = data_dict.get('adapter_domain')
            if self.domain_adapter.method == 'iada':
                ego_feature = adaptation_context['iada_ego_features']
                adaptation_context['iada_ego_cls_preds'] = self.cls_head(
                    ego_feature
                )
                adaptation_context['iada_ego_reg_preds'] = self.reg_head(
                    ego_feature
                )
                consistency_feature = adaptation_context.get(
                    'iada_consistency_features'
                )
                if consistency_feature is not None:
                    adaptation_context['iada_consistency_cls_preds'] = (
                        self.cls_head(consistency_feature)
                    )
                    adaptation_context['iada_consistency_reg_preds'] = (
                        self.reg_head(consistency_feature)
                    )
                teacher_feature = adaptation_context.get(
                    'iada_teacher_features'
                )
                if teacher_feature is not None:
                    with torch.no_grad():
                        adaptation_context['iada_teacher_cls_preds'] = (
                            self.cls_head(teacher_feature)
                        )
                        adaptation_context['iada_teacher_reg_preds'] = (
                            self.reg_head(teacher_feature)
                        )
            agent_confidence_logits = None
            needs_confidence = self.domain_adapter.requires_agent_confidence
            if (
                self.domain_adapter.method == 'dusa'
                and adapter_domain == 'source'
            ):
                needs_confidence = False
            if needs_confidence:
                agent_confidence_logits = self.cls_head(agent_features)
            output_dict.update(
                self.domain_adapter(
                    agent_features=agent_features,
                    fused_features=fused_feature,
                    record_len=record_len,
                    pairwise_t_matrix=data_dict['pairwise_t_matrix'],
                    grl_lambda=data_dict.get('grl_lambda', 1.0),
                    agent_confidence_logits=agent_confidence_logits,
                    fused_class_logits=psm,
                    context=adaptation_context,
                    detection_features=fused_feature,
                    adapter_domain=adapter_domain,
                )
            )

        return output_dict
