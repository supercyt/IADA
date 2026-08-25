import copy
import unittest

import torch

from opencood.loss.point_pillar_loss import PointPillarLoss
from opencood.loss.point_pillar_pyramid_loss import PointPillarPyramidLoss
from opencood.models.fuse_modules.fusion_in_one import CoBEVT
from opencood.models.fuse_modules.pyramid_fuse import PyramidFusion
from opencood.models.point_pillar_baseline import PointPillarBaseline


def _identity_affine(batch_size, max_agents):
    affine = torch.zeros(batch_size, max_agents, max_agents, 2, 3)
    affine[..., 0, 0] = 1
    affine[..., 1, 1] = 1
    return affine


def _identity_pairwise():
    return torch.eye(4, dtype=torch.float64).view(1, 1, 1, 4, 4).repeat(
        2, 2, 2, 1, 1
    )


def _model_input(domain="source"):
    return {
        "processed_lidar": {
            "voxel_features": torch.randn(6, 4, 4),
            "voxel_num_points": torch.full((6,), 4, dtype=torch.int32),
            "voxel_coords": torch.tensor(
                [
                    [0, 0, 1, 1],
                    [0, 0, 2, 2],
                    [1, 0, 3, 3],
                    [1, 0, 4, 4],
                    [2, 0, 5, 5],
                    [2, 0, 6, 6],
                ],
                dtype=torch.int32,
            ),
        },
        "record_len": torch.tensor([2, 1]),
        "pairwise_t_matrix": _identity_pairwise(),
        "grl_lambda": 0.5,
        "adapter_domain": domain,
    }


def _point_pillar_args(method):
    args = {
        "voxel_size": [1.0, 1.0, 4.0],
        "lidar_range": [0.0, 0.0, -3.0, 8.0, 8.0, 1.0],
        "anchor_number": 2,
        "pillar_vfe": {
            "use_norm": True,
            "with_distance": False,
            "use_absolute_xyz": True,
            "num_filters": [64],
        },
        "point_pillar_scatter": {
            "num_features": 64,
            "grid_size": [8, 8, 1],
        },
        "domain_adapter": {
            "enabled": True,
            "method": "iada",
            "iada_hidden_dim": 8,
            "iada_effect_dim": 8,
            "iada_local_topk": 4,
            "iada_source_supervision_enabled": True,
            "iada_target_consistency_enabled": False,
            "iada_effect_memory_enabled": False,
        },
        "dir_args": {"num_bins": 2},
    }
    if method == "cobevt":
        args.update(
            {
                "fusion_method": "cobevt",
                "base_bev_backbone": {
                    "layer_nums": [0, 0, 0],
                    "layer_strides": [2, 2, 2],
                    "num_filters": [64, 128, 256],
                    "upsample_strides": [1, 2, 4],
                    "num_upsample_filter": [128, 128, 128],
                },
                "shrink_header": {
                    "kernal_size": [3],
                    "stride": [1],
                    "padding": [1],
                    "dim": [8],
                    "input_dim": 384,
                },
                "cobevt": {
                    "input_dim": 8,
                    "mlp_dim": 8,
                    "agent_size": 2,
                    "window_size": 2,
                    "dim_head": 4,
                    "drop_out": 0.0,
                    "depth": 1,
                },
            }
        )
    else:
        args.update(
            {
                "fusion_method": "pyramid",
                "base_bev_backbone": {
                    "resnet": True,
                    "layer_nums": [1],
                    "layer_strides": [2],
                    "num_filters": [64],
                    "upsample_strides": [1],
                    "num_upsample_filter": [64],
                },
                "pyramid": {
                    "resnext": False,
                    "layer_nums": [1, 1, 1],
                    "layer_strides": [1, 2, 2],
                    "num_filters": [8, 16, 32],
                    "upsample_strides": [1, 2, 4],
                    "num_upsample_filter": [8, 8, 8],
                },
                "shrink_header": {
                    "kernal_size": [3],
                    "stride": [1],
                    "padding": [1],
                    "dim": [8],
                    "input_dim": 24,
                },
                "pyramid_aux_loss": {"enabled": False},
            }
        )
    return args


class FusionModuleTest(unittest.TestCase):
    def test_cobevt_handles_padding_and_backpropagates(self):
        fusion = CoBEVT(
            {
                "input_dim": 8,
                "mlp_dim": 8,
                "agent_size": 2,
                "window_size": 2,
                "dim_head": 4,
                "drop_out": 0.0,
                "depth": 1,
            }
        )
        features = torch.randn(3, 8, 4, 4, requires_grad=True)
        output = fusion(
            features, torch.tensor([2, 1]), _identity_affine(2, 2)
        )
        self.assertEqual(output.shape, (2, 8, 4, 4))
        output.square().mean().backward()
        self.assertIsNotNone(features.grad)

    def test_pyramid_single_agent_matches_collaborative_path(self):
        fusion = PyramidFusion(
            {
                "resnext": False,
                "layer_nums": [1, 1, 1],
                "layer_strides": [1, 2, 2],
                "num_filters": [8, 16, 32],
                "upsample_strides": [1, 2, 4],
                "num_upsample_filter": [8, 8, 8],
            },
            input_channels=8,
        ).eval()
        features = torch.randn(2, 8, 8, 8)
        single, occupancy = fusion.forward_single(features)
        collaborative, collab_occupancy = fusion.forward_collab(
            features,
            torch.tensor([1, 1]),
            _identity_affine(2, 1),
        )
        torch.testing.assert_close(collaborative, single)
        self.assertEqual(len(occupancy), 3)
        self.assertEqual(len(collab_occupancy), 3)


class PointPillarFusionIntegrationTest(unittest.TestCase):
    def test_new_fusions_run_iada_and_backpropagate(self):
        for method in ("cobevt", "pyramid"):
            with self.subTest(method=method):
                model = PointPillarBaseline(_point_pillar_args(method)).train()
                output = model(copy.deepcopy(_model_input()))
                self.assertEqual(output["cls_preds"].shape, (2, 2, 4, 4))
                self.assertIn("iada_effect_features", output)
                loss = (
                    output["cls_preds"].square().mean()
                    + output["reg_preds"].square().mean()
                    + output["iada_effect_features"].square().mean()
                )
                loss.backward()

    def test_disabled_pyramid_auxiliary_is_standard_detection_loss(self):
        config = {
            "pos_cls_weight": 2.0,
            "cls": {
                "alpha": 0.25,
                "gamma": 2.0,
                "weight": 2.0,
            },
            "reg": {"sigma": 3.0, "weight": 2.0},
            "pyramid_aux_loss": {"enabled": False},
        }
        standard = PointPillarLoss(config)
        pyramid = PointPillarPyramidLoss(config)
        output = {
            "cls_preds": torch.randn(1, 2, 2, 2),
            "reg_preds": torch.randn(1, 14, 2, 2),
        }
        target = {
            "pos_equal_one": torch.zeros(1, 2, 2, 2),
            "neg_equal_one": torch.ones(1, 2, 2, 2),
            "targets": torch.zeros(1, 2, 2, 14),
        }
        torch.testing.assert_close(
            pyramid(copy.deepcopy(output), target),
            standard(copy.deepcopy(output), target),
        )


if __name__ == "__main__":
    unittest.main()
