import copy
import unittest

import torch

from opencood.models.point_pillar_iada import PointPillarIADA
from opencood.models.point_pillar_intermediate import (
    PointPillarIntermediate,
)


def _model_args(interaction_enabled=True):
    return {
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
        "base_bev_backbone": {
            "voxel_size": [1.0, 1.0, 4.0],
            "layer_nums": [0, 0, 0],
            "layer_strides": [2, 2, 2],
            "num_filters": [64, 128, 256],
            "upsample_strides": [1, 2, 4],
            "num_upsample_filter": [128, 128, 128],
        },
        "interaction_da": {
            "enabled": interaction_enabled,
            "graph_dim": 16,
            "geometry_scale": 10.0,
        },
        "dir_args": {"num_bins": 2},
    }


def _model_input():
    torch.manual_seed(31)
    # Two scenes with record_len=[2,1]. Give each flattened agent two voxels.
    voxel_features = torch.randn(6, 4, 4)
    voxel_num_points = torch.full((6,), 4, dtype=torch.int32)
    voxel_coords = torch.tensor(
        [
            [0, 0, 1, 1],
            [0, 0, 2, 2],
            [1, 0, 3, 3],
            [1, 0, 4, 4],
            [2, 0, 5, 5],
            [2, 0, 6, 6],
        ],
        dtype=torch.int32,
    )
    pairwise = torch.eye(4, dtype=torch.float64).view(
        1, 1, 1, 4, 4
    ).repeat(2, 2, 2, 1, 1)
    pairwise[0, 0, 1, 0, 3] = 1.0
    pairwise[0, 1, 0, 0, 3] = -1.0

    return {
        "processed_lidar": {
            "voxel_features": voxel_features,
            "voxel_coords": voxel_coords,
            "voxel_num_points": voxel_num_points,
        },
        "record_len": torch.tensor([2, 1]),
        "pairwise_t_matrix": pairwise,
        "lidar_pose": torch.zeros(3, 6, dtype=torch.float64),
        "grl_lambda": 0.5,
    }


class PointPillarIADATest(unittest.TestCase):
    def test_existing_attfuse_state_only_misses_new_interaction_branch(self):
        args = _model_args(interaction_enabled=True)
        reference = PointPillarIntermediate(args)
        adapted = PointPillarIADA(args)

        incompatible = adapted.load_state_dict(
            reference.state_dict(), strict=False
        )

        self.assertEqual(incompatible.unexpected_keys, [])
        self.assertTrue(incompatible.missing_keys)
        self.assertTrue(
            all(
                key.startswith("interaction_da.")
                for key in incompatible.missing_keys
            )
        )

    def test_baseline_matches_existing_multiscale_attfuse(self):
        args = _model_args(interaction_enabled=False)
        reference = PointPillarIntermediate(args).eval()
        baseline = PointPillarIADA(args).eval()
        incompatible = baseline.load_state_dict(
            reference.state_dict(), strict=False
        )
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])

        model_input = _model_input()
        with torch.no_grad():
            expected = reference(copy.deepcopy(model_input))
            actual = baseline(copy.deepcopy(model_input))

        self.assertEqual(set(actual), set(expected))
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key])

    def test_forward_shapes_and_single_agent_mask(self):
        model = PointPillarIADA(_model_args())
        model.eval()

        with torch.no_grad():
            output = model(_model_input())

        self.assertEqual(output["cls_preds"].shape, (2, 2, 4, 4))
        self.assertEqual(output["reg_preds"].shape, (2, 14, 4, 4))
        self.assertEqual(output["dir_preds"].shape, (2, 4, 4, 4))
        self.assertEqual(output["domain_logits"].shape, (2, 1))
        self.assertEqual(output["graph_embedding"].shape, (2, 16))
        self.assertNotIn("interaction_logits", output)
        self.assertTrue(
            torch.equal(
                output["valid_graph_mask"], torch.tensor([True, False])
            )
        )
        self.assertTrue(
            torch.isfinite(output["domain_logits"]).all().item()
        )

    def test_graph_alignment_does_not_change_native_attfuse_detection(self):
        baseline = PointPillarIADA(
            _model_args(interaction_enabled=False)
        ).eval()
        adapted = PointPillarIADA(
            _model_args(interaction_enabled=True)
        ).eval()
        incompatible = adapted.load_state_dict(
            baseline.state_dict(), strict=False
        )
        self.assertEqual(incompatible.unexpected_keys, [])
        self.assertTrue(incompatible.missing_keys)
        self.assertTrue(
            all(
                key.startswith("interaction_da.")
                for key in incompatible.missing_keys
            )
        )
        model_input = _model_input()

        with torch.no_grad():
            baseline_output = baseline(copy.deepcopy(model_input))
            adapted_output = adapted(copy.deepcopy(model_input))

        for key in ("cls_preds", "reg_preds", "dir_preds"):
            torch.testing.assert_close(
                baseline_output[key], adapted_output[key]
            )
        self.assertNotIn("interaction_logits", adapted_output)

    def test_baseline_mode_omits_domain_outputs(self):
        model = PointPillarIADA(
            _model_args(interaction_enabled=False)
        ).eval()

        with torch.no_grad():
            output = model(_model_input())

        self.assertNotIn("domain_logits", output)
        self.assertNotIn("interaction_logits", output)
        self.assertEqual(output["cls_preds"].shape[0], 2)


if __name__ == "__main__":
    unittest.main()
