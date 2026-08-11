import copy
import math
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn

from opencood.loss.domain_adaptation_loss import (
    balanced_domain_loss,
    compute_adaptation_loss,
    cudax_bin_loss,
    dusa_agent_loss,
    entropy_weighted_domain_loss,
)
from opencood.tools.compute_cudax_bounds import (
    compute_source_residual_bounds,
)
from opencood.models.point_pillar_baseline import PointPillarBaseline
from opencood.models.fuse_modules.fusion_in_one import V2XViTFusion
from opencood.models.sub_modules.domain_adaptation import (
    CUDAXAdapter,
    DUSAAdapter,
    FrequencyShiftAdjustment,
    HaarWaveletReconstruction,
    IADAAdapter,
    NaiveDomainAdapter,
    SSDAAdapter,
    build_domain_adapter,
    extract_ego_features,
    scene_and_local_indices,
)


def _identity_pairwise(record_len, dtype=torch.float32):
    counts = torch.as_tensor(record_len)
    batch_size = int(counts.numel())
    max_agents = int(counts.max().item())
    return torch.eye(4, dtype=dtype).view(1, 1, 1, 4, 4).repeat(
        batch_size, max_agents, max_agents, 1, 1
    )


def _identity_affine(batch_size, max_agents, dtype=torch.float32):
    affine = torch.zeros(
        batch_size, max_agents, max_agents, 2, 3, dtype=dtype
    )
    affine[..., 0, 0] = 1
    affine[..., 1, 1] = 1
    return affine


def _adapter_inputs():
    torch.manual_seed(71)
    record_len = torch.tensor([2, 2])
    agent_features = torch.randn(4, 4, 2, 2)
    fused_features = torch.randn(2, 4, 2, 2)
    pairwise = _identity_pairwise(record_len)
    return agent_features, fused_features, record_len, pairwise


class AdapterIndexTest(unittest.TestCase):
    def test_flat_agent_indices_and_ego_extraction_respect_scene_boundaries(self):
        features = torch.arange(4.0).reshape(4, 1, 1, 1)
        record_len = torch.tensor([2, 1, 1])

        scene, local = scene_and_local_indices(
            record_len, features.device
        )

        torch.testing.assert_close(scene, torch.tensor([0, 0, 1, 2]))
        torch.testing.assert_close(local, torch.tensor([0, 1, 0, 0]))
        torch.testing.assert_close(
            extract_ego_features(features, record_len).flatten(),
            torch.tensor([0.0, 2.0, 3.0]),
        )


class AdapterFactoryTest(unittest.TestCase):
    def test_disabled_adapter_is_none(self):
        self.assertIsNone(
            build_domain_adapter(
                {"enabled": False}, in_channels=4, anchor_number=2
            )
        )

    def test_builds_all_comparison_methods_and_aliases(self):
        cases = (
            (
                {"enabled": True, "method": "discriminator", "hidden_dim": 4},
                NaiveDomainAdapter,
            ),
            (
                {
                    "enabled": True,
                    "method": "dusa",
                    "feature_size": [2, 2],
                    "dusa_lsa_hidden_dim": 4,
                    "dusa_cia_hidden_dim": 4,
                },
                DUSAAdapter,
            ),
            (
                {
                    "enabled": True,
                    "method": "iada",
                    "graph_dim": 4,
                    "hidden_dim": 4,
                },
                IADAAdapter,
            ),
            (
                {
                    "enabled": True,
                    "method": "CUDA-X",
                    "feature_size": [2, 2],
                    "cudax_hidden_dim": 4,
                    "cpa_hidden_dim": 4,
                    "ckt_groups": 2,
                    "bin_count": 3,
                },
                CUDAXAdapter,
            ),
            (
                {
                    "enabled": True,
                    "method": "ssda",
                    "ssda_hidden_dim": 4,
                },
                SSDAAdapter,
            ),
        )
        for config, expected_type in cases:
            with self.subTest(method=config["method"]):
                adapter = build_domain_adapter(
                    config,
                    in_channels=4,
                    detection_channels=4,
                    anchor_number=2,
                    lidar_range=(-2, -2, -1, 2, 2, 1),
                )
                self.assertIsInstance(adapter, expected_type)

    def test_unknown_method_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must be one of"):
            build_domain_adapter(
                {"enabled": True, "method": "not-a-method"},
                in_channels=4,
                anchor_number=2,
            )


class AdapterForwardTest(unittest.TestCase):
    def test_haar_reconstructed_bands_sum_to_original_feature(self):
        features = torch.randn(2, 3, 5, 7)

        bands = HaarWaveletReconstruction()(features)

        self.assertEqual(bands.shape, (2, 12, 5, 7))
        reconstructed = bands.reshape(2, 4, 3, 5, 7).sum(dim=1)
        torch.testing.assert_close(reconstructed, features)

    def test_fsa_refines_agents_without_crossing_scene_boundaries(self):
        features = torch.randn(4, 4, 3, 5, requires_grad=True)
        adapter = FrequencyShiftAdjustment(4).train()

        refined = adapter(features, torch.tensor([2, 1, 1]))

        self.assertEqual(refined.shape, features.shape)
        self.assertTrue(torch.isfinite(refined).all().item())
        refined.square().mean().backward()
        self.assertGreater(features.grad.abs().sum().item(), 0.0)

    def test_naive_grl_aligns_every_flattened_agent(self):
        agent_features, fused_features, record_len, pairwise = (
            _adapter_inputs()
        )
        agent_features.requires_grad_()
        adapter = NaiveDomainAdapter(in_channels=4, hidden_dim=4)
        with torch.no_grad():
            for parameter in adapter.parameters():
                parameter.fill_(0.1)

        output = adapter(
            agent_features,
            fused_features,
            record_len,
            pairwise,
            grl_lambda=0.5,
        )

        self.assertEqual(output["domain_logits"].shape, (4, 1))
        torch.testing.assert_close(
            output["domain_scene_index"], torch.tensor([0, 0, 1, 1])
        )
        self.assertTrue(output["domain_valid_mask"].all().item())
        output["domain_logits"].sum().backward()
        self.assertIsNotNone(agent_features.grad)
        self.assertTrue(torch.isfinite(agent_features.grad).all().item())
        self.assertGreater(agent_features.grad.abs().sum().item(), 0.0)

    def test_dusa_exposes_scene_lsa_and_target_role_metadata(self):
        agent_features, fused_features, record_len, pairwise = (
            _adapter_inputs()
        )
        confidence_logits = torch.randn(4, 2, 2, 2, requires_grad=True)
        adapter = DUSAAdapter(
            in_channels=4,
            lsa_hidden_dim=4,
            cia_hidden_dim=4,
            lidar_range=(-2, -2, -1, 2, 2, 1),
            feature_size=(2, 2),
        ).eval()

        output = adapter(
            agent_features,
            fused_features,
            record_len,
            pairwise,
            grl_lambda=0.5,
            agent_confidence_logits=confidence_logits,
        )

        self.assertEqual(output["domain_logits"].shape, (2, 1))
        self.assertEqual(output["agent_domain_logits"].shape, (4, 1, 2, 2))
        self.assertEqual(output["agent_domain_weights"].shape, (4, 1, 2, 2))
        torch.testing.assert_close(
            output["agent_scene_index"], torch.tensor([0, 0, 1, 1])
        )
        torch.testing.assert_close(
            output["agent_local_index"], torch.tensor([0, 1, 0, 1])
        )
        torch.testing.assert_close(
            output["agent_domain_weights"][0],
            output["agent_domain_weights"][1],
        )
        self.assertFalse(output["agent_domain_weights"].requires_grad)

    def test_dusa_reverses_backbone_but_not_location_selector_gradient(self):
        agent_features = torch.ones(2, 2, 2, 2, requires_grad=True)
        record_len = torch.tensor([2])
        adapter = DUSAAdapter(
            in_channels=2,
            lsa_hidden_dim=2,
            cia_hidden_dim=2,
            lidar_range=(-2, -2, -1, 2, 2, 1),
            feature_size=(2, 2),
        ).eval()
        with torch.no_grad():
            for layer in adapter.sim_real_discriminator:
                if isinstance(layer, nn.Linear):
                    layer.weight.fill_(0.1)
                    layer.bias.fill_(0.1)

        output = adapter(
            agent_features,
            torch.ones(1, 2, 2, 2),
            record_len,
            _identity_pairwise(record_len),
            grl_lambda=0.0,
            agent_confidence_logits=torch.zeros(2, 2, 2, 2),
        )
        output["domain_logits"].sum().backward()

        self.assertLess(agent_features.grad[0].mean().item(), 0.0)
        self.assertGreater(
            adapter.location_selection_map.grad.mean().item(), 0.0
        )
        torch.testing.assert_close(
            agent_features.grad[1], torch.zeros_like(agent_features.grad[1])
        )

    def test_dusa_confidence_uses_anchor_mean_before_pair_minimum(self):
        eighty_percent_logit = math.log(0.8 / 0.2)
        logits = torch.tensor(
            [
                [[[10.0]], [[-10.0]]],
                [
                    [[eighty_percent_logit]],
                    [[eighty_percent_logit]],
                ],
            ]
        )

        weights = DUSAAdapter._confidence_weights(
            logits, torch.tensor([2])
        )

        expected = torch.full_like(weights, 0.5)
        torch.testing.assert_close(weights, expected, atol=1.0e-4, rtol=0)

    def test_iada_returns_only_graph_alignment_outputs(self):
        agent_features, fused_features, _, _ = _adapter_inputs()
        record_len = torch.tensor([2, 1, 1])
        pairwise = _identity_pairwise(record_len)
        adapter = IADAAdapter(
            in_channels=4,
            graph_dim=6,
            discriminator_hidden_dim=4,
        )

        output = adapter(
            agent_features,
            fused_features.new_empty(3, 4, 2, 2),
            record_len,
            pairwise,
            grl_lambda=0.5,
        )

        self.assertEqual(output["domain_logits"].shape, (3, 1))
        self.assertEqual(output["graph_embedding"].shape, (3, 6))
        torch.testing.assert_close(
            output["valid_graph_mask"],
            torch.tensor([True, False, False]),
        )
        self.assertNotIn("interaction_logits", output)

    def test_ssda_exposes_entropy_weighted_global_and_local_heads(self):
        agent_features, fused_features, record_len, pairwise = (
            _adapter_inputs()
        )
        adapter = SSDAAdapter(in_channels=4, hidden_dim=4).eval()
        refined, _ = adapter.adapt_agents(agent_features, record_len)
        class_logits = torch.randn(4, 2, 2, 2)

        output = adapter(
            refined,
            fused_features,
            record_len,
            pairwise,
            grl_lambda=0.5,
            agent_confidence_logits=class_logits,
        )

        self.assertEqual(output["ssda_global_logits"].shape, (2, 1, 2, 2))
        self.assertEqual(output["ssda_global_attention"].shape, (2, 1, 2, 2))
        self.assertEqual(output["ssda_local_logits"].shape, (4, 1, 2, 2))
        self.assertEqual(output["ssda_local_attention"].shape, (4, 1, 2, 2))
        self.assertTrue(
            ((output["ssda_local_attention"] >= 0)
             & (output["ssda_local_attention"] <= 1)).all().item()
        )

    def test_cudax_heads_observe_fused_outputs_without_replacing_them(self):
        agent_features, fused_features, record_len, pairwise = (
            _adapter_inputs()
        )
        adapter = CUDAXAdapter(
            in_channels=4,
            detection_channels=4,
            anchor_number=2,
            hidden_dim=4,
            discriminator_hidden_dim=4,
            ckt_groups=2,
            bin_count=3,
            feature_size=(2, 2),
            cpa_hidden_dim=4,
        )

        adapted_features, context = adapter.adapt_fused(
            agent_features,
            fused_features,
            record_len,
            grl_lambda=0.5,
        )
        fused_class_logits = torch.randn(2, 2, 2, 2, requires_grad=True)
        output = adapter(
            agent_features,
            adapted_features,
            record_len,
            pairwise,
            grl_lambda=0.5,
            fused_class_logits=fused_class_logits,
            context=context,
            detection_features=adapted_features,
        )

        self.assertIs(adapted_features, fused_features)
        self.assertEqual(output["ckt_domain_logits"].shape, (2, 1))
        self.assertEqual(output["blc_domain_logits"].shape, (2, 1, 2, 2))
        self.assertEqual(output["cpa_domain_logits"].shape, (2, 2))
        self.assertEqual(output["bin_logits"].shape, (2, 36, 2, 2))
        self.assertNotIn("interaction_logits", output)
        output["cpa_domain_logits"].sum().backward()
        self.assertIsNotNone(fused_class_logits.grad)
        self.assertTrue(torch.isfinite(fused_class_logits.grad).all().item())


class AdaptationLossTest(unittest.TestCase):
    def test_entropy_weighted_domain_loss_balances_source_and_target(self):
        logits = torch.zeros(3, 1, 2, 2, requires_grad=True)
        attention = torch.tensor([1.0, 0.5, 0.25]).reshape(3, 1, 1, 1)

        loss, accuracy, valid_count = entropy_weighted_domain_loss(
            logits,
            attention,
            torch.tensor([0.0, 1.0]),
            scene_indices=torch.tensor([0, 0, 1]),
        )

        self.assertAlmostEqual(loss.item(), math.log(2.0), places=6)
        self.assertAlmostEqual(accuracy.item(), 0.5, places=6)
        self.assertEqual(valid_count, 12)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all().item())

    def test_balanced_domain_loss_supports_agent_pixel_logits(self):
        logits = torch.zeros(3, 1, 2, 2, requires_grad=True)
        labels = torch.tensor([0.0, 1.0])
        scene_indices = torch.tensor([0, 0, 1])

        loss, accuracy, valid_count = balanced_domain_loss(
            logits, labels, scene_indices=scene_indices
        )

        self.assertAlmostEqual(loss.item(), math.log(2.0), places=6)
        self.assertAlmostEqual(accuracy.item(), 0.5, places=6)
        self.assertEqual(valid_count, 12)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all().item())

    def test_dusa_cia_uses_only_complete_target_pairs(self):
        logits = torch.zeros(4, 1, 2, 2, requires_grad=True)
        weights = torch.ones_like(logits)
        scene_indices = torch.tensor([0, 0, 1, 1])
        local_indices = torch.tensor([0, 1, 0, 1])
        domain_labels = torch.tensor([0.0, 1.0])
        record_len = torch.tensor([2, 2])

        loss, accuracy, valid_count = dusa_agent_loss(
            logits,
            weights,
            scene_indices,
            local_indices,
            domain_labels,
            record_len,
        )
        changed_source = logits.detach().clone()
        changed_source[:2] = 100.0
        changed_loss, _, _ = dusa_agent_loss(
            changed_source,
            weights,
            scene_indices,
            local_indices,
            domain_labels,
            record_len,
        )

        self.assertAlmostEqual(loss.item(), math.log(2.0), places=5)
        self.assertAlmostEqual(accuracy.item(), 0.5, places=6)
        self.assertEqual(valid_count, 8)
        torch.testing.assert_close(changed_loss, loss.detach())

    def test_cudax_bin_loss_is_differentiable_zero_without_positives(self):
        logits = torch.randn(1, 24, 2, 2, requires_grad=True)
        targets = {
            "targets": torch.zeros(1, 2, 2, 14),
            "pos_equal_one": torch.zeros(1, 2, 2, 2),
        }

        loss = cudax_bin_loss(
            logits, targets, bin_count=2, residual_bounds=[1.0] * 6
        )

        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        torch.testing.assert_close(logits.grad, torch.zeros_like(logits))

    def test_loss_dispatch_matches_supported_methods(self):
        labels = torch.tensor([0.0, 1.0])
        record_len = torch.tensor([2, 2])
        empty_targets = {}

        grl_loss, _ = compute_adaptation_loss(
            "grl",
            {
                "domain_logits": torch.zeros(4, 1),
                "domain_scene_index": torch.tensor([0, 0, 1, 1]),
                "domain_valid_mask": torch.ones(4, dtype=torch.bool),
            },
            labels,
            source_scene_count=1,
            record_len=record_len,
            source_label_dict=empty_targets,
            config={"domain_loss_weight": 0.3},
        )
        self.assertAlmostEqual(
            grl_loss.item(), 0.3 * math.log(2.0), places=6
        )

        dusa_loss, _ = compute_adaptation_loss(
            "dusa",
            {
                "domain_logits": torch.zeros(2, 1),
                "domain_scene_index": torch.arange(2),
                "domain_valid_mask": torch.ones(2, dtype=torch.bool),
                "agent_domain_logits": torch.zeros(4, 1, 1, 1),
                "agent_domain_weights": torch.ones(4, 1, 1, 1),
                "agent_scene_index": torch.tensor([0, 0, 1, 1]),
                "agent_local_index": torch.tensor([0, 1, 0, 1]),
            },
            labels,
            source_scene_count=1,
            record_len=record_len,
            source_label_dict=empty_targets,
            config={"dusa_lsa_weight": 0.5, "dusa_cia_weight": 0.25},
        )
        self.assertAlmostEqual(
            dusa_loss.item(), 0.75 * math.log(2.0), places=6
        )

        iada_loss, metrics = compute_adaptation_loss(
            "iada",
            {
                "domain_logits": torch.zeros(2, 1),
                "domain_scene_index": torch.arange(2),
                "domain_valid_mask": torch.ones(2, dtype=torch.bool),
                "graph_embedding": torch.zeros(2, 3),
                "valid_graph_mask": torch.ones(2, dtype=torch.bool),
            },
            labels,
            source_scene_count=1,
            record_len=record_len,
            source_label_dict=empty_targets,
            config={
                "domain_loss_weight": 0.2,
                "graph_variance_target_std": 0.0,
                "graph_variance_weight": 1.0,
            },
        )
        self.assertAlmostEqual(
            iada_loss.item(), 0.2 * math.log(2.0), places=6
        )
        self.assertEqual(metrics["graph_variance_update_applied"].item(), 0)

        ssda_loss, metrics = compute_adaptation_loss(
            "ssda",
            {
                "ssda_global_logits": torch.zeros(2, 1, 1, 1),
                "ssda_global_attention": torch.ones(2, 1, 1, 1),
                "ssda_global_scene_index": torch.arange(2),
                "ssda_local_logits": torch.zeros(4, 1, 1, 1),
                "ssda_local_attention": torch.ones(4, 1, 1, 1),
                "ssda_local_scene_index": torch.tensor([0, 0, 1, 1]),
            },
            labels,
            source_scene_count=1,
            record_len=record_len,
            source_label_dict=empty_targets,
            config={"ssda_global_weight": 0.5, "ssda_local_weight": 1.0},
        )
        self.assertAlmostEqual(
            ssda_loss.item(), 1.5 * math.log(2.0), places=6
        )
        self.assertAlmostEqual(
            metrics["domain_loss"].item(), 2.0 * math.log(2.0), places=6
        )

    def test_cudax_bin_supervision_cannot_observe_target_labels(self):
        labels = torch.tensor([0.0, 1.0])
        record_len = torch.tensor([2, 2])
        output = {
            "ckt_domain_logits": torch.zeros(2, 1),
            "blc_domain_logits": torch.zeros(2, 1, 2, 2),
            "cpa_domain_logits": torch.zeros(2, 2),
            "bin_logits": torch.zeros(2, 24, 2, 2),
            "domain_scene_index": torch.arange(2),
            "domain_valid_mask": torch.ones(2, dtype=torch.bool),
        }
        source_targets = {
            "targets": torch.zeros(1, 2, 2, 14),
            "pos_equal_one": torch.zeros(1, 2, 2, 2),
        }
        source_targets["pos_equal_one"][0, 0, 0, 0] = 1
        config = {
            "cudax_bin_count": 2,
            "cudax_residual_bounds": [1.0] * 6,
            "cudax_bin_loss_weight": 0.25,
            "cudax_domain_loss_weight": 0.1,
        }

        loss, metrics = compute_adaptation_loss(
            "cudax",
            output,
            labels,
            source_scene_count=1,
            record_len=record_len,
            source_label_dict=source_targets,
            config=config,
        )
        changed_target_output = dict(output)
        changed_bins = output["bin_logits"].clone()
        changed_bins[1] = torch.randn_like(changed_bins[1]) * 1000.0
        changed_target_output["bin_logits"] = changed_bins
        changed_loss, _ = compute_adaptation_loss(
            "cudax",
            changed_target_output,
            labels,
            source_scene_count=1,
            record_len=record_len,
            source_label_dict=source_targets,
            config=config,
        )

        expected = 0.55 * math.log(2.0)
        self.assertAlmostEqual(loss.item(), expected, places=6)
        self.assertAlmostEqual(metrics["bin_loss"].item(), math.log(2.0), places=6)
        torch.testing.assert_close(changed_loss, loss)


class CUDAXBoundUtilityTest(unittest.TestCase):
    def test_uses_only_positive_source_anchors_and_excludes_yaw(self):
        targets = np.array(
            [
                [
                    [
                        1.0,
                        -2.0,
                        3.0,
                        -4.0,
                        5.0,
                        -6.0,
                        1000.0,
                        99.0,
                        99.0,
                        99.0,
                        99.0,
                        99.0,
                        99.0,
                        99.0,
                    ]
                ]
            ]
        )
        dataset = [
            {
                "ego": {
                    "label_dict": {
                        "targets": targets,
                        "pos_equal_one": np.array([[[1, 0]]]),
                    }
                }
            }
        ]

        bounds, positive_count, sample_count = (
            compute_source_residual_bounds(dataset)
        )

        np.testing.assert_allclose(bounds, [1, 2, 3, 4, 5, 6])
        self.assertEqual(positive_count, 1)
        self.assertEqual(sample_count, 1)


def _point_pillar_args(domain_adapter):
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
        "fusion_method": "max",
        "domain_adapter": domain_adapter,
        "dir_args": {"num_bins": 2},
    }


def _point_pillar_input():
    torch.manual_seed(73)
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
        "pairwise_t_matrix": _identity_pairwise(
            torch.tensor([2, 1]), dtype=torch.float64
        ),
        "grl_lambda": 0.5,
    }


class _CaptureV2XTransformer(nn.Module):
    def forward(self, features, mask, spatial_correction_matrix):
        self.features = features.detach().clone()
        self.mask = mask.detach().clone()
        self.spatial_correction_matrix = spatial_correction_matrix
        return features[:, 0, ..., :-3]


class V2XViTPriorEncodingTest(unittest.TestCase):
    def _fusion(self, fallback="local_index_1_infra"):
        capture = _CaptureV2XTransformer()
        with patch(
            "opencood.models.sub_modules.v2xvit_basic.V2XTransformer",
            return_value=capture,
        ):
            fusion = V2XViTFusion(
                {
                    "transformer": {},
                    "prior_encoding_fallback": fallback,
                }
            )
        return fusion, capture

    def test_fallback_respects_scene_mask_and_explicit_prior_overrides_it(self):
        fusion, capture = self._fusion()
        features = torch.randn(3, 4, 2, 2)
        record_len = torch.tensor([2, 1])
        affine = _identity_affine(2, 2)

        output = fusion(features, record_len, affine)

        self.assertEqual(output.shape, (2, 4, 2, 2))
        torch.testing.assert_close(
            capture.mask, torch.tensor([[1, 1], [1, 0]])
        )
        expected = torch.zeros(2, 2, 2, 2, 3)
        expected[0, 1, ..., 2] = 1
        torch.testing.assert_close(capture.features[..., -3:], expected)

        explicit_source_prior = torch.zeros(2, 2, 3, 1, 1)
        fusion(
            features,
            record_len,
            affine,
            prior_encoding=explicit_source_prior,
        )
        torch.testing.assert_close(
            capture.features[..., -3:], torch.zeros_like(expected)
        )

    def test_rejects_invalid_shape_dtype_device_padding_and_role(self):
        fusion, _ = self._fusion("zeros")
        features = torch.randn(3, 4, 2, 2)
        record_len = torch.tensor([2, 1])
        affine = _identity_affine(2, 2)
        invalid_cases = (
            (torch.zeros(2, 2, 2), ValueError),
            (torch.zeros(2, 2, 3, dtype=torch.int64), TypeError),
            (torch.zeros(2, 2, 3, dtype=torch.float64), TypeError),
        )
        for prior, error_type in invalid_cases:
            with self.subTest(shape=tuple(prior.shape), dtype=prior.dtype):
                with self.assertRaises(error_type):
                    fusion(features, record_len, affine, prior)

        padded = torch.zeros(2, 2, 3)
        padded[1, 1, 0] = 1
        with self.assertRaisesRegex(ValueError, "padding"):
            fusion(features, record_len, affine, padded)

        invalid_role = torch.zeros(2, 2, 3)
        invalid_role[0, 1, 2] = 2
        with self.assertRaisesRegex(ValueError, "exactly 0 or 1"):
            fusion(features, record_len, affine, invalid_role)

        with self.assertRaisesRegex(ValueError, "same device"):
            fusion(
                features,
                record_len,
                affine,
                torch.empty(2, 2, 3, device="meta"),
            )


class PointPillarBaselineAdapterTest(unittest.TestCase):
    def test_ssda_runs_fsa_before_fusion_and_saa_after_detection_head(self):
        args = _point_pillar_args(
            {
                "enabled": True,
                "method": "ssda",
                "ssda_hidden_dim": 4,
            }
        )
        model = PointPillarBaseline(args).eval()

        with torch.no_grad():
            output = model(_point_pillar_input())

        self.assertEqual(output["cls_preds"].shape, (2, 2, 4, 4))
        self.assertEqual(output["ssda_global_logits"].shape, (2, 1, 4, 4))
        self.assertEqual(output["ssda_local_logits"].shape, (3, 1, 4, 4))
        torch.testing.assert_close(
            output["ssda_local_scene_index"], torch.tensor([0, 0, 1])
        )

    def test_native_fusion_modules_are_constructed_outside_adapter(self):
        cases = (
            ("att", "AttFusion", "att", {"feat_dim": 384}, 384),
            (
                "disconet",
                "DiscoFusion",
                "disconet",
                {"feat_dim": 384},
                384,
            ),
            (
                "v2xvit",
                "V2XViTFusion",
                "v2xvit",
                {"transformer": {"sentinel": True}},
                {"transformer": {"sentinel": True}},
            ),
        )
        for method, class_name, config_key, config, expected_arg in cases:
            with self.subTest(method=method):
                args = _point_pillar_args(
                    {"enabled": True, "method": "grl", "hidden_dim": 4}
                )
                args["fusion_method"] = method
                args[config_key] = config
                replacement = nn.Identity()
                patch_path = (
                    "opencood.models.point_pillar_baseline." + class_name
                )
                with patch(patch_path, return_value=replacement) as factory:
                    model = PointPillarBaseline(args)
                factory.assert_called_once_with(expected_arg)
                self.assertIs(model.fusion_net, replacement)
                self.assertIsInstance(
                    model.domain_adapter, NaiveDomainAdapter
                )

    def test_named_native_fusions_run_through_the_common_adapter(self):
        v2xvit = {
            "transformer": {
                "encoder": {
                    "num_blocks": 1,
                    "depth": 1,
                    "use_roi_mask": True,
                    "use_RTE": False,
                    "RTE_ratio": 0,
                    "cav_att_config": {
                        "dim": 384,
                        "use_hetero": False,
                        "use_RTE": False,
                        "RTE_ratio": 0,
                        "heads": 1,
                        "dim_head": 384,
                        "dropout": 0.0,
                    },
                    "pwindow_att_config": {
                        "dim": 384,
                        "heads": [1],
                        "dim_head": [384],
                        "dropout": 0.0,
                        "window_size": [1],
                        "relative_pos_embedding": True,
                        "fusion_method": "naive",
                    },
                    "feed_forward": {
                        "mlp_dim": 384,
                        "dropout": 0.0,
                    },
                    "sttf": {
                        "voxel_size": [1.0, 1.0, 4.0],
                        "downsample_rate": 2,
                    },
                }
            }
        }
        cases = (
            ("att", "att", {"feat_dim": 384}),
            ("disconet", "disconet", {"feat_dim": 384}),
            ("v2xvit", "v2xvit", v2xvit),
        )
        for method, config_key, config in cases:
            with self.subTest(method=method):
                args = _point_pillar_args(
                    {"enabled": True, "method": "grl", "hidden_dim": 4}
                )
                args["fusion_method"] = method
                args[config_key] = config
                model = PointPillarBaseline(args).eval()

                with torch.no_grad():
                    output = model(copy.deepcopy(_point_pillar_input()))

                self.assertEqual(output["cls_preds"].shape, (2, 2, 4, 4))
                self.assertEqual(output["domain_logits"].shape, (3, 1))

    def test_v2xvit_receives_prior_encoding_from_model_input(self):
        class CaptureFusion(nn.Module):
            def forward(self, features, record_len, affine,
                        prior_encoding=None):
                self.prior_encoding = prior_encoding
                ego_indices = torch.cat(
                    [record_len.new_zeros(1), record_len.cumsum(0)[:-1]]
                )
                return features.index_select(0, ego_indices)

        args = _point_pillar_args({"enabled": False})
        args["fusion_method"] = "v2xvit"
        args["v2xvit"] = {"transformer": {}}
        capture = CaptureFusion()
        with patch(
            "opencood.models.point_pillar_baseline.V2XViTFusion",
            return_value=capture,
        ):
            model = PointPillarBaseline(args).eval()
        model_input = _point_pillar_input()
        model_input["prior_encoding"] = torch.zeros(2, 2, 3)
        model_input["prior_encoding"][0, 1, 2] = 1

        with torch.no_grad():
            model(model_input)

        torch.testing.assert_close(
            capture.prior_encoding, model_input["prior_encoding"]
        )

    def test_all_adapters_preserve_native_max_fusion_detection(self):
        baseline = PointPillarBaseline(
            _point_pillar_args({"enabled": False})
        ).eval()
        model_input = _point_pillar_input()
        with torch.no_grad():
            baseline_output = baseline(copy.deepcopy(model_input))

        configs = {
            "grl": {"hidden_dim": 8},
            "dusa": {
                "feature_size": [4, 4],
                "dusa_lsa_hidden_dim": 8,
                "dusa_cia_hidden_dim": 8,
            },
            "iada": {"hidden_dim": 8, "graph_dim": 8},
            "cudax": {
                "hidden_dim": 8,
                "feature_size": [4, 4],
                "cudax_hidden_dim": 8,
                "cpa_hidden_dim": 8,
                "ckt_groups": 8,
                "bin_count": 3,
            },
        }
        expected_adapter_outputs = {
            "grl": {"domain_logits"},
            "dusa": {"domain_logits", "agent_domain_logits"},
            "iada": {"domain_logits", "graph_embedding"},
            "cudax": {
                "ckt_domain_logits",
                "blc_domain_logits",
                "cpa_domain_logits",
                "bin_logits",
            },
        }

        for method, method_config in configs.items():
            with self.subTest(method=method):
                adapter_config = {
                    "enabled": True,
                    "method": method,
                    **method_config,
                }
                adapted = PointPillarBaseline(
                    _point_pillar_args(adapter_config)
                ).eval()
                incompatible = adapted.load_state_dict(
                    baseline.state_dict(), strict=False
                )
                self.assertEqual(incompatible.unexpected_keys, [])
                self.assertTrue(incompatible.missing_keys)
                self.assertTrue(
                    all(
                        key.startswith("domain_adapter.")
                        for key in incompatible.missing_keys
                    )
                )

                with torch.no_grad():
                    output = adapted(copy.deepcopy(model_input))

                for key in ("cls_preds", "reg_preds", "dir_preds"):
                    torch.testing.assert_close(
                        output[key], baseline_output[key]
                    )
                self.assertTrue(
                    expected_adapter_outputs[method].issubset(output)
                )
                self.assertNotIn("interaction_logits", output)


if __name__ == "__main__":
    unittest.main()
