import copy
import math
import os
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn
import yaml

from opencood.tools.train_iada import (
    TRAINING_STATE_FILENAME,
    _balanced_domain_loss,
    _configure_stage,
    _find_model_checkpoint,
    _graph_variance_floor_loss,
    _grl_lambda,
    _load_model_checkpoint,
    _load_training_state,
    _resolve_best_validation_path,
    _restore_random_state,
    _save_training_state,
    _select_model_inputs,
    _setup_stage_optimizer,
    _slice_detection_output,
    _validate_baseline_warm_start,
)


def _hypes():
    return {
        "name": "experiment",
        "model": {
            "args": {
                "domain_adapter": {
                    "enabled": True,
                    "method": "iada",
                    "feature_size": [2, 2],
                    "cudax_bin_count": 5,
                }
            }
        },
        "domain_adaptation": {
            "stage": "iada",
            "mode": "uda",
            "cudax_bin_count": 5,
            "cudax_residual_bounds": [1.0] * 6,
        },
    }


def _protocol_hypes(stage):
    return {
        "data_dir": "/data/dair",
        "root_dir": "/data/dair/train.json",
        "validate_dir": "/data/dair/val.json",
        "test_dir": "/data/dair/val.json",
        "domain_adaptation": {
            "stage": stage,
            "source": {
                "root_dir": "/data/opv2v/train",
                "validate_dir": "/data/opv2v/validate",
                "test_dir": "/data/opv2v/test",
                "fusion": {"dataset": "opv2v"},
            },
        },
        "fusion": {
            "core_method": "intermediate",
            "dataset": "dairv2x",
            "args": {"proj_first": False},
        },
        "preprocess": {
            "core_method": "SpVoxelPreprocessor",
            "cav_lidar_range": [-100.8, -40, -3.5, 100.8, 40, 1.5],
            "args": {"voxel_size": [0.4, 0.4, 5]},
        },
        "postprocess": {
            "core_method": "VoxelPostprocessor",
            "anchor_args": {
                "l": 4.5,
                "w": 2.0,
                "h": 1.56,
                "r": [0, 90],
            },
        },
        "model": {
            "core_method": "point_pillar_baseline",
            "args": {
                "anchor_number": 2,
                "base_bev_backbone": {"num_filters": [64, 128, 256]},
                "domain_adapter": {
                    "enabled": stage != "baseline",
                    "method": "none" if stage == "baseline" else stage,
                },
            },
        },
        "train_params": {"max_cav": 2},
        "comm_range": 100,
        "noise_setting": {"add_noise": False},
        "input_source": ["lidar"],
        "label_type": "lidar",
    }


class ConfigureStageTest(unittest.TestCase):
    def test_stage_switches_are_explicit(self):
        for stage in (
            "baseline",
            "grl",
            "dusa",
            "cudax",
            "iada",
            "ssda",
        ):
            with self.subTest(stage=stage):
                hypes = copy.deepcopy(_hypes())
                self.assertEqual(_configure_stage(hypes, stage), stage)
                adapter = hypes["model"]["args"]["domain_adapter"]
                self.assertEqual(adapter["enabled"], stage != "baseline")
                self.assertEqual(
                    adapter["method"],
                    "none" if stage == "baseline" else stage,
                )
                self.assertEqual(
                    hypes["domain_adaptation"]["mode"],
                    "source_only" if stage == "baseline" else "uda",
                )
                self.assertTrue(hypes["name"].endswith(f"_{stage}"))

    def test_unknown_stage_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported training stage"):
            _configure_stage(copy.deepcopy(_hypes()), "not-a-stage")

    def test_cudax_rejects_missing_source_residual_bounds(self):
        hypes = _hypes()
        hypes["domain_adaptation"]["cudax_residual_bounds"] = []

        with self.assertRaisesRegex(ValueError, "source-only"):
            _configure_stage(hypes, "cudax")

    def test_cudax_rejects_model_loss_bin_count_mismatch(self):
        hypes = _hypes()
        hypes["domain_adaptation"]["cudax_bin_count"] = 3

        with self.assertRaisesRegex(ValueError, "bin count differs"):
            _configure_stage(hypes, "cudax")


class DomainLossTest(unittest.TestCase):
    def test_balances_domains_and_ignores_single_agent_graphs(self):
        logits = torch.tensor([[0.0], [100.0], [0.0], [-100.0]])
        labels = torch.tensor([0.0, 0.0, 1.0, 1.0])
        valid = torch.tensor([True, False, True, False])

        loss, accuracy, valid_count = _balanced_domain_loss(
            logits,
            labels,
            valid,
            source_scene_count=2,
        )

        self.assertEqual(valid_count, 2)
        self.assertAlmostEqual(loss.item(), math.log(2.0), places=6)
        self.assertAlmostEqual(accuracy.item(), 0.5, places=6)

    def test_all_invalid_graphs_return_differentiable_zero(self):
        logits = torch.randn(4, 1, requires_grad=True)
        loss, accuracy, valid_count = _balanced_domain_loss(
            logits,
            torch.tensor([0.0, 0.0, 1.0, 1.0]),
            torch.zeros(4, dtype=torch.bool),
            source_scene_count=2,
        )

        self.assertEqual(loss.item(), 0.0)
        self.assertTrue(torch.isnan(accuracy))
        self.assertEqual(valid_count, 0)
        loss.backward()
        torch.testing.assert_close(
            logits.grad, torch.zeros_like(logits)
        )

    def test_one_sided_valid_batch_skips_adversarial_update(self):
        logits = torch.randn(4, 1, requires_grad=True)
        loss, _, valid_count = _balanced_domain_loss(
            logits,
            torch.tensor([0.0, 0.0, 1.0, 1.0]),
            torch.tensor([True, True, False, False]),
            source_scene_count=2,
        )

        self.assertEqual(valid_count, 2)
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        torch.testing.assert_close(
            logits.grad, torch.zeros_like(logits)
        )


class GraphVarianceFloorLossTest(unittest.TestCase):
    def test_penalizes_collapsed_domains_and_backpropagates(self):
        embedding = torch.tensor(
            [
                [0.0, 0.0],
                [0.001, -0.001],
                [2.0, 2.0],
                [2.001, 1.999],
            ],
            requires_grad=True,
        )
        loss, applied = _graph_variance_floor_loss(
            embedding,
            torch.ones(4, dtype=torch.bool),
            source_scene_count=2,
            target_std=0.01,
        )

        self.assertEqual(applied, 1)
        self.assertGreater(loss.item(), 0.0)
        loss.backward()
        self.assertTrue(torch.isfinite(embedding.grad).all().item())
        self.assertGreater(embedding.grad.abs().sum().item(), 0.0)

    def test_is_zero_when_both_domains_already_meet_floor(self):
        embedding = torch.tensor(
            [[-1.0], [1.0], [-2.0], [2.0]], requires_grad=True
        )
        loss, applied = _graph_variance_floor_loss(
            embedding,
            torch.ones(4, dtype=torch.bool),
            source_scene_count=2,
            target_std=0.01,
        )

        self.assertEqual(applied, 1)
        self.assertEqual(loss.item(), 0.0)

    def test_aggregate_floor_does_not_force_every_channel_active(self):
        embedding = torch.tensor(
            [
                [-1.0, 0.0],
                [1.0, 0.0],
                [-2.0, 3.0],
                [2.0, 3.0],
            ],
            requires_grad=True,
        )
        loss, applied = _graph_variance_floor_loss(
            embedding,
            torch.ones(4, dtype=torch.bool),
            source_scene_count=2,
            target_std=0.01,
        )

        self.assertEqual(applied, 1)
        self.assertEqual(loss.item(), 0.0)

    def test_skips_unbalanced_or_too_small_valid_sets(self):
        embedding = torch.randn(4, 3, requires_grad=True)
        loss, applied = _graph_variance_floor_loss(
            embedding,
            torch.tensor([True, True, True, False]),
            source_scene_count=2,
            target_std=0.01,
        )

        self.assertEqual(applied, 0)
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        torch.testing.assert_close(
            embedding.grad, torch.zeros_like(embedding)
        )


class DetectionSliceTest(unittest.TestCase):
    def test_target_predictions_are_excluded(self):
        output = {
            "cls_preds": torch.arange(4).reshape(4, 1, 1, 1),
            "reg_preds": torch.arange(8).reshape(4, 2, 1, 1),
            "domain_logits": torch.randn(4, 1),
        }

        source_output = _slice_detection_output(
            output, source_scene_count=2
        )

        self.assertEqual(set(source_output), {"cls_preds", "reg_preds"})
        self.assertEqual(source_output["cls_preds"].shape[0], 2)
        self.assertNotIn("domain_logits", source_output)


class ModelInputSelectionTest(unittest.TestCase):
    def test_source_only_input_has_explicit_zero_prior(self):
        ego_batch = {
            "processed_lidar": {"sentinel": torch.tensor(1)},
            "record_len": torch.tensor([2, 1]),
            "pairwise_t_matrix": torch.eye(4).view(
                1, 1, 1, 4, 4
            ).repeat(2, 3, 3, 1, 1),
            "lidar_pose": torch.zeros(3, 6),
        }

        model_inputs = _select_model_inputs(ego_batch)

        self.assertEqual(model_inputs["prior_encoding"].shape, (2, 3, 3))
        torch.testing.assert_close(
            model_inputs["prior_encoding"],
            torch.zeros(2, 3, 3),
        )

    def test_source_only_input_rejects_a_supplied_infrastructure_type(self):
        ego_batch = {
            "processed_lidar": {"sentinel": torch.tensor(1)},
            "record_len": torch.tensor([2]),
            "pairwise_t_matrix": torch.eye(4).view(
                1, 1, 1, 4, 4
            ).repeat(1, 2, 2, 1, 1),
            "lidar_pose": torch.zeros(2, 6),
            "prior_encoding": torch.zeros(1, 2, 3),
        }
        ego_batch["prior_encoding"][0, 1, 2] = 1.0

        with self.assertRaisesRegex(ValueError, "canonical Sim2Real"):
            _select_model_inputs(ego_batch)


class GradientScheduleTest(unittest.TestCase):
    def test_dann_schedule_starts_zero_and_approaches_max(self):
        self.assertEqual(_grl_lambda(0, 100, 1.0, 10.0), 0.0)
        self.assertGreater(_grl_lambda(99, 100, 1.0, 10.0), 0.999)


class StageOptimizerTest(unittest.TestCase):
    def test_all_adaptation_stages_use_discriminative_learning_rates(self):
        for stage in ("grl", "dusa", "cudax", "iada", "ssda"):
            with self.subTest(stage=stage):
                model = _WarmStartModel()
                hypes = {
                    "optimizer": {
                        "core_method": "Adam",
                        "lr": 2.0e-4,
                        "args": {"weight_decay": 1.0e-4},
                    },
                    "domain_adaptation": {
                        "pretrained_lr_scale": 0.1,
                        "adapter_lr_scale": 1.0,
                    },
                }

                optimizer = _setup_stage_optimizer(hypes, model, stage)

                self.assertEqual(
                    [
                        group["group_name"]
                        for group in optimizer.param_groups
                    ],
                    ["pretrained", "adapter"],
                )
                self.assertAlmostEqual(
                    optimizer.param_groups[0]["lr"], 2.0e-5
                )
                self.assertAlmostEqual(
                    optimizer.param_groups[1]["lr"], 2.0e-4
                )
                grouped_ids = {
                    id(parameter)
                    for group in optimizer.param_groups
                    for parameter in group["params"]
                }
                self.assertEqual(
                    grouped_ids,
                    {id(parameter) for parameter in model.parameters()},
                )


class _WarmStartModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(2, 2)
        self.domain_adapter = nn.Linear(2, 1)


class CheckpointCompatibilityTest(unittest.TestCase):
    def test_warm_start_allows_only_adapter_keys_to_be_missing(self):
        model = _WarmStartModel()
        source_state = {
            key: value.clone()
            for key, value in model.state_dict().items()
            if not key.startswith("domain_adapter.")
        }

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = os.path.join(directory, "net_epoch1.pth")
            torch.save(source_state, checkpoint)
            missing, unexpected = _load_model_checkpoint(
                model,
                checkpoint,
                allow_missing_adapter=True,
            )

        self.assertEqual(unexpected, [])
        self.assertEqual(
            set(missing),
            {"domain_adapter.weight", "domain_adapter.bias"},
        )

    def test_warm_start_rejects_missing_shared_encoder_keys(self):
        model = _WarmStartModel()
        source_state = {
            key: value.clone()
            for key, value in model.state_dict().items()
            if key != "encoder.bias"
        }

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = os.path.join(directory, "net_epoch1.pth")
            torch.save(source_state, checkpoint)
            with self.assertRaisesRegex(RuntimeError, "encoder.bias"):
                _load_model_checkpoint(
                    model,
                    checkpoint,
                    allow_missing_adapter=True,
                )

    def test_adaptation_warm_start_rejects_an_existing_adapter(self):
        model = _WarmStartModel()

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = os.path.join(directory, "net_epoch1.pth")
            torch.save(model.state_dict(), checkpoint)
            with self.assertRaisesRegex(ValueError, "fresh adapter branch"):
                _load_model_checkpoint(
                    model,
                    checkpoint,
                    allow_missing_adapter=True,
                    require_fresh_adapter=True,
                )

    def test_resume_prefers_latest_periodic_but_warm_start_prefers_best(self):
        with tempfile.TemporaryDirectory() as directory:
            for filename in (
                "net_epoch2.pth",
                "net_epoch7.pth",
                "net_epoch_bestval_at4.pth",
            ):
                torch.save({}, os.path.join(directory, filename))

            resume_path = _find_model_checkpoint(
                directory, prefer_best=False
            )
            warm_start_path = _find_model_checkpoint(
                directory, prefer_best=True
            )

        self.assertEqual(os.path.basename(resume_path), "net_epoch7.pth")
        self.assertEqual(
            os.path.basename(warm_start_path),
            "net_epoch_bestval_at4.pth",
        )

    def test_adaptation_requires_matching_common_geometry_baseline_config(self):
        baseline = _protocol_hypes("baseline")
        current = _protocol_hypes("iada")

        with tempfile.TemporaryDirectory() as directory:
            config_path = os.path.join(directory, "config.yaml")
            with open(config_path, "w") as stream:
                yaml.safe_dump(baseline, stream)

            _validate_baseline_warm_start(directory, current)

            baseline["preprocess"]["cav_lidar_range"][0] = -140.8
            with open(config_path, "w") as stream:
                yaml.safe_dump(baseline, stream)
            with self.assertRaisesRegex(ValueError, "ROI"):
                _validate_baseline_warm_start(directory, current)

    def test_adaptation_rejects_another_adaptation_as_warm_start(self):
        current = _protocol_hypes("iada")

        with tempfile.TemporaryDirectory() as directory:
            with open(
                os.path.join(directory, "config.yaml"), "w"
            ) as stream:
                yaml.safe_dump(_protocol_hypes("dusa"), stream)

            with self.assertRaisesRegex(ValueError, "source-only baseline"):
                _validate_baseline_warm_start(directory, current)

    def test_adaptation_rejects_a_different_source_or_target_protocol(self):
        current = _protocol_hypes("cudax")

        for mutation in ("source", "target"):
            baseline = _protocol_hypes("baseline")
            if mutation == "source":
                baseline["domain_adaptation"]["source"]["fusion"][
                    "dataset"
                ] = "dairv2x"
            else:
                baseline["root_dir"] = "/data/dair/other_train.json"

            with tempfile.TemporaryDirectory() as directory:
                with open(
                    os.path.join(directory, "config.yaml"), "w"
                ) as stream:
                    yaml.safe_dump(baseline, stream)
                with self.assertRaisesRegex(
                    ValueError, "same OPV2V source"
                ):
                    _validate_baseline_warm_start(directory, current)


class FullTrainingStateTest(unittest.TestCase):
    def test_round_trip_preserves_model_training_and_random_state(self):
        random.seed(51)
        np.random.seed(51)
        torch.manual_seed(51)
        model = nn.Linear(2, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=1
        )
        scaler = torch.amp.GradScaler("cuda", enabled=False)

        loss = model(torch.ones(2, 2)).sum()
        loss.backward()
        optimizer.step()
        scheduler.step()
        expected_parameters = {
            key: value.clone() for key, value in model.state_dict().items()
        }

        with tempfile.TemporaryDirectory() as directory:
            _save_training_state(
                directory,
                stage="iada",
                completed_epoch=3,
                global_step=30,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                best_validation_loss=1.25,
                best_validation_path=None,
            )
            self.assertTrue(
                os.path.isfile(
                    os.path.join(directory, TRAINING_STATE_FILENAME)
                )
            )

            expected_random = (
                random.random(),
                np.random.random(),
                torch.rand(1),
            )
            random.random()
            np.random.random()
            torch.rand(1)

            restored_model = nn.Linear(2, 1)
            state = _load_training_state(
                directory, restored_model, "iada"
            )

        self.assertEqual(state["epoch"], 3)
        self.assertEqual(state["global_step"], 30)
        self.assertEqual(state["best_validation_loss"], 1.25)
        self.assertTrue(state["optimizer"]["state"])
        for key, expected in expected_parameters.items():
            torch.testing.assert_close(
                restored_model.state_dict()[key], expected
            )

        _restore_random_state(state["random_state"])
        actual_random = (
            random.random(),
            np.random.random(),
            torch.rand(1),
        )
        self.assertEqual(actual_random[0], expected_random[0])
        self.assertEqual(actual_random[1], expected_random[1])
        torch.testing.assert_close(actual_random[2], expected_random[2])

    def test_resume_rejects_cross_stage_state(self):
        model = nn.Linear(2, 1)
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=1
        )
        scaler = torch.amp.GradScaler("cuda", enabled=False)

        with tempfile.TemporaryDirectory() as directory:
            _save_training_state(
                directory,
                stage="dusa",
                completed_epoch=1,
                global_step=1,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                best_validation_loss=2.0,
                best_validation_path=None,
            )
            with self.assertRaisesRegex(
                ValueError, "Use --pretrained_model_dir"
            ):
                _load_training_state(directory, model, "cudax")

    def test_resume_archives_best_file_not_committed_in_training_state(self):
        with tempfile.TemporaryDirectory() as directory:
            committed = os.path.join(
                directory, "net_epoch_bestval_at2.pth"
            )
            interrupted = os.path.join(
                directory, "net_epoch_bestval_at3.pth"
            )
            torch.save({}, committed)
            torch.save({}, interrupted)

            with patch("builtins.print"):
                selected = _resolve_best_validation_path(
                    directory,
                    {
                        "best_validation_checkpoint": os.path.basename(
                            committed
                        )
                    },
                )

            self.assertEqual(selected, committed)
            self.assertTrue(os.path.isfile(committed))
            self.assertFalse(os.path.exists(interrupted))
            self.assertTrue(
                os.path.isfile(interrupted + ".orphaned")
            )


if __name__ == "__main__":
    unittest.main()
