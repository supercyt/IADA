import unittest

import torch

from opencood.tools.sim2real_utils import (
    ForeverDataIterator,
    build_prior_encoding,
    build_source_config,
    merge_source_target_batches,
)


class _CountingIterable:
    def __init__(self, values):
        self.values = list(values)
        self.iter_calls = 0

    def __iter__(self):
        self.iter_calls += 1
        return iter(self.values)


def _anchor_grid(height=2, width=3, anchor_count=2):
    anchor = torch.zeros(height, width, anchor_count, 7, dtype=torch.float64)
    for row in range(height):
        for column in range(width):
            anchor[row, column, :, 0] = column
            anchor[row, column, :, 1] = row
    anchor[..., 3] = 1.56
    anchor[..., 4] = 2.0
    anchor[..., 5] = 4.5
    return anchor


def _ego_batch(record_len, voxel_agent_indices, max_agents=3):
    record_len = torch.tensor(record_len, dtype=torch.int64)
    agent_count = int(record_len.sum().item())
    voxel_agent_indices = list(voxel_agent_indices)
    voxel_count = len(voxel_agent_indices)

    voxel_coords = torch.zeros(voxel_count, 4, dtype=torch.int32)
    if voxel_count:
        voxel_coords[:, 0] = torch.tensor(
            voxel_agent_indices, dtype=torch.int32
        )
        voxel_coords[:, 2] = torch.arange(
            voxel_count, dtype=torch.int32
        ) % 2
        voxel_coords[:, 3] = torch.arange(
            voxel_count, dtype=torch.int32
        ) % 3

    pairwise = torch.eye(4, dtype=torch.float64).view(
        1, 1, 1, 4, 4
    ).repeat(record_len.numel(), max_agents, max_agents, 1, 1)

    return {
        "processed_lidar": {
            "voxel_features": torch.arange(
                voxel_count * 8, dtype=torch.float32
            ).reshape(voxel_count, 2, 4),
            "voxel_coords": voxel_coords,
            "voxel_num_points": torch.full(
                (voxel_count,), 2, dtype=torch.int32
            ),
            "ignored_lidar_field": torch.tensor(123),
        },
        "record_len": record_len,
        "pairwise_t_matrix": pairwise,
        "lidar_pose": torch.arange(
            agent_count * 6, dtype=torch.float64
        ).reshape(agent_count, 6),
        "anchor_box": _anchor_grid(),
        "label_dict": {"private_target": torch.tensor(1.0)},
        "object_bbx_center": torch.ones(1, 100, 7),
    }


class BuildSourceConfigTest(unittest.TestCase):
    def test_deep_merge_preserves_shared_target_geometry(self):
        target_config = {
            "name": "target",
            "root_dir": "dair/train.json",
            "fusion": {
                "core_method": "intermediate",
                "dataset": "dairv2x",
                "args": {"proj_first": False, "clip_pc": False},
            },
            "preprocess": {
                "cav_lidar_range": [-100.8, -40, -3.5, 100.8, 40, 1.5],
                "args": {
                    "voxel_size": [0.4, 0.4, 5],
                    "max_voxel_train": 32000,
                },
            },
            "domain_adaptation": {
                "lambda_domain": 0.1,
                "source": {
                    "name": "source",
                    "root_dir": "opv2v/train",
                    "fusion": {
                        "dataset": "opv2v",
                        "args": {"clip_pc": True},
                    },
                    "preprocess": {
                        "args": {"max_voxel_train": 16000}
                    },
                },
            },
        }

        source_config = build_source_config(target_config)

        self.assertEqual(source_config["name"], "source")
        self.assertEqual(source_config["root_dir"], "opv2v/train")
        self.assertEqual(source_config["fusion"]["dataset"], "opv2v")
        self.assertEqual(
            source_config["fusion"]["core_method"], "intermediate"
        )
        self.assertFalse(source_config["fusion"]["args"]["proj_first"])
        self.assertTrue(source_config["fusion"]["args"]["clip_pc"])
        self.assertEqual(
            source_config["preprocess"]["args"]["voxel_size"],
            [0.4, 0.4, 5],
        )
        self.assertEqual(
            source_config["preprocess"]["args"]["max_voxel_train"], 16000
        )
        self.assertNotIn("domain_adaptation", source_config)

        source_config["preprocess"]["args"]["voxel_size"][0] = 99
        self.assertEqual(
            target_config["preprocess"]["args"]["voxel_size"][0], 0.4
        )
        self.assertEqual(
            target_config["fusion"]["args"]["clip_pc"], False
        )

    def test_can_keep_harmless_domain_adaptation_section(self):
        target_config = {
            "fusion": {"dataset": "dairv2x"},
            "domain_adaptation": {
                "lambda_domain": 0.1,
                "source": {"fusion": {"dataset": "opv2v"}},
            },
        }

        source_config = build_source_config(
            target_config, keep_domain_adaptation=True
        )

        self.assertEqual(source_config["fusion"]["dataset"], "opv2v")
        self.assertIn("domain_adaptation", source_config)
        self.assertIsNot(
            source_config["domain_adaptation"],
            target_config["domain_adaptation"],
        )

    def test_requires_source_override_mapping(self):
        with self.assertRaises(KeyError):
            build_source_config({"domain_adaptation": {}})

    def test_can_drop_target_only_keys_from_source_config(self):
        target_config = {
            "ego_selection": {"train": "1", "eval": "1"},
            "remove_ego_object": {"center": [0, 0, 0]},
            "domain_adaptation": {
                "source": {"root_dir": "opv2v/train"},
                "source_drop_keys": [
                    "ego_selection",
                    "remove_ego_object",
                ],
            },
        }

        source_config = build_source_config(target_config)

        self.assertNotIn("ego_selection", source_config)
        self.assertNotIn("remove_ego_object", source_config)

    def test_rejects_invalid_source_drop_key(self):
        target_config = {
            "domain_adaptation": {
                "source": {},
                "source_drop_keys": [""],
            }
        }

        with self.assertRaisesRegex(ValueError, "source_drop_keys"):
            build_source_config(target_config)


class ForeverDataIteratorTest(unittest.TestCase):
    def test_rebuilds_iterator_after_exhaustion(self):
        data_loader = _CountingIterable([10, 20])
        iterator = ForeverDataIterator(data_loader)

        self.assertEqual(next(iterator), 10)
        self.assertEqual(next(iterator), 20)
        self.assertEqual(next(iterator), 10)
        self.assertEqual(data_loader.iter_calls, 2)

    def test_empty_loader_propagates_stop_iteration(self):
        data_loader = _CountingIterable([])
        iterator = ForeverDataIterator(data_loader)

        with self.assertRaises(StopIteration):
            next(iterator)
        self.assertEqual(data_loader.iter_calls, 2)


class MergeSourceTargetBatchesTest(unittest.TestCase):
    def test_merges_model_inputs_and_offsets_target_agent_indices(self):
        source = _ego_batch([2, 1], [0, 1, 2], max_agents=3)
        target = _ego_batch([1, 2], [0, 1, 2, 2], max_agents=3)
        original_target_coords = target["processed_lidar"][
            "voxel_coords"
        ].clone()

        merged, source_scene_count, domain_labels = (
            merge_source_target_batches(source, target)
        )

        self.assertEqual(source_scene_count, 2)
        torch.testing.assert_close(
            domain_labels, torch.tensor([0.0, 0.0, 1.0, 1.0])
        )
        self.assertEqual(domain_labels.dtype, torch.float32)
        torch.testing.assert_close(
            merged["record_len"], torch.tensor([2, 1, 1, 2])
        )
        self.assertEqual(
            tuple(merged["pairwise_t_matrix"].shape),
            (4, 3, 3, 4, 4),
        )
        self.assertEqual(tuple(merged["lidar_pose"].shape), (6, 6))
        expected_prior = torch.zeros(4, 3, 3)
        # Source scene 0 also has a local-index-1 collaborator, but OPV2V is
        # vehicle-only. Target scene 0 is single-agent; only target scene 1
        # has a valid roadside agent at local index 1.
        expected_prior[3, 1, 2] = 1.0
        torch.testing.assert_close(
            merged["prior_encoding"], expected_prior
        )

        merged_coords = merged["processed_lidar"]["voxel_coords"]
        torch.testing.assert_close(
            merged_coords[:, 0],
            torch.tensor([0, 1, 2, 3, 4, 5, 5], dtype=torch.int32),
        )
        torch.testing.assert_close(
            target["processed_lidar"]["voxel_coords"],
            original_target_coords,
        )

        self.assertEqual(
            set(merged),
            {
                "processed_lidar",
                "record_len",
                "pairwise_t_matrix",
                "lidar_pose",
                "prior_encoding",
            },
        )
        self.assertEqual(
            set(merged["processed_lidar"]),
            {"voxel_features", "voxel_coords", "voxel_num_points"},
        )
        self.assertNotIn("label_dict", merged)
        self.assertNotIn("anchor_box", merged)
        self.assertNotIn("object_bbx_center", merged)

    def test_prior_builder_keeps_source_collaborators_as_vehicles(self):
        source = _ego_batch([2, 1], [0, 1, 2], max_agents=3)

        prior = build_prior_encoding(source, "source")

        self.assertEqual(prior.shape, (2, 3, 3))
        torch.testing.assert_close(prior, torch.zeros_like(prior))

    def test_prior_builder_marks_only_valid_target_roadside_agents(self):
        target = _ego_batch([1, 2], [0, 1, 2], max_agents=3)

        prior = build_prior_encoding(target, "target")

        expected = torch.zeros(2, 3, 3)
        expected[1, 1, 2] = 1.0
        torch.testing.assert_close(prior, expected)

    def test_prior_builder_keeps_v2v_target_agents_as_vehicles(self):
        target = _ego_batch([1, 2], [0, 1, 2], max_agents=3)

        prior = build_prior_encoding(target, "target", agent_type="v2v")

        torch.testing.assert_close(prior, torch.zeros_like(prior))
        # The local-index-1 slot in the single-agent scene and every padded
        # slot must remain zero.
        self.assertEqual(prior[0, 1:].abs().sum().item(), 0.0)
        self.assertEqual(prior[1, 2].abs().sum().item(), 0.0)

    def test_batch_merge_accepts_v2v_target_policy(self):
        source = _ego_batch([2], [0, 1], max_agents=2)
        target = _ego_batch([2], [0, 1], max_agents=2)

        merged, _, _ = merge_source_target_batches(
            source, target, target_agent_type="v2v"
        )

        torch.testing.assert_close(
            merged["prior_encoding"],
            torch.zeros_like(merged["prior_encoding"]),
        )

    def test_existing_prior_is_validated_instead_of_silently_overridden(self):
        source = _ego_batch([2], [0, 1], max_agents=3)
        source["prior_encoding"] = torch.zeros(1, 3, 3)
        torch.testing.assert_close(
            build_prior_encoding(source, "source"),
            source["prior_encoding"],
        )

        source["prior_encoding"][0, 1, 2] = 1.0
        with self.assertRaisesRegex(ValueError, "canonical Sim2Real"):
            build_prior_encoding(source, "source")
        with self.assertRaisesRegex(ValueError, "canonical Sim2Real"):
            merge_source_target_batches(
                source,
                _ego_batch([2], [0, 1], max_agents=3),
            )

        target = _ego_batch([2], [0, 1], max_agents=3)
        target["prior_encoding"] = torch.zeros(1, 3, 3)
        with self.assertRaisesRegex(ValueError, "canonical Sim2Real"):
            build_prior_encoding(target, "target")
        with self.assertRaisesRegex(ValueError, "canonical Sim2Real"):
            merge_source_target_batches(
                _ego_batch([2], [0, 1], max_agents=3),
                target,
            )

        # A supplied type on a padded slot is rejected even when the target
        # scene itself contains only one valid agent.
        single_target = _ego_batch([1], [0], max_agents=3)
        single_target["prior_encoding"] = torch.zeros(1, 3, 3)
        single_target["prior_encoding"][0, 1, 2] = 1.0
        with self.assertRaisesRegex(ValueError, "canonical Sim2Real"):
            build_prior_encoding(single_target, "target")

    def test_rejects_different_grid_shapes(self):
        source = _ego_batch([1], [0], max_agents=2)
        target = _ego_batch([1], [0], max_agents=2)
        target["anchor_box"] = _anchor_grid(width=4)

        with self.assertRaisesRegex(ValueError, "different shapes"):
            merge_source_target_batches(source, target)

    def test_rejects_same_shape_but_different_physical_grid(self):
        source = _ego_batch([1], [0], max_agents=2)
        target = _ego_batch([1], [0], max_agents=2)
        target["anchor_box"] = target["anchor_box"].clone()
        target["anchor_box"][0, 0, 0, 0] += 0.4

        with self.assertRaisesRegex(ValueError, "anchor grids differ"):
            merge_source_target_batches(source, target)

    def test_rejects_different_pairwise_padding_size(self):
        source = _ego_batch([1], [0], max_agents=2)
        target = _ego_batch([1], [0], max_agents=3)

        with self.assertRaisesRegex(
            ValueError, "pairwise_t_matrix trailing shapes differ"
        ):
            merge_source_target_batches(source, target)

    def test_rejects_voxel_agent_index_outside_record_len(self):
        source = _ego_batch([1], [0], max_agents=2)
        target = _ego_batch([1], [1], max_agents=2)

        with self.assertRaisesRegex(
            ValueError, "voxel agent index exceeds"
        ):
            merge_source_target_batches(source, target)


if __name__ == "__main__":
    unittest.main()
