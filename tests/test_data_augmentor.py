import unittest

import numpy as np

from opencood.data_utils.augmentor.data_augmentor import DataAugmentor
from opencood.data_utils.datasets.intermediate_fusion_dataset import (
    sample_scene_augmentation,
)


class SceneAugmentationTest(unittest.TestCase):
    def setUp(self):
        self.config = [
            {"NAME": "random_world_flip", "ALONG_AXIS_LIST": ["x"]},
            {
                "NAME": "random_world_rotation",
                "WORLD_ROT_ANGLE": [-0.5, 0.5],
            },
            {
                "NAME": "random_world_scaling",
                "WORLD_SCALE_RANGE": [0.95, 1.05],
            },
        ]

    def test_eval_does_not_sample_geometric_augmentation(self):
        self.assertIsNone(sample_scene_augmentation(self.config, False))

    def test_fixed_scene_parameters_transform_all_cavs_identically(self):
        augmentor = DataAugmentor(self.config, train=True)
        parameters = {
            "flip": {"x": True},
            "noise_rotation": 0.25,
            "noise_scale": 1.02,
        }
        first = {
            "lidar_np": np.array([[2.0, 1.0, 0.5, 1.0]]),
            "object_bbx_center": np.array(
                [[2.0, 1.0, 0.5, 4.0, 2.0, 1.5, 0.1]]
            ),
            "object_bbx_mask": np.array([1.0]),
            **parameters,
        }
        second = {
            key: value.copy() if isinstance(value, np.ndarray) else value
            for key, value in first.items()
        }

        first_result = augmentor.forward(first)
        second_result = augmentor.forward(second)

        np.testing.assert_allclose(
            first_result["lidar_np"], second_result["lidar_np"]
        )
        np.testing.assert_allclose(
            first_result["object_bbx_center"],
            second_result["object_bbx_center"],
        )

    def test_without_fixed_parameters_keeps_existing_random_behavior(self):
        augmentor = DataAugmentor(
            [{"NAME": "random_world_scaling",
              "WORLD_SCALE_RANGE": [2.0, 2.1]}],
            train=True,
        )
        data = {
            "lidar_np": np.array([[1.0, 1.0, 1.0, 1.0]]),
            "object_bbx_center": np.array(
                [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]]
            ),
            "object_bbx_mask": np.array([1.0]),
        }

        result = augmentor.forward(data)

        self.assertGreater(result["lidar_np"][0, 0], 2.0)
        self.assertLess(result["lidar_np"][0, 0], 2.1)


if __name__ == "__main__":
    unittest.main()
