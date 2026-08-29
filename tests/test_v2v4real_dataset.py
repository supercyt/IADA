import copy
import os
import tempfile
import unittest

import numpy as np

from opencood.data_utils.datasets.basedataset.v2v4real_basedataset import (
    V2V4REALBaseDataset,
    _relative_matrix,
    _transform_local_objects,
)
from opencood.utils.transformation_utils import x1_to_x2, x_to_world


class V2V4RealGeometryTest(unittest.TestCase):
    def test_relative_matrix_matches_official_matrix_convention(self):
        ego = x_to_world([10.0, -3.0, 0.5, 0.0, 20.0, 0.0])
        cav = x_to_world([14.0, 2.0, 0.7, 0.0, 35.0, 0.0])

        relative = _relative_matrix(cav, ego)

        np.testing.assert_allclose(relative, np.linalg.inv(ego) @ cav)

    def test_local_objects_are_moved_to_synthetic_ego_world(self):
        cav_to_ego = x_to_world([5.0, -2.0, 0.0, 0.0, 30.0, 0.0])
        objects = {
            7: {
                "location": [4.0, 1.0, -1.0],
                "angle": [0.0, 15.0, 0.0],
                "center": [0.2, 0.0, 0.5],
                "extent": [2.0, 1.0, 0.8],
            }
        }
        original = copy.deepcopy(objects)

        transformed = _transform_local_objects(objects, cav_to_ego)

        expected = cav_to_ego @ x_to_world(
            [4.2, 1.0, -0.5, 0.0, 15.0, 0.0]
        )
        actual = x_to_world(
            transformed[7]["location"] + transformed[7]["angle"]
        )
        np.testing.assert_allclose(actual, expected, atol=1e-6)
        self.assertEqual(transformed[7]["center"], [0.0, 0.0, 0.0])
        self.assertEqual(objects, original)

    def test_normalized_pose_reproduces_local_object_projection(self):
        cav_to_ego = x_to_world([8.0, 3.0, 0.2, 1.0, -25.0, -0.5])
        objects = {
            "car": {
                "location": [12.0, -4.0, -1.2],
                "angle": [0.0, 40.0, 0.0],
                "center": [0.0, 0.0, 0.0],
                "extent": [2.1, 0.9, 0.8],
            }
        }
        transformed = _transform_local_objects(objects, cav_to_ego)
        world_object = transformed["car"]["location"] + transformed["car"][
            "angle"
        ]
        from opencood.utils.transformation_utils import tfm_to_pose

        recovered_local = x1_to_x2(world_object, tfm_to_pose(cav_to_ego))
        expected_local = x_to_world(
            objects["car"]["location"] + objects["car"]["angle"]
        )
        np.testing.assert_allclose(recovered_local, expected_local, atol=2e-3)


class V2V4RealIndexTest(unittest.TestCase):
    def test_reinitialize_uses_only_timestamps_shared_by_all_cavs(self):
        with tempfile.TemporaryDirectory() as root:
            scenario = os.path.join(root, "scene")
            for cav_id, timestamps in (
                ("0", ("000000", "000001")),
                ("1", ("000000", "000001", "000002")),
            ):
                cav_path = os.path.join(scenario, cav_id)
                os.makedirs(cav_path)
                for timestamp in timestamps:
                    for extension in (".yaml", ".pcd"):
                        path = os.path.join(
                            cav_path, timestamp + extension
                        )
                        with open(path, "w"):
                            pass

            dataset = V2V4REALBaseDataset.__new__(V2V4REALBaseDataset)
            dataset.scenario_folders = [scenario]
            dataset.max_cav = 2
            dataset.train = False
            dataset.root_dir = root
            dataset.params = {}
            dataset.reinitialize()

        self.assertEqual(dataset.len_record, [2])
        for cav_content in dataset.scenario_database[0].values():
            self.assertEqual(
                [key for key in cav_content if key != "ego"],
                ["000000", "000001"],
            )


if __name__ == "__main__":
    unittest.main()
