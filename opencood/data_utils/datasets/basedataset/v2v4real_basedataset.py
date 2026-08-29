"""V2V4Real adapter for the current OpenCOOD dataset contract.

V2V4Real stores each LiDAR pose as a 4x4 matrix and its object annotations in
the local CAV frame.  The modern fusion datasets in this repository expect a
six-degree-of-freedom pose and objects in a shared world frame.  This adapter
keeps the raw matrices for geometry, then represents every frame in a synthetic
world frame whose origin is the selected ego LiDAR.
"""

import copy
import os
from collections import OrderedDict

import numpy as np

from opencood.data_utils.datasets.basedataset.opv2v_basedataset import (
    OPV2VBaseDataset,
)
from opencood.utils.transformation_utils import tfm_to_pose, x_to_world


def _relative_matrix(cav_to_world, ego_to_world):
    """Match the matrix-to-matrix convention used by official V2V4Real."""

    cav_to_world = np.asarray(cav_to_world, dtype=np.float64)
    ego_to_world = np.asarray(ego_to_world, dtype=np.float64)
    if cav_to_world.shape != (4, 4) or ego_to_world.shape != (4, 4):
        raise ValueError("V2V4Real lidar_pose must have shape (4, 4)")
    return np.linalg.inv(ego_to_world) @ cav_to_world


def _transform_local_objects(objects, cav_to_ego):
    """Move local V2V4Real boxes into the synthetic ego-world frame."""

    transformed = {}
    for object_id, object_content in objects.items():
        object_copy = copy.deepcopy(object_content)
        location = object_copy["location"]
        angle = object_copy["angle"]
        center = object_copy.get("center", [0.0, 0.0, 0.0])
        local_pose = [
            location[0] + center[0],
            location[1] + center[1],
            location[2] + center[2],
            angle[0],
            angle[1],
            angle[2],
        ]
        object_to_ego = cav_to_ego @ x_to_world(local_pose)
        ego_pose = tfm_to_pose(object_to_ego)
        object_copy["location"] = ego_pose[:3]
        object_copy["angle"] = ego_pose[3:]
        object_copy["center"] = [0.0, 0.0, 0.0]
        transformed[object_id] = object_copy
    return transformed


class V2V4REALBaseDataset(OPV2VBaseDataset):
    """Read V2V4Real while exposing the OPV2V-style fusion interface."""

    def reinitialize(self):
        """Index only timestamps shared by every CAV in a scenario."""

        self.scenario_database = OrderedDict()
        self.len_record = []
        for scenario_folder in self.scenario_folders:
            cav_list = sorted(
                name
                for name in os.listdir(scenario_folder)
                if os.path.isdir(os.path.join(scenario_folder, name))
            )
            if not cav_list:
                continue
            if self.train:
                # Preserve OpenCOOD's epoch-wise random ego selection without
                # using target annotations.
                np.random.shuffle(cav_list)
            cav_list = cav_list[: self.max_cav]

            timestamp_sets = []
            for cav_id in cav_list:
                cav_path = os.path.join(scenario_folder, cav_id)
                timestamp_sets.append(
                    {
                        name[:-5]
                        for name in os.listdir(cav_path)
                        if name.endswith(".yaml")
                        and "additional" not in name
                        and "camera_gt" not in name
                        and os.path.isfile(
                            os.path.join(cav_path, name[:-5] + ".pcd")
                        )
                    }
                )
            timestamps = sorted(set.intersection(*timestamp_sets))
            if not timestamps:
                continue

            scenario_content = OrderedDict()
            for cav_index, cav_id in enumerate(cav_list):
                cav_path = os.path.join(scenario_folder, cav_id)
                cav_content = OrderedDict()
                for timestamp in timestamps:
                    cav_content[timestamp] = OrderedDict(
                        yaml=os.path.join(cav_path, timestamp + ".yaml"),
                        lidar=os.path.join(cav_path, timestamp + ".pcd"),
                        cameras=self.find_camera_files(cav_path, timestamp),
                        depths=self.find_camera_files(
                            cav_path, timestamp, sensor="depth"
                        ),
                    )
                cav_content["ego"] = cav_index == 0
                scenario_content[cav_id] = cav_content

            scenario_index = len(self.scenario_database)
            self.scenario_database[scenario_index] = scenario_content
            previous = self.len_record[-1] if self.len_record else 0
            self.len_record.append(previous + len(timestamps))

        if not self.len_record:
            raise RuntimeError(
                f"No synchronized V2V4Real frames found in {self.root_dir}"
            )

    def retrieve_base_data(self, idx):
        data = super().retrieve_base_data(idx)
        ego_content = next(
            content for content in data.values() if content["ego"]
        )
        ego_raw_pose = np.asarray(
            ego_content["params"]["lidar_pose"], dtype=np.float64
        )

        for cav_content in data.values():
            params = cav_content["params"]
            raw_pose = np.asarray(params["lidar_pose"], dtype=np.float64)
            cav_to_ego = _relative_matrix(raw_pose, ego_raw_pose)
            params["v2v4real_lidar_pose_matrix"] = raw_pose
            params["v2v4real_cav_to_ego"] = cav_to_ego
            params["vehicles"] = _transform_local_objects(
                params.get("vehicles", {}), cav_to_ego
            )
            # The generic fusion path can now compute distances, pairwise
            # transforms, clean/noisy poses, and single-view labels normally.
            params["lidar_pose"] = tfm_to_pose(cav_to_ego)
            params["true_ego_pos"] = list(params["lidar_pose"])
        return data


__all__ = [
    "V2V4REALBaseDataset",
    "_relative_matrix",
    "_transform_local_objects",
]
