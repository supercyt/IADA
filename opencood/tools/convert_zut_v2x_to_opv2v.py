#!/usr/bin/env python3
"""Convert the ZUT real-world cooperative ROS 2 bags to OPV2V layout.

The cooperative frame clock is the timestamp in the forward LiDAR
``PointCloud2.header``.  GPS and IMU values are deliberately decoded only from
``/fusion_data``.  The SQLite receive timestamp is used only for same-vehicle
association of the two side LiDAR topics because those derived messages have
a zero-valued header stamp.  It is compared with the forward LiDAR SQLite
receive timestamp, not with the forward LiDAR sensor/header timestamp.

The bags contain no human-labelled 3-D boxes.  Consequently ``vehicles`` is
left empty in the OPV2V metadata.  Detector/grid output in ``fusion_data`` is
not exported as annotation metadata.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import sqlite3
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Sequence

import numpy as np
import yaml


FORWARD_TOPIC = "/forward/rslidar_points"
LEFT_TOPIC = "/left_point_after_trans"
RIGHT_TOPIC = "/right_point_after_trans"
FUSION_TOPIC = "/fusion_data"

DEFAULT_INPUT = Path("/home/caoyitong/DataProjects/v2x_datasets/zut_v2x_real")
EXPERIMENTS = {
    "experiment_1": {
        "front": Path("coop1/bag8.17/sub_0.db3"),
        "rear": Path("coop1/bag11/bag11_0.db3"),
    },
    "experiment_2": {
        "front": Path("coop2/bag8.17_1/sub_0.db3"),
        "rear": Path("coop2/bag22/bag22_0.db3"),
    },
}


@dataclass(frozen=True)
class MessageRef:
    timestamp: float
    message_id: int
    record_timestamp: float


@dataclass(frozen=True)
class MatchedPair:
    front_index: int
    rear_index: int
    delta_seconds: float


@dataclass
class FusionData:
    timestamp: float
    id: int
    carlength: float
    carwidth: float
    carheight: float
    centeroffset: float
    signnumber: int
    signdata: list[float]
    lightdata: list[float]
    obstacledata: list[float]
    yaw: float
    pitch: float
    roll: float
    wx: float
    wy: float
    wz: float
    ax: float
    ay: float
    az: float
    longitude: float
    latitude: float
    height: float
    eastvelocity: float
    northvelocity: float
    skyvelocity: float
    carspeed: float
    steerangle: float
    gearpos: int
    braketq: float
    parkingstate: int
    soc: int
    batteryvol: int
    batterydischargecur: int
    car_run_mode: int
    throttle_percentage: int
    braking_percentage: int
    left_light: bool
    right_light: bool
    reversing_light: bool
    speaker: bool
    start_button: bool
    stop_button: bool
    state: int
    error: int
    process_time: float

    @property
    def obstacles(self) -> list[list[float]]:
        return _groups(self.obstacledata, 9)

    @property
    def signs(self) -> list[list[float]]:
        return _groups(self.signdata, 9)

    @property
    def lights(self) -> list[list[float]]:
        return _groups(self.lightdata, 6)


class CdrReader:
    """Small little-endian CDR1 reader for the two message types in the bags."""

    def __init__(self, serialized: bytes | memoryview):
        view = memoryview(serialized)
        if len(view) < 4 or bytes(view[:2]) != b"\x00\x01":
            raise ValueError("only little-endian CDR1 messages are supported")
        self.data = view[4:]
        self.pos = 0

    def _align(self, size: int) -> None:
        self.pos = (self.pos + size - 1) // size * size

    def unpack(self, fmt: str, size: int) -> Any:
        self._align(size)
        value = struct.unpack_from("<" + fmt, self.data, self.pos)[0]
        self.pos += size
        return value

    def uint8(self) -> int:
        return int(self.unpack("B", 1))

    def int8(self) -> int:
        return int(self.unpack("b", 1))

    def uint16(self) -> int:
        return int(self.unpack("H", 2))

    def uint32(self) -> int:
        return int(self.unpack("I", 4))

    def int32(self) -> int:
        return int(self.unpack("i", 4))

    def float32(self) -> float:
        return float(self.unpack("f", 4))

    def float64(self) -> float:
        return float(self.unpack("d", 8))

    def string(self) -> str:
        size = self.uint32()
        if size == 0:
            return ""
        value = bytes(self.data[self.pos : self.pos + size - 1]).decode(
            "utf-8", errors="replace"
        )
        self.pos += size
        return value

    def float32_sequence(self) -> list[float]:
        size = self.uint32()
        if size == 0:
            return []
        self._align(4)
        values = struct.unpack_from(f"<{size}f", self.data, self.pos)
        self.pos += size * 4
        return [float(value) for value in values]

    def uint8_sequence_view(self) -> memoryview:
        size = self.uint32()
        value = self.data[self.pos : self.pos + size]
        self.pos += size
        return value


def parse_fusion(serialized: bytes | memoryview) -> FusionData:
    reader = CdrReader(serialized)
    timestamp = reader.float64()
    vehicle_id = reader.uint8()
    dimensions = [reader.float32() for _ in range(4)]
    signnumber = reader.uint8()
    signdata = reader.float32_sequence()
    lightdata = reader.float32_sequence()
    obstacledata = reader.float32_sequence()
    attitude_and_imu = [reader.float32() for _ in range(9)]
    geodetic = [reader.float64() for _ in range(3)]
    velocity_and_control = [reader.float32() for _ in range(5)]
    gearpos = reader.int8()
    braketq = reader.float32()
    parkingstate = reader.uint8()
    soc = reader.uint8()
    batteryvol = reader.uint8()
    batterydischargecur = reader.uint16()
    discrete = [reader.uint8() for _ in range(11)]
    process_time = reader.float32()

    if reader.pos != len(reader.data):
        raise ValueError(
            f"FusionInterface decoder stopped at {reader.pos}, "
            f"message has {len(reader.data)} payload bytes"
        )
    for field_name, values, group_size in (
        ("signdata", signdata, 9),
        ("lightdata", lightdata, 6),
        ("obstacledata", obstacledata, 9),
    ):
        if len(values) % group_size:
            raise ValueError(
                f"{field_name} length {len(values)} is not divisible by {group_size}"
            )

    return FusionData(
        timestamp=timestamp,
        id=vehicle_id,
        carlength=dimensions[0],
        carwidth=dimensions[1],
        carheight=dimensions[2],
        centeroffset=dimensions[3],
        signnumber=signnumber,
        signdata=signdata,
        lightdata=lightdata,
        obstacledata=obstacledata,
        yaw=attitude_and_imu[0],
        pitch=attitude_and_imu[1],
        roll=attitude_and_imu[2],
        wx=attitude_and_imu[3],
        wy=attitude_and_imu[4],
        wz=attitude_and_imu[5],
        ax=attitude_and_imu[6],
        ay=attitude_and_imu[7],
        az=attitude_and_imu[8],
        longitude=geodetic[0],
        latitude=geodetic[1],
        height=geodetic[2],
        eastvelocity=velocity_and_control[0],
        northvelocity=velocity_and_control[1],
        skyvelocity=velocity_and_control[2],
        carspeed=velocity_and_control[3],
        steerangle=velocity_and_control[4],
        gearpos=gearpos,
        braketq=braketq,
        parkingstate=parkingstate,
        soc=soc,
        batteryvol=batteryvol,
        batterydischargecur=batterydischargecur,
        car_run_mode=discrete[0],
        throttle_percentage=discrete[1],
        braking_percentage=discrete[2],
        left_light=bool(discrete[3]),
        right_light=bool(discrete[4]),
        reversing_light=bool(discrete[5]),
        speaker=bool(discrete[6]),
        start_button=bool(discrete[7]),
        stop_button=bool(discrete[8]),
        state=discrete[9],
        error=discrete[10],
        process_time=process_time,
    )


def pointcloud_header_timestamp(serialized_prefix: bytes | memoryview) -> float:
    if len(serialized_prefix) < 12:
        raise ValueError("PointCloud2 prefix is too short")
    seconds, nanoseconds = struct.unpack_from("<iI", serialized_prefix, 4)
    return seconds + nanoseconds / 1_000_000_000.0


def parse_pointcloud2(serialized: bytes | memoryview) -> np.ndarray:
    """Decode x/y/z/intensity from sensor_msgs/msg/PointCloud2."""
    reader = CdrReader(serialized)
    reader.int32()  # header.stamp.sec
    reader.uint32()  # header.stamp.nanosec
    reader.string()  # header.frame_id
    height = reader.uint32()
    width = reader.uint32()
    field_count = reader.uint32()
    fields: dict[str, tuple[int, int, int]] = {}
    for _ in range(field_count):
        name = reader.string()
        offset = reader.uint32()
        datatype = reader.uint8()
        count = reader.uint32()
        fields[name] = (offset, datatype, count)
    is_bigendian = bool(reader.uint8())
    point_step = reader.uint32()
    row_step = reader.uint32()
    raw = reader.uint8_sequence_view()
    reader.uint8()  # is_dense

    if is_bigendian:
        raise ValueError("big-endian PointCloud2 is not supported")
    required = ("x", "y", "z")
    if any(name not in fields for name in required):
        raise ValueError(f"PointCloud2 is missing one of {required}: {sorted(fields)}")
    if any(fields[name][1:] != (7, 1) for name in required):
        raise ValueError("x/y/z must be scalar FLOAT32 PointCloud2 fields")
    count = height * width
    if len(raw) < count * point_step or row_step < width * point_step:
        raise ValueError("PointCloud2 data/row size is inconsistent")

    names = ["x", "y", "z"]
    if "intensity" in fields and fields["intensity"][1:] == (7, 1):
        names.append("intensity")
    dtype = np.dtype(
        {
            "names": names,
            "formats": ["<f4"] * len(names),
            "offsets": [fields[name][0] for name in names],
            "itemsize": point_step,
        }
    )
    # These bags have tightly packed rows.  Keep an explicit check rather
    # than silently misreading a future bag with row padding.
    if row_step != width * point_step:
        raise ValueError("PointCloud2 rows with padding are not supported")
    values = np.frombuffer(raw, dtype=dtype, count=count)
    points = np.empty((count, 4), dtype=np.float32)
    for column, name in enumerate(("x", "y", "z")):
        points[:, column] = values[name]
    points[:, 3] = values["intensity"] if "intensity" in names else 0.0
    return points[np.isfinite(points[:, :3]).all(axis=1)]


class Bag:
    def __init__(self, path: Path):
        self.path = path
        uri = f"file:{path}?mode=ro"
        self.connection = sqlite3.connect(uri, uri=True)
        self.topic_ids = {
            name: topic_id
            for topic_id, name in self.connection.execute("SELECT id, name FROM topics")
        }
        missing = {FORWARD_TOPIC, LEFT_TOPIC, RIGHT_TOPIC, FUSION_TOPIC} - set(
            self.topic_ids
        )
        if missing:
            raise ValueError(f"{path} is missing topics: {sorted(missing)}")
        self._fusion_by_id: dict[int, FusionData] = {}

    def close(self) -> None:
        self.connection.close()

    def forward_refs(self) -> list[MessageRef]:
        topic_id = self.topic_ids[FORWARD_TOPIC]
        rows = self.connection.execute(
            "SELECT id, timestamp, substr(data, 1, 12) "
            "FROM messages WHERE topic_id = ? ORDER BY timestamp",
            (topic_id,),
        )
        refs = []
        for message_id, record_ns, prefix in rows:
            timestamp = pointcloud_header_timestamp(prefix)
            if timestamp <= 1_000_000_000:
                raise ValueError(f"invalid forward LiDAR header stamp in message {message_id}")
            refs.append(MessageRef(timestamp, message_id, record_ns / 1e9))
        return sorted(refs, key=lambda item: item.timestamp)

    def record_time_refs(self, topic: str) -> list[MessageRef]:
        topic_id = self.topic_ids[topic]
        rows = self.connection.execute(
            "SELECT id, timestamp FROM messages WHERE topic_id = ? ORDER BY timestamp",
            (topic_id,),
        )
        return [MessageRef(ts / 1e9, mid, ts / 1e9) for mid, ts in rows]

    def fusion_refs(self) -> list[MessageRef]:
        topic_id = self.topic_ids[FUSION_TOPIC]
        rows = self.connection.execute(
            "SELECT id, timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp",
            (topic_id,),
        )
        refs = []
        for message_id, record_ns, serialized in rows:
            fusion = parse_fusion(serialized)
            self._fusion_by_id[message_id] = fusion
            refs.append(MessageRef(fusion.timestamp, message_id, record_ns / 1e9))
        return sorted(refs, key=lambda item: item.timestamp)

    def fusion(self, message_id: int) -> FusionData:
        return self._fusion_by_id[message_id]

    def pointcloud(self, message_id: int) -> np.ndarray:
        row = self.connection.execute(
            "SELECT data FROM messages WHERE id = ?", (message_id,)
        ).fetchone()
        if row is None:
            raise KeyError(message_id)
        return parse_pointcloud2(row[0])


def pair_monotonic_nearest(
    front: Sequence[MessageRef], rear: Sequence[MessageRef], max_delta_seconds: float
) -> list[MatchedPair]:
    """One-to-one nearest matching under a tight, monotonic sensor clock."""
    rear_times = [item.timestamp for item in rear]
    candidates: list[tuple[float, int, int]] = []
    for front_index, front_ref in enumerate(front):
        insertion = bisect.bisect_left(rear_times, front_ref.timestamp)
        for rear_index in (insertion - 1, insertion):
            if 0 <= rear_index < len(rear):
                delta = abs(front_ref.timestamp - rear[rear_index].timestamp)
                if delta < max_delta_seconds:  # user requested strict '< 10 ms'
                    candidates.append((delta, front_index, rear_index))

    used_front: set[int] = set()
    used_rear: set[int] = set()
    matches = []
    for delta, front_index, rear_index in sorted(candidates):
        if front_index in used_front or rear_index in used_rear:
            continue
        used_front.add(front_index)
        used_rear.add(rear_index)
        matches.append(MatchedPair(front_index, rear_index, delta))
    return sorted(matches, key=lambda match: front[match.front_index].timestamp)


def nearest_ref(timestamp: float, refs: Sequence[MessageRef]) -> tuple[MessageRef, float]:
    times = [item.timestamp for item in refs]
    insertion = bisect.bisect_left(times, timestamp)
    indices = [index for index in (insertion - 1, insertion) if 0 <= index < len(refs)]
    if not indices:
        raise ValueError("cannot associate against an empty message stream")
    index = min(indices, key=lambda candidate: abs(timestamp - times[candidate]))
    return refs[index], abs(timestamp - times[index])


def _groups(values: Sequence[float], size: int) -> list[list[float]]:
    return [list(values[index : index + size]) for index in range(0, len(values), size)]


def _percentiles_ms(values_seconds: Sequence[float]) -> dict[str, float | int]:
    if not values_seconds:
        return {"count": 0}
    values = sorted(value * 1000.0 for value in values_seconds)

    def at(fraction: float) -> float:
        index = min(len(values) - 1, int((len(values) - 1) * fraction))
        return round(values[index], 6)

    return {
        "count": len(values),
        "min_ms": round(values[0], 6),
        "median_ms": round(float(median(values)), 6),
        "p95_ms": at(0.95),
        "max_ms": round(values[-1], 6),
    }


def _valid_geodetic(fusion: FusionData) -> bool:
    return (
        math.isfinite(fusion.longitude)
        and math.isfinite(fusion.latitude)
        and math.isfinite(fusion.height)
        and -180 <= fusion.longitude <= 180
        and -90 <= fusion.latitude <= 90
        and abs(fusion.longitude) > 1e-8
        and abs(fusion.latitude) > 1e-8
    )


def geodetic_to_ecef(longitude: float, latitude: float, height: float) -> np.ndarray:
    semi_major = 6378137.0
    eccentricity_sq = 6.69437999014e-3
    lon = math.radians(longitude)
    lat = math.radians(latitude)
    prime_vertical = semi_major / math.sqrt(1 - eccentricity_sq * math.sin(lat) ** 2)
    return np.array(
        [
            (prime_vertical + height) * math.cos(lat) * math.cos(lon),
            (prime_vertical + height) * math.cos(lat) * math.sin(lon),
            (prime_vertical * (1 - eccentricity_sq) + height) * math.sin(lat),
        ],
        dtype=np.float64,
    )


def geodetic_to_enu(
    longitude: float,
    latitude: float,
    height: float,
    origin: tuple[float, float, float],
) -> list[float]:
    origin_lon, origin_lat, origin_height = origin
    delta = geodetic_to_ecef(longitude, latitude, height) - geodetic_to_ecef(
        origin_lon, origin_lat, origin_height
    )
    lon = math.radians(origin_lon)
    lat = math.radians(origin_lat)
    rotation = np.array(
        [
            [-math.sin(lon), math.cos(lon), 0],
            [
                -math.sin(lat) * math.cos(lon),
                -math.sin(lat) * math.sin(lon),
                math.cos(lat),
            ],
            [
                math.cos(lat) * math.cos(lon),
                math.cos(lat) * math.sin(lon),
                math.sin(lat),
            ],
        ]
    )
    return [float(value) for value in rotation @ delta]


def fusion_pose(fusion: FusionData, origin: tuple[float, float, float]) -> list[float]:
    east, north, up = geodetic_to_enu(
        fusion.longitude, fusion.latitude, fusion.height, origin
    )
    # fusion_data yaw is already conventional ENU yaw: counter-clockwise from
    # the positive world x/east axis.  This is independently confirmed by
    # atan2(north_velocity, east_velocity), with about 0.1 degree median error
    # in the healthy vehicle streams.  Applying ``90 - yaw`` rotates the
    # inter-CAV baseline by roughly 90 degrees.
    yaw = (fusion.yaw + 180.0) % 360.0 - 180.0
    return [east, north, up, fusion.roll, yaw, fusion.pitch]


def write_binary_pcd(path: Path, points: np.ndarray) -> None:
    """Write an Open3D/OpenCOOD-compatible binary x/y/z/rgb PCD."""
    points = np.asarray(points, dtype=np.float32)
    intensity = np.nan_to_num(points[:, 3], nan=0.0, posinf=255.0, neginf=0.0)
    # RoboSense intensity is normally 0..255.  Preserve already-normalized
    # inputs by scaling only when the observed range indicates byte values.
    if intensity.size and float(np.nanmax(intensity)) <= 1.0:
        intensity = intensity * 255.0
    grey = np.clip(np.rint(intensity), 0, 255).astype(np.uint32)
    packed_rgb = (grey << 16) | (grey << 8) | grey
    output = np.empty(points.shape[0], dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<f4")])
    output["x"] = points[:, 0]
    output["y"] = points[:, 1]
    output["z"] = points[:, 2]
    output["rgb"] = packed_rgb.view(np.float32)
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z rgb\n"
        "SIZE 4 4 4 4\n"
        "TYPE F F F F\n"
        "COUNT 1 1 1 1\n"
        f"WIDTH {len(output)}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {len(output)}\n"
        "DATA binary\n"
    ).encode("ascii")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as file:
        file.write(header)
        file.write(output.tobytes())
    os.replace(temporary, path)


def write_yaml(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        yaml.safe_dump(value, file, allow_unicode=True, sort_keys=False)
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")
    os.replace(temporary, path)


def fusion_metadata(fusion: FusionData, pose: list[float]) -> dict[str, Any]:
    """Return only fields consumed by the standard OPV2V loader.

    The raw fusion detector output is not ground truth and must not be mixed
    into an OPV2V annotation YAML.  It is therefore intentionally omitted.
    """
    return {
        "lidar_pose": pose,
        "lidar_pose_clean": pose,
        "true_ego_pos": pose,
        "ego_speed": fusion.carspeed * 3.6,
        "vehicles": {},
    }


def inspect_experiment(
    input_root: Path,
    experiment_name: str,
    paths: dict[str, Path],
    sync_seconds: float,
    association_seconds: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    bags = {role: Bag(input_root / relative) for role, relative in paths.items()}
    try:
        streams: dict[str, dict[str, list[MessageRef]]] = {}
        for role, bag in bags.items():
            streams[role] = {
                "forward": bag.forward_refs(),
                "fusion": bag.fusion_refs(),
                "left": bag.record_time_refs(LEFT_TOPIC),
                "right": bag.record_time_refs(RIGHT_TOPIC),
            }
        matches = pair_monotonic_nearest(
            streams["front"]["forward"], streams["rear"]["forward"], sync_seconds
        )

        frame_records = []
        association_deltas: dict[str, dict[str, list[float]]] = {
            role: {name: [] for name in ("fusion", "left", "right")}
            for role in ("front", "rear")
        }
        rejected_no_fusion = 0
        side_coverage = {role: {"left": 0, "right": 0} for role in ("front", "rear")}
        for match in matches:
            refs = {
                "front": streams["front"]["forward"][match.front_index],
                "rear": streams["rear"]["forward"][match.rear_index],
            }
            associated: dict[str, dict[str, tuple[MessageRef, float] | None]] = {}
            fusion_is_valid = True
            for role in ("front", "rear"):
                associated[role] = {}
                for name in ("fusion", "left", "right"):
                    # fusion_data and the cooperative clock use sensor/header
                    # time.  Derived side clouds have zero header stamps, so
                    # compare their receive times with the forward cloud's
                    # receive time from the same rosbag.
                    association_clock = (
                        refs[role].timestamp
                        if name == "fusion"
                        else refs[role].record_timestamp
                    )
                    item = nearest_ref(association_clock, streams[role][name])
                    association_deltas[role][name].append(item[1])
                    if item[1] <= association_seconds:
                        associated[role][name] = item
                        if name in ("left", "right"):
                            side_coverage[role][name] += 1
                    else:
                        associated[role][name] = None
                        if name == "fusion":
                            fusion_is_valid = False
            if not fusion_is_valid:
                rejected_no_fusion += 1
                continue
            frame_records.append(
                {
                    "match": match,
                    "forward": refs,
                    "associated": associated,
                }
            )

        geodetic = []
        for record in frame_records:
            for role in ("front", "rear"):
                fusion_ref = record["associated"][role]["fusion"][0]
                fusion = bags[role].fusion(fusion_ref.message_id)
                if _valid_geodetic(fusion):
                    geodetic.append((fusion.longitude, fusion.latitude, fusion.height))
        if not geodetic:
            raise ValueError(f"{experiment_name} has no valid GPS in associated fusion_data")
        # A median origin is robust to a bad first GNSS fix and is shared by both CAVs.
        origin = tuple(float(np.median(np.asarray(geodetic), axis=0)[index]) for index in range(3))

        pose_health = {}
        for role in ("front", "rear"):
            associated_fusion = [
                bags[role].fusion(record["associated"][role]["fusion"][0].message_id)
                for record in frame_records
            ]
            unique_positions = len(
                {
                    (item.longitude, item.latitude, item.height)
                    for item in associated_fusion
                }
            )
            unique_attitudes = len(
                {(item.yaw, item.pitch, item.roll) for item in associated_fusion}
            )
            max_speed = max((abs(item.carspeed) for item in associated_fusion), default=0.0)
            pose_health[role] = {
                "unique_geodetic_positions": unique_positions,
                "unique_attitudes": unique_attitudes,
                "max_speed_mps": float(max_speed),
                "frozen_while_moving": bool(
                    max_speed > 1.0
                    and unique_positions == 1
                    and unique_attitudes == 1
                ),
            }

        report = {
            "experiment": experiment_name,
            "source_bags": {role: str(bag.path) for role, bag in bags.items()},
            "stream_counts": {
                role: {name: len(values) for name, values in role_streams.items()}
                for role, role_streams in streams.items()
            },
            "sync_rule": f"abs(front_lidar_header - rear_lidar_header) < {sync_seconds * 1000:g} ms",
            "matched_pairs": len(matches),
            "usable_pairs": len(frame_records),
            "rejected_pairs_without_nearby_fusion": rejected_no_fusion,
            "sync_delta": _percentiles_ms([match.delta_seconds for match in matches]),
            "association_delta": {
                role: {
                    name: _percentiles_ms(values)
                    for name, values in per_stream.items()
                }
                for role, per_stream in association_deltas.items()
            },
            "side_lidar_coverage_within_threshold": side_coverage,
            "association_threshold_ms": association_seconds * 1000.0,
            "fusion_pose_health": pose_health,
            "enu_origin": {
                "longitude": origin[0],
                "latitude": origin[1],
                "height": origin[2],
            },
            "notes": [
                "GPS and IMU are decoded exclusively from /fusion_data.",
                "Left/right transformed PointCloud2 headers are zero, so their SQLite receive times are compared with the same vehicle's forward-LiDAR SQLite receive time.",
                "The bags contain detector/grid output but no human-labelled 3-D boxes; OPV2V vehicles is intentionally empty.",
            ],
        }
        frozen_roles = [
            role for role, health in pose_health.items() if health["frozen_while_moving"]
        ]
        if frozen_roles:
            report["notes"].append(
                "WARNING: fusion_data pose is frozen while the vehicle moves for: "
                + ", ".join(frozen_roles)
                + "; do not perform cross-CAV point-cloud fusion without recovering that pose."
            )
        state = {
            "bags": bags,
            "streams": streams,
            "frames": frame_records,
            "origin": origin,
        }
        return report, state
    except Exception:
        for bag in bags.values():
            bag.close()
        raise


def export_experiment(
    state: dict[str, Any],
    report: dict[str, Any],
    scenario_dir: Path,
) -> list[dict[str, Any]]:
    if scenario_dir.exists() and any(scenario_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty scenario: {scenario_dir}")
    cav_dirs = {"front": scenario_dir / "0", "rear": scenario_dir / "1"}
    for directory in cav_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    bags: dict[str, Bag] = state["bags"]
    manifest_frames = []
    try:
        for frame_index, record in enumerate(state["frames"]):
            stem = f"{frame_index:06d}"
            manifest_frame: dict[str, Any] = {
                "frame": stem,
                "sync_delta_ms": record["match"].delta_seconds * 1000.0,
                "cavs": {},
            }
            for role in ("front", "rear"):
                forward_ref: MessageRef = record["forward"][role]
                associations = record["associated"][role]
                fusion_ref: MessageRef = associations["fusion"][0]
                fusion = bags[role].fusion(fusion_ref.message_id)
                if not _valid_geodetic(fusion):
                    raise ValueError(f"invalid GPS in fusion message {fusion_ref.message_id}")
                pose = fusion_pose(fusion, state["origin"])

                pointclouds = [bags[role].pointcloud(forward_ref.message_id)]
                used_topics = [FORWARD_TOPIC]
                for name, topic in (("left", LEFT_TOPIC), ("right", RIGHT_TOPIC)):
                    item = associations[name]
                    if item is not None:
                        pointclouds.append(bags[role].pointcloud(item[0].message_id))
                        used_topics.append(topic)
                merged = np.concatenate(pointclouds, axis=0)
                write_binary_pcd(cav_dirs[role] / f"{stem}.pcd", merged)
                metadata = fusion_metadata(fusion, pose)
                write_yaml(cav_dirs[role] / f"{stem}.yaml", metadata)
                manifest_frame["cavs"][role] = {
                    "cav_id": 0 if role == "front" else 1,
                    "frame_timestamp": forward_ref.timestamp,
                    "fusion_timestamp": fusion.timestamp,
                    "point_count": int(len(merged)),
                    "included_lidar_topics": used_topics,
                }
            manifest_frames.append(manifest_frame)
            if (frame_index + 1) % 25 == 0 or frame_index + 1 == len(state["frames"]):
                print(
                    f"  {report['experiment']}: {frame_index + 1}/{len(state['frames'])} frames",
                    flush=True,
                )

        protocol = {
            "dataset": "ZUT V2X real",
            "format": "OPV2V-compatible",
            "scenario": report["experiment"],
            "cavs": {0: "front", 1: "rear"},
            "sync_rule": report["sync_rule"],
            "enu_origin": report["enu_origin"],
            "labels": "unlabelled; vehicles is empty and detector output is not exported",
        }
        write_yaml(scenario_dir / "data_protocol.yaml", protocol)
        return manifest_frames
    finally:
        for bag in bags.values():
            bag.close()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="dataset root to create (default: INPUT_ROOT/opv2v)",
    )
    parser.add_argument("--split", default="train", help="OPV2V split directory")
    parser.add_argument(
        "--sync-ms",
        type=float,
        default=10.0,
        help="strict cross-vehicle forward-LiDAR threshold in milliseconds",
    )
    parser.add_argument(
        "--max-association-ms",
        type=float,
        default=100.0,
        help="nearest same-vehicle fusion/side-LiDAR association threshold",
    )
    parser.add_argument(
        "--experiment",
        action="append",
        choices=sorted(EXPERIMENTS),
        help="convert only this experiment (repeatable)",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="print the extraction report without writing point clouds/metadata",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="optional path for the JSON analysis report",
    )
    args = parser.parse_args(argv)
    if args.sync_ms <= 0 or args.max_association_ms <= 0:
        parser.error("time thresholds must be positive")
    if args.output_root is None:
        args.output_root = args.input_root / "opv2v"
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    selected = args.experiment or list(EXPERIMENTS)
    reports = []
    states = {}
    for name in selected:
        print(f"Inspecting {name} ...", flush=True)
        report, state = inspect_experiment(
            args.input_root,
            name,
            EXPERIMENTS[name],
            args.sync_ms / 1000.0,
            args.max_association_ms / 1000.0,
        )
        reports.append(report)
        states[name] = state
        print(
            f"  matched={report['matched_pairs']}, usable={report['usable_pairs']}, "
            f"sync median={report['sync_delta'].get('median_ms', float('nan')):.3f} ms",
            flush=True,
        )

    report_document = {
        "input_root": str(args.input_root),
        "experiments": reports,
    }
    if args.analyze_only:
        print(json.dumps(report_document, ensure_ascii=False, indent=2))
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            write_json(args.report, report_document)
        for state in states.values():
            for bag in state["bags"].values():
                bag.close()
        return 0

    split_dir = args.output_root / args.split
    split_dir.mkdir(parents=True, exist_ok=True)
    manifests = {}
    for report in reports:
        name = report["experiment"]
        print(f"Exporting {name} ...", flush=True)
        manifests[name] = export_experiment(states[name], report, split_dir / name)
    report_document["output_root"] = str(args.output_root)
    report_document["split"] = args.split
    report_document["frames"] = manifests
    write_json(args.output_root / "manifest.json", report_document)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.report, report_document)
    print(f"Done: {args.output_root}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
