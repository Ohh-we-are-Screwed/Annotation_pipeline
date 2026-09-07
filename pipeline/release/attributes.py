"""Velocity-derived attributes (spec §4). Never emits parked / sitting / without_rider."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from pipeline.release.config import AttributeConfig

VEHICLE_CLASSES = ("car", "bus", "truck", "covered_van", "microbus", "cng_autorickshaw",
                   "battery_rickshaw", "tempo", "human_hauler", "pushcart")
PEDESTRIAN_CLASSES = ("pedestrian",)
CYCLE_CLASSES = ("bicycle", "motorcycle", "cycle_rickshaw")
NO_ATTRIBUTE_CLASSES = ("traffic_cone", "barrier", "construction_element", "animal")
ATTRIBUTE_NAMES = ("vehicle.moving", "vehicle.stopped", "vehicle.parked", "pedestrian.moving",
                   "pedestrian.standing", "pedestrian.sitting_lying_down", "cycle.with_rider",
                   "cycle.without_rider")
BASIS_HUMAN = "human"
BASIS_CHAIN = "chain_velocity"


def chain_velocities(rows: list, global_center_of: dict, cfg: AttributeConfig) -> dict:
    by_inst: dict = defaultdict(list)
    for r in rows:
        by_inst[r["instance_token"]].append(r)
    out: dict = {}
    max_dt = cfg.max_time_diff_s
    for members in by_inst.values():
        members.sort(key=lambda r: r["t_ns"])
        n = len(members)
        for i, r in enumerate(members):
            if n == 1:
                out[r["token"]] = None
                continue
            j0, j1 = max(i - 1, 0), min(i + 1, n - 1)
            dt = (members[j1]["t_ns"] - members[j0]["t_ns"]) / 1e9
            if dt <= 0 or dt > max_dt * (2 if (j1 - j0 == 2) else 1):
                out[r["token"]] = None
                continue
            d = global_center_of[members[j1]["token"]][:2] - global_center_of[members[j0]["token"]][:2]
            out[r["token"]] = np.asarray(d, dtype=np.float64) / dt
    return out


def attribute_state(speed_mps, cfg: AttributeConfig):
    if speed_mps is None:
        return None
    return "moving" if speed_mps > cfg.moving_speed_threshold_mps else "stopped"


def attribute_name_for(dbench_class: str, state):
    if state is None or dbench_class in NO_ATTRIBUTE_CLASSES:
        return None
    if dbench_class in PEDESTRIAN_CLASSES:
        return "pedestrian.moving" if state == "moving" else "pedestrian.standing"
    if dbench_class in CYCLE_CLASSES:
        return "cycle.with_rider"
    if dbench_class in VEHICLE_CLASSES:
        return f"vehicle.{state}"
    raise ValueError(f"unknown dbench class {dbench_class!r}")


def assign_attributes(rows: list, class_of: dict, global_center_of: dict, cfg: AttributeConfig) -> None:
    vel = chain_velocities(rows, global_center_of, cfg)
    for r in rows:
        v = vel.get(r["token"])
        r["velocity_chain_mps"] = None if v is None else [float(v[0]), float(v[1])]
        human = r.get("attribute")
        if human:
            if human not in ATTRIBUTE_NAMES:
                raise ValueError(f"record {r['token']}: attribute {human!r} is not a nuScenes attribute name")
            r["attr_state"], r["attr_name"], r["attribute_basis"] = None, human, BASIS_HUMAN
            continue
        speed = None if v is None else float(np.hypot(v[0], v[1]))
        state = attribute_state(speed, cfg)
        name = attribute_name_for(class_of[r["instance_token"]], state)
        r["attr_state"] = state
        r["attr_name"] = name
        r["attribute_basis"] = BASIS_CHAIN if name else None
