from __future__ import annotations

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pipeline.release.attributes import (  # noqa: E402
    assign_attributes, attribute_name_for, attribute_state, chain_velocities,
)
from pipeline.release.config import AttributeConfig  # noqa: E402

CFG = AttributeConfig(moving_speed_threshold_mps=0.5, max_time_diff_s=1.5)


def _row(tok, inst, t_s, attribute=None):
    return {"token": tok, "instance_token": inst, "t_ns": int(t_s * 1e9), "attribute": attribute}


def test_chain_velocity_matches_devkit_semantics():
    rows = [_row("a", "i", 0.0), _row("b", "i", 0.4), _row("c", "i", 0.8)]
    pos = {"a": np.array([0.0, 0, 0]), "b": np.array([2.0, 0, 0]), "c": np.array([4.0, 0, 0])}
    v = chain_velocities(rows, pos, CFG)
    assert np.allclose(v["b"], [5.0, 0.0])          # central difference
    assert np.allclose(v["a"], [5.0, 0.0]) and np.allclose(v["c"], [5.0, 0.0])   # one-sided at the ends


def test_single_annotation_and_big_gap_are_undefined():
    rows = [_row("a", "i", 0.0)]
    assert chain_velocities(rows, {"a": np.zeros(3)}, CFG) == {"a": None}
    rows = [_row("a", "j", 0.0), _row("b", "j", 2.0)]
    v = chain_velocities(rows, {"a": np.zeros(3), "b": np.array([3.0, 0, 0])}, CFG)
    assert v == {"a": None, "b": None}


def test_state_threshold():
    assert attribute_state(None, CFG) is None
    assert attribute_state(0.49, CFG) == "stopped"
    assert attribute_state(0.51, CFG) == "moving"


@pytest.mark.parametrize("cls,state,name", [
    ("car", "moving", "vehicle.moving"), ("cng_autorickshaw", "stopped", "vehicle.stopped"),
    ("pushcart", "moving", "vehicle.moving"),
    ("pedestrian", "moving", "pedestrian.moving"), ("pedestrian", "stopped", "pedestrian.standing"),
    ("bicycle", "moving", "cycle.with_rider"), ("cycle_rickshaw", "stopped", "cycle.with_rider"),
    ("traffic_cone", "moving", None), ("animal", "stopped", None), ("car", None, None),
])
def test_names(cls, state, name):
    assert attribute_name_for(cls, state) == name


def test_assign_attributes_sets_basis_and_respects_human_value():
    rows = [_row("a", "i", 0.0), _row("b", "i", 0.4), _row("h", "k", 0.0, attribute="vehicle.parked")]
    pos = {"a": np.zeros(3), "b": np.array([0.1, 0, 0]), "h": np.zeros(3)}
    assign_attributes(rows, {"i": "car", "k": "car"}, pos, CFG)
    a, b, h = rows
    assert a["attr_name"] == "vehicle.stopped" and a["attribute_basis"] == "chain_velocity"
    assert b["velocity_chain_mps"] == pytest.approx([0.25, 0.0])
    assert h["attr_name"] == "vehicle.parked" and h["attribute_basis"] == "human"
    assert h["velocity_chain_mps"] is None


def test_unknown_human_attribute_name_raises():
    rows = [_row("h", "k", 0.0, attribute="vehicle.flying")]
    with pytest.raises(ValueError):
        assign_attributes(rows, {"k": "car"}, {"h": np.zeros(3)}, CFG)
