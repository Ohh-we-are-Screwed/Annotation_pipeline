"""Tests for local_yolox_build/scripts/export_armb.py (torch-free parts).

Run: /home/mt/miniconda3/envs/ano_pipe/bin/python -m pytest tests/test_export_armb.py -v
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "local_yolox_build", "scripts"))

from export_armb import checkpoint_names, provenance_from_run  # noqa: E402

CSV = """epoch,time,train/box_loss,train/cls_loss,train/dfl_loss,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B),val/box_loss,val/cls_loss,val/dfl_loss,lr/pg0,lr/pg1,lr/pg2
1,100.0,1.0,1.0,1.0,0.8,0.7,0.90,0.60,1.0,0.7,1.1,0.005,0.005,0.005
2,200.0,0.9,0.9,0.9,0.9,0.8,0.93,0.74,0.9,0.6,1.0,0.006,0.006,0.006
3,300.0,0.8,0.8,0.8,0.9,0.8,0.92,0.73,0.8,0.5,1.0,0.007,0.007,0.007
"""

ARGS = """task: detect
model: /somewhere/yolo11x.pt
data: /somewhere/rsud20k_yolo11x.yaml
epochs: 80
classes:
- 0
- 1
- 3
- 6
- 7
"""


def _fake_run(tmp_path):
    run = tmp_path / "r-test"
    (run / "weights").mkdir(parents=True)
    (run / "weights" / "best.pt").write_bytes(b"not a real checkpoint")
    (run / "results.csv").write_text(CSV)
    (run / "args.yaml").write_text(ARGS)
    return str(run)


def test_best_row_is_max_fitness(tmp_path):
    prov = provenance_from_run(_fake_run(tmp_path))
    # ultralytics 8.4.120 detect fitness is mAP50-95 ALONE (metrics.py:1009,
    # weights [0,0,0,1]) -> epoch 2 (0.74) beats epoch 3 (0.73). NOT the older
    # 0.1*mAP50 + 0.9*mAP50-95 blend: that picks a different epoch in general,
    # and would disagree with the best_fitness the checkpoint itself carries.
    assert prov["best_row"]["epoch"] == 2
    assert abs(prov["best_fitness"] - 0.74) < 1e-9
    assert prov["classes_trained"] == [0, 1, 3, 6, 7]
    assert prov["source_run"].endswith("r-test")
    assert prov["ship_names"] == ["rickshaw", "cng"]


def test_refuses_run_without_best_pt(tmp_path):
    import pytest
    run = tmp_path / "empty"
    (run / "weights").mkdir(parents=True)
    (run / "results.csv").write_text(CSV)
    (run / "args.yaml").write_text(ARGS)
    with pytest.raises(SystemExit):
        provenance_from_run(str(run))


class _FakeModel:
    def __init__(self, names):
        self.names = names


NAMES = {0: "person", 1: "rickshaw", 3: "cng"}


def test_checkpoint_names_prefers_model_when_stripped():
    ck = {"model": _FakeModel(NAMES), "ema": None}
    assert checkpoint_names(ck) == NAMES


def test_checkpoint_names_falls_back_to_ema_mid_training():
    # An interrupted run's best.pt carries model=None and the weights in `ema`;
    # strip_optimizer only moves them into `model` when training finishes.
    ck = {"model": None, "ema": _FakeModel(NAMES)}
    assert checkpoint_names(ck) == NAMES


def test_checkpoint_names_refuses_when_neither_names_the_classes():
    import pytest
    with pytest.raises(SystemExit):
        checkpoint_names({"model": None, "ema": None})
