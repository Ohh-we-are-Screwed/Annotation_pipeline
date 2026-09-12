"""author_priors_dhaka.py --from-table: priors with no template (2026-09-12).

The nuScenes-derived template lived on the old box and is gone. The documented
population means (docs/Annotation_pipeline.md:141) plus the operator's rickshaw
and CNG lengths (2.40 m, stated 2026-09-12) are enough for Stage 6's epsilon and
Stage 8's dims. Unreachable phrases get a dims=None block, like the template's
trailer did, so load_priors accepts the file and Stage 6 refuses by name.
"""
from __future__ import annotations
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pipeline.stage6_cluster.priors import load_priors  # noqa: E402
from scripts.author_priors_dhaka import (  # noqa: E402
    LITERATURE, TABLE, TABLE_SOURCE, OPERATOR_SOURCE, author_from_table, main,
)

FP = "ab12" * 16
BINDING = {"dataroot_realpath": "/data/x", "version": "v1.0-dhaka-fixed2",
           "fingerprint_spec": "sha256-of-sorted-name-digest-manifest/v1"}
PHRASES = ["a car", "a pedestrian", "a road barrier", "a traffic cone", "a truck",
           "a motorcycle", "a bus", "a bicycle", "a construction vehicle", "a trailer",
           "a rickshaw", "an auto rickshaw"]


def test_operator_lengths_replace_literature():
    assert LITERATURE["a rickshaw"]["l"] == 2.40
    assert LITERATURE["an auto rickshaw"]["l"] == 2.40
    assert LITERATURE["a rickshaw"]["source"] == OPERATOR_SOURCE


def test_table_payload_loads_and_covers_every_phrase(tmp_path):
    payload = author_from_table(PHRASES, fingerprint=FP, binding=BINDING, authored_on="2026-09-12")
    out = tmp_path / "priors_pilot_v0.json"
    out.write_text(json.dumps(payload))
    priors = load_priors(str(out))
    assert priors.metadata_fingerprint == FP
    car = priors.get("a car")
    assert car is not None and abs(car.mu("l") - 4.63) < 1e-9 and abs(car.mu("w") - 1.93) < 1e-9
    rick = priors.get("a rickshaw")
    assert abs(rick.mu("l") - 2.40) < 1e-9
    assert payload["classes"]["a car"]["source"] == TABLE_SOURCE
    assert payload["classes"]["a rickshaw"]["source"] == OPERATOR_SOURCE
    # unreachable phrases are present with no dims, never silently absent
    for ph in ("a road barrier", "a traffic cone", "a construction vehicle", "a trailer"):
        assert ph in payload["classes"] and payload["classes"][ph]["dims"] is None
    eps, src = priors.eps_bev("a car", fallback_m=9.9)
    assert abs(eps - 0.6 * (1.93 ** 2 + 4.63 ** 2) ** 0.5) < 1e-6 and not src.startswith("config_fallback")
    assert payload["derived_from"]["scene_subset"] == "priors"
    assert "no scene" in payload["derived_from"]["subset_note"]


def test_cli_from_table_writes_file(tmp_path):
    out = tmp_path / "p.json"
    rc = main(["--from-table", "--out", str(out), "--fingerprint", FP,
               "--dataroot", "/data/x", "--version", "v1.0-dhaka-fixed2"])
    assert rc == 0 and out.is_file()
    load_priors(str(out))


def test_from_table_rebind_carries_previous_history(tmp_path):
    """A second --from-table run over an existing file with a different
    fingerprint must carry the previous REBOUND entry forward, exactly like
    author_dhaka_priors — not restart rebound_history at []."""
    out = tmp_path / "p.json"
    args = ["--from-table", "--out", str(out), "--dataroot", "/data/x", "--version", "v1.0-dhaka-fixed2"]
    fp2 = "cd34" * 16
    assert main(args + ["--fingerprint", FP]) == 0
    assert main(args + ["--fingerprint", fp2]) == 0
    hist = json.loads(out.read_text())["derived_from"]["REBOUND"]["rebound_history"]
    assert [h["to"] for h in hist] == [FP, fp2]
    assert hist[1]["from"] == FP
