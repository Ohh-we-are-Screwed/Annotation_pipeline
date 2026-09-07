"""scripts/author_priors_dhaka.py — the AUTHORED Dhaka priors file, reproducibly.

Stage 6 takes eps_bev and Stage 8 takes dims.mu from priors_pilot_v0.json,
refuse without it, and refuse one not bound to the current dataroot
fingerprint. Dhaka has no sample_annotation to derive one from, so on the
operator's 2026-08-30 decision the pilot's file was hand-authored (handover
§6): the nuScenes-derived classes transferred unchanged and stamped, the two
rickshaw classes from literature with an ASSUMED sigma, gt_derived false, a
source_note saying box sizes from this run are not evidence about Dhaka
objects, and a REBOUND history. That file went with the 2026-09-05 wipe. This
script is that recipe as code, bound per chunk, checked by the real loader.
"""

from __future__ import annotations

import json
import math
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pipeline.stage6_cluster.priors import PRIORS_SPEC, load_priors  # noqa: E402
from scripts.author_priors_dhaka import (  # noqa: E402
    LITERATURE,
    TRANSFERRED_SOURCE,
    ASSUMED_SOURCE,
    author_dhaka_priors,
    main,
    taxonomy_phrases,
)

FP_NUSC = "4c5a" * 16
FP_CHUNK = "ab12" * 16
BINDING = {"dataroot_realpath": "/data/chunk_0006", "version": "v1.0-dhaka-fixed",
           "fingerprint_spec": "sha256-of-sorted-name-digest-manifest/v1"}


def _dims(w, l, h, s=0.1):
    return {"w": {"mu": w, "sigma": s}, "l": {"mu": l, "sigma": s}, "h": {"mu": h, "sigma": s}}


def _template():
    return {
        "spec": PRIORS_SPEC, "name": "priors_pilot_v0", "source": "nuscenes_gt_pilot", "eps_scale": 0.6,
        "eps_formula": "eps_bev = eps_scale * mean_i sqrt(w_i^2 + l_i^2)",
        "derived_from": {"metadata_fingerprint": FP_NUSC, "version": "v1.0-mini", "scene_subset": "priors"},
        "classes": {
            "a car": {"category": "vehicle.car", "dims": _dims(1.92, 4.61, 1.71), "eps_bev": 2.995,
                      "n_instances": 1040, "conf_thresh": None, "gaps": [], "measured": {"x": 1}},
            "a bicycle": {"category": "vehicle.bicycle", "dims": _dims(0.59, 1.74, 1.54), "eps_bev": 1.1,
                          "n_instances": 42, "conf_thresh": None, "gaps": []},
            "a trailer": {"category": "vehicle.trailer", "dims": None, "eps_bev": None,
                          "n_instances": 0, "conf_thresh": None, "gaps": ["no_instances"]},
        },
    }


TAXONOMY = {"prompt_phrase": {"vehicle.car": "a car", "human.pedestrian.adult": "a pedestrian",
                              "human.pedestrian.child": "a pedestrian", "vehicle.bicycle": "a bicycle",
                              "dhaka.cycle_rickshaw": "a rickshaw", "dhaka.cng": "an auto rickshaw"}}


class TestTaxonomyPhrases:
    def test_unique_in_first_seen_order(self):
        assert taxonomy_phrases(TAXONOMY) == ["a car", "a pedestrian", "a bicycle", "a rickshaw", "an auto rickshaw"]


class TestAuthor:
    def _author(self, phrases=("a car", "a bicycle", "a rickshaw", "an auto rickshaw"), **kw):
        return author_dhaka_priors(_template(), list(phrases), fingerprint=FP_CHUNK, binding=BINDING,
                                   authored_on="2026-09-06T01:10:00+06:00", **kw)

    def test_transferred_classes_keep_their_dims_and_are_stamped(self):
        out = self._author()
        car = out["classes"]["a car"]
        assert car["dims"] == _template()["classes"]["a car"]["dims"]
        assert car["eps_bev"] == 2.995 and car["n_instances"] == 1040
        assert car["source"] == TRANSFERRED_SOURCE
        assert car["transferred_from_fingerprint"] == FP_NUSC

    def test_literature_classes_are_added_with_assumed_sigma_and_eps(self):
        out = self._author()
        r = out["classes"]["a rickshaw"]
        assert (r["dims"]["l"]["mu"], r["dims"]["w"]["mu"], r["dims"]["h"]["mu"]) == (2.70, 1.15, 1.75)
        assert r["category"] == "dhaka.cycle_rickshaw"
        assert r["source"] == ASSUMED_SOURCE and r["n_instances"] == 0
        assert r["eps_bev"] == pytest.approx(0.6 * math.hypot(1.15, 2.70))
        assert all(r["dims"][a]["sigma"] > 0 for a in ("w", "l", "h"))
        assert "sigma_assumption" in r
        a = out["classes"]["an auto rickshaw"]
        assert (a["dims"]["l"]["mu"], a["dims"]["w"]["mu"], a["dims"]["h"]["mu"]) == (2.65, 1.30, 1.75)
        assert a["category"] == "dhaka.cng"

    def test_only_taxonomy_phrases_are_carried(self):
        out = self._author(phrases=("a car", "a rickshaw"))
        assert set(out["classes"]) == {"a car", "a rickshaw"}

    def test_a_phrase_with_no_source_refuses_loudly(self):
        with pytest.raises(ValueError, match="a pedestrian"):
            self._author(phrases=("a car", "a pedestrian"))

    def test_top_level_binding_and_honesty_fields(self):
        out = self._author()
        assert out["spec"] == PRIORS_SPEC and out["name"] == "priors_pilot_v0"
        assert out["source"] != "S0" and out["gt_derived"] is False
        assert "not evidence" in out["source_note"]
        assert out["eps_scale"] == 0.6
        df = out["derived_from"]
        assert df["metadata_fingerprint"] == FP_CHUNK
        assert df["dataroot_realpath"] == "/data/chunk_0006" and df["version"] == "v1.0-dhaka-fixed"
        assert df["transferred_from"]["metadata_fingerprint"] == FP_NUSC
        assert len(df["REBOUND"]["rebound_history"]) == 1
        assert df["REBOUND"]["rebound_history"][0]["to"] == FP_CHUNK
        assert out["classes_without_instances"] == ["a rickshaw", "an auto rickshaw"]

    # Stage 6 (P1-5, §11 decision 3) refuses priors whose derived_from.scene_subset
    # is not 'priors': epsilons tuned on scored scenes. Every epsilon here comes
    # from nuScenes' priors partition or from literature — none from a scene
    # this pipeline scores — so the authored file carries the template's subset,
    # and a template that was NOT derived on 'priors' is refused rather than
    # laundered through this script.
    def test_carries_the_templates_priors_partition(self):
        out = self._author()
        assert out["derived_from"]["scene_subset"] == "priors"
        assert out["derived_from"]["transferred_from"]["scene_subset"] == "priors"

    def test_refuses_a_template_not_derived_on_the_priors_partition(self):
        bad = _template()
        bad["derived_from"]["scene_subset"] = "run"
        with pytest.raises(ValueError, match="scene_subset"):
            author_dhaka_priors(bad, ["a car"], fingerprint=FP_CHUNK, binding=BINDING, authored_on="x")

    def test_rebinding_appends_history_and_keeps_classes(self):
        first = self._author()
        second = author_dhaka_priors(_template(), ["a car", "a bicycle", "a rickshaw", "an auto rickshaw"],
                                     fingerprint="cd34" * 16, binding=BINDING,
                                     authored_on="2026-09-07T00:00:00+06:00", previous=first)
        hist = second["derived_from"]["REBOUND"]["rebound_history"]
        assert [h["to"] for h in hist] == [FP_CHUNK, "cd34" * 16]
        assert hist[1]["from"] == FP_CHUNK
        assert second["classes"] == first["classes"]

    def test_round_trips_through_the_real_loader(self, tmp_path):
        p = tmp_path / "priors_pilot_v0.json"
        p.write_text(json.dumps(self._author()))
        priors = load_priors(str(p))
        assert priors.metadata_fingerprint == FP_CHUNK
        eps, prov = priors.eps_bev("a rickshaw", fallback_m=9.9)
        assert prov.startswith("priors:") and eps == pytest.approx(0.6 * math.hypot(1.15, 2.70))
        assert priors.get("a car").mu("l") == 4.61
        assert priors.get("an auto rickshaw").sigma("w") > 0


class TestMain:
    def _files(self, tmp_path):
        tpl = tmp_path / "nusc_priors.json"
        tpl.write_text(json.dumps(_template()))
        tax = tmp_path / "taxonomy.yaml"
        tax.write_text("prompt_phrase:\n" + "".join(f'  {k}: "{v}"\n' for k, v in TAXONOMY["prompt_phrase"].items()
                                                    if v != "a pedestrian"))
        out = tmp_path / "out" / "priors" / "priors_pilot_v0.json"
        return tpl, tax, out

    def test_writes_a_loadable_file_bound_to_the_given_fingerprint(self, tmp_path):
        tpl, tax, out = self._files(tmp_path)
        rc = main(["--template", str(tpl), "--taxonomy", str(tax), "--out", str(out),
                   "--fingerprint", FP_CHUNK, "--dataroot", "/data/chunk_0006", "--version", "v1.0-dhaka-fixed"])
        assert rc == 0
        priors = load_priors(str(out))
        assert priors.metadata_fingerprint == FP_CHUNK
        assert set(priors.classes) == {"a car", "a bicycle", "a rickshaw", "an auto rickshaw"}

    def test_same_fingerprint_is_a_no_op_and_a_new_one_rebinds(self, tmp_path):
        tpl, tax, out = self._files(tmp_path)
        args = ["--template", str(tpl), "--taxonomy", str(tax), "--out", str(out),
                "--dataroot", "/data/chunk_0006", "--version", "v1.0-dhaka-fixed"]
        assert main(args + ["--fingerprint", FP_CHUNK]) == 0
        before = out.read_bytes()
        assert main(args + ["--fingerprint", FP_CHUNK]) == 0
        assert out.read_bytes() == before
        assert main(args + ["--fingerprint", "cd34" * 16]) == 0
        hist = json.loads(out.read_text())["derived_from"]["REBOUND"]["rebound_history"]
        assert [h["to"] for h in hist] == [FP_CHUNK, "cd34" * 16]


def test_literature_values_are_the_handover_numbers():
    assert LITERATURE["a rickshaw"]["l"] == 2.70 and LITERATURE["a rickshaw"]["w"] == 1.15
    assert LITERATURE["an auto rickshaw"]["l"] == 2.65 and LITERATURE["an auto rickshaw"]["w"] == 1.30
