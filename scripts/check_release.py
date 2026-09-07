#!/usr/bin/env python3
"""Release checklist (spec §8): errors exit 2, warnings exit 1, clean exit 0.

Reads one export root (`<out>/` — the directory holding `<version>/`, the
sidecars and `release_meta.json`) and re-derives, from the shipped tables alone,
every claim the benchmark's checklist and DELIVERY_NOTE.md make about it. It
imports nothing from the exporter's in-memory state on purpose: a table that
disagrees with the note is exactly what this is for.

    python scripts/check_release.py export/day1_chunk_0000/boxes [--version v1.0-dhaka-fixed]

`scripts/export_release.py` runs it at the end of every export.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.common.schemas import HUMAN_SOURCES  # noqa: E402

STATIC = ("traffic_cone", "barrier", "construction_element", "animal")
INTERP_WARN_FRACTION = 0.20


@dataclass
class Report:
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    info: dict = field(default_factory=dict)


def _load(d, name):
    with open(os.path.join(d, f"{name}.json"), "r", encoding="utf-8") as fh:
        return json.load(fh)


def _detect_version(out_root):
    cands = [e for e in os.listdir(out_root) if os.path.isfile(os.path.join(out_root, e, "sample.json"))]
    if len(cands) != 1:
        raise SystemExit(f"cannot detect the version dir under {out_root}: {cands}")
    return cands[0]


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_release(out_root: str, version=None) -> Report:
    rep = Report()
    version = version or _detect_version(out_root)
    d = os.path.join(out_root, version)
    anns, inst, cats = _load(d, "sample_annotation"), _load(d, "instance"), _load(d, "category")
    attrs, samples = _load(d, "attribute"), _load(d, "sample")
    meta_p = os.path.join(out_root, "release_meta.json")
    meta = json.load(open(meta_p)) if os.path.isfile(meta_p) else {}
    admitted = meta.get("tiers_admitted", "auto_accept")
    inst_by = {i["token"]: i for i in inst}
    cat_by = {c["token"]: c["name"] for c in cats}
    attr_by = {a["token"]: a["name"] for a in attrs}
    sample_by = {s["token"]: s for s in samples}
    ann_by = {a["token"]: a for a in anns}
    # Samples a human owns at least one row on: an interpolated machine row that
    # survives on such a keyframe outlived the endpoint the human superseded.
    human_samples = {a["sample_token"] for a in anns if a.get("dhakascenes_source") in HUMAN_SOURCES}
    by_inst = defaultdict(list)
    for a in anns:
        t = a["token"]
        if any(float(v) <= 0 for v in a["size"]):
            rep.errors.append(f"{t}: non-positive size {a['size']}")
        elif float(a["size"][0]) > float(a["size"][1]):
            rep.warnings.append(f"{t}: w > l size {a['size']}")
        n = math.sqrt(sum(float(v) ** 2 for v in a["rotation"]))
        if abs(n - 1.0) > 1e-3:
            rep.errors.append(f"{t}: quaternion norm {n:.4f}")
        if a["instance_token"] not in inst_by:
            rep.errors.append(f"{t}: dangling instance_token {a['instance_token']}")
        if a["sample_token"] not in sample_by:
            rep.errors.append(f"{t}: dangling sample_token {a['sample_token']}")
        for at in a.get("attribute_tokens", []):
            if at not in attr_by:
                rep.errors.append(f"{t}: dangling attribute token {at}")
        for k in ("prev", "next"):
            other = a.get(k)
            if other:
                o = ann_by.get(other)
                back = "next" if k == "prev" else "prev"
                if o is None or o.get(back) != t or o["instance_token"] != a["instance_token"]:
                    rep.errors.append(f"{t}: prev/next asymmetry on {k}={other}")
        src = a.get("dhakascenes_source", "pipeline")
        if src in HUMAN_SOURCES:
            if not a.get("dhakascenes_verified_by"):
                rep.errors.append(f"{t}: human row without verified_by")
        else:
            if admitted != "all" and a.get("dhakascenes_tier") != "auto_accept":
                rep.errors.append(f"{t}: pipeline tier {a.get('dhakascenes_tier')!r} inside sample_annotation")
            if a.get("dhakascenes_interpolated") and a["sample_token"] in human_samples:
                rep.warnings.append(f"{t}: interpolated row on sample {a['sample_token']}, whose other rows are "
                                    "human-sourced (a stitched interpolation that outlived a human-superseded endpoint)")
        if a.get("annotator_pass") == "B":
            rep.warnings.append(f"{t}: pass B row inline (a harness without a pass filter double-counts this frame)")
        by_inst[a["instance_token"]].append(a)
    for i in inst:
        if i["category_token"] not in cat_by:
            rep.errors.append(f"instance {i['token']}: dangling category_token")
        rows = sorted(by_inst.get(i["token"], []), key=lambda a: sample_by.get(a["sample_token"], {}).get("timestamp", 0))
        if i["nbr_annotations"] != len(rows):
            rep.errors.append(f"instance {i['token']}: nbr_annotations {i['nbr_annotations']} != {len(rows)} rows")
        if rows:
            heads = [a for a in rows if not a.get("prev")]
            tails = [a for a in rows if not a.get("next")]
            if len(heads) != 1 or len(tails) != 1 or heads[0]["token"] != i["first_annotation_token"] or tails[0]["token"] != i["last_annotation_token"]:
                rep.errors.append(f"instance {i['token']}: first/last/chain inconsistent")
            cls = cat_by.get(i["category_token"])
            if cls not in STATIC and len(rows) >= 2:
                for a in rows:
                    if not a.get("attribute_tokens"):
                        rep.warnings.append(f"{a['token']}: non-static multi-annotation box without attribute")
    present = Counter(cat_by.get(i["category_token"]) for i in inst)
    for name in sorted(cat_by.values()):
        if present[name] == 0:
            rep.warnings.append(f"class {name}: zero instances")
    n_interp = sum(1 for a in anns if a.get("dhakascenes_interpolated"))
    if anns and n_interp / len(anns) > INTERP_WARN_FRACTION:
        rep.warnings.append(f"interpolated rows are {n_interp / len(anns):.0%} of sample_annotation")
    dpath = os.path.join(out_root, "double_annotation.json")
    if os.path.isfile(dpath):
        doc = json.load(open(dpath))
        flagged = {s["token"] for s in samples if s.get("dbench_double_annotated")}
        need = math.ceil(float(doc.get("fraction", 0)) * len(samples))
        if len(flagged) < need:
            rep.errors.append(f"double annotation: {len(flagged)} flagged samples < required {need}")
        if set(s["sample_token"] for s in doc.get("selected", [])) != flagged:
            rep.errors.append("double annotation: sample.json flags disagree with double_annotation.json")
        if (meta.get("human") or {}).get("stats", {}).get("n_rows_double_A", 0) > 0:
            passes = defaultdict(set)
            for a in anns:
                if a.get("annotator_pass"):
                    passes[a["sample_token"]].add(a["annotator_pass"])
            for s in sorted(flagged):
                if passes.get(s) and passes[s] != {"A", "B"}:
                    rep.errors.append(f"double annotation: sample {s} has rows from pass {sorted(passes[s])} only")
    # The release pins the benchmark definition it was built against by digest.
    # A file that no longer matches means the release config was never re-pinned
    # after the benchmark moved. Absent file: this is another machine, say nothing.
    bench = ((meta.get("release_config") or {}).get("benchmark_source") or {})
    bpath, bsha = bench.get("path"), bench.get("sha256")
    if bpath and bsha and os.path.isfile(bpath):
        try:
            got = _sha256(bpath)
        except OSError:
            got = None
        if got and got != bsha:
            rep.warnings.append(f"benchmark definition {bpath} has sha256 {got}, release_meta pins {bsha} "
                                "(the release config was not re-pinned after the benchmark changed)")
    if not os.path.isfile(os.path.join(out_root, "DELIVERY_NOTE.md")):
        rep.warnings.append("DELIVERY_NOTE.md missing")
    rep.info = {"n_annotations": len(anns), "n_instances": len(inst), "n_interpolated": n_interp, "present": dict(present)}
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out_root")
    ap.add_argument("--version", default=None)
    args = ap.parse_args(argv)
    rep = check_release(args.out_root, args.version)
    for e in rep.errors:
        print(f"ERROR   {e}")
    for w in rep.warnings:
        print(f"WARNING {w}")
    print(f"check_release: {len(rep.errors)} error(s), {len(rep.warnings)} warning(s); {rep.info}")
    return 2 if rep.errors else (1 if rep.warnings else 0)


if __name__ == "__main__":
    raise SystemExit(main())
