"""DELIVERY_NOTE.md — the per-chunk text block the benchmark requires (spec §8).

Every number is read out of `<out>/release_meta.json` (the exporter's own record
of what it shipped), the Stage 9 `run_manifest.json` (the gates that decided
which boxes exist at all), the CVAT `import_manifest.json` and
`double_annotation.json`. Only the fixed sentences quoted in spec §8 — the
annotation rule and the anonymisation order — are typed here; nothing else in
the note is hand-written, so it cannot drift from the export it describes.
"""

from __future__ import annotations

import json
import os

from pipeline.common.eval_region import _RHO_RADIUS_M

# Two sentences, because the release ships two kinds of row and only the first
# faced Stage 9's gates. The old unconditional form claimed a confidence and a
# footprint test for rows that never had either (final review C2).
RULE_SENTENCE = ("A measured box ships iff it has >= {n} LiDAR returns (single-sweep, ground-filtered, "
                 "pre-inflation) AND detector confidence >= {c} AND its BEV footprint is <= {m}x class "
                 "prior. There are no camera-only boxes, so the visibility term V does not apply; "
                 "`visibility_token` is a camera field-of-view proxy, not an occlusion estimate.")
INTERP_SENTENCE = ("The other rows are interpolated: geometric fills written at a keyframe between two gated "
                   "endpoints of one stitched identity, flagged `dhakascenes_interpolated: true` with "
                   "`dhakascenes_tier_basis: \"inherited_from_endpoints\"` (filter on either). They are not "
                   "detections, so the confidence and footprint clauses do not apply to them; they inherit "
                   "the worse of their two endpoints\' tiers. Their `num_lidar_pts` IS measured, inside the "
                   "box against that keyframe\'s own ground-filtered cloud, and must clear the same >= {f} "
                   "return floor: a fill below it does not ship, it goes to "
                   "`sample_annotation_excluded.json` with `dhakascenes_excluded_reason: "
                   "interpolated_below_point_floor`.")
ANON_SENTENCE = ("After annotation. The annotated images are un-blurred; face and plate blurring is applied "
                 "to the released images afterwards, so any box drawn from image evidence saw the original pixels.")


def _kv(rows):
    return "\n".join(f"- {k}: {v}" for k, v in rows)


def _m(value) -> str:
    """A metre count without a decimal tail: 30.0 -> '30', 41.25 -> '41.25'."""
    try:
        return f"{float(value):g}"
    except (TypeError, ValueError):
        return str(value)


def render_note(meta, stage9_manifest, import_manifest, double_doc, stage_tree, chunk_name) -> str:
    s9 = (stage9_manifest or {}).get("config", {})
    rule = RULE_SENTENCE.format(n=s9.get("min_lidar_returns", "?"), c=s9.get("conf_gate", "?"),
                                m=s9.get("spatial_multiplier", "?"))
    cls = meta.get("classes", {})
    rng = meta.get("range", {})
    st = meta.get("stitch", {})
    hu = meta.get("human", {})
    at = meta.get("attributes", {})
    dbl = meta.get("double_annotation")
    ex = meta.get("excluded", {})
    rc = meta.get("release_config", {})
    lines = [f"# {chunk_name} — delivery note", ""]
    lines += ["## Provenance", _kv([
        ("export created (UTC)", meta.get("created_utc")), ("nuScenes version dir", meta.get("version")),
        ("pipeline git sha", meta.get("git_sha")), ("Stage 9 spec", (stage9_manifest or {}).get("spec") or meta.get("pipeline_version")),
        ("stage tree", stage_tree or "(not recorded)"),
        ("release config sha256", rc.get("sha256")),
        ("benchmark definition", f"{rc.get('benchmark_source', {}).get('path')} sha256 {rc.get('benchmark_source', {}).get('sha256')}"),
    ]), ""]
    if hu.get("enabled"):
        stt = hu.get("stats", {})
        task_lines = [f"task {t.get('task_id')} ({t.get('kind')}, {t.get('scene')}, assignee {t.get('assignee')})"
                      for t in (import_manifest or {}).get("tasks", [])]
        lines += ["## Human pass", _kv([
            ("review rows / samples", f"{stt.get('n_rows_review', 0)} / {stt.get('n_samples_review', 0)}"),
            ("double pass A rows / samples", f"{stt.get('n_rows_double_A', 0)} / {stt.get('n_samples_double_A', 0)}"),
            ("double pass B rows / samples", f"{stt.get('n_rows_double_B', 0)} / {stt.get('n_samples_double_B', 0)}"),
            ("half-imported double frames (B without A)", ", ".join(hu.get("half_imported_samples", [])) or "none"),
            ("CVAT tasks imported", "; ".join(task_lines) or "(no import manifest)"),
        ]), ""]
    else:
        lines += ["## Human pass", "No human pass has run on this export: every row is a machine pre-annotation "
                  "(`dhakascenes_source: pipeline`).", ""]
    # The rho radius and the annotation cap were both 30 m before 2026-09-07 and are
    # routinely conflated; after the cap moved to 50 m they are two different numbers.
    # Both come out of release_meta's `range` block: `cap_m` is the pipeline cap the
    # exporter ran under (eval_region._R_MAX_M), `max_exported_range_m` the furthest
    # box that actually shipped.
    rho_radius = ((rc.get("strata") or {}).get("density_radius_m")
                  if isinstance(rc.get("strata"), dict) else None)
    rho_radius = _RHO_RADIUS_M if rho_radius is None else rho_radius
    cap = _m(rng.get("cap_m"))
    # The floor the exporter actually applied (Stage 9's own min_lidar_returns
    # when it had a manifest, else configs/release.yaml), so the sentence cannot
    # claim a number the table was not gated on.
    fl = st.get("interpolated_point_floor")
    floor = fl.get("value") if isinstance(fl, dict) else None
    if floor is None:
        floor = s9.get("min_lidar_returns", "?")
    rule_block = [rule]
    if (st.get("totals") or {}).get("n_interpolated"):
        rule_block += ["", INTERP_SENTENCE.format(f=floor)]
    lines += ["## Annotation rule", *rule_block, "", "## Range", _kv([
        ("pipeline range cap", f"{cap} m (Stage 1 `range_cap_m` / eval region `_R_MAX_M`, "
                               "the benchmark's class_range maximum)"),
        ("furthest exported box (observed, not a cap)",
         f"{rng.get('max_exported_range_m')} m in the ego BEV plane"),
        ("effective p99 range per class (included boxes)", ", ".join(f"{k} {v} m" for k, v in sorted((rng.get('effective_p99_m_by_class') or {}).items())) or "n/a"),
        ("density radius (rho)", f"{_m(rho_radius)} m, unchanged — `eval_region._RHO_RADIUS_M`, benchmark "
                                 f"`stratification.density.radius_m`. It used to equal the annotation range cap; "
                                 f"the cap is now {cap} m, so rho is a density over a {_m(rho_radius)} m "
                                 f"disc inside a {cap} m region, not over the whole annotated region."),
    ]), ""]
    lines += ["## Tiers", _kv([
        ("admitted to sample_annotation", meta.get("tiers_admitted")),
        ("excluded rows", f"{ex.get('n', 0)} in {ex.get('table')} — " + ", ".join(f"{k} {v}" for k, v in sorted((ex.get('by_reason') or {}).items()))),
    ]), "`sample_annotation_excluded.json` is not a nuScenes table; the devkit does not load it.", ""]
    present = cls.get("present", {})
    lines += ["## Classes", _kv([
        ("present (instances)", ", ".join(f"{k} {v}" for k, v in sorted(present.items())) or "none"),
        ("in the detector vocabulary but absent on this route", ", ".join(cls.get("absent_on_route", [])) or "none"),
        ("not producible by the 12-phrase detector vocabulary", ", ".join(cls.get("not_producible", [])) or "none"),
    ]), ""]
    tot = st.get("totals", {})
    lines += ["## Identity (stitching)", _kv([
        ("enabled", st.get("enabled")), ("fragments -> chains", f"{tot.get('n_fragments')} -> {tot.get('n_chains')}"),
        ("joins by gap (keyframes)", json.dumps(tot.get("joins_by_gap", {}))), ("interpolated rows", tot.get("n_interpolated")),
        ("median chain length before -> after, per scene", "; ".join(f"{k} {v.get('median_len_before')} -> {v.get('median_len_after')}" for k, v in (st.get("per_scene") or {}).items()) or "n/a"),
        ("max gap", f"{(st.get('config') or {}).get('max_gap_keyframes')} keyframes"),
    ]), "A class flip along one object stays two instances (class-agnostic stitching is off).", ""]
    lines += ["## Attributes", _kv([
        ("derivation", f"chain velocity (nuScenes box_velocity semantics); moving if > {at.get('threshold_mps')} m/s else stopped; "
                       "single-annotation instances get none; static classes and animals never; parked / sitting / without_rider are human-only"),
        ("rows with an attribute", at.get("n_with_attribute")), ("by name", json.dumps(at.get("by_name", {}))), ("by basis", json.dumps(at.get("by_basis", {}))),
    ]), ""]
    lines += ["## Uncertainty", "`is_uncertain` is false on every pipeline row: no class-uncertainty signal survives to the "
              "pre-labels in this phase (the VLM check is off). Human rows carry what the annotator set.", ""]
    if dbl:
        cells = "; ".join(f"{c['density']}/{c['illumination']} {c['selected']}/{c['n']}" for c in dbl.get("cells", []))
        lines += ["## Double annotation", _kv([
            ("frames selected", f"{dbl.get('n_selected')} of {(double_doc or {}).get('n_keyframes', '?')} "
                                f"(fraction {(double_doc or {}).get('fraction', '?')}, seed {(double_doc or {}).get('seed', '?')})"),
            ("stratification", f"density quartiles {meta.get('strata', {}).get('density_bin_edges')} x illumination luma edges {meta.get('strata', {}).get('illumination_bin_edges')}"),
            ("cells (selected/n)", cells), ("selection reused from an earlier export", dbl.get("reused")),
            ("A/B rows imported", "yes" if hu.get("enabled") and (hu.get("stats") or {}).get("n_rows_double_A") else "no"),
        ]), ""]
    else:
        lines += ["## Double annotation", "Disabled for this export (double fraction 0).", ""]
    # The heading row answers the checklist question in the checklist's own words;
    # ANON_SENTENCE below it is the spec §8 quote, kept verbatim.
    lines += ["## Anonymisation", _kv([("face and plate blurring", "applied after annotation")]),
              ANON_SENTENCE, ""]
    used = meta.get("mapper", {}).get("used", {})
    lines += ["## Extra layers", "- road/: driveable-surface lidarseg (nuScenes-lidarseg tables + .bin), reads against boxes/samples/.",
              "- coco_2d/: 2D-only auxiliary layer with the detector's phrase names; not consumed by the 3D benchmark. Phrase -> class: "
              + ", ".join(f"{k} -> {v}" for k, v in sorted(used.items())), ""]
    lines += ["## Files", "- boxes/<version>/: the 13 nuScenes tables; sample.json carries `dbench_double_annotated`.",
              "- boxes/sample_annotation_excluded.json, boxes/stitch_map.json, boxes/double_annotation.json, boxes/release_meta.json.",
              f"- num_lidar_pts basis: {', '.join(meta.get('num_lidar_pts_basis', []))}; visibility basis: {meta.get('visibility', {}).get('basis')}.", ""]
    return "\n".join(lines)


def write_note(out_root: str, *, stage9_manifest_path, import_manifest_path, stage_tree, chunk_name) -> str:
    with open(os.path.join(out_root, "release_meta.json"), "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    s9 = _read_json(stage9_manifest_path)
    imp = _read_json(import_manifest_path)
    dbl = _read_json(os.path.join(out_root, "double_annotation.json"))
    path = os.path.join(out_root, "DELIVERY_NOTE.md")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(render_note(meta, s9, imp, dbl, stage_tree, chunk_name))
    os.replace(tmp, path)
    return path


def _read_json(path):
    if not path or not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
