#!/usr/bin/env python3
"""Author `priors_pilot_v0.json` for a Dhaka substrate — the handover §6 recipe as code.

Stage 6 takes `eps_bev` and Stage 8 takes `dims.mu` from this file, refuse
without it, and refuse one whose `derived_from.metadata_fingerprint` is not
the current dataroot's. Dhaka carries no `sample_annotation`, so
`pipeline.stage6_cluster.priors` cannot DERIVE one; on the operator's
2026-08-30 decision the pilot's file was hand-AUTHORED, and that file went with
the 2026-09-05 wipe. This reproduces it, bound to any chunk:

  * every taxonomy phrase the nuScenes-derived template knows is TRANSFERRED
    unchanged and stamped `nuscenes_gt_pilot_TRANSFERRED_not_measured_on_dhaka`
    with the template's fingerprint;
  * `a rickshaw` and `an auto rickshaw` come from LITERATURE (handover §6:
    2.70x1.15x1.75 m and 2.65x1.30x1.75 m, Bajaj RE class) with an ASSUMED
    sigma, stamped `literature_ASSUMED_no_measurement`;
  * a taxonomy phrase with neither source REFUSES — an unmatched class would
    otherwise take the config fallback epsilon silently (X-6);
  * `gt_derived: false`, a `source_note` saying box dimensions from this run
    are NOT evidence about Dhaka object sizes, and a `REBOUND` history.

The written file is re-read through the pipeline's own `load_priors`, so what
Stage 6 will accept is what this script guarantees.

    python scripts/author_priors_dhaka.py --paths configs/paths_day1_chunk_0006.yaml
    # -> <out_root>/priors/priors_pilot_v0.json, bound to that chunk's fingerprint

Idempotent: an existing file already bound to the same fingerprint is left
byte-identical (Stage 6/8 manifests record its sha256).
"""

from __future__ import annotations

import argparse
import copy
import datetime as _dt
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.stage6_cluster.priors import PRIORS_NAME, PRIORS_SPEC, PriorsError, load_priors  # noqa: E402

DEFAULT_TEMPLATE = "/home/mt/dhakascenes/out_nuscenes/priors/priors_pilot_v0.json"
DEFAULT_TAXONOMY = "configs/taxonomy_pilot_dhaka.yaml"

TRANSFERRED_SOURCE = "nuscenes_gt_pilot_TRANSFERRED_not_measured_on_dhaka"
ASSUMED_SOURCE = "literature_ASSUMED_no_measurement"
AUTHORED_SOURCE = "authored_dhaka:nuscenes_transferred+literature_assumed"
SIGMA_FRACTION = 0.10  # ASSUMED: no Dhaka measurement exists to set it

# handover/2026-08-30-dhaka-pilot-handoff.md §6, verbatim numbers (l x w x h).
LITERATURE: dict[str, dict] = {
    "a rickshaw": {"category": "dhaka.cycle_rickshaw", "l": 2.70, "w": 1.15, "h": 1.75,
                   "note": "cycle rickshaw, literature/typical Dhaka build"},
    "an auto rickshaw": {"category": "dhaka.cng", "l": 2.65, "w": 1.30, "h": 1.75,
                         "note": "Bajaj RE class CNG auto rickshaw"},
}

SOURCE_NOTE = (
    "AUTHORED, not derived. Box dimensions produced by a run against this file are not "
    "evidence about Dhaka object sizes: the nuScenes classes are transferred from Boston/"
    "Singapore GT and the rickshaw classes are literature values with an assumed sigma. "
    "Replace by labelling Dhaka keyframes with 3D cuboids and running "
    "pipeline.stage6_cluster.priors (§6)."
)


def taxonomy_phrases(taxonomy: dict) -> list[str]:
    """Unique prompt phrases in first-seen order — the class names every stage keys on."""
    seen: list[str] = []
    for phrase in taxonomy["prompt_phrase"].values():
        if phrase not in seen:
            seen.append(phrase)
    return seen


def _literature_block(phrase: str, eps_scale: float) -> dict:
    lit = LITERATURE[phrase]
    dims = {ax: {"mu": float(lit[ax]), "sigma": round(SIGMA_FRACTION * lit[ax], 4)} for ax in ("w", "l", "h")}
    return {
        "category": lit["category"],
        "categories": [lit["category"]],
        "dims": dims,
        "eps_bev": eps_scale * math.hypot(lit["w"], lit["l"]),
        "n_instances": 0,
        "conf_thresh": None,
        "gaps": [],
        "source": ASSUMED_SOURCE,
        "sigma_assumption": f"{SIGMA_FRACTION:.0%} of mu per axis, ASSUMED (no Dhaka measurement)",
        "dims_note": lit["note"],
    }


def author_dhaka_priors(
    template: dict,
    phrases: list[str],
    *,
    fingerprint: str,
    binding: dict,
    authored_on: str,
    previous: dict | None = None,
) -> dict:
    """The authored payload. `binding` carries dataroot_realpath / version /
    fingerprint_spec; `previous` (the file being rebound, if any) keeps the
    REBOUND history growing instead of restarting."""
    tdf = template.get("derived_from", {})
    template_fp = str(tdf.get("metadata_fingerprint", ""))
    eps_scale = float(template.get("eps_scale", 0.6))
    # Stage 6 refuses priors not derived on the 'priors' partition (P1-5, §11
    # decision 3: epsilon must never be tuned on scored scenes). Every epsilon
    # here comes from the template's priors partition or from literature, so
    # the authored file carries that subset — and a template that was not
    # itself derived on 'priors' is refused rather than laundered through here.
    template_subset = tdf.get("scene_subset")
    if template_subset != "priors":
        raise ValueError(
            f"template derived_from.scene_subset={template_subset!r}, expected 'priors': its "
            "epsilons may have been tuned on scored scenes and cannot be transferred (P1-5)"
        )

    classes: dict[str, dict] = {}
    unsourced: list[str] = []
    for phrase in phrases:
        if phrase in template.get("classes", {}):
            block = copy.deepcopy(template["classes"][phrase])
            block["source"] = TRANSFERRED_SOURCE
            block["transferred_from_fingerprint"] = template_fp
            classes[phrase] = block
        elif phrase in LITERATURE:
            classes[phrase] = _literature_block(phrase, eps_scale)
        else:
            unsourced.append(phrase)
    if unsourced:
        raise ValueError(
            f"no prior source for taxonomy phrase(s) {unsourced}: not in the template and not in "
            "LITERATURE. Refusing — an unmatched class would take the config fallback epsilon "
            "silently (X-6)"
        )

    history: list[dict] = []
    prev_fp = None
    if previous:
        prev_df = previous.get("derived_from", {})
        prev_fp = prev_df.get("metadata_fingerprint")
        history = list(prev_df.get("REBOUND", {}).get("rebound_history", []))
    if not history or history[-1].get("to") != fingerprint:
        history.append({
            "from": prev_fp,
            "to": fingerprint,
            "on": authored_on,
            "reason": "bound to this dataroot's metadata fingerprint (Stage 6/8 refuse otherwise)",
        })

    return {
        "spec": PRIORS_SPEC,
        "name": PRIORS_NAME,
        "source": AUTHORED_SOURCE,
        "gt_derived": False,
        "source_note": SOURCE_NOTE,
        "eps_scale": eps_scale,
        "eps_formula": template.get("eps_formula", "eps_bev = eps_scale * mean_i sqrt(w_i^2 + l_i^2)"),
        "classes": classes,
        "classes_without_instances": [p for p in phrases if classes[p].get("n_instances", 0) == 0],
        "derived_from": {
            "metadata_fingerprint": fingerprint,
            "fingerprint_spec": binding.get("fingerprint_spec"),
            "dataroot_realpath": binding.get("dataroot_realpath"),
            "version": binding.get("version"),
            # Inherited from the template, truthfully: no epsilon in this file
            # was tuned on a scene this pipeline scores (see the guard above).
            "scene_subset": template_subset,
            "scenes": [],
            "authored_on": authored_on,
            "authored_by": "scripts/author_priors_dhaka.py",
            "transferred_from": {
                "metadata_fingerprint": template_fp,
                "version": tdf.get("version"),
                "name": template.get("name"),
                "source": template.get("source"),
                "scene_subset": template_subset,
                "scenes": list(tdf.get("scenes", [])),
            },
            "REBOUND": {
                "note": "the fingerprint must be rebound whenever the metadata tables change",
                "rebound_history": history,
            },
        },
        "release_guard": template.get("release_guard"),
        "license_note": template.get("license_note"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--paths", default=None,
                        help="paths yaml: fingerprint, dataroot, version and the default --out come from it")
    parser.add_argument("--template", default=DEFAULT_TEMPLATE, help="the nuScenes-DERIVED priors file to transfer from")
    parser.add_argument("--taxonomy", default=DEFAULT_TAXONOMY)
    parser.add_argument("--out", default=None, help=f"default <out_root>/priors/{PRIORS_NAME}.json")
    parser.add_argument("--fingerprint", default=None, help="override the computed fingerprint (manual rebind)")
    parser.add_argument("--dataroot", default=None, help="recorded in derived_from when --paths is not given")
    parser.add_argument("--version", default=None, help="recorded in derived_from when --paths is not given")
    args = parser.parse_args(argv)

    fingerprint_spec = None
    if args.paths:
        from pipeline.common.paths import FINGERPRINT_SPEC, load_paths, metadata_fingerprint  # noqa: E402
        paths = load_paths(args.paths)
        fingerprint = args.fingerprint or metadata_fingerprint(paths)
        dataroot = os.path.realpath(paths.dataroot)
        version = paths.version
        out = args.out or os.path.join(paths.out_root, "priors", f"{PRIORS_NAME}.json")
        fingerprint_spec = FINGERPRINT_SPEC
    else:
        if not (args.fingerprint and args.dataroot and args.version and args.out):
            print("without --paths, give --fingerprint, --dataroot, --version and --out", file=sys.stderr)
            return 2
        fingerprint, dataroot, version, out = args.fingerprint, args.dataroot, args.version, args.out

    with open(args.template, encoding="utf-8") as fh:
        template = json.load(fh)
    import yaml  # noqa: E402 — configs are YAML everywhere in this repo

    with open(args.taxonomy, encoding="utf-8") as fh:
        phrases = taxonomy_phrases(yaml.safe_load(fh))

    previous = None
    if os.path.isfile(out):
        with open(out, encoding="utf-8") as fh:
            previous = json.load(fh)
        if previous.get("derived_from", {}).get("metadata_fingerprint") == fingerprint:
            print(f"{out}: already bound to {fingerprint[:16]}…; left byte-identical")
            return 0

    payload = author_dhaka_priors(
        template, phrases, fingerprint=fingerprint,
        binding={"dataroot_realpath": dataroot, "version": version, "fingerprint_spec": fingerprint_spec},
        authored_on=_dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        previous=previous,
    )
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    os.replace(tmp, out)
    try:
        priors = load_priors(out)  # the pipeline's own reader is the acceptance test
    except PriorsError as exc:
        print(f"authored file fails the pipeline's own loader: {exc}", file=sys.stderr)
        return 2
    n_lit = sum(1 for b in payload["classes"].values() if b["source"] == ASSUMED_SOURCE)
    print(f"{out}")
    print(f"  bound to {fingerprint}  ({dataroot} @ {version})")
    print(f"  classes: {len(priors.classes)} = {len(priors.classes) - n_lit} transferred from nuScenes GT "
          f"+ {n_lit} literature/assumed; rebound history: {len(payload['derived_from']['REBOUND']['rebound_history'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
