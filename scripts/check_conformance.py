#!/usr/bin/env python3
"""Validate docs/conformance.yaml integrity and render docs/CONFORMANCE.md.

The ledger maps every checkable assertion in pilot_plan.md to evidence about
this repository (BUILD_PROMPT.md §4). This checker enforces the rules that keep
the ledger from degenerating into a reading exercise:

  R1  status=CONFORMS requires evidence_class TEST or MEASUREMENT.
      "I read the code and it does this" (CODE-SITE) supports PLAUSIBLE at most.
  R2  No row has an empty evidence_ref — even ABSENT rows name the path that
      should exist.
  R3  Every audit finding ID (P0-1..9, P1-1..15, M-1..15, X-1..10) appears in
      at least one row's `closes`.
  R4  Every top-level plan section (§0..§14) appears in at least one row's
      `source`.
  R5  IDs are unique and match <section>-r<em>N</em> or audit-<finding>.
  R6  Enum fields hold only declared values.

Silent failure this guards against: a ledger that *looks* complete — hundreds
of rows, tidy statuses — while quietly skipping the sections or findings where
the code is weakest. Coverage is asserted, not assumed.

Usage:
  python scripts/check_conformance.py            # validate + render
  python scripts/check_conformance.py --validate # validate only, exit 1 on fail
"""

from __future__ import annotations

import argparse
import collections
import datetime
import os
import re
import subprocess
import sys

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEDGER = os.path.join(REPO, "docs", "conformance.yaml")
RENDERED = os.path.join(REPO, "docs", "CONFORMANCE.md")

STATUSES = {"CONFORMS", "PLAUSIBLE", "VIOLATES", "ABSENT", "UNVERIFIABLE", "N/A"}
KINDS = {"contract", "behaviour", "artifact", "value", "test", "decision",
         "claim-hygiene", "structure"}
EVIDENCE = {"TEST", "MEASUREMENT", "CODE-SITE", "ARTIFACT", "HUMAN"}
CONFORMS_EVIDENCE = {"TEST", "MEASUREMENT"}

AUDIT_IDS = (
    [f"P0-{i}" for i in range(1, 10)]
    + [f"P1-{i}" for i in range(1, 16)]
    + [f"M-{i}" for i in range(1, 16)]
    + [f"X-{i}" for i in range(1, 11)]
)

# §0..§14 top-level sections of pilot_plan.md; each must be touched.
PLAN_SECTIONS = [str(n) for n in range(15)]

ID_RE = re.compile(r"^(?:[0-9]+(?:\.[0-9]+(?:-[0-9.]+)?)?-r[0-9]+|audit-(?:P0|P1|M|X)-[0-9]+)$")


def validate(rows: list[dict]) -> list[str]:
    errors: list[str] = []
    seen_ids: set[str] = set()
    closed: set[str] = set()
    sections_touched: set[str] = set()

    for i, r in enumerate(rows):
        tag = f"row {i} (id={r.get('id', '?')})"
        for field in ("id", "source", "claim", "kind", "status",
                      "evidence_class", "evidence_ref", "closes", "phase"):
            if field not in r:
                errors.append(f"{tag}: missing field {field!r}")
        rid = str(r.get("id", ""))
        if rid in seen_ids:
            errors.append(f"{tag}: duplicate id")
        seen_ids.add(rid)
        if rid and not ID_RE.match(rid):
            errors.append(f"{tag}: id does not match <section>-rN or audit-<finding>")
        if r.get("status") not in STATUSES:
            errors.append(f"{tag}: bad status {r.get('status')!r}")
        if r.get("kind") not in KINDS:
            errors.append(f"{tag}: bad kind {r.get('kind')!r}")
        if r.get("evidence_class") not in EVIDENCE:
            errors.append(f"{tag}: bad evidence_class {r.get('evidence_class')!r}")
        # R1 — the load-bearing rule.
        if (r.get("status") == "CONFORMS"
                and r.get("evidence_class") not in CONFORMS_EVIDENCE):
            errors.append(f"{tag}: CONFORMS with evidence_class "
                          f"{r.get('evidence_class')!r} — CODE-SITE proves PLAUSIBLE at most (R1)")
        # R2
        if not str(r.get("evidence_ref", "")).strip():
            errors.append(f"{tag}: empty evidence_ref (R2)")
        closes = r.get("closes") or []
        if not isinstance(closes, list):
            errors.append(f"{tag}: closes must be a list")
        else:
            closed.update(closes)
        m = re.match(r"^(\d+)", rid) if rid else None
        if m:
            sections_touched.add(m.group(1))
        src = str(r.get("source", ""))
        m2 = re.search(r"§\s*(\d+)", src)
        if m2:
            sections_touched.add(m2.group(1))
        if rid.startswith("audit-"):
            sections_touched.add("audit")

    # R3 — audit coverage
    for fid in AUDIT_IDS:
        if fid not in closed:
            errors.append(f"coverage: audit finding {fid} appears in no row's closes (R3)")
    # R4 — section coverage
    for s in PLAN_SECTIONS:
        if s not in sections_touched:
            errors.append(f"coverage: plan §{s} appears in no row (R4)")

    return errors


def render(rows: list[dict]) -> str:
    counts = collections.Counter(r["status"] for r in rows)
    ev_counts = collections.Counter(r["evidence_class"] for r in rows)
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True, cwd=REPO
                                ).stdout.strip() or "(no commit)"
    except OSError:
        commit = "(no git)"

    order = ["VIOLATES", "ABSENT", "UNVERIFIABLE", "PLAUSIBLE", "CONFORMS", "N/A"]
    lines = [
        "# CONFORMANCE — rendered from conformance.yaml, do not edit",
        "",
        f"Rendered {now} at commit `{commit}` by `scripts/check_conformance.py`. "
        f"{len(rows)} rows.",
        "",
        "| Status | Rows | Meaning |",
        "|---|---:|---|",
        f"| CONFORMS | {counts.get('CONFORMS', 0)} | demonstrated by TEST or MEASUREMENT |",
        f"| PLAUSIBLE | {counts.get('PLAUSIBLE', 0)} | code appears to implement it; nothing demonstrates it |",
        f"| VIOLATES | {counts.get('VIOLATES', 0)} | implemented contrary to the claim |",
        f"| ABSENT | {counts.get('ABSENT', 0)} | not implemented |",
        f"| UNVERIFIABLE | {counts.get('UNVERIFIABLE', 0)} | not checkable on this substrate |",
        f"| N/A | {counts.get('N/A', 0)} | waived by plan §4 |",
        "",
        "Evidence classes: "
        + ", ".join(f"{k} {v}" for k, v in sorted(ev_counts.items())),
        "",
        "**Reading order: VIOLATES first.** A PLAUSIBLE row is an open question, "
        "not a pass — it converts to CONFORMS only when a test or measurement "
        "lands (BUILD_PROMPT.md §4.2).",
        "",
    ]

    for status in order:
        group = [r for r in rows if r["status"] == status]
        if not group:
            continue
        lines += [f"## {status} ({len(group)})", "",
                  "| id | claim | evidence | closes | phase |",
                  "|---|---|---|---|---:|"]
        for r in sorted(group, key=lambda r: r["id"]):
            closes = ", ".join(r.get("closes") or []) or "—"
            ev = f"{r['evidence_class']}: {r['evidence_ref']}"
            if len(ev) > 110:
                ev = ev[:107] + "…"
            claim = r["claim"] if len(r["claim"]) <= 130 else r["claim"][:127] + "…"
            lines.append(f"| `{r['id']}` | {claim} | {ev} | {closes} | {r['phase']} |")
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true", help="validate only")
    args = ap.parse_args()

    with open(LEDGER) as f:
        doc = yaml.safe_load(f)
    rows = doc["rows"] if isinstance(doc, dict) else doc

    errors = validate(rows)
    if errors:
        print(f"conformance.yaml: {len(errors)} integrity error(s)", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        return 1
    print(f"conformance.yaml: {len(rows)} rows, integrity OK")

    if not args.validate:
        tmp = RENDERED + ".tmp"
        with open(tmp, "w") as f:
            f.write(render(rows))
        os.replace(tmp, RENDERED)
        print(f"wrote {RENDERED}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
