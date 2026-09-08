# Stage 3m crashed with `KeyError: 15` — C34 x C36, a reporting bug

Date: 2026-09-08
Chunk: `dhaka_20260905_174950_chunk_0000` (`configs/paths_b_fused_chunk_0000.yaml`,
substrate `dhaka6`, work root `/home/mt/dhakascenes/work_b/chunk_0000`)
Crashing run: `logs/run_20260908_112225.log` — the chain aborted at 3m after
Stages 0/1/3/3f had completed (~11 min of GPU work), with no marker written.

## Root cause

`merge_rows()` runs three passes: the C34 veto (a confident PROTECTED arm A
box removes every arm B box contesting it, recorded in `vetoed`), the C28
class-pair arbitration, then the C36 part suppression (an arm A
bicycle/motorcycle whose own area is >= 60 % covered by a SURVIVING arm B
rickshaw is that rickshaw's wheel, recorded in `absorbed`). `pos_a` is built
only over `keep_a` — the arm A boxes that survive both `suppressed` and
`absorbed` — so `pos_a[i]` in the `suppressed_arm_b` block assumed every
protector survives.

It does not. A confident bicycle can veto the arm B rickshaw it *overlaps*
(IoU > 0.5) and then be absorbed by a **different, larger** rickshaw that
never contested it, because C36 measures containment and C34/C28 measure IoU:

| frame | protector | vetoed arm B box | absorbing arm B box |
|---|---|---|---|
| CAM_BACK `a2cc8778cb09e6152d6ac302e1c88d8e` | arm A **15**, `a bicycle` 0.517 | j=0 `a rickshaw` 0.853, IoU **0.536** | j=1 `a rickshaw`, containment **0.977**, IoU 0.076 |
| CAM_LEFT `2d4428ad14a52caad479138cfe2a1434` | arm A **4**, `a bicycle` 0.827 | j=0 `a rickshaw` 0.630, IoU **0.503** | j=2 `a rickshaw`, containment **0.631**, IoU 0.221 |

The bicycle leaves `keep_a` (and `pos_a`) while `vetoed` still names it →
`KeyError: 15` on the first such row, in 0 s. **2 rows in 8,184** hit it.

### Is C36 absorption the only path?

Yes, and the fix relies on it. The C28 pass opens with
`if _is_protected(i): continue`, so a protector can never enter `suppressed`;
C36 is the only pass that can remove an arm A box that already vetoed
something. `_protector_of()` raises `MergeContractError` rather than emitting
a silent `None` if a protector ever leaves by a third route — the invariant is
asserted, not assumed.

### Sibling blocks — checked, no change needed

* `suppressed_parts` → `pos_b[j]`: `j` comes from `authorities`, a subset of
  `keep_b`, and **no pass after the C34 veto removes an arm B box** (C36
  removes arm A boxes only, and it runs last). An absorbing rickshaw cannot
  be removed later. Safe.
* `suppressed_arm_a` → `pos_b[j]`: `j` wins an `argmax` over
  `np.where(live, iou[i], -1.0)` and is kept only when `best > iou_threshold`;
  vetoed boxes score -1.0, so for any threshold in the documented [0, 1] they
  can never be selected. `live` is exactly `keep_b`. Safe.

Both now carry a one-line comment saying *why* the lookup cannot fail, so the
asymmetry with `protected_by` reads as deliberate.

## The reporting decision

The entry **stays** in `suppressed_arm_b`. The veto really happened and the
arm B box really is gone from the arrays; dropping the entry (or counting it
only in aggregate) would erase the sole record of a removed box. What changes
is that the pointer stops lying:

| key | before | after |
|---|---|---|
| `protected_index_in_arm_a` | — | **new**: the protector's ORIGINAL arm A index, always present |
| `protected_by` | `pos_a[i]` (crashed) | merged index, or `null` when the protector is not in the output |
| `protected_survived` | — | **new**: bool |
| `protected_removed_by` | — | **new**: `null`, or `{reason: "absorbed_as_part", absorbed_by: <merged index>, absorbed_class_name, overlap}` |

`null` alone would have been the defensive `pos_a.get(i)` papering-over; it is
honest only because it always travels with `protected_survived: false` and a
record naming what removed the protector and where that absorber sits in the
merged arrays. A consumer can now tell "protected by a box that is in the
output" from "protected by a box that was itself removed" — and follow the
second to the box that replaced it.

Row block and manifest totals gain `n_protected_arm_a_removed`: of the
`n_protected_arm_a` boxes that exercised a veto, how many are not in the
output. `run()` prints a second summary line when it is non-zero. Every other
count keeps its meaning: `n_protected_arm_a` and `n_suppressed_arm_b` count
arbitration EVENTS, which are unchanged.

**No merged box changed.** The diff touches the ledger construction, the two
new counters and comments only; `keep_a` / `keep_b` and every suppression
decision are byte-for-byte the code that was there.

## Tests

TDD: the repro landed first and failed with the same `KeyError` at
`merge.py:349`, from synthetic rows carrying the real chunk_0000 geometry.

* `tests/test_stage3_merge_parts.py::TestProtectorRemovedAsPart` (3)
  — the crash state end to end through `merge_rows` (boxes unchanged, ledger
  honest, `absorbed_by` resolves to a real merged rickshaw); a surviving
  protector whose arm A index (1) and merged index (0) differ, proving the two
  keys are distinct quantities; and the invariant that a protector is never
  suppressed by the C28 table.
* `tests/test_stage3_merge.py::TestDriverAbsorbedProtector` (1) — the same
  state through `main()`: rc 0, the tree written, and
  `totals.n_protected_arm_a_removed == 1` in the manifest.

Suite: **606 passed, 1 failed** under `DHAKASCENES_SUBSTRATE=dhaka6` (603 + 4
new). The failure is the PRE-EXISTING, unrelated
`tests/test_stage1_thin_stereo.py::test_legacy_config_lets_every_ring_vote`
(`assert [0, 1, 2, 3, 101] == []` on `IngestConfig.ground_fit_rings`); neither
that test nor `pipeline/stage1_ingestion/` is touched by this change
(`git diff --name-only` = merge.py + the two merge test files).

## Re-run on the live chunk

```
DHAKASCENES_PATHS_CONFIG=configs/paths_b_fused_chunk_0000.yaml \
DHAKASCENES_SUBSTRATE=dhaka6 bash scripts/run_stages.sh 3m \
  --scenes dhaka_20260905_174950_chunk_0000
```
(the merge CLI itself takes no `--scenes` — it pairs whole trees or refuses;
the driver supplies `--arm-a-dir/--arm-b-dir/--out-dir/--taxonomy` and
`--accept-degraded-upstream`, the latter because the upstream markers on disk
are degraded.)

**Completed in 1 s, DEGRADED (inherited, by design), marker written.** Stages
0/1/3/3f were not re-run and nothing was deleted.

```
stage3_merge: 8184 rows, 69230 boxes out, 1883 arm A suppressed, 73 kept-both,
0 out-of-table overlaps, 386 arm B dropped under 390 protected arm A boxes
stage3_merge: 2 of those protected arm A boxes were themselves absorbed as
rickshaw parts (C36) and are not in the output; their suppressed_arm_b entries
carry protected_survived false
```

`stage3_merged/run_manifest.json` totals:

| key | value |
|---|---|
| `n_rows` | 8184 |
| `n_arm_a_in` / `n_arm_b_in` | 66542 / 6136 |
| `n_out` | 69230 |
| `n_suppressed_arm_a` | 1883 |
| **`n_suppressed_arm_b`** | **386** |
| **`n_suppressed_parts`** | **1179** |
| **`n_protected_arm_a`** | **390** |
| `n_protected_arm_a_removed` | 2 |
| `n_kept_both` / `n_overlap_out_of_table` | 73 / 0 |

The two affected rows now read, e.g. (CAM_BACK):
`protected_by: null`, `protected_survived: false`,
`protected_index_in_arm_a: 15`,
`protected_removed_by: {reason: absorbed_as_part, absorbed_by: 19,
absorbed_class_name: "a rickshaw", overlap: 0.9767}` — and merged index 19 is
indeed the arm_b rickshaw `[303.18, 372.17, 566.75, 720.0]`.

The chain can resume from Stage 4.

## Note for the operator

`docs/DECISIONS.md` stops at C34 and `docs/RUNNING.md` has no `part_floor`
row: **C35 and C36 were never written up**. This fix follows C36's precedent
(code comment + tests) rather than inventing a C37 entry against a numbering
you may already have spent. Worth back-filling C36 (and this) when the chain
is not blocked.
