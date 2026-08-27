# SAM 3.1 click-fix review — repairing low-IoU boxes by clicking

> **Frames mode** (`python -m scripts.review_fix_sam31 frames`) is the
> checker-gate variant of this tool: every kept box over every keyframe image,
> navigated frame-by-frame (`F`/`D`), `Enter` attests a whole frame as correct.
> It exists because the production DhakaScenes case has no GT to build an IoU
> queue from — the checker is the truth source. Same ledger, same export, same
> publisher; frame attestations are recorded as `frame_ok` records and skipped
> by `export`. It can also ADD objects the pipeline missed (drag a box, SAM
> segments it, `A` saves — `source: human`, `review: human_added`) and TRACK a
> saved object across its keyframe chain with `P` (SAM 3.1 video propagation;
> per-frame `add` records, `review: human_tracked`, `hops` = keyframe distance
> from the seed, each copy individually reviewable). Keymap and colours:
> `docs/REVIEW_HANDBOOK.md` §C′. The IoU-band queue below remains for
> GT-targeted passes.

`scripts/review_fix_sam31.py` is the CVAT review loop (`CVAT_GUIDE.md`)
narrowed to the boxes that need a human, with SAM 3.1 point prompts instead of
vertex dragging. It runs locally (stdlib HTTP server + the `sam31_example/`
click client), needs no CVAT server, and writes a COCO export that the existing
CVAT publisher and `eval_2d.py` both consume.

## The three steps

```bash
ano_pipe=/home/mt/miniconda3/envs/ano_pipe/bin/python     # has `sam3` + torch
set -a && . ./.env && set +a                             # HF_HOME -> pinned sam3.1 checkpoint

# 1. which boxes? kept Stage 3/4 boxes whose best IoU vs the GT twin is in [0.3, 0.5)
PYTHONNOUSERSITE=1 $ano_pipe -m scripts.review_fix_sam31 queue            # prints per-scene counts
PYTHONNOUSERSITE=1 $ano_pipe -m scripts.review_fix_sam31 queue --iou-min 0 # ...including unmatched boxes

# 2. fix them in the browser (http://localhost:8765; --host 0.0.0.0 to reach it over the tailnet)
PYTHONNOUSERSITE=1 $ano_pipe -m scripts.review_fix_sam31 serve [--scenes scene-0061] [--iou-min 0.3 --iou-max 0.5]

# 3. fold the decisions into a COCO export, then score / publish it
PYTHONNOUSERSITE=1 $ano_pipe -m scripts.review_fix_sam31 export
PYTHONNOUSERSITE=1 $ano_pipe -m scripts.eval_2d --pred-export cvat_export_fixed
CVAT_PASSWORD=... cvat_sdk_env/bin/python scripts/cvat_setup.py --host http://localhost:8081 --user mt \
    --export-dir cvat_export_fixed --project "OUR PIPELINE — SAM 3.1 click-fixed (2D)" \
    --task-suffix "— SAM 3.1 click-fixed"
```

The queue is built with `eval_2d.py`'s own dedup and IoU code, so "IoU" here is
the same number the metric reports. On the current run: 1 585 of 10 150 kept
boxes fall in [0.3, 0.5); 3 767 fall below 0.5 (809 of those overlap no GT box).

## The page

Left: the queue (filter by scene / status / class). Centre: the image, zoomed
onto the current box. Bottom: IoU before, and after each re-segmentation.

| colour | meaning |
|---|---|
| red | the machine box (Stage 3 proposal) under review |
| green | the best-overlapping nuScenes GT box (amodal: projected cuboid, occluded extent included) |
| cyan | SAM 3.1 mask from the current prompts |
| yellow dashed | tight box of that mask — what "accept fix" stores |
| faint white / green | the other predictions / GT boxes on this image |

Prompts: **left click** = object, **right click** or shift-click = background,
**Z** undoes the last click, **R** clears. "Seed from box" (**B**) sends the
existing box as SAM-2's corner-pair prompt together with the clicks — on load
every box is re-segmented that way, so a loose box often needs zero clicks.
Every refinement sends the FULL click set (replace semantics, like an
annotation UI), and switching boxes resets the session so nothing from the
previous object conditions the next mask.

Decisions: **A** accept fix · **K** keep as is (the box is right, the GT is
just amodal) · **D** delete (false positive) · **S** skip. **N** jumps to the
next pending box; **F** toggles auto-zoom; wheel zooms; ctrl/middle-drag pans.

Speed: the 3.3 GB model loads once (~10 s, on the first refine); switching to
another (scene, camera) chain costs ~1 s (frames are symlinked from the
read-only dataroot under `work/review_sam31/chains/`); a refine is 20–400 ms.

## What gets stored

`<work_root>/review_sam31/fixes.jsonl` — append-only, one record per decision,
**last record per box wins**; `spec: dhakascenes-pilot/sam31-click-fix/v1`.
Each record carries the box identity (`scene`, `file_name`, `channel`,
`keyframe_token`, `proposal_index`, COCO `ann_id`), `bbox_before_xyxy`,
`iou_before`, the GT box, and for a fix: the `clicks`, whether the box seed was
used, `bbox_after_xyxy`, the polygon `segmentation`, `mask_area_px`,
`iou_after`, a `mask_png` (`review_sam31/masks/<scene>/<ann_id>.png`, 1600×900,
ready for a Stage 5 re-lift), and `provenance` = `{tool: sam31_click, human:
true, checkpoint_sha256}`.

`<work_root>/cvat_export_fixed/<scene>/instances.json` — `cvat_export/` with
the decisions applied: fixed boxes get the new bbox on the rectangle AND the
new polygon on its mask twin; deleted boxes lose both rows; human-added and
SAM-tracked objects are appended as fresh rect+polygon pairs (`source: human`,
`track_id` = the seed's ledger id, `hops` = keyframe distance from the seed);
every annotation gains `review` (`none` / `sam31_fixed` / `kept_by_reviewer` /
`human_added` / `human_tracked`), `iou_before`, `iou_after`. Those attributes
are declared in `info.cvat_label_attributes` and `cvat_setup.py` adds them to a
**new** project's label schema — the existing pipeline and answer-key projects
are untouched (they also predate the `human` source value and the two new
review values, so publishing into them needs `--accept-missing-attributes`).
Ledger provenance is honest about who made each mask: hand-drawn adds carry
`human: true`; SAM-propagated copies carry `human: false, human_initiated:
true` until the frame they landed on is re-attested — tracking into an
attested frame automatically reopens it (`frame_pending`).

`eval_2d.py --pred-export cvat_export_fixed` writes
`metrics/detect2d_metrics.cvat_export_fixed.json` next to the baseline file, never
over it.

## Provenance rule (C13)

Everything that leaves this tool is human-edited. The fixed export is a separate
directory with a separate CVAT project name; nothing here writes into a stage
tree, `cvat_export/`, or the answer-key project. Feeding the fixed masks back
into Stages 5–9 (`allow_human_provenance`) is a decision to record in
`DECISIONS.md` first.
