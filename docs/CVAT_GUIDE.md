# CVAT guide — DhakaScenes pilot review

Server: **http://100.123.243.18:8081** (Zamiul's tailnet view of `sim`; also
`localhost:8081` on the box). Login `mt`.

## What exists

Machine output and human truth live in **separate projects**, so the first
question — *whose annotations am I looking at?* — is answered before a task is
even opened:

| project | tasks | content | origin |
|---|---|---|---|
| **OUR PIPELINE — machine pre-annotations (2D)** | `<scene> — OUR PIPELINE output` | mask **polygons** + suppressed-duplicate **rectangles**, 6 cams/keyframe, distinct color per class | our pipeline (Stage 3 boxes → Stage 4 masks) |
| **nuScenes GT — HUMAN answer key (2D)** | `<scene> — nuScenes HUMAN answer key` | 2D **rectangles** projected from the human 3D ground truth, **every label the same green** | NOT our pipeline — the dataset's annotators |

The uniform green is deliberate: human truth speaks in one visual voice, so a
screenshot or a second browser tab is instantly attributable. Multicolored
shapes = machine, green rectangles = human. (Any 3D tasks follow the same
suffix convention when present; they are published by a separate exporter.)

Within each project the task **suffix** still carries provenance and the
publish step keys off it: `— OUR PIPELINE output` tasks are deleted and
recreated on every `scripts/run_stages.sh cvat`; `— nuScenes HUMAN answer key`
tasks are never touched. Task *ids* shift after each republish — navigate by
project, then name.

Frame order is identical everywhere (keyframes chronological, cameras
alphabetical), so frame *N* in a pre-annotated task is the same photo as frame
*N* in its GT twin.

## Opening a job (the annotation screen)

Projects → the project → pick a task → scroll to **Jobs** → click **Job #…**.
The screen: canvas in the middle, **Objects** sidebar on the right, a vertical
toolbar on the left, playback controls on top.

Navigation: `F` next frame, `D` previous frame, `Space` play, or type a frame
number in the top bar. `Ctrl+S` saves (edits live in CVAT's database, the
pipeline's files are never touched).

## Seeing boxes vs segmentation

Both live in the same job; what you see is controlled from the right side:

- **Appearance panel** (right sidebar, top): drag **Opacity** up to see masks
  as filled regions, down to outlines only. "Selected opacity" highlights the
  shape under your cursor.
- **Objects tab**: every shape on this frame, one row each — click a row to
  jump to it, eye icon to hide one shape, lock icon to protect it.
- **Labels tab** (next to Objects): eye icon per class — hide "a traffic cone"
  everywhere, show only "an articulated bus", etc.
- Click any shape: its **attributes** show in the sidebar — `score` is the
  detector confidence; `suppressed = true` marks a cross-camera duplicate the
  Stage 4 contest removed (drawn as a rectangle, no polygon).

Filters (funnel icon, top bar) accept expressions:

    attr["score"] == 0        -> only human-drawn shapes (yours)
    attr["suppressed"] == true -> only the contest's removed duplicates
    label == "a police car"    -> one class

## Comparing pipeline vs ground truth

Open the scene's two tasks in two browser tabs (e.g. task #1 and #11), put
them side by side, and step both with `F`. Same photo, pipeline's opinion left,
human GT right. Expect the GT tab to be much denser — nuScenes labels to full
range and through occlusion; our under-tier detector catches the near-obvious
slice (36.9% GT coverage, `work/metrics/paint_metrics.json`).

## The 3D view (task #21)

Open its job. The layout: a **perspective view** (drag to orbit, wheel to
zoom) plus **Top / Side / Front** ortho panels, and the six camera photos as a
strip — click one to enlarge. `F`/`D` steps keyframes just like 2D.

The cloud is the ego-frame ground-filtered single sweep — the exact points
Stage 5 painted. There are no boxes in it yet: cuboids arrive when Stage 6
lands. You can draw one by hand (cuboid tool in the left toolbar, then drag in
the ortho panels to fit) to get a feel for the eventual QA workflow.

For a *colored* 3D result today (points tinted by predicted instance), use the
static BEV renders instead: `work/viz/<scene>/<token>/bev.png`.

## Editing and exporting

- Fix a mask: click it, drag vertices; `N` starts a new shape of the last type.
- Your edits vs pipeline shapes: pipeline shapes carry `score > 0`; anything
  you draw has default `score = 0` (the filter above). In exports
  ("CVAT for images 1.1" XML) each shape carries `source="file"` (imported) or
  `source="manual"` (human).
- Export: task → Actions → **Export task dataset** → pick a format.

**Provenance rule (register C13):** anything exported from CVAT after human
editing is `human_verified` data. The pilot runs `allow_human_provenance:
false`, so those exports are for review and comparison — feeding them back
into the pipeline is a decision to record first, not a default.

## Fixing low-IoU boxes with SAM 3.1 clicks

For the boxes that need geometry repaired (best IoU vs GT in 0.3–0.5 by
default), `scripts/review_fix_sam31.py` offers the same review narrowed to
those boxes, with click-prompted SAM 3.1 re-segmentation instead of vertex
dragging, and publishes the result as a separate CVAT project. See
[`SAM31_REVIEW.md`](SAM31_REVIEW.md).

## Regenerating

The normal path is `scripts/run_stages.sh cvat`, which publishes both projects
with the right names, suffixes and colors. By hand, the equivalent is:

```bash
# 2D pre-annotations (after a new pipeline run)
ano_pipe python scripts/export_cvat_coco.py
CVAT_PASSWORD=... cvat_sdk_env/bin/python scripts/cvat_setup.py --host http://localhost:8081 --user mt \
    --project "OUR PIPELINE — machine pre-annotations (2D)" \
    --task-suffix "— OUR PIPELINE output" --replace

# GT twins (uniform green: the answer key's visual voice)
ano_pipe python scripts/export_gt_coco.py
CVAT_PASSWORD=... cvat_sdk_env/bin/python scripts/cvat_setup.py --host http://localhost:8081 --user mt \
    --project "nuScenes GT — HUMAN answer key (2D)" --label-color "#2ecc71" \
    --export-dir cvat_export_gt --task-suffix "— nuScenes HUMAN answer key"

# another scene in 3D
ano_pipe python scripts/export_cvat_3d.py --scene scene-0553
```

(`ano_pipe` = `PYTHONNOUSERSITE=1 /home/mt/miniconda3/envs/ano_pipe/bin/python`;
`cvat_sdk_env` = `/home/mt/dhakascenes/cache/cvat_sdk_env`.)
Existing tasks are skipped, so re-runs only add what's missing.
