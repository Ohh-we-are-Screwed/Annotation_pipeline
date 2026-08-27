# Review handbook — operating CVAT and the SAM 3.1 click-fix page

Two tools, one workflow. CVAT shows you what the pipeline produced; the click-fix
page is where you repair it. They are separate programs on separate ports, and
almost every point of confusion comes from expecting one to behave like the other.

Companion to [`CVAT_GUIDE.md`](CVAT_GUIDE.md) (what exists in CVAT and why) and
[`SAM31_REVIEW.md`](SAM31_REVIEW.md) (what the click-fix tool stores). This file
is the operator's view: which window am I in, what do I press, why does nothing
seem to improve.

---

## A. Which tool am I in?

Start here. The controls below only make sense once you know which window you
are looking at.

| | **CVAT** | **SAM 3.1 click-fix** |
|---|---|---|
| address | `100.123.243.18:8081` | `100.123.243.18:8765` |
| what it is | viewer + comparison surface | the actual segmentation tool |
| you navigate | **frames** — one photo at a time | **boxes** — a queue, not a filmstrip |
| how you edit | drag polygon vertices by hand | click the object, SAM re-segments (~230 ms) |
| saving | `Ctrl+S`, or the Save button | **no save button** — every decision writes instantly |
| use it to | inspect, compare against GT, spot-check | repair the 1 585 loosely-wrong boxes |

### The single biggest source of confusion

In the click-fix page there is **no such thing as "the next image."** You move
through a **queue of boxes**. When the next box happens to sit on a different
photo, the photo changes by itself. That is why looking for frame controls there
leads nowhere — press `N` for the next box that still needs a decision, and the
picture follows.

---

## B. CVAT — the three things that trip you up

### The polygons are invisible

They are being drawn; they are just fully transparent. Right sidebar → scroll to
**Appearance** → drag **Opacity** up from zero to ~50%. The masks fill in
immediately.

**Selected opacity** is a *separate* slider that only affects the shape currently
under your cursor — which is why hovering seemed to work while everything else
stayed hollow.

### Moving between photos

| key | does |
|---|---|
| `F` | next frame |
| `D` | previous frame |
| `Space` | play / pause through the frames |
| frame box, top bar | type a number, Enter to jump straight there |

Frame order is identical across every task in the project (keyframes
chronological, cameras alphabetical), so frame 67 in one task is the same photo
as frame 67 in its GT twin. That is what makes side-by-side tabs work.

### Saving

`Ctrl+S`, or the **Save** button top-left next to Menu. Nothing you do in CVAT
touches the pipeline's files — edits live in CVAT's own database.

### Finding the repaired boxes

Click-fix results are published into CVAT with attributes on every shape. Funnel
icon, top bar:

```
attr["review"] == "sam31_fixed"       the boxes SAM repaired
attr["review"] == "kept_by_reviewer"  ones a human confirmed were already right
attr["score"] == 0                    shapes you drew yourself
```

Click any shape and its `iou_before` / `iou_after` show in the sidebar — that is
the repair, recorded.

---

## C′. Frames mode — the checker gate (use this one)

`frames` is the workflow the pilot is actually rehearsing: a **human checker
between Stage 4 (2D masks) and Stage 5 (the 3D lift)**. You flip through whole
images fast, everything drawn at once, and touch only what is wrong. Moving on
with `Enter` *attests the frame* — untouched boxes on an OK'd frame are
**approved**, not merely unseen, and that attestation is recorded in the ledger.

```bash
$ano_pipe -m scripts.review_fix_sam31 frames --host 0.0.0.0 --port 8765
```

Unlike the band queue it shows **every kept box** (10 150 over 2 424 frames, not
an IoU-selected 1 585) — because in production DhakaScenes there is no GT to
select by; the checker is the truth source. GT overlay is therefore **off** by
default (`T` shows it for the pilot), and box-seed is **off** by default.

| key | does |
|---|---|
| `F` / `→` | next image (CVAT muscle memory) |
| `D` / `←` | previous image |
| `Enter` | **frame is fine → record + next image** (the fast path) |
| `Shift+Enter` | un-mark a frame attested by mistake |
| click a box | select it (smallest box under the cursor wins); auto-zooms |
| `Tab` | cycle through the boxes on this frame |
| `Esc` | deselect, zoom back out |
| left / right click (selected) | SAM prompt: object / background |
| `A` `K` `X` `S` | accept fix · keep · **delete (X, not D — D is prev-frame)** · skip |
| `Z` `R` `B` | undo click · clear clicks · toggle box seed |
| `G` / `P` | **track the selected object across the clip** (while drawing, `G` re-segments) |
| `T` `M` `V` | toggle GT · toggle mask polygons · zoom-to-selection |

### Adding an object the pipeline missed

**Drag a rectangle on empty canvas** around the object — SAM segments inside it
immediately, no clicks needed. The cyan mask and yellow tight box appear; add
left/right clicks to refine if the mask is off. A **class palette pops up next
to the box**: the AI's guess (SigLIP zero-shot on the crop — measured 73% top-1
/ 87% top-3 on this substrate, 9 ms) is pre-selected with its confidence shown;
**press a number key `1`–`9`/`0`** (or click a row) to override. Then **`A` or
`Enter` saves it**. `Esc` throws the draft away. While composing, frame
navigation is locked so a half-drawn object can't be lost by accident.

The record keeps the class's provenance: `category_source` says whether the
label was `ai_suggested` (you accepted the guess) or `human_picked` (you chose),
and the AI's top-3 is stored alongside for audit.

Saved objects draw in **purple**. Click one to select it; `X` removes it (a
retraction record — the ledger keeps both, last wins). They export as normal
rect + polygon rows with `source = human`, `review = human_added`, `score = 0`,
so CVAT, `eval_2d`, and Stage 5 all see them like any other annotation — with
provenance that says a person drew them.

### Fixing a half-annotated box, then tracking it

A truncated pipeline box (car half in frame, box covering only part of it) is
the fix flow: click the box, click the object, `A`. The accepted geometry
**replaces the stale box on screen immediately** and survives restarts, the box
stays selected, and **`G` now tracks it** — the repaired object propagates
across the chain exactly like a drawn one. Its copies land as dashed purple
additions on the other frames; removing the track (`Shift+X`) never removes the
fixed box itself, which is a real pipeline annotation.

### Tracking an added object across the clip (`G`)

SAM 3.1 is a video model, and each camera's keyframes form one chain — so after
saving an object it stays selected, and **`G` (or `P`) propagates it across
the whole chain** (both directions, ~5 s for 39 keyframes). Every frame that keeps a
confident, non-empty mask becomes its own annotation: **dashed purple**,
`review = human_tracked`, `track_id` = the seed's id, `hops` = keyframe distance
from the seed. Flip through with `F`/`D` to review them. A copy whose mask is wrong is
**repairable**: with it selected, click the object — SAM re-segments — and `A`
saves the repair (the record becomes human-verified). `X` on a dashed copy
removes just that copy, `X` on the solid seed (or `Shift+X` anywhere) removes
the whole track. Re-pressing `G` on the seed re-tracks and replaces the
previous propagation — no duplicates. Tracking into a frame you had already
marked OK **reopens that frame** — its ✓ disappears, because it no longer shows
what you attested. Propagated masks are recorded `human: false` in the ledger
until their frame is re-attested; only the hand-drawn seed is `human: true`.

**Review the propagated frames — do not trust them blind.** Keyframes are 0.5 s
apart, and measurement on this substrate showed nearby frames track cleanly
while frames many seconds away can drift onto a different object *while still
reporting confidence 1.0*. The far end of a track is exactly where your eyes are
needed; empty-mask frames (object left the view) are dropped automatically.

Box colours on the frame: **white** pending · **yellow** fixed · **blue** kept ·
**dark red ✕** deleted · **purple** human-added · **bold red** currently
selected. Stage 4 polygons are drawn under their boxes so mask quality is
judged without opening CVAT.

Everything lands in the same `fixes.jsonl`, so `export`, `eval_2d`, and the CVAT
publish work unchanged; frame attestations ride along as `frame_ok` records
(and are skipped by `export`). The band-queue mode (`serve`) still exists for
GT-targeted passes.

---

## C. The click-fix page, end to end

Left sidebar is the queue. Middle is the image, zoomed to the box under review.
Top bar is every control. Bottom strip is the colour legend and the live IoU.

### What the colours mean

| colour | meaning |
|---|---|
| **red** | the machine box — the thing you are deciding about |
| **green** | nuScenes GT. Amodal: includes hidden parts of the object |
| **cyan** | SAM's mask from your current clicks |
| **yellow dashed** | tight box of that mask — what "accept fix" stores |
| faint white | the other predictions on this photo (context, not your problem) |

### Clicking

| input | does |
|---|---|
| **left click** | "this pixel **is** the object" — re-segments immediately |
| **right click** or `Shift`+click | "this pixel is **background**" — carves away over-reach |
| `Z` | undo the last click and re-segment |
| `R` | clear all clicks, start this object over |
| `G` | re-segment from the current clicks |
| `B` | toggle "seed from box" — **see the warning below** |
| wheel | zoom |
| `Ctrl`-drag or middle-drag | pan |
| `F` | toggle auto-zoom to the box |

> **Turn "seed from box" OFF before you start.**
>
> It is **checked by default**, and measurement says it hurts. Across 200 boxes,
> box-seeded re-segmentation made the result **worse 124 times and better only
> 75**. In **62 of 200** cases the box prompt so dominated that adding a click
> moved the mask by less than 0.001 IoU — the click was effectively ignored.
>
> Press `B` once, or untick it in the top bar. A bare click beat box-plus-click
> on every aggregate measured.

### Deciding — and why there is no Save

Each key writes a record to `fixes.jsonl`, flushes it, and jumps you to the next
pending box. Nothing to save, no way to lose work. There is no undo either — but
a later decision on the same box overrides the earlier one, so pressing the wrong
key is fixed by pressing the right one.

| key | decision | when |
|---|---|---|
| `A` | **accept fix** | store SAM's mask + tight box. Needs a mask first — click the object |
| `K` | **keep as is** | box was already right, GT just looks bigger. **Your most-used key** |
| `D` | **delete box** | there is no object here. Drops box and mask from the export |
| `S` | **skip** | undecided. Stays handled-but-unchanged |

### Getting around the queue

| key | does |
|---|---|
| `N` | **next box still needing a decision** — the one you want |
| `←` `→` | previous / next box in the list, decided or not |
| sidebar row | click to jump straight to any box |
| sidebar dropdowns | filter by scene, by status, or type a class substring |

---

## D. Why you will press `K` far more than `A`

The green GT box is **amodal** — a 3D cuboid projected onto the photo, covering
the whole object *including the parts hidden behind other cars*. SAM only ever
sees visible pixels. A perfect mask on a half-occluded car therefore scores a
*low* IoU against that green box, through no fault of the mask.

Measured across 200 queued boxes:

- SAM's mask sat **inside** the GT box almost always — containment median `1.000`
- while covering only about **44%** of its area
- the pipeline's original boxes had the same profile (containment `0.947`)
- in **137 of 200** cases no visible-only mask could reach IoU 0.5 — arithmetically
  out of reach

**At the keyboard:** roughly **61%** of this queue is correctly-segmented boxes
that only look wrong because of amodality — those are `K`. About **27%** are
genuine geometry repairs worth clicking — those are `A`. If you find yourself
pressing `K` constantly, the tool is not broken and neither are you; the queue's
selection rule is picking up occlusion rather than bad geometry.

---

## E. The full pipeline, start to finish

Four commands. The first three are ordered — the queue must exist before you can
fix it, and fixes must exist before they can be exported.

Prelude for every command:

```bash
cd /home/mt/Zami/Annotation_pipeline
set -a && . ./.env && set +a
ano_pipe="PYTHONNOUSERSITE=1 /home/mt/miniconda3/envs/ano_pipe/bin/python"
```

### 1. Build the queue

Finds every kept box whose best IoU against GT falls in the band. Prints
per-scene counts, writes `queue.json`.

```bash
$ano_pipe -m scripts.review_fix_sam31 queue
$ano_pipe -m scripts.review_fix_sam31 queue --iou-min 0   # include unmatched boxes
```

### 2. Open the page and click

Starts the local server. Leave the terminal open — this *is* the server. The
3.3 GB model loads on your first click (~10 s), once per session.

```bash
$ano_pipe -m scripts.review_fix_sam31 serve --host 0.0.0.0 --port 8765
```

Then open **http://100.123.243.18:8765/**. Use `--host 127.0.0.1` instead if you
are working directly on the machine and would rather not expose the port.

### 3. Fold the decisions into an export

Reads the ledger, writes `cvat_export_fixed/` — same COCO shape as the original
export, repaired boxes swapped in, deleted ones dropped.

```bash
$ano_pipe -m scripts.review_fix_sam31 export
```

### 4. Score it, then publish it back to CVAT

Scoring writes a metrics file *next to* the baseline, never over it.

```bash
$ano_pipe -m scripts.eval_2d --pred-export cvat_export_fixed

CVAT_PASSWORD=... /home/mt/dhakascenes/cache/cvat_sdk_env/bin/python \
    scripts/cvat_setup.py --host http://localhost:8081 --user mt \
    --export-dir cvat_export_fixed \
    --project "OUR PIPELINE — SAM 3.1 click-fixed (2D)" \
    --task-suffix "— SAM 3.1 click-fixed"
```

If a publish fails partway it can leave an **empty task** behind, and a re-run
will skip it by name rather than noticing it is broken. Add `--replace` to delete
and rebuild tasks carrying that suffix.

---

## F. When something is unreachable

Failures actually hit while setting this up:

| symptom | cause and fix |
|---|---|
| page won't load at the tailnet address | server bound to `127.0.0.1`, which accepts local connections only. Restart with `--host 0.0.0.0`. CVAT works because it binds that way |
| `Address already in use` | an older server still holds the port. `pkill -f review_fix_sam31`, then start again |
| CVAT `No such file or directory` on task creation | CVAT reads images from a share mounted into the container (`ResourceType.SHARE`), not from uploads. If the dataset moves, that mount breaks and no frames are visible — see the `cvat_share` device in `/home/mt/cvat/docker-compose.override.yml` |
| model reloads on every box | expected when crossing scene/camera boundaries — chains reload in ~0.7 s. Within one chain a refine is ~230 ms |
| `refine a mask first` | `A` pressed with no mask on screen. Click the object, then accept |
| stale `chains/` symlinks after a dataroot move | `chain_dir()` rebuilds only when the file *count* differs, so dangling links survive silently. `find <work>/review_sam31/chains -xtype l -delete` |

---

## Provenance (decision C13)

Everything leaving the click-fix page is human-edited data, recorded as such in
the ledger along with your clicks and the checkpoint hash. The fixed export is a
separate directory and a separate CVAT project; it never overwrites the
pipeline's own output or the answer key. Feeding those masks back into Stages 5–9
(`allow_human_provenance`) is a decision to record in `DECISIONS.md` first, not a
default.
