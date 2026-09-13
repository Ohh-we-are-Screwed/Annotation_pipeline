# Export status and decisions

Working notes for whoever picks this up next. Covers `export/`, the `release`
step of `scripts/run_stages.sh`, and the fused-capture queue. Written
2026-09-09. Companion to [`NUSCENES_EXPORT.md`](NUSCENES_EXPORT.md) (how to run
an export) and [`DECISIONS.md`](DECISIONS.md) (the contradiction register).

## 1. Where the chunks stand

The fused capture `dhaka_20260905_174950` is **11 scenes = 11 chunks**, one
nuScenes root holding all of them, selected per run with
`--scenes dhaka_20260905_174950_chunk_<NNNN>`.

| chunk | state |
|---|---|
| 0000, 0008, 0009, 0010 | complete, exported, published to CVAT |
| 0001 | running (queue pid started 16:25, Stage 3b) |
| 0002 – 0007 | queued behind 0001 in the same driver |

All 11 directories exist under `export/`. The seven pending ones hold **zero
files** (24 KB of empty dirs) — `scripts/prepare_run_exports.py` creates them at
queue launch and points the work tree at them:

```
work_b/chunk_0001/cvat_export -> export/dhaka_20260905_174950_chunk_0001/cvat_export
```

**Do not delete those empty directories while a queue is running.** Stages write
*through* those symlinks straight into the delivery folder; removing them breaks
the live run. They fill in as each chunk completes.

### Why 0001–0007 were late

Stage 0's `files_parse` predicate refused 0001/0002/0004/0005/0006/0007 in 3–5
seconds each (`HARD STOP: zero usable scenes`, rc=2, nothing written). Twelve
blobs in the capture do not parse — eleven CAM_FRONT_LEFT JPEGs (six with no SOI
marker, two truncated, one 6,680 bytes) and one LiDAR cloud truncated at
5,505,024 bytes. `files_parse` is all-or-nothing per scene and each chunk is one
scene, so twelve files refused ~8,900 keyframes. `scripts/fixup_unparseable.py`
produced `v1.0-dhaka-fixed3`, which drops those twelve keyframes from the tables
(nothing is repaired — the corrupt blobs are simply no longer named). 0003 never
failed; it was interrupted mid-Stage-3b when its queue was stopped, and its
partial tree was cleared before the requeue.

## 2. What changed on 2026-09-09

### 2.1 `road` moved into the default step list — RESOLVED

```bash
# before
ALL_STEPS=(0 1 3 4 5 6 7 8 release eval viz cvat cvat3d)
OPT_IN_STEPS=(3b 3f 3m 3c road cvatroad)
# after
ALL_STEPS=(0 1 3 4 5 6 7 8 road release eval viz cvat cvat3d)
OPT_IN_STEPS=(3b 3f 3m 3c cvatroad)
```

`road` was opt-in, so a default run — or `all` — never produced `stage_road`,
and `release` then correctly skipped the road layer. That is why the fused
chunks shipped without the 3D segmentation the day-1 exports had.
`run_fused_chunks.py` only escaped it by passing `road` explicitly.

This **reverses** the documented rule that opt-in arms stay out of the default
chain ("a baseline run has to stay the one nobody had to ask for"). The override
is deliberate and is recorded in the script header: the release export is the
delivery contract, so the stage feeding it belongs in the chain. The other four
arms (`3b 3f 3m 3c`) are untouched and still opt-in. `road` sits **before**
`release` because release exports whatever `stage_road` left; a `road` after
`release` ships a delivery with no segmentation.

Verified: default and `all` both yield
`0 1 3 4 5 6 7 8 road release eval viz cvat cvat3d`; `--no-cvat` keeps `road`
and drops the publishes; explicit step lists are unchanged; 612 tests pass.

### 2.2 The `release` step now exports the extras — RESOLVED (earlier same day)

`release` writes three sibling layers: `boxes/` (fatal), `road/` (soft) and
`coco_2d/` (soft). The extras run **before** the boxes export because that
export writes `DELIVERY_NOTE.md` and the note names the layers beside it —
build them afterwards and every note describes a `road/` that did not exist when
it was written.

The extras stay `soft` on purpose: the boxes are the deliverable and a missing
extra must not cost a chunk its GPU hours. A skip is visible in the note rather
than silent. **Do not promote them to `fatal`.**

### 2.3 Frozen-runner drift — the trap to know about

`run_fused_chunks.py` copies `scripts/run_stages.sh` to a tempfile at queue
launch (`scripts/.run_stages_<rand>.sh`) so repo edits during a multi-day queue
cannot change a running chain — bash reads a long script incrementally.

Consequence: a fix landing mid-queue does **not** reach the running queue.
chunk_0010 finished at 16:23 under a snapshot taken at 01:39, before §2.2
landed, so it came out with no `road/` and no `coco_2d/` and needed a manual
backfill. **Never edit `scripts/.run_stages_*.sh` while a queue holds it.**

### 2.4 chunk_0010 backfilled

`road/` (497M, 146 keyframes, 13,713,410 road points) and `coco_2d/` (32M) built
directly, then `DELIVERY_NOTE.md` regenerated via `pipeline.release.note.write_note`
so it stops claiming the layers are absent. Only the note was rewritten;
`release_meta.json` and the nuScenes tables were left alone because nothing about
the boxes changed. All four finished chunks now have identical shape.

## 3. The segmentation was not loadable — RESOLVED 2026-09-09

**Outcome first:** the fold-into-`boxes/` plan was tried and is **impossible**;
`road/` was made a genuine standalone lidarseg root instead, and that works.
The four already-exported chunks still carry the OLD unscoped `road/` and need
a re-export (§3.3).

The four "complete" exports did **not** have usable 3D segmentation, despite
`road/` existing:

- `boxes/` — the root anyone loads — has no `lidarseg.json` and no `lidarseg/`
  directory. By the nuScenes-lidarseg spec that is a dataset with zero
  segmentation.
- `road/` is **not** a scoped copy of the chunk. It carries the whole capture's
  tables and no blobs:

  | table | `boxes/` (chunk_0010) | `road/` |
  |---|---|---|
  | sample | 146 | 14,966 (all 11 chunks) |
  | sample_data | 5,385 | 551,580 |
  | sample_annotation | 3,200 | 0 |
  | category | 18 Dhaka classes | 32 nuScenes classes (overlap: 1) |
  | lidarseg | absent | 146 |

  No `samples/`, no `sweeps/` (they are only symlinked under `--link-blobs`,
  which `run_stages.sh` does not pass).

So `boxes/` has the point clouds but no labels; `road/` has labels but no
clouds, no annotations and a different taxonomy. The two do not compose, and
nothing standard loads segmentation from this delivery.

**Second, independent blocker:** `map.json` is `[]` in *every* export, day-1
included, because the source dataroot's `map.json` is empty — the capture has no
map. `nuscenes-devkit` raises `IndexError` at `nuscenes.py:198`
(`__make_reverse_index__` does `self.map[0].keys()`) before it reaches
segmentation, so **no export in this repo currently loads with the devkit.**
The devkit needs one map record carrying `log_tokens`; it only reads the map
image if the map API is used.

### 3.1 The fold-in is impossible — do not retry it

The operator chose "fold lidarseg into `boxes/`, keep `road/` too, fix
`map.json`". The fold-in half **cannot be built**. `nuscenes.py:110-111` runs
only when `lidarseg.json` is present in the table root:

```python
self.colormap = dict({c['name']: self.colormap[c['name']]
                      for c in sorted(self.category, key=lambda k: k['index'])})
```

That forces every category row to carry an `index` AND every category **name**
to exist in the devkit's hard-coded 32-name `get_colormap()`. Seventeen of the
eighteen Dhaka box classes are absent from it (`animal` is the sole overlap).
Measured on devkit 1.2.0:

| `category.json` beside `lidarseg.json` | bare `NuScenes(version, dataroot)` |
|---|---|
| union: 32 lidarseg @ 0..31 + 18 box appended | `KeyError: 'car'` at nuscenes.py:110 |
| canonical 32 only (box rows dropped) | `KeyError: <token>` at nuscenes.py:228 |

No free variable is left: `index == position` is forced by the stats/render
APIs and by line 111's sort; the box tokens are forced by `sample_annotation`.
nuScenes-lidarseg's `category.json` is a closed vocabulary welded to the
devkit's colormap, and a box release with a custom taxonomy cannot share it.
**This is why `road/` was a separate root in the first place — that design was
correct.** (It does load if every consumer passes `colormap={**get_colormap(),
**dhaka}`, which is a per-consumer burden and absent on older devkits. Rejected.)

### 3.1a The object layer ships in `boxes/` anyway — devkit-INVISIBLE (2026-09-13)

The operator asked for per-point labels for OBJECTS, which `road/`'s canonical-32
taxonomy cannot express (`cycle_rickshaw` and `cng_autorickshaw` have no
canonical name), so `scripts/export_lidarseg.py` writes the layer into `boxes/`
after all. **The blocker above is unchanged and was re-measured on devkit
1.1.11 — it was WORKED AROUND, not solved.**

The trick is the AUTO-DETECTION, not the colormap. `NuScenes.__init__` looks for
the literal names `lidarseg.json` / `panoptic.json` in the table root; only if
it finds one does it reach line 110 and die on our class names. So the index
table is written as **`<version>/dhakascenes_lidarseg.json`** — identical schema,
identical rows, a name the devkit does not look for. The `index` fields and the
prepended `noise` row in `category.json` then ride along INERT
(`load_lidarseg_cat_name_mapping` never runs), and a bare
`NuScenes(version, dataroot=boxes)` loads exactly as it did before the layer
existed. That matters because `boxes/` is what the operator's BEVFusion data
conversion reads.

Measured on chunk 14 after the export:

```
(a) bare NuScenes('v1.0-dhaka-fixed2', dataroot=boxes, verbose=False): OK
    sample_annotation 9446 (unchanged), 19 category, hasattr(nusc,'lidarseg') == False
(b) recipe, in a THROWAWAY copy of the table dir:
    cp dhakascenes_lidarseg.json lidarseg.json  +  the get_colormap shim
    -> 2582 lidarseg records; get_sample_lidarseg_stats: 39,936 points, 380 labelled
```

Both halves of the recipe are required — the copy alone still `KeyError`s. It is
spelled out verbatim in the delivery note and in `lidarseg/lidarseg_meta.json`
(`devkit_recipe`), next to the no-devkit path (`read_without_the_devkit`: one
`numpy.fromfile`, `category.json`'s `index` for the names). Because a
`lidarseg.json` left in a DELIVERED root breaks the bare load for everyone else,
the exporter **refuses to run (exit 2) while one is present** rather than
writing beside it. `category.json.pre_lidarseg.bak` sits beside the rewritten
table if the layer needs undoing entirely.

`road/` stays a separate root; the two layers are the surface and the objects,
not rivals. Note for any future in-place table rewrite: `write_json_atomic`
creates through `mkstemp` and lands **0600**, which in these group-ACL'd
delivery folders is unreadable by anyone but the exporting user — the tables the
release itself wrote are 0660. `export_lidarseg.write_table` restores the mode
from a sibling table; `export_annotations_2d.write_json_compact` documents the
same trap.

### 3.2 What was built instead — two valid roots

`boxes/` for cuboids, `road/` for segmentation, each loading on its own.

1. **`map.json` synthesis** — `SourceRoot.__init__` in `scripts/export_release.py`.
   When the scoped map table is empty, ONE row is written binding every exported
   log: `{token, log_tokens, category: "semantic_prior", filename: ""}`. The
   capture has no map image; the row exists solely because the devkit
   dereferences `self.map[0]` before reading anything else. Recorded as
   `map_synthesised` in `release_meta.json` — never silent. `export_release()`
   also rewrites `map.json` explicitly, because the copy-through branch would
   otherwise ship the source root's empty table over the top of it.

2. **`road/` scoped and loadable** — `scripts/export_road_lidarseg.py` gains
   `--scenes` and now builds the eight tables `SourceRoot` owns (`scene`,
   `sample`, `sample_data`, `ego_pose`, `calibrated_sensor`, `sensor`, `log`,
   `map`) **through that same `SourceRoot`** rather than copying them
   byte-for-byte. One implementation of both the scoping filter and the map row,
   shared with the box exporter, so the two roots cannot drift. The five
   annotation tables still ride along byte-for-byte (this layer ships them
   empty, as always) and `category.json` is still replaced with the canonical 32.
   Labels narrow with the tables: a bin left in for a scoped-out keyframe would
   name a `sample_data` row the root no longer has.

3. **`run_stages.sh`** passes `${SCENE_ARGS[@]}` to the road export. One line.

Verified on real chunk_0010 data, not just fixtures:

```
LOADED   lidarseg=146  samples=146  sample_data=5385  categories=32
map row  filename=''   log->map linked: True
first bin: 383688 pts, labels {0: 295814, 24: 87874}
cat[24] = flat.driveable_surface index 24
```

Before the fix that same root reported 14,966 samples — the whole capture's
tables wrapped around one chunk's 146 labels. 617 tests pass.

### 3.3 Outstanding: the four exported chunks still have the old `road/`

0000, 0008, 0009 and 0010 carry the pre-fix `road/` — unscoped tables, empty
`map.json`, unloadable. They need the layer re-exported. This needs **no
pipeline re-run**: `export_road_lidarseg.py` reads existing `stage_road` output
(~1.5 s for chunk_0010). The exporter refuses a non-empty `<out>/<version>`, so
the old `road/` has to be moved aside or removed first — that is a delete
against a delivery folder and needs the operator's explicit say-so per item.
Chunks 0001-0007 come out correct from the queue and need nothing.

## 4. What the segmentation actually is

Road-surface only, in both day-1 and day-2: the `.bin` files carry exactly two
labels, `24 flat.driveable_surface` and `0` unlabelled. The other 30 categories
are present for devkit compatibility and have zero points. Roughly 19% of points
per cloud are road. This is **not** full semantic segmentation of objects, and
nothing labels pedestrians or vehicles at the point level — the pipeline
produces 3D *boxes* for those.

## 5. Delivery facts worth not rediscovering

- **No symlinks anywhere** in a finished export; blobs are **hardlinks** to the
  dataroot (`links=2` LiDAR, `links=3` cameras — dataroot + export + CVAT
  share). A hardlink is a second directory entry for the same inode, so deleting
  the dataroot does not harm the export.
- The hardlink saving is **local to one filesystem**. Copying an export off the
  box materialises real bytes: ~20G per full-size chunk, ~72G for the four
  finished, ~220G projected for all 11.
- `coco_2d/instances.json` uses paths relative to the nuScenes root
  (`samples/CAM_BACK/014637.jpg`), so it needs `boxes/` beside it. Upload whole
  chunk folders; do not split them. `instances.share.json` points at the local
  CVAT share and is meaningless off this machine.
- `cvat_export*` are review artefacts, not delivery. `cvat_export_3d_double` is
  empty in every chunk, so a sync that skips empty dirs will change the shape.
- `cvatroad` is still opt-in and is **not** in the default chain. The fused
  queue publishes the road layer only because `run_fused_chunks.py` passes
  `cvatroad` explicitly.
