# CVAT "Datumaro 3D 1.0" round trip: do tracks and custom attributes survive?

Spike for spec §7 (Tasks 14 and 16). Tasks 14/16 encode object IDENTITY and the
reviewer's ANSWERS in cuboid tracks and custom label attributes, so the encoding
had to be chosen from measurement rather than from the format description.

Measured 2026-09-08 against the operator's live CVAT with
`scripts/spike_cvat_3d_roundtrip.py` — a 3-frame task, 200 random points per
frame, no related images, labels `a car` / `a pedestrian`, three objects:

| object | `record_token` | how it went in |
|---|---|---|
| car | `k0:CAM_FRONT:0` | TRACK on frames 0,1,2 (`track_id: 7`, `keyframe: true`) |
| pedestrian | `k1:CAM_FRONT:3` | plain SHAPE on frame 1 (no `track_id`, no `keyframe`) — the control |
| pedestrian | `k2:CAM_FRONT:9` | SHORT TRACK on frames 0,1 only (`track_id: 9`, `keyframe: true`) |

Reproduce:

```bash
set -a; . ./.env; set +a
python scripts/spike_cvat_3d_roundtrip.py            # nothing is deleted; see "Artifacts"
```

---

## 1. Versions

| | |
|---|---|
| CVAT server (`GET /api/server/about`) | **2.72.1** |
| `cvat_sdk` (client, conda env `ano_pipe`) | 2.73.0 — prints a "not compatible with SDK version" warning against a 2.72.1 server; every call in this spike worked anyway |
| `datumaro` inside the `cvat/server:dev` image | **0.3** — this is the library that does the geometry rounding in §4 |

## 2. Tracks: yes, they round-trip — but the `track_id` VALUE does not

A cuboid imported with `track_id` + `keyframe` comes back **as a track, not as
three independent shapes**. Server-side (`GET /api/tasks/293/annotations`) the
car is one entry under `"tracks"` with three shapes; the control pedestrian is
the only entry under `"shapes"`:

```
[2] server-side: 2 track(s), 1 shape(s), 0 tag(s)
    track id=2 label_id=541 frames=[0, 1, 2]  track-level attributes=[{'spec_id': 1077, 'value': 'k0:CAM_FRONT:0'}, {'spec_id': 1078, 'value': '7'}]
    track id=3 label_id=542 frames=[0, 1, 2]  track-level attributes=[{'spec_id': 1082, 'value': 'k2:CAM_FRONT:9'}, {'spec_id': 1083, 'value': '9'}]
    shape id=1564987 label_id=542 frame=1
```

**How track membership is represented in the exported Datumaro JSON.** There is
no `tracks` container in the export. The dataset is flat — one `items` entry per
frame — and every shape of a track is written into its own frame's `annotations`
carrying two extra attributes:

* `"track_id": <int>` — the same integer on every frame of one track;
* `"keyframe": true`.

Shapes that are not part of a track carry **no `keyframe` key at all**. That
absence, not the value of `track_id`, is the reliable discriminator (see the
trap below).

**The exported `track_id` is CVAT's own per-task track index, not the value that
went in.** Car went in as `track_id: 7`, came back `0`; the short pedestrian
track went in as `9`, came back `1`. The exporter overwrites the attribute:
`bindings.py:2208` does `dm_attr["track_id"] = shape.track_id` *after*
`_convert_attrs()` has already put our declared value there, and
`bindings.py:541` sets `tracked_shape["track_id"] = idx` (a 0-based enumeration
of the task's tracks) because `use_server_track_ids` defaults to False. The
value is dense, 0-based, per task, and stable only within one export.

Our value is **not lost on the server** — CVAT stored `track_id = "7"` as a
track-level label attribute (spec_id 1078 above) and shows it in the UI — it is
lost only in the Datumaro export. A reader that needs the pipeline's own integer
can get it from `GET /api/tasks/{id}/annotations` (`tracks[].attributes`,
resolving `spec_id` through the project's labels), not from the dataset zip.

**Trap — `track_id` 0 is ambiguous in the export.** Because `track_id` is also
one of our declared label attributes, an untracked shape comes back with the
attribute's `default_value` — `"track_id": 0.0` on the control pedestrian —
which collides with the first track's index `0`. Distinguish by `keyframe`:
present ⇒ track shape, absent ⇒ plain shape.

**A track that ends before the last frame does NOT leave a phantom cuboid.** On
import CVAT auto-closes it with an `outside: true` shape on the next frame
(`_close_last_interval`, bindings.py:2704) — that is why the short pedestrian
track reads `frames=[0, 1, 2]` server-side — and the exporter drops outside
shapes. 6 cuboids in, 6 cuboids out, no interpolated extras, none missing.

## 3. Attribute survival

All five declared attributes survive, with these exact JSON shapes:

| attribute | declared `input_type` | went in | came back | type in JSON |
|---|---|---|---|---|
| `record_token` | text | `"k0:CAM_FRONT:0"` | `"k0:CAM_FRONT:0"` | `str` — exact |
| `track_id` | number | `7` | `0` | `int` on a track shape (**the index, see §2**); `float` (`0.0`, the `default_value`) on a plain shape |
| `attribute` | select | `"vehicle.moving"` | `"vehicle.moving"` | `str` — exact, including the empty-string option |
| `uncertain` | checkbox | `true` / `false` | `true` / `false` | **`bool`**, not the string `"true"` |
| `uncertain_reason` | text | `"occluded by bus"` / `""` | same | `str` — exact, empty string preserved |

So: four of five survive byte-for-byte; only `track_id` is clobbered, and only
because CVAT reserves that name (`CVAT_INTERNAL_ATTRIBUTES` = `occluded`,
`outside`, `keyframe`, `track_id`, `rotation`, `source`, `score`).

Notes for Task 16's parser:

* Types come from the label's `input_type`, not from what was imported: number →
  Python `float`, checkbox → `bool`, text/select → `str`
  (`_convert_attrs`, bindings.py:2166).
* An attribute the label does not declare is dropped; a declared attribute that
  was never set comes back as its `default_value`, so **every declared attribute
  is always present** in the export — absence means "not declared", never
  "not answered". `uncertain_reason: ""` therefore cannot be told apart from an
  unanswered one.
* Immutable attributes (`mutable: False`) are stored once per TRACK, mutable
  ones per shape (visible in the server dump in §2). The exporter merges the
  track-level ones into every shape (`tracked_shape["attributes"] += track["attributes"]`),
  so the flat export shows all five on every cuboid regardless.
* `occluded` is always present as a `bool` alongside them.

## 4. Geometry: order preserved, values QUANTIZED to 2 decimals

`position`, `rotation` and `scale` come back in the **same slot order** they went
in — no re-ordering of `scale`, so Task 16 needs no inverse permutation. But they
do **not** match to 1e-4: every component is rounded to 2 decimal places.

```
ok@1e-2  frame=0 k0:CAM_FRONT:0.position: [10.1234, 2.5678, 0.9012] -> [10.12, 2.57, 0.9]   (worst |delta| = 0.0034)
ok@1e-2  frame=0 k0:CAM_FRONT:0.rotation: [0.0, 0.0, 0.3456]        -> [0.0, 0.0, 0.35]     (worst |delta| = 0.0044)
ok@1e-2  frame=0 k0:CAM_FRONT:0.scale:    [4.2345, 1.8123, 1.6789]  -> [4.23, 1.81, 1.68]   (worst |delta| = 0.0045)
```

Worst |delta| over all 18 triples: **0.0045**, i.e. exactly the half-ulp of a
2-decimal rounding. Cause, found in the server image rather than inferred:
`datumaro/components/annotation.py:61` sets `COORDINATE_ROUNDING_DIGITS = 2`, and
`Cuboid3d._points_validator` (line 854) applies `np.around(points, 2)` when the
annotation object is constructed. That happens on **import**, so CVAT's database
already holds the quantized numbers (`points: [10.12, 2.57, 0.9, 0.0, 0.0, 0.35,
4.23, 1.81, 1.68, …]` in the server dump) — the export merely reproduces them.
The loss is one-way and unavoidable through this format:

* position: 1 cm grid;
* yaw: 0.01 rad ≈ **0.573°**;
* extents: 1 cm grid.

## 5. Label-attribute declaration CVAT accepted

Nothing was rejected. The project was created in one shot with all five
attributes on both cuboid labels, using exactly:

```python
{"name": ..., "input_type": "text"|"number"|"select"|"checkbox",
 "mutable": True|False, "default_value": ..., "values": [...]}
```

on labels declared as `{"name": ..., "type": "cuboid", "attributes": [...]}`.
Read back from the server (`project #53`, labels 541/542):

```json
{"name": "record_token",     "input_type": "text",     "mutable": false, "default_value": "",      "values": [""]}
{"name": "track_id",         "input_type": "number",   "mutable": false, "default_value": "0",     "values": ["0", "100000", "1"]}
{"name": "attribute",        "input_type": "select",   "mutable": true,  "default_value": "",      "values": ["vehicle.moving", "vehicle.stopped", "vehicle.parked", "pedestrian.moving", "pedestrian.standing", "pedestrian.sitting_lying_down", "cycle.with_rider", "cycle.without_rider", ""]}
{"name": "uncertain",        "input_type": "checkbox", "mutable": true,  "default_value": "false", "values": [""]}
{"name": "uncertain_reason", "input_type": "text",     "mutable": true,  "default_value": "",      "values": [""]}
```

Observations:

* A label attribute named `track_id` **is accepted** even though the name is
  internally reserved — CVAT stores it and shows it; only the Datumaro export
  overwrites it (§2). There is no error to catch here, which is why this had to
  be measured.
* `values: []` is normalised to `[""]` for text and checkbox.
* `number` was declared with `values: ["0", "100000", "1"]` (min, max, step) and
  accepted. Declaring a number with an empty `values` list was NOT tested.
* `default_value` is always a STRING, including `"false"` for a checkbox and
  `"0"` for a number.

## 6. Inner path of the JSON inside the export zip

`task.export_dataset("Datumaro 3D 1.0", filename, include_images=False)` produces
a zip whose entire member list is:

```
['annotations/default.json']
```

So Task 16 reads **`annotations/default.json`**. `default` is the subset name; a
task split into subsets would yield one file per subset, so the importer should
glob `annotations/*.json` and refuse (or merge) if there is more than one, rather
than hardcoding the single name blindly.

The export's `categories` block does **not** carry per-label attribute
declarations — `categories.label.labels[*].attributes` comes back `[]` and the
names are hoisted to a flat union at `categories.label.attributes`:

```json
"label": {
  "labels": [{"name": "a car", "parent": "", "attributes": []},
             {"name": "a pedestrian", "parent": "", "attributes": []}],
  "label_groups": [],
  "attributes": ["attribute", "occluded", "record_token", "track_id", "uncertain", "uncertain_reason"]
}
```

Task 16 must therefore read attribute values off the annotations, and must not
try to learn which attribute belongs to which label from the export.

## Raw exported JSON of one cuboid

Car, frame 0 (`items[0].annotations[0]` of `annotations/default.json`), verbatim:

```json
{
  "id": 0,
  "type": "cuboid_3d",
  "attributes": {
    "attribute": "vehicle.moving",
    "keyframe": true,
    "occluded": false,
    "record_token": "k0:CAM_FRONT:0",
    "track_id": 0,
    "uncertain": true,
    "uncertain_reason": "occluded by bus"
  },
  "group": 0,
  "label_id": 0,
  "position": [10.12, 2.57, 0.9],
  "rotation": [0.0, 0.0, 0.35],
  "scale": [4.23, 1.81, 1.68]
}
```

It went in as
`position [10.1234, 2.5678, 0.9012]`, `rotation [0.0, 0.0, 0.3456]`,
`scale [4.2345, 1.8123, 1.6789]`, `track_id 7`.

And the control (untracked) pedestrian, frame 1 — note no `keyframe`, and
`track_id` as a float default:

```json
{
  "id": 2,
  "type": "cuboid_3d",
  "attributes": {
    "attribute": "pedestrian.standing",
    "occluded": false,
    "record_token": "k1:CAM_FRONT:3",
    "track_id": 0.0,
    "uncertain": false,
    "uncertain_reason": ""
  },
  "group": 0,
  "label_id": 1,
  "position": [5.56, -3.33, 0.78],
  "rotation": [0.0, 0.0, 1.23],
  "scale": [0.71, 0.62, 1.73]
}
```

---

## DECISION

**Tasks 14 and 16 use TRACKS, and carry identity in `record_token`, not in
`track_id`.**

1. **Task 14 emits tracks.** A cuboid gets `"track_id": <int>` +
   `"keyframe": true` in its Datumaro attributes; CVAT builds a real track from
   them, the reviewer gets track-aware 3D tools, and the grouping survives the
   round trip intact (§2). No `outside` terminator needs writing — CVAT adds one
   and the export drops it again.
2. **Task 16 reads identity from `record_token`, never from the exported
   `track_id`.** The exported `track_id` is CVAT's per-task track index, not the
   value Task 14 wrote (§2), and on untracked shapes it is the attribute default
   `0.0`, which collides with the first track's index. `record_token` is a text
   attribute and comes back byte-exact, so it is the identity carrier. Task 14
   must therefore write a `record_token` that is unique per object-instance and
   is the pipeline's real key.
3. **`track_id` stays declared and stays written** — it is what the reviewer
   sees in the CVAT UI, it is preserved verbatim server-side, and writing it is
   what makes CVAT build the track in the first place. It is simply not a value
   Task 16 may read back out of the export. Task 16 may use the exported
   `track_id` only as a *within-this-export* grouping key (all shapes of one
   track share it), and must gate that on `keyframe` being present.
4. **Task 16's geometry tolerance is 1e-2, not 1e-4.** Everything is quantized
   to 2 decimals by datumaro on import (§4): compare with `atol=5e-3` per
   component (0.0045 observed worst case), and do not treat a 4-decimal
   round-trip mismatch as corruption. Any check that a reviewed box is
   "unchanged" must use the same 1e-2 grid. `scale` is **not** re-ordered, so no
   inverse permutation is needed — Task 16 reads
   `position (x,y,z)`, `rotation (roll,pitch,yaw)`, `scale (x,y,z extent)` in the
   slots Task 14 wrote, per `scripts/export_cvat_3d.py`'s existing convention.
5. **The human answers ride on the other four attributes as-is** —
   `attribute` (str), `uncertain` (bool), `uncertain_reason` (str) — with the
   caveat that a declared-but-unanswered attribute is indistinguishable from one
   answered with the default. If Task 16 must tell "reviewer said no" from
   "reviewer said nothing", the default has to be an explicit sentinel outside
   the answer set (e.g. `uncertain_reason` default `"<unset>"`), not `""`.
6. **Task 16 reads `annotations/default.json`** from the export zip (§6),
   globbing `annotations/*.json` and refusing more than one subset.

The spec's shapes-only fallback is not needed: tracks round-trip.

## Artifacts left behind (nothing was deleted)

The operator's standing rule is that nothing on the CVAT server is deleted
without an explicit per-item ask, so `scripts/spike_cvat_3d_roundtrip.py`
implements `--cleanup` as a printed refusal. Left in place, for the operator to
remove by hand whenever they like:

* project **#53** `SPIKE — 3D roundtrip (delete me)`
* task **#292** `spike-3d-roundtrip-1322185` (first run: car track + control shape)
* task **#293** `spike-3d-roundtrip-1345913` (second run: + the short track of §2)

Both tasks are 3 frames of 200 random points, a few kB each.
