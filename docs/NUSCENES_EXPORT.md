# nuScenes exports

`scripts/run_stages.sh` includes `road` and then `release` in its default run,
after Stage 8. `release` runs the Stage 9 QA gate, exports the two extra layers,
and then `scripts/export_release.py`. Explicit step lists must include `release`
where the dataset should be written, and `road` ahead of it for the 3D
segmentation layer.

To export chunk 0000's existing annotations without running the models again:

```bash
DHAKASCENES_SUBSTRATE=dhaka6 \
DHAKASCENES_PATHS_CONFIG=configs/paths_b_fused_chunk_0000.yaml \
bash scripts/run_stages.sh release --scenes dhaka_20260905_174950_chunk_0000
```

`release` writes THREE sibling layers under `export/<name>/`, and all three are
the delivery:

- `boxes/` — the nuScenes root: tables in `v1.0-dhaka-fixed2/`, referenced data
  in `samples/` and `sweeps/`, release metadata and `DELIVERY_NOTE.md`
  alongside them. Load with
  `NuScenes(version="v1.0-dhaka-fixed2", dataroot="boxes")`.
- `road/` — EXTRA, not stock nuScenes: the road-surface layer as a SECOND,
  standalone nuScenes-lidarseg root — its own scoped tables plus
  `road/lidarseg/<version>/<sample_data_token>_lidarseg.bin`, one `uint8` label
  per point. `NuScenes(version=..., dataroot="road")` loads it with no extra
  arguments and no blobs. Only two labels are ever populated —
  `24 flat.driveable_surface` and `0` unlabelled; the other 30 categories exist
  because the devkit's stats and render APIs assume `index == list position`.
  The bins are positional over the raw `LIDAR_TOP` blob, so point rendering
  reads against `boxes/samples/`.

  The segmentation is a separate root **by necessity, not by preference**.
  `nuscenes.py:110` runs only when `lidarseg.json` is present and does
  `self.colormap[c['name']]` over every category row sorted by `index`, so a
  lidarseg root's `category.json` must be exactly the devkit's closed 32-name
  vocabulary. 17 of the 18 Dhaka box classes are not in it, so a single root
  carrying both taxonomies raises `KeyError: 'car'`. Do not try to merge them.
- `coco_2d/` — EXTRA: per-scene COCO `instances.json`, 2D boxes + masks in
  image space. Paths are relative to the nuScenes root, so this layer needs
  `boxes/` beside it.

Only selected scenes are exported. CVAT review files remain in sibling
`cvat_export*` folders and are NOT part of the delivery.

`road/` is exported from whatever the `road` step left in `<work_root>/stage_road`.
It is a soft step: if that stage never ran, the layer is silently absent and only
`DELIVERY_NOTE.md` records it. `road` is in the default step list since
2026-09-09 — before that it was opt-in, which is how the fused chunks shipped
without the 3D segmentation that the day-1 exports had. A bare
`run_stages.sh release` still skips `road/` unless an earlier run left the
marker, so pass `road release` when exporting a tree that has no `stage_road`.

`EXPORT_ROOT` overrides the repository's `export/` default. `EXPORT_NAME`
overrides the dataset-and-work-folder name. `RELEASE_BLOBS=hardlink` is the
default: blobs are real files sharing read-only source bytes, with a copy
fallback across filesystems. Use `RELEASE_BLOBS=copy` for independent copies.
The exporter refuses to overwrite an existing nonempty release.

`scripts/run_fused_chunks.py 0004 0005 0006 0007 0008 0009 0010` runs those
chunks sequentially, produces each nuScenes release, and publishes its CVAT
tasks. It records failed chunks and continues with the next; it never bypasses
the input-data checks. Run it with the `ano_pipe` Python environment.
