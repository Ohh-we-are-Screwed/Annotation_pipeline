# nuScenes exports

`scripts/run_stages.sh` includes `release` in its default run, after Stage 8.
This runs the Stage 9 QA gate and `scripts/export_release.py`. Explicit step
lists must include `release` where the dataset should be written.

To export chunk 0000's existing annotations without running the models again:

```bash
DHAKASCENES_SUBSTRATE=dhaka6 \
DHAKASCENES_PATHS_CONFIG=configs/paths_b_fused_chunk_0000.yaml \
bash scripts/run_stages.sh release --scenes dhaka_20260905_174950_chunk_0000
```

The nuScenes root is
`export/dhaka_20260905_174950_chunk_0000/boxes/`, with tables in
`v1.0-dhaka-fixed2/`, referenced data in `samples/` and `sweeps/`, and release
metadata and the delivery note alongside those directories. Only selected
scenes are exported. CVAT review files remain in sibling `cvat_export*` folders.

`EXPORT_ROOT` overrides the repository's `export/` default. `EXPORT_NAME`
overrides the dataset-and-work-folder name. `RELEASE_BLOBS=hardlink` is the
default: blobs are real files sharing read-only source bytes, with a copy
fallback across filesystems. Use `RELEASE_BLOBS=copy` for independent copies.
The exporter refuses to overwrite an existing nonempty release.

`scripts/run_fused_chunks.py 0004 0005 0006 0007 0008 0009 0010` runs those
chunks sequentially, produces each nuScenes release, and publishes its CVAT
tasks. It records failed chunks and continues with the next; it never bypasses
the input-data checks. Run it with the `ano_pipe` Python environment.
