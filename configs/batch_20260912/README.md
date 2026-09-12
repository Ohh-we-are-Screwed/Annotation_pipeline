# configs/batch_20260912/ — generated, do not edit

`chunk_NN.yaml` in this directory is written by `scripts/run_all_chunks.py`
(including on `--dry-run`) and git-ignored. One file per global chunk number,
`NN` = 01..38 over the four 2026-09-11 Dhaka export sessions in this order:

| chunks | session                 | scenes |
|--------|-------------------------|--------|
| 01-11  | `dhaka_20260911_141259` | `chunk_0000`..`chunk_0010` |
| 12-13  | `dhaka_20260911_151029` | `chunk_0000`..`chunk_0001` |
| 14-27  | `dhaka_20260911_154512` | `chunk_0000`..`chunk_0013` |
| 28-38  | `dhaka_20260911_170051` | `chunk_0000`..`chunk_0010` |

Each file is `configs/paths_zami_20260911.yaml` with the read side pointed at
that chunk's session (`dataroot`, `meta_root`; `version: v1.0-dhaka-fixed2`)
and the write side moved to roots nothing else shares:

    work_root:      /mnt/hdd/dhakascenes/batch_20260912/NN/work
    out_root:       /mnt/hdd/dhakascenes/batch_20260912/NN/out
    probe_out_root: /mnt/hdd/dhakascenes/batch_20260912/NN/probe

The isolation is not tidiness: `scripts/run_stages.sh` takes a flock on its
work root, so two chunks sharing one would run one at a time — and a Stage 0
probe report, which is per work root, would describe the wrong chunk.

Regenerate (writes nothing else, runs nothing):

    PYTHONNOUSERSITE=1 python scripts/run_all_chunks.py --dry-run --chunks 1-38

To point a batch somewhere else, pass `--batch-root` / `--template`; the
dashboard reads only what the runner wrote to the SSD, not this directory.
