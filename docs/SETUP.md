# Setup — a fresh clone, in one command

```bash
git clone <repo> && cd <repo>
./setup.sh
```

`setup.sh` exists because exactly one fact about this repository is
machine-dependent: **where the substrate lives**. Everything else — the conda
interpreter, the pinned dependency set *in its mandated install order*, `.env`,
the write roots, the two checkpoint files, the tmux session every long run is
launched into — is derivable, and deriving it by hand is how a second machine
ends up subtly different from the first. Companion to
[`RUNNING.md`](RUNNING.md) (how a run is launched) and
[`DECISIONS.md`](DECISIONS.md) (why anything is the way it is).

The script is **idempotent**: a second run reports `SKIP` almost everywhere and
finishes in about two seconds. It never overwrites an existing `.env`, never
rebuilds an existing conda env without `--update`, and never deletes anything.

## Flags

| flag | effect |
|---|---|
| `--help` | the flag list |
| `--update` | re-resolve an existing env (`conda env update --prune`) and re-run the pip installs. Without it an existing env is reported and left alone. |
| `--cpu-only` | CPU torch wheels instead of the pinned cu124 ones. Every GPU stage (2, 3, 3b, 3f, 4, 5, road) stops being runnable and the VRAM-cap contract becomes meaningless. |
| `--with-cvat` | clone `cvat-ai/cvat` next to the repo and `docker compose up -d`. Without it the script only probes `$CVAT_HOST`. |
| `--skip-tests` | do not run `pytest tests/ -q` (the import smoke test still runs) |
| `--skip-weights` | do not download `mobile_sam.pt` / `yolo11x.pt` |
| `--weights-dir DIR` | where the two checkpoint FILES go. Default: the checkpoints directory `.env` names, **outside** the repo. |
| `--yes` | non-interactive; required for an unattended run |

Exit status is 0 only if every fatal step passed. Non-fatal problems are `WARN`
and do not change it.

## What it does, and the contract behind each step

1. **Preflight.** bash ≥ 4, `git`, `tmux` (fatal); `docker` and `nvidia-smi`
   (warn-only — a CPU-only setup is allowed, and the script names what will not
   run); free space on the repo, conda-base and write-root filesystems; Python
   3.10, which `environment.yml` pins because `nuscenes-devkit 1.1.11` has no
   wheels above 3.11.
2. **Conda env `ano_pipe`.** Created from `environment.yml` **with the `pip:`
   block stripped**. That block lists `requirements-torch.txt` and
   `requirements.txt` together, and conda hands both to one pip process — which
   is precisely the process-global `--index-url` footgun
   `requirements-torch.txt` exists to prevent. Conda's job is the interpreter;
   step 3 does pip's job in the documented order.
3. **Dependencies**, in the order `requirements.txt`'s header mandates:
   `requirements-torch.txt` alone → `requirements.txt` → `--no-deps
   requirements-devkit.txt`. Before installing anything it checks the installed
   distributions against every pin **offline**, so a satisfied env is a `SKIP`
   rather than three network round-trips.
4. **`.env`.** Absent → copied from `.env.example` and filled with the
   machine-derived values (cache roots, thread pins, VRAM cap, determinism
   block). Present → never touched; the script reports which `.env.example`
   keys are missing and which secrets are still blank. **No secret value is ever
   copied from anywhere.**
5. **Write roots** from `configs/paths.yaml`, parsed with PyYAML through the
   env's interpreter (never grep), created only after `work_root`, `out_root`
   and `probe_out_root` are shown to be outside the repo and pairwise disjoint
   from `dataroot` (§1.8). `dataroot` is never created — an empty one turns "you
   have not pointed me at the data" into stage 0's much less obvious "no usable
   scenes".
6. **Checkpoints.** `yolo11x.pt` from the ultralytics release URL recorded in
   `.env.example`, verified against the sha256 recorded beside it (mismatch is
   fatal); `mobile_sam.pt` from the *same commit* `requirements-torch.txt` pins
   the `mobile_sam` package to, with its digest printed for the manifest. Both
   are FILES, not hub ids: `YOLO("yolo11x.pt")` on a missing file downloads into
   the CWD, i.e. into the repo tree, which §1.8 forbids.
7. **Verification.** `pytest tests/ -q`, an import smoke test
   (`torch numpy scipy yaml PIL cvat_sdk nuscenes` plus `torch.cuda.is_available()`),
   and the resolved path contract printed back.
8. **tmux `pipe`**, created if absent. Existing windows are never touched — the
   operator's running chain lives in them.
9. **CVAT**, only with `--with-cvat`. Otherwise it just reports whether
   `$CVAT_HOST` answers. Nothing is ever deleted.
10. **Summary** — a readiness checklist and the numbered by-hand list.

## Derivations, so they can be argued with

| value | derived from |
|---|---|
| `HF_HOME`, `HUGGINGFACE_HUB_CACHE`, `TORCH_HOME`, `YOLO_CONFIG_DIR` | `dirname(work_root)/cache/…` — `.env.example` describes the cache roots as *siblings of work_root/out_root*, so the parent of `work_root` is the one honest anchor that is not somebody's home directory |
| `OMP_NUM_THREADS`, `MKL_NUM_THREADS` | `nproc / 4`, clamped to `[1, 8]` — reproduces the `8` `.env.example` records for the 32-core reference host, and keeps a 4-core laptop from oversubscribing BLAS |
| `DHAKASCENES_VRAM_CAP_MIB` | `nvidia-smi --query-gpu=memory.total`: ≥ 24 GB → `22000` (C19 production budget, display headroom); ≥ 8 GB → `4096` (the binding C1 pilot contract); smaller or no GPU → **empty**, because on the laptop itself the physical card is the ceiling |

## What the script cannot do for you

1. **Secrets.** `HF_TOKEN` (first download of the gated `facebook/sam3` and
   `facebook/sam3.1` repos), then `CVAT_HOST` / `CVAT_USER` / `CVAT_PASSWORD`.
   Follow with `<env>/bin/python -m huggingface_hub.cli auth login`.
2. **`configs/paths.yaml`** — `dataroot`, `meta_root`, `version`. `version` must
   equal the on-disk directory name *verbatim*; a mismatch yields zero token
   matches, which reads downstream as "no usable scenes" rather than as a config
   error. The substrate is not redistributable and is not in this repo.
3. **CVAT login** — create the superuser once the stack is up:
   `docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'`.

## Two contract gaps this script works around

Both are reported at runtime rather than hidden, and both should be fixed in the
tracked files rather than in `setup.sh`:

- **`cvat_sdk` is missing from `requirements.txt`.** It is imported by
  `scripts/cvat_setup.py`, `cvat_setup_3d.py`, `cvat_purge.py` and
  `import_cvat_3d.py`, and `run_stages.sh` publishes to CVAT in its *default*
  chain — but it appears only in `requirements-lock.txt` (`cvat_sdk==2.73.0`).
  A clone that installed only the three declared files would fail at the publish
  step. `setup.sh` installs the lock's exact pin and `WARN`s about it.
  `requirements.txt` rule 3 says anything this repo imports belongs there.
- **`CVAT_HOST` / `CVAT_USER` / `CVAT_PASSWORD` are not in `.env.example`,**
  which claims to be the canonical list of every variable the project reads.
  `setup.sh` appends them as empty placeholders in a clearly marked block when
  it creates a `.env`.
