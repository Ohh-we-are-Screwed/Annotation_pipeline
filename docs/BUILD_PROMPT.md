# BUILD PROMPT — DhakaScenes Pilot: verify everything, reconcile, restructure, finish

**Audience:** an autonomous coding agent with filesystem, shell, and GPU access on this machine.
**Written:** 2026-08-12, after a full survey of the repository on disk.
**Bible:** [`docs/pilot_plan.md`](pilot_plan.md) rev 2. Everything below serves it.

---

## 0. Your mandate

Four activities, **in this order**, and you do not begin one before the previous is complete:

| | Activity | Output | Blocks |
|---|---|---|---|
| **A** | **Verify** every claim the bible makes against what is on disk | `docs/conformance.yaml` + `scripts/check_conformance.py` | everything |
| **B** | **Reconcile** every contradiction the verification exposes | `docs/DECISIONS.md` | C, D |
| **C** | **Restructure** the repository if — and only if — B says it must be | one mechanical commit per move | D |
| **D** | **Build** what is missing, in the bible's phase order, under its test discipline | code + tests + manifests | — |

Most of the effort belongs in **A**. There are ~15,000 lines of pipeline code on disk that the bible says do not exist, three phases that have been *run* but never *gated*, and no tests at all. Until you know — mechanically, with evidence — which parts conform, any building you do is layered on assumptions.

`docs/pilot_plan.md` is unusually good: it names the *silent* failure mode behind almost every decision it makes. Your job is not to redesign it. Your job is to prove or disprove that the repository implements it, resolve every place where the plan contradicts itself, the machine, or the code, restructure where the structure itself violates the plan, and then finish it.

The plan's governing principle applies to you:

> **fidelity over quality.** Same stages, same contracts, same order as the real spec. The goal is proving the *plumbing*, not producing usable labels.

And its test philosophy is binding:

> before implementing a stage, write down (a) the one or two failure modes that would be **silent** — wrong output that still looks plausible and does not crash — and (b) a test that specifically catches that failure mode.

A test that would pass against a broken implementation is worse than no test: it converts an open question into a false answer. The plan's §9 table has an entire column of these (`Replace — would pass on broken code`). Do not add more.

---

## 1. Governing documents, and their precedence

| Rank | Document | Role |
|---|---|---|
| 1 | `docs/pilot_plan.md` (rev 2) | **The bible.** Pilot scope, phases, contracts, decisions. |
| 2 | `docs/comprehensive.md` | Canonical v1.0 spec. The pilot mirrors its *shape*. Deviations are **declared**, never absorbed. |
| 3 | `docs/GAP_ANALYSIS.md` | Literature positioning. Governs what may be *claimed*. |
| 4 | `docs/dhakascenes-pilot-validated-comet.md` | Pre-implementation audit. Finding IDs `P0-n / P1-n / M-n / X-n`, referenced throughout. |
| — | `docs/Annotation_pipeline.md` | **Superseded** by `comprehensive.md` §7. Cite only where the plan explicitly rejects one of its values (its DBSCAN ε table is rejected outright — §5.7, X-6). |

Precedence: if the plan and `comprehensive.md` disagree, the plan wins **for the pilot**, and the divergence goes in the deviation register. If the plan disagrees with **itself**, that is a Contradiction (§5).

---

## 2. Environment — fixed. Do not renegotiate it, do not work outside it.

A conda environment named **`ano_pipe`** exists and is verified. It is the only interpreter this project runs under.

```bash
export PATH="/home/mt/miniconda3/bin:$PATH"
conda activate ano_pipe
export PYTHONNOUSERSITE=1          # non-negotiable, see §2.2
cd /home/mt/Zami
```

Verified 2026-08-12 — **16/16 repository modules import clean**:

| | |
|---|---|
| Python | 3.10.20 (`/home/mt/miniconda3/envs/ano_pipe`) |
| torch | 2.5.1+cu124, `cuda_available=True`, CUDA 12.4 |
| device | NVIDIA GeForce RTX 4090, 24080 MiB, sm_89, driver 595.84 |
| numpy / scipy / sklearn | 1.26.4 / 1.14.1 / 1.5.2 |
| transformers / tokenizers / timm | 4.46.3 / 0.20.3 / 1.0.11 |
| umap-learn / hdbscan / numba | 0.5.7 / 0.8.40 / 0.60.0 |
| mobile_sam | git `f706ad9c4eb7f219c00d9050e46328518ffb65d2` |
| nuscenes-devkit | 1.1.11, installed `--no-deps` |

### 2.1 The dependency manifest is part of the contract

Four files; `requirements.txt` is the index naming the other three:

- **`requirements.txt`** — canonical. Every entry exact-pinned, every entry carrying a provenance note saying *which part of the plan needs it*.
- **`requirements-torch.txt`** — CUDA wheels. Separate because pip's `--index-url` is process-global, not per-line.
- **`requirements-devkit.txt`** — `nuscenes-devkit`, `--no-deps`, because its metadata pins `matplotlib<3.6.0` — a stale bound that would drag the environment back to 2022.
- **`requirements-lock.txt`** — generated: `pip freeze --exclude-editable > requirements-lock.txt`.

Rules you maintain:

1. **Any package you need goes into `requirements.txt` with an exact pin and a provenance comment, before you use it.** §1.9 requires package versions in every `run_manifest.json`; a floating range makes two runs of the same commit produce different manifests.
2. **No bare `pip install <name>`.** A hand-installed package is invisible to the manifest and will not survive a rebuild.
3. **Regenerate `requirements-lock.txt` after every change** and mention it in your report.
4. **`numpy` stays `<2`.** `hdbscan` and `nuscenes-devkit` are validated against the 1.x ABI. This machine already demonstrated the failure: the system interpreter carried numpy 2.2.6 against a scipy built for `<1.25`, and `pipeline/stage7_track/track.py` — which imports `scipy.optimize.linear_sum_assignment` — died with `ImportError: numpy.core.multiarray failed to import`. Under `ano_pipe` it imports clean.

### 2.2 `PYTHONNOUSERSITE=1` is not optional

`ano_pipe` is Python 3.10 and so is the system interpreter, so `/home/mt/.local/lib/python3.10/site-packages` lands on `sys.path` **ahead of** the environment's own site-packages. Two consequences, both observed here:

- A user-site package silently shadows the pinned one. `run_manifest.json` then records the pinned version while the interpreter imported a different one — a manifest that lies.
- `pip install` into the environment **uninstalls the shadowing copy from `~/.local`**. This already happened during setup: `umap-learn 0.5.12` and several numeric packages were removed. Surviving user-site packages are now `Cython, joblib, pynndescent, pyzed, threadpoolctl` only.

### 2.3 `.env` is a declared contract, not a scratchpad

`.env.example` is committed and is the **canonical list of every environment variable this project reads**. `.env` is git-ignored (`cp .env.example .env`).

The dividing line, which you hold:

> **`.env` = machine-local facts and secrets.** Differ per machine. Never appear in a manifest.
> **`configs/` = experiment parameters.** Identical across machines. Recorded in every `run_manifest.json`.
> **If changing a value would change the OUTPUT, it belongs in `configs/`, not `.env`.**

That is why the global seed is in `configs/pipeline_pilot.yaml` and `CUBLAS_WORKSPACE_CONFIG` is in `.env`.

Declared: `PYTHONNOUSERSITE`, `HF_TOKEN`, `HF_HOME`, `HUGGINGFACE_HUB_CACHE`, `TORCH_HOME`, `MOBILE_SAM_CHECKPOINT`, `DHAKASCENES_PATHS_CONFIG`, `CUBLAS_WORKSPACE_CONFIG`, `DHAKASCENES_ALLOW_TF32`, `DHAKASCENES_CUDNN_BENCHMARK`, `TOKENIZERS_PARALLELISM`, `CUDA_VISIBLE_DEVICES`, `PYTORCH_CUDA_ALLOC_CONF`, `DHAKASCENES_VRAM_CAP_MIB`, `DHAKASCENES_RUN_INTEGRATION`, `DHAKASCENES_RUN_GPU_TESTS`, `DHAKASCENES_RUN_INDIGENOUS_PROBE`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS`.

`MOBILE_SAM_CHECKPOINT` points at `/home/mt/dhakascenes/cache/checkpoints/mobile_sam.pt`, which **does not exist yet**. MobileSAM ships weights as a file, not a hub id, and `stage4_masks/masks.py` raises `ModelUnavailable` saying exactly that. Fetch it at the Phase 5b gate and record its SHA-256 in `configs/models_pilot.yaml`.

---

## 3. Measured ground truth of the repository

**Read this before you read any code.** Measured directly on 2026-08-12. It is not what the plan's status header says.

### 3.1 What exists

Repository root is `/home/mt/Zami` — **not** `dhakascenes_pilot/` as §2 of the plan draws it.

```
docs/          5 documents, 2368 lines
configs/       paths.yaml (43), taxonomy_pilot_nuscenes.yaml (69)
nuscenes/      15 GB — the dataroot, INSIDE the repo root (see C11)
pipeline/
  common/      conventions.py 428 · schemas.py 1126 · eval_region.py 361
               paths.py 351 · model_interfaces.py 1171          [manifest.py ABSENT]
  stage0_data_probe/probe.py     960
  stage1_ingestion/ingest.py    1008
  stage2_ood/ood.py              697
  stage3_proposals/proposals.py 1086
  stage4_masks/masks.py          883
  stage5_lift/lift.py           1114
  stage6_cluster/cluster.py     1176 · priors.py 820
  stage7_track/track.py         1601
  stage8_inflate/inflate.py     1124                            [stage9_qa ABSENT]
scripts/       probe_substrate.py 654 · measure_vram.py 1185
```

~15,000 lines. **Quality is high and it tracks the plan closely** — constants carry provenance strings (`"comprehensive.md §7.3.1, unvalidated on this substrate"`), DBSCAN and ICP are hand-implemented *specifically for determinism* (§1.6, §1.9), `conventions.py` implements the four-hop projection chain, the taxonomy file carries the injectivity and no-chunking rules. **This code is an asset to verify, not scaffolding to replace.**

### 3.2 What has actually been run

| Artifact | Path | State |
|---|---|---|
| Phase 1 reconnaissance | `probe_out/phase1_substrate.json` | present |
| Stage 0 allowlist | `work/stage0_data_probe/usable_scenes.json` | **10 usable scenes**, fingerprint `4c5a5cfe…d7891` |
| Stage 0 report | `work/stage0_data_probe/probe_report.json` | present |
| Stage 1 manifest | `work/stage1_ingestion/run_manifest.json` | 404 keyframes, retained fraction 0.571 |
| Stage 1 clouds | `work/stage1_ingestion/clouds/` | **1.7 GB**, all 10 scenes |
| Priors | `out/priors/priors_pilot_v0.json` | 23 classes, `source: "nuscenes_gt_pilot"` |

(All under `/home/mt/dhakascenes/`.)

The scene partition on disk **matches §11 decision 3 exactly** — priors `0061/0103/0553/1077`, tuning `0655/1094`, run `0757/0796/0916/1100`, each spanning both locations with one night scene. Verified; do not re-derive.

Stage 1's diagnostics are real: 146,137,792 input points → 83,505,441 after ground removal; `compensation_max_residual_m = 7.02e-13` (ego-motion compensation numerically exact); 223 sector fits rejected.

**Stages 2, 3, 4, 5, 6(cluster), 7, 8 have never been executed.** No checkpoints downloaded. No GPU work done.

### 3.3 What is missing outright

| Missing | Mandated by | Severity |
|---|---|---|
| `tests/` — **the entire tree** | §2, §9, and the `Tests:` line of every phase 2–10 | **Blocks every gate** |
| `pipeline/common/manifest.py` | §2, §1.9, Phase 2 | **Blocking** (C3) |
| `pipeline/stage9_qa/` | §5.10, Phase 10 | Blocks Phase 10 |
| `probes/indigenous_prompt_probe/` | §8 | Blocks Phase 10 |
| `configs/models_pilot.yaml` | §7.2 — **referenced by code that cannot run without it** | Blocks Phase 5 |
| `configs/models_production.yaml` | §7.3 | Blocks Phase 5 |
| `configs/pipeline_pilot.yaml` | §10 — **referenced by code that cannot run without it** | Blocking |
| `configs/taxonomy_probe_indigenous.yaml` | §8 | Blocks Phase 10 |
| `scripts/check_vram_paper.py` | Phase 5a | Blocks Phase 5 |
| `scripts/run_pilot.py` | §2 | Blocks Phase 10 |
| `README.md` with the §13.2 banner | Phase 10 exit gate | Blocks Phase 10 |
| `pyproject.toml` / any packaging | — (see C11) | Structural |
| git repository | §1.9 (`git commit` in every manifest) | **See C5** |

`git rev-parse` fails: `/home/mt/Zami` is not a git repository.

---

## 4. ACTIVITY A — the verification sweep

**This is the largest and most important part of your work. Do not shortcut it.**

You are not "reviewing the code." You are building a **machine-checkable ledger** that maps every assertion in the bible to evidence about this repository. The reason it must be mechanical: a human or an agent reading 15,000 careful, well-commented lines will conclude they are correct, because they *read* correct. The failures this project is built to catch are by construction the ones that look fine.

### 4.1 The conformance ledger

Create **`docs/conformance.yaml`** — one row per checkable assertion — plus **`scripts/check_conformance.py`**, which validates the ledger's own integrity and renders `docs/CONFORMANCE.md`.

Row schema:

```yaml
- id: "1.1-r3"                       # <plan-section>-<ordinal>, stable, never reused
  source: "pilot_plan.md §1.1 rule 3"
  claim: "Stage 1 applies T_ego_lidar exactly once; no later stage re-applies it."
  kind: contract                      # contract|behaviour|artifact|value|test|decision|claim-hygiene|structure
  status: CONFORMS                    # see 4.2
  evidence:
    class: TEST                       # see 4.3
    ref: "tests/unit/test_frames.py::test_ego_transform_applied_once"
  closes: [P0-1]                      # audit finding IDs, if any
  phase: 4                            # which plan phase owns it
  note: ""
```

### 4.2 Status values — and the rule that makes the ledger worth anything

| Status | Meaning |
|---|---|
| `CONFORMS` | Implemented **and demonstrated**. |
| `PLAUSIBLE` | Code appears to implement it; nothing demonstrates it. |
| `VIOLATES` | Implemented contrary to the claim. |
| `ABSENT` | Not implemented at all. |
| `UNVERIFIABLE` | Cannot be checked on this substrate — say why. |
| `N/A` | Explicitly waived by the plan (§4) — cite the waiver. |

> **The load-bearing rule: `CONFORMS` requires evidence of class `TEST` or `MEASUREMENT`. A `CODE-SITE` may only ever support `PLAUSIBLE`.**

"I read the code and it does this" is not verification. Without this rule the sweep degenerates into a reading exercise that ratifies whatever is already there — which is exactly the failure that let three phases run un-gated.

### 4.3 Evidence classes

| Class | What it is | Can support |
|---|---|---|
| `TEST` | A test that fails if the claim is false. Name it `file::test_name`. | `CONFORMS`, `VIOLATES` |
| `MEASUREMENT` | A number produced by running code on real data, recorded with how it was obtained. | `CONFORMS`, `VIOLATES` |
| `CODE-SITE` | `path:line` where the behaviour is implemented. | `PLAUSIBLE`, `ABSENT`, `VIOLATES` |
| `ARTIFACT` | A file on disk and the field inside it. | `CONFORMS` for existence claims only |
| `HUMAN` | A decision only a person can make. | `UNVERIFIABLE` pending answer |

### 4.4 Scope — enumerate, do not summarise

Walk the bible **section by section, line by line**, and emit a row for every assertion that could be false. Coverage floor:

| Plan section | Enumerate |
|---|---|
| §0 | all 14 substrate facts, each re-measured on this disk |
| §1.1 | 4 frame rules, the mandatory non-identity test, each named silent failure |
| §1.2 | 5 time rules, all 6 per-camera Δt medians, the real-record conversion test |
| §1.3 | the four-hop chain, extrinsic inversion direction, `z ≤ 0` cull, K-after-extrinsics |
| §1.4 | fit-on-accumulated / apply-to-single-sweep / lift-from-single-sweep / count-on-single-sweep, and `cloud_kind`, `n_sweeps_actual`, `window_ns` |
| §1.5 | all 6 two-D rules |
| §1.6 | per-instance scope, ε-means-class-conditional, deterministic tie-break |
| §1.7 | the raising `write_records()` boundary, **both** directions of the invariant, `tier ≠ source` |
| §1.8 | every `validate_paths()` assertion, fingerprint binding, read-only dataroot, roots outside repo |
| §1.9 | **every field** of the manifest list, atomic write, `_SUCCESS`, upstream refusal, per-scene isolation, idempotence, OOM hard stop, determinism |
| §1.10 | `coverage_config: R2`, E derived not constant, ρ as count-over-area |
| §2 | every path in the tree — present / absent / moved |
| §3 | I-1…I-7 rows, §3.1's 4 downgrade rules, §3.2's 4 field gaps |
| §4 | all 8 waive/restore rows |
| §5.1–§5.10 | every named requirement per stage — §5.1's six predicates individually |
| §6 | 3 capabilities, 2 guards, the licence `[VERIFY]` |
| §7 | 4 roles, 3 interface fixes, preprocessing-belongs-to-role, asserted teardown, §7.2's per-role fields |
| §8 | all 5 separation mechanisms |
| §9 | **every cell** of the table — Keep, Replace, Add — as its own row |
| §10 | **every tunable in the list** (~45), each: is it config? does it carry provenance? |
| §11 | 8 decisions, each with its locking phase |
| §12 | 10 phases × entry / deliverables / exit gate |
| §13 | 4 corrected figures, the banner's 3 required locations, all 11 claim-table rows |
| §14 | all 10 survivals — still true? |
| audit | all 49 finding IDs: `P0-1…9`, `P1-1…15`, `M-1…15`, `X-1…10` |

**Expect 300–400 rows.** A ledger with 80 rows means you summarised instead of enumerating, and you should redo it. `check_conformance.py` must assert: every audit finding ID appears at least once; every plan section `§n` appears at least once; no row has `status: CONFORMS` with evidence class `CODE-SITE`; no row has an empty `evidence.ref`.

### 4.5 Verification order

1. **Re-measure §0 yourself.** The appendix has the values; confirm, don't trust.
2. **Re-derive the three run artifacts' claims** — allowlist, Stage 1 manifest, priors — against the tests that *should* have gated them. If a retro-fitted test fails, that artifact is invalid: say so loudly and regenerate. Never patch the test.
3. **Trace each contract boundary end to end.** For each of I-1…I-5, name producer, consumer, and the field-by-field match. This is where representation drift hides.
4. **Grep for the plan's forbidden patterns** and prove absence, not just non-observation: square resizes, prompt chunking, `sigma_pos_m = 0.0`, dotted category names reaching a prompt, `max_memory_allocated()` used as an occupancy measure, hardcoded ε from `Annotation_pipeline.md`, `tier` conflated with `source`, silent OOM retry.
5. **Only then** open the contradiction register.

---

## 5. ACTIVITY B — the contradiction register

`docs/DECISIONS.md`. **C1–C13 below are the seed, not the complete list** — the sweep will find more, and every `VIOLATES` / `ABSENT` / `UNVERIFIABLE` row that implies a choice becomes a new entry. Format:

```
### C<n> — <one-line title>
Status:      RESOLVED | DEFERRED-TO-PHASE-<n> | ESCALATED-TO-HUMAN
Plan says:   <quote + section>
Disk says:   <measured fact + how measured>
Resolution:  <what you chose>
Because:     <why, in terms of what silently breaks otherwise>
Recorded in: <file:line where a future reader trips over it>
Gate:        <which phase gate re-checks this>
```

Nothing may be closed by silence.

---

### C1 — The hardware premise vs the machine — **RESOLVED BY HUMAN, 2026-08-12**

**Plan says:** scope line — *"3050ti laptop GPU (4 GB VRAM)"*. §7.2 — `total_device_mb: 4096`. §5.5 — MobileSAM over SAM-ViT-B *"because the ~9× parameter difference matters at 4 GB"*. Phase 5b — *"On a 4 GB card the gap between 'allocated' and 'occupied' **is** the margin being budgeted."* §13.2 lists *"It runs on 4 GB VRAM"* as supportable.

**Disk says:** `nvidia-smi` → **RTX 4090, 24564 MiB, driver 595.84**. Six times the plan's budget.

**Human's decision:** the fleet is **two machines**. The 3050ti laptop (4 GB) is real and is where the pilot must ultimately run; this 4090 box is where the work happens for now. So the plan's premise is not wrong — it just isn't *this* machine — and the resolution is **(a): keep the 4 GB tier as the binding contract, enforce the ceiling synthetically on the 4090.** Restating scope to 24 GB (option b) is rejected: it would sever the pilot from the device it exists to run on.

**What this binds you to:**

1. **`models_pilot.yaml` keeps `total_device_mb: 4096`.** The model tier — Grounding DINO Tiny, MobileSAM, the letterbox fallback — stands, and its 4 GB argument stands with it.
2. **Every GPU process enforces the cap at startup** — `torch.cuda.set_per_process_memory_fraction()` derived from `DHAKASCENES_VRAM_CAP_MIB=4096` — and a forward pass that exceeds it is the §7.2 hard stop, exercised, not just defined.
3. **Every manifest records** `vram_cap: {value_mib: 4096, enforced: "synthetic", physical_device_mib: 24564, device_name: "NVIDIA GeForce RTX 4090"}`. A manifest whose cap block says the card is 4096 physical is a lie; a manifest with no cap block on this machine is worse.
4. **The §13.2 claim is downgraded, not preserved:** what a capped 4090 run supports is *"runs under a synthetic 4096 MiB ceiling on a 24 GB card"*. A synthetic cap does not reproduce a small card's fragmentation behaviour, a laptop's display-server allocation, or its thermals. The claim *"runs on 4 GB VRAM"* becomes quotable only after **one end-to-end confirmation run on the physical 3050ti** — record this as a declared deferred verification in the claim table and in `BUILD_STATE.md`, so it cannot be silently forgotten.
5. Phase 5b's `system_reserve_mb` measurement is of **this** machine and does not transfer to the laptop. Record it; do not quote it as the laptop's margin.
6. The 24 GB card makes the `models_production.yaml` tier *measurable* for the first time. That is out of pilot scope — note the possibility in `DECISIONS.md`, do not act on it.

**Unacceptable, unchanged:** running uncapped while the config says `4096`, or a `verified: true` flag on a number measured without the cap in force.

---

### C2 — The bible's own status header is false

**Plan says:** line 3 — *"**Status:** planning document. Nothing described here has been built yet."*

**Disk says:** ~15,000 lines, Phases 1/3/4 executed, 1.7 GB of derived clouds, a priors file.

**Why load-bearing:** a reader trusting the header rebuilds from scratch and destroys working, provenance-annotated code. And nobody can tell from the bible which of the ten phases has passed its gate.

**Resolution:** a `## Status` block at the top of `pilot_plan.md` pointing at **`docs/BUILD_STATE.md`** — a per-phase table of `NOT STARTED / CODE WRITTEN / RUN / GATE PASSED`, **generated** by `scripts/build_state.py` from what is on disk (module presence, `_SUCCESS` markers, `run_manifest.json` presence, that phase's test selection result, and the conformance ledger). Hand-maintained status drifts; that drift *is* C2.

---

### C3 — `manifest.py` does not exist, and its responsibilities leaked into stage code

**Plan says:** §2 places `manifest.py` in `pipeline/common/` — *"run_manifest.json, atomic write, `_SUCCESS`, seeds (§1.9)"*. Phase 2 lists it as a deliverable. §14 calls contracts-first *"the correct dependency structure and… also where every P0 finding turned out to be fixable."*

**Disk says:** no `pipeline/common/manifest.py`. Instead:
- `write_json_atomic()` at **`pipeline/stage0_data_probe/probe.py:739`**
- `write_jsonl_atomic()` at **`pipeline/stage3_proposals/proposals.py:752`**
- Stages 1, 2, 4, 5, 6, 7, 8 import them from stage0 and stage3. `stage5_lift` imports from stage0, stage1, **and** stage3.

**Why load-bearing:** the dependency arrow points stage→stage instead of stage→common. Stage 5 cannot be tested without importing Stage 3. `_SUCCESS` semantics, upstream-fingerprint refusal, seed threading and the §1.9 field set have no single owner, so each stage implemented whichever subset its author needed — which is how Stage 1's manifest ended up missing four mandatory fields (C5). It also proves Phase 2's exit gate (*"all `common/` tests green… **no stage code written yet**"*) was never met.

**Resolution:** write `pipeline/common/manifest.py` as sole owner of atomic write, `_SUCCESS`, the §1.9 field set as a typed record, seed threading, idempotent re-run keying, upstream refusal, and the §13.2 banner constant. Migrate the two writers into it, update all eight importers, and add the test that **nothing under a stage package imports another stage package**. Without that test the leak grows back.

---

### C4 — There are no tests, so no phase gate from Phase 2 onward has been passed

**Plan says:** §9 is a full test plan with a `Replace — would pass on broken code` column. Every phase 2–10 carries a `Tests:` line. Phase 2's gate is *"all `common/` tests green"*. Phase 7's is *"paint-inside-GT rate is high and **stated as a number**"* — *"the single strongest correctness signal available in the whole pilot."*

**Disk says:** `tests/` does not exist. Zero test files.

**Why load-bearing:** Phases 3 and 4 have been **run**, and their outputs are already consumed as trustworthy. They may well be correct — the code reads as careful — but nothing has demonstrated it, and the failures the plan guards against are the ones that look fine.

**Resolution:** build `tests/` before any new stage code, and **retro-fit the Phase 2/3/4 tests against the artifacts already on disk.** Build the §9 table in full, including every `Add` entry. Wire `DHAKASCENES_RUN_INTEGRATION` / `DHAKASCENES_RUN_GPU_TESTS` so integration tests **fail rather than skip** when the switch is on and data is absent (§9: *"so they never silently false-pass against absent data"*).

Highest-value first — the ones that would catch a live wrong answer:
1. **GT-box reprojection** and **paint-inside-GT rate** (Phase 7). Use `nuscenes-devkit` as the **oracle**: an independent implementation sharing no code with `pipeline/common/conventions.py`. A test computing its expected value with the code under test proves nothing — which is why `pyquaternion` is pinned but never imported by the pipeline.
2. **µs→ns against a real `sample_data` record pair** (§1.2). The plan calls rev 1's synthetic version *"the clearest instance in rev 1 of a test that proves nothing."*
3. **`LIDAR_TOP` calibration is non-identity**, expecting the measured **−89.883°** (§1.1).
4. **Positive-rejection provenance tests** (§1.7): a `val` + `pipeline_accepted` record **must raise**; `human_verified` with `verified_by=None` **must raise**. The negative form is vacuously true against a completely broken validator.
5. **Determinism**: same scene twice, byte-compare (§1.9).
6. **Non-square rectangle at 30° yaw** for the L-shape fit (§5.7). A square fixture hides both the `[w,l]` swap and the 90° ambiguity.

---

### C5 — Not a git repo, and Stage 1's manifest is missing four mandatory §1.9 fields

**Plan says:** §1.9 — every manifest carries *"seed, resolved config hash, **git commit**, package versions (Python / PyTorch / CUDA / UMAP / HDBSCAN / sklearn), checkpoint IDs + revisions + SHA-256, dataroot fingerprint, `usable_scenes.json` hash, scene partition + its seed, priors version and `source`, `W_acc` count and duration, actual image resolution and prompt configuration, measured VRAM peaks per role, per-stage input/output counts."*

**Disk says:** `git rev-parse --is-inside-work-tree` → `fatal: not a git repository`. Stage 1's manifest, audited field by field:

| Field | Present |
|---|---|
| seed, dataroot fingerprint, upstream `usable_scenes` ref, `w_acc_count`, `w_acc_duration_ns`, per-filter I/O counts | ✅ |
| **git commit** | ❌ |
| **resolved config hash** | ❌ |
| **package versions** | ❌ — only `python_version`, `numpy_version` |
| **scene partition** | ❌ — in `usable_scenes.json`, not carried forward |

**Why load-bearing:** Phase 10's gate is *"the run is reproducible from the manifest by a second person."* Without a commit hash the manifest identifies the config but not the code — and this repository has 15,000 lines and no version history at all, so one bad edit is unrecoverable. **This is also the precondition for Activity C: you may not restructure an unversioned tree.**

**Resolution:** `git init` and commit the current tree as the pre-existing baseline **before touching anything**, with the existing `.gitignore`. Then have `manifest.py` (C3) emit the complete §1.9 set, refusing to write when the tree is dirty unless `dirty: true` is recorded beside the commit. Regenerate Stage 1's manifest.

---

### C6 — Two configs are referenced by code that cannot run without them

**Disk says:** code references `configs/models_pilot.yaml` (2 sites) and `configs/pipeline_pilot.yaml` (1 site). Neither exists. Only `paths.yaml` and `taxonomy_pilot_nuscenes.yaml` do.

**Why load-bearing:** those paths have never executed, and the §10 tunables currently live as defaults scattered through Python rather than as config. Stage 1's manifest shows the target: sixteen config keys, each with a provenance string, several reading `"arbitrary, needs tuning"`. That is the standard.

**Resolution:** write `configs/pipeline_pilot.yaml` covering the full §10 list, and `models_pilot.yaml` / `models_production.yaml` per §7.2/§7.3 — each per-role entry carrying checkpoint id **+ revision + SHA-256 + VRAM estimate + `verified: true/false`**. Then the test that **every** §10 tunable resolves from config and **no** stage holds a magic number.

---

### C7 — Phase order has already been violated; decide what that invalidates

**Plan says:** §12 — *"**Do not start phase N+1 until phase N's exit gate passes** — the failure this ordering prevents is building against an *assumed* upstream output shape."*

**Disk says:** code for Stages 2–8 exists and `priors_pilot_v0.json` (a **Phase 8** deliverable) has been produced, while Phases 5/6/7 have never run and no gate from 2 onward is verifiable (C4).

**Why load-bearing:** the priors file feeds ε and dimension means to Stages 6 and 8. It was derived without Phase 7's paint-inside-GT number, so nothing has confirmed the projection chain those clusters depend on.

**Resolution:** do **not** delete the work. Re-gate it: every existing artifact is *unverified until its phase's tests pass against it*, recorded in `BUILD_STATE.md`. Then walk the gates in order from Phase 2. Where a gate fails, regenerate — don't patch the gate.

---

### C8 — M-5 is unresolved, and it decides what ρ actually measures

**Audit (M-5):** *"ρ's definition in the pilot: is it computed over GT boxes or pipeline output? Over pipeline output it is a function of the detector, not the scene."*

**Plan says:** §1.10 — *"ρ remains **count over area**. This survived the audit unchanged and should not be touched."* That answers *how* ρ is normalised. It never answers *what is counted*.

**`comprehensive.md` §8.3.1 says:** ρ = *"(# **annotated** agents whose box center lies in E within radius R = 30 m of ego) / area(E ∩ disc(R))"*.

**Disk says:** `pipeline/common/eval_region.py:315` implements `rho(centers_xy_m, spec, n_min=...)` taking centres from the caller — so the question is pushed to every call site, and no call site exists yet.

**Why load-bearing:** the pilot has no human annotations, so "annotated agents" can only mean pipeline output — making ρ a function of Grounding DINO Tiny's recall, not of the scene. And `comprehensive.md` C2/G5 makes ρ-binned degradation curves *the money figure*. A ρ from a deliberately under-tier detector, binned and plotted, looks exactly like the real thing.

**Resolution:** `rho()` takes an explicit, recorded `rho_source ∈ {"gt", "pipeline_output"}`; both are computable here (nuScenes GT is on disk); every ρ reaching a figure carries its source in the record **and** the figure header. Default `"gt"`, with `"pipeline_output"` available for the plumbing demonstration. **This is a genuine gap in the bible — resolve it and propose the §1.10 amendment.**

`M-4`, `M-9`, `M-10` are likewise never cited by ID in rev 2 but are substantively covered (M-4 by §4's waiver table; M-9/M-10 by P1-9/P1-8). Cross-reference; no further action.

---

### C9 — Stage 9, the probe, and the driver do not exist

`pipeline/stage9_qa/`, `probes/indigenous_prompt_probe/`, `configs/taxonomy_probe_indigenous.yaml`, `scripts/run_pilot.py`, `scripts/check_vram_paper.py` — all absent. Ordinary build work, with two traps:

- **Stage 9's spatial gate runs on *pre-inflation* dimensions** (P1-12). Stage 8 inflates *toward* the class prior and the gate is *"BEV box exceeds 2× class prior"*, so any box through Stage 8 passes more easily and the gate is weakest exactly where it is needed. The plan is explicit this is an **inherited spec flaw the pilot surfaces rather than reproduces**: record both dimensions, gate on the measured one.
- **The probe's separation is by mechanism, not naming** (§8): separate output root, separate code path, separate taxonomy file, `experiment` and `not_evidence_for: "S1"` on every record, and a **test** asserting no indigenous prompt string appears in the pilot taxonomy and no probe record appears in main outputs. `DHAKASCENES_RUN_INDIGENOUS_PROBE=0` by default.

---

### C10 — The claim-hygiene banner exists in exactly one file

**Plan says (§13.2):** the banner goes *"in the pilot README, in `run_manifest.json`, and in the header of **any figure exported from a pilot run**"*.

**Disk says:** the string appears only in `scripts/measure_vram.py`. There is no `README.md`.

**Resolution:** define it once as a constant in `pipeline/common/manifest.py`; emit into every manifest; put it in `README.md`; route figure export through one helper that stamps it. Test that a figure produced without the banner cannot be written.

> **This demonstrates pipeline plumbing only. Label quality is not evidence of anything; models are deliberately under-tier; the substrate is nuScenes v1.0-mini, not Dhaka.**

---

### C11 — The repository structure itself violates the plan — **this is what Activity C is for**

Four separate defects, each measured:

1. **No package, and ten `sys.path.insert` hacks.** `pipeline/stage{2,3,4,5,6,7,8}/*.py`, `stage6_cluster/priors.py`, and both scripts each do
   `sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))`
   followed by imports carrying `# noqa: E402`. There is no `pyproject.toml`, `setup.py`, or `setup.cfg`. Every module mutates global interpreter state at import time to find its own siblings, and import success depends on file location rather than on installation. This is the same class of defect as C3 — a path model that is a convention instead of a contract, which §1.8 exists to forbid.
2. **Package root name.** §2 draws `dhakascenes_pilot/` as the top directory; on disk `pipeline/`, `configs/`, `scripts/`, `docs/` sit directly under `/home/mt/Zami`.
3. **The 15 GB dataroot is inside the repo root.** `/home/mt/Zami/nuscenes/`. §1.8's stated hazard is precisely this: *"A symlink inside the project tree is copied by `shutil.copytree`, `rsync -L`, or a Docker build context."* A real directory is worse than a symlink. `validate_paths()`'s disjointness check passes only because it compares dataroot against the *write* roots, which are at `/home/mt/dhakascenes/` — it never checks dataroot against the repo.
4. **`priors/` location is ambiguous in the bible.** §2 places `priors/` inside the repo and §6 says *"Write `priors/priors_pilot_v0.json`"*, while §1.8 requires outputs outside the repo. The code chose `out_root` (`/home/mt/dhakascenes/out/priors/`). That is the better reading — priors are a generated artifact, and the nuScenes licence bears on redistributing GT-derived statistics (§6 `[VERIFY]`, and C12). Ratify it and amend §2.

**Resolution:** restructure, under the protocol in §6. Defects 1 and 3 are not stylistic — they are the plan's own §1.8 contract, unimplemented.

---

### C12 — The §6 licence `[VERIFY]` is still open and now blocks a commit

**Plan says (§6):** *"**Licence note [VERIFY]:** the nuScenes licence (CC BY-NC-SA class) bears on redistributing GT-derived statistics. Check before `priors_pilot_v0.json` lands in any public repo."*

**Disk says:** `priors_pilot_v0.json` exists (23 classes from `sample_annotation.json`) and carries `license_note` and `release_guard` fields — good. `nuscenes/LICENSE` is on disk. But C5 requires `git init`, making "lands in a repo" imminent.

**Resolution:** read `nuscenes/LICENSE`, resolve the `[VERIFY]`, record the finding. `.gitignore` already excludes the out root and `/nuscenes/` — keep it that way and say why.

---

### C13 — Tripwires the plan installed deliberately; do not quietly normalise them

Not contradictions. Each must survive to the end:

- **Ground-removal parameters are foreign.** 0.3 m / 40 m / 4 m were chosen for a **Livox Mid-360 on Dhaka roads**, applied here to a **32-beam spinning LiDAR in Boston/Singapore**. A 0.3 m band removes every wheel return and most of a traffic cone, pushing small classes below the ≥5-point gate. Stage 1's manifest already carries this provenance. Check the per-class consequence at the Phase 8 gate.
- **Stage 2 is not OOD detection.** It is clustering of a non-metric 2-D projection. Outlier decisions happen in **embedding space** (HDBSCAN GLOSH on raw 384-D embeddings); UMAP is visualisation only. Expected sample count is **242 images** (2,424 ÷ 10), not 40 keyframes — state and check it before running; `n_neighbors` must be `< n_samples`.
- **Stage 7 runs at 2 Hz.** 3D IoU between consecutive detections is zero for most vehicles, so association collapses to appearance-only unless predict-then-match propagates the box first. Record effective inter-frame Δt in every track record; the IoU gate is config with provenance `derived for 2 Hz`.
- **Prompt chunking is forbidden** unless re-tuned and re-recorded — it changes confidence semantics, because scores come from token-level logits over the full concatenated caption.
- **Square resizes of 1600×900 imagery are forbidden.** Letterbox with recorded padding, or resize-shortest-side.
- **OOM is a hard stop**, never a silent retry at lower resolution.
- **`tier` ≠ `source`**, and `allow_human_provenance: false` is pilot-wide.

---

## 6. ACTIVITY C — restructure

**You have explicit authority to restructure the repository, including moving every file.** C11 establishes that the current structure violates §1.8 and §2 of the bible. Restructuring is not optional if the verification sweep confirms C11.

### 6.1 Preconditions — all four, no exceptions

1. `git init` done, baseline committed, working tree clean (C5).
2. The conformance ledger is complete (Activity A). **You must know what you have before you move it.**
3. C1's resolution (two-device fleet, synthetic 4096 MiB cap) carried into `DECISIONS.md` verbatim.
4. **Golden outputs captured**: run Stage 0 and Stage 1 on one scene under the *current* structure and record SHA-256 of every output file. This is your behaviour-preservation oracle, and it reuses the determinism machinery §1.9 already requires.

### 6.2 Target layout

```
/home/mt/Zami/                        # git root
  pyproject.toml                      # installable -> deletes all 10 sys.path hacks
  README.md                           # §13.2 banner
  requirements.txt  requirements-torch.txt  requirements-devkit.txt  requirements-lock.txt
  environment.yml   .env.example  .gitignore
  src/
    dhakascenes_pilot/                # §2's package name, made real
      common/
        conventions.py  schemas.py  eval_region.py
        paths.py  manifest.py  model_interfaces.py
      stages/
        stage0_data_probe/ … stage9_qa/
      priors/                         # consumed by BOTH stage6 and stage8 -> not stage6's property
      probes/indigenous_prompt_probe/ # separate code path (§8)
  configs/
  scripts/
  tests/  unit/  integration/  fixtures/
  docs/
```

Then `pip install -e .` into `ano_pipe`, add it to `requirements.txt`, and delete every `sys.path.insert` and its `# noqa: E402`.

**Move the dataroot out of the repo** — `/home/mt/Zami/nuscenes` → `/home/mt/dhakascenes/nuscenes` (same filesystem, so `mv` is instant), and update `configs/paths.yaml`. Extend `validate_paths()` to assert dataroot is disjoint from the **repository root**, not only from the write roots — the current check passes while the hazard is live.

> **Consequence you must handle, not discover:** `usable_scenes.json` records the dataroot **realpath**, and §1.8 says *"every consumer verifies the match."* Moving the dataroot invalidates the allowlist by design — which is the contract working correctly. Re-run Stage 0 (no GPU, cheap) and let the fingerprint bind to the new path. If moving the dataroot does **not** invalidate the allowlist, that is a `VIOLATES` row: the binding is not enforced.

### 6.3 Migration protocol

- **`git mv` only.** One commit per coherent move, message naming the plan section it satisfies.
- **Restructure commits contain no behaviour change.** Ever. If you find a bug mid-move, finish the move, commit, then fix it in a separate commit. Mixed commits make the golden-output comparison meaningless.
- **Fix imports by codemod, not by hand.** Ten files, one mechanical rewrite; hand-editing is where a stage quietly keeps importing another stage.
- **Re-run the golden oracle after every commit.** Byte-identical, or the move was not mechanical.
- **Do not restructure and build simultaneously.** Activity C completes and is verified before Activity D begins.

### 6.4 When restructuring is *not* warranted

If the sweep shows the `sys.path` hacks and the in-repo dataroot are the only structural findings, defects 1 and 3 alone still justify the move — they are unimplemented §1.8 contract, not taste. But **do not invent further reorganisation.** Renaming modules, splitting files by size, or introducing an abstraction layer the plan does not ask for is scope you were not given, and it destroys the provenance comments that make this codebase auditable.

---

## 7. Method of work — non-negotiable

1. **Silent-failure-first.** Before writing a stage, write down the failure modes producing *plausible wrong output without crashing*, and the test catching each. Put them in the module docstring. The existing code does this well — match it.
2. **Contracts before consumers.** Nothing in a stage package may import another stage package. Enforce with a test.
3. **Gate discipline.** Do not start phase N+1 until phase N's gate passes. When a gate fails, regenerate the artifact — never weaken the gate.
4. **Every number you state is measured.** Phase 7's gate is *"paint-inside-GT rate is high and **stated as a number**"*. "High" is not a result.
5. **Every tunable is config with a provenance note** — `spec §x.y` / `measured on N frames, date` / `arbitrary, needs tuning`. Stage 1's config block is the reference implementation.
6. **Determinism is a feature.** One global seed threaded into RANSAC, UMAP and all sampling; deterministic tie-breaks; `PYTHONNOUSERSITE=1`; `CUBLAS_WORKSPACE_CONFIG` set; TF32 and cuDNN benchmark off. Byte-compare a re-run.
7. **Fail closed.** A `null` σ from I-2 is an error, not a default. A missing upstream manifest refuses the start. A missing required field raises at `write_records()`.
8. **Small, verified commits**, message naming the plan section and audit finding closed.

---

## 8. ACTIVITY D — the build, in order

Phase numbering is the plan's §12.

**Phase 0 — reconciliation (new, blocking).** `git init` + baseline. Conformance ledger (Activity A). `docs/DECISIONS.md` with C1–C13 **plus everything the sweep found** (C1 is already human-resolved — transcribe it, don't reopen it). `docs/BUILD_STATE.md` + `scripts/build_state.py`. `README.md` with the banner. Restructure per §6 if warranted.
*Gate:* every ledger row has a status; every register entry has a resolution; C1's decision transcribed into `DECISIONS.md`; golden outputs byte-identical across the restructure.

**Phase 2 (re-open) — complete `pipeline/common/`.** `manifest.py` (C3, C5, C10). `configs/pipeline_pilot.yaml` + the two model YAMLs (C6). `tests/unit/` + `tests/fixtures/` and the §9 cross-cutting row. `rho_source` on `eval_region.rho()` (C8). The no-stage-imports-stage test and `test_env_contract.py`.
*Gate:* all `common/` tests green; every §10 tunable resolves from config; C1's decision reflected in `models_pilot.yaml`.

**Phase 3/4 (re-gate) — validate what already ran.** Retro-fit Stage 0's tests (truncated `.pcd.bin` at a multiple of 20 bytes, dangling `ego_pose`, missing sweeps, fingerprint mismatch) and Stage 1's (sloped multi-sector ground, real-record µs→ns, static-structure sharpness, scene-start `n_sweeps_actual`, byte-compare determinism) **against the artifacts on disk**. Backfill manifests to the full §1.9 set.
*Gate:* every retro-fitted test green against existing outputs, or the outputs regenerated.

**Phase 5 — registry and VRAM.** 5a: `check_vram_paper.py` against §13.1's *corrected* figures — Grounding DINO Tiny ≈ **172 M params / ~690 MB FP32** (not "172 MB"); SAM-ViT-B ≈ **91 M params / ~375 MB** (not "375 M params"). 5b: download checkpoints, one forward pass per role at real resolution and the real 23-phrase prompt, measured with `max_memory_reserved()` and `mem_get_info()` free-deltas — **not** `max_memory_allocated()`. Flip `verified:` per role. Assert `memory_allocated() ≈ 0` between teardowns. Fetch `mobile_sam.pt` and record its SHA-256.
*Gate:* measured peaks in the manifest **with the synthetic cap in force and its block recorded**; hard-stop behaviour defined **and exercised** (drive one role past 4096 MiB deliberately and show the hard stop fires); no `verified: true` on any uncapped measurement.

**Phase 6 — Stages 3 and 4.** Per-class thresholds tuned on the **`tuning` subset only** (`scene-0655`, `scene-1094`). Phrase→class span mapping tested. No chunking. Masks asserted at 1600×900. IoA-NMS > 0.5 across cameras.

**Phase 7 — Stage 5 lift.** The decisive phase. GT-box reprojection and paint-inside-GT against the devkit oracle, plus behind-camera cull, near-zero depth, out-of-bounds, two-camera overlap determinism, lidar→pixel→lidar round-trip.
*Gate:* **paint-inside-GT rate stated as a number.**

**Phase 8 — Stages 6 and 8.** Re-derive priors under E and the 40 m cap from the `priors` subset if Phase 7 invalidates the existing file. ε from `comprehensive.md` §7.2's formula, never `Annotation_pipeline.md`'s table. Non-square-rectangle-at-30° test. Anchor-direction test. `inflated` + `inflation_fraction` on every box.

**Phase 9 — Stages 7 and 2.** Predict-then-match, Hungarian, stated gates, birth/death, minimum crop size, ICP with a **declared frame** (ego-frame ICP measures motion relative to ego; nuScenes mAVE is absolute), fully-specified Kalman fallback, yaw-consistency enforcement. Stage 2 with image-level sampling and GLOSH on raw embeddings.
*Gate:* the **≥3-frame stable-track** integration test passes on one real scene.

**Phase 10 — Stage 9, end-to-end, claim hygiene.** Stage 9 gating on pre-inflation dimensions and single-sweep counts. `scripts/run_pilot.py` over the **`run` subset**. Failure isolation exercised by deliberately corrupting one scene. The probe in its separate root. Determinism re-run.
*Gate:* reproducible from the manifest by a second person; banner in README, manifest, and every figure header.

---

## 9. Prohibitions

- **Do not report `CONFORMS` on the strength of having read the code.** `CODE-SITE` evidence supports `PLAUSIBLE` and nothing stronger.
- Do not delete or rewrite existing pipeline code because it is easier than reading it. It encodes decisions, and its provenance comments are the audit trail.
- Do not weaken a test to make a gate pass.
- Do not mix a restructure commit with a behaviour change.
- Do not add a dependency without a pin and a provenance note in `requirements.txt`.
- Do not install into the system or user-site interpreter. `ano_pipe` + `PYTHONNOUSERSITE=1`.
- Do not write to the dataroot. It is read-only by convention and there is a test for it.
- Do not let probe output reach `out_root` or the priors.
- Do not quote any VRAM figure marked `verified: false`, and do not quote any metric without §11 decision 4's framing in the same sentence.
- Do not lift or raise the synthetic VRAM cap (**C1**) without the human — it is what keeps this 4090's results transferable to the 4 GB laptop the pilot exists for.
- Do not report a phase complete when only its code exists. **Code written ≠ run ≠ gate passed** — that distinction is what `BUILD_STATE.md` exists to keep honest.

---

## 10. What you report back

1. **`docs/conformance.yaml` + `docs/CONFORMANCE.md`** — the ledger, with a summary count by status. This is the primary deliverable of Activity A.
2. **`docs/DECISIONS.md`** — every contradiction, every status.
3. **`docs/BUILD_STATE.md`** — machine-generated, per-phase.
4. **The restructure record** — what moved, why, and the golden-output comparison proving behaviour was preserved.
5. **Measured numbers, not adjectives** — usable scene count, per-camera Δt medians against §1.2's table, paint-inside-GT rate, per-role measured VRAM, track lengths, per-filter retention.
6. **A deviation register** — every departure from `comprehensive.md`, with the reason. §4 of the plan is the seed; add to it.
7. **Anything this survey missed**, stated as plainly as C1–C13 are stated here.

---

## Appendix — measured substrate facts

Verified 2026-08-12 by direct computation over `nuscenes/v1.0-mini/*.json`. Confirm on your own run; do not re-derive from scratch.

| Fact | Value |
|---|---|
| Metadata | `v1.0-mini`, 13 tables, fingerprint `4c5a5cfe88c837989e692a98159b454c9ae39e57c6ad51fb066ccb9cb48d7891` |
| Scenes / keyframes / `sample_data` | **10 / 404 / 31,206** |
| Categories / instances / annotations | **23 / 911 / 18,538** |
| Missing files across 12 channels | 0 |
| Keyframe images / resolution | **2,424** / **1600 × 900**, uniform |
| `CAM_FRONT` intrinsics | fx = fy = **1266.417**, cx = **816.267**, cy = **491.507** — **first record only; see erratum E1** |
| `LIDAR_TOP` extrinsic quaternion | `[0.70780, −0.00649, 0.01065, −0.70631]` — first record; see erratum E1 |
| `LIDAR_TOP` extrinsic yaw | **−89.883°** — not identity; the §1.1 test that must exist. **Per-scene: see erratum E1** |
| LiDAR cadence / sweeps per keyframe | 49.79 ms median (≈20 Hz) / 9.74 |
| Point record | 20 bytes (5 × float32) |
| Timestamps | 16-digit microseconds, Unix epoch |
| Per-camera Δt medians (ms) | FL −43.06 · F −35.44 · FR −27.53 · BR −19.95 · B −10.36 · BL −0.48 |
| Partition (verified on disk) | priors `0061/0103/0553/1077` · tuning `0655/1094` · run `0757/0796/0916/1100` |
| Stage 1 retention | 146,137,792 → 83,505,441 points (0.571); compensation residual 7.02e-13 m |
| Host | RTX 4090, 24080 MiB, sm_89, driver 595.84 · 32 cores · 62 GB RAM · 407 GB free |

### Errata to the bible's §0/§1 constants (E1) — verified independently by two sessions, 2026-08-12

Calibration in nuScenes is **per-scene** (`calibrated_sensor` is per log), and the plan's
single constants are only the first record:

| Erratum | Measured |
|---|---|
| `CAM_FRONT` has **two** calibrations across the 10 scenes | fx 1266.417 (×6) **and** 1252.813 (×4) |
| `LIDAR_TOP` has 10 per-scene extrinsics, **two distinct yaws** | −89.883° (×6) **and** −90.031° (×4) |
| `CAM_BACK_LEFT` Δt can be **positive** | max **+1.20 ms** — "every camera fires before the anchor" fails at the boundary |

Consequences (binding on Phase 2 tests — see `DECISIONS.md` C14): the §1.1 non-identity
test asserts yaw ≈ **−90° ± 0.5° per record**, never one constant; nothing caches one
scene's `K` or `T_ego_lidar` for another, and a test pins that; per-camera Δt sanity
bands must admit the +1.2 ms boundary.
