# BUILD STATE — generated, do not edit

Generated 2026-08-12 15:38 by `scripts/build_state.py` at commit `9be4b06` (dirty tree).

> **This demonstrates pipeline plumbing only. Label quality is not evidence
> of anything; models are deliberately under-tier; the substrate is nuScenes
> v1.0-mini, not Dhaka.**

`CODE WRITTEN ≠ RUN ≠ GATE PASSED`. A phase whose artifacts exist but whose
tests do not is **RUN (ungated)** — its outputs are unverified (DECISIONS C7).

| Phase | Title | State | Gate | Missing |
|---|---|---|---|---|
| 1 | Substrate truth + path contract | **RUN (ungated)** | NO-HARNESS | — |
| 2 | pipeline/common/ — the representation contract | **CODE PARTIAL** | NO-HARNESS | `configs/pipeline_pilot.yaml`, `configs/models_pilot.yaml`, `configs/models_production.yaml` |
| 3 | Stage 0 probe + scene partition | **RUN (ungated)** | NO-HARNESS | — |
| 4 | Stage 1 ingestion | **RUN (ungated)** | NO-HARNESS | — |
| 5 | Model registry + VRAM (5a paper, 5b measured) | **CODE PARTIAL** | NO-HARNESS | `scripts/check_vram_paper.py`, `configs/models_pilot.yaml`, `run_manifest.json` |
| 6 | Stages 3+4 — proposals + masks | **CODE WRITTEN** | NO-HARNESS | `run_manifest.json`, `run_manifest.json` |
| 7 | Stage 5 lift (paint-inside-GT is the gate number) | **CODE WRITTEN** | NO-HARNESS | `run_manifest.json` |
| 8 | Stages 6+8 — cluster, priors, inflation | **CODE WRITTEN** | NO-HARNESS | `run_manifest.json`, `run_manifest.json` |
| 9 | Stages 7+2 — tracking + OOD branch | **CODE WRITTEN** | NO-HARNESS | `run_manifest.json`, `run_manifest.json` |
| 10 | Stage 9 QA + end-to-end + claim hygiene | **NOT STARTED** | NO-HARNESS | `pipeline/stage9_qa/qa.py`, `scripts/run_pilot.py`, `probes/indigenous_prompt_probe`, `configs/taxonomy_probe_indigenous.yaml`, `run_manifest.json` |

## Deferred verifications (may not be silently forgotten)

- **C1**: the claim *"runs on 4 GB VRAM"* requires one end-to-end run on
  the physical 3050ti laptop. Every run on this machine is under a synthetic
  4096 MiB cap on a 24 GB RTX 4090 and supports only the capped claim.
- **C7**: artifacts produced before tests existed remain unverified until
  their phase's retro-fitted tests pass against them.
