#!/usr/bin/env python3
"""Stage 2 -- DINOv2 OOD / long-tail frame discovery (S5.3, Phase 9). A BRANCH.

**Stage 2 feeds nothing else in the pilot.** It is not link 2 of a 0->9 chain:
it can be built and run out of order (scheduled next to Stage 7 in Phase 9
because both are DINOv2 consumers, per pilot_plan.md S5.3), it consumes only
Stage 1's keyframes, and no confirmed-novel-label loop exists downstream of it
-- nuScenes' taxonomy is already known and there is nothing this pilot can add
to it. Running this stage proves the embedding -> projection -> clustering
path end to end; it changes no taxonomy and gates no other stage.

**This is not OOD detection, and the code below never claims it is.** HDBSCAN
membership in a UMAP projection has no formal relationship to outlyingness in
the 384-D embedding space UMAP preserves neither density nor global structure.
**UMAP output is written to every row for visualisation only and is never read
by the flagging decision** -- the outlier call is GLOSH (`outlier_scores_`),
computed by HDBSCAN directly on the RAW embeddings, before any 2-D projection
exists. `flag_ood()`, the one function that decides `is_ood`, takes precomputed
scores and a threshold; it has no UMAP coordinate in its argument list, which
is the enforcement mechanism, not a comment promising one.

**Sampling is over images, not keyframes** (pilot_plan.md S5.3, M-15). At
1-in-10, 404 keyframes give 40 samples -- HDBSCAN on 40 points returns one
cluster plus noise and any "not zero, not thousands" success criterion is
unfalsifiable. 404 keyframes x 6 ring cameras give ~2,424 images; 1-in-10 over
THAT list gives ~242, which is still at the edge but at least the right order
of magnitude. `list_images()` enumerates every (keyframe, channel) pair across
every requested scene and `sample_images()` strides over that flat list with a
seeded phase offset -- never a stride over the keyframe list with a 6x
multiplier applied after the fact, which is the same 40-sample failure wearing
a bigger denominator.

**Degenerate-N is a declared failure mode, not a crash.** UMAP requires
`n_neighbors < n_samples` and a sample count near the configured
`n_neighbors`/`min_cluster_size` produces a meaningless fit either way, so
both are clamped when the sampled set is smaller than their configured value,
the clamp is recorded (`umap.n_neighbors_used` /
`hdbscan.min_cluster_size_used`), and the run degrades rather than raising.
The expected sample count is compared against `min_meaningful_samples` before
either library runs and the shortfall is stated in the manifest, not silently
absorbed into "0 flagged."

    python3 -m pipeline.stage2_ood.ood [--paths configs/paths.yaml]

Exit codes:
    0  ran with a meaningful sample count and produced a clustering
    1  ran, but the sampled set was below `min_meaningful_samples`
    2  upstream contract broken, or a required library/checkpoint is
       unavailable; nothing was written
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pipeline.common.model_interfaces import (  # noqa: E402
    EMBEDDING_OOD,
    ORIGINAL_SIZE_PX,
    WHOLE_IMAGE_CLS,
    CheckpointSpec,
    EmbeddingBatch,
    RoleContractError,
    register,
)
from pipeline.common.paths import (  # noqa: E402
    PathValidationError,
    Paths,
    assert_dataroot_read_only,
    load_paths,
    metadata_fingerprint,
)
from pipeline.common.schemas import RING_CAMERAS, KeyframeRecord, read_records  # noqa: E402
from pipeline.common.manifest import (  # noqa: E402
    UpstreamRefusal,
    clear_markers,
    require_upstream,
    write_json_atomic,
    write_jsonl_atomic,
    write_marker,
)
from pipeline.stage3_proposals.proposals import ModelUnavailable  # noqa: E402

STAGE = "stage2_ood"
STAGE_SPEC = "dhakascenes-pilot/stage2_ood/v1"

EXIT_OK = 0
EXIT_DEGRADED = 1
EXIT_REFUSED = 2


class OODContractError(RuntimeError):
    """A Stage 2 input, or a library's output, violated the contract this stage assumes."""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OODConfig:
    """Stage 2 tunables. None of these may appear as a literal in the code below."""

    # --- model ---
    model_id: str = "facebook/dinov2-small"
    revision: str | None = None
    device: str = "cuda"
    embed_batch_size: int = 16

    # --- sampling (S5.3, M-15) ---
    sample_every_n_images: int = 10
    min_meaningful_samples: int = 100

    # --- UMAP: visualisation only, never read by the flagging decision ---
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    umap_n_components: int = 2

    # --- HDBSCAN + GLOSH, on the RAW embeddings ---
    hdbscan_min_cluster_size: int = 5
    hdbscan_min_samples: int | None = None
    glosh_outlier_threshold: float = 0.75

    # --- upstream gate (C16) ---
    accept_degraded_upstream: bool = False

    # --- determinism (S1.9) ---
    global_seed: int = 20260812

    provenance: dict = field(
        default_factory=lambda: {
            "model_id": "pilot tier, S7.1 role table; DINOv2 ViT-S/14",
            "sample_every_n_images": (
                "pilot_plan.md S5.3 / M-15: 1-in-10 OVER IMAGES (~242 of ~2,424), not over "
                "keyframes (~40 of 404) -- the latter is unfalsifiable for HDBSCAN"
            ),
            "min_meaningful_samples": (
                "arbitrary threshold below which a clustering result is declared, not trusted; "
                "pilot_plan.md S5.3 asks only that the expected count be STATED and CHECKED"
            ),
            "umap_n_neighbors": "config, not spec-copied; Small-model embeddings will not "
            "transfer Large's tuning. Clamped when the sample is smaller than this value",
            "hdbscan_min_cluster_size": "config, not spec-copied; same caveat as umap_n_neighbors",
            "glosh_outlier_threshold": (
                "arbitrary pilot value, needs tuning -- NEITHER governing document states a "
                "numeric GLOSH threshold; this pilot invents one rather than leaving the flag "
                "undefined, and says so here rather than presenting it as derived"
            ),
            "accept_degraded_upstream": "C16 -- consuming a DEGRADED (complete, quality-flagged) "
            "Stage 1 output is an explicit recorded decision, never a default",
            "global_seed": "S1.9, one global seed; threaded into UMAP's random_state and the "
            "sampling stride's phase offset",
        }
    )

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.sample_every_n_images < 1:
            errors.append(f"sample_every_n_images={self.sample_every_n_images} must be >= 1")
        if self.min_meaningful_samples < 2:
            errors.append(f"min_meaningful_samples={self.min_meaningful_samples} must be >= 2")
        if self.umap_n_neighbors < 2:
            errors.append(f"umap_n_neighbors={self.umap_n_neighbors} must be >= 2")
        if not 0.0 <= self.umap_min_dist < 1.0:
            errors.append(f"umap_min_dist={self.umap_min_dist} must be in [0, 1)")
        if self.umap_n_components < 2:
            errors.append(f"umap_n_components={self.umap_n_components} must be >= 2 (a projection, not a scalar)")
        if self.hdbscan_min_cluster_size < 2:
            errors.append(f"hdbscan_min_cluster_size={self.hdbscan_min_cluster_size} must be >= 2")
        if not 0.0 <= self.glosh_outlier_threshold <= 1.0:
            errors.append(f"glosh_outlier_threshold={self.glosh_outlier_threshold} must be in [0, 1]")
        if self.embed_batch_size < 1:
            errors.append(f"embed_batch_size={self.embed_batch_size} must be >= 1")
        return errors

    def as_dict(self) -> dict:
        return dict(self.__dict__)


# ---------------------------------------------------------------------------
# Image-level enumeration and sampling (S5.3, M-15)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ImageRef:
    """One (keyframe, channel) image -- the sampling UNIT this stage uses."""

    scene_name: str
    scene_token: str
    keyframe_token: str
    t_ns: int
    time_base: str
    channel: str
    image_path: str  # relative to dataroot

    def as_dict(self) -> dict:
        return {
            "scene": self.scene_name,
            "scene_token": self.scene_token,
            "keyframe_token": self.keyframe_token,
            "t_ns": self.t_ns,
            "time_base": self.time_base,
            "channel": self.channel,
            "image_path": self.image_path,
        }


def list_images(stage1_dir: str, scene_names: Sequence[str]) -> list[ImageRef]:
    """Every (keyframe, channel) pair across the requested scenes, canonically ordered.

    The IMAGE is the sampling unit (S5.3): this enumerates one entry per
    camera exposure, not one per keyframe with a camera count multiplied in
    afterwards -- the latter reintroduces the keyframe-level stride the
    correction exists to remove.
    """
    images: list[ImageRef] = []
    for scene_name in scene_names:
        path = os.path.join(stage1_dir, "scenes", scene_name, "keyframes.jsonl")
        if not os.path.isfile(path):
            raise UpstreamRefusal(f"{path} not found; run `python3 -m pipeline.stage1_ingestion.ingest` first")
        for keyframe in read_records(path, expect_type=KeyframeRecord):
            for channel in RING_CAMERAS:
                observation = keyframe.cameras.get(channel)
                if observation is None:
                    continue
                images.append(
                    ImageRef(
                        scene_name=scene_name,
                        scene_token=keyframe.scene_token,
                        keyframe_token=keyframe.keyframe_token,
                        t_ns=keyframe.t_ns,
                        time_base=keyframe.time_base,
                        channel=channel,
                        image_path=observation.path,
                    )
                )
    images.sort(key=lambda im: (im.scene_name, im.t_ns, im.channel))
    return images


def sample_images(images: Sequence[ImageRef], cfg: OODConfig) -> tuple[list[ImageRef], int]:
    """Deterministic 1-in-N stride over the canonically ordered image list.

    A fixed stride, not a random draw: `n_neighbors`/`min_cluster_size` care
    about HOW MANY samples exist and their spread, not about an unbiased
    random subset, and a stride over an already-canonically-sorted (scene,
    time, channel) list spreads the sample evenly across scenes and time
    rather than clustering by whichever scene happens to sort first. The seed
    only picks the phase offset, so the run is reproducible (S1.9) without
    being a fixed "always index 0, 10, 20, ...".
    """
    n = len(images)
    if n == 0 or cfg.sample_every_n_images <= 0:
        return [], 0
    rng = np.random.default_rng(cfg.global_seed)
    offset = int(rng.integers(0, cfg.sample_every_n_images))
    sampled = [images[i] for i in range(n) if (i - offset) % cfg.sample_every_n_images == 0]
    return sampled, offset


# ---------------------------------------------------------------------------
# The embedding_ood role: DINOv2 whole-image CLS token
# ---------------------------------------------------------------------------


class Dinov2OodAdapter:
    """DINOv2 as the `embedding_ood` role: CLS token over the WHOLE frame, ~518x518 input.

    Preprocessing belongs to the role, not the model (P1-2): this transform is
    a whole-image resize, never Stage 7's per-object crop, even on the tiers
    where both roles share a checkpoint.
    """

    def __init__(self, spec: CheckpointSpec, cfg: OODConfig) -> None:
        self._spec = spec
        self._cfg = cfg
        self._device = cfg.device
        self._model: Any = None
        self._processor: Any = None
        self._torch: Any = None

    @property
    def roles(self) -> tuple[str, ...]:
        return (EMBEDDING_OOD,)

    @property
    def spec(self) -> CheckpointSpec:
        return self._spec

    @property
    def device(self) -> str:
        return self._device

    def load(self) -> None:
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:
            raise ModelUnavailable(f"{self._spec.model_id}: {exc}") from exc
        self._torch = torch
        kwargs: dict[str, Any] = {}
        if self._spec.revision:
            kwargs["revision"] = self._spec.revision
        self._processor = AutoImageProcessor.from_pretrained(self._spec.model_id, **kwargs)
        self._model = AutoModel.from_pretrained(self._spec.model_id, **kwargs).to(self._device).eval()

    def unload(self) -> None:
        self._model = None
        self._processor = None
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

    def embed_images(self, images: Sequence[np.ndarray], *, ids: Sequence[str] | None = None) -> EmbeddingBatch:
        if self._model is None:
            raise RuntimeError("adapter is not loaded")
        torch = self._torch
        inputs = self._processor(images=list(images), return_tensors="pt").to(self._device)
        with torch.inference_mode():
            outputs = self._model(**inputs)
        vectors = outputs.last_hidden_state[:, 0, :].float().cpu().numpy()  # CLS token
        model_input = inputs["pixel_values"].shape[-2:]
        return EmbeddingBatch(
            vectors=vectors,
            semantics=WHOLE_IMAGE_CLS,
            preprocessing=f"{self._spec.provider}:whole_image_cls_processor",
            model_input_size_px=(int(model_input[1]), int(model_input[0])),
            source_size_px=ORIGINAL_SIZE_PX,
            ids=list(ids) if ids is not None else None,
        )


def embed_all(
    sampled: Sequence[ImageRef], dataroot: str, adapter: Dinov2OodAdapter, cfg: OODConfig
) -> np.ndarray:
    """(N, D) embeddings, batched, in the same order as `sampled`."""
    from PIL import Image

    vectors: list[np.ndarray] = []
    for start in range(0, len(sampled), cfg.embed_batch_size):
        batch_refs = sampled[start : start + cfg.embed_batch_size]
        images = []
        for ref in batch_refs:
            with Image.open(os.path.join(dataroot, ref.image_path)) as im:
                images.append(np.asarray(im.convert("RGB"), dtype=np.uint8))
        batch = adapter.embed_images(images, ids=[ref.keyframe_token + ":" + ref.channel for ref in batch_refs])
        batch.assert_valid(prefix=f"images[{start}:{start + len(batch_refs)}]: ")
        vectors.append(np.asarray(batch.vectors, dtype=np.float64))
    return np.concatenate(vectors, axis=0) if vectors else np.zeros((0, 0))


# ---------------------------------------------------------------------------
# UMAP -- visualisation only. Nothing downstream of this function decides
# anything; its return value is written to the output rows and read by no
# other function in this module.
# ---------------------------------------------------------------------------


def project_umap(vectors: np.ndarray, cfg: OODConfig) -> tuple[np.ndarray, dict]:
    try:
        import umap
    except ImportError as exc:
        raise ModelUnavailable(f"umap-learn: {exc}") from exc

    n = vectors.shape[0]
    n_neighbors = cfg.umap_n_neighbors
    clamped = False
    if n_neighbors >= n:
        # UMAP requires n_neighbors < n_samples. A sampled set this small is
        # already outside what the stage's own min_meaningful_samples check
        # allows through unremarked (see run()); this clamp only keeps the
        # library from raising when that check has been overridden by hand.
        n_neighbors = max(2, n - 1)
        clamped = True
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=cfg.umap_min_dist,
        n_components=cfg.umap_n_components,
        random_state=cfg.global_seed,
    )
    coords = reducer.fit_transform(vectors)
    ledger = {
        "n_neighbors_configured": cfg.umap_n_neighbors,
        "n_neighbors_used": n_neighbors,
        "n_neighbors_clamped": clamped,
        "min_dist": cfg.umap_min_dist,
        "n_components": cfg.umap_n_components,
        "random_state": cfg.global_seed,
        "role": "visualisation only; never read by flag_ood()",
    }
    return coords, ledger


# ---------------------------------------------------------------------------
# HDBSCAN + GLOSH, on the RAW embeddings -- the actual outlier decision
# ---------------------------------------------------------------------------


def cluster_and_score(vectors: np.ndarray, cfg: OODConfig) -> tuple[np.ndarray, np.ndarray, dict]:
    """(cluster_labels, glosh_outlier_scores, ledger). Runs on RAW (N, D) embeddings.

    `outlier_scores_` is HDBSCAN's GLOSH implementation, computed from the
    condensed cluster tree over the embeddings actually passed in here -- at
    no point does a 2-D coordinate reach this function.
    """
    try:
        import hdbscan as hdbscan_lib
    except ImportError as exc:
        raise ModelUnavailable(f"hdbscan: {exc}") from exc

    n = vectors.shape[0]
    min_cluster_size = cfg.hdbscan_min_cluster_size
    clamped = False
    if min_cluster_size >= n:
        min_cluster_size = max(2, n - 1)
        clamped = True
    clusterer = hdbscan_lib.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=cfg.hdbscan_min_samples,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(vectors)
    if not hasattr(clusterer, "outlier_scores_") or clusterer.outlier_scores_ is None:
        raise OODContractError(
            "hdbscan.HDBSCAN produced no outlier_scores_ (GLOSH) for this fit; the flagging "
            "decision has no scores to read and must not silently default to 'nothing is OOD'"
        )
    scores = np.asarray(clusterer.outlier_scores_, dtype=np.float64)
    scores = np.nan_to_num(scores, nan=0.0)
    ledger = {
        "algorithm": "hdbscan",
        "outlier_method": "glosh",
        "min_cluster_size_configured": cfg.hdbscan_min_cluster_size,
        "min_cluster_size_used": min_cluster_size,
        "min_cluster_size_clamped": clamped,
        "min_samples": cfg.hdbscan_min_samples,
        "n_clusters": int(len(set(labels.tolist()) - {-1})),
        "n_noise": int(np.count_nonzero(labels == -1)),
        "space": "raw_embeddings_384d",
        "space_note": "UMAP output is never read here (P1-8-adjacent OOD-space caveat, S5.3)",
    }
    return labels, scores, ledger


def flag_ood(outlier_scores: np.ndarray, cfg: OODConfig) -> np.ndarray:
    """The ENTIRE flagging decision. Takes precomputed GLOSH scores and a threshold.

    No UMAP coordinate appears in this function's signature. That absence is
    the enforcement of "UMAP for visualisation only" -- not a comment next to
    a function that could, if miswired, read one anyway.
    """
    return outlier_scores > cfg.glosh_outlier_threshold


# ---------------------------------------------------------------------------
# Upstream
# ---------------------------------------------------------------------------


def load_upstream(paths: Paths, stage1_dir: str, *, accept_degraded: bool = False):
    """The C16 gate over Stage 1: manifest, marker, fingerprint, degraded policy."""
    manifest, marker = require_upstream(
        stage1_dir,
        stage_name="Stage 1",
        module_hint="pipeline.stage1_ingestion.ingest",
        current_fingerprint=metadata_fingerprint(paths),
        accept_degraded=accept_degraded,
    )
    return manifest, marker


def scene_names_from_stage1(stage1_dir: str, wanted: Sequence[str] | None) -> list[str]:
    root = os.path.join(stage1_dir, "scenes")
    if not os.path.isdir(root):
        raise UpstreamRefusal(f"{root} not found; Stage 1 wrote no scenes")
    names = sorted(n for n in os.listdir(root) if os.path.isdir(os.path.join(root, n)))
    if wanted:
        missing = sorted(set(wanted) - set(names))
        if missing:
            raise UpstreamRefusal(f"requested scene(s) not present in Stage 1 output: {missing}")
        names = [n for n in names if n in wanted]
    return names


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run(
    paths: Paths,
    stage1_manifest: dict,
    stage1_marker,
    cfg: OODConfig,
    stage1_dir: str,
    out_dir: str,
    scene_names: Sequence[str] | None,
) -> tuple[dict, int]:
    started = time.time()
    errors = cfg.validate()
    if errors:
        raise UpstreamRefusal("; ".join(errors))
    # Any marker still standing describes the PREVIOUS run of this stage; it
    # comes down before the first write (C16).
    clear_markers(out_dir)

    names = scene_names_from_stage1(stage1_dir, scene_names)
    all_images = list_images(stage1_dir, names)
    sampled, phase_offset = sample_images(all_images, cfg)

    degraded = len(sampled) < cfg.min_meaningful_samples

    spec = CheckpointSpec(
        role=EMBEDDING_OOD, provider="dinov2_ood", model_id=cfg.model_id,
        revision=cfg.revision, provenance="pilot tier, S7.1",
    )
    spec_errors = spec.validate(prefix="checkpoint: ")
    if spec_errors:
        raise UpstreamRefusal("; ".join(spec_errors) + " -- pass --revision with the hub commit sha")

    adapter = Dinov2OodAdapter(spec, cfg)
    adapter.load()
    try:
        vectors = embed_all(sampled, paths.dataroot, adapter, cfg)
    finally:
        adapter.unload()

    if vectors.shape[0] >= 3:
        umap_coords, umap_ledger = project_umap(vectors, cfg)
        labels, outlier_scores, hdbscan_ledger = cluster_and_score(vectors, cfg)
        flagged = flag_ood(outlier_scores, cfg)
    else:
        # Too few embeddings for either library to fit anything meaningful.
        # Declared as degraded rather than crashed; every row still gets a
        # well-formed (all-null) clustering block.
        umap_coords = np.zeros((vectors.shape[0], cfg.umap_n_components))
        umap_ledger = {"skipped": True, "reason": "fewer_than_3_samples"}
        labels = np.full(vectors.shape[0], -1, dtype=np.int64)
        outlier_scores = np.zeros(vectors.shape[0])
        hdbscan_ledger = {"skipped": True, "reason": "fewer_than_3_samples"}
        flagged = np.zeros(vectors.shape[0], dtype=bool)

    rows: list[dict] = []
    for i, ref in enumerate(sampled):
        rows.append(
            {
                "spec": STAGE_SPEC,
                **ref.as_dict(),
                "embedding_dim": int(vectors.shape[1]) if vectors.ndim == 2 else 0,
                "cluster_label": int(labels[i]),
                "is_noise": bool(labels[i] == -1),
                "glosh_outlier_score": round(float(outlier_scores[i]), 6),
                "is_ood": bool(flagged[i]),
                "umap_xy": [round(float(v), 5) for v in umap_coords[i]],
                "checkpoint": {"model_id": spec.model_id, "revision": spec.revision, "sha256": spec.sha256},
                "seed": cfg.global_seed,
            }
        )
    write_jsonl_atomic(os.path.join(out_dir, "ood_flags.jsonl"), rows)

    per_scene: dict[str, dict] = {}
    for ref, row in zip(sampled, rows):
        bucket = per_scene.setdefault(ref.scene_name, {"n_sampled": 0, "n_flagged": 0})
        bucket["n_sampled"] += 1
        bucket["n_flagged"] += int(row["is_ood"])
    for name in names:
        per_scene.setdefault(name, {"n_sampled": 0, "n_flagged": 0})

    manifest = {
        "spec": STAGE_SPEC,
        "stage": STAGE,
        "seed": cfg.global_seed,
        "config": cfg.as_dict(),
        "upstream": {
            "metadata_fingerprint": stage1_manifest["upstream"]["metadata_fingerprint"],
            "fingerprint_spec": stage1_manifest["upstream"]["fingerprint_spec"],
            "stage1_spec": stage1_manifest["spec"],
            # C16: a run built on accepted degradation says so in its provenance.
            "stage1_degraded": stage1_marker.degraded,
            "stage1_degraded_causes": list(stage1_marker.causes),
            "accepted_degraded_upstream": cfg.accept_degraded_upstream,
        },
        "paths": paths.as_dict(),
        "checkpoint": {"model_id": spec.model_id, "revision": spec.revision, "sha256": spec.sha256},
        "role": "branch, not a chain link (pilot_plan.md S5.3): consumes only Stage 1, feeds "
        "nothing downstream, and can be run independently of Stages 3-9",
        "sampling": {
            "unit": "image (keyframe, channel), never keyframe alone",
            "unit_source": "pilot_plan.md S5.3 / M-15",
            "sample_every_n_images": cfg.sample_every_n_images,
            "phase_offset": phase_offset,
            "n_images_total": len(all_images),
            "n_images_sampled": len(sampled),
            "expected_sample_count": (
                f"~{len(all_images) // cfg.sample_every_n_images} at 1-in-{cfg.sample_every_n_images}, "
                f"stated and checked against min_meaningful_samples={cfg.min_meaningful_samples} "
                "before clustering, per pilot_plan.md S5.3"
            ),
            "min_meaningful_samples": cfg.min_meaningful_samples,
            "meets_minimum": not degraded,
        },
        "umap": umap_ledger,
        "hdbscan": hdbscan_ledger,
        "flagging": {
            "rule": "glosh_outlier_score > glosh_outlier_threshold",
            "rule_source": "pilot_plan.md S5.3: HDBSCAN GLOSH outlier scores on the ORIGINAL "
            "embeddings; UMAP retained for visualisation only, never read by this rule",
            "threshold": cfg.glosh_outlier_threshold,
            "n_flagged": int(sum(r["is_ood"] for r in rows)),
        },
        "not_ood_detection": (
            "this stage does not claim to detect out-of-distribution frames. HDBSCAN membership "
            "in a UMAP projection has no formal relationship to outlyingness in the 384-D "
            "embedding space; the plumbing (embed -> project -> cluster) is what this run proves, "
            "not a taxonomy-gap finding -- nuScenes' taxonomy is known and there is nothing to "
            "discover (pilot_plan.md S5.3)"
        ),
        "numpy_version": np.__version__,
        "python_version": sys.version.split()[0],
        "elapsed_s": round(time.time() - started, 2),
        "scenes": per_scene,
        "totals": {
            "n_scenes": len(names),
            "n_images_total": len(all_images),
            "n_images_sampled": len(sampled),
            "n_clusters": hdbscan_ledger.get("n_clusters", 0),
            "n_noise": hdbscan_ledger.get("n_noise", 0),
            "n_flagged": int(sum(r["is_ood"] for r in rows)),
        },
    }
    return manifest, EXIT_DEGRADED if degraded else EXIT_OK


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--paths", default="configs/paths.yaml")
    parser.add_argument("--stage1-dir", default=None, help="default <work_root>/stage1_ingestion")
    parser.add_argument("--out-dir", default=None, help="default <work_root>/stage2_ood")
    parser.add_argument("--scenes", nargs="*", default=None, help="subset of Stage 1 scene names")
    parser.add_argument("--sample-every-n-images", type=int, default=None)
    parser.add_argument("--glosh-outlier-threshold", type=float, default=None)
    parser.add_argument("--revision", default=None, help="hub commit sha for the checkpoint; required")
    parser.add_argument("--seed", type=int, default=None, help="override the global seed (recorded)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--accept-degraded-upstream",
        action="store_true",
        help="consume a DEGRADED (complete, quality-flagged) Stage 1 output; recorded (C16)",
    )
    args = parser.parse_args(argv)

    try:
        paths = load_paths(args.paths)
    except PathValidationError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_REFUSED

    stage1_dir = args.stage1_dir or os.path.join(paths.work_root, "stage1_ingestion")
    out_dir = args.out_dir or os.path.join(paths.work_root, STAGE)
    assert_dataroot_read_only(paths, out_dir)

    cfg = OODConfig(
        device=args.device,
        revision=args.revision,
        accept_degraded_upstream=args.accept_degraded_upstream,
        **({"sample_every_n_images": args.sample_every_n_images} if args.sample_every_n_images is not None else {}),
        **({"glosh_outlier_threshold": args.glosh_outlier_threshold} if args.glosh_outlier_threshold is not None else {}),
        **({"global_seed": args.seed} if args.seed is not None else {}),
    )

    try:
        stage1_manifest, stage1_marker = load_upstream(
            paths, stage1_dir, accept_degraded=cfg.accept_degraded_upstream
        )
        manifest, code = run(
            paths, stage1_manifest, stage1_marker, cfg, stage1_dir, out_dir, args.scenes
        )
    except UpstreamRefusal as exc:
        print(f"REFUSING TO START: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    except ModelUnavailable as exc:
        print(f"REFUSING TO START: model/library unavailable: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    except (OODContractError, RoleContractError, PathValidationError) as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_REFUSED

    write_json_atomic(os.path.join(out_dir, "run_manifest.json"), manifest)
    # Three-state marker (S1.9, C16): clean / degraded-with-causes / absent.
    write_marker(
        out_dir,
        manifest["upstream"]["metadata_fingerprint"],
        degraded=code == EXIT_DEGRADED,
        causes=[
            f"sampled {manifest['sampling']['n_images_sampled']} images < "
            f"min_meaningful_samples {manifest['sampling']['min_meaningful_samples']}"
        ]
        if code == EXIT_DEGRADED
        else [],
    )

    s = manifest["sampling"]
    t = manifest["totals"]
    print(f"images total         : {s['n_images_total']}  (unit: keyframe x camera, S5.3)")
    print(f"images sampled       : {s['n_images_sampled']}  ({s['expected_sample_count']})")
    print(f"meets minimum        : {s['meets_minimum']}  (>= {s['min_meaningful_samples']})")
    print(f"clusters / noise     : {t['n_clusters']} / {t['n_noise']}")
    print(f"flagged OOD          : {t['n_flagged']}  (GLOSH > {cfg.glosh_outlier_threshold}, raw embedding space)")
    print("not OOD detection    : plumbing evidence only -- see run_manifest.json.not_ood_detection")
    print(f"wrote {out_dir}")
    return code


register("dinov2_ood", EMBEDDING_OOD, Dinov2OodAdapter)


if __name__ == "__main__":
    raise SystemExit(main())
