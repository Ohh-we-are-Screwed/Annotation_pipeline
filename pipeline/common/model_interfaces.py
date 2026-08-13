"""Role Protocols and the provider registry (§7, Phase 5a).

Four roles cover every model-dependent stage. Stages 6, 8 and 9 are geometric:
no role, same code on both tiers.

    embedding_ood    Stage 2   DINOv2 ViT-S/14   ->  DINOv2 ViT-L/14
    proposal_2d      Stage 3   G-DINO Tiny       ->  G-DINO Swin-L / SAM 3
    mask_2d          Stage 4   MobileSAM         ->  SAM 2.1 / SAM 3
    reid_embedding   Stage 7   DINOv2 ViT-S/14   ->  DINOv2 ViT-L/14

Three interface properties this module exists to make expressible on day one
(§7.1, closing P1-3 and X-10). Each is cheap now and a stage rewrite later:

  1. **`mask_2d` takes temporal state and a window from the start.** MobileSAM
     has no propagation and its adapter ignores both arguments; SAM 2.1 is
     `(frames, boxes, memory_state) -> masks over time` with a propagation
     window and state re-init at block boundaries. Without the parameters in
     the signature, that swap is a Stage 4 rewrite rather than a config edit.
  2. **One provider may register against several roles.** `GAP_ANALYSIS.md` §6
     names SAM 3 — a unified proposal + mask + track engine — as the likely
     production choice. A one-model-per-role registry cannot express it.
  3. **`proposal_2d` may return masks**, in which case Stage 4 is a
     pass-through (`Proposals.masks is not None`).

**Preprocessing belongs to the role, not to the model (P1-2).** `embedding_ood`
embeds a whole image (CLS token, ~518x518 input); `reid_embedding` embeds a
small object crop, ideally mask-pooled patch tokens. Same weights, different
transform, different output semantics. A registry that hands back "the DINOv2
model" and lets Stage 7 inherit Stage 2's transform upsamples a 28x28 crop ~20x
and produces a similarity dominated by interpolation artifacts. Hence
`EmbeddingBatch.semantics` and `.preprocessing` are required fields, and the
two roles are separate registry entries even when they share a checkpoint.

**The 2D contract (§1.5) is enforced here, not in stage code.** Every 2D
quantity crossing a stage boundary is absolute pixels at the original
1600x900 — `xyxy`, never `cxcywh`, never normalised. Each adapter owns its own
forward and inverse transform. Square resizes of 16:9 imagery are rejected by
`validate()`, not by review: a non-aspect-preserving resize with a uniform
inverse stretches every box along one axis, every mask still looks like a mask,
and every 3D extent is wrong in one axis, consistently.

**The shared-instance VRAM claim is dropped (X-2).** Registry singletons are a
convenience for repeated `create()` calls inside one stage. They are not a VRAM
argument: the policy is serial, one role resident at a time, and under that
policy two copies can never coexist, so reloading costs disk I/O and nothing
else. `release()` drops the reference and calls `unload()`; it cannot enforce a
ceiling. The ceiling assertion — `torch.cuda.memory_allocated() ~ 0` between
role teardowns (§7.1) — lives in `scripts/measure_vram.py`, because this module
is torch-free and importable on a machine with no GPU.

No torch, no CUDA, no checkpoint downloads: numpy, PyYAML and stdlib only.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np
import yaml

from pipeline.common.schemas import IMAGE_HEIGHT_PX, IMAGE_WIDTH_PX

__all__ = [
    "ROLES",
    "EMBEDDING_OOD",
    "PROPOSAL_2D",
    "MASK_2D",
    "REID_EMBEDDING",
    "EMBEDDING_SEMANTICS",
    "WHOLE_IMAGE_CLS",
    "CROP_CLS",
    "MASK_POOLED_PATCH",
    "RESIZE_POLICIES",
    "ORIGINAL_SIZE_PX",
    "PER_FRAME_WINDOW",
    "RoleContractError",
    "RegistryError",
    "CheckpointSpec",
    "PromptConfig",
    "Proposals",
    "TemporalWindow",
    "MaskResult",
    "EmbeddingBatch",
    "ModelRole",
    "EmbeddingOOD",
    "Proposal2D",
    "Mask2D",
    "ReidEmbedding",
    "ROLE_PROTOCOLS",
    "ROLE_METHODS",
    "check_role_conformance",
    "assert_role_conformance",
    "ProviderEntry",
    "register",
    "unregister",
    "provider_names",
    "roles_of",
    "create",
    "release",
    "release_all",
    "resident_providers",
    "VramBudget",
    "ModelConfig",
    "load_model_config",
    "VRAM_CAP_ENV",
    "apply_vram_cap",
]

# ---------------------------------------------------------------------------
# §7.1 Roles
# ---------------------------------------------------------------------------

EMBEDDING_OOD = "embedding_ood"
PROPOSAL_2D = "proposal_2d"
MASK_2D = "mask_2d"
REID_EMBEDDING = "reid_embedding"

ROLES: tuple[str, ...] = (EMBEDDING_OOD, PROPOSAL_2D, MASK_2D, REID_EMBEDDING)

# Output semantics of an embedding. Two roles share one checkpoint and do NOT
# share this (P1-2); a stage that receives the wrong one gets a plausible vector
# of the right dimensionality computed over the wrong pixels.
WHOLE_IMAGE_CLS = "whole_image_cls"        # embedding_ood: CLS over the full frame
CROP_CLS = "crop_cls"                      # reid_embedding: CLS over an object crop
MASK_POOLED_PATCH = "mask_pooled_patch"    # reid_embedding: patch tokens pooled over the mask
EMBEDDING_SEMANTICS: tuple[str, ...] = (WHOLE_IMAGE_CLS, CROP_CLS, MASK_POOLED_PATCH)

# §1.5 rule 3. "square" is absent deliberately and is rejected by name below so
# the failure is a message rather than a silently stretched box.
RESIZE_POLICIES: tuple[str, ...] = ("letterbox", "shortest_side", "none")

ORIGINAL_SIZE_PX: tuple[int, int] = (IMAGE_WIDTH_PX, IMAGE_HEIGHT_PX)  # (w, h)

# A nuScenes category name: dotted, hierarchical, and NOT a prompt string
# (§0.3). Fed to Grounding DINO it tokenises to nonsense and returns
# near-random but non-empty detections — boxes with confidences, on the right
# imagery, meaning nothing. The mapping to natural-language phrases lives in
# configs/taxonomy_pilot_nuscenes.yaml; this pattern is the tripwire.
_DOTTED_CATEGORY_RE = re.compile(r"^[a-z][a-z_]*(\.[a-z][a-z_]*)+$")


class RoleContractError(ValueError):
    """A role input or output violated the interface contract (§1.5, §7.1)."""


class RegistryError(RuntimeError):
    """A provider registration or lookup failed."""


def _err(out: list[str], prefix: str, message: str) -> None:
    out.append(f"{prefix}{message}" if prefix else message)


def _check_role_name(role: object) -> str:
    if role not in ROLES:
        raise RegistryError(f"role={role!r} is not one of {ROLES}")
    return str(role)


# ---------------------------------------------------------------------------
# Checkpoint identity (§7.2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckpointSpec:
    """Identity of one checkpoint bound to one role.

    `verified` mirrors `comprehensive.md` §7.1.1: unverified is fine to *run*,
    never fine to *quote*. It is flipped by `scripts/measure_vram.py` against a
    measured peak on the target card — never by hand, and never from a
    weight-file size, which is not inference VRAM (§13.1).
    """

    role: str
    provider: str
    model_id: str
    revision: str | None = None
    sha256: str | None = None
    vram_estimate_mb: float | None = None
    vram_measured_mb: float | None = None
    verified: bool = False
    provenance: str = ""
    options: Mapping[str, Any] = field(default_factory=dict)

    def validate(self, *, prefix: str = "") -> list[str]:
        e: list[str] = []
        p = prefix
        if self.role not in ROLES:
            _err(e, p, f"role={self.role!r} is not one of {ROLES}")
        for name in ("provider", "model_id"):
            v = getattr(self, name)
            if not isinstance(v, str) or not v.strip():
                _err(e, p, f"{name} must be a non-empty string, got {v!r}")
        for name in ("vram_estimate_mb", "vram_measured_mb"):
            v = getattr(self, name)
            if v is not None and (not isinstance(v, (int, float)) or v <= 0):
                _err(e, p, f"{name} must be a positive number or None, got {v!r}")
        if not isinstance(self.verified, bool):
            _err(e, p, f"verified must be a bool, got {self.verified!r}")
        # §7.2 lists revision as part of every entry, verified or not, and it is
        # knowable without downloading anything: an unpinned model_id tracks the
        # hub's default branch, so the weights can change under a recorded
        # measurement without a single field in the manifest changing with them.
        # sha256 is only knowable after a download, so it is required at
        # verified=true and not before.
        if not self.revision:
            _err(e, p, "revision is required: an unpinned model_id silently tracks the hub default branch")
        if self.verified:
            # A verified entry is a quotable claim, so it carries the three
            # things a claim needs: what ran, which bytes, and what was measured.
            if not self.sha256:
                _err(e, p, "verified=true requires sha256; an unhashed checkpoint is not a measurement")
            if self.vram_measured_mb is None:
                _err(e, p, "verified=true requires vram_measured_mb from scripts/measure_vram.py (§13.1)")
        return e

    def assert_valid(self) -> "CheckpointSpec":
        errors = self.validate(prefix=f"{self.role}/{self.provider}: ")
        if errors:
            raise RoleContractError("; ".join(errors))
        return self

    def as_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "provider": self.provider,
            "model_id": self.model_id,
            "revision": self.revision,
            "sha256": self.sha256,
            "vram_estimate_mb": self.vram_estimate_mb,
            "vram_measured_mb": self.vram_measured_mb,
            "verified": self.verified,
            "provenance": self.provenance,
            "options": dict(self.options),
        }


# ---------------------------------------------------------------------------
# proposal_2d payloads (§1.5, §5.4)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PromptConfig:
    """The exact prompt configuration a Stage 3 record must carry (P1-14).

    Two silent failures are guarded here:

    * **Dotted category names as prompts** (§0.3). They tokenise to nonsense and
      still produce boxes. `validate()` rejects them by pattern.
    * **Prompt chunking** (§5.4). Grounding DINO's scores come from token-level
      logits over the *whole* prompt, so a 6-phrase chunk's confidences are not
      comparable to a 23-phrase prompt's and a threshold tuned under one regime
      is meaningless under the other. Chunking is forbidden unless the caller
      supplies a re-tuning provenance string, which then travels in the record.
    """

    phrases: tuple[str, ...]
    thresholds: Mapping[str, float] = field(default_factory=dict)
    default_threshold: float = 0.40  # inherited, unvalidated on this substrate (§10)
    source: str = ""                 # e.g. "configs/taxonomy_pilot_nuscenes.yaml@<sha>"
    chunked: bool = False
    chunk_retune_provenance: str = ""

    def validate(self, *, prefix: str = "") -> list[str]:
        e: list[str] = []
        p = prefix
        if not self.phrases:
            _err(e, p, "phrases is empty; a proposal role needs at least one prompt phrase")
        for phrase in self.phrases:
            if not isinstance(phrase, str) or not phrase.strip():
                _err(e, p, f"phrase must be a non-empty string, got {phrase!r}")
            elif _DOTTED_CATEGORY_RE.match(phrase.strip()):
                _err(
                    e,
                    p,
                    f"phrase {phrase!r} is a dotted nuScenes category name, not a natural-language "
                    "prompt; map it through taxonomy_pilot_nuscenes.yaml (§0.3)",
                )
        if not isinstance(self.default_threshold, (int, float)) or not 0.0 < float(self.default_threshold) < 1.0:
            _err(e, p, f"default_threshold must be in (0, 1), got {self.default_threshold!r}")
        for name, value in dict(self.thresholds).items():
            if not isinstance(value, (int, float)) or not 0.0 < float(value) < 1.0:
                _err(e, p, f"thresholds[{name!r}] must be in (0, 1), got {value!r}")
        if not self.source:
            _err(e, p, "source is required: every Stage 3 record states which prompt file it ran (P1-14)")
        if self.chunked and not self.chunk_retune_provenance:
            _err(
                e,
                p,
                "chunked=true without chunk_retune_provenance: chunking changes confidence semantics, "
                "so thresholds must be re-tuned and the re-tuning recorded (§5.4)",
            )
        return e

    def threshold_for(self, class_name: str) -> float:
        """Per-class threshold with a global default (`comprehensive.md` §7.3.3).

        A single tuned scalar is indefensible: Grounding DINO scores are not
        calibrated across phrases and longer phrases score systematically lower.
        A scalar is acceptable as a *pilot default* only because the interface
        takes a dict, so production needs no Stage 3 change (X-5).
        """
        return float(self.thresholds.get(class_name, self.default_threshold))

    def as_dict(self) -> dict[str, Any]:
        return {
            "phrases": list(self.phrases),
            "n_phrases": len(self.phrases),
            "thresholds": dict(self.thresholds),
            "default_threshold": float(self.default_threshold),
            "source": self.source,
            "chunked": bool(self.chunked),
            "chunk_retune_provenance": self.chunk_retune_provenance,
        }


def _check_image_size(e: list[str], p: str, size: Any, field_name: str) -> None:
    if (
        not isinstance(size, (tuple, list))
        or len(size) != 2
        or not all(isinstance(v, (int, np.integer)) and not isinstance(v, bool) for v in size)
    ):
        _err(e, p, f"{field_name} must be a (width_px, height_px) integer pair, got {size!r}")
        return
    if (int(size[0]), int(size[1])) != ORIGINAL_SIZE_PX:
        _err(
            e,
            p,
            f"{field_name}={tuple(size)} but every 2D quantity crossing a stage boundary is "
            f"absolute pixels at {ORIGINAL_SIZE_PX[0]}x{ORIGINAL_SIZE_PX[1]} (§1.5 rule 1); "
            "the adapter owns the inverse transform, not the stage",
        )


def _check_resize_policy(e: list[str], p: str, policy: Any) -> None:
    if policy == "square":
        _err(
            e,
            p,
            "resize_policy='square': a square resize of 16:9 imagery with a uniform inverse "
            "stretches every box along one axis and every 3D extent with it (§1.5 rule 3). "
            "Letterbox with recorded padding, or resize shortest side",
        )
    elif policy not in RESIZE_POLICIES:
        _err(e, p, f"resize_policy={policy!r} is not one of {RESIZE_POLICIES}")


def _check_masks(e: list[str], p: str, masks: Any, n: int, field_name: str) -> None:
    if not isinstance(masks, np.ndarray):
        _err(e, p, f"{field_name} must be a numpy array, got {type(masks).__name__}")
        return
    if masks.dtype != np.bool_:
        _err(e, p, f"{field_name}.dtype must be bool, got {masks.dtype}")
    if masks.ndim != 3:
        _err(e, p, f"{field_name} must be (N, H, W), got shape {masks.shape}")
        return
    if masks.shape[0] != n:
        _err(e, p, f"{field_name} has {masks.shape[0]} masks for {n} boxes; one mask per box, in order")
    # §1.5 rule 4: asserted, not assumed. MobileSAM postprocessing returns 1024-
    # space masks unless the adapter inverts its own transform.
    if (int(masks.shape[2]), int(masks.shape[1])) != ORIGINAL_SIZE_PX:
        _err(
            e,
            p,
            f"{field_name} is {masks.shape[2]}x{masks.shape[1]}; masks are returned at "
            f"{ORIGINAL_SIZE_PX[0]}x{ORIGINAL_SIZE_PX[1]} before Stage 5 indexes them (§1.5 rule 4)",
        )


@dataclass
class Proposals:
    """`proposal_2d` output: absolute-pixel boxes at original resolution.

    `phrase_spans` carries the CHARACTER span [start, end) in the concatenated
    caption that each box's class phrase occupies — char, not token: tokens are
    a property of one tokenizer revision, while the caption string is in the
    record, so a char span is checkable by any reader with no model loaded
    (C18, doc corrected 2026-08-12). It is not decoration: Grounding DINO emits
    per-token logits over the concatenation, and a span-bookkeeping bug yields
    well-placed boxes with **wrong labels** — which then select the wrong DBSCAN
    epsilon, the wrong dimension prior and the wrong inflation target, with
    every downstream stage running perfectly (§5.4). Carrying the span makes
    the mapping testable instead of trusted.

    `masks` is populated only by providers that return box + mask jointly
    (SAM-3-class, DINO-X-class), in which case Stage 4 is a pass-through
    (§7.1 fix 3).
    """

    boxes_xyxy_px: np.ndarray                 # (N, 4) float32, absolute px, x1<x2, y1<y2
    scores: np.ndarray                        # (N,) float32
    class_names: list[str]                    # (N,) prompt-phrase class, not a dotted category
    phrase_spans: list[tuple[int, int]]       # (N,) [start, end) CHAR span in the caption string
    prompt_config: PromptConfig
    channel: str = ""                         # source camera, for the multi-camera contest rule (§1.5 rule 5)
    image_size_px: tuple[int, int] = ORIGINAL_SIZE_PX
    resize_policy: str = "letterbox"
    model_input_size_px: tuple[int, int] | None = None   # what the model actually saw
    masks: np.ndarray | None = None

    def __len__(self) -> int:
        return int(np.asarray(self.boxes_xyxy_px).shape[0]) if np.asarray(self.boxes_xyxy_px).size else 0

    def validate(self, *, prefix: str = "") -> list[str]:
        e: list[str] = []
        p = prefix
        boxes = self.boxes_xyxy_px
        if not isinstance(boxes, np.ndarray):
            _err(e, p, f"boxes_xyxy_px must be a numpy array, got {type(boxes).__name__}")
            return e
        if boxes.ndim != 2 or (boxes.size and boxes.shape[1] != 4):
            _err(e, p, f"boxes_xyxy_px must be (N, 4) xyxy — not cxcywh, not normalised — got {boxes.shape}")
            return e
        n = int(boxes.shape[0])
        if not np.issubdtype(boxes.dtype, np.floating):
            _err(e, p, f"boxes_xyxy_px.dtype must be floating, got {boxes.dtype}")
        if n:
            if not np.isfinite(boxes).all():
                _err(e, p, "boxes_xyxy_px contains non-finite values")
            else:
                if not (boxes[:, 2] > boxes[:, 0]).all() or not (boxes[:, 3] > boxes[:, 1]).all():
                    _err(e, p, "boxes_xyxy_px is not ordered x1<x2, y1<y2 — a normalised or cxcywh box reads this way")
                w, h = ORIGINAL_SIZE_PX
                # No count gate: a single box whose every coordinate is <= 1.0 is
                # sub-pixel under the absolute-pixel reading, so it is a
                # normalised box either way. Gating this on n > 1 let a one-box
                # normalised set through, which is exactly the frame — one
                # detection — where nobody would notice.
                if float(boxes.max()) <= 1.0:
                    _err(
                        e,
                        p,
                        "every boxes_xyxy_px value is <= 1.0: this is a normalised box set, and the "
                        f"contract is absolute pixels at {w}x{h} (§1.5 rule 1)",
                    )
                if boxes[:, 0].min() < 0 or boxes[:, 1].min() < 0 or boxes[:, 2].max() > w or boxes[:, 3].max() > h:
                    _err(e, p, f"boxes_xyxy_px falls outside the {w}x{h} image; clip in the adapter, not in stage code")
        scores = np.asarray(self.scores)
        if scores.shape != (n,):
            _err(e, p, f"scores must be (N,) for N={n}, got {scores.shape}")
        elif n and (scores.min() < 0.0 or scores.max() > 1.0):
            _err(e, p, "scores must lie in [0, 1]")
        if len(self.class_names) != n:
            _err(e, p, f"class_names has {len(self.class_names)} entries for {n} boxes")
        for name in self.class_names:
            if _DOTTED_CATEGORY_RE.match(str(name)):
                _err(e, p, f"class_names contains a dotted category {name!r}; classes are prompt phrases (§0.3)")
        if len(self.phrase_spans) != n:
            _err(
                e,
                p,
                f"phrase_spans has {len(self.phrase_spans)} entries for {n} boxes; the span is what makes "
                "the phrase->class mapping testable (§5.4)",
            )
        for span in self.phrase_spans:
            if (
                not isinstance(span, (tuple, list))
                or len(span) != 2
                or not all(isinstance(v, (int, np.integer)) for v in span)
                or int(span[0]) >= int(span[1])
            ):
                _err(e, p, f"phrase_span must be a [start, end) integer pair with start<end, got {span!r}")
        e += self.prompt_config.validate(prefix=f"{p}prompt_config.")
        _check_image_size(e, p, self.image_size_px, "image_size_px")
        _check_resize_policy(e, p, self.resize_policy)
        if self.masks is not None:
            _check_masks(e, p, self.masks, n, "masks")
        return e

    def assert_valid(self, *, prefix: str = "") -> "Proposals":
        errors = self.validate(prefix=prefix)
        if errors:
            raise RoleContractError("; ".join(errors))
        return self


# ---------------------------------------------------------------------------
# mask_2d payloads (§7.1 fix 1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TemporalWindow:
    """Propagation window for a temporal `mask_2d` provider.

    MobileSAM ignores this. SAM 2.1 needs it, together with an explicit
    re-initialisation rule at block boundaries — otherwise a memory state built
    on one block leaks into the next and mask identity drifts across a cut that
    the pipeline treats as a hard edge. The pilot's capability gap is stated
    plainly (§4): no propagation, masking is independent per frame, and Stage 7
    will look worse for a structural reason rather than a tuning one.
    """

    frames: int = 1                       # 1 == per-frame, no propagation
    stride: int = 1
    reinit_at_block_boundary: bool = True
    block_frames: int | None = None       # None == the caller's block, e.g. one scene

    def validate(self, *, prefix: str = "") -> list[str]:
        e: list[str] = []
        p = prefix
        if not isinstance(self.frames, int) or isinstance(self.frames, bool) or self.frames < 1:
            _err(e, p, f"frames must be a positive int, got {self.frames!r}")
        if not isinstance(self.stride, int) or isinstance(self.stride, bool) or self.stride < 1:
            _err(e, p, f"stride must be a positive int, got {self.stride!r}")
        if self.block_frames is not None and (not isinstance(self.block_frames, int) or self.block_frames < 1):
            _err(e, p, f"block_frames must be a positive int or None, got {self.block_frames!r}")
        if not isinstance(self.reinit_at_block_boundary, bool):
            _err(e, p, f"reinit_at_block_boundary must be a bool, got {self.reinit_at_block_boundary!r}")
        return e

    @property
    def is_temporal(self) -> bool:
        return self.frames > 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "frames": self.frames,
            "stride": self.stride,
            "reinit_at_block_boundary": self.reinit_at_block_boundary,
            "block_frames": self.block_frames,
        }


PER_FRAME_WINDOW = TemporalWindow(frames=1)


@dataclass
class MaskResult:
    """`mask_2d` output: one mask per input box, in order, at 1600x900.

    `state` is opaque: a provider's memory/propagation state, threaded back in
    by the caller on the next call and dropped at a block boundary. MobileSAM
    returns None. Nothing outside the provider interprets it.

    `propagated` says whether these masks came from propagation rather than from
    a fresh per-frame segmentation. Stage 7 needs to know, because propagated
    masks and flickering per-frame masks fail differently.
    """

    masks: np.ndarray                      # (N, H, W) bool at 1600x900
    state: Any | None = None
    window: TemporalWindow = PER_FRAME_WINDOW
    propagated: bool = False
    scores: np.ndarray | None = None       # (N,) provider mask quality, if any

    def __len__(self) -> int:
        return int(np.asarray(self.masks).shape[0]) if np.asarray(self.masks).size else 0

    def validate(self, *, n_boxes: int | None = None, prefix: str = "") -> list[str]:
        e: list[str] = []
        p = prefix
        masks = np.asarray(self.masks)
        n = int(masks.shape[0]) if masks.ndim == 3 else -1
        _check_masks(e, p, self.masks, n_boxes if n_boxes is not None else max(n, 0), "masks")
        e += self.window.validate(prefix=f"{p}window.")
        if self.propagated and not self.window.is_temporal:
            _err(e, p, "propagated=true with a per-frame window (frames=1): a provider cannot propagate over one frame")
        if self.propagated and self.state is None:
            _err(e, p, "propagated=true with state=None: propagation without carried state is not propagation")
        if self.scores is not None:
            scores = np.asarray(self.scores)
            if n >= 0 and scores.shape != (n,):
                _err(e, p, f"scores must be (N,) for N={n}, got {scores.shape}")
        return e

    def assert_valid(self, *, n_boxes: int | None = None, prefix: str = "") -> "MaskResult":
        errors = self.validate(n_boxes=n_boxes, prefix=prefix)
        if errors:
            raise RoleContractError("; ".join(errors))
        return self


# ---------------------------------------------------------------------------
# embedding payloads (P1-2)
# ---------------------------------------------------------------------------


@dataclass
class EmbeddingBatch:
    """Embeddings plus the preprocessing that produced them.

    `semantics` and `preprocessing` are required because the two embedding roles
    share a checkpoint and share nothing else. A Stage 7 crop embedding computed
    under Stage 2's whole-image transform is a well-formed vector of the right
    dimensionality whose similarity is dominated by ~20x upsampling artifacts —
    the exact regime (small distant crops) where association already fails at
    2 Hz (§5.8).
    """

    vectors: np.ndarray                     # (N, D) float32
    semantics: str
    preprocessing: str                      # named, versioned transform owned by the role
    model_input_size_px: tuple[int, int]    # what the model actually saw, e.g. (518, 518)
    source_size_px: tuple[int, int] = ORIGINAL_SIZE_PX
    ids: list[str] | None = None            # sample_data tokens / instance ids, order-matched
    normalized: bool = False

    def __len__(self) -> int:
        return int(np.asarray(self.vectors).shape[0])

    @property
    def dim(self) -> int:
        v = np.asarray(self.vectors)
        return int(v.shape[1]) if v.ndim == 2 else -1

    def validate(self, *, prefix: str = "") -> list[str]:
        e: list[str] = []
        p = prefix
        v = self.vectors
        if not isinstance(v, np.ndarray):
            _err(e, p, f"vectors must be a numpy array, got {type(v).__name__}")
            return e
        if v.ndim != 2:
            _err(e, p, f"vectors must be (N, D), got {v.shape}")
            return e
        if not np.issubdtype(v.dtype, np.floating):
            _err(e, p, f"vectors.dtype must be floating, got {v.dtype}")
        if v.size and not np.isfinite(v).all():
            _err(e, p, "vectors contains non-finite values")
        if self.semantics not in EMBEDDING_SEMANTICS:
            _err(e, p, f"semantics={self.semantics!r} is not one of {EMBEDDING_SEMANTICS}")
        if not isinstance(self.preprocessing, str) or not self.preprocessing.strip():
            _err(
                e,
                p,
                "preprocessing is required: preprocessing belongs to the role, not the model, and an "
                "unnamed transform is how Stage 7 silently inherits Stage 2's (P1-2)",
            )
        for name in ("model_input_size_px", "source_size_px"):
            size = getattr(self, name)
            if (
                not isinstance(size, (tuple, list))
                or len(size) != 2
                or not all(isinstance(x, (int, np.integer)) and not isinstance(x, bool) and x > 0 for x in size)
            ):
                _err(e, p, f"{name} must be a positive (width_px, height_px) integer pair, got {size!r}")
        if self.ids is not None and len(self.ids) != v.shape[0]:
            _err(e, p, f"ids has {len(self.ids)} entries for {v.shape[0]} vectors")
        return e

    def assert_valid(self, *, prefix: str = "") -> "EmbeddingBatch":
        errors = self.validate(prefix=prefix)
        if errors:
            raise RoleContractError("; ".join(errors))
        return self


# ---------------------------------------------------------------------------
# The four role Protocols (§7.1)
# ---------------------------------------------------------------------------


@runtime_checkable
class ModelRole(Protocol):
    """Common surface. `spec` identifies the checkpoint; `role` identifies the job.

    `load()` / `unload()` are explicit because the pilot's policy is serial —
    one role resident at a time, freed before the next is constructed — and
    because teardown must be *asserted*, not assumed: `empty_cache()` does not
    free memory still referenced by a live Python object (§7.1). The assertion
    itself is in `scripts/measure_vram.py`; the hook is here.
    """

    @property
    def roles(self) -> tuple[str, ...]:
        """Every role this instance fills.

        A single-role provider returns a 1-tuple. A **composite** returns each of
        its roles (§7.1 fix 2): `create()` hands the same instance back for all of
        them, because "one unified proposal + mask + track engine" means one
        resident engine, not one object per facet. A scalar `role` here would
        make the composite case unrepresentable — which is exactly the defect
        rev 1's registry had.
        """
        ...

    @property
    def spec(self) -> CheckpointSpec:
        """Checkpoint identity. One per instance, including for a composite —
        a composite is one checkpoint filling several roles."""
        ...

    @property
    def device(self) -> str: ...

    def load(self) -> None: ...

    def unload(self) -> None: ...


@runtime_checkable
class EmbeddingOOD(ModelRole, Protocol):
    """Stage 2. Whole-image embedding — CLS token, ~518x518 input.

    Consumed by GLOSH outlier scoring on the raw embeddings; UMAP is for
    visualisation only, because it preserves neither density nor global
    structure and cluster membership in its 2-D projection has no formal
    relationship to outlyingness in the 384-D space (§5.3).
    """

    def embed_images(
        self,
        images: Sequence[np.ndarray],
        *,
        ids: Sequence[str] | None = None,
    ) -> EmbeddingBatch:
        """Embed whole frames given as HxWx3 uint8 RGB arrays at 1600x900.

        Returns `semantics == "whole_image_cls"`.
        """
        ...


@runtime_checkable
class Proposal2D(ModelRole, Protocol):
    """Stage 3. Open-vocabulary 2D proposals from natural-language phrases.

    `returns_masks` is a capability flag, not a request: a provider that returns
    box + mask jointly (SAM 3, DINO-X) makes Stage 4 a pass-through (§7.1 fix 3).
    """

    @property
    def returns_masks(self) -> bool: ...

    def propose(
        self,
        image: np.ndarray,
        prompt: PromptConfig,
        *,
        channel: str = "",
    ) -> Proposals:
        """One 1600x900 HxWx3 uint8 RGB frame -> absolute-pixel xyxy proposals.

        The adapter owns the forward and inverse transform and the phrase-span
        bookkeeping. Neither ever leaks into stage code (§1.5 rule 2, §5.4).
        """
        ...


@runtime_checkable
class Mask2D(ModelRole, Protocol):
    """Stage 4. Box-prompted masks, with the temporal parameters present on day one.

    `segment()` takes a *sequence* of frames, an optional carried `state`, and a
    `window`, so that swapping MobileSAM (`frames=1`, state ignored) for SAM 2.1
    (`(frames, boxes, memory_state) -> masks over time`) is a provider change and
    not a Stage 4 rewrite (§7.1 fix 1).
    """

    @property
    def supports_temporal(self) -> bool: ...

    def segment(
        self,
        images: Sequence[np.ndarray],
        boxes_xyxy_px: np.ndarray,
        *,
        state: Any | None = None,
        window: TemporalWindow = PER_FRAME_WINDOW,
        channel: str = "",
    ) -> MaskResult:
        """Boxes in absolute 1600x900 pixels -> one mask per box, in order, at 1600x900.

        A non-temporal provider MUST accept `state` and `window` and ignore them,
        returning `state=None`, `propagated=False`. It must not raise on them:
        the argument's presence is the whole point of the interface.
        """
        ...


@runtime_checkable
class ReidEmbedding(ModelRole, Protocol):
    """Stage 7. Object-crop embedding — different preprocessing, different semantics.

    `min_crop_px` is the stated size below which appearance similarity is not
    trusted (§5.8, decision 5): DINOv2's patch-14 tokenisation reduces a 28x28
    crop to a 2x2 patch grid, so cosine similarity approaches degeneracy exactly
    where the 2 Hz IoU term has already collapsed. The association code gates on
    this value rather than discovering it empirically per run.
    """

    @property
    def min_crop_px(self) -> int: ...

    def embed_crops(
        self,
        image: np.ndarray,
        boxes_xyxy_px: np.ndarray,
        *,
        masks: np.ndarray | None = None,
        ids: Sequence[str] | None = None,
    ) -> EmbeddingBatch:
        """Crops from one 1600x900 frame -> per-object embeddings.

        `masks` enables mask-pooled patch tokens (`semantics ==
        "mask_pooled_patch"`); without it the provider returns `"crop_cls"`.
        """
        ...


ROLE_PROTOCOLS: dict[str, type] = {
    EMBEDDING_OOD: EmbeddingOOD,
    PROPOSAL_2D: Proposal2D,
    MASK_2D: Mask2D,
    REID_EMBEDDING: ReidEmbedding,
}

# Explicit member lists. `issubclass()` against a Protocol with non-method
# members is a TypeError and `isinstance()` only reports a bool, which is a
# useless diagnostic when a provider is one method away from conforming.
_COMMON_MEMBERS: tuple[str, ...] = ("roles", "spec", "device", "load", "unload")
ROLE_METHODS: dict[str, tuple[str, ...]] = {
    EMBEDDING_OOD: ("embed_images",),
    PROPOSAL_2D: ("propose", "returns_masks"),
    MASK_2D: ("segment", "supports_temporal"),
    REID_EMBEDDING: ("embed_crops", "min_crop_px"),
}
_CALLABLE_MEMBERS: frozenset[str] = frozenset(
    {"load", "unload", "embed_images", "propose", "segment", "embed_crops"}
)


def check_role_conformance(obj: object, role: str) -> list[str]:
    """Return the missing/malformed members for `role`, most useful first."""
    role = _check_role_name(role)
    errors: list[str] = []
    for member in _COMMON_MEMBERS + ROLE_METHODS[role]:
        if not hasattr(obj, member):
            errors.append(f"missing member: {member}")
        elif member in _CALLABLE_MEMBERS and not callable(getattr(obj, member)):
            errors.append(f"member {member} is not callable")
    declared = getattr(obj, "roles", None)
    if declared is not None:
        if isinstance(declared, str) or not isinstance(declared, (tuple, list)):
            errors.append(f"roles must be a tuple of role names, got {declared!r}")
        elif role not in declared:
            errors.append(f"provider declares roles={tuple(declared)} but was requested as {role!r}")
    spec = getattr(obj, "spec", None)
    if isinstance(spec, CheckpointSpec):
        errors += spec.validate(prefix="spec: ")
        if isinstance(declared, (tuple, list)) and spec.role not in declared:
            errors.append(f"spec.role={spec.role!r} is not among the provider's roles {tuple(declared)}")
    elif spec is not None:
        errors.append(f"spec must be a CheckpointSpec, got {type(spec).__name__}")
    return errors


def assert_role_conformance(obj: object, role: str) -> object:
    errors = check_role_conformance(obj, role)
    if errors:
        raise RoleContractError(f"{type(obj).__name__} does not satisfy role {role!r}: " + "; ".join(errors))
    return obj


# ---------------------------------------------------------------------------
# Registry (§7.1 fix 2 — composite providers)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProviderEntry:
    """One registered provider and the roles it can fill.

    A provider registered for several roles is a *composite* (SAM 3's
    proposal + mask + track engine). `create()` returns the same singleton
    instance for each of its roles, because that is what "one unified engine"
    means. It does not mean two roles are free: the policy is still serial.
    """

    name: str
    roles: tuple[str, ...]
    factory: Callable[..., Any]

    @property
    def composite(self) -> bool:
        return len(self.roles) > 1


_PROVIDERS: dict[str, ProviderEntry] = {}
_INSTANCES: dict[str, Any] = {}


def register(
    name: str,
    roles: str | Iterable[str],
    factory: Callable[..., Any],
    *,
    replace: bool = False,
) -> ProviderEntry:
    """Register `factory` under `name` for one or more roles."""
    if not isinstance(name, str) or not name.strip():
        raise RegistryError(f"provider name must be a non-empty string, got {name!r}")
    role_tuple = (roles,) if isinstance(roles, str) else tuple(roles)
    if not role_tuple:
        raise RegistryError(f"provider {name!r} registered for no roles")
    for role in role_tuple:
        _check_role_name(role)
    if len(set(role_tuple)) != len(role_tuple):
        raise RegistryError(f"provider {name!r} lists a role twice: {role_tuple}")
    if not callable(factory):
        raise RegistryError(f"provider {name!r}: factory is not callable")
    if name in _PROVIDERS and not replace:
        raise RegistryError(
            f"provider {name!r} is already registered for {_PROVIDERS[name].roles}; "
            "pass replace=True to override deliberately"
        )
    if name in _INSTANCES:
        raise RegistryError(f"provider {name!r} has a resident instance; release() it before re-registering")
    entry = ProviderEntry(name=name.strip(), roles=role_tuple, factory=factory)
    _PROVIDERS[entry.name] = entry
    return entry


def unregister(name: str) -> None:
    if name in _INSTANCES:
        release(name)
    _PROVIDERS.pop(name, None)


def provider_names(role: str | None = None) -> tuple[str, ...]:
    if role is None:
        return tuple(sorted(_PROVIDERS))
    role = _check_role_name(role)
    return tuple(sorted(n for n, e in _PROVIDERS.items() if role in e.roles))


def roles_of(name: str) -> tuple[str, ...]:
    if name not in _PROVIDERS:
        raise RegistryError(f"provider {name!r} is not registered")
    return _PROVIDERS[name].roles


def create(role: str, name: str | None = None, *, singleton: bool = True, **kwargs: Any) -> Any:
    """Construct (or return the resident) provider for `role`.

    Conformance is checked at construction, so a provider that is one method
    short fails here rather than three stages later with an AttributeError
    inside a loop over 2,424 images.
    """
    role = _check_role_name(role)
    if name is None:
        candidates = provider_names(role)
        if len(candidates) != 1:
            raise RegistryError(
                f"role {role!r} has {len(candidates)} registered providers {candidates}; "
                "name one explicitly — a default here would make the pipeline depend on import order"
            )
        name = candidates[0]
    if name not in _PROVIDERS:
        raise RegistryError(f"provider {name!r} is not registered (registered: {provider_names()})")
    entry = _PROVIDERS[name]
    if role not in entry.roles:
        raise RegistryError(f"provider {name!r} is registered for {entry.roles}, not for {role!r}")

    if singleton and name in _INSTANCES:
        instance = _INSTANCES[name]
    else:
        instance = entry.factory(role=role, **kwargs) if _factory_takes_role(entry.factory) else entry.factory(**kwargs)
        if singleton:
            _INSTANCES[name] = instance
    assert_role_conformance(instance, role)
    return instance


def _factory_takes_role(factory: Callable[..., Any]) -> bool:
    """Composite factories need to know which role they were asked for."""
    import inspect

    try:
        params = inspect.signature(factory).parameters
    except (TypeError, ValueError):
        return False
    if "role" in params:
        return True
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def release(name: str) -> None:
    """Unload and drop the resident instance.

    This is the *reference*-dropping half of teardown. It cannot verify that the
    device is quiescent — that needs torch, and this module is torch-free. The
    `memory_allocated() ~ 0` assertion between roles lives in
    `scripts/measure_vram.py` (§7.1).
    """
    instance = _INSTANCES.pop(name, None)
    if instance is None:
        return
    unload = getattr(instance, "unload", None)
    if callable(unload):
        unload()


def release_all() -> None:
    for name in list(_INSTANCES):
        release(name)


def resident_providers() -> tuple[str, ...]:
    return tuple(sorted(_INSTANCES))


# ---------------------------------------------------------------------------
# configs/models_{pilot,production}.yaml (§7.2, §7.3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VramBudget:
    """The 4 GB budget, with the consequence of exceeding it stated (§7.2).

    Rev 1 defined a ceiling and not what happens at it. `on_exceed` has exactly
    one legal value: an OOM or an over-ceiling measurement is a hard stop with
    stage / role / resolution logged, never a silent retry at lower resolution,
    which would change the output distribution mid-run (§1.9).
    """

    total_device_mb: float
    system_reserve_mb: float
    on_exceed: str = "hard_stop"
    reserve_provenance: str = ""

    @property
    def hard_ceiling_mb(self) -> float:
        return float(self.total_device_mb) - float(self.system_reserve_mb)

    def validate(self, *, prefix: str = "") -> list[str]:
        e: list[str] = []
        p = prefix
        for name in ("total_device_mb", "system_reserve_mb"):
            v = getattr(self, name)
            if not isinstance(v, (int, float)) or isinstance(v, bool) or v < 0:
                _err(e, p, f"{name} must be a non-negative number, got {v!r}")
        if self.on_exceed != "hard_stop":
            _err(e, p, f"on_exceed={self.on_exceed!r}: the only legal behaviour is 'hard_stop' (§1.9)")
        if self.hard_ceiling_mb <= 0:
            _err(e, p, f"hard_ceiling_mb={self.hard_ceiling_mb} is not positive")
        if not self.reserve_provenance:
            _err(e, p, "system_reserve_mb needs a provenance note; it is an unvalidated inherited estimate (§10)")
        return e

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_device_mb": float(self.total_device_mb),
            "system_reserve_mb": float(self.system_reserve_mb),
            "hard_ceiling_mb": self.hard_ceiling_mb,
            "on_exceed": self.on_exceed,
            "reserve_provenance": self.reserve_provenance,
        }


@dataclass(frozen=True)
class ModelConfig:
    """A resolved models_*.yaml. Same role names on both tiers (§7.3)."""

    tier: str
    budget: VramBudget
    roles: Mapping[str, CheckpointSpec]
    seed: int
    config_path: str = ""

    def validate(self, *, prefix: str = "") -> list[str]:
        e: list[str] = []
        p = prefix
        if self.tier not in ("pilot", "production"):
            _err(e, p, f"tier={self.tier!r} must be 'pilot' or 'production'")
        e += self.budget.validate(prefix=f"{p}vram_budget.")
        missing = [r for r in ROLES if r not in self.roles]
        if missing:
            _err(e, p, f"roles missing: {missing}; both tiers declare all four roles by the same names (§7.3)")
        unknown = [r for r in self.roles if r not in ROLES]
        if unknown:
            _err(e, p, f"unknown roles: {unknown}")
        for role, spec in self.roles.items():
            e += spec.validate(prefix=f"{p}roles.{role}.")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            _err(e, p, f"seed must be an int, got {self.seed!r} — one global seed, recorded (§1.9)")
        return e

    def assert_valid(self) -> "ModelConfig":
        errors = self.validate(prefix=f"{self.config_path or 'model config'}: ")
        if errors:
            raise RoleContractError("\n  - ".join(["model config is invalid:"] + errors))
        return self

    def as_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "config_path": self.config_path,
            "seed": self.seed,
            "vram_budget": self.budget.as_dict(),
            "roles": {r: s.as_dict() for r, s in sorted(self.roles.items())},
        }


def load_model_config(path: str | os.PathLike) -> ModelConfig:
    """Read models_pilot.yaml / models_production.yaml. No defaults, fail closed.

    Expected shape:

        tier: pilot
        seed: 20260812
        vram_budget:
          total_device_mb: 4096
          system_reserve_mb: 420
          on_exceed: hard_stop
          reserve_provenance: "measured on <card>, <date>"
        roles:
          embedding_ood:
            provider: dinov2_vits14
            model_id: facebook/dinov2-small
            revision: <git sha>
            sha256: <weights sha>
            vram_estimate_mb: 800
            verified: false
            provenance: "planning estimate, §13.1"
    """
    path = os.fspath(path)
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise RoleContractError(f"{path}: expected a YAML mapping, got {type(raw).__name__}")

    required_top = ("tier", "seed", "vram_budget", "roles")
    missing = [k for k in required_top if k not in raw]
    if missing:
        raise RoleContractError(f"{path}: missing required key(s): {missing}")

    budget_raw = raw["vram_budget"]
    if not isinstance(budget_raw, dict):
        raise RoleContractError(f"{path}: vram_budget must be a mapping")
    for key in ("total_device_mb", "system_reserve_mb"):
        if key not in budget_raw:
            raise RoleContractError(f"{path}: vram_budget.{key} is required; there is no default ceiling")
    budget = VramBudget(
        total_device_mb=float(budget_raw["total_device_mb"]),
        system_reserve_mb=float(budget_raw["system_reserve_mb"]),
        on_exceed=str(budget_raw.get("on_exceed", "hard_stop")),
        reserve_provenance=str(budget_raw.get("reserve_provenance", "")),
    )

    roles_raw = raw["roles"]
    if not isinstance(roles_raw, dict):
        raise RoleContractError(f"{path}: roles must be a mapping of role -> checkpoint entry")
    specs: dict[str, CheckpointSpec] = {}
    for role, entry in roles_raw.items():
        if not isinstance(entry, dict):
            raise RoleContractError(f"{path}: roles.{role} must be a mapping")
        if "model_id" not in entry:
            raise RoleContractError(f"{path}: roles.{role}.model_id is required")
        specs[str(role)] = CheckpointSpec(
            role=str(role),
            provider=str(entry.get("provider", role)),
            model_id=str(entry["model_id"]),
            revision=entry.get("revision"),
            sha256=entry.get("sha256"),
            vram_estimate_mb=entry.get("vram_estimate_mb"),
            vram_measured_mb=entry.get("vram_measured_mb"),
            verified=bool(entry.get("verified", False)),
            provenance=str(entry.get("provenance", "")),
            options=dict(entry.get("options", {}) or {}),
        )

    return ModelConfig(
        tier=str(raw["tier"]),
        budget=budget,
        roles=specs,
        seed=int(raw["seed"]),
        config_path=os.path.realpath(path),
    ).assert_valid()


# ---------------------------------------------------------------------------
# C1 — the synthetic VRAM ceiling
# ---------------------------------------------------------------------------

VRAM_CAP_ENV = "DHAKASCENES_VRAM_CAP_MIB"


def apply_vram_cap(torch_module: Any, device: str) -> dict:
    """Enforce C1's synthetic VRAM ceiling and return the manifest block.

    C1 (register, RESOLVED BY HUMAN): the pilot's binding contract is the 4 GB
    laptop card, and every GPU process on the 24 GB dev box enforces a synthetic
    4096 MiB ceiling so nothing is ever measured, or found to fit, on a machine
    the pilot does not target. The env var is the .env contract; C17 records
    that until 2026-08-12 no code read it — this function is the code that reads
    it, called by every adapter BEFORE its first allocation (a cap applied after
    the model loaded caps nothing).

    Returns C1's manifest schema:
        {value_mib, enforced: "synthetic"|"physical"|"none",
         physical_device_mib, device_name}
    An empty/unset env var means the physical card is the ceiling; the run is
    then honest but its fit claims carry verified: false (C1).
    """
    raw = os.environ.get(VRAM_CAP_ENV, "").strip()
    if not str(device).startswith("cuda") or not torch_module.cuda.is_available():
        return {"value_mib": None, "enforced": "none", "physical_device_mib": None,
                "device_name": str(device)}
    index = int(str(device).split(":", 1)[1]) if ":" in str(device) else torch_module.cuda.current_device()
    props = torch_module.cuda.get_device_properties(index)
    physical_mib = int(props.total_memory // (1024 * 1024))
    if not raw:
        return {"value_mib": None, "enforced": "none", "physical_device_mib": physical_mib,
                "device_name": props.name}
    cap_mib = int(raw)
    if cap_mib <= 0:
        raise ValueError(f"{VRAM_CAP_ENV}={raw!r}: the cap must be a positive MiB count")
    if cap_mib >= physical_mib:
        # The card is smaller than the cap: the physical limit is the ceiling
        # and pretending otherwise would label a real device as synthetic.
        return {"value_mib": cap_mib, "enforced": "physical", "physical_device_mib": physical_mib,
                "device_name": props.name}
    torch_module.cuda.set_per_process_memory_fraction(cap_mib / physical_mib, index)
    return {"value_mib": cap_mib, "enforced": "synthetic", "physical_device_mib": physical_mib,
            "device_name": props.name}
