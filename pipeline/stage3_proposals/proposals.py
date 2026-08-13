#!/usr/bin/env python3
"""Stage 3 — open-vocabulary 2D proposals (§5.4, Phase 6).

A Grounding-DINO-family detector over the six ring cameras of every Stage 1
keyframe, with the nuScenes prompt set from
`configs/taxonomy_pilot_nuscenes.yaml`. Default checkpoint on this box:
iSEE-Laboratory/llmdet_large (MM-Grounding-DINO Swin-L + LLMDet fine-tune,
C19); the pilot IDEA-Research/grounding-dino-tiny stays selectable via
--model-id. Both share the caption convention, the processor, and the raw
logits/pred_boxes output contract, so one adapter serves both.

The three silent failures this stage is built around — wrong output that still
looks plausible and never crashes:

  1. **Dotted category names as prompts** (§0.3). `vehicle.emergency.ambulance`
     tokenises to nonsense and returns near-random but non-empty detections. The
     prompt strings come from the mapping table, always, and `PromptConfig`
     rejects a dotted string by pattern.
  2. **Phrase-span bookkeeping** (§5.4) — the sharpest one. Grounding DINO emits
     per-token logits over the *concatenated* caption; recovering which phrase a
     box belongs to is token-span arithmetic, and multi-word classes make it
     error-prone. A span bug yields well-placed boxes with the WRONG LABEL, which
     then selects the wrong DBSCAN epsilon, the wrong dimension prior and the
     wrong inflation target — with every downstream stage running perfectly. So
     the adapter owns the mapping, builds it from the tokenizer's own character
     offsets, and asserts it is total and injective before a single image runs.
     `build_phrase_span_map()` is pure and testable without a GPU or a network.
  3. **A square resize** (§1.5 rule 3). A non-aspect-preserving resize with a
     uniform inverse stretches every box along one axis; boxes still look like
     boxes, SAM still returns masks, and every 3D extent is wrong in one axis,
     consistently. The processed tensor's aspect ratio is asserted against
     1600:900 on every image, and padding is asserted absent.

Everything crossing the stage boundary is absolute pixels at 1600x900, `xyxy`
(§1.5 rule 1). The adapter owns both directions of its transform; no transform
leaks into stage code (§1.5 rule 2).

Also here, because rev 1 left it unspecified: **deduplication of overlapping
proposals**, with a deterministic tie-break — a greedy NMS keyed on model output
order is not reproducible, and §1.9 requires byte-identical re-runs.

Not here: IoA-NMS across overlapping cameras. That is Stage 4 (§5.5), because it
needs the mask set, and because across two image planes it is not a pixel-space
quantity at all.

    python3 -m pipeline.stage3_proposals.proposals [--paths configs/paths.yaml]

Exit codes:
    0  every camera of every keyframe produced proposals under contract
    1  ran, but at least one image was degraded (zero proposals, or truncated)
    2  upstream contract broken, or a model/checkpoint is unavailable; nothing written
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pipeline.common.model_interfaces import (  # noqa: E402
    PROPOSAL_2D,
    CheckpointSpec,
    PromptConfig,
    Proposals,
    RoleContractError,
    apply_vram_cap,
    register,
)
from pipeline.common.paths import (  # noqa: E402
    PathValidationError,
    Paths,
    assert_dataroot_read_only,
    load_paths,
    metadata_fingerprint,
)
from pipeline.common.schemas import (  # noqa: E402
    IMAGE_HEIGHT_PX,
    IMAGE_WIDTH_PX,
    RING_CAMERAS,
    KeyframeRecord,
    SchemaValidationError,
    read_records,
)
from pipeline.common.manifest import (  # noqa: E402
    UpstreamRefusal as _UpstreamRefusal,
    clear_markers,
    require_upstream,
    write_json_atomic,
    write_jsonl_atomic as _write_jsonl_atomic,
    write_marker,
)

STAGE = "stage3_proposals"
STAGE_SPEC = "dhakascenes-pilot/stage3_proposals/v1"

EXIT_OK = 0
EXIT_DEGRADED = 1  # ran, but at least one image produced nothing or was truncated
EXIT_REFUSED = 2  # upstream contract broken or model unavailable; nothing was written

# Aspect-ratio tolerance for the processed tensor. 1600:900 is 16:9 exactly; a
# DETR-style shortest-side resize lands within a pixel of it, a square resize
# misses by 0.78. This is a floating-point tolerance, not a tuned threshold.
ASPECT_TOLERANCE = 0.02


# Canonical class lives in pipeline.common.manifest (C3); the local name stays
# because stages 4 and 5 import it from here.
UpstreamRefusal = _UpstreamRefusal


class ModelUnavailable(RuntimeError):
    """The checkpoint or its library is not installed.

    Distinct from a contract violation: nothing is wrong with the pipeline, the
    card just cannot answer. Never degraded into "zero proposals", which is a
    legitimate output value and would be indistinguishable.
    """


class PhraseSpanError(RuntimeError):
    """The caption's token spans do not cover the phrases exactly (§5.4)."""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProposalConfig:
    """Stage 3 tunables. None of these may appear as a literal in the code below."""

    # --- model ---
    model_id: str = "iSEE-Laboratory/llmdet_large"
    revision: str | None = None
    device: str = "cuda"
    # fp16 via autocast ONLY; model.half() is broken for this architecture (see
    # provenance). Off by default: fp32 fits this box with headroom.
    autocast_fp16: bool = False

    # --- thresholds (§7.3.3: per-class dict with a global default) ---
    default_threshold: float = 0.40
    thresholds: tuple[tuple[str, float], ...] = ()  # phrase -> threshold, from the taxonomy file

    # --- deduplication (§5.4; rev 1 left this undefined) ---
    same_class_iou: float = 0.65
    cross_class_iou: float = 0.90
    min_box_side_px: float = 4.0
    max_proposals_per_image: int = 300

    # --- 2D contract (§1.5) ---
    resize_policy: str = "shortest_side"
    image_width_px: int = IMAGE_WIDTH_PX
    image_height_px: int = IMAGE_HEIGHT_PX

    # --- prompt (§5.4) ---
    allow_prompt_chunking: bool = False

    # --- upstream gate (C16) ---
    accept_degraded_upstream: bool = False

    # --- determinism (§1.9) ---
    global_seed: int = 20260812

    provenance: dict = field(
        default_factory=lambda: {
            "model_id": (
                "default tier for this box, human decision 2026-08-13 (DECISIONS C19): "
                "iSEE-Laboratory/llmdet_large — LLMDet (CVPR 2025) fine-tune of MM-Grounding-DINO "
                "Swin-L. Measured on the resident 4090: 7.5 GiB peak alloc fp32, 349 ms per "
                "1600x900 frame. LVIS minival zero-shot 51.1 AP / 45.1 AP-rare vs the pilot "
                "tiny's 28.8/18.8. The pilot tier (IDEA-Research/grounding-dino-tiny, §7.1 role "
                "table, C1 4-GB contract) stays selectable via model_id; it no longer governs "
                "defaults on this 24-GB machine"
            ),
            "autocast_fp16": (
                "human decision 2026-08-13 (C19): model.half() BREAKS mm-grounding-dino (dtype "
                "mixing in the text path); torch.autocast('cuda', torch.float16) around the "
                "forward is the only valid half-precision path. Default False — fp32 is the "
                "measured, validated configuration (7.5 GiB fits the 24-GB card with headroom)"
            ),
            "default_threshold": "inherited 0.40, unvalidated on this substrate (§10)",
            "thresholds": "per-class dict from taxonomy file; tuned on the `tuning` subset only (§11 d3)",
            "same_class_iou": "arbitrary, needs tuning; suppresses duplicate boxes of one class",
            "cross_class_iou": (
                "arbitrary, needs tuning; deliberately HIGH — cross-class suppression uses IoU, not "
                "IoA, because a pedestrian inside a car box has IoA~1.0 and must survive"
            ),
            "min_box_side_px": "arbitrary; drops degenerate boxes that cannot hold a mask",
            "max_proposals_per_image": "arbitrary cap; truncation is recorded, never silent",
            "resize_policy": "§1.5 rule 3: shortest-side, aspect preserved; square resize forbidden",
            "allow_prompt_chunking": "§5.4: forbidden by default; changes confidence semantics",
            "accept_degraded_upstream": "C16 — consuming a DEGRADED (complete, quality-flagged) "
            "Stage 1 output is an explicit recorded decision, never a default",
            "global_seed": "§1.9, one global seed, recorded",
        }
    )

    def as_dict(self) -> dict:
        out = {k: v for k, v in self.__dict__.items()}
        out["thresholds"] = {k: float(v) for k, v in self.thresholds}
        return out


# ---------------------------------------------------------------------------
# Taxonomy: nuScenes category -> prompt phrase (§0.3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Taxonomy:
    """The mapping table, loaded and checked.

    `phrases` is ordered and that order is part of the measurement: the caption
    is the concatenation, and Grounding DINO's scores are token-level logits over
    it. Reordering the file changes the numbers, so the order is hashed into
    every record's `prompt_sha256`.
    """

    path: str
    sha256: str
    category_to_phrase: tuple[tuple[str, str], ...]
    thresholds: tuple[tuple[str, float], ...]
    default_threshold: float
    # Categories DELIBERATELY outside the class space (C21). The distinction
    # matters: a category with GT and no phrase is normally a refusal (§0.3 — it
    # would contribute to no prior and never be missed), and that guard must keep
    # firing for accidental omissions. Declaring an exclusion in the taxonomy is
    # how an author says "this one is on purpose".
    excluded_categories: tuple[str, ...] = ()

    @property
    def phrases(self) -> tuple[str, ...]:
        """The class space: DEDUPLICATED phrases, in first-appearance order.

        The mapping is many-to-one by design (C21) — five pedestrian categories
        share "a pedestrian", two bus categories share "a bus". Returning one
        entry per CATEGORY would put a phrase into the caption several times,
        and each copy would get its own token span: several spans competing to
        win the argmax for one class, with the winner decided by token position.
        A clean run, and a class whose score depends on where it happened to
        land in the caption.
        """
        seen: dict[str, None] = {}
        for _, phrase in self.category_to_phrase:
            seen.setdefault(phrase, None)
        return tuple(seen)

    @property
    def phrase_to_categories(self) -> dict[str, tuple[str, ...]]:
        """phrase -> every nuScenes category that collapses into it."""
        out: dict[str, list[str]] = {}
        for category, phrase in self.category_to_phrase:
            out.setdefault(phrase, []).append(category)
        return {phrase: tuple(cats) for phrase, cats in out.items()}

    def as_dict(self) -> dict:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "n_categories": len(self.category_to_phrase),
            "n_phrases": len(self.phrases),
            "category_to_phrase": dict(self.category_to_phrase),
            # The grouping is the measurement's class space; recording only the
            # forward map would leave "which categories were merged" implicit.
            "phrase_to_categories": {p: list(c) for p, c in self.phrase_to_categories.items()},
        }


def load_taxonomy(path: str) -> Taxonomy:
    """Read the mapping table and assert the properties downstream depends on.

    * **No dotted category names as phrases** — the §0.3 trap.
    * **Every phrase non-empty**, since an empty one cannot own a token span.

    What is deliberately NOT asserted any more: that phrase -> category is
    injective. Rev 1 refused a phrase shared by two categories, on the grounds
    that the reverse map is how Stage 6 picks an epsilon and Stage 8 picks a
    prior. That reasoning was wrong in its premise — those lookups are keyed by
    PHRASE, not by category (X-6) — and the rule blocked the fix for the pilot's
    largest measured defect: a 23-way class space in which five phrases had no
    instances at all and the most-predicted class was impossible by construction
    (C21). Collapsing a class space IS a many-to-one mapping.

    The only thing the old rule really protected was the record field
    `nuscenes_categories`, which nothing downstream reads. The genuine hazard —
    the same phrase appearing twice in the caption — is handled where it lives,
    in `Taxonomy.phrases` (deduplicated) and `build_caption` (refuses repeats).
    """
    if not os.path.isfile(path):
        raise UpstreamRefusal(
            f"taxonomy file not found: {path}. It is mandatory (§0.3): without it the prompts are "
            "dotted category names, which tokenise to nonsense and still return boxes"
        )
    with open(path, "rb") as fh:
        raw_bytes = fh.read()
    mapping_doc = yaml.safe_load(raw_bytes.decode("utf-8"))
    if not isinstance(mapping_doc, dict) or "prompt_phrase" not in mapping_doc:
        raise UpstreamRefusal(f"{path}: expected a mapping with a `prompt_phrase` block")

    pairs = tuple((str(k), str(v).strip()) for k, v in mapping_doc["prompt_phrase"].items())
    if not pairs:
        raise UpstreamRefusal(f"{path}: prompt_phrase is empty")

    seen: dict[str, str] = {}
    for category, phrase in pairs:
        if not phrase:
            raise UpstreamRefusal(f"{path}: category {category!r} has an empty prompt phrase")
        # Many-to-one is legal (C21). Recorded rather than refused, because a
        # collapse the author did not intend should still be visible.
        seen.setdefault(phrase, category)

    thresholds = tuple((str(k), float(v)) for k, v in (mapping_doc.get("thresholds") or {}).items())
    unknown = [k for k, _ in thresholds if k not in seen]
    if unknown:
        raise UpstreamRefusal(f"{path}: thresholds key(s) {unknown} are not prompt phrases")

    excluded = tuple(str(k) for k in (mapping_doc.get("excluded_categories") or {}))
    overlap = sorted(set(excluded) & {c for c, _ in pairs})
    if overlap:
        raise UpstreamRefusal(
            f"{path}: category/categories {overlap} are BOTH mapped to a prompt phrase and listed "
            "in excluded_categories. The class space cannot both contain and exclude a category"
        )

    return Taxonomy(
        path=os.path.realpath(path),
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
        category_to_phrase=pairs,
        thresholds=thresholds,
        default_threshold=float(mapping_doc.get("default_threshold", 0.40)),
        excluded_categories=excluded,
    )


def build_prompt_config(taxonomy: Taxonomy, cfg: ProposalConfig) -> PromptConfig:
    """The §1.5/§5.4 prompt contract object, validated before any image runs."""
    thresholds = dict(taxonomy.thresholds)
    thresholds.update(dict(cfg.thresholds))
    prompt = PromptConfig(
        phrases=taxonomy.phrases,
        thresholds=thresholds,
        default_threshold=cfg.default_threshold,
        source=f"{taxonomy.path}@{taxonomy.sha256[:16]}",
        chunked=bool(cfg.allow_prompt_chunking),
    )
    errors = prompt.validate(prefix="prompt: ")
    if errors:
        raise UpstreamRefusal("; ".join(errors))
    return prompt


# ---------------------------------------------------------------------------
# The caption and its phrase spans (§5.4) — pure, testable, no model needed
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Caption:
    """The concatenated prompt, with each phrase's character span recorded.

    Grounding DINO's caption convention: lowercase phrases, ". "-separated, with
    a trailing period. The spans are computed while building the string rather
    than recovered afterwards with `str.find()` — `find` returns the FIRST match,
    so a phrase that is a substring of an earlier one ("a bus" inside "an
    articulated bus") silently resolves to the wrong span, and every box of that
    class is then labelled as the other.
    """

    text: str
    phrase_char_spans: tuple[tuple[int, int], ...]
    phrases: tuple[str, ...]

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


def build_caption(phrases: Sequence[str]) -> Caption:
    parts: list[str] = []
    spans: list[tuple[int, int]] = []
    cursor = 0
    for phrase in phrases:
        clean = phrase.strip().lower().rstrip(".")
        if not clean:
            raise PhraseSpanError(f"empty phrase in prompt set: {phrase!r}")
        if clean in parts:
            # The failure mode a many-to-one taxonomy makes reachable (C21): the
            # same phrase concatenated twice gets two disjoint token spans, both
            # scoring the same class, and the argmax silently resolves to
            # whichever copy sits at the luckier token position. The caption
            # would look sane and the class would be scored on half its evidence.
            # Taxonomy.phrases deduplicates; this refuses if anything else does not.
            raise PhraseSpanError(
                f"phrase {clean!r} appears twice in the prompt set. The class space must be the "
                "DEDUPLICATED phrase set: a repeated phrase gets two token spans competing for "
                "one class (§5.4)"
            )
        spans.append((cursor, cursor + len(clean)))
        parts.append(clean)
        cursor += len(clean) + len(". ")
    text = ". ".join(parts) + "."
    for (start, end), phrase in zip(spans, parts):
        if text[start:end] != phrase:
            raise PhraseSpanError(
                f"caption span [{start},{end}) is {text[start:end]!r}, expected {phrase!r}"
            )
    return Caption(text=text, phrase_char_spans=tuple(spans), phrases=tuple(parts))


@dataclass(frozen=True)
class PhraseSpanMap:
    """token index -> phrase index, and its inverse.

    `tokens_of_phrase[i]` is the list of token indices belonging to phrase `i`.
    Separators, padding and special tokens belong to no phrase and are dropped:
    scoring a box on a "." token is how a box acquires a confidently wrong class.
    """

    tokens_of_phrase: tuple[tuple[int, ...], ...]
    phrase_of_token: tuple[int, ...]  # -1 == not part of any phrase
    n_tokens: int

    def as_dict(self) -> dict:
        return {
            "n_tokens": self.n_tokens,
            "tokens_per_phrase": [len(t) for t in self.tokens_of_phrase],
        }


def build_phrase_span_map(
    caption: Caption,
    offsets: Sequence[tuple[int, int]],
) -> PhraseSpanMap:
    """Map tokenizer character offsets onto phrase indices.

    `offsets` is the tokenizer's `offset_mapping` for `caption.text`: one
    `(char_start, char_end)` per token, with `(0, 0)` for specials. A token
    belongs to phrase `i` when its character span lies inside phrase `i`'s span;
    overlap across a boundary is impossible for a well-formed tokenizer and is
    an error rather than a coin flip.

    Asserts the map is **total over phrases**: every phrase owns at least one
    token. A phrase that owns none can never win an argmax, so its class silently
    never appears in the output — a stage that runs clean and cannot detect the
    class it was told to find.
    """
    tokens_of_phrase: list[list[int]] = [[] for _ in caption.phrase_char_spans]
    phrase_of_token: list[int] = []

    for token_index, offset in enumerate(offsets):
        start, end = int(offset[0]), int(offset[1])
        if end <= start:  # special / padding token
            phrase_of_token.append(-1)
            continue
        owner = -1
        for phrase_index, (p_start, p_end) in enumerate(caption.phrase_char_spans):
            if start >= p_start and end <= p_end:
                owner = phrase_index
                break
            if start < p_end and end > p_start:
                raise PhraseSpanError(
                    f"token {token_index} spans [{start},{end}) crossing the boundary of phrase "
                    f"{phrase_index} [{p_start},{p_end}); the caption and the tokenizer disagree"
                )
        phrase_of_token.append(owner)
        if owner >= 0:
            tokens_of_phrase[owner].append(token_index)

    empty = [i for i, toks in enumerate(tokens_of_phrase) if not toks]
    if empty:
        raise PhraseSpanError(
            f"phrase(s) {[caption.phrases[i] for i in empty]} own no tokens; they can never win an "
            "argmax and their class would silently never be produced"
        )
    return PhraseSpanMap(
        tokens_of_phrase=tuple(tuple(t) for t in tokens_of_phrase),
        phrase_of_token=tuple(phrase_of_token),
        n_tokens=len(offsets),
    )


def phrase_scores(token_probs: np.ndarray, span_map: PhraseSpanMap) -> np.ndarray:
    """(Q, T) token probabilities -> (Q, P) per-phrase scores.

    Max over the phrase's tokens: a multi-word phrase is recognised when any of
    its tokens fires, and a mean would dilute long phrases against short ones on
    top of Grounding DINO's already-uncalibrated cross-phrase scores (§7.3.3).
    """
    probs = np.asarray(token_probs, dtype=np.float32)
    if probs.ndim != 2:
        raise ValueError(f"token_probs must be (Q, T), got {probs.shape}")
    if probs.shape[1] < span_map.n_tokens:
        raise PhraseSpanError(
            f"token_probs has {probs.shape[1]} token columns but the caption tokenised to "
            f"{span_map.n_tokens}; the span map does not describe this tensor"
        )
    out = np.zeros((probs.shape[0], len(span_map.tokens_of_phrase)), dtype=np.float32)
    for phrase_index, tokens in enumerate(span_map.tokens_of_phrase):
        out[:, phrase_index] = probs[:, list(tokens)].max(axis=1)
    return out


# ---------------------------------------------------------------------------
# Box geometry and deduplication
# ---------------------------------------------------------------------------


def cxcywh_norm_to_xyxy_px(boxes: np.ndarray, width_px: int, height_px: int) -> np.ndarray:
    """Normalised cxcywh (the model's space) -> absolute xyxy pixels (§1.5 rule 1).

    Both conversions in one place. Doing the two independently is how a box ends
    up half-converted — plausible, on the image, and wrong by a factor of two in
    one axis.
    """
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    cx, cy, w, h = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    xyxy = np.stack(
        [(cx - w / 2) * width_px, (cy - h / 2) * height_px, (cx + w / 2) * width_px, (cy + h / 2) * height_px],
        axis=1,
    )
    return xyxy.astype(np.float32)


def _areas(boxes: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])


def pairwise_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between two absolute-pixel xyxy box sets, in the SAME image plane."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0.0, None)
    inter = wh[..., 0] * wh[..., 1]
    union = _areas(a)[:, None] + _areas(b)[None, :] - inter
    return np.where(union > 0.0, inter / np.maximum(union, 1e-12), 0.0)


def deduplicate(
    boxes_xyxy_px: np.ndarray,
    scores: np.ndarray,
    class_names: Sequence[str],
    *,
    same_class_iou: float,
    cross_class_iou: float,
    max_keep: int,
) -> tuple[np.ndarray, dict]:
    """Greedy NMS with a deterministic order. Returns the kept indices and a ledger.

    Two thresholds, for two different failure modes:

    * **Same class, `same_class_iou`** — ordinary duplicate suppression.
    * **Different class, `cross_class_iou` (high)** — Grounding DINO routinely
      emits one box under several phrases ("a car" / "a truck" on the same van).
      Suppressing those needs near-identity, so the test is IoU. Using IoA here
      instead would delete a pedestrian standing in front of a bus: IoA(person,
      bus) is ~1.0 while IoU is ~0.02, and the deletion would be invisible.

    Determinism (§1.9): the order is score descending, then the box coordinates,
    then the class name — never the model's output order, which varies with
    query permutation and would break the byte-compare re-run test.
    """
    boxes = np.asarray(boxes_xyxy_px, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    names = list(class_names)
    n = len(boxes)
    ledger = {"n_in": n, "n_suppressed_same_class": 0, "n_suppressed_cross_class": 0, "truncated": False}
    if n == 0:
        return np.zeros((0,), dtype=np.int64), ledger

    order = sorted(
        range(n),
        key=lambda i: (-float(scores[i]), float(boxes[i][0]), float(boxes[i][1]),
                       float(boxes[i][2]), float(boxes[i][3]), names[i]),
    )
    iou = pairwise_iou(boxes, boxes)
    kept: list[int] = []
    suppressed = np.zeros(n, dtype=bool)
    for i in order:
        if suppressed[i]:
            continue
        kept.append(i)
        for j in order:
            if j == i or suppressed[j]:
                continue
            same = names[j] == names[i]
            threshold = same_class_iou if same else cross_class_iou
            if iou[i, j] > threshold:
                suppressed[j] = True
                ledger["n_suppressed_same_class" if same else "n_suppressed_cross_class"] += 1

    if len(kept) > max_keep:
        # A cap that silently truncates reads downstream as "this image had few
        # objects". Recorded, both the flag and the count (§1.9).
        ledger["truncated"] = True
        ledger["n_dropped_by_cap"] = len(kept) - max_keep
        kept = kept[:max_keep]
    ledger["n_out"] = len(kept)
    return np.asarray(kept, dtype=np.int64), ledger


# ---------------------------------------------------------------------------
# The adapter (role `proposal_2d`)
# ---------------------------------------------------------------------------


class GroundingDinoAdapter:
    """A Grounding-DINO-family checkpoint as the `proposal_2d` role.

    Serves both the default tier (iSEE-Laboratory/llmdet_large, an MM-Grounding-
    DINO Swin-L; model_type `mm-grounding-dino`) and the pilot tier
    (IDEA-Research/grounding-dino-tiny): AutoProcessor resolves both to the
    GroundingDinoProcessor and AutoModelForZeroShotObjectDetection to the
    matching *ForObjectDetection class, and both emit per-token logits over the
    caption plus normalised cxcywh `pred_boxes`.

    Owns, and never leaks into stage code (§1.5 rule 2):
      * the caption, its character spans and its token spans;
      * the forward transform (the processor's shortest-side resize) and the
        inverse (normalised cxcywh in the processed space -> absolute xyxy at
        1600x900);
      * per-class thresholding, applied to per-phrase scores rather than to the
        max over the whole caption.
    """

    def __init__(self, spec: CheckpointSpec, prompt: PromptConfig, cfg: ProposalConfig) -> None:
        self._spec = spec
        self._cfg = cfg
        self._prompt = prompt
        self._caption = build_caption(prompt.phrases)
        self._span_map: PhraseSpanMap | None = None
        self._model: Any = None
        self._processor: Any = None
        self._torch: Any = None
        self._device = cfg.device
        # Filled by load() from DHAKASCENES_VRAM_CAP_MIB (C1); recorded verbatim
        # in the run manifest.
        self.vram_cap: dict = {"value_mib": None, "enforced": "none",
                               "physical_device_mib": None, "device_name": ""}

    # --- ModelRole surface ---

    @property
    def roles(self) -> tuple[str, ...]:
        return (PROPOSAL_2D,)

    @property
    def spec(self) -> CheckpointSpec:
        return self._spec

    @property
    def device(self) -> str:
        return self._device

    @property
    def returns_masks(self) -> bool:
        # Grounding DINO returns boxes only, so Stage 4 is NOT a pass-through
        # here. A SAM-3-class provider would return True and collapse the
        # Stage 3/4 boundary (§7.1 fix 3).
        return False

    @property
    def caption(self) -> Caption:
        return self._caption

    @property
    def span_map(self) -> PhraseSpanMap:
        if self._span_map is None:
            raise RuntimeError("span map is built by load(); call load() first")
        return self._span_map

    def load(self) -> None:
        try:
            import torch
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError as exc:
            raise ModelUnavailable(f"{self._spec.model_id}: {exc}") from exc

        self._torch = torch
        # C1: the synthetic 4 GiB ceiling, applied BEFORE the first allocation.
        self.vram_cap = apply_vram_cap(torch, self._device)
        kwargs: dict[str, Any] = {}
        if self._spec.revision:
            kwargs["revision"] = self._spec.revision
        self._processor = AutoProcessor.from_pretrained(self._spec.model_id, **kwargs)
        # dtype is explicit: transformers v5 defaults to dtype='auto', which loads
        # whatever precision the checkpoint stores. Our checkpoints store fp32,
        # but the contract is fp32-in-memory (half precision only ever via
        # autocast, see ProposalConfig.autocast_fp16 provenance), so it is stated
        # rather than inherited from checkpoint metadata.
        self._model = (
            AutoModelForZeroShotObjectDetection.from_pretrained(
                self._spec.model_id, dtype=torch.float32, **kwargs
            )
            .to(self._device)
            .eval()
        )

        # The span map is built ONCE, from the tokenizer's own character offsets,
        # and asserted total before any image is seen (§5.4).
        tokenizer = getattr(self._processor, "tokenizer", self._processor)
        encoded = tokenizer(self._caption.text, return_offsets_mapping=True, return_tensors=None)
        offsets = encoded["offset_mapping"]
        self._span_map = build_phrase_span_map(self._caption, offsets)

    def unload(self) -> None:
        self._model = None
        self._processor = None
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

    # --- the role method ---

    def propose(self, image: np.ndarray, prompt: PromptConfig, *, channel: str = "") -> Proposals:
        """One 1600x900 RGB frame -> deduplicated absolute-pixel proposals."""
        if self._model is None:
            raise RuntimeError("adapter is not loaded")
        torch = self._torch
        height_px, width_px = int(image.shape[0]), int(image.shape[1])
        if (width_px, height_px) != (self._cfg.image_width_px, self._cfg.image_height_px):
            raise RoleContractError(
                f"image is {width_px}x{height_px}; Stage 3 runs at "
                f"{self._cfg.image_width_px}x{self._cfg.image_height_px} (§1.5 rule 1)"
            )

        inputs = self._processor(images=image, text=self._caption.text, return_tensors="pt")
        processed_h, processed_w = int(inputs["pixel_values"].shape[-2]), int(inputs["pixel_values"].shape[-1])
        _assert_aspect_preserved(processed_w, processed_h, width_px, height_px)
        _assert_unpadded(inputs)
        inputs = inputs.to(self._device)

        # Half precision is autocast-only (see ProposalConfig.autocast_fp16
        # provenance): model.half() mixes dtypes in the text path and breaks the
        # forward, so the weights stay fp32 and only the compute is downcast.
        use_autocast = bool(self._cfg.autocast_fp16) and str(self._device).startswith("cuda")
        autocast_ctx = (
            torch.autocast("cuda", dtype=torch.float16) if use_autocast else contextlib.nullcontext()
        )
        with torch.inference_mode(), autocast_ctx:
            outputs = self._model(**inputs)

        # (Q, T) token probabilities -> (Q, P) phrase scores, through the span map.
        token_probs = outputs.logits.sigmoid()[0].float().cpu().numpy()
        scores_by_phrase = phrase_scores(token_probs, self.span_map)
        best_phrase = np.argmax(scores_by_phrase, axis=1)
        best_score = scores_by_phrase[np.arange(len(best_phrase)), best_phrase]

        phrases = list(self._caption.phrases)
        # Per-class thresholds, applied per phrase (§7.3.3). A single scalar over
        # the caption max is what X-5 rejects: scores are not calibrated across
        # phrases and longer phrases score systematically lower.
        keep = np.array(
            [best_score[q] >= prompt.threshold_for(phrases[best_phrase[q]]) for q in range(len(best_score))],
            dtype=bool,
        )

        boxes_norm = outputs.pred_boxes[0].float().cpu().numpy()[keep]
        boxes = cxcywh_norm_to_xyxy_px(boxes_norm, width_px, height_px)
        scores = best_score[keep].astype(np.float32)
        phrase_index = best_phrase[keep]
        class_names = [phrases[i] for i in phrase_index]

        # Clip and drop degenerates in the ADAPTER (§1.5 rule 2), before dedup.
        if len(boxes):
            boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0.0, float(width_px))
            boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0.0, float(height_px))
            side_ok = ((boxes[:, 2] - boxes[:, 0]) >= self._cfg.min_box_side_px) & (
                (boxes[:, 3] - boxes[:, 1]) >= self._cfg.min_box_side_px
            )
            boxes, scores = boxes[side_ok], scores[side_ok]
            phrase_index = phrase_index[side_ok]
            class_names = [n for n, k in zip(class_names, side_ok) if k]

        kept, ledger = deduplicate(
            boxes,
            scores,
            class_names,
            same_class_iou=self._cfg.same_class_iou,
            cross_class_iou=self._cfg.cross_class_iou,
            max_keep=self._cfg.max_proposals_per_image,
        )
        boxes, scores = boxes[kept], scores[kept]
        phrase_index = phrase_index[kept]
        class_names = [class_names[i] for i in kept]

        proposals = Proposals(
            boxes_xyxy_px=boxes,
            scores=np.clip(scores, 0.0, 1.0),
            class_names=class_names,
            # The span carried per box is the CHARACTER span of its phrase in the
            # caption, recovered by index rather than by string search, so the
            # phrase -> class mapping is testable rather than trusted (§5.4).
            phrase_spans=[self._caption.phrase_char_spans[i] for i in phrase_index],
            prompt_config=prompt,
            channel=channel,
            image_size_px=(width_px, height_px),
            resize_policy=self._cfg.resize_policy,
            model_input_size_px=(processed_w, processed_h),
        )
        proposals.assert_valid(prefix=f"{channel}: ")
        # The ledger rides along for the record; Proposals is the contract object
        # and deliberately has no room for stage bookkeeping.
        setattr(proposals, "dedup_ledger", ledger)
        return proposals


def _assert_aspect_preserved(
    processed_w: int, processed_h: int, source_w: int, source_h: int
) -> None:
    """§1.5 rule 3, enforced on every image rather than trusted to a config.

    A square resize of 16:9 imagery with a uniform inverse stretches every box
    along one axis. The boxes still look like boxes and every downstream 3D
    extent is wrong in one axis, consistently — the failure never surfaces as an
    error, only as slightly wrong dimensions everywhere.
    """
    if processed_w <= 0 or processed_h <= 0:
        raise RoleContractError(f"processed tensor is {processed_w}x{processed_h}")
    source_aspect = source_w / source_h
    processed_aspect = processed_w / processed_h
    if abs(processed_aspect - source_aspect) > ASPECT_TOLERANCE:
        raise RoleContractError(
            f"the processor resized {source_w}x{source_h} (aspect {source_aspect:.4f}) to "
            f"{processed_w}x{processed_h} (aspect {processed_aspect:.4f}). Square or otherwise "
            "non-aspect-preserving resizes are forbidden (§1.5 rule 3): letterbox with recorded "
            "padding, or resize the shortest side"
        )


def _assert_unpadded(inputs: Any) -> None:
    """Padding shifts the normalised box origin; with one image there must be none.

    A DETR-style processor pads a batch to its largest member. Batched with a
    different-sized image, the normalised coordinates would be relative to the
    padded canvas and the inverse transform above would be wrong by the padding
    fraction — small, uniform, and undetectable by eye.
    """
    mask = inputs.get("pixel_mask") if hasattr(inputs, "get") else None
    if mask is None:
        return
    array = np.asarray(mask)
    if array.size and not bool(array.all()):
        raise RoleContractError(
            "pixel_mask contains padding; Stage 3 processes one image at a time precisely so that "
            "normalised boxes are relative to the image and not to a padded canvas"
        )


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------


# Canonical implementation now in pipeline.common.manifest (C3). The name stays
# importable from here because stages 4 and 5 already import it from this module.
write_jsonl_atomic = _write_jsonl_atomic


def proposal_row(
    keyframe: KeyframeRecord,
    channel: str,
    proposals: Proposals,
    *,
    taxonomy: Taxonomy,
    caption: Caption,
    span_map: PhraseSpanMap,
    spec: CheckpointSpec,
    cfg: ProposalConfig,
) -> dict:
    """One jsonl row: the proposals for one camera of one keyframe.

    Every Stage 3 record states the actual image resolution and the prompt
    configuration used (P1-14) — not because it is likely to change, but because
    a resolution fallback or a chunked prompt is otherwise invisible in the
    output it produced.
    """
    phrase_to_categories = taxonomy.phrase_to_categories
    # Strict, not .get(name, ""): every class name descends from the taxonomy's
    # own phrases, so a miss here is a broken invariant — and an empty-string
    # category would ride silently into Stage 8's priors lookup, where an
    # unmatched key means "use the default" (C18 [V-adjacent], fixed 2026-08-12).
    for name in proposals.class_names:
        if name not in phrase_to_categories:
            raise PhraseSpanError(
                f"{channel}: class name {name!r} is not a phrase of taxonomy "
                f"{taxonomy.sha256[:12]}; refusing to emit an unmappable category"
            )
    obs = keyframe.cameras[channel]
    return {
        "spec": STAGE_SPEC,
        "keyframe_token": keyframe.keyframe_token,
        "scene_token": keyframe.scene_token,
        "t_ns": keyframe.t_ns,
        "time_base": keyframe.time_base,
        "coverage_config": keyframe.coverage_config,
        "channel": channel,
        "sample_data_token": obs.sample_data_token,
        # Carried forward because Stage 4 needs the intrinsic/extrinsic to put
        # this camera's boxes into the ego frame the ring shares, and Stage 5
        # needs both poses for the four-hop chain. Re-deriving them from the
        # metadata by channel name would silently pick the wrong keyframe's
        # calibration on any substrate where calibration is not constant.
        "calibrated_sensor_token": obs.calibrated_sensor_token,
        "ego_pose_token": obs.ego_pose_token,
        "dt_ns": obs.dt_ns,
        "image_path": obs.path,
        "image_size_px": list(proposals.image_size_px),
        "model_input_size_px": list(proposals.model_input_size_px or ()),
        "resize_policy": proposals.resize_policy,
        "checkpoint": {"model_id": spec.model_id, "revision": spec.revision, "sha256": spec.sha256},
        "prompt": {
            **proposals.prompt_config.as_dict(),
            "caption_sha256": caption.sha256,
            "taxonomy_sha256": taxonomy.sha256,
            "span_map": span_map.as_dict(),
        },
        "n_proposals": len(proposals),
        "dedup": getattr(proposals, "dedup_ledger", {}),
        "boxes_xyxy_px": np.asarray(proposals.boxes_xyxy_px, dtype=np.float32).round(3).tolist(),
        "scores": np.asarray(proposals.scores, dtype=np.float32).round(5).tolist(),
        "class_names": list(proposals.class_names),
        # The class space is many-to-one (C21), so a box's phrase no longer names
        # ONE nuScenes category: "a pedestrian" covers five. The field therefore
        # carries every category the phrase collapses — the honest answer to
        # "what could this be?" — instead of a single arbitrary winner. Record
        # only; nothing downstream reads it (downstream keys on the phrase, X-6).
        "nuscenes_categories": [list(phrase_to_categories[n]) for n in proposals.class_names],
        "phrase_char_spans": [list(s) for s in proposals.phrase_spans],
        "seed": cfg.global_seed,
    }


# ---------------------------------------------------------------------------
# Upstream binding (§1.8, §1.9)
# ---------------------------------------------------------------------------


def load_upstream(paths: Paths, stage1_dir: str, *, accept_degraded: bool = False):
    """Refuse to start unless Stage 1 COMPLETED and was bound to THIS substrate.

    The gate itself lives in `pipeline.common.manifest.require_upstream` (C16):
    no marker means incomplete (unconditional refusal); a degraded marker means
    complete-but-flagged and needs the explicit `accept_degraded` opt-in, which
    this stage then records in its own manifest.
    """
    return require_upstream(
        stage1_dir,
        stage_name="Stage 1",
        module_hint="pipeline.stage1_ingestion.ingest",
        current_fingerprint=metadata_fingerprint(paths),
        accept_degraded=accept_degraded,
    )


def scene_dirs(stage1_dir: str, wanted: Sequence[str] | None) -> list[tuple[str, str]]:
    root = os.path.join(stage1_dir, "scenes")
    if not os.path.isdir(root):
        raise UpstreamRefusal(f"{root} not found; Stage 1 wrote no scenes")
    names = sorted(n for n in os.listdir(root) if os.path.isdir(os.path.join(root, n)))
    if wanted:
        missing = sorted(set(wanted) - set(names))
        if missing:
            raise UpstreamRefusal(f"requested scene(s) not present in Stage 1 output: {missing}")
        names = [n for n in names if n in wanted]
    return [(n, os.path.join(root, n)) for n in names]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run(
    paths: Paths,
    upstream: dict,
    upstream_marker,
    taxonomy: Taxonomy,
    cfg: ProposalConfig,
    stage1_dir: str,
    out_dir: str,
    scene_names: Sequence[str] | None,
) -> tuple[dict, int]:
    started = time.time()
    # Any marker still standing describes the PREVIOUS run of this stage; it
    # comes down before the first write (C16).
    clear_markers(out_dir)
    prompt = build_prompt_config(taxonomy, cfg)
    # The provider name is derived from the model id, not hardcoded, so the
    # manifest stays honest for BOTH the default (llmdet_large, C19) and an
    # explicitly-passed pilot checkpoint (grounding-dino-tiny, §7.1/C1). Both
    # derivations match a registered provider name below.
    provider = cfg.model_id.rsplit("/", 1)[-1].replace("-", "_").lower()
    default_model_id = type(cfg).__dataclass_fields__["model_id"].default
    spec = CheckpointSpec(
        role=PROPOSAL_2D,
        provider=provider,
        model_id=cfg.model_id,
        revision=cfg.revision,
        provenance=(
            cfg.provenance.get("model_id", "")
            if cfg.model_id == default_model_id
            else "explicit model_id override; default tier is C19 (see config.provenance.model_id)"
        ),
    )
    spec_errors = spec.validate(prefix="checkpoint: ")
    if spec_errors:
        raise UpstreamRefusal(
            "; ".join(spec_errors) + " — pass --revision with the hub commit sha for this checkpoint"
        )
    adapter = GroundingDinoAdapter(spec, prompt, cfg)
    adapter.load()

    per_scene: list[dict] = []
    degraded = False
    n_images = 0
    n_proposals = 0
    n_empty = 0
    n_truncated = 0

    from PIL import Image  # local: Stage 3 is the first stage that needs imagery

    for scene_name, scene_dir in scene_dirs(stage1_dir, scene_names):
        keyframes = read_records(os.path.join(scene_dir, "keyframes.jsonl"), expect_type=KeyframeRecord)
        rows: list[dict] = []
        scene_images = 0
        scene_proposals = 0
        scene_empty = 0
        per_class: dict[str, int] = {}

        for keyframe in keyframes:
            for channel in RING_CAMERAS:
                obs = keyframe.cameras.get(channel)
                if obs is None:
                    continue
                image_path = os.path.join(paths.dataroot, obs.path)
                with Image.open(image_path) as im:
                    image = np.asarray(im.convert("RGB"), dtype=np.uint8)
                proposals = adapter.propose(image, prompt, channel=channel)
                row = proposal_row(
                    keyframe,
                    channel,
                    proposals,
                    taxonomy=taxonomy,
                    caption=adapter.caption,
                    span_map=adapter.span_map,
                    spec=spec,
                    cfg=cfg,
                )
                rows.append(row)
                scene_images += 1
                scene_proposals += row["n_proposals"]
                if row["n_proposals"] == 0:
                    scene_empty += 1
                if row["dedup"].get("truncated"):
                    n_truncated += 1
                for name in row["class_names"]:
                    per_class[name] = per_class.get(name, 0) + 1

        out_scene = os.path.join(out_dir, "scenes", scene_name)
        write_jsonl_atomic(os.path.join(out_scene, "proposals.jsonl"), rows)

        summary = {
            "scene": scene_name,
            "n_keyframes": len(keyframes),
            "n_images": scene_images,
            "n_proposals": scene_proposals,
            "n_images_with_zero_proposals": scene_empty,
            "proposals_per_image": round(scene_proposals / scene_images, 2) if scene_images else 0.0,
            "per_class": dict(sorted(per_class.items())),
            # Zero proposals on an entire image is a legitimate value and a
            # symptom; either way it is surfaced, never averaged away.
            "degraded": scene_empty > 0,
        }
        per_scene.append(summary)
        degraded = degraded or summary["degraded"]
        n_images += scene_images
        n_proposals += scene_proposals
        n_empty += scene_empty
        print(
            f"  {scene_name}  {len(keyframes):>3} kf  {scene_images:>4} img  "
            f"{scene_proposals:>5} proposals  {summary['proposals_per_image']:>6.2f}/img  "
            f"{scene_empty:>3} empty" + ("  DEGRADED" if summary["degraded"] else "")
        )

    adapter.unload()

    manifest = {
        "spec": STAGE_SPEC,
        "stage": STAGE,
        "seed": cfg.global_seed,
        "config": cfg.as_dict(),
        "upstream": {
            "metadata_fingerprint": upstream["upstream"]["metadata_fingerprint"],
            "fingerprint_spec": upstream["upstream"]["fingerprint_spec"],
            "stage1_spec": upstream["spec"],
            # C16: a run built on accepted degradation says so in its provenance.
            "degraded": upstream_marker.degraded,
            "degraded_causes": list(upstream_marker.causes),
            "accepted_degraded_upstream": cfg.accept_degraded_upstream,
        },
        "paths": paths.as_dict(),
        "taxonomy": taxonomy.as_dict(),
        "prompt": {**prompt.as_dict(), "caption": adapter.caption.text, "caption_sha256": adapter.caption.sha256},
        "checkpoint": {"model_id": spec.model_id, "revision": spec.revision, "sha256": spec.sha256},
        "vram_cap": adapter.vram_cap,  # C1 — synthetic ceiling, or the honest absence of one
        "image_size_px": [cfg.image_width_px, cfg.image_height_px],
        "numpy_version": np.__version__,
        "python_version": sys.version.split()[0],
        "elapsed_s": round(time.time() - started, 2),
        "scenes": per_scene,
        "totals": {
            "n_images": n_images,
            "n_proposals": n_proposals,
            "n_images_with_zero_proposals": n_empty,
            "n_images_truncated_by_cap": n_truncated,
            "proposals_per_image": round(n_proposals / n_images, 2) if n_images else 0.0,
        },
    }
    return manifest, EXIT_DEGRADED if degraded or n_truncated else EXIT_OK


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--paths", default=os.environ.get("DHAKASCENES_PATHS_CONFIG", "configs/paths.yaml"))
    parser.add_argument("--taxonomy", default="configs/taxonomy_pilot_nuscenes.yaml")
    parser.add_argument("--stage1-dir", default=None, help="default <work_root>/stage1_ingestion")
    parser.add_argument("--out-dir", default=None, help="default <work_root>/stage3_proposals")
    parser.add_argument("--scenes", nargs="*", default=None, help="subset of Stage 1 scene names")
    parser.add_argument("--seed", type=int, default=None, help="override the global seed (recorded)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--model-id",
        default=None,
        help="override the proposal_2d checkpoint (default iSEE-Laboratory/llmdet_large, C19; "
        "the pilot IDEA-Research/grounding-dino-tiny stays selectable)",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="hub commit sha for the checkpoint; required, an unpinned id tracks the default branch",
    )
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

    cfg = ProposalConfig(
        device=args.device,
        revision=args.revision,
        accept_degraded_upstream=args.accept_degraded_upstream,
        **({"model_id": args.model_id} if args.model_id else {}),
        **({"global_seed": args.seed} if args.seed is not None else {}),
    )

    try:
        taxonomy = load_taxonomy(args.taxonomy)
        upstream, upstream_marker = load_upstream(
            paths, stage1_dir, accept_degraded=cfg.accept_degraded_upstream
        )
        manifest, code = run(
            paths, upstream, upstream_marker, taxonomy, cfg, stage1_dir, out_dir, args.scenes
        )
    except UpstreamRefusal as exc:
        print(f"REFUSING TO START: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    except ModelUnavailable as exc:
        print(f"REFUSING TO START: model unavailable: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    except (SchemaValidationError, PathValidationError, RoleContractError, PhraseSpanError) as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_REFUSED

    write_json_atomic(os.path.join(out_dir, "run_manifest.json"), manifest)
    # Three-state marker (§1.9, C16): clean / degraded-with-causes / absent.
    write_marker(
        out_dir,
        manifest["upstream"]["metadata_fingerprint"],
        degraded=code == EXIT_DEGRADED,
        causes=[
            f"{s['scene']}: {s['n_images_with_zero_proposals']} image(s) with zero proposals"
            for s in manifest["scenes"]
            if s["degraded"]
        ],
    )

    t = manifest["totals"]
    print(f"images               : {t['n_images']}")
    print(f"proposals            : {t['n_proposals']}  ({t['proposals_per_image']}/image)")
    print(f"images with none     : {t['n_images_with_zero_proposals']}")
    print(f"images hitting cap   : {t['n_images_truncated_by_cap']}")
    print(f"prompt phrases       : {len(manifest['prompt']['phrases'])}  chunked={manifest['prompt']['chunked']}")
    print(f"wrote {out_dir}")
    return code


# One adapter class serves the whole Grounding-DINO family: the default tier
# (LLMDet Swin-L, C19) and the pilot tier (tiny, §7.1/C1) share the processor,
# the caption convention, and the raw-logits output contract.
register("llmdet_large", PROPOSAL_2D, GroundingDinoAdapter)
register("grounding_dino_tiny", PROPOSAL_2D, GroundingDinoAdapter)


if __name__ == "__main__":
    raise SystemExit(main())
