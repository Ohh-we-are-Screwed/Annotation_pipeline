#!/usr/bin/env python3
"""Phase 5b — measured VRAM per role, on the card, at 1600x900.

Phase 5a checks arithmetic on paper. This script is the other half: it downloads
the pilot checkpoints, runs **one real forward pass per role at the real
1600x900 source resolution with the real prompt set**, and measures what the
card is actually holding.

**Why not `max_memory_allocated()`** (the measurement rev 1 asked for, X-3): it
counts only live tensor bytes. It excludes the CUDA context (~300 MB), it
excludes reserved-but-unallocated caching-allocator blocks, it excludes
cuBLAS/cuDNN workspaces and kernel modules, and it knows nothing about the
display server or any other process on the card. On a 4 GB device the gap
between "allocated" and "occupied" *is* the margin being budgeted. So every role
is measured two ways and the larger is believed:

  * `torch.cuda.max_memory_reserved()` — the allocator's peak reservation.
  * `torch.cuda.mem_get_info()` free-deltas — device-level occupancy, which
    catches everything the allocator never sees.

Protocol, in order:

  1. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is pinned **before torch
     is imported** (below), because the allocator reads it at first CUDA init and
     ignores later mutation. Grounding DINO's variable-length text tensors are
     exactly the workload that fragments a non-expandable pool.
  2. The **display-server / other-process allocation is measured first**, via
     `nvidia-smi`, before this process creates a context. `mem_get_info()` cannot
     do it: calling it creates the context it was supposed to measure.
  3. The CUDA context cost is then measured on its own.
  4. Each role is loaded, exercised, measured, and torn down **serially** — one
     role resident at a time, per §7.1's policy.
  5. Between every teardown, `torch.cuda.memory_allocated()` must return to ~0.
     `empty_cache()` does not free memory still referenced by a live Python
     object, so this is asserted, not assumed (§7.1). Failure is exit code 5.
  6. A role whose occupancy exceeds `hard_ceiling_mb` is a **hard stop** with
     role / resolution / prompt count logged (§1.9). Never a silent retry at a
     lower resolution: that would change the output distribution mid-run. The
     first legitimate mitigation is a **letterboxed** reduction recorded in every
     Stage 3 record — never a square resize (§1.5 rule 3), never silent prompt
     chunking (§5.4).

**The adapters in this file are measurement harnesses, not the Stage 3/4
adapters.** They exist to put a real workload of the right shape on the card. The
production adapters (Phase 6) own phrase-span bookkeeping and the inverse
transform, and when they land this measurement is re-run against them. Nothing
here is imported by pipeline code, and the JSON records `adapter_kind:
"measurement_harness"` so a peak measured here is never quoted as a peak for the
shipped adapter.

Usage:
    python3 scripts/measure_vram.py [--model-config configs/models_pilot.yaml]
                                    [--paths configs/paths.yaml]
                                    [--taxonomy configs/taxonomy_pilot_nuscenes.yaml]
                                    [--roles embedding_ood,proposal_2d,...]
                                    [--image PATH] [--iters 3] [--out PATH]
                                    [--allow-placeholder-prompts]
                                    [--allow-nontarget-device]

Exit codes:
    0  every requested role measured and every peak inside the hard ceiling
    2  config / path / role-contract violated, or the substrate is not addressable
    3  HARD STOP: a role exceeded the hard ceiling, or OOMed
    4  a role could not be measured (missing dependency or checkpoint)
    5  teardown assertion failed: the device did not return to quiescent
"""

from __future__ import annotations

# --- allocator configuration, before any torch import ----------------------
# The caching allocator parses PYTORCH_CUDA_ALLOC_CONF once, at first CUDA
# initialisation. Setting it after `import torch` has already initialised CUDA
# is a no-op that leaves no trace, so the ordering is enforced rather than
# documented: if torch is already in sys.modules, this process cannot honour the
# contract and refuses to produce a number.
import os as _os
import sys as _sys

ALLOC_CONF = "expandable_segments:True"


def _pin_alloc_conf() -> str:
    if "torch" in _sys.modules:
        raise SystemExit(
            "measure_vram: torch was imported before PYTORCH_CUDA_ALLOC_CONF could be pinned; "
            "run this file as a script, not as an import into a torch process"
        )
    existing = _os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    if existing and existing != ALLOC_CONF:
        raise SystemExit(
            f"measure_vram: PYTORCH_CUDA_ALLOC_CONF is already {existing!r} in the environment; "
            f"this measurement is defined at {ALLOC_CONF!r} and will not silently measure another "
            "allocator configuration"
        )
    _os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ALLOC_CONF
    return ALLOC_CONF


_PINNED_ALLOC_CONF = _pin_alloc_conf()

import argparse  # noqa: E402
import gc  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from typing import Any, Sequence  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.common.model_interfaces import (  # noqa: E402
    EMBEDDING_OOD,
    MASK_2D,
    ORIGINAL_SIZE_PX,
    PROPOSAL_2D,
    REID_EMBEDDING,
    ROLES,
    CheckpointSpec,
    EmbeddingBatch,
    MaskResult,
    ModelConfig,
    PromptConfig,
    Proposals,
    RoleContractError,
    TemporalWindow,
    assert_role_conformance,
    load_model_config,
)
from pipeline.common.paths import (  # noqa: E402
    PathValidationError,
    Paths,
    assert_dataroot_read_only,
    load_paths,
    metadata_fingerprint,
)
from pipeline.common.schemas import IMAGE_HEIGHT_PX, IMAGE_WIDTH_PX  # noqa: E402
from pipeline.stage0_data_probe.probe import write_json_atomic  # noqa: E402

MB = 1024.0 * 1024.0

# `memory_allocated()` tolerance between role teardowns. Not zero: a few pinned
# bookkeeping bytes can survive an empty_cache() legitimately. One MiB is far
# below any model tensor and far above any bookkeeping.
QUIESCENT_TOLERANCE_MB = 1.0

# Placeholder prompt set, used ONLY with --allow-placeholder-prompts and only to
# get the right *prompt length* on the card before Phase 6 writes the real
# mapping table. 23 phrases, matching the 23 nuScenes categories, because text
# length drives the text-encoder's activation footprint. Any run using these is
# recorded quotable=false: these are not the tuned prompt strings and their
# confidences mean nothing (§0.3, §5.4).
PLACEHOLDER_PHRASES: tuple[str, ...] = (
    "an animal", "an adult pedestrian", "a child", "a construction worker",
    "a person on a personal mobility device", "a police officer", "a stroller",
    "a wheelchair", "a movable traffic barrier", "a debris object on the road",
    "a pushable or pullable object", "a traffic cone", "a bicycle rack",
    "a bicycle", "a bendy bus", "a rigid bus", "a car", "a construction vehicle",
    "a motorcycle", "a trailer", "a truck", "an emergency ambulance",
    "a police car",
)


# ---------------------------------------------------------------------------
# Device bookkeeping
# ---------------------------------------------------------------------------


class MeasurementError(RuntimeError):
    """A role could not be measured (missing dependency, checkpoint, or device)."""


class TeardownError(RuntimeError):
    """The device did not return to quiescent between roles (§7.1)."""


class CeilingExceeded(RuntimeError):
    """A measured peak exceeded the configured hard ceiling (§7.2, §1.9)."""


def _nvidia_smi_used_mb(device_index: int) -> float | None:
    """Bytes already resident on the card, before this process makes a context.

    `mem_get_info()` cannot answer this: calling it initialises the context whose
    cost is the thing being separated out. On a 4 GB laptop card the display
    server is a material fraction of the budget, so it is measured, not assumed.
    """
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                f"--id={device_index}",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        ).stdout.strip().splitlines()
        return float(out[0].strip())
    except Exception:
        return None


@dataclass
class DeviceBaseline:
    index: int
    name: str
    total_mb: float
    pre_context_used_mb: float | None      # display server + other processes
    post_context_used_mb: float           # after this process holds a context
    context_mb: float | None              # our context alone
    capability: str
    driver: str


def _establish_baseline(torch, device_index: int) -> DeviceBaseline:
    pre_used = _nvidia_smi_used_mb(device_index)

    # Force context creation deliberately, and measure it on its own.
    torch.cuda.set_device(device_index)
    torch.cuda.init()
    probe = torch.empty(1, device=f"cuda:{device_index}")
    torch.cuda.synchronize()
    free_b, total_b = torch.cuda.mem_get_info(device_index)
    del probe
    gc.collect()
    torch.cuda.empty_cache()
    free_b, total_b = torch.cuda.mem_get_info(device_index)

    total_mb = total_b / MB
    post_used_mb = (total_b - free_b) / MB
    props = torch.cuda.get_device_properties(device_index)
    return DeviceBaseline(
        index=device_index,
        name=props.name,
        total_mb=total_mb,
        pre_context_used_mb=pre_used,
        post_context_used_mb=post_used_mb,
        context_mb=None if pre_used is None else max(0.0, post_used_mb - pre_used),
        capability=f"{props.major}.{props.minor}",
        driver=getattr(torch.version, "cuda", "") or "",
    )


def _assert_quiescent(torch, when: str, device_index: int) -> float:
    """`memory_allocated()` must be ~0 between roles (§7.1).

    A live Python reference — a cached output tensor, a closure, a module still
    on the device — survives `empty_cache()` and quietly shifts the whole budget
    for every role measured after it. That failure is silent by construction, so
    it is checked at every boundary rather than at the end.
    """
    gc.collect()
    torch.cuda.synchronize(device_index)
    torch.cuda.empty_cache()
    allocated_mb = torch.cuda.memory_allocated(device_index) / MB
    if allocated_mb > QUIESCENT_TOLERANCE_MB:
        raise TeardownError(
            f"{when}: memory_allocated()={allocated_mb:.1f} MB > {QUIESCENT_TOLERANCE_MB} MB. "
            "empty_cache() does not free memory still referenced by a live Python object; every "
            "measurement after this point is inflated by an unknown amount"
        )
    return allocated_mb


# ---------------------------------------------------------------------------
# Inputs: one real 1600x900 keyframe, and the real prompt set
# ---------------------------------------------------------------------------


def _load_real_keyframe(paths: Paths, override: str | None) -> tuple[np.ndarray, str]:
    """A real CAM_FRONT keyframe at 1600x900, read-only, asserted not assumed."""
    from PIL import Image

    if override:
        path = os.path.realpath(override)
    else:
        cam_dir = os.path.join(paths.samples_dir, "CAM_FRONT")
        if not os.path.isdir(cam_dir):
            raise MeasurementError(f"no CAM_FRONT samples directory at {cam_dir}")
        names = sorted(n for n in os.listdir(cam_dir) if n.lower().endswith(".jpg"))
        if not names:
            raise MeasurementError(f"no JPEGs under {cam_dir}")
        path = os.path.join(cam_dir, names[0])

    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        arr = np.asarray(im, dtype=np.uint8)
    if (w, h) != ORIGINAL_SIZE_PX:
        raise MeasurementError(
            f"{path} is {w}x{h}; this measurement is defined at "
            f"{IMAGE_WIDTH_PX}x{IMAGE_HEIGHT_PX} and a peak measured at another resolution is a "
            "different number (§1.5, §13.1)"
        )
    return arr, path


def _load_prompt_config(taxonomy_path: str, allow_placeholder: bool) -> PromptConfig:
    """The real 23-phrase prompt set from the mapping table (§0.3).

    Dotted nuScenes category names fed to Grounding DINO tokenise to nonsense and
    still return boxes with confidences. `PromptConfig.validate()` rejects them;
    this loader only decides where the phrases come from.
    """
    if os.path.isfile(taxonomy_path):
        import yaml

        with open(taxonomy_path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        mapping = raw.get("prompt_phrase", raw.get("classes", raw)) if isinstance(raw, dict) else raw
        if isinstance(mapping, dict):
            phrases = tuple(str(v) for v in mapping.values())
        elif isinstance(mapping, list):
            phrases = tuple(str(v) for v in mapping)
        else:
            raise MeasurementError(f"{taxonomy_path}: cannot read a phrase list out of this file")
        source = f"{os.path.realpath(taxonomy_path)}"
    else:
        if not allow_placeholder:
            raise MeasurementError(
                f"{taxonomy_path} does not exist. It is a Phase 6 artifact and the measurement needs the "
                "REAL prompt set, because prompt length drives the text-encoder footprint. Re-run with "
                "--allow-placeholder-prompts to measure with an in-script placeholder set of the same "
                "length; that run is recorded quotable=false"
            )
        phrases = PLACEHOLDER_PHRASES
        source = "placeholder-in-script (scripts/measure_vram.py PLACEHOLDER_PHRASES)"

    return PromptConfig(
        phrases=phrases,
        thresholds={},
        default_threshold=0.40,  # inherited, unvalidated on this substrate (§10)
        source=source,
        chunked=False,
    )


def _letterbox(image: np.ndarray, size: int) -> np.ndarray:
    """Aspect-preserving resize into a square canvas with recorded padding.

    §1.5 rule 3: square resizes of 16:9 imagery are forbidden. Embedding roles
    invert nothing, so the damage there is subtler than a stretched box — but a
    role that letterboxes at measurement time and squashes at run time is not the
    same workload, so the harness does the same thing the adapter must.
    """
    from PIL import Image

    h, w = image.shape[:2]
    scale = min(size / w, size / h)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = np.asarray(Image.fromarray(image).resize((new_w, new_h), Image.BILINEAR), dtype=np.uint8)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    top, left = (size - new_h) // 2, (size - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas


def _fallback_boxes(n: int = 32) -> np.ndarray:
    """Deterministic box prompts, used only when no proposals were produced.

    Plausible object footprints spread across the frame — the point is to put the
    right number of prompt encodings and mask decodings on the card, not to be
    right about any object.
    """
    w, h = ORIGINAL_SIZE_PX
    boxes = []
    cols, rows = 8, 4
    for i in range(n):
        cx = (i % cols + 0.5) / cols * w
        cy = (i // cols % rows + 0.5) / rows * h
        bw, bh = 120.0 + 10.0 * (i % 5), 90.0 + 8.0 * (i % 7)
        boxes.append(
            [
                max(0.0, cx - bw / 2),
                max(0.0, cy - bh / 2),
                min(float(w), cx + bw / 2),
                min(float(h), cy + bh / 2),
            ]
        )
    return np.asarray(boxes, dtype=np.float32)


# ---------------------------------------------------------------------------
# Measurement-harness adapters (NOT the Stage 3/4 adapters)
# ---------------------------------------------------------------------------


class _HarnessRole:
    """Shared plumbing: role name, spec, device, load/unload (§7.1's ModelRole)."""

    _role_name = ""

    def __init__(self, spec: CheckpointSpec, device: str, torch) -> None:
        self._spec = spec
        self._device = device
        self._torch = torch
        self._model: Any = None
        self._processor: Any = None

    @property
    def roles(self) -> tuple[str, ...]:
        # Every harness adapter fills exactly one role. A composite provider
        # (SAM 3) would return several here and be created once for all of them.
        return (self._role_name,)

    @property
    def spec(self) -> CheckpointSpec:
        return self._spec

    @property
    def device(self) -> str:
        return self._device

    def load(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def unload(self) -> None:
        """Drop every device reference this object holds.

        Deliberately explicit rather than relying on `del adapter`: a forgotten
        attribute is precisely what the post-teardown assertion catches, and the
        assertion is only useful if teardown is trying.
        """
        self._model = None
        self._processor = None
        gc.collect()
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

    def _hf_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if self._spec.revision:
            kwargs["revision"] = self._spec.revision
        return kwargs


class DinoV2WholeImage(_HarnessRole):
    """`embedding_ood` (Stage 2): CLS token over the whole frame."""

    _role_name = EMBEDDING_OOD

    def __init__(self, spec, device, torch, input_size: int = 518) -> None:
        super().__init__(spec, device, torch)
        self._input_size = int(spec.options.get("input_size_px", input_size))

    def load(self) -> None:
        try:
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:  # pragma: no cover
            raise MeasurementError(f"{self._role_name}: transformers is not installed ({exc})")
        self._processor = AutoImageProcessor.from_pretrained(self._spec.model_id, **self._hf_kwargs())
        self._model = AutoModel.from_pretrained(self._spec.model_id, **self._hf_kwargs()).to(self._device).eval()

    def embed_images(self, images: Sequence[np.ndarray], *, ids: Sequence[str] | None = None) -> EmbeddingBatch:
        torch = self._torch
        batch = np.stack([_letterbox(im, self._input_size) for im in images])
        inputs = self._processor(
            images=list(batch), return_tensors="pt", do_resize=False, do_center_crop=False
        ).to(self._device)
        with torch.inference_mode():
            out = self._model(**inputs)
        cls = out.last_hidden_state[:, 0, :].float().cpu().numpy()
        return EmbeddingBatch(
            vectors=cls,
            semantics="whole_image_cls",
            preprocessing=f"letterbox_{self._input_size}/imagenet_norm/v1",
            model_input_size_px=(self._input_size, self._input_size),
            source_size_px=ORIGINAL_SIZE_PX,
            ids=list(ids) if ids else None,
        )


class DinoV2Crops(_HarnessRole):
    """`reid_embedding` (Stage 7): CLS over object crops — different preprocessing.

    Same checkpoint as `embedding_ood`, deliberately a separate object with a
    separate transform (P1-2). If Stage 7 ever inherits Stage 2's transform, a
    28x28 crop is upsampled ~20x and the similarity measures interpolation.
    """

    _role_name = REID_EMBEDDING

    def __init__(self, spec, device, torch, input_size: int = 224, min_crop_px: int = 28) -> None:
        super().__init__(spec, device, torch)
        self._input_size = int(spec.options.get("crop_input_size_px", input_size))
        self._min_crop_px = int(spec.options.get("min_crop_px", min_crop_px))

    @property
    def min_crop_px(self) -> int:
        return self._min_crop_px

    def load(self) -> None:
        try:
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:  # pragma: no cover
            raise MeasurementError(f"{self._role_name}: transformers is not installed ({exc})")
        self._processor = AutoImageProcessor.from_pretrained(self._spec.model_id, **self._hf_kwargs())
        self._model = AutoModel.from_pretrained(self._spec.model_id, **self._hf_kwargs()).to(self._device).eval()

    def embed_crops(
        self,
        image: np.ndarray,
        boxes_xyxy_px: np.ndarray,
        *,
        masks: np.ndarray | None = None,
        ids: Sequence[str] | None = None,
    ) -> EmbeddingBatch:
        torch = self._torch
        crops = []
        for x1, y1, x2, y2 in np.asarray(boxes_xyxy_px, dtype=np.float32):
            xa, ya = int(max(0, np.floor(x1))), int(max(0, np.floor(y1)))
            xb, yb = int(min(IMAGE_WIDTH_PX, np.ceil(x2))), int(min(IMAGE_HEIGHT_PX, np.ceil(y2)))
            crop = image[ya:yb, xa:xb]
            if crop.size == 0:
                crop = np.zeros((self._min_crop_px, self._min_crop_px, 3), dtype=np.uint8)
            crops.append(_letterbox(crop, self._input_size))
        inputs = self._processor(
            images=crops, return_tensors="pt", do_resize=False, do_center_crop=False
        ).to(self._device)
        with torch.inference_mode():
            out = self._model(**inputs)
        vectors = out.last_hidden_state[:, 0, :].float().cpu().numpy()
        return EmbeddingBatch(
            vectors=vectors,
            semantics="crop_cls",
            preprocessing=f"crop/letterbox_{self._input_size}/imagenet_norm/v1",
            model_input_size_px=(self._input_size, self._input_size),
            source_size_px=ORIGINAL_SIZE_PX,
            ids=list(ids) if ids else None,
        )


class GroundingDinoProposals(_HarnessRole):
    """`proposal_2d` (Stage 3): the full prompt set, one 1600x900 frame.

    The phrase spans produced here are **character** spans located in the prompt
    string. That is not the token-span bookkeeping Stage 3 needs (§5.4) — it is
    enough to satisfy the contract's shape for a VRAM measurement and nothing
    more, which is why this class is not importable by pipeline code.
    """

    _role_name = PROPOSAL_2D

    def __init__(self, spec, device, torch, prompt: PromptConfig) -> None:
        super().__init__(spec, device, torch)
        self._prompt = prompt
        # Grounding DINO's caption convention: lowercase phrases, ". "-separated,
        # trailing period. Chunking is forbidden (§5.4), so the whole set goes in.
        self._caption = ". ".join(p.strip().lower().rstrip(".") for p in prompt.phrases) + "."
        self._last_input_size: tuple[int, int] | None = None

    @property
    def returns_masks(self) -> bool:
        return False

    def load(self) -> None:
        try:
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError as exc:  # pragma: no cover
            raise MeasurementError(f"{self._role_name}: transformers is not installed ({exc})")
        self._processor = AutoProcessor.from_pretrained(self._spec.model_id, **self._hf_kwargs())
        self._model = (
            AutoModelForZeroShotObjectDetection.from_pretrained(self._spec.model_id, **self._hf_kwargs())
            .to(self._device)
            .eval()
        )

    def propose(self, image: np.ndarray, prompt: PromptConfig, *, channel: str = "") -> Proposals:
        torch = self._torch
        from PIL import Image as PILImage

        pil = PILImage.fromarray(image)
        inputs = self._processor(images=pil, text=self._caption, return_tensors="pt").to(self._device)
        # The processor's DETR-style resize preserves aspect (shortest side), so
        # nothing is stretched; the actual tensor size is recorded, not assumed.
        pv = inputs["pixel_values"]
        self._last_input_size = (int(pv.shape[-1]), int(pv.shape[-2]))
        with torch.inference_mode():
            outputs = self._model(**inputs)

        results = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=float(prompt.default_threshold),
            text_threshold=float(self._spec.options.get("text_threshold", 0.25)),
            target_sizes=[(IMAGE_HEIGHT_PX, IMAGE_WIDTH_PX)],
        )[0]

        boxes = np.asarray(results["boxes"].float().cpu().numpy(), dtype=np.float32).reshape(-1, 4)
        scores = np.asarray(results["scores"].float().cpu().numpy(), dtype=np.float32).reshape(-1)
        labels = list(results.get("text_labels", results.get("labels", [])))
        labels = [str(x) for x in labels][: len(boxes)]
        while len(labels) < len(boxes):
            labels.append("")

        # Clip into the frame in the adapter, never in stage code (§1.5 rule 2).
        if len(boxes):
            boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0.0, float(IMAGE_WIDTH_PX))
            boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0.0, float(IMAGE_HEIGHT_PX))
            keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            boxes, scores = boxes[keep], scores[keep]
            labels = [l for l, k in zip(labels, keep) if k]

        spans: list[tuple[int, int]] = []
        for label in labels:
            start = self._caption.find(label.strip().lower()) if label else -1
            spans.append((start, start + len(label)) if start >= 0 and label else (0, len(self._caption)))

        return Proposals(
            boxes_xyxy_px=boxes,
            scores=np.clip(scores, 0.0, 1.0),
            class_names=labels,
            phrase_spans=spans,
            prompt_config=prompt,
            channel=channel,
            image_size_px=ORIGINAL_SIZE_PX,
            resize_policy="shortest_side",
            model_input_size_px=self._last_input_size,
        )


class MobileSamMasks(_HarnessRole):
    """`mask_2d` (Stage 4): box-prompted masks at 1600x900.

    Accepts `state` and `window` and ignores them, returning `state=None`,
    `propagated=False` — the non-temporal contract of §7.1 fix 1. MobileSAM has
    no propagation; that is a **capability** gap, stated, not a tuning gap (§4).
    """

    _role_name = MASK_2D

    def __init__(self, spec, device, torch) -> None:
        super().__init__(spec, device, torch)
        self._predictor: Any = None

    @property
    def supports_temporal(self) -> bool:
        return False

    def load(self) -> None:
        checkpoint = self._spec.options.get("checkpoint_path")
        model_type = str(self._spec.options.get("model_type", "vit_t"))
        registry = None
        predictor_cls = None
        try:
            from mobile_sam import SamPredictor, sam_model_registry  # type: ignore

            registry, predictor_cls = sam_model_registry, SamPredictor
        except ImportError:
            try:
                from segment_anything import SamPredictor, sam_model_registry  # type: ignore

                registry, predictor_cls = sam_model_registry, SamPredictor
            except ImportError as exc:
                raise MeasurementError(
                    f"{self._role_name}: neither mobile_sam nor segment_anything is installed ({exc}); "
                    "MobileSAM is the committed pilot choice (§5.5) and cannot be measured without it"
                )
        if not checkpoint or not os.path.isfile(str(checkpoint)):
            raise MeasurementError(
                f"{self._role_name}: roles.mask_2d.options.checkpoint_path is missing or not a file "
                f"({checkpoint!r}); MobileSAM ships weights as a file, not a hub id"
            )
        sam = registry[model_type](checkpoint=str(checkpoint))
        sam.to(self._device).eval()
        self._model = sam
        self._predictor = predictor_cls(sam)

    def unload(self) -> None:
        self._predictor = None
        super().unload()

    def segment(
        self,
        images: Sequence[np.ndarray],
        boxes_xyxy_px: np.ndarray,
        *,
        state: Any | None = None,
        window: TemporalWindow = TemporalWindow(),
        channel: str = "",
    ) -> MaskResult:
        torch = self._torch
        image = np.asarray(images[0])
        boxes = np.asarray(boxes_xyxy_px, dtype=np.float32).reshape(-1, 4)
        with torch.inference_mode():
            self._predictor.set_image(image)
            if len(boxes) == 0:
                masks_np = np.zeros((0, IMAGE_HEIGHT_PX, IMAGE_WIDTH_PX), dtype=bool)
            else:
                # The predictor owns the forward transform into its 1024-longest-
                # side space and the inverse back to the source resolution; masks
                # come back at 1600x900 and that is asserted below (§1.5 rule 4).
                box_t = self._predictor.transform.apply_boxes_torch(
                    torch.as_tensor(boxes, device=self._device), image.shape[:2]
                )
                masks, _, _ = self._predictor.predict_torch(
                    point_coords=None, point_labels=None, boxes=box_t, multimask_output=False
                )
                masks_np = masks[:, 0].detach().cpu().numpy().astype(bool)
        self._predictor.reset_image()
        return MaskResult(
            masks=masks_np,
            state=None,        # no propagation: MobileSAM carries nothing forward
            window=window,
            propagated=False,
        )


# ---------------------------------------------------------------------------
# Per-role measurement
# ---------------------------------------------------------------------------


@dataclass
class RoleMeasurement:
    role: str
    status: str                     # "measured" | "unavailable"
    model_id: str
    revision: str | None
    reserved_peak_mb: float = 0.0
    free_delta_mb: float = 0.0
    occupancy_mb: float = 0.0       # max(reserved peak, free-delta), context excluded
    total_with_context_mb: float = 0.0
    allocated_after_load_mb: float = 0.0
    allocated_after_teardown_mb: float = 0.0
    estimate_mb: float | None = None
    load_s: float = 0.0
    forward_s: float = 0.0
    within_ceiling: bool = False
    detail: dict[str, Any] | None = None
    error: str = ""

    def as_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["detail"] = self.detail or {}
        return d


def _build_adapter(role: str, spec: CheckpointSpec, device: str, torch, prompt: PromptConfig):
    if role == EMBEDDING_OOD:
        return DinoV2WholeImage(spec, device, torch)
    if role == REID_EMBEDDING:
        return DinoV2Crops(spec, device, torch)
    if role == PROPOSAL_2D:
        return GroundingDinoProposals(spec, device, torch, prompt)
    if role == MASK_2D:
        return MobileSamMasks(spec, device, torch)
    raise MeasurementError(f"no measurement harness for role {role!r}")


def _exercise(role: str, adapter, image: np.ndarray, prompt: PromptConfig, boxes: np.ndarray) -> Any:
    """One real forward pass at 1600x900, per role, with contract validation."""
    if role == EMBEDDING_OOD:
        out = adapter.embed_images([image], ids=["measure"])
        return out.assert_valid(prefix="embedding_ood: ")
    if role == PROPOSAL_2D:
        out = adapter.propose(image, prompt, channel="CAM_FRONT")
        return out.assert_valid(prefix="proposal_2d: ")
    if role == MASK_2D:
        out = adapter.segment([image], boxes, state=None, window=TemporalWindow(frames=1))
        return out.assert_valid(n_boxes=len(boxes), prefix="mask_2d: ")
    if role == REID_EMBEDDING:
        out = adapter.embed_crops(image, boxes, ids=[f"box{i}" for i in range(len(boxes))])
        return out.assert_valid(prefix="reid_embedding: ")
    raise MeasurementError(f"no exercise defined for role {role!r}")


def measure_role(
    torch,
    role: str,
    spec: CheckpointSpec,
    *,
    device_index: int,
    image: np.ndarray,
    prompt: PromptConfig,
    boxes: np.ndarray,
    ceiling_mb: float,
    context_mb: float,
    iters: int,
) -> tuple[RoleMeasurement, Any]:
    """Load, exercise, measure, tear down. One role, resident alone.

    Returns the measurement and the last output, so a later role can reuse real
    boxes rather than synthetic ones. The output is dropped by the caller before
    the next role loads.
    """
    device = f"cuda:{device_index}"
    m = RoleMeasurement(
        role=role,
        status="unavailable",
        model_id=spec.model_id,
        revision=spec.revision,
        estimate_mb=spec.vram_estimate_mb,
    )

    _assert_quiescent(torch, f"before loading role {role}", device_index)
    torch.cuda.reset_peak_memory_stats(device_index)
    free_before, total = torch.cuda.mem_get_info(device_index)

    adapter = _build_adapter(role, spec, device, torch, prompt)
    assert_role_conformance(adapter, role)

    t0 = time.perf_counter()
    adapter.load()
    torch.cuda.synchronize(device_index)
    m.load_s = time.perf_counter() - t0
    free_after_load, _ = torch.cuda.mem_get_info(device_index)
    m.allocated_after_load_mb = torch.cuda.memory_allocated(device_index) / MB

    output: Any = None
    free_low = free_after_load
    try:
        # One warm-up (cuBLAS/cuDNN workspaces, autotune, lazy module init) then
        # the measured passes. The peak counters are NOT reset after warm-up:
        # workspace allocations are part of what the card must hold.
        for _ in range(max(1, iters)):
            t_fwd = time.perf_counter()
            output = _exercise(role, adapter, image, prompt, boxes)
            torch.cuda.synchronize(device_index)
            m.forward_s = time.perf_counter() - t_fwd  # the last pass: warm, not first-call
            free_now, _ = torch.cuda.mem_get_info(device_index)
            free_low = min(free_low, free_now)
    except torch.cuda.OutOfMemoryError as exc:
        # OOM is a hard stop with stage/role/resolution logged (§1.9). It is never
        # retried at a lower resolution: that changes the output distribution.
        m.error = (
            f"CUDA OOM during {role} forward at {IMAGE_WIDTH_PX}x{IMAGE_HEIGHT_PX} with "
            f"{len(prompt.phrases)} prompt phrases and {len(boxes)} boxes: {exc}"
        )
        adapter.unload()
        del adapter
        raise CeilingExceeded(m.error) from exc

    m.reserved_peak_mb = torch.cuda.max_memory_reserved(device_index) / MB
    m.free_delta_mb = (free_before - free_low) / MB
    # The two measurements disagree by exactly what the allocator cannot see:
    # cuBLAS/cuDNN workspaces, kernel modules, and anything another process did
    # meanwhile. Believe the larger; a budget built on the smaller is not a budget.
    m.occupancy_mb = max(m.reserved_peak_mb, m.free_delta_mb)
    m.total_with_context_mb = m.occupancy_mb + context_mb
    m.within_ceiling = m.total_with_context_mb <= ceiling_mb
    m.status = "measured"
    m.detail = {
        "device_free_before_mb": free_before / MB,
        "device_free_low_mb": free_low / MB,
        "device_total_mb": total / MB,
        "iters": max(1, iters),
        "source_size_px": list(ORIGINAL_SIZE_PX),
        "n_prompt_phrases": len(prompt.phrases),
        "n_boxes": int(len(boxes)),
        "inference_mode": True,
    }
    if role == PROPOSAL_2D and isinstance(output, Proposals):
        m.detail["model_input_size_px"] = list(output.model_input_size_px or ())
        m.detail["n_proposals"] = len(output)
        m.detail["resize_policy"] = output.resize_policy
    elif isinstance(output, EmbeddingBatch):
        m.detail["model_input_size_px"] = list(output.model_input_size_px)
        m.detail["embedding_dim"] = output.dim
        m.detail["semantics"] = output.semantics
        m.detail["preprocessing"] = output.preprocessing
    elif isinstance(output, MaskResult):
        m.detail["n_masks"] = len(output)
        m.detail["propagated"] = output.propagated
        m.detail["supports_temporal"] = adapter.supports_temporal

    # --- teardown, then the assertion that makes the next role's number mean
    # something (§7.1). The output is deliberately kept alive by the caller only
    # AFTER it has been converted to numpy inside the adapters.
    adapter.unload()
    del adapter
    m.allocated_after_teardown_mb = _assert_quiescent(torch, f"after tearing down role {role}", device_index)
    return m, output


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _banner() -> str:
    return (
        "This demonstrates pipeline plumbing only. Label quality is not evidence of anything; "
        "models are deliberately under-tier; the substrate is nuScenes v1.0-mini, not Dhaka."
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-config", default="configs/models_pilot.yaml")
    ap.add_argument("--paths", default=os.environ.get("DHAKASCENES_PATHS_CONFIG", "configs/paths.yaml"))
    ap.add_argument("--taxonomy", default="configs/taxonomy_pilot_nuscenes.yaml")
    ap.add_argument("--roles", default=",".join(ROLES), help="comma-separated subset of the four roles")
    ap.add_argument("--image", default=None, help="override the 1600x900 keyframe (must still be 1600x900)")
    ap.add_argument("--iters", type=int, default=3, help="forward passes per role, including warm-up")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--out", default=None, help="default: <work_root>/vram/measure_vram.json")
    ap.add_argument("--allow-placeholder-prompts", action="store_true")
    ap.add_argument(
        "--allow-nontarget-device",
        action="store_true",
        help="measure on a card whose total VRAM differs from vram_budget.total_device_mb",
    )
    args = ap.parse_args()

    # --- config and substrate contracts, before anything touches the GPU ---
    # A malformed YAML is a config contract violation like any other and exits 2;
    # it does not get to surface as a bare traceback that reads like a crash.
    import yaml

    try:
        paths = load_paths(args.paths)
    except (PathValidationError, OSError, yaml.YAMLError) as exc:
        print(f"path contract: {exc}", file=sys.stderr)
        return 2
    try:
        config: ModelConfig = load_model_config(args.model_config)
    except (RoleContractError, OSError, yaml.YAMLError, ValueError, TypeError) as exc:
        print(f"model config: {exc}", file=sys.stderr)
        return 2

    requested = [r.strip() for r in args.roles.split(",") if r.strip()]
    unknown = [r for r in requested if r not in ROLES]
    if unknown:
        print(f"unknown role(s): {unknown}; roles are {ROLES}", file=sys.stderr)
        return 2

    try:
        prompt = _load_prompt_config(args.taxonomy, args.allow_placeholder_prompts)
        image, image_path = _load_real_keyframe(paths, args.image)
    except MeasurementError as exc:
        print(f"input contract: {exc}", file=sys.stderr)
        return 2
    prompt_errors = prompt.validate(prefix="prompt: ")
    if prompt_errors:
        print("\n".join(prompt_errors), file=sys.stderr)
        return 2

    try:
        import torch  # noqa: E402  (imported only after the allocator config is pinned)
    except ImportError as exc:
        print(f"measure_vram: torch is not installed ({exc}); Phase 5b needs the real card", file=sys.stderr)
        return 4

    if not torch.cuda.is_available():
        print(
            "measure_vram: no CUDA device. This is Phase 5b — the measured half. The paper half is "
            "scripts/check_vram_paper.py and it does not need a card.",
            file=sys.stderr,
        )
        return 4

    # §1.9: one global seed, and no autotuning, so a re-measurement is comparable.
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.benchmark = False

    baseline = _establish_baseline(torch, args.device)
    context_mb = baseline.context_mb if baseline.context_mb is not None else baseline.post_context_used_mb
    ceiling_mb = config.budget.hard_ceiling_mb

    on_target_device = abs(baseline.total_mb - float(config.budget.total_device_mb)) <= 128.0
    if not on_target_device and not args.allow_nontarget_device:
        print(
            f"measure_vram: this card reports {baseline.total_mb:.0f} MB total but vram_budget."
            f"total_device_mb is {config.budget.total_device_mb:.0f} MB. A peak measured on a different "
            "card is not a measurement of the budgeted card — allocator behaviour, the display-server "
            "share and the free-delta baseline all differ. Re-run on the target card, or pass "
            "--allow-nontarget-device to record the run with quotable=false.",
            file=sys.stderr,
        )
        return 2

    measurements: list[RoleMeasurement] = []
    boxes = _fallback_boxes()
    box_source = "deterministic_fallback_grid"
    exit_code = 0

    # Roles are measured in this order so that mask_2d and reid_embedding get
    # REAL boxes from proposal_2d rather than the synthetic grid. Only one role
    # is resident at any moment; the ordering buys fidelity, not memory.
    order = [r for r in (PROPOSAL_2D, MASK_2D, EMBEDDING_OOD, REID_EMBEDDING) if r in requested]

    for role in order:
        spec = config.roles[role]
        try:
            m, output = measure_role(
                torch,
                role,
                spec,
                device_index=args.device,
                image=image,
                prompt=prompt,
                boxes=boxes,
                ceiling_mb=ceiling_mb,
                context_mb=context_mb,
                iters=args.iters,
            )
        except RoleContractError as exc:
            # The harness adapter does not satisfy the role it was built for.
            # That is a defect in this file, not a property of the card, and
            # every later role's number would be measured by broken scaffolding.
            print(f"[{role}] CONTRACT VIOLATION: {exc}", file=sys.stderr)
            measurements.append(
                RoleMeasurement(
                    role=role,
                    status="contract_violation",
                    model_id=spec.model_id,
                    revision=spec.revision,
                    error=str(exc),
                )
            )
            exit_code = 2
            break
        except MeasurementError as exc:
            measurements.append(
                RoleMeasurement(
                    role=role,
                    status="unavailable",
                    model_id=spec.model_id,
                    revision=spec.revision,
                    estimate_mb=spec.vram_estimate_mb,
                    error=str(exc),
                )
            )
            print(f"[{role}] UNAVAILABLE: {exc}", file=sys.stderr)
            exit_code = max(exit_code, 4)
            continue
        except CeilingExceeded as exc:
            measurements.append(
                RoleMeasurement(
                    role=role,
                    status="oom",
                    model_id=spec.model_id,
                    revision=spec.revision,
                    estimate_mb=spec.vram_estimate_mb,
                    error=str(exc),
                )
            )
            print(f"[{role}] HARD STOP: {exc}", file=sys.stderr)
            exit_code = 3
            continue
        except TeardownError as exc:
            print(f"[{role}] TEARDOWN FAILED: {exc}", file=sys.stderr)
            measurements.append(
                RoleMeasurement(
                    role=role,
                    status="teardown_failed",
                    model_id=spec.model_id,
                    revision=spec.revision,
                    error=str(exc),
                )
            )
            return 5

        measurements.append(m)
        if role == PROPOSAL_2D and isinstance(output, Proposals) and len(output):
            boxes = np.asarray(output.boxes_xyxy_px[:32], dtype=np.float32)
            box_source = f"proposal_2d/{spec.model_id}"
        del output
        gc.collect()

        verdict = "OK" if m.within_ceiling else "OVER CEILING"
        print(
            f"[{role}] reserved_peak={m.reserved_peak_mb:8.1f} MB  free_delta={m.free_delta_mb:8.1f} MB  "
            f"+context={context_mb:6.1f} MB  total={m.total_with_context_mb:8.1f} MB  "
            f"ceiling={ceiling_mb:.1f} MB  {verdict}"
        )
        if not m.within_ceiling:
            # §7.2's stated consequence. The mitigation is a letterboxed resize
            # recorded in every Stage 3 record — not a square resize, not silent
            # prompt chunking, and not a quiet retry here.
            print(
                f"[{role}] HARD STOP: {m.total_with_context_mb:.1f} MB > hard_ceiling_mb {ceiling_mb:.1f} MB "
                f"at {IMAGE_WIDTH_PX}x{IMAGE_HEIGHT_PX} with {len(prompt.phrases)} prompt phrases",
                file=sys.stderr,
            )
            exit_code = 3

    measured = [m for m in measurements if m.status == "measured"]
    quotable = (
        exit_code == 0
        and on_target_device
        and not prompt.source.startswith("placeholder")
        and bool(measured)
        and all(m.revision for m in measured)
    )

    payload: dict[str, Any] = {
        "artifact": "measure_vram/v1",
        "phase": "5b",
        "banner": _banner(),
        "quotable": quotable,
        "quotable_reasons": {
            "measured_on_target_device": on_target_device,
            "real_prompt_set": not prompt.source.startswith("placeholder"),
            "all_revisions_pinned": bool(measured) and all(m.revision for m in measured),
            "all_roles_within_ceiling": exit_code == 0,
        },
        "adapter_kind": "measurement_harness",
        "adapter_note": (
            "Peaks measured with the harness adapters in scripts/measure_vram.py, not with the Phase 6 "
            "Stage 3/4 adapters. Re-measure when those land."
        ),
        "alloc_conf": _PINNED_ALLOC_CONF,
        "measurement_method": (
            "max(torch.cuda.max_memory_reserved(), mem_get_info() free-delta) + measured CUDA context. "
            "max_memory_allocated() is NOT used: it excludes the context, reserved-but-unallocated "
            "blocks, cuBLAS/cuDNN workspaces and other processes."
        ),
        "seed": config.seed,
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "tf32": {
            "matmul": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn": bool(torch.backends.cudnn.allow_tf32),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "platform": platform.platform(),
        },
        "device": {
            "index": baseline.index,
            "name": baseline.name,
            "capability": baseline.capability,
            "total_mb": baseline.total_mb,
            "pre_context_used_mb": baseline.pre_context_used_mb,
            "post_context_used_mb": baseline.post_context_used_mb,
            "context_mb": baseline.context_mb,
            "pre_context_source": "nvidia-smi" if baseline.pre_context_used_mb is not None else "unavailable",
        },
        "budget": config.budget.as_dict(),
        "substrate": {
            "dataroot": paths.dataroot,
            "version": paths.version,
            "metadata_fingerprint": metadata_fingerprint(paths),
            "image_path": image_path,
            "image_size_px": list(ORIGINAL_SIZE_PX),
        },
        "prompt": prompt.as_dict(),
        "box_source": box_source,
        "model_config": config.as_dict(),
        "roles": {m.role: m.as_dict() for m in measurements},
        # `verified:` is never flipped from inside this script: models_pilot.yaml
        # is a hand-maintained, comment-carrying config and a machine rewrite
        # would drop the provenance notes that make it worth reading. The exact
        # values to set are printed and recorded instead.
        "verified_flags_to_set": {
            m.role: bool(quotable and m.within_ceiling and m.revision and config.roles[m.role].sha256)
            for m in measured
        },
        "exit_code": exit_code,
    }

    out_path = args.out or os.path.join(paths.work_root, "vram", "measure_vram.json")
    out_path = assert_dataroot_read_only(paths, out_path)
    write_json_atomic(out_path, payload)
    print(f"\nwrote {out_path}")

    if not quotable:
        print(
            "quotable=false: this run may be RUN against, never QUOTED. "
            f"reasons={json.dumps(payload['quotable_reasons'])}",
            file=sys.stderr,
        )
    else:
        print(
            "Set verified: true in "
            f"{os.path.relpath(config.config_path)} for: "
            + ", ".join(f"{r} (vram_measured_mb: {payload['roles'][r]['total_with_context_mb']:.0f})" for r in
                        payload["verified_flags_to_set"] if payload["verified_flags_to_set"][r])
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
