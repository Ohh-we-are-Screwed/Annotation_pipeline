#!/usr/bin/env python3
"""Stage 4 — box-prompted masks and cross-camera IoA-NMS (§5.5, Phase 6).

Box-prompted masks over Stage 3's proposals. The default provider is now
**SAM 2.1-hiera-large** via transformers (DECISIONS C19, human decision
2026-08-13): on the 24 GB box the best locally runnable tier governs defaults,
not the 4 GB pilot contract. facebook/sam3's tracker is a selectable provider
("sam3_tracker", SA-V J&F 84.4 vs SAM2.1-L's 78.4), pending the gated-repo
license grant. MobileSAM (~10 M params, ~40 MB) stays registered as the
pilot-tier fallback — at 4 GB the §5.5 commitment over SAM-ViT-B (~91 M, the
~9x parameter difference, §13.1) is unchanged: C1 is not abandoned, it just no
longer picks the default on this machine.

Three things this stage is responsible for, and the silent failure each prevents:

  1. **Masks come back at 1600x900, asserted rather than assumed** (§1.5 rule 4).
     MobileSAM prompts live in a 1024-longest-side transformed space and its
     decoder emits 1024-space logits; if the inverse transform is skipped, Stage
     5 indexes a mask with pixel coordinates it does not own, and the painted
     points are wrong by a scale factor everywhere — no crash, no empty output.
  2. **One mask per box, in the same order.** The strongest test rev 1 had.
     Silently permuted masks assign every object its neighbour's class.
  3. **IoA-NMS > 0.5 across overlapping cameras** (§7.3.4, restored in §1.5
     rule 6; rev 1 dropped it without comment, M-2). Its absence is what makes
     duplicate assignment reachable: the ring cameras overlap, so one car is
     proposed twice and lifted twice, and the two boxes disagree slightly.

**Where the IoA is computed, and why it is not pixels.** Two boxes in two
different cameras live in two different image planes. Their pixel coordinates
are not comparable, and a pixel-space IoA between them is a number the code will
happily produce and that means nothing — it suppresses whatever happens to share
image coordinates, which for a ring rig is "objects at the same bearing relative
to two different optical axes", i.e. arbitrary. So the overlap is computed where
the two cameras DO share a frame: each mask's tight bounding box is back-
projected through K^-1 and the camera->ego rotation into an **(azimuth,
elevation) footprint in ego frame** (§1.1), and IoA is area-over-area there.
Within one camera the two formulations coincide; across cameras only this one
exists.

The footprint is computed from the MASK's tight box, not from Stage 3's
proposal box: SAM routinely tightens a loose proposal, and a tighter footprint
is a strictly better overlap test. That costs one segmentation for a mask that
may then be suppressed, which is the trade recorded in the manifest.

**Capability gap, stated plainly (§4) — pilot tier only, since C19.** MobileSAM
has no cross-frame propagation; on that provider masking is independent per
frame and Stage 7 looks worse for a structural reason, not a tuning one. The
SAM 2.1 / SAM 3 adapters DO support video propagation (`supports_temporal`),
which is exactly what §7.1 fix 1 bought by keeping `state` and `window` in the
signature: the swap was a provider change, not a Stage 4 rewrite. The driver
still segments per frame (`window_frames=1`); the video-session propagation
path exists but is best-effort until exercised end to end.

**SAM 3.1 Object Multiplex is a selectable provider since C26** ("sam31_multiplex",
`--model-id facebook/sam3.1`), reached through Meta's own `sam3` package rather than
transformers: no transformers integration for SAM 3.1 exists, and facebook/sam3.1's
config.json is a stale SAM 3 copy that would silently load SAM 3's classes. It is the
one provider here that also implements `MaskVideoTracker` (C27, Stage 3b box recovery).

    python3 -m pipeline.stage4_masks.masks [--paths configs/paths.yaml]

Exit codes:
    0  every proposal produced a mask under contract
    1  ran, but at least one image was degraded (empty masks, or all suppressed),
       or the model could not be RELEASED after a complete run — the marker
       carries the causes, per-scene and run-level alike (C16)
    2  upstream contract broken, or the model is unavailable; nothing written
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field, replace
from typing import Any, Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pipeline.common.conventions import CAMERA, EGO, Transform  # noqa: E402
from pipeline.common.model_interfaces import (  # noqa: E402
    MASK_2D,
    CheckpointSpec,
    Mask2D,
    MaskResult,
    PER_FRAME_WINDOW,
    RoleContractError,
    TemporalWindow,
    apply_vram_cap,
    register,
)
from pipeline.common.paths import (  # noqa: E402
    PathValidationError,
    Paths,
    assert_dataroot_read_only,
    load_paths,
)
from pipeline.common.schemas import IMAGE_HEIGHT_PX, IMAGE_WIDTH_PX  # noqa: E402
from pipeline.common.manifest import (  # noqa: E402
    UpstreamRefusal,
    clear_markers,
    require_upstream,
    write_json_atomic,
    write_jsonl_atomic,
    write_marker,
)
from pipeline.stage0_data_probe.probe import Substrate  # noqa: E402
from pipeline.stage3_proposals.proposals import ModelUnavailable  # noqa: E402

STAGE = "stage4_masks"
STAGE_SPEC = "dhakascenes-pilot/stage4_masks/v1"

EXIT_OK = 0
EXIT_DEGRADED = 1  # ran, but an image produced no usable mask, or the release failed
EXIT_REFUSED = 2  # upstream contract broken or model unavailable; nothing was written

# Ring-camera adjacency is not assumed from the names: two cameras are treated as
# overlapping when their angular footprints actually overlap, which is what the
# IoA test measures anyway. This constant only bounds the pair search.
_TWO_PI = 2.0 * math.pi


class MaskContractError(RuntimeError):
    """A mask violated the Stage 4 output contract (resolution, count, or order)."""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MaskConfig:
    """Stage 4 tunables. None of these may appear as a literal in the code below."""

    # --- model (C19, human decision 2026-08-13) ---
    model_id: str = "facebook/sam3"
    provider: str = ""           # "" == infer from model_id; mobile_sam | sam2_video | sam3_tracker
    model_type: str = "vit_t"    # MobileSAM-only
    checkpoint_path: str = ""    # MobileSAM-only: a weights file, never a hub id
    revision: str | None = None
    device: str = "cuda"

    # --- IoA-NMS across overlapping cameras (§7.3.4, §1.5 rule 6) ---
    ioa_threshold: float = 0.5
    same_class_only: bool = True

    # --- mask hygiene ---
    min_mask_px: int = 16
    multimask_output: bool = False
    sam31_max_objects: int = 128

    # --- 2D contract (§1.5) ---
    image_width_px: int = IMAGE_WIDTH_PX
    image_height_px: int = IMAGE_HEIGHT_PX

    # --- temporal (§7.1 fix 1): present, and unused on this provider ---
    window_frames: int = 1
    reinit_at_block_boundary: bool = True

    # --- storage ---
    bit_pack_masks: bool = True

    # --- upstream gate (C16) ---
    accept_degraded_upstream: bool = False

    # --- determinism (§1.9) ---
    global_seed: int = 20260812

    provenance: dict = field(
        default_factory=lambda: {
            "model_id": (
                "C19 (human decision, 2026-08-13): facebook/sam3 (tracker path) is the default "
                "mask_2d on the 24 GB box — gated-repo license granted same day, box-prompt smoke "
                "passed (SA-V J&F 84.4 vs SAM2.1-L 78.4); facebook/sam2.1-hiera-large is the "
                "ungated alternate; MobileSAM stays the pilot-tier fallback (§5.5, C1)"
            ),
            "provider": (
                "C26 (supersedes C19's rule): '' infers from model_id — EXACTLY 'facebook/sam3' "
                "-> sam3_tracker; prefix 'facebook/sam3.1' -> sam31_multiplex; 'mobile_sam' -> "
                "mobile_sam; 'facebook/sam2' -> sam2_video; anything else is REFUSED, it no "
                "longer falls through to sam2_video; an explicit value wins"
            ),
            "ioa_threshold": "comprehensive.md §7.3.4, > 0.5; spec value, unvalidated on this substrate",
            "same_class_only": (
                "suppress only same-class duplicates: a pedestrian and the bus behind it share a "
                "bearing sector and are not duplicates of each other"
            ),
            "min_mask_px": "arbitrary; a mask below this cannot carry 5 LiDAR returns anyway",
            "multimask_output": "single mask per box: the box IS the disambiguation (§7.3.4)",
            "sam31_max_objects": (
                "C26: SAM 3.1 multiplex object budget; must exceed the densest per-camera "
                "proposal count (yolo11x averages ~5/image, max cap 300); VRAM at 128 measured "
                "7456 MiB peak on the 4090 smoke"
            ),
            "window_frames": (
                "1 == per-frame, the validated default; >1 enables the best-effort SAM 2/3 "
                "video-propagation path (C19). MobileSAM ignores it (§4)"
            ),
            "bit_pack_masks": "np.packbits: a 1600x900 bool mask is 1.4 MB raw, 180 kB packed",
            "accept_degraded_upstream": "C16 — consuming a DEGRADED (complete, quality-flagged) "
            "Stage 3 output is an explicit recorded decision, never a default",
            "global_seed": "§1.9, one global seed, recorded",
        }
    )

    def as_dict(self) -> dict:
        return dict(self.__dict__)


# ---------------------------------------------------------------------------
# The shared frame: (azimuth, elevation) footprints in ego frame (§1.1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AngularBox:
    """A 2D box's footprint as an ego-frame bearing/elevation rectangle.

    Azimuth is measured about +z from +x, ISO 8855 (§1.1), and is stored
    UNWRAPPED: `az_hi` may exceed pi so that a footprint straddling the rear
    ( CAM_BACK, azimuth ~ pi ) stays a single interval instead of splitting into
    two and silently comparing as disjoint against everything.
    """

    az_lo: float
    az_hi: float
    el_lo: float
    el_hi: float

    @property
    def area(self) -> float:
        return max(0.0, self.az_hi - self.az_lo) * max(0.0, self.el_hi - self.el_lo)

    def as_dict(self) -> dict:
        return {
            "az_lo_rad": round(self.az_lo, 6),
            "az_hi_rad": round(self.az_hi, 6),
            "el_lo_rad": round(self.el_lo, 6),
            "el_hi_rad": round(self.el_hi, 6),
        }


def _box_sample_points(box_xyxy_px: Sequence[float], n_per_edge: int = 3) -> np.ndarray:
    """Corners plus edge samples.

    Corners alone underestimate the footprint of a wide box near the image edge,
    where the projection is most non-linear — and the edges are exactly where two
    ring cameras overlap.
    """
    x1, y1, x2, y2 = (float(v) for v in box_xyxy_px)
    xs = np.linspace(x1, x2, n_per_edge)
    ys = np.linspace(y1, y2, n_per_edge)
    grid_x, grid_y = np.meshgrid(xs, ys)
    return np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)


def angular_footprint(
    box_xyxy_px: Sequence[float],
    intrinsic_k: np.ndarray,
    camera_to_ego: Transform,
) -> AngularBox:
    """Back-project a 2D box into an ego-frame (azimuth, elevation) rectangle.

    `camera_to_ego` is nuScenes' `calibrated_sensor` in the direction it is
    stored (camera -> ego), and it is used in that direction here: the ray
    direction is rotated INTO ego, which is the opposite of the projection chain's
    need for the inverse (§1.3). Getting this backwards yields a footprint
    reflected about the optical axis — still a valid rectangle, still comparable,
    and wrong for every camera except the one pointing along +x.
    """
    K = np.asarray(intrinsic_k, dtype=np.float64)
    if K.shape != (3, 3):
        raise ValueError(f"intrinsic must be (3, 3), got {K.shape}")
    if (camera_to_ego.source_frame, camera_to_ego.parent_frame) != (CAMERA, EGO):
        raise RoleContractError(
            f"camera_to_ego must be {CAMERA} -> {EGO}, got "
            f"{camera_to_ego.source_frame} -> {camera_to_ego.parent_frame}"
        )

    uv = _box_sample_points(box_xyxy_px)
    homogeneous = np.concatenate([uv, np.ones((len(uv), 1))], axis=1)  # (M, 3)
    rays_camera = homogeneous @ np.linalg.inv(K).T  # unit-depth rays in camera frame
    rays_ego = rays_camera @ camera_to_ego.matrix()[:3, :3].T

    azimuth = np.arctan2(rays_ego[:, 1], rays_ego[:, 0])
    elevation = np.arctan2(rays_ego[:, 2], np.hypot(rays_ego[:, 0], rays_ego[:, 1]))

    # Unwrap around the first sample: a box spans far less than pi, so any jump
    # bigger than pi is the +-pi seam and not the object's real extent.
    reference = float(azimuth[0])
    unwrapped = reference + np.remainder(azimuth - reference + math.pi, _TWO_PI) - math.pi
    return AngularBox(
        az_lo=float(unwrapped.min()),
        az_hi=float(unwrapped.max()),
        el_lo=float(elevation.min()),
        el_hi=float(elevation.max()),
    )


def angular_ioa(a: AngularBox, b: AngularBox) -> float:
    """Intersection-over-Area of `a`: how much of `a` lies inside `b`.

    IoA, not IoU, because the two cameras see the same object from different
    angles at different truncations: the near-edge camera may hold a third of the
    car and the other camera all of it. IoU between those is ~0.33 and would fail
    any sane threshold; IoA of the truncated one against the whole one is ~1.0,
    which is the actual question — "is this box already accounted for?".
    """
    if a.area <= 0.0:
        return 0.0
    # Unwrap b onto a's branch before comparing; the two footprints may have been
    # unwrapped around different references.
    shift = _TWO_PI * round(((a.az_lo + a.az_hi) / 2.0 - (b.az_lo + b.az_hi) / 2.0) / _TWO_PI)
    b_lo, b_hi = b.az_lo + shift, b.az_hi + shift
    az_overlap = max(0.0, min(a.az_hi, b_hi) - max(a.az_lo, b_lo))
    el_overlap = max(0.0, min(a.el_hi, b.el_hi) - max(a.el_lo, b.el_lo))
    return float(az_overlap * el_overlap / a.area)


@dataclass
class MaskCandidate:
    """One mask, with everything the cross-camera contest needs to be decided."""

    channel: str
    index: int  # index into that channel's proposal list; the mask order contract
    class_name: str
    score: float
    box_xyxy_px: tuple[float, float, float, float]
    mask_box_xyxy_px: tuple[float, float, float, float]
    footprint: AngularBox
    n_mask_px: int
    suppressed_by: tuple[str, int] | None = None
    ioa: float = 0.0
    # Stage 3b (C27) per-box provenance, carried through untouched: a recovered
    # box is a box no detector proposed on this frame, and a reader of the CVAT
    # export cannot tell one from a detection unless Stage 4 passes it along.
    box_source: str = "yolo"
    track_id: int | None = None
    n_propagated_hops: int = 0

    @property
    def kept(self) -> bool:
        return self.suppressed_by is None


def ioa_nms_across_cameras(
    candidates: Sequence[MaskCandidate],
    *,
    ioa_threshold: float,
    same_class_only: bool,
    camera_priority: Sequence[str],
) -> dict:
    """Suppress duplicates of one object seen by two overlapping cameras.

    Only pairs from DIFFERENT channels are considered: within one image, Stage 3
    already deduplicated, and re-running it here on the same geometry would
    double-suppress with a different threshold.

    Order is fully determined (§1.9): score descending, then the fixed camera
    priority list, then the proposal index. Rev 1 left the tie undefined, which
    makes the surviving box depend on camera iteration order — the output is
    clean, plausible, and different on every run.
    """
    order = sorted(
        range(len(candidates)),
        key=lambda i: (
            -float(candidates[i].score),
            camera_priority.index(candidates[i].channel) if candidates[i].channel in camera_priority else 99,
            candidates[i].channel,
            candidates[i].index,
        ),
    )
    n_suppressed = 0
    for winner_pos, i in enumerate(order):
        if not candidates[i].kept:
            continue
        for j in order[winner_pos + 1 :]:
            other = candidates[j]
            if not other.kept or other.channel == candidates[i].channel:
                continue
            if same_class_only and other.class_name != candidates[i].class_name:
                continue
            ioa = angular_ioa(other.footprint, candidates[i].footprint)
            if ioa > ioa_threshold:
                other.suppressed_by = (candidates[i].channel, candidates[i].index)
                other.ioa = ioa
                n_suppressed += 1
    return {
        "n_candidates": len(candidates),
        "n_suppressed_cross_camera": n_suppressed,
        "n_kept": sum(1 for c in candidates if c.kept),
        "ioa_threshold": ioa_threshold,
        "ioa_space": "ego_frame_azimuth_elevation_rad",
        "same_class_only": same_class_only,
    }


# ---------------------------------------------------------------------------
# The adapter (role `mask_2d`)
# ---------------------------------------------------------------------------


class MobileSamAdapter:
    """MobileSAM as the `mask_2d` role — the pilot-tier fallback since C19.

    Owns its box-prompt transform into the 1024-longest-side space and the
    inverse back to 1600x900; the transform never leaks into stage code
    (§1.5 rule 2). Accepts `state` and `window` and ignores them, returning
    `state=None, propagated=False` — the non-temporal contract of §7.1 fix 1.
    """

    def __init__(self, spec: CheckpointSpec, cfg: MaskConfig) -> None:
        self._spec = spec
        self._cfg = cfg
        self._device = cfg.device
        self._model: Any = None
        self._predictor: Any = None
        self._torch: Any = None
        # Filled by load() from DHAKASCENES_VRAM_CAP_MIB (C1); recorded verbatim
        # in the run manifest.
        self.vram_cap: dict = {"value_mib": None, "enforced": "none",
                               "physical_device_mib": None, "device_name": ""}

    # --- ModelRole surface ---

    @property
    def roles(self) -> tuple[str, ...]:
        return (MASK_2D,)

    @property
    def spec(self) -> CheckpointSpec:
        return self._spec

    @property
    def device(self) -> str:
        return self._device

    @property
    def supports_temporal(self) -> bool:
        # MobileSAM has NO propagation. This is a capability gap, not a quality
        # gap, and it is reported rather than worked around (§4, §5.5).
        return False

    def load(self) -> None:
        try:
            import torch
        except ImportError as exc:
            raise ModelUnavailable(f"torch is not installed: {exc}") from exc
        try:
            from mobile_sam import SamPredictor, sam_model_registry  # type: ignore
        except ImportError:
            try:
                from segment_anything import SamPredictor, sam_model_registry  # type: ignore
            except ImportError as exc:
                raise ModelUnavailable(
                    f"neither mobile_sam nor segment_anything is installed ({exc}); MobileSAM is the "
                    "committed pilot choice (§5.5)"
                ) from exc
        if not self._cfg.checkpoint_path or not os.path.isfile(self._cfg.checkpoint_path):
            raise ModelUnavailable(
                f"checkpoint_path is missing or not a file: {self._cfg.checkpoint_path!r}. MobileSAM "
                "ships weights as a file, not a hub id; pass --checkpoint"
            )
        self._torch = torch
        # C1: the synthetic 4 GiB ceiling, applied BEFORE the first allocation.
        self.vram_cap = apply_vram_cap(torch, self._device)
        sam = sam_model_registry[self._cfg.model_type](checkpoint=self._cfg.checkpoint_path)
        sam.to(self._device).eval()
        self._model = sam
        self._predictor = SamPredictor(sam)

    def unload(self) -> None:
        self._predictor = None
        self._model = None
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

    # --- the role method ---

    def segment(
        self,
        images: Sequence[np.ndarray],
        boxes_xyxy_px: np.ndarray,
        *,
        state: Any | None = None,
        window: TemporalWindow = TemporalWindow(),
        channel: str = "",
    ) -> MaskResult:
        """Absolute-pixel boxes at 1600x900 -> one mask per box, in order, at 1600x900.

        `state` and `window` are accepted and ignored; that is the interface's
        whole point, and a provider that raised on them would make the SAM 2.1
        swap a Stage 4 rewrite (§7.1 fix 1).
        """
        if self._predictor is None:
            raise RuntimeError("adapter is not loaded")
        torch = self._torch
        image = np.asarray(images[0])
        height_px, width_px = int(image.shape[0]), int(image.shape[1])
        if (width_px, height_px) != (self._cfg.image_width_px, self._cfg.image_height_px):
            raise MaskContractError(
                f"{channel}: image is {width_px}x{height_px}; Stage 4 runs at "
                f"{self._cfg.image_width_px}x{self._cfg.image_height_px} (§1.5)"
            )
        boxes = np.asarray(boxes_xyxy_px, dtype=np.float32).reshape(-1, 4)

        if len(boxes) == 0:
            masks_np = np.zeros((0, height_px, width_px), dtype=bool)
        else:
            with torch.inference_mode():
                self._predictor.set_image(image)
                # The predictor's ResizeLongestSide carries the boxes into the
                # 1024 space and the mask postprocessing carries the masks back.
                # Both directions belong to the adapter.
                box_tensor = self._predictor.transform.apply_boxes_torch(
                    torch.as_tensor(boxes, device=self._device), (height_px, width_px)
                )
                masks, _, _ = self._predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=box_tensor,
                    multimask_output=self._cfg.multimask_output,
                )
                masks_np = masks[:, 0].detach().cpu().numpy().astype(bool)
            self._predictor.reset_image()

        # §1.5 rule 4 and the one-mask-per-box-in-order contract, both asserted
        # here rather than trusted to the library version installed today.
        if masks_np.shape[0] != len(boxes):
            raise MaskContractError(
                f"{channel}: {masks_np.shape[0]} masks for {len(boxes)} boxes; Stage 5 indexes masks "
                "by proposal position and a count mismatch silently reassigns every class"
            )
        if masks_np.ndim != 3 or (masks_np.shape[2], masks_np.shape[1]) != (width_px, height_px):
            raise MaskContractError(
                f"{channel}: masks are {masks_np.shape[2]}x{masks_np.shape[1]}, not "
                f"{width_px}x{height_px}. MobileSAM decodes in a 1024-longest-side space; the inverse "
                "transform is the adapter's job (§1.5 rule 4)"
            )

        return MaskResult(
            masks=masks_np,
            state=None,  # nothing is carried forward: no propagation (§4)
            window=window,
            propagated=False,
        )


class TransformersSamAdapter:
    """SAM 2.1 / SAM 3 tracker (transformers) as the `mask_2d` role.

    The default provider since C19 (human decision, 2026-08-13). The model
    family is chosen from `cfg.model_id`: `facebook/sam3*` loads the SAM 3
    tracker classes (gated repo; license grant pending), anything else loads
    SAM 2 — the tracker is a drop-in replacement for SAM 2 (same processor
    call shapes, same video-session surface). Weights come from the hub id,
    never from `checkpoint_path` (that stays MobileSAM-only).

    Owns the transform into model space and the inverse back to 1600x900
    through `processor.post_process_masks` (§1.5 rule 2), and asserts the same
    count / order / resolution contract as MobileSamAdapter rather than
    trusting the library. `supports_temporal` is True: the per-frame path is
    the safe default, and a `window.frames > 1` call carrying several frames
    takes the best-effort video-session propagation path, with the session
    returned as `state`.
    """

    def __init__(self, spec: CheckpointSpec, cfg: MaskConfig) -> None:
        self._spec = spec
        self._cfg = cfg
        self._device = cfg.device
        self._model: Any = None
        self._processor: Any = None
        self._video_model: Any = None
        self._video_processor: Any = None
        self._torch: Any = None
        # Filled by load() from DHAKASCENES_VRAM_CAP_MIB (C1); recorded verbatim
        # in the run manifest.
        self.vram_cap: dict = {"value_mib": None, "enforced": "none",
                               "physical_device_mib": None, "device_name": ""}

    @property
    def _is_sam3(self) -> bool:
        # Exact match since C26: facebook/sam3.1 is NOT this path (it has no
        # transformers integration at all) and startswith() would claim it.
        return self._cfg.model_id == "facebook/sam3"

    # --- ModelRole surface ---

    @property
    def roles(self) -> tuple[str, ...]:
        return (MASK_2D,)

    @property
    def spec(self) -> CheckpointSpec:
        return self._spec

    @property
    def device(self) -> str:
        return self._device

    @property
    def supports_temporal(self) -> bool:
        # SAM 2/3 carry a video memory state across frames. The capability is
        # real even though the driver still calls per frame (C19).
        return True

    def load(self) -> None:
        try:
            import torch
        except ImportError as exc:
            raise ModelUnavailable(f"torch is not installed: {exc}") from exc
        if self._cfg.model_id.startswith("facebook/sam3.1"):
            raise ModelUnavailable(
                "facebook/sam3.1 has no transformers integration (its config.json is a stale "
                "SAM 3 copy); use --provider sam31_multiplex"
            )
        try:
            if self._is_sam3:
                from transformers import Sam3TrackerModel as model_cls  # type: ignore
                from transformers import Sam3TrackerProcessor as processor_cls  # type: ignore
            else:
                from transformers import Sam2Model as model_cls  # type: ignore
                from transformers import Sam2Processor as processor_cls  # type: ignore
        except ImportError as exc:
            raise ModelUnavailable(
                f"transformers does not provide the {'SAM 3 tracker' if self._is_sam3 else 'SAM 2'} "
                f"classes ({exc}); SAM 2.1 needs transformers>=4.56, the SAM 3 tracker the 5.x line "
                "(C19 pins 5.15.0)"
            ) from exc
        self._torch = torch
        # C1: the synthetic ceiling, applied BEFORE the first allocation.
        self.vram_cap = apply_vram_cap(torch, self._device)
        try:
            # Hub id, never checkpoint_path (MobileSAM-only). dtype is explicit
            # because transformers 5.x from_pretrained defaults to dtype='auto'
            # (C19 note; our checkpoints store fp32, but defensively stated).
            self._processor = processor_cls.from_pretrained(
                self._cfg.model_id, revision=self._cfg.revision
            )
            model = model_cls.from_pretrained(
                self._cfg.model_id, revision=self._cfg.revision, dtype=torch.float32
            )
        except (OSError, ValueError) as exc:
            gated = (
                " facebook/sam3 is a gated repo and the license grant is pending (C19); select "
                "sam2_video until it lands"
                if self._is_sam3
                else ""
            )
            raise ModelUnavailable(
                f"could not load {self._cfg.model_id!r}: {exc}.{gated}"
            ) from exc
        model.to(self._device).eval()
        self._model = model

    def unload(self) -> None:
        self._model = None
        self._processor = None
        self._video_model = None
        self._video_processor = None
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

    def _ensure_video(self) -> tuple[Any, Any]:
        """Lazily load the video-session classes; only the propagation path pays for them."""
        if self._video_model is not None:
            return self._video_model, self._video_processor
        torch = self._torch
        if self._is_sam3:
            from transformers import Sam3TrackerVideoModel as model_cls  # type: ignore
            from transformers import Sam3TrackerVideoProcessor as processor_cls  # type: ignore
        else:
            from transformers import Sam2VideoModel as model_cls  # type: ignore
            from transformers import Sam2VideoProcessor as processor_cls  # type: ignore
        self._video_processor = processor_cls.from_pretrained(
            self._cfg.model_id, revision=self._cfg.revision
        )
        model = model_cls.from_pretrained(
            self._cfg.model_id, revision=self._cfg.revision, dtype=torch.float32
        )
        model.to(self._device).eval()
        self._video_model = model
        return self._video_model, self._video_processor

    # --- the role method ---

    def segment(
        self,
        images: Sequence[np.ndarray],
        boxes_xyxy_px: np.ndarray,
        *,
        state: Any | None = None,
        window: TemporalWindow = TemporalWindow(),
        channel: str = "",
    ) -> MaskResult:
        """Absolute-pixel boxes at 1600x900 -> one mask per box, in order, at 1600x900.

        `window.frames == 1` or a single frame runs the per-frame path and
        returns `state=None, propagated=False` — the safe default. Several
        frames under `window.frames > 1` take the best-effort video-session
        propagation path (C19).
        """
        if self._model is None or self._processor is None:
            raise RuntimeError("adapter is not loaded")
        frames = [np.asarray(im) for im in images]
        for frame in frames:
            frame_h, frame_w = int(frame.shape[0]), int(frame.shape[1])
            if (frame_w, frame_h) != (self._cfg.image_width_px, self._cfg.image_height_px):
                raise MaskContractError(
                    f"{channel}: image is {frame_w}x{frame_h}; Stage 4 runs at "
                    f"{self._cfg.image_width_px}x{self._cfg.image_height_px} (§1.5)"
                )
        height_px, width_px = int(frames[0].shape[0]), int(frames[0].shape[1])
        boxes = np.asarray(boxes_xyxy_px, dtype=np.float32).reshape(-1, 4)

        session: Any = None
        propagated = False
        if len(boxes) == 0:
            # Same empty-box contract as MobileSamAdapter: (0, H, W), nothing carried.
            masks_np = np.zeros((0, height_px, width_px), dtype=bool)
        elif window.frames > 1 and len(frames) > 1:
            masks_np, session = self._propagate(frames, boxes, state=state, channel=channel)
            propagated = True
        else:
            masks_np = self._segment_single(frames[0], boxes)

        # §1.5 rule 4 and the one-mask-per-box-in-order contract, both asserted
        # here rather than trusted to the library version installed today.
        if masks_np.shape[0] != len(boxes):
            raise MaskContractError(
                f"{channel}: {masks_np.shape[0]} masks for {len(boxes)} boxes; Stage 5 indexes masks "
                "by proposal position and a count mismatch silently reassigns every class"
            )
        if masks_np.ndim != 3 or (masks_np.shape[2], masks_np.shape[1]) != (width_px, height_px):
            raise MaskContractError(
                f"{channel}: masks are {masks_np.shape[2]}x{masks_np.shape[1]}, not "
                f"{width_px}x{height_px}. SAM decodes at model resolution; carrying the masks back "
                "through processor.post_process_masks is the adapter's job (§1.5 rule 4)"
            )

        return MaskResult(
            masks=masks_np,
            state=session,
            window=window,
            propagated=propagated,
        )

    def _segment_single(self, image: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """One frame, box prompts, masks back at the original resolution."""
        torch = self._torch
        inputs = self._processor(
            images=image,
            input_boxes=[[[float(v) for v in box] for box in boxes]],
            return_tensors="pt",
        )
        # Grabbed before .to(device) so post-processing sees CPU sizes.
        original_sizes = inputs.get("original_sizes")
        if original_sizes is None:
            original_sizes = [(int(image.shape[0]), int(image.shape[1]))]
        inputs = inputs.to(self._device)
        with torch.inference_mode():
            outputs = self._model(**inputs, multimask_output=self._cfg.multimask_output)
        post = self._processor.post_process_masks(
            outputs.pred_masks.detach().cpu(), original_sizes
        )
        masks_t = post[0]
        if masks_t.ndim == 4:
            # (n_boxes, n_masks_per_box, H, W): first mask per box — the same
            # `masks[:, 0]` selection MobileSamAdapter makes.
            masks_t = masks_t[:, 0]
        masks_np = masks_t.cpu().numpy() if hasattr(masks_t, "cpu") else np.asarray(masks_t)
        return masks_np.astype(bool)

    def _propagate(
        self,
        frames: Sequence[np.ndarray],
        boxes: np.ndarray,
        *,
        state: Any | None,
        channel: str,
    ) -> tuple[np.ndarray, Any]:
        """Best-effort video-session propagation (C19).

        Prompts frame 0 with the boxes, propagates across the window, and
        returns the LAST frame's masks with the session carried as `state`.
        The per-frame path stays the safe default: any surprise in the video
        API surfaces here as an explicit error and never breaks that path.
        """
        torch = self._torch
        height_px, width_px = int(frames[0].shape[0]), int(frames[0].shape[1])
        try:
            video_model, video_processor = self._ensure_video()
            session = state
            if session is None:
                session = video_processor.init_video_session(
                    video=list(frames), inference_device=self._device
                )
                video_processor.add_inputs_to_inference_session(
                    inference_session=session,
                    frame_idx=0,
                    obj_ids=list(range(len(boxes))),
                    input_boxes=[[[float(v) for v in box] for box in boxes]],
                )
            last_masks: Any = None
            with torch.inference_mode():
                for output in video_model.propagate_in_video_iterator(session):
                    last_masks = video_processor.post_process_masks(
                        [output.pred_masks], original_sizes=[[height_px, width_px]]
                    )[0]
            if last_masks is None:
                raise RuntimeError("propagate_in_video_iterator yielded no frames")
            masks_t = last_masks
            if masks_t.ndim == 4:
                masks_t = masks_t[:, 0]
            masks_np = masks_t.cpu().numpy() if hasattr(masks_t, "cpu") else np.asarray(masks_t)
            return masks_np.astype(bool), session
        except MaskContractError:
            raise
        except Exception as exc:  # noqa: BLE001 — best-effort branch, explicit by design
            raise NotImplementedError(
                f"{channel}: SAM video propagation failed ({type(exc).__name__}: {exc}). The "
                "propagation path is best-effort (C19); the per-frame path (window_frames=1, or "
                "one frame per call) is the safe default and remains fully supported"
            ) from exc


class Sam31MultiplexAdapter:
    """SAM 3.1 Object Multiplex as the `mask_2d` role — selectable since C26.

    Reached through Meta's own `sam3` package (facebookresearch/sam3@8f0b7f4),
    not transformers: no transformers integration for SAM 3.1 exists, and
    facebook/sam3.1's config.json is a stale SAM 3 copy, so the transformers
    path would load SAM 3's architecture under SAM 3.1's name. Weights come from
    the hub id at a pinned revision (`sam3.1_multiplex.pt`) and are stream-hashed,
    so a run is quotable against bytes rather than against a repo name (§7.2).

    The predictor upstream ships is built for TEXT-prompted, detector-driven
    tracking. Box-prompted instance tracking — what Stage 4 and Stage 3b need —
    takes six measured compensations, EVERY one of which fails silently if it is
    skipped (wrong dtype, vanishing tracklets, discarded masks, a leaked
    autocast). Each is applied below and commented where it is applied; all six
    were measured on this 4090 by scripts/smoke_sam31.py --mode harness, which
    drives this adapter through the identical contracts in --mode adapter.

    Implements `Mask2D` and, additionally, `MaskVideoTracker` (C27): the video
    surface is the same predictor session, exposed for Stage 3b's box recovery.
    """

    # SAM 3.1's box prompt IS SAM-2's corner pair: top-left labelled 2,
    # bottom-right labelled 3. See _add_box_prompts for why not `bounding_boxes=`.
    _BOX_POINT_LABELS: tuple[int, int] = (2, 3)
    _CHECKPOINT_FILENAME: str = "sam3.1_multiplex.pt"
    _HASH_CHUNK_BYTES: int = 1024 * 1024      # the checkpoint is ~3.3 GB; never read whole
    # The C26-recorded sha256 of facebook/sam3.1's sam3.1_multiplex.pt at revision
    # daa63191845a41281374e725f4c9e51c7a824460. THIS is what makes a MobileSAM-for-
    # SAM-3.1 swap impossible — not the filename, which is a label the caller
    # controls and which the HF cache does not even preserve (its snapshot entry is
    # a symlink onto blobs/<sha256>). _resolve_checkpoint verifies it before the
    # builder is handed a path, so no unestablished bytes are ever loaded (§7.2).
    _EXPECTED_SHA256: str = "0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6"

    def __init__(self, spec: CheckpointSpec, cfg: MaskConfig) -> None:
        self._spec = spec
        self._cfg = cfg
        self._device = cfg.device
        self._predictor: Any = None
        self._torch: Any = None
        # Deterministic session ids (§1.9): a counter, never uuid4 — the same
        # inputs must name the same sessions in the same order on a re-run.
        self._session_counter = 0
        # Every session this adapter opened and has not closed. A caller that
        # drops the session id (the Stage 4 driver discards MaskResult.state)
        # would otherwise leave its frames and cached masks on the card until the
        # process ends; unload() sweeps whatever is still in here.
        self._owned_sessions: set[str] = set()
        # Which presence source the frames actually carried, in the vocabulary
        # Stage 3b's manifest already reads off the sibling tracker adapter
        # (getattr(adapter, "presence_source", "unknown")). Recorded so a run
        # whose presence is partly a constant cannot be mistaken for one whose
        # presence is a measurement.
        self.presence_source: str = "unknown"
        # Filled by load() from the stream hash of the weights that actually ran;
        # the run manifest prefers it over the (unset) spec.sha256.
        self.checkpoint_sha256: str | None = None
        # Filled by load() from DHAKASCENES_VRAM_CAP_MIB (C1); recorded verbatim
        # in the run manifest.
        self.vram_cap: dict = {"value_mib": None, "enforced": "none",
                               "physical_device_mib": None, "device_name": ""}

    # --- ModelRole surface ---

    @property
    def roles(self) -> tuple[str, ...]:
        return (MASK_2D,)

    @property
    def spec(self) -> CheckpointSpec:
        return self._spec

    @property
    def device(self) -> str:
        return self._device

    @property
    def supports_temporal(self) -> bool:
        # Real propagation, exercised end to end by the C26 smoke — not merely
        # declared: this adapter also implements MaskVideoTracker (C27).
        return True

    def load(self) -> None:
        try:
            import torch
        except ImportError as exc:
            raise ModelUnavailable(f"torch is not installed: {exc}") from exc
        if not str(self._device).startswith("cuda"):
            raise ModelUnavailable(
                f"device={self._device!r}: sam31_multiplex requires CUDA; "
                "facebookresearch/sam3's builder hard-codes .cuda()"
            )
        try:
            from sam3.model_builder import build_sam3_multiplex_video_predictor  # type: ignore
        except ImportError as exc:
            hint = (
                "the setuptools==80.9.0 pin in requirements.txt restores pkg_resources, which "
                "sam3.model_builder resolves its BPE vocabulary through"
                if "pkg_resources" in str(exc)
                else "sam3 is pinned in requirements-devkit.txt @8f0b7f4 and is installed --no-deps"
            )
            raise ModelUnavailable(f"the sam3 package is not importable ({exc}); {hint}") from exc
        self._torch = torch
        # C1: the synthetic ceiling, applied BEFORE the first allocation — the
        # builder's own .cuda() is that allocation, so this cannot wait for it.
        self.vram_cap = apply_vram_cap(torch, self._device)
        if not self._cfg.model_id.startswith("facebook/sam3.1"):
            # run() refuses the same mismatch before construction; belt and
            # braces, because a manifest that attributes SAM 3.1 masks to some
            # other checkpoint is unfalsifiable after the fact (§7.2).
            raise ModelUnavailable(
                f"provider 'sam31_multiplex' with model_id {self._cfg.model_id!r}: this adapter "
                f"loads facebook/sam3.1's {self._CHECKPOINT_FILENAME} and nothing else"
            )
        # Resolution AND verification: _resolve_checkpoint stream-hashes the file,
        # refuses anything but the pinned digest and records it in
        # checkpoint_sha256 — so nothing unverified reaches the builder below, and
        # the ~3.3 GB are read once rather than twice.
        checkpoint = self._resolve_checkpoint()
        torch.manual_seed(self._cfg.global_seed)
        try:
            # C26 compensation 1 — use_fa3=False is MANDATORY: FA3 is the
            # fp8/Hopper kernel path and on this sm_89 card its import alone
            # fails. compile/warm_up off keep the build deterministic, and
            # async_loading_frames off keeps frame order the caller's order.
            predictor = build_sam3_multiplex_video_predictor(
                checkpoint_path=checkpoint,
                max_num_objects=self._cfg.sam31_max_objects,
                use_fa3=False,
                compile=False,
                warm_up=False,
                async_loading_frames=False,
            )
        except Exception as exc:  # noqa: BLE001 — any build failure is unavailability
            raise ModelUnavailable(
                f"could not build the SAM 3.1 multiplex predictor from {checkpoint!r} "
                f"({type(exc).__name__}: {exc})"
            ) from exc
        # C26 compensation 2, both knobs measured on this card (2026-08-19).
        # `hotstart_delay` buffers and de-duplicates newly appearing objects for
        # that many frames — right for detector proposals, wrong for box prompts,
        # which must appear on the frame they were prompted on. Keep-alive
        # suppression (sam3_multiplex_base.py:2301-2309) then hides any tracklet
        # the detector does not re-confirm, and an instance-only session has no
        # detection prompt at all, so with the shipped default EVERY box-prompted
        # tracklet vanishes after a few frames (measured: 1/7 frames carried
        # objects). Box-prompted instance tracking needs both off; asserted
        # rather than set, because a renamed knob would fail silently.
        for knob in ("hotstart_delay", "suppress_unmatched_only_within_hotstart"):
            if not hasattr(predictor.model, knob):
                raise ModelUnavailable(
                    f"API drift: sam3's predictor.model has no {knob!r}; the C26 instance-mode "
                    "contract was measured against facebookresearch/sam3@8f0b7f4"
                )
        predictor.model.hotstart_delay = 0
        predictor.model.suppress_unmatched_only_within_hotstart = True
        self._predictor = predictor

    def unload(self) -> None:
        """C26 compensation 6: shutdown() is not enough, and the gap is silent.

        `Sam3MultiplexVideoPredictor` enters a bf16 autocast context in its
        __init__ (sam3_multiplex_video_predictor.py:51) and its model wrapper
        enters a second one (sam3_multiplex_base.py:2944), but it inherits the
        BASE shutdown (sam3_base_predictor.py:482), which clears sessions and
        nothing else. Upstream fixed exactly this leak for the non-multiplex
        predictor (sam3_video_predictor.py:99) and not for this class. A leaked
        autocast silently changes the dtype of every model loaded later in the
        process, and its weight-cast cache held ~1.28 GiB on the C26 smoke.
        """
        predictor, torch = self._predictor, self._torch
        # Cleanup is TOTAL: nothing between here and the autocast teardown below
        # may escape. An unguarded sweep let one raising close_session skip
        # compensation 6 entirely and leave the process in bf16 autocast with the
        # ~1.28 GiB weight-cast cache resident — the exact leak this method exists
        # to close, now triggered by the leak-closing code itself. Failures are
        # collected and re-raised AFTER the teardown, so a broken release is loud
        # and never silent, but is never what prevents the release.
        failures: list[str] = []
        # Sessions nobody closed — a caller that dropped the id _propagate handed
        # it, or a window that raised past its close_video — are released through
        # the predictor's own close_session first, while the predictor is still
        # alive. shutdown() below drops the registry, which frees the states but
        # never runs the release path they were registered against.
        if predictor is not None:
            for session_id in sorted(self._owned_sessions):
                try:
                    self._close_session(session_id)
                except Exception as exc:  # noqa: BLE001 — collected, never swallowed
                    # _close_session keeps a session it could NOT close owned, so
                    # the id is still on the adapter after this returns rather
                    # than forgotten by the bookkeeping that was meant to sweep it.
                    failures.append(f"close_session({session_id!r}) {type(exc).__name__}: {exc}")
        self._predictor = None
        if predictor is None:
            return
        if hasattr(predictor, "shutdown"):
            try:
                predictor.shutdown()
            except Exception as exc:  # noqa: BLE001 — same reason as the sweep above
                failures.append(f"shutdown() {type(exc).__name__}: {exc}")
        # Autocast NESTS, so every entered context has to be exited. Walk the
        # object chain: vars() for the context (the wrapper proxies unknown
        # attribute reads to its inner model and would alias two contexts as
        # one), getattr for the "model" hop (nn.Module keeps submodules in
        # _modules, not in __dict__).
        obj: Any = predictor
        seen_ctx_ids: set[int] = set()
        for _ in range(4):
            if obj is None:
                break
            ctx = vars(obj).get("bf16_context") if hasattr(obj, "__dict__") else None
            if ctx is not None and id(ctx) not in seen_ctx_ids:
                seen_ctx_ids.add(id(ctx))
                try:
                    ctx.__exit__(None, None, None)
                finally:
                    obj.bf16_context = None
            obj = getattr(obj, "model", None)
        if torch is not None:
            if self._autocast_enabled(torch):
                # Unbalanced enters beyond the discoverable contexts: restore the
                # end state directly rather than leave the process running in it.
                try:
                    torch.set_autocast_enabled("cuda", False)
                except TypeError:
                    torch.set_autocast_enabled(False)
            # The bf16 weight-cast cache is dropped when autocast nesting reaches
            # zero — which unbalanced enters prevent. ~1.28 GiB on the C26 smoke.
            torch.clear_autocast_cache()
        del predictor, obj
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        if failures:
            # Surfaced only now, and only after the whole teardown above has run:
            # the caller learns that something is still held (and which sessions
            # are still listed in _owned_sessions) without that report being the
            # reason autocast stayed on.
            raise RuntimeError(
                "sam31_multiplex unload did not release everything ("
                + "; ".join(failures)
                + "); the bf16 autocast contexts and the weight-cast cache were torn down "
                "regardless, and every session that failed to close is still owned"
            )

    # --- checkpoint identity (§7.2) ---

    def _resolve_checkpoint(self) -> str:
        """The pinned weights file, decided by DIGEST: an explicit path wins, else the hub id.

        `checkpoint_path` is MobileSAM-only for the other providers, but
        scripts/smoke_sam31.py hands this adapter the file it stream-hashed
        itself (`--checkpoint`), and silently downloading a second copy at a
        different revision would measure something the gate never measured.

        Returns the path and, as a side effect, records its verified sha256 in
        `checkpoint_sha256`: the identity the manifest quotes (§7.2).
        """
        explicit = self._cfg.checkpoint_path
        if explicit:
            # C26: `--checkpoint` defaults to $MOBILE_SAM_CHECKPOINT, which this
            # repo's .env always sets, so an unscoped path arrives on EVERY CLI
            # run whatever the model is. Honouring it blindly fed MobileSAM's
            # 40 MB .pt to the multiplex builder and stream-hashed MobileSAM's
            # bytes into checkpoint_sha256 — SAM 3.1 provenance made of another
            # model's weights, unfalsifiable after the fact (§7.2).
            #
            # The NAME is only a cheap pre-filter, and was never the invariant:
            # taken alone it also refused the RIGHT bytes. The HF cache stores
            # snapshots/<rev>/sam3.1_multiplex.pt as a SYMLINK onto
            # blobs/<sha256>, so a caller that resolves the symlink before handing
            # the path over — scripts/smoke_sam31.py's own _resolve_checkpoint
            # returns os.path.realpath(override) — passes a path whose basename is
            # a hex digest. That spelling is accepted too; the digest below is the
            # decision on both, and it is what makes the MobileSAM swap
            # impossible, not the filename.
            resolved = os.path.realpath(explicit)
            if (
                os.path.basename(explicit) != self._CHECKPOINT_FILENAME
                and os.path.basename(resolved) != self._EXPECTED_SHA256
            ):
                raise ModelUnavailable(
                    f"checkpoint_path={explicit!r} is neither {self._CHECKPOINT_FILENAME} nor the "
                    f"pinned blob {self._EXPECTED_SHA256}: sam31_multiplex loads facebook/sam3.1's "
                    f"{self._CHECKPOINT_FILENAME} and nothing else, and loading or hashing any "
                    "other file would record those bytes as SAM 3.1's provenance. --checkpoint / "
                    "$MOBILE_SAM_CHECKPOINT is the mobile_sam channel: leave it unset for this "
                    "provider and the pinned file is resolved from the hub cache"
                )
            if not os.path.isfile(explicit):
                raise ModelUnavailable(f"checkpoint_path={explicit!r} is not a file")
            path = resolved
        else:
            try:
                from huggingface_hub import hf_hub_download  # type: ignore
            except ImportError as exc:
                raise ModelUnavailable(f"huggingface_hub is not installed: {exc}") from exc
            try:
                path = hf_hub_download(
                    repo_id=self._cfg.model_id,
                    filename=self._CHECKPOINT_FILENAME,
                    revision=self._cfg.revision,
                )
            except Exception as exc:  # noqa: BLE001 — any resolution failure is unavailability
                raise ModelUnavailable(
                    f"cannot resolve {self._cfg.model_id}/{self._CHECKPOINT_FILENAME} at revision "
                    f"{self._cfg.revision!r} ({exc}). facebook/sam3.1 is a gated repo: HF_TOKEN "
                    "must be set for the first download — the grant is held by this account "
                    "(C26) — and HF_HOME must point at the cache it populated"
                ) from exc
        # The decision, on BOTH branches and before the builder or any weight
        # load: the six C26 compensations were measured against these bytes, so a
        # file that hashes to anything else is a different model wearing the right
        # name — including a --revision that resolves to different weights.
        digest = self._sha256_file(path)
        if digest != self._EXPECTED_SHA256:
            raise ModelUnavailable(
                f"checkpoint {path!r} hashes to {digest}, not the pinned "
                f"{self._EXPECTED_SHA256}: that is facebook/sam3.1's {self._CHECKPOINT_FILENAME} "
                "at revision daa63191845a41281374e725f4c9e51c7a824460 (C26), the bytes this "
                "adapter's compensations were measured against. Nothing is built from, or "
                "recorded against, weights nobody established"
            )
        self.checkpoint_sha256 = digest
        return path

    @classmethod
    def _sha256_file(cls, path: str) -> str:
        """Stream-hash the weights, chunked: the identity of the bytes that ran."""
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(cls._HASH_CHUNK_BYTES), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _autocast_enabled(torch: Any) -> bool:
        # torch >= 2.4 wants a device type; the no-arg form is the legacy spelling.
        try:
            return bool(torch.is_autocast_enabled("cuda"))
        except TypeError:
            return bool(torch.is_autocast_enabled())

    # --- the role method ---

    def segment(
        self,
        images: Sequence[np.ndarray],
        boxes_xyxy_px: np.ndarray,
        *,
        state: Any | None = None,
        window: TemporalWindow = PER_FRAME_WINDOW,
        channel: str = "",
    ) -> MaskResult:
        """Absolute-pixel boxes at 1600x900 -> one mask per box, in order, at 1600x900.

        Same shape as TransformersSamAdapter.segment: `window.frames == 1` or a
        single frame runs the per-frame path and returns `state=None,
        propagated=False` — the safe default. Several frames under
        `window.frames > 1` take the best-effort video-propagation path (C26),
        with the session id carried as `state`.
        """
        if self._predictor is None:
            raise RuntimeError("adapter is not loaded")
        frames = [np.asarray(im) for im in images]
        self._assert_frame_sizes(frames, channel)
        height_px, width_px = int(frames[0].shape[0]), int(frames[0].shape[1])
        boxes = np.asarray(boxes_xyxy_px, dtype=np.float32).reshape(-1, 4)

        session: Any = None
        propagated = False
        if len(boxes) == 0:
            # Same empty-box contract as the other two adapters: (0, H, W), nothing carried.
            masks_np = np.zeros((0, height_px, width_px), dtype=bool)
        elif window.frames > 1 and len(frames) > 1:
            masks_np, session = self._propagate(frames, boxes, state=state, channel=channel)
            propagated = True
        else:
            masks_np = self._segment_single(frames[0], boxes, channel)

        # §1.5 rule 4 and the one-mask-per-box-in-order contract, both asserted
        # here rather than trusted to the library version installed today.
        if masks_np.shape[0] != len(boxes):
            raise MaskContractError(
                f"{channel}: {masks_np.shape[0]} masks for {len(boxes)} boxes; Stage 5 indexes masks "
                "by proposal position and a count mismatch silently reassigns every class"
            )
        if masks_np.ndim != 3 or (masks_np.shape[2], masks_np.shape[1]) != (width_px, height_px):
            raise MaskContractError(
                f"{channel}: masks are {masks_np.shape[2]}x{masks_np.shape[1]}, not "
                f"{width_px}x{height_px}. SAM 3.1 decodes at model resolution and the multiplex "
                "predictor resizes back to the source frame; carrying that through is the "
                "adapter's job (§1.5 rule 4)"
            )

        return MaskResult(
            masks=masks_np,
            state=session,
            window=window,
            propagated=propagated,
        )

    def _assert_frame_sizes(self, frames: Sequence[np.ndarray], channel: str) -> None:
        """Every frame at 1600x900, or the run stops here (§1.5)."""
        for frame in frames:
            frame_h, frame_w = int(frame.shape[0]), int(frame.shape[1])
            if (frame_w, frame_h) != (self._cfg.image_width_px, self._cfg.image_height_px):
                raise MaskContractError(
                    f"{channel}: image is {frame_w}x{frame_h}; Stage 4 runs at "
                    f"{self._cfg.image_width_px}x{self._cfg.image_height_px} (§1.5)"
                )

    def _segment_single(self, image: np.ndarray, boxes: np.ndarray, channel: str) -> np.ndarray:
        """One frame, corner-pair box prompts, masks in the source coordinate space."""
        height_px, width_px = int(image.shape[0]), int(image.shape[1])
        session_id = self._start_session([image])
        try:
            outputs = self._add_box_prompts(session_id, 0, range(len(boxes)), boxes, channel)
            mask_by_id = self._outputs_to_masks(outputs)
        finally:
            # One session per call: a 1-frame session left open holds the whole
            # image feature cache on the card until the predictor expires it.
            self._close_session(session_id)
        return self._stack_by_obj_id(mask_by_id, len(boxes), height_px, width_px)

    def _propagate(
        self,
        frames: Sequence[np.ndarray],
        boxes: np.ndarray,
        *,
        state: Any | None,
        channel: str,
    ) -> tuple[np.ndarray, Any]:
        """Best-effort video-session propagation (C26).

        Prompts frame 0 with the boxes, propagates forward over the window and
        returns the LAST frame's masks with the session id carried as `state`.
        The per-frame path stays the safe default: any surprise in the video API
        surfaces here as an explicit error and never breaks that path.
        """
        height_px, width_px = int(frames[0].shape[0]), int(frames[0].shape[1])
        session = state
        # Session ownership, chosen once and stated here (C26): the SUCCESS path
        # hands the live session to the caller, who owns it from the return
        # onward and releases it with close_video; every FAILURE path releases a
        # session opened HERE before the exception leaves. Closing it on success
        # instead and returning state=None is not open to us — MaskResult.validate
        # rejects propagated=True with state=None ("propagation without carried
        # state is not propagation") and process_keyframe calls assert_valid on
        # every result, so that policy would force a false propagated=False into
        # Stage 7's provenance. The current Stage 4 driver hands segment() one
        # frame per call and so never reaches this path at all; for anything that
        # does and then drops the id, unload() sweeps what is still owned.
        opened_here = session is None
        handed_off = False
        try:
            if opened_here:
                session = self.init_video(frames)
                self.add_video_boxes(session, 0, list(range(len(boxes))), boxes)
            last_index, last_masks = -1, {}
            for frame_idx, mask_by_id, _presence in self.propagate_video(
                session, start_frame_idx=0, max_frames=len(frames)
            ):
                if frame_idx >= last_index:
                    last_index, last_masks = frame_idx, mask_by_id
            if last_index < 0:
                raise RuntimeError("propagate_in_video yielded no frames")
            masks_np = self._stack_by_obj_id(last_masks, len(boxes), height_px, width_px)
            handed_off = True
            return masks_np, session
        except MaskContractError:
            raise
        except Exception as exc:  # noqa: BLE001 — best-effort branch, explicit by design
            raise NotImplementedError(
                f"{channel}: SAM 3.1 video propagation failed ({type(exc).__name__}: {exc}). The "
                "propagation path is best-effort (C26); the per-frame path (window_frames=1, or "
                "one frame per call) is the safe default and remains fully supported"
            ) from exc
        finally:
            # ~0.4 GB of frames and cached masks for a 7-frame/30-box window, on
            # every raise, including the two re-raises above.
            if opened_here and not handed_off and session is not None:
                self._close_session(session)

    # --- MaskVideoTracker (C27): the temporal facet, on the same sessions ---

    def init_video(self, frames: Sequence[np.ndarray]) -> Any:
        """Open a propagation session over an ordered window of 1600x900 frames."""
        if self._predictor is None:
            raise RuntimeError("adapter is not loaded")
        as_arrays = [np.asarray(frame) for frame in frames]
        self._assert_frame_sizes(as_arrays, "")
        return self._start_session(as_arrays)

    def add_video_boxes(
        self, session: Any, frame_idx: int, obj_ids: Sequence[int], boxes_xyxy_px: np.ndarray
    ) -> None:
        """Box-prompt `obj_ids` on `frame_idx`; the caller's ids are kept verbatim."""
        if self._predictor is None:
            raise RuntimeError("adapter is not loaded")
        self._add_box_prompts(session, int(frame_idx), obj_ids, boxes_xyxy_px, "")

    def propagate_video(
        self,
        session: Any,
        *,
        start_frame_idx: int,
        max_frames: int | None = None,
        reverse: bool = False,
    ):
        """Yield `(frame_idx, {obj_id: mask}, {obj_id: presence})` per visited frame.

        An obj_id absent from a yield means "no mask on this frame": the driver
        zero-fills and the adapter never fabricates (C27's protocol). Forward and
        reverse are separate calls, so the direction is never inferred.
        """
        if self._predictor is None:
            raise RuntimeError("adapter is not loaded")
        request = {
            "type": "propagate_in_video",
            "session_id": session,
            "propagation_direction": "backward" if reverse else "forward",
            "start_frame_index": int(start_frame_idx),
            "max_frame_num_to_track": max_frames,
        }
        for item in self._predictor.handle_stream_request(request):
            frame_index = int(item["frame_index"])
            outputs = item["outputs"]
            if outputs is None:
                # A non-rank-0 worker yields the frame index with no payload. On
                # this single-GPU box it never fires; fabricating masks for it
                # would be the one failure this protocol cannot tolerate.
                yield frame_index, {}, {}
                continue
            ids = [int(v) for v in np.asarray(outputs["out_obj_ids"]).reshape(-1).tolist()]
            probs_arr = np.asarray(outputs["out_probs"]).reshape(-1)
            probs = probs_arr.tolist()
            # out_probs is already a probability in [0, 1] (a sigmoid times the
            # presence score, sam3_image_processor.py:195-197): reported, never rescaled.
            if len(probs) == len(ids):
                presence = {oid: float(probs[k]) for k, oid in enumerate(ids)}
                # STICKY, deliberately unlike the Stage 3b sibling's last-write-wins:
                # one clean frame after a mismatched one must not erase the record
                # that some presence in this run was fabricated.
                if self.presence_source in ("unknown", "out_probs"):
                    self.presence_source = "out_probs"
            else:
                # Dropping the trailing ids here was invisible downstream: Stage 3b
                # fills the gap with `presence.get(obj_id, 1.0)` and writes that
                # fabricated 1.0 into its 12 Hz artifact with nothing counted. Fill
                # them here instead, and say so where the manifest reads it.
                presence = {
                    oid: (float(probs[k]) if k < len(probs) else 1.0)
                    for k, oid in enumerate(ids)
                }
                self.presence_source = (
                    f"partial_1.0_out_probs_shape_mismatch:probs{tuple(probs_arr.shape)}"
                    f"_ids{(len(ids),)}"
                )
            yield frame_index, self._outputs_to_masks(outputs), presence

    def close_video(self, session: Any) -> None:
        """Release the session's device memory; sessions never outlive this call."""
        self._close_session(session)
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

    # --- session plumbing: the C26 compensations 3, 4 and 5 ---

    def _start_session(self, frames: Sequence[Any]) -> str:
        """`start_session` for the multiplex predictor, minus the kwarg it rejects.

        C26 compensation 3: `Sam3BasePredictor.start_session` passes
        offload_state_to_cpu unconditionally (sam3_base_predictor.py:126-131)
        and the multiplex model's init_state (sam3_multiplex_tracking.py:207-215)
        does not accept it -> TypeError. So mirror the registration
        start_session does, filtering the kwargs through the model's own
        signature — the same defensive pattern upstream's add_prompt already
        uses (sam3_base_predictor.py:196-201). Session ids are a deterministic
        counter, never uuid4 (§1.9).
        """
        predictor = self._predictor
        init_kwargs: dict[str, Any] = {
            "resource_path": self._as_pil(frames),
            "offload_video_to_cpu": False,
            "offload_state_to_cpu": False,
        }
        if hasattr(predictor, "async_loading_frames"):
            init_kwargs["async_loading_frames"] = predictor.async_loading_frames
        if hasattr(predictor, "video_loader_type"):
            init_kwargs["video_loader_type"] = predictor.video_loader_type
        valid = set(inspect.signature(predictor.model.init_state).parameters)
        state = predictor.model.init_state(**{k: v for k, v in init_kwargs.items() if k in valid})
        session_id = f"stage4-sam31-{self._session_counter}"
        self._session_counter += 1
        now = time.time()
        predictor._all_inference_states[session_id] = {
            "state": state, "session_id": session_id, "start_time": now, "last_use_time": now,
        }
        # Registered here and dropped in _close_session: unload()'s sweep is only
        # as good as this bookkeeping, and this method IS the registration.
        self._owned_sessions.add(session_id)
        # C26 compensation 4: `_build_sam2_output` returns {} for any frame absent
        # from cached_frame_outputs (sam3_multiplex_tracking.py:1244-1245). A
        # text-prompted session caches every frame through the detector; an
        # instance-only session caches only the prompted frame, so propagation
        # computes refined tracker masks and then DISCARDS them at that gate —
        # measured: 1/7 frames carried objects without this seed, 7/7 with it.
        # Seeding empty entries passes the gate; real outputs overwrite the seeds.
        for frame_index in range(int(state["num_frames"])):
            state["cached_frame_outputs"].setdefault(frame_index, {})
        return session_id

    def _close_session(self, session_id: Any) -> None:
        predictor = self._predictor
        if predictor is None:
            # No predictor left to close against: the registry the session lived
            # in went with it, so holding the id would only make unload()'s sweep
            # retry a close that can never run.
            self._owned_sessions.discard(session_id)
            return
        try:
            predictor.close_session(session_id=session_id)
        except (RuntimeError, KeyError):
            # close_session is idempotent by contract, but a session already
            # closed (or expired) must not turn a finally: into the failure the
            # caller sees instead of the real one. The state is gone either way,
            # so the id is dropped below.
            pass
        # Dropped only once the close has actually RUN. Discarding first made a
        # raising close forget the session, and _owned_sessions is the only input
        # unload()'s sweep has: a session lost here could never be swept again.
        self._owned_sessions.discard(session_id)

    @staticmethod
    def _as_pil(frames: Sequence[Any]) -> list[Any]:
        """Frames as PIL Images — what init_state's frame loader reads."""
        from PIL import Image

        return [
            Image.fromarray(np.asarray(frame, dtype=np.uint8)) if isinstance(frame, np.ndarray)
            else frame
            for frame in frames
        ]

    def _add_box_prompts(
        self,
        session_id: Any,
        frame_idx: int,
        obj_ids: Sequence[int],
        boxes: np.ndarray,
        channel: str,
    ) -> Any:
        """C26 compensation 5: one corner-pair point prompt per object.

        SAM 3.1's box prompt IS SAM-2's corner pair — top-left labelled 2,
        bottom-right labelled 3, in coordinates relative to the frame. The
        `bounding_boxes=` / `boxes_xywh=` spelling also returns masks, and is the
        wrong one: it routes to the semantic exemplar path, which opens with
        `self.reset_state(inference_state)` (sam3_multiplex_tracking.py:1695) and
        discards every instance prompted so far. Returns the LAST call's outputs,
        which carry every object prompted on this frame so far.
        """
        predictor = self._predictor
        boxes_list = np.asarray(boxes, dtype=np.float32).reshape(-1, 4).tolist()
        ids = [int(v) for v in obj_ids]
        if len(ids) != len(boxes_list):
            raise MaskContractError(
                f"{channel}: {len(ids)} object ids for {len(boxes_list)} boxes; the id -> box "
                "alignment IS the mask order contract"
            )
        width_px = float(self._cfg.image_width_px)
        height_px = float(self._cfg.image_height_px)
        outputs = None
        for position, (obj_id, box) in enumerate(zip(ids, boxes_list)):
            x1, y1, x2, y2 = box
            outputs = predictor.add_prompt(
                session_id=session_id,
                frame_idx=int(frame_idx),
                obj_id=int(obj_id),
                points=[[x1 / width_px, y1 / height_px], [x2 / width_px, y2 / height_px]],
                point_labels=list(self._BOX_POINT_LABELS),
                rel_coordinates=True,
            )["outputs"]
            if outputs is None:
                raise MaskContractError(
                    f"{channel}: SAM 3.1 refused object {position} of {len(ids)}: "
                    f"sam31_max_objects={self._cfg.sam31_max_objects} exhausted"
                )
        return outputs

    @staticmethod
    def _outputs_to_masks(outputs: Any) -> dict[int, np.ndarray]:
        """`{obj_id: (900, 1600) bool}` from one predictor output dict."""
        masks = np.asarray(outputs["out_binary_masks"])
        ids = [int(v) for v in np.asarray(outputs["out_obj_ids"]).reshape(-1).tolist()]
        return {obj_id: np.asarray(masks[k], dtype=bool) for k, obj_id in enumerate(ids)}

    @staticmethod
    def _stack_by_obj_id(
        mask_by_id: dict, n_boxes: int, height_px: int, width_px: int
    ) -> np.ndarray:
        """(N, H, W) bool in obj_id order, zero-filling the ids the model dropped.

        An absent id is legal, not a fault: SAM drops zero-area masks, and Stage
        4 counts an empty mask as empty/tiny downstream (n_empty_or_tiny_masks)
        rather than inventing pixels for it.

        An id OUTSIDE [0, n_boxes) is a different thing: this stacking pairs mask
        to box BY POSITION, so a session prompted with ids that are not box
        positions (Stage 3b prompts track ids like 37, 42) has no placement at
        all. Skipping those ids fabricated an (n_boxes, H, W) array from nothing,
        which also made segment()'s one-mask-per-box assert unfalsifiable for
        this adapter — an identity mismatch surfaced as an all-empty mask set
        instead of an error (C26). It raises now.
        """
        stray = sorted({int(o) for o in mask_by_id if not 0 <= int(o) < n_boxes})
        if stray:
            raise MaskContractError(
                f"obj_id(s) {stray} outside the expected range [0, {n_boxes}): masks are stacked "
                "by obj_id AS box position, so an id that is not a box position cannot be placed "
                "and zero-filling it would return an all-empty mask set for a real mismatch"
            )
        masks_np = np.zeros((n_boxes, height_px, width_px), dtype=bool)
        for obj_id, mask in mask_by_id.items():
            masks_np[int(obj_id)] = np.asarray(mask, dtype=bool)
        return masks_np


# PREFIX rows only, and `facebook/sam3` is deliberately not one of them: it is
# claimed by the exact-match branch below. As a prefix it silently undid C26's
# refusal contract — `facebook/sam3-video` (a typo, or a future variant) inferred
# sam3_tracker and was then loaded through the SAM 2 transformers classes, which
# is exactly the false provider recorded against a checkpoint nobody established
# that the refusal exists to prevent. facebook/sam3.1 keeps its prefix row: a
# `facebook/sam3.1-...` variant is still this adapter's.
_PROVIDER_BY_MODEL_PREFIX: tuple[tuple[str, str], ...] = (
    ("facebook/sam3.1", "sam31_multiplex"),
    ("mobile_sam", "mobile_sam"),
    ("facebook/sam2", "sam2_video"),
)


def infer_mask_provider(model_id: str) -> str:
    """model_id -> registered provider name (C19, C26). An explicit cfg.provider wins.

    Behaviour change, deliberate and recorded in C26: an unknown model_id now
    REFUSES instead of falling through to sam2_video. The fallthrough recorded
    whichever provider the default happened to be against a checkpoint nobody
    established a provider for, and the manifest is the only place a reader can
    see which model actually produced the masks (§7.2) — the same reasoning as
    infer_reid_provider (Stage 7).
    """
    if model_id == "facebook/sam3":
        # Exact: only the plain SAM 3 tracker has a transformers integration.
        return "sam3_tracker"
    for prefix, provider in _PROVIDER_BY_MODEL_PREFIX:
        if model_id.startswith(prefix):
            return provider
    raise UpstreamRefusal(
        f"unknown mask_2d checkpoint {model_id!r}: no registered provider claims it (known: "
        f"exactly 'facebook/sam3', or prefixes {[p for p, _ in _PROVIDER_BY_MODEL_PREFIX]}). "
        "A checkpoint whose provider "
        "nobody established would be recorded under another model's name; pass --provider "
        "explicitly to override this inference"
    )


# The only providers whose weights are a local FILE rather than a hub id, and so
# the only ones cfg.checkpoint_path may reach (C26; see run()).
_FILE_CHECKPOINT_PROVIDERS: frozenset[str] = frozenset({"mobile_sam"})

_MASK_ADAPTERS: dict[str, type] = {
    "mobile_sam": MobileSamAdapter,
    "sam2_video": TransformersSamAdapter,
    "sam3_tracker": TransformersSamAdapter,
    "sam31_multiplex": Sam31MultiplexAdapter,
}

_PROVIDER_PROVENANCE: dict[str, str] = {
    "mobile_sam": "pilot tier, §5.5; MobileSAM ~10 M params, ~40 MB (fallback since C19)",
    "sam2_video": (
        "C19 (human, 2026-08-13): SAM 2.1-hiera-large default mask_2d on the 24 GB box; "
        "SA-V J&F 78.4"
    ),
    "sam3_tracker": (
        "C19 (human, 2026-08-13): selectable pending the gated-repo license grant; "
        "SA-V J&F 84.4 vs SAM2.1-L 78.4"
    ),
    "sam31_multiplex": (
        "C26 (human, 2026-08-19): SAM 3.1 Object Multiplex via facebookresearch/sam3@8f0b7f4 + "
        "facebook/sam3.1@daa6319 (no transformers integration exists); instance-mode contract "
        "measured on this 4090 (smoke: 32/32 masks, 0.105 s/frame video, 7456 MiB peak); "
        "selectable pending A/B vs sam3_tracker"
    ),
}


def preflight_mask_adapter(adapter: Any, provider: str, cfg: MaskConfig) -> dict:
    """Prove what can be proved about the model BEFORE `clear_markers` (C16, C26).

    `run()` took the previous run's marker down and only THEN resolved the
    provider and loaded the model, so a ModelUnavailable — the sam3 package
    missing, a --revision that does not exist, a gated download refused, a
    checkpoint whose digest is not the pinned one, MobileSAM's weights file gone —
    exited 2 over a tree carrying no marker at all, which every downstream stage
    reads as "Stage 4 never ran". The house contract is that exit 2 leaves nothing
    written and the previous state intact, so every refusal that lives in the
    CHEAP half of an adapter's load() is raised here instead. Same shape as Stage
    3b's `preflight_tracker`, for the same reason.

    It is a preflight, not a load. The expensive half is deliberately not paid
    twice and what that leaves uncovered is named in `not_covered` and recorded in
    the manifest: weights that RESOLVE but cannot be BUILT still fail after
    clear_markers and still exit 2 over a marker-less tree. That residue is
    stated, not hidden.
    """
    started = time.time()
    checks: list[str] = ["adapter_constructed"]
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise ModelUnavailable(f"torch is not installed: {exc}") from exc
    checks.append("torch_import")

    if provider == "sam31_multiplex":
        if not str(cfg.device).startswith("cuda"):
            # Mirrored from the adapter's own load(), which would raise it only
            # once the marker was already gone; the point of a preflight is that
            # it raises here instead.
            raise ModelUnavailable(
                f"device={cfg.device!r}: sam31_multiplex requires CUDA; "
                "facebookresearch/sam3's builder hard-codes .cuda()"
            )
        checks.append("device_is_cuda")
        try:
            from sam3.model_builder import build_sam3_multiplex_video_predictor  # noqa: F401
        except ImportError as exc:
            raise ModelUnavailable(
                f"the sam3 package is not importable ({exc}); it is pinned in "
                "requirements-devkit.txt @8f0b7f4 and is installed --no-deps"
            ) from exc
        checks.append("sam3_package_import")
        # The adapter's OWN resolution and digest verification, called on the
        # instance rather than reimplemented here: hf_hub_download of the pinned
        # sam3.1_multiplex.pt is where a gated repo, a bad --revision or a missing
        # HF_TOKEN actually fails, and the pinned sha256 is where the wrong bytes
        # fail (§7.2). PROBED, never assumed, as Stage 3b probes it — a rename
        # there degrades this to an honest gap rather than to a preflight that
        # silently checks less than it claims. Cost, stated rather than hidden:
        # the ~3.3 GB file is stream-hashed here and again inside load(),
        # page-cached the second time (~7 s on this box), and downloaded at most
        # once.
        resolve = getattr(adapter, "_resolve_checkpoint", None)
        if callable(resolve):
            resolve()
            checks.append("checkpoint_resolved")
            checks.append("checkpoint_digest_verified")
            not_covered = "the multiplex predictor itself is built by adapter.load()"
        else:
            not_covered = (
                "API drift: this adapter exposes no _resolve_checkpoint, so checkpoint "
                "resolution AND the predictor build both happen in adapter.load()"
            )
    elif provider == "mobile_sam":
        # MobileSAM ships weights as a FILE, not a hub id; a missing one is the
        # whole of its cheap half. Mirrored from MobileSamAdapter.load().
        if not cfg.checkpoint_path or not os.path.isfile(cfg.checkpoint_path):
            raise ModelUnavailable(
                f"checkpoint_path is missing or not a file: {cfg.checkpoint_path!r}. MobileSAM "
                "ships weights as a file, not a hub id; pass --checkpoint"
            )
        checks.append("checkpoint_file_present")
        not_covered = (
            "the mobile_sam / segment_anything import and the sam_model_registry build happen "
            "in adapter.load()"
        )
    else:
        # sam2_video / sam3_tracker (TransformersSamAdapter). The class import is
        # the only cheap half there is: transformers fetches config, processor AND
        # ~3.4 GB of weights through the same from_pretrained, so hoisting the
        # gated-repo / bad-revision refusal would mean paying for the weights
        # here. Exact match on facebook/sam3, as the adapter's _is_sam3 is (C26).
        is_sam3 = cfg.model_id == "facebook/sam3"
        try:
            if is_sam3:
                from transformers import Sam3TrackerModel  # noqa: F401  # type: ignore
                from transformers import Sam3TrackerProcessor  # noqa: F401  # type: ignore
            else:
                from transformers import Sam2Model  # noqa: F401  # type: ignore
                from transformers import Sam2Processor  # noqa: F401  # type: ignore
        except ImportError as exc:
            raise ModelUnavailable(
                f"transformers does not provide the {'SAM 3 tracker' if is_sam3 else 'SAM 2'} "
                f"classes ({exc}); SAM 2.1 needs transformers>=4.56, the SAM 3 tracker the 5.x "
                "line (C19 pins 5.15.0)"
            ) from exc
        checks.append("transformers_classes_import")
        not_covered = (
            "from_pretrained (processor + ~3.4 GB of weights) is the whole load: a gated repo, a "
            "bad --revision or an OOM still refuses after clear_markers"
        )
    return {
        "provider": provider,
        "checks": checks,
        "not_covered": not_covered,
        "elapsed_s": round(time.time() - started, 2),
    }


def mask_tight_box(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    """Tight xyxy box of a boolean mask, or None if the mask is empty."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return None
    y_indices = np.nonzero(rows)[0]
    x_indices = np.nonzero(cols)[0]
    return (
        float(x_indices[0]),
        float(y_indices[0]),
        float(x_indices[-1] + 1),
        float(y_indices[-1] + 1),
    )


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------


def write_masks_npz(path: str, per_channel: dict[str, np.ndarray], *, bit_pack: bool) -> str:
    """One npz per keyframe: `<channel>` -> (N, H, W) bool, optionally packed.

    Bit-packed along the last axis, so unpacking needs the original width:
    `np.unpackbits(arr, axis=-1, count=1600)`. The width is stored in the file
    rather than assumed by the reader.
    """
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    payload: dict[str, np.ndarray] = {}
    for channel, masks in per_channel.items():
        payload[channel] = np.packbits(masks, axis=-1) if bit_pack else masks
    payload["__width_px__"] = np.asarray([IMAGE_WIDTH_PX], dtype=np.int32)
    payload["__height_px__"] = np.asarray([IMAGE_HEIGHT_PX], dtype=np.int32)
    payload["__bit_packed__"] = np.asarray([1 if bit_pack else 0], dtype=np.int8)
    # The suffix is not cosmetic: np.savez_compressed APPENDS ".npz" to any name
    # that does not already end in it, so a ".tmp" temp file is written as
    # ".tmp.npz" and the rename below looks for a file that was never created.
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **payload)
    if not os.path.isfile(tmp):
        raise RuntimeError(f"{tmp}: np.savez_compressed did not write the name it was given")
    os.replace(tmp, path)
    return path


def read_proposal_rows(path: str) -> list[dict]:
    if not os.path.isfile(path):
        raise UpstreamRefusal(f"{path} not found; run `python3 -m pipeline.stage3_proposals.proposals` first")
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_upstream(stage3_dir: str, *, accept_degraded: bool = False):
    """The C16 gate over Stage 3, plus the resolution cross-check."""
    manifest, marker = require_upstream(
        stage3_dir,
        stage_name="Stage 3",
        module_hint="pipeline.stage3_proposals.proposals",
        accept_degraded=accept_degraded,
    )
    resolution = manifest.get("image_size_px")
    if resolution != [IMAGE_WIDTH_PX, IMAGE_HEIGHT_PX]:
        # A Stage 3 resolution fallback is legal and recorded (§13.1); consuming
        # its boxes at a different resolution is not.
        raise UpstreamRefusal(
            f"Stage 3 ran at {resolution}, Stage 4 runs at {[IMAGE_WIDTH_PX, IMAGE_HEIGHT_PX]}"
        )
    return manifest, marker


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def optional_c27_array(row: dict, key: str, n_boxes: int) -> list:
    """One of Stage 3b's optional per-box arrays, or [] — but never a SHORT one.

    ABSENT is legal: a plain Stage 3 row (a pre-C27 tree) carries no recovery
    provenance and the caller's defaults are the honest reading of that. PRESENT
    but shorter than `boxes_xyxy_px` is not: every box past the array's end then
    took the default `"yolo"` / None / 0, so a RECOVERED box — one no detector
    proposed on this frame — was recorded as detector output. That is exactly the
    provenance laundering C27 exists to prevent, and a truncated or partially
    written upstream row is how it happens. Refused as the upstream-contract
    failure it is, the way load_upstream and read_proposal_rows refuse.
    """
    values = row.get(key)
    if values is None:
        return []
    values = list(values)
    if len(values) != n_boxes:
        raise UpstreamRefusal(
            f"sample_data_token={row.get('sample_data_token')!r}: {key} has {len(values)} "
            f"entries for {n_boxes} boxes in boxes_xyxy_px. These are parallel arrays (C27): a "
            "mismatch does not shift the provenance, it silently DROPS it — every box past the "
            "end reads as a plain detection. Re-run "
            "`python3 -m pipeline.stage3b_track2d.track2d` over this scene"
        )
    return values


def process_keyframe(
    rows: Sequence[dict],
    adapter: Mask2D,
    substrate: Substrate,
    cfg: MaskConfig,
    dataroot: str,
    camera_priority: Sequence[str],
) -> tuple[dict[str, np.ndarray], list[MaskCandidate], dict]:
    """Segment every camera of one keyframe, then contest across cameras."""
    from PIL import Image

    calibrated = substrate.by_token("calibrated_sensor.json")
    per_channel_masks: dict[str, np.ndarray] = {}
    candidates: list[MaskCandidate] = []
    n_empty_masks = 0

    for row in rows:
        channel = row["channel"]
        boxes = np.asarray(row["boxes_xyxy_px"], dtype=np.float32).reshape(-1, 4)
        with Image.open(os.path.join(dataroot, row["image_path"])) as im:
            image = np.asarray(im.convert("RGB"), dtype=np.uint8)

        result = adapter.segment(
            [image],
            boxes,
            state=None,
            window=TemporalWindow(
                frames=cfg.window_frames, reinit_at_block_boundary=cfg.reinit_at_block_boundary
            ),
            channel=channel,
        )
        result.assert_valid(n_boxes=len(boxes), prefix=f"{channel}: ")
        per_channel_masks[channel] = result.masks

        cs_record = calibrated[row["calibrated_sensor_token"]]
        intrinsic = np.asarray(cs_record["camera_intrinsic"], dtype=np.float64)
        camera_to_ego = Transform.from_nuscenes(cs_record, source_frame=CAMERA, parent_frame=EGO)

        # Stage 3b (C27) OPTIONAL parallel arrays, one entry per box. Absent is
        # "no recovery provenance", not a fault, and the defaults below are the
        # honest reading of it; present-but-short is an upstream contract failure
        # and is refused, not filled in (see optional_c27_array).
        n_boxes = len(boxes)
        box_sources = optional_c27_array(row, "box_sources", n_boxes)
        track_ids = optional_c27_array(row, "track_ids", n_boxes)
        propagated_hops = optional_c27_array(row, "n_propagated_hops", n_boxes)

        for index in range(len(boxes)):
            mask = result.masks[index]
            n_px = int(mask.sum())
            tight = mask_tight_box(mask)
            if tight is None or n_px < cfg.min_mask_px:
                # An empty or hair-thin mask cannot carry the >= 5 single-sweep
                # returns Stage 9 gates on. Counted, not silently dropped.
                n_empty_masks += 1
                continue
            candidates.append(
                MaskCandidate(
                    channel=channel,
                    index=index,
                    class_name=row["class_names"][index],
                    score=float(row["scores"][index]),
                    box_xyxy_px=tuple(float(v) for v in boxes[index]),
                    mask_box_xyxy_px=tight,
                    footprint=angular_footprint(tight, intrinsic, camera_to_ego),
                    n_mask_px=n_px,
                    box_source=str(box_sources[index]) if index < len(box_sources) else "yolo",
                    track_id=int(track_ids[index]) if index < len(track_ids) else None,
                    n_propagated_hops=(
                        int(propagated_hops[index]) if index < len(propagated_hops) else 0
                    ),
                )
            )

    nms_ledger = ioa_nms_across_cameras(
        candidates,
        ioa_threshold=cfg.ioa_threshold,
        same_class_only=cfg.same_class_only,
        camera_priority=camera_priority,
    )
    nms_ledger["n_empty_or_tiny_masks"] = n_empty_masks
    return per_channel_masks, candidates, nms_ledger


def candidate_rows(keyframe_token: str, scene_token: str, candidates: Sequence[MaskCandidate]) -> list[dict]:
    return [
        {
            "keyframe_token": keyframe_token,
            "scene_token": scene_token,
            "channel": c.channel,
            "proposal_index": c.index,
            "class_name": c.class_name,
            "score": round(c.score, 5),
            "proposal_box_xyxy_px": [round(v, 3) for v in c.box_xyxy_px],
            "mask_box_xyxy_px": [round(v, 3) for v in c.mask_box_xyxy_px],
            "n_mask_px": c.n_mask_px,
            # Stage 3b (C27) provenance, carried through for the CVAT export:
            # purely additive, nothing downstream keys on them yet.
            "box_source": c.box_source,
            "track_id": c.track_id,
            "n_propagated_hops": c.n_propagated_hops,
            "angular_footprint": c.footprint.as_dict(),
            "kept": c.kept,
            "suppressed_by": list(c.suppressed_by) if c.suppressed_by else None,
            "suppression_ioa": round(c.ioa, 4) if c.suppressed_by else None,
        }
        for c in candidates
    ]


def run(
    paths: Paths,
    upstream: dict,
    upstream_marker,
    cfg: MaskConfig,
    stage3_dir: str,
    out_dir: str,
    scene_names: Sequence[str] | None,
) -> tuple[dict, int]:
    started = time.time()
    # Everything from here to `clear_markers` is READ-ONLY, and deliberately so:
    # a refusal that has already taken the previous run's marker down leaves a
    # marker-less tree that downstream reads as "Stage 4 never ran", when the
    # house contract for exit 2 is that nothing was written and the previous
    # state stands. Provider resolution, the attribution guards, the spec, the
    # adapter's constructability and its checkpoint all decide up here now.
    provider = cfg.provider or infer_mask_provider(cfg.model_id)
    if provider not in _MASK_ADAPTERS:
        raise UpstreamRefusal(
            f"unknown mask_2d provider {provider!r}; one of {sorted(_MASK_ADAPTERS)}"
        )
    if provider == "mobile_sam" and not cfg.model_id.startswith("mobile_sam"):
        raise UpstreamRefusal(
            f"provider 'mobile_sam' with model_id {cfg.model_id!r}: the manifest would attribute "
            "MobileSAM masks to a hub checkpoint; pass --model-id mobile_sam_vit_t"
        )
    # The same attribution guard for C26, in both directions: SAM 3.1's masks
    # may only be recorded against facebook/sam3.1, and facebook/sam3.1 may only
    # be run by the one adapter that can load it (§7.2).
    if provider == "sam31_multiplex" and not cfg.model_id.startswith("facebook/sam3.1"):
        raise UpstreamRefusal(
            f"provider 'sam31_multiplex' with model_id {cfg.model_id!r}: the manifest would "
            "attribute SAM 3.1 multiplex masks to another checkpoint; pass --model-id facebook/sam3.1"
        )
    if cfg.model_id.startswith("facebook/sam3.1") and cfg.provider not in ("", "sam31_multiplex"):
        raise UpstreamRefusal(
            f"model_id {cfg.model_id!r} with provider {cfg.provider!r}: facebook/sam3.1 has no "
            "transformers integration (its config.json is a stale SAM 3 copy); the only adapter "
            "that loads it is sam31_multiplex (C26)"
        )
    # C26: `--checkpoint` defaults to $MOBILE_SAM_CHECKPOINT, which the .env
    # contract always sets, so MobileSAM's weights file rides along on every CLI
    # run whatever --model-id says. Only a provider that consumes a local weights
    # file may see it; on any other, that path reached an adapter that would load
    # AND stream-hash another model's bytes as this run's provenance (§7.2). The
    # hub id resolves the right file for the rest, so this drops the mis-scoped
    # path rather than failing a run over an env default the user never typed —
    # loudly, and out of the config the manifest records.
    if cfg.checkpoint_path and provider not in _FILE_CHECKPOINT_PROVIDERS:
        print(
            f"note: ignoring checkpoint_path={cfg.checkpoint_path!r}: a local weights file is the "
            f"{sorted(_FILE_CHECKPOINT_PROVIDERS)} channel and provider {provider!r} resolves "
            f"{cfg.model_id!r} from the hub (C26)",
            file=sys.stderr,
        )
        cfg = replace(cfg, checkpoint_path="")

    spec = CheckpointSpec(
        role=MASK_2D,
        provider=provider,
        model_id=cfg.model_id,
        revision=cfg.revision,
        provenance=_PROVIDER_PROVENANCE[provider],
    )
    spec_errors = spec.validate(prefix="checkpoint: ")
    if spec_errors:
        raise UpstreamRefusal(
            "; ".join(spec_errors)
            + " — pass --revision (a hub git sha, or an identifier for the MobileSAM weights file)"
        )

    adapter = _MASK_ADAPTERS[provider](spec, cfg)
    preflight = preflight_mask_adapter(adapter, provider, cfg)
    substrate = Substrate.load(paths)

    # §1.5 rule 5's fixed camera-priority list, used here only as the final
    # deterministic tie-break when two cameras score a duplicate identically.
    camera_priority = tuple(upstream.get("config", {}).get("camera_priority", ()) or (
        "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
    ))

    propagation_note = (
        "MobileSAM has no cross-frame propagation; masking is independent per frame "
        "with no temporal consistency (§4, §5.5)"
        if not adapter.supports_temporal
        else (
            "provider supports video propagation (C19, best-effort); this run segmented per "
            f"frame — the driver passes one frame per call (window_frames={cfg.window_frames})"
        )
    )

    root = os.path.join(stage3_dir, "scenes")
    names = sorted(n for n in os.listdir(root) if os.path.isdir(os.path.join(root, n)))
    if scene_names:
        missing = sorted(set(scene_names) - set(names))
        if missing:
            raise UpstreamRefusal(f"requested scene(s) not present in Stage 3 output: {missing}")
        names = [n for n in names if n in scene_names]

    # Nothing above this line has written a byte, and that is the last refusal
    # this stage can decide for free. Any marker still standing describes the
    # PREVIOUS run; it comes down here, immediately before the first write (C16).
    clear_markers(out_dir)
    # What the preflight deliberately did not pay for twice: the weight build.
    # A failure HERE still exits 2 over a tree with no marker — the residue named
    # in preflight["not_covered"] and recorded in the manifest below.
    adapter.load()

    per_scene: list[dict] = []
    degraded = False
    # Run-level, not per-scene: the unload below is one call after the last
    # keyframe and has no scene to hang from, so its cause travels here and
    # main() reads it off the manifest when it writes the marker (C16, F2).
    run_causes: list[str] = []
    totals = {"n_keyframes": 0, "n_masks": 0, "n_kept": 0, "n_suppressed_cross_camera": 0, "n_empty_or_tiny": 0,
              # Run-level for the same reason, and counted so a release failure
              # is a number in the manifest and not only prose in the marker.
              "n_unload_failed": 0}

    for scene_name in names:
        rows = read_proposal_rows(os.path.join(root, scene_name, "proposals.jsonl"))
        by_keyframe: dict[str, list[dict]] = {}
        for row in rows:
            by_keyframe.setdefault(row["keyframe_token"], []).append(row)

        index_rows: list[dict] = []
        scene_totals = {"n_keyframes": 0, "n_masks": 0, "n_kept": 0, "n_suppressed_cross_camera": 0,
                        "n_empty_or_tiny": 0}

        for keyframe_token, keyframe_rows in sorted(by_keyframe.items()):
            keyframe_rows.sort(key=lambda r: r["channel"])
            masks, candidates, ledger = process_keyframe(
                keyframe_rows, adapter, substrate, cfg, paths.dataroot, camera_priority
            )
            mask_path = os.path.join(out_dir, "scenes", scene_name, "masks", f"{keyframe_token}.npz")
            write_masks_npz(mask_path, masks, bit_pack=cfg.bit_pack_masks)

            scene_token = keyframe_rows[0]["scene_token"]
            index_rows.append(
                {
                    "spec": STAGE_SPEC,
                    "keyframe_token": keyframe_token,
                    "scene_token": scene_token,
                    "t_ns": keyframe_rows[0]["t_ns"],
                    "time_base": keyframe_rows[0]["time_base"],
                    "coverage_config": keyframe_rows[0]["coverage_config"],
                    "mask_path": os.path.relpath(mask_path, out_dir),
                    "mask_size_px": [IMAGE_WIDTH_PX, IMAGE_HEIGHT_PX],
                    "bit_packed": cfg.bit_pack_masks,
                    "channels": sorted(masks),
                    "n_masks_per_channel": {ch: int(m.shape[0]) for ch, m in sorted(masks.items())},
                    "propagation": {
                        "propagated": False,
                        "state": None,
                        "window_frames": cfg.window_frames,
                        "capability_gap": propagation_note,
                    },
                    "ioa_nms": ledger,
                    "candidates": candidate_rows(keyframe_token, scene_token, candidates),
                }
            )
            scene_totals["n_keyframes"] += 1
            scene_totals["n_masks"] += sum(int(m.shape[0]) for m in masks.values())
            scene_totals["n_kept"] += ledger["n_kept"]
            scene_totals["n_suppressed_cross_camera"] += ledger["n_suppressed_cross_camera"]
            scene_totals["n_empty_or_tiny"] += ledger["n_empty_or_tiny_masks"]

        write_jsonl_atomic(os.path.join(out_dir, "scenes", scene_name, "masks.jsonl"), index_rows)
        summary = {
            "scene": scene_name,
            **scene_totals,
            "suppression_rate": round(
                scene_totals["n_suppressed_cross_camera"] / max(1, scene_totals["n_masks"]), 4
            ),
            # Zero kept masks across a scene means either the proposals were
            # empty or the contest ate everything; both are reportable.
            "degraded": scene_totals["n_kept"] == 0,
        }
        per_scene.append(summary)
        degraded = degraded or summary["degraded"]
        # Over scene_totals' keys, not totals': the run-level counters have no
        # per-scene counterpart to add in.
        for key in scene_totals:
            totals[key] += scene_totals[key]
        print(
            f"  {scene_name}  {scene_totals['n_keyframes']:>3} kf  {scene_totals['n_masks']:>5} masks  "
            f"{scene_totals['n_kept']:>5} kept  {scene_totals['n_suppressed_cross_camera']:>4} cross-cam "
            f"suppressed  {scene_totals['n_empty_or_tiny']:>4} empty"
            + ("  DEGRADED" if summary["degraded"] else "")
        )

    try:
        adapter.unload()
    except Exception as exc:  # noqa: BLE001 — a leaked resident is not a lost run
        # The manifest and the three-state marker are written by main() AFTER
        # run() returns, and main() catches only the refusals: an unguarded
        # failure here discarded a run that had segmented every keyframe, with
        # no manifest and no marker to show for it (F2). It is not hypothetical
        # on this provider — Sam31MultiplexAdapter.unload() raises RuntimeError
        # BY DESIGN when a close_session or shutdown() failed (C26 compensation
        # 6), i.e. exactly when the run most needs to say what is still held.
        # Releasing the model is the last thing this run needs from the device;
        # the failure is recorded as a degrading cause, surfaced verbatim, and
        # the run is DEGRADED, never lost. Same guard, same shape, as Stage 3b's
        # (C27) — the two stages behave identically on a failed RELEASE.
        totals["n_unload_failed"] += 1
        run_causes.append(
            f"mask adapter unload failed ({type(exc).__name__}: {exc}); device memory may still "
            "be held by this process"
        )
        degraded = True

    manifest = {
        "spec": STAGE_SPEC,
        "stage": STAGE,
        # Read by scripts/save_run_results.py:68 as s4.get("provider"); absent
        # here until C26, so every saved Results/*/run_config.json recorded
        # mask_2d.provider as null while the field it names existed all along.
        "provider": provider,
        "seed": cfg.global_seed,
        "config": cfg.as_dict(),
        "upstream": {
            "metadata_fingerprint": upstream["upstream"]["metadata_fingerprint"],
            "fingerprint_spec": upstream["upstream"]["fingerprint_spec"],
            "stage3_spec": upstream["spec"],
            "prompt_caption_sha256": upstream["prompt"]["caption_sha256"],
            # C16: a run built on accepted degradation says so in its provenance.
            "degraded": upstream_marker.degraded,
            "degraded_causes": list(upstream_marker.causes),
            "accepted_degraded_upstream": cfg.accept_degraded_upstream,
        },
        "paths": paths.as_dict(),
        # The adapter's stream hash of the weights that actually ran wins over
        # spec.sha256, which nothing on this path fills in (§7.2).
        "checkpoint": {
            "model_id": spec.model_id,
            "revision": spec.revision,
            "sha256": getattr(adapter, "checkpoint_sha256", None) or spec.sha256,
        },
        # Which refusals were decided before clear_markers, and what the preflight
        # left to adapter.load() — i.e. what an exit 2 can still cost (C16).
        "preflight": preflight,
        "vram_cap": adapter.vram_cap,  # C1 — synthetic ceiling, or the honest absence of one
        "image_size_px": [IMAGE_WIDTH_PX, IMAGE_HEIGHT_PX],
        "camera_priority": list(camera_priority),
        "ioa_nms": {
            "threshold": cfg.ioa_threshold,
            "space": "ego_frame_azimuth_elevation_rad",
            "rationale": (
                "Pixel-space IoA between two different image planes is not a geometric quantity; the "
                "footprints are compared in the ego frame the two cameras share (§1.1, §1.5 rule 6)"
            ),
            "source_box": "mask tight box, not the Stage 3 proposal box",
        },
        "capability_gaps": [
            (
                "no SAM 2.1-style mask propagation: masks are independent per frame (§4)"
                if not adapter.supports_temporal
                else "video propagation supported but not exercised: the driver segments per frame (C19)"
            ),
        ],
        "numpy_version": np.__version__,
        "python_version": sys.version.split()[0],
        "elapsed_s": round(time.time() - started, 2),
        "scenes": per_scene,
        "totals": totals,
        "run_causes": run_causes,
    }
    return manifest, EXIT_DEGRADED if degraded else EXIT_OK


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # Env defaults are the .env contract (C17): a declared key either has a
    # reader or is removed — these are the readers.
    parser.add_argument(
        "--paths", default=os.environ.get("DHAKASCENES_PATHS_CONFIG", "configs/paths.yaml")
    )
    parser.add_argument("--stage3-dir", default=None, help="default <work_root>/stage3_proposals")
    parser.add_argument("--out-dir", default=None, help="default <work_root>/stage4_masks")
    parser.add_argument(
        "--model-id",
        default=None,
        help="mask_2d model: hub id, or mobile_sam_vit_t; default facebook/sam3 (C19); "
             "facebook/sam3.1 selects the sam31_multiplex provider (C26)",
    )
    parser.add_argument(
        "--provider",
        default=None,
        choices=("mobile_sam", "sam2_video", "sam3_tracker", "sam31_multiplex"),
        help="adapter override; default inferred from --model-id (C19, C26)",
    )
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("MOBILE_SAM_CHECKPOINT"),
        help="MobileSAM weights file (mobile_sam provider only); default $MOBILE_SAM_CHECKPOINT",
    )
    parser.add_argument(
        "--revision", default=None,
        help="hub git sha, or an identifier for the MobileSAM weights file; required",
    )
    parser.add_argument("--scenes", nargs="*", default=None, help="subset of Stage 3 scene names")
    parser.add_argument("--seed", type=int, default=None, help="override the global seed (recorded)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--accept-degraded-upstream",
        action="store_true",
        help="consume a DEGRADED (complete, quality-flagged) Stage 3 output; recorded (C16)",
    )
    args = parser.parse_args(argv)

    try:
        paths = load_paths(args.paths)
    except PathValidationError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_REFUSED

    stage3_dir = args.stage3_dir or os.path.join(paths.work_root, "stage3_proposals")
    out_dir = args.out_dir or os.path.join(paths.work_root, STAGE)
    assert_dataroot_read_only(paths, out_dir)

    cfg = MaskConfig(
        device=args.device,
        checkpoint_path=args.checkpoint or "",
        revision=args.revision,
        accept_degraded_upstream=args.accept_degraded_upstream,
        **({"model_id": args.model_id} if args.model_id else {}),
        **({"provider": args.provider} if args.provider else {}),
        **({"global_seed": args.seed} if args.seed is not None else {}),
    )

    try:
        upstream, upstream_marker = load_upstream(
            stage3_dir, accept_degraded=cfg.accept_degraded_upstream
        )
        manifest, code = run(
            paths, upstream, upstream_marker, cfg, stage3_dir, out_dir, args.scenes
        )
    except UpstreamRefusal as exc:
        print(f"REFUSING TO START: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    except ModelUnavailable as exc:
        print(f"REFUSING TO START: model unavailable: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    except (MaskContractError, RoleContractError, PathValidationError) as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_REFUSED

    write_json_atomic(os.path.join(out_dir, "run_manifest.json"), manifest)
    # Three-state marker (§1.9, C16): clean / degraded-with-causes / absent.
    write_marker(
        out_dir,
        manifest["upstream"]["metadata_fingerprint"],
        degraded=code == EXIT_DEGRADED,
        # Run-level causes have no scene to hang from — an unload that failed
        # after the last keyframe is the whole reason such a run is degraded
        # (F2) — so the marker carries both lists or the degradation has no
        # stated cause.
        causes=[
            f"{s['scene']}: no masks survived (n_masks {s['n_masks']}, kept {s['n_kept']})"
            for s in manifest["scenes"]
            if s["degraded"]
        ]
        + list(manifest.get("run_causes") or []),
    )

    t = manifest["totals"]
    print(f"keyframes            : {t['n_keyframes']}")
    print(f"masks                : {t['n_masks']}  (all asserted at {IMAGE_WIDTH_PX}x{IMAGE_HEIGHT_PX})")
    print(f"kept after IoA-NMS   : {t['n_kept']}")
    print(f"cross-camera dupes   : {t['n_suppressed_cross_camera']}  (IoA > {cfg.ioa_threshold}, ego angular)")
    print(f"empty / tiny masks   : {t['n_empty_or_tiny']}")
    print(f"propagation          : {manifest['capability_gaps'][0]}")
    print(f"wrote {out_dir}")
    return code


register("mobile_sam", MASK_2D, MobileSamAdapter)
register("sam2_video", MASK_2D, TransformersSamAdapter)
register("sam3_tracker", MASK_2D, TransformersSamAdapter)
register("sam31_multiplex", MASK_2D, Sam31MultiplexAdapter)


if __name__ == "__main__":
    raise SystemExit(main())
