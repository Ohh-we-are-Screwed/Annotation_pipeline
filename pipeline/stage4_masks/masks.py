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

    python3 -m pipeline.stage4_masks.masks [--paths configs/paths.yaml]

Exit codes:
    0  every proposal produced a mask under contract
    1  ran, but at least one image was degraded (empty masks, or all suppressed)
    2  upstream contract broken, or the model is unavailable; nothing written
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pipeline.common.conventions import CAMERA, EGO, Transform  # noqa: E402
from pipeline.common.model_interfaces import (  # noqa: E402
    MASK_2D,
    CheckpointSpec,
    Mask2D,
    MaskResult,
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
EXIT_DEGRADED = 1  # ran, but at least one image produced no usable mask
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
                "C19: '' infers from model_id — facebook/sam3* -> sam3_tracker, mobile_sam* -> "
                "mobile_sam, anything else -> sam2_video; an explicit value wins"
            ),
            "ioa_threshold": "comprehensive.md §7.3.4, > 0.5; spec value, unvalidated on this substrate",
            "same_class_only": (
                "suppress only same-class duplicates: a pedestrian and the bus behind it share a "
                "bearing sector and are not duplicates of each other"
            ),
            "min_mask_px": "arbitrary; a mask below this cannot carry 5 LiDAR returns anyway",
            "multimask_output": "single mask per box: the box IS the disambiguation (§7.3.4)",
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
        return self._cfg.model_id.startswith("facebook/sam3")

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


def infer_mask_provider(model_id: str) -> str:
    """model_id -> registered provider name (C19). An explicit cfg.provider wins."""
    if model_id.startswith("facebook/sam3"):
        return "sam3_tracker"
    if model_id.startswith("mobile_sam"):
        return "mobile_sam"
    return "sam2_video"


_MASK_ADAPTERS: dict[str, type] = {
    "mobile_sam": MobileSamAdapter,
    "sam2_video": TransformersSamAdapter,
    "sam3_tracker": TransformersSamAdapter,
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
    # Any marker still standing describes the PREVIOUS run of this stage; it
    # comes down before the first write (C16).
    clear_markers(out_dir)
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
    adapter.load()
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

    per_scene: list[dict] = []
    degraded = False
    totals = {"n_keyframes": 0, "n_masks": 0, "n_kept": 0, "n_suppressed_cross_camera": 0, "n_empty_or_tiny": 0}

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
        for key in totals:
            totals[key] += scene_totals[key]
        print(
            f"  {scene_name}  {scene_totals['n_keyframes']:>3} kf  {scene_totals['n_masks']:>5} masks  "
            f"{scene_totals['n_kept']:>5} kept  {scene_totals['n_suppressed_cross_camera']:>4} cross-cam "
            f"suppressed  {scene_totals['n_empty_or_tiny']:>4} empty"
            + ("  DEGRADED" if summary["degraded"] else "")
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
            "stage3_spec": upstream["spec"],
            "prompt_caption_sha256": upstream["prompt"]["caption_sha256"],
            # C16: a run built on accepted degradation says so in its provenance.
            "degraded": upstream_marker.degraded,
            "degraded_causes": list(upstream_marker.causes),
            "accepted_degraded_upstream": cfg.accept_degraded_upstream,
        },
        "paths": paths.as_dict(),
        "checkpoint": {"model_id": spec.model_id, "revision": spec.revision, "sha256": spec.sha256},
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
        help="mask_2d model: hub id, or mobile_sam_vit_t; default facebook/sam2.1-hiera-large (C19)",
    )
    parser.add_argument(
        "--provider",
        default=None,
        choices=("mobile_sam", "sam2_video", "sam3_tracker"),
        help="adapter override; default inferred from --model-id (C19)",
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
        causes=[
            f"{s['scene']}: no masks survived (n_masks {s['n_masks']}, kept {s['n_kept']})"
            for s in manifest["scenes"]
            if s["degraded"]
        ],
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


if __name__ == "__main__":
    raise SystemExit(main())
