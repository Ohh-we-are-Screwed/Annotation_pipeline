#!/usr/bin/env python3
"""Stage 3b — 12 Hz identity propagation between Stage 3's 2 Hz keyframes (C27).

Stage 3 runs a detector on the 404 keyframes of v1.0-mini at 2 Hz. Between two
consecutive keyframes the substrate holds ~5 more frames of the same camera at
12 Hz, and nothing in the pipeline has ever looked at them. This stage does:
it prompts a SAM video tracker with every Stage 3 box at keyframe `t`, walks the
sweep frames to keyframe `t+1`, and asks what happened to each box on the way.

Two things come out of that walk:

  1. **Recovery.** An object the detector found at `t` and missed at `t+1` is
     re-emitted at `t+1` from the propagated mask's tight box, tagged
     `box_sources == "recovered"`, with a decayed score
     (`last_yolo_score * recovered_score_decay ** hops`) and the number of hops
     it has coasted. A recovered box is never presented as a detection: the
     provenance travels with it, in the row, per box.
  2. **Identity.** Every box in the output — detected or recovered — carries a
     `track_ids` entry that is stable across keyframes for one (scene, channel).
     Stage 7 associates in 3D and does not read this; it is here because the
     recovery decision needs it and because a 2D identity is what makes the
     12 Hz artifact auditable.

**12 Hz sweep frames are connective tissue for 2D identity only; no LiDAR or
eval quantity exists off-keyframe (pilot_plan §11 decision 5 is not reopened.)**
Every row this stage writes lands at a KEYFRAME, in Stage 3's exact
`proposals.jsonl` schema plus additive keys, in Stage 3's row order, in this
stage's own output directory. Stage 4 consumes it unchanged with
`--stage3-dir <this stage's out dir>`; nothing downstream needs to know Stage 3b
ran, which is the property that makes the A/B measurable at all.

**Two-phase residency, and why it is not one phase.** Phase A re-runs Stage 3's
OWN detector (identity read from Stage 3's run manifest, never re-chosen here)
on the sweep frames, so that a track born mid-gap has real evidence behind it
and a coasting track can be re-anchored before it drifts. Phase B loads the
video tracker. The two models are never resident together: Phase A unloads and
empties the CUDA cache before Phase B loads, because C1's 4 GiB synthetic
ceiling is the pilot's binding contract and two ~3 GB residents do not fit under
it. That is also why the window is bounded (Annotation_pipeline.md §7.3:
"cap sliding window to N = 16 frames; reinitialize attention states at block
transitions"), and why the registry is discarded at every scene boundary —
`reinit_at_block_boundary`, the vocabulary `TemporalWindow` already carries.

Window failures are isolated, never fatal: one window that raises is a BLOCK
BOUNDARY — its camera's live tracks are retired, that keyframe's boxes seed
fresh ones, the failure is counted, it is named in the manifest, and it degrades
the run. The stage never exits 2 after it has started writing, which is what the
Phase B PREFLIGHT is for: the video tracker is proved loadable, and Phase A's
detector rebuilt, BEFORE `clear_markers` and before the first byte, so an
unavailable model costs neither the previous run's marker nor 8 minutes of
sweep detection over 14k images.

    python3 -m pipeline.stage3b_track2d.track2d --revision <hub git sha>
    python3 -m pipeline.stage3b_track2d.track2d --self-test   # pure logic, no GPU

Exit codes:
    0  every window propagated under contract
    1  ran, but at least one window failed / anchor was missing / row hit the cap,
       or the upstream it consumed was DEGRADED — the marker carries the causes,
       this stage's own and the ones it inherited, and rc 1 keeps the exit code
       and the marker state saying the same thing (C16)
    2  upstream contract broken, the out dir holds another config's scenes, or a
       model failed its preflight — refused before the first write, and the
       previous run's marker is still standing
"""

from __future__ import annotations

import argparse
import dataclasses
import inspect
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pipeline.common.manifest import (  # noqa: E402
    MARKER_CLEAN,
    MARKER_DEGRADED,
    UpstreamRefusal,
    clear_markers,
    require_upstream,
    write_json_atomic,
    write_jsonl_atomic,
    write_marker,
)
from pipeline.common.model_interfaces import (  # noqa: E402
    MASK_2D,
    PROPOSAL_2D,
    CheckpointSpec,
    RoleContractError,
    apply_vram_cap,
)
from pipeline.common.paths import (  # noqa: E402
    PathValidationError,
    Paths,
    assert_dataroot_read_only,
    load_paths,
)
from pipeline.common.schemas import IMAGE_HEIGHT_PX, IMAGE_WIDTH_PX, RING_CAMERAS  # noqa: E402
from pipeline.common.sequences import (  # noqa: E402
    FrameRef,
    SequenceError,
    Window,
    camera_frame_chain,
    keyframe_windows,
)
from pipeline.stage0_data_probe.probe import Substrate  # noqa: E402
from pipeline.stage3_proposals.proposals import (  # noqa: E402
    YOLO_PROVIDER,
    ModelUnavailable,
    ProposalConfig,
    Yolo11Adapter,
    build_prompt_config,
    load_class_map,
    load_taxonomy,
    pairwise_iou,
)

STAGE = "stage3b_track2d"
STAGE_SPEC = "dhakascenes-pilot/stage3b_track2d/v1"
SWEEP_SPEC = "dhakascenes-pilot/stage3b_sweep_proposals/v1"
TRACKS12_SPEC = "dhakascenes-pilot/stage3b_tracks12hz/v1"

EXIT_OK = 0
EXIT_DEGRADED = 1  # ran, but at least one window / image / row was not clean
EXIT_REFUSED = 2  # refused BEFORE the first write: nothing written, previous marker intact


class Track2DContractError(RuntimeError):
    """A frame, a mask or a rewritten row violated the Stage 3b output contract."""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Track2DConfig:
    """Stage 3b tunables. None of these may appear as a literal in the code below."""

    # --- model (C19 family; C26 for the 3.1 line) ---
    model_id: str = "facebook/sam3"
    provider: str = ""  # "" == infer from model_id
    revision: str | None = None
    device: str = "cuda"
    video_storage_device: str = "cuda"

    # --- window (Annotation_pipeline.md §7.3, comprehensive.md §7.3.4) ---
    window_max_frames: int = 16

    # --- association ---
    match_iou: float = 0.30
    cross_class_suppress_iou: float = 0.90
    miss_tolerance_keyframes: int = 2
    recovered_score_decay: float = 0.7

    # --- recovered-box hygiene ---
    min_recovered_mask_px: int = 16
    min_recovered_box_side_px: float = 4.0
    max_proposals_per_image: int = 300
    max_active_tracks_per_camera: int = 64

    # --- recovered-box UPPER bounds: the tracker-leak gate (C27) ---
    max_recovered_area_ratio: float = 2.5
    max_recovered_frame_fraction: float = 0.25

    # --- box refinement: THE A/B flag ---
    refine_matched_boxes: bool = False
    refine_min_iou: float = 0.50
    refine_area_band: tuple[float, float] = (0.6, 1.4)

    # --- gated to a later revision ---
    reverse_pass: bool = False

    # --- Phase A: detection on the sweep frames ---
    detect_on_sweeps: bool = True
    sweep_birth_min_hits: int = 2

    # --- side artifact ---
    emit_12hz_tracks: bool = True

    # --- upstream gate (C16) ---
    accept_degraded_upstream: bool = False

    # --- 2D contract (§1.5) ---
    image_width_px: int = IMAGE_WIDTH_PX
    image_height_px: int = IMAGE_HEIGHT_PX

    # --- determinism (§1.9) ---
    global_seed: int = 20260812

    provenance: dict = field(
        default_factory=lambda: {
            "model_id": (
                "C19 family (human decision, 2026-08-13): facebook/sam3. Stage 3b loads ONLY the "
                "VIDEO tracker classes (Sam3TrackerVideoModel + Sam3TrackerVideoProcessor, ~3.4 GB "
                "fp32), never the image-tier classes Stage 4 uses — the two stages share a "
                "checkpoint family, not a resident model. facebook/sam3.1 is selectable per C26 "
                "and routes to the multiplex adapter"
            ),
            "provider": (
                "'' infers from model_id — facebook/sam3.1* -> sam31_multiplex, exactly "
                "'facebook/sam3' -> sam3_tracker_video, anything else REFUSES by name. Derived "
                "rather than defaulted: a checkpoint whose provider nobody established would be "
                "recorded in this manifest under another model's name"
            ),
            "revision": (
                "mandatory, enforced by CheckpointSpec.validate: an unpinned model_id silently "
                "tracks the hub's default branch, so the weights can change under a recorded "
                "measurement without one field in this manifest changing with them (§7.2)"
            ),
            "device": "the card this run used; recorded because a CPU fallback would be invisible otherwise",
            "video_storage_device": (
                "where the processed frames of one video session live. A window is 5-8 frames on "
                "this substrate, so keeping them on the inference device costs little and avoids a "
                "host round trip per propagated frame. Consumed ONLY by the transformers "
                "processor's init_video_session, i.e. by provider sam3_tracker_video: the SAM 3.1 "
                "multiplex predictor has no such knob and pipeline/stage4_masks owns that adapter, "
                "so the manifest records whether the resolved provider actually consumed this "
                "setting rather than describing a memory layout that never existed"
            ),
            "window_max_frames": (
                "Annotation_pipeline.md §7.3 / comprehensive.md §7.3.4: '<= 16 frames, reinitialize "
                "attention states at block transitions', the VRAM-containment rule for the "
                "prototype card. Measured on v1.0-mini 2026-08-19: every keyframe-to-keyframe "
                "window is 4-8 frames (2364 windows, mode 7), so the interior thinning path in "
                "sequences.keyframe_windows never fires here — the cap is a ceiling, not a policy"
            ),
            "match_iou": (
                "admissibility floor for the Hungarian assignment, same-class only. Deliberately "
                "BELOW Stage 3's same_class_iou (0.65): a propagated box that lands on top of a "
                "detected box of its own class must be MATCHED, because the match IS the "
                "deduplication — anything that matches is never injected, so this threshold is "
                "what keeps recovery from manufacturing near-duplicates. Untuned"
            ),
            "cross_class_suppress_iou": (
                "mirror of Stage 3's cross_class_iou (0.90) and high for the same reason: a "
                "recovered box that is near-identical to a DIFFERENT-class detected box is one "
                "object seen under two labels and is dropped, while a pedestrian standing in front "
                "of a bus (IoU ~0.02, IoA ~1.0) survives. IoU, never IoA"
            ),
            "miss_tolerance_keyframes": (
                "a track survives at most 2 consecutive unmatched keyframes = 1.0 s at 2 Hz, then "
                "dies. Identical to the maximum number of CONSECUTIVE recovered keyframes one "
                "track can contribute, which is the property that bounds how far a single "
                "detection can be stretched. Untuned"
            ),
            "recovered_score_decay": (
                "recovered score = last_yolo_score * 0.7 ** hops, so a box that has coasted twice "
                "carries 0.49 of its parent's confidence. Untuned and deliberately measured "
                "A/B rather than argued: the number decides where recovered boxes sit relative to "
                "Stage 3's per-class thresholds in every downstream ranking"
            ),
            "min_recovered_mask_px": (
                "Stage 4's min_mask_px in spirit and in value: below this a mask cannot carry the "
                ">= 5 single-sweep LiDAR returns Stage 9 gates on, so injecting the box would "
                "manufacture a proposal that cannot survive the lift"
            ),
            "min_recovered_box_side_px": (
                "Stage 3's min_box_side_px: a degenerate side means a box that cannot hold a mask, "
                "and a recovered box is held to the same floor as a detected one"
            ),
            "max_recovered_area_ratio": (
                "the two gates above are FLOORS; nothing bounded a propagated box from ABOVE, so a "
                "mask that leaked off its object was injected as a detection-shaped proposal "
                "carrying full class and score provenance. Measured on the 2026-08-19 trial, "
                "scene-0061/CAM_FRONT_LEFT track 0, a truck leaving frame right: the leaked mask's "
                "tight box was 3.07x its parent detection's area and covered the WHOLE 1600x900 "
                "frame, Stage 4 segmented it to 74% of the frame, and it suppressed a legitimate "
                "CAM_FRONT recovered truck at IoA 1.0. A propagated tight box more than 2.5x the "
                "area of its track's last DETECTED (yolo-sourced) box is a tracker leak, not an "
                "object that grew: 2.5x leaves room for genuine approach-toward-camera growth "
                "across one 0.5 s keyframe hop while rejecting the 3.07x and 3.80x leaks that "
                "trial actually produced. SHRINK is deliberately NOT bounded — 28 of that trial's 71 "
                "recovered boxes sit below 0.4x their parent and are objects leaving the frame or "
                "going behind something, which min_recovered_mask_px and min_recovered_box_side_px "
                "already bound. Untuned beyond that measurement; A/B-visible"
            ),
            "max_recovered_frame_fraction": (
                "a single RECOVERED instance covering more than a quarter of a 1600x900 frame is a "
                "leak by construction on this substrate, and the same trial says where the line "
                "falls: the largest legitimate recovered box measured is 21.7% of frame, and the "
                "next two are 61.9% and 100% — both the SAME leaked truck. 0.25 sits in that empty "
                "band. Stated honestly, this bound is not free: 4 of that scene's 273 DETECTED "
                "boxes are themselves over a quarter of the frame (largest 38.2%, a car in "
                "CAM_BACK_LEFT), so a legitimate recovery of one of those objects at its own "
                "detected size is refused by this bound and counted as a leak. That trade is taken "
                "deliberately — an object filling a third of a ring camera is metres away and is "
                "re-detected at nearly every keyframe, so it is the least likely to NEED recovery, "
                "while the failure it prevents is unbounded: Stage 4 segmented the measured "
                "full-frame leak to 74% of the frame across 64 deg x 39 deg, it suppressed a "
                "legitimate CAM_FRONT truck at IoA 1.0, and Stage 5 would have painted every LiDAR "
                "return in that cone as one vehicle.truck at 0.494. Absolute where "
                "max_recovered_area_ratio is relative, so the two are not one gate twice: the "
                "ratio catches a leak that grew out of a small parent, this catches one whose "
                "parent was "
                "already large — on the trial each of them catches a leak the other misses"
            ),
            "max_proposals_per_image": (
                "Stage 3's per-image cap, honoured AFTER injection so the two stages cannot "
                "disagree about how many proposals an image may carry. Overflow drops the LOWEST "
                "decayed score first and only ever drops recovered boxes — a detected box is "
                "Stage 3's output and is never deleted here. Counted, and degrades the run"
            ),
            "max_active_tracks_per_camera": (
                "Annotation_pipeline.md §7.3's object-multiplex bound, applied per camera. "
                "Overflow is resolved deterministically — keep the highest last_yolo_score, ties "
                "broken by the LOWEST track_id (birth order) — never by dict order, which would "
                "vary between runs. Drops are counted"
            ),
            "refine_matched_boxes": (
                "THE A/B flag, default OFF: when on, a detected box whose propagated mask agrees "
                "with it is replaced by the mask's tight box and the original is kept alongside. "
                "Off by default because it changes Stage 3's own numbers, which no recovery "
                "measurement should silently do"
            ),
            "refine_min_iou": (
                "refine only when the tracker and the detector already agree about the object: "
                "below this the tight box describes something else and replacing the detection "
                "with it is a relabel, not a refinement"
            ),
            "refine_area_band": (
                "tight/detected area ratio must land in [0.6, 1.4]; outside it the mask has either "
                "leaked into the background or collapsed onto a part of the object, and the "
                "detected box is kept. Untuned"
            ),
            "reverse_pass": (
                "v1 accepts the flag and REFUSES to run with it: a backward pass would recover "
                "objects that appear late in a gap, but it also rewrites the meaning of hops and "
                "of miss_tolerance, and neither has been measured. Gated to a later revision "
                "(C27) — refused loudly rather than silently ignored"
            ),
            "detect_on_sweeps": (
                "user decision: run Stage 3's OWN detector on the sweep frames (Phase A). Without "
                "it a track can only ever be seeded at a keyframe and never re-anchored mid-gap, "
                "so every recovery rests on one detection and no mid-gap birth is possible"
            ),
            "sweep_birth_min_hits": (
                "a track born between keyframes may only INJECT at a keyframe once it has been "
                "detected on >= 2 sweep frames. One sweep-frame detection is exactly the shape of "
                "a false positive, and injecting on it would turn detector noise into an annotated "
                "object with a provenance tag that says 'recovered'"
            ),
            "emit_12hz_tracks": (
                "user decision: write tracks_12hz.jsonl, one row per (sweep frame, live track). "
                "Nothing downstream reads it — it exists so a human can see what the propagation "
                "actually did between two keyframes instead of inferring it from the endpoints"
            ),
            "accept_degraded_upstream": (
                "C16 — consuming a DEGRADED (complete, quality-flagged) Stage 3 output is an "
                "explicit recorded decision, never a default"
            ),
            "image_width_px": "§1.5 rule 1: every 2D quantity is absolute pixels at 1600x900",
            "image_height_px": "§1.5 rule 1: every 2D quantity is absolute pixels at 1600x900",
            "global_seed": "§1.9, one global seed, recorded, threaded into torch",
        }
    )

    def as_dict(self) -> dict:
        out = dict(self.__dict__)
        # A tuple round-trips through JSON as a list; storing the list makes the
        # manifest byte-comparable against itself after a read-back.
        out["refine_area_band"] = list(self.refine_area_band)
        return out


# ---------------------------------------------------------------------------
# The adapters (the `MaskVideoTracker` surface, model_interfaces.py)
# ---------------------------------------------------------------------------


class Sam3VideoTrackerAdapter:
    """facebook/sam3's video tracker as Stage 3b's `MaskVideoTracker` (C19 family, C27).

    Loads ONLY `Sam3TrackerVideoModel` + `Sam3TrackerVideoProcessor`. Stage 4's
    image-tier classes are a different resident and are never touched here.

    Owns, and never leaks into stage code (§1.5 rule 2): the frames' transform
    into model space, the inverse of the masks back to 1600x900 through
    `post_process_masks`, and the `obj_id -> mask` pairing. The driver above
    speaks only in absolute pixels and integer track ids.
    """

    def __init__(self, spec: CheckpointSpec, cfg: Track2DConfig) -> None:
        self._spec = spec
        self._cfg = cfg
        self._device = cfg.device
        self._model: Any = None
        self._processor: Any = None
        self._torch: Any = None
        self._session_params: frozenset[str] = frozenset()
        # Which of the two presence sources the installed model actually gave us.
        # Recorded in the manifest so a run whose presence is a constant cannot be
        # mistaken for one whose presence is a measurement.
        self.presence_source: str = "unknown"
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
        return True

    def load(self) -> None:
        try:
            import torch
        except ImportError as exc:
            raise ModelUnavailable(f"torch is not installed: {exc}") from exc
        try:
            from transformers import Sam3TrackerVideoModel  # type: ignore
            from transformers import Sam3TrackerVideoProcessor  # type: ignore
        except ImportError as exc:
            raise ModelUnavailable(
                f"transformers does not provide the SAM 3 video-tracker classes ({exc}); they "
                "need the transformers 5.x line (C19 pins 5.15.0)"
            ) from exc
        self._torch = torch
        # C1: the synthetic 4 GiB ceiling, applied BEFORE the first allocation —
        # a cap applied after the model loaded caps nothing.
        self.vram_cap = apply_vram_cap(torch, self._device)
        try:
            self._processor = Sam3TrackerVideoProcessor.from_pretrained(
                self._cfg.model_id, revision=self._cfg.revision
            )
            # dtype is explicit because transformers 5.x from_pretrained defaults
            # to dtype='auto' (C19 note); our checkpoints store fp32.
            model = Sam3TrackerVideoModel.from_pretrained(
                self._cfg.model_id, revision=self._cfg.revision, dtype=torch.float32
            )
        except (OSError, ValueError) as exc:
            raise ModelUnavailable(
                f"could not load the SAM 3 video tracker {self._cfg.model_id!r} at revision "
                f"{self._cfg.revision!r}: {exc}. facebook/sam3 is a gated repo; the license grant "
                "is recorded in C19"
            ) from exc
        model.to(self._device).eval()
        self._model = model
        # §1.9: one seed, recorded, and set on the library that owns the RNG.
        torch.manual_seed(self._cfg.global_seed)
        # The session kwargs are read off the INSTALLED signature rather than
        # assumed from a docstring: this processor gained
        # max_vision_features_cache_size and dtype in the 5.x line, and calling
        # with a kwarg an older build lacks is a TypeError one frame into the run.
        self._session_params = frozenset(
            inspect.signature(self._processor.init_video_session).parameters
        )

    def unload(self) -> None:
        self._model = None
        self._processor = None
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

    # --- the MaskVideoTracker surface ---

    def init_video(self, frames: Sequence[np.ndarray]) -> Any:
        if self._processor is None:
            raise RuntimeError("adapter is not loaded")
        kwargs: dict[str, Any] = {
            "video": [np.asarray(frame) for frame in frames],
            "inference_device": self._device,
            "video_storage_device": self._cfg.video_storage_device,
        }
        if "dtype" in self._session_params:
            kwargs["dtype"] = self._torch.float32
        # max_vision_features_cache_size IS in the installed signature and is
        # deliberately left at the processor's own default of 1: caching a whole
        # window of Hiera features would spend the C1 ceiling on frames that are
        # each visited once per pass, and C27's config table has no knob for it.
        # The cost is paid only by the re-propagation after a mid-gap birth,
        # which re-encodes the frames from the birth frame onward.
        return self._processor.init_video_session(**kwargs)

    def add_video_boxes(
        self, session: Any, frame_idx: int, obj_ids: Sequence[int], boxes_xyxy_px: np.ndarray
    ) -> None:
        ids = [int(v) for v in obj_ids]
        boxes = np.asarray(boxes_xyxy_px, dtype=np.float64).reshape(-1, 4)
        if len(ids) != len(boxes):
            raise Track2DContractError(
                f"add_video_boxes: {len(ids)} obj_ids for {len(boxes)} boxes; the processor pairs "
                "them by position and a mismatch would prompt the wrong object"
            )
        # Nesting verified against the installed source
        # (processing_sam3_tracker_video.py: `input_boxes.shape[1] != len(obj_ids)`):
        # one batch element, one box per obj_id inside it.
        self._processor.add_inputs_to_inference_session(
            inference_session=session,
            frame_idx=int(frame_idx),
            obj_ids=ids,
            input_boxes=[[[float(v) for v in box] for box in boxes]],
        )

    def propagate_video(
        self,
        session: Any,
        *,
        start_frame_idx: int,
        max_frames: int | None = None,
        reverse: bool = False,
    ) -> Iterator[tuple[int, dict[int, np.ndarray], dict[int, float]]]:
        if self._model is None:
            raise RuntimeError("adapter is not loaded")
        torch = self._torch
        height_px, width_px = self._cfg.image_height_px, self._cfg.image_width_px
        with torch.inference_mode():
            for output in self._model.propagate_in_video_iterator(
                session,
                start_frame_idx=int(start_frame_idx),
                max_frame_num_to_track=max_frames,
                reverse=bool(reverse),
            ):
                # post_process_masks carries the model-resolution logits back to
                # 1600x900 (§1.5 rule 4). original_sizes is (height, width).
                post = self._processor.post_process_masks(
                    [output.pred_masks], original_sizes=[[height_px, width_px]]
                )[0]
                object_ids = [int(v) for v in (output.object_ids or ())]
                if len(post) != len(object_ids):
                    raise Track2DContractError(
                        f"frame {output.frame_idx}: {len(post)} masks for {len(object_ids)} object "
                        "ids; the pairing is positional inside output.object_ids and a mismatch "
                        "would hand every track its neighbour's mask"
                    )
                # `object_score_logits` is (num_objects,) in the installed
                # modeling source (Sam3TrackerVideoSegmentationOutput), aligned
                # with object_ids; sigmoid of it is the model's own statement
                # that the object is present in this frame. A build that does not
                # expose it yields a constant 1.0, and says so in the manifest.
                logits = getattr(output, "object_score_logits", None)
                scores = (
                    torch.sigmoid(logits.reshape(-1)).detach().cpu().numpy()
                    if logits is not None
                    else None
                )
                if scores is not None and scores.size != len(object_ids):
                    # Presence is a RECORD-only field, so a shape this driver
                    # cannot align is not worth failing a window over — but a
                    # mis-paired presence would be a fabricated measurement, so
                    # the constant is used and the manifest says exactly why.
                    self.presence_source = f"constant_1.0_logits_shape_mismatch:{tuple(logits.shape)}"
                    scores = None
                else:
                    self.presence_source = (
                        "object_score_logits_sigmoid" if scores is not None else "constant_1.0"
                    )
                masks: dict[int, np.ndarray] = {}
                presence: dict[int, float] = {}
                for position, obj_id in enumerate(object_ids):
                    mask_t = post[position]
                    while getattr(mask_t, "ndim", 2) > 2:  # (1, H, W) -> (H, W)
                        mask_t = mask_t[0]
                    mask = mask_t.detach().cpu().numpy().astype(bool)
                    if (mask.shape[1], mask.shape[0]) != (width_px, height_px):
                        raise Track2DContractError(
                            f"mask is {mask.shape[1]}x{mask.shape[0]}, not {width_px}x{height_px}; "
                            "carrying the masks back through post_process_masks is the adapter's "
                            "job (§1.5 rule 4)"
                        )
                    masks[obj_id] = mask
                    presence[obj_id] = float(scores[position]) if scores is not None else 1.0
                yield int(output.frame_idx), masks, presence

    def close_video(self, session: Any) -> None:
        # The window's `finally` drops the driver's reference; the session owns
        # device tensors, so the freed blocks are returned to the allocator here.
        del session
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()


def _sam31_factory(spec: CheckpointSpec, cfg: Track2DConfig) -> Any:
    """Build the SAM 3.1 multiplex adapter (C26) — imported LAZILY, on purpose.

    `Sam31MultiplexAdapter` lands with C26 step A4 and may not exist in this
    tree yet. A module-level import would make Stage 3b unimportable on the
    default provider because of a class the default provider never uses.
    """
    try:
        from pipeline.stage4_masks.masks import MaskConfig, Sam31MultiplexAdapter  # type: ignore
    except (ImportError, AttributeError) as exc:
        raise ModelUnavailable(
            f"sam31_multiplex adapter not available: {exc}; it lands with C26 step A4"
        ) from exc
    # Only the fields that describe the CHECKPOINT and the determinism contract
    # cross over; every Stage 4 tunable (IoA-NMS, bit packing, mask hygiene)
    # stays at its own default, because none of them describes this stage.
    mask_cfg = MaskConfig(
        model_id=cfg.model_id,
        provider="sam31_multiplex",
        revision=cfg.revision,
        device=cfg.device,
        global_seed=cfg.global_seed,
    )
    # `spec` is already role=MASK_2D (Stage 3b registers no role of its own), so
    # it is the mask_2d spec this adapter expects, unmodified.
    return Sam31MultiplexAdapter(spec, mask_cfg)


_TRACK_ADAPTERS: dict[str, Any] = {
    "sam3_tracker_video": Sam3VideoTrackerAdapter,
    "sam31_multiplex": _sam31_factory,
}

_PROVIDER_PROVENANCE: dict[str, str] = {
    "sam3_tracker_video": (
        "C19 family (human, 2026-08-13): facebook/sam3's video tracker, SA-V J&F 84.4; Stage 3b "
        "loads only its video classes (C27)"
    ),
    "sam31_multiplex": (
        "C26: SAM 3.1 multiplex, gated by scripts/smoke_sam31.py; selected here whenever model_id "
        "names the 3.1 line"
    ),
}


def infer_track_provider(model_id: str) -> str:
    """model_id -> adapter name. An explicit cfg.provider wins.

    Derived, never defaulted: recording `sam3_tracker_video` for a 3.1 checkpoint
    would put a false provider in the manifest that decided the numbers, and this
    manifest is the only place a reader can see which model actually ran.
    """
    if model_id.startswith("facebook/sam3.1"):
        return "sam31_multiplex"
    if model_id == "facebook/sam3":
        return "sam3_tracker_video"
    raise UpstreamRefusal(
        f"unknown video-tracker checkpoint {model_id!r}: no registered provider claims it (known: "
        f"{sorted(_TRACK_ADAPTERS)}). Pass --provider explicitly if this checkpoint really is one "
        "of them; a checkpoint whose provider nobody established would be recorded under another "
        "model's name"
    )


# The providers that actually CONSUME cfg.video_storage_device (F10): the kwarg
# is the transformers processor's init_video_session, and nothing else takes it.
# `_sam31_factory` builds a MaskConfig with no storage field and the SAM 3.1
# multiplex predictor has no equivalent knob — pipeline/stage4_masks owns that
# adapter and this stage does not get to add one — so on that provider the
# setting is inert and the manifest says so rather than describing a memory
# layout that never existed.
_STORAGE_DEVICE_CONSUMERS: frozenset[str] = frozenset({"sam3_tracker_video"})


def preflight_tracker(spec: CheckpointSpec, cfg: Track2DConfig, provider: str) -> dict:
    """Prove the Phase B tracker can load, BEFORE Phase A writes a byte (C27).

    Phase B is constructed and loaded ~8 minutes and ten scenes of
    sweep_proposals.jsonl into the run, with `clear_markers` long since past. A
    tracker that cannot load — the sam3 package missing, a gated download
    refused, a --revision that does not exist, a transformers build without the
    Sam3TrackerVideo classes — therefore used to burn Phase A, destroy the
    previous run's marker and exit 2 over a tree mixing this config's sweep files
    with the previous run's proposals.jsonl. Every one of those failures lives in
    the CHEAP half of an adapter's load(), so every one of them is provable here,
    while `rc 2 == nothing written` is still true.

    It is a preflight, not a load. The expensive half — 3.4 GB of weights through
    `from_pretrained`, or the multiplex predictor build — is deliberately not
    paid twice, and what that leaves uncovered is named in the returned block and
    in the manifest: weights that RESOLVE but cannot be BUILT still fail in Phase
    B and still exit 2, over a tree that then carries no marker at all — which is
    exactly the state §1.9's marker exists to make downstream refuse.

    Nothing stays resident: Phase A's detector loads next, and C1's ceiling is a
    serial-residency contract.
    """
    started = time.time()
    checks: list[str] = []
    # Construction first. For sam31_multiplex this IS the lazy import of
    # pipeline.stage4_masks.masks — the adapter lands with C26 step A4 and may
    # not exist in this tree — plus the MaskConfig plumbing; for both it is the
    # spec/role wiring that the driver would otherwise reach 8 minutes later.
    adapter = _TRACK_ADAPTERS[provider](spec, cfg)
    checks.append("adapter_constructed")
    checkpoint_sha256: str | None = None
    not_covered = ""

    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise ModelUnavailable(f"torch is not installed: {exc}") from exc
    checks.append("torch_import")

    if provider == "sam3_tracker_video":
        try:
            from transformers import Sam3TrackerVideoModel  # noqa: F401  # type: ignore
            from transformers import Sam3TrackerVideoProcessor  # type: ignore
        except ImportError as exc:
            raise ModelUnavailable(
                f"transformers does not provide the SAM 3 video-tracker classes ({exc}); they "
                "need the transformers 5.x line (C19 pins 5.15.0)"
            ) from exc
        checks.append("video_tracker_classes_import")
        # The PROCESSOR is this adapter's cheap half: a few kB of preprocessor
        # config, fetched from the same gated repo, at the same revision, under
        # the same auth as the 3.4 GB of weights that follow it. It is what turns
        # "the repo is gated / HF_TOKEN is unset / that revision does not exist"
        # from a Phase B traceback into a refusal. The weights are NOT fetched.
        try:
            Sam3TrackerVideoProcessor.from_pretrained(cfg.model_id, revision=cfg.revision)
        except Exception as exc:  # noqa: BLE001 — any resolution failure is unavailability
            raise ModelUnavailable(
                f"could not resolve the SAM 3 video tracker {cfg.model_id!r} at revision "
                f"{cfg.revision!r} ({type(exc).__name__}: {exc}). facebook/sam3 is a gated repo; "
                "the license grant is recorded in C19"
            ) from exc
        checks.append("processor_from_pretrained")
        not_covered = "Sam3TrackerVideoModel's ~3.4 GB of weights are downloaded and built in Phase B"
    elif provider == "sam31_multiplex":
        if not str(cfg.device).startswith("cuda"):
            # Mirrored from the adapter's own load(), which would raise it only
            # after Phase A had written; the point of the preflight is that it
            # raises here instead.
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
        # The adapter's OWN checkpoint resolution, called on the instance rather
        # than reimplemented here: hf_hub_download of the pinned
        # sam3.1_multiplex.pt is where a gated repo, a bad --revision or a missing
        # HF_TOKEN actually fails, and _resolve_checkpoint then stream-hashes the
        # file, REFUSES anything but the pinned digest, and records that verified
        # digest on `checkpoint_sha256` — the identity a manifest is quoted
        # against (§7.2). Private surface in a file this stage may not edit, so it
        # is PROBED, never assumed — a rename degrades this to hashing here, or to
        # the honest full load, rather than to a preflight that silently checks
        # less than it claims.
        #
        # Cost, stated rather than hidden: resolution stream-hashes the ~3.3 GB
        # file here and the adapter's load() hashes it again in Phase B, so the
        # run reads it twice (page-cached the second time) and downloads it at
        # most once. Hashing it a THIRD time here — which is what calling
        # _sha256_file on the path _resolve_checkpoint had just verified amounted
        # to — measured nothing that was not already decided, and is gone. What
        # remains is bought against the ~8 min of Phase A YOLO over ~14k sweep
        # images which this refusal now precedes instead of following.
        resolve = getattr(adapter, "_resolve_checkpoint", None)
        if callable(resolve):
            resolved = resolve()
            checks.append("checkpoint_resolved")
            not_covered = "the multiplex predictor itself is built in Phase B"
            # The recorded digest is the EVIDENCE that the pinned-bytes refusal
            # ran, so the check is claimed only when it is there; a resolution
            # that stopped recording it gets hashed here instead, and a manifest
            # never carries a digest this preflight did not establish.
            checkpoint_sha256 = getattr(adapter, "checkpoint_sha256", None)
            if checkpoint_sha256:
                checks.append("checkpoint_digest_verified")
            else:
                digest = getattr(adapter, "_sha256_file", None)
                checkpoint_sha256 = str(digest(resolved)) if callable(digest) else None
                checks.append(
                    "checkpoint_sha256" if checkpoint_sha256 else "checkpoint_sha256_unavailable"
                )
                if not checkpoint_sha256:
                    not_covered += (
                        "; API drift: this adapter records no checkpoint_sha256 and exposes no "
                        "_sha256_file, so this run's weights are not quoted against a digest"
                    )
        else:
            adapter.load()
            checks.append("full_load")
            checkpoint_sha256 = getattr(adapter, "checkpoint_sha256", None)
            adapter.unload()
            checks.append("unload")
    else:
        # A provider registered after this function was written: paying the whole
        # load and releasing it again is slower, never dishonest.
        adapter.load()
        checks.append("full_load")
        checkpoint_sha256 = getattr(adapter, "checkpoint_sha256", None)
        adapter.unload()
        checks.append("unload")

    del adapter  # C1: nothing is resident when Phase A's detector arrives
    return {
        "provider": provider,
        "checks": checks,
        "checkpoint_sha256": checkpoint_sha256,
        "weights_built": "full_load" in checks,
        "not_covered": not_covered,
        "elapsed_s": round(time.time() - started, 2),
    }


# ---------------------------------------------------------------------------
# Pure geometry and matching (no model, no substrate — the self-test drives these)
# ---------------------------------------------------------------------------


def tight_box_from_mask(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    """Tight xyxy box of a boolean mask, or None if the mask is empty.

    Same semantics as `stage4_masks.masks.mask_tight_box`, reimplemented here
    rather than imported: importing it would drag Stage 4's module-level
    `conventions`/`Transform` imports into a stage that owns no geometry.
    The `+1` on the far edge is the half-open convention Stage 3's boxes use —
    a one-pixel mask is a one-pixel-wide box, not a zero-width one.
    """
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


def hungarian_match(
    track_boxes: Sequence[Sequence[float]],
    track_classes: Sequence[str],
    det_boxes: Sequence[Sequence[float]],
    det_classes: Sequence[str],
    *,
    min_iou: float,
) -> tuple[list[tuple[int, int]], np.ndarray]:
    """Optimal one-to-one (track, detection) assignment on -IoU. Returns (pairs, iou).

    Admissible iff the two carry the SAME class name and IoU >= `min_iou`.
    Cross-class association is not a tuning question here: a track whose class
    changes mid-gap is two objects, and silently relabelling it would move a
    detection's class onto a different object with no record anywhere.

    Determinism (§1.9): rows are tracks in ascending track_id, columns are
    detections in row order, and an inadmissible cell costs +1.0 — strictly
    worse than any admissible cell, whose cost lies in [-1, -min_iou]. A finite
    sentinel rather than inf because `linear_sum_assignment` raises on a matrix
    it cannot cover. Ties are resolved by scipy's own deterministic traversal of
    that canonical ordering, so the same inputs give the same pairs every run.
    """
    pairs: list[tuple[int, int]] = []
    n_tracks, n_dets = len(track_boxes), len(det_boxes)
    if n_tracks == 0 or n_dets == 0:
        return pairs, np.zeros((n_tracks, n_dets), dtype=np.float64)
    iou = pairwise_iou(np.asarray(track_boxes, dtype=np.float64),
                       np.asarray(det_boxes, dtype=np.float64))
    same_class = np.array(
        [[track_classes[r] == det_classes[c] for c in range(n_dets)] for r in range(n_tracks)],
        dtype=bool,
    )
    admissible = same_class & (iou >= min_iou)
    cost = np.where(admissible, -iou, 1.0)
    rows, cols = linear_sum_assignment(cost)
    for r, c in zip(rows, cols):
        if admissible[r, c]:
            pairs.append((int(r), int(c)))
    pairs.sort()
    return pairs, iou


def _area(box: Sequence[float]) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))


def oversize_recovery_reason(
    box: Sequence[float], *, last_detected_area_px2: float, cfg: Track2DConfig
) -> str:
    """Empty if a propagated tight box is a plausible recovery, else the bound it broke.

    The one place the UPPER bounds live, because two callers ask the same
    question about the same box: `CameraTracker.step_keyframe`, which decides
    whether the box may be injected AND whether it may become the next window's
    prompt, and the 12 Hz artifact, which labels what the propagation did.

    Named-reason rather than a bool: "this mask leaked" and "this mask is
    implausible relative to the detection it descends from" are different
    findings, and a counter that cannot tell them apart cannot be read.

    Pure arithmetic on the values already in hand (§1.9): no clock, no RNG, no
    iteration order, so the verdict is the same on every run of the same inputs.
    """
    area = _area(box)
    frame_area = float(cfg.image_width_px) * float(cfg.image_height_px)
    # Frame fraction first: it is absolute, so it is the one bound that still
    # answers when the reference area is missing. Every birth and every detector
    # re-anchor sets that reference, so a zero here means a degenerate birth box
    # and nothing else — but the bound that needs no reference is the one that
    # should decide when there is none, and a ratio against 0 is not a number.
    if frame_area > 0.0:
        fraction = area / frame_area
        if fraction > cfg.max_recovered_frame_fraction:
            return f"frame_fraction={fraction:.4f}>{cfg.max_recovered_frame_fraction}"
    if last_detected_area_px2 > 0.0:
        ratio = area / last_detected_area_px2
        if ratio > cfg.max_recovered_area_ratio:
            return f"area_ratio={ratio:.3f}>{cfg.max_recovered_area_ratio}"
    return ""


# ---------------------------------------------------------------------------
# The records the driver moves around
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SweepDetection:
    """One Phase-A detection on one sweep frame.

    `phrase_char_span` and `nuscenes_categories` are carried in memory although
    the on-disk sweep row does not record them: a mid-gap-born track that later
    injects at a keyframe must write the same class fields Stage 3 writes, and
    re-deriving them at injection time from the keyframe's own rows would fail
    for exactly the class that is missing there.
    """

    box_xyxy_px: tuple[float, float, float, float]
    score: float
    class_name: str
    phrase_char_span: tuple[int, int]
    nuscenes_categories: tuple[str, ...]


@dataclass(frozen=True)
class MaskObservation:
    """What one propagated mask reduces to. The pixels are never kept.

    A window of 8 frames x 64 tracks of (900, 1600) bool masks is 737 MB held at
    once; nothing downstream of the yield needs a pixel, only the tight box, the
    area and the presence. Reducing at the yield keeps a window's footprint flat
    and makes the driver's state small enough to reason about.
    """

    box_xyxy_px: tuple[float, float, float, float]
    n_px: int
    presence: float


@dataclass
class Track:
    """One 2D identity inside ONE (scene, channel).

    `track_id` is unique per (scene, channel) only, and the emitted row carries
    `channel` and `scene_token` next to it — so the composable key is the triple,
    never the integer alone. Two cameras reusing id 0 for different objects is
    the expected state, not a collision.
    """

    track_id: int
    class_name: str
    phrase_span: list[int]
    nuscenes_categories: list[str]
    last_yolo_score: float
    evidence_hits: int
    misses: int
    seed_box: tuple[float, float, float, float]
    hops_since_evidence: int
    born_midgap: bool


COUNTER_KEYS: tuple[str, ...] = (
    "n_keyframes",
    "n_camera_rows",
    "n_yolo_boxes",
    "n_tracks",
    "n_recovered",
    "n_refined",
    "n_deaths",
    "n_deaths_boundary",
    "n_windows",
    "n_windows_skipped_empty",
    "n_windows_failed",
    "n_windows_missing_anchor",
    "n_sessions_unclosed",
    "n_lost_contest",
    "n_suppressed_cross_class",
    "n_rejected_min_hits",
    "n_recovered_recovered_suppressed",
    "n_below_min_px",
    "n_rejected_oversize",
    "n_sweep_images",
    "n_sweep_images_failed",
    "n_sweep_detections",
    "n_midgap_births",
    "n_midgap_injected",
    "n_repropagations",
    "n_12hz_rows",
    "n_row_cap_drops",
    "n_tracks_dropped_overflow",
    "n_camera_chains_failed",
    # Run-level, not per-scene: an unload is one call after the last scene, so it
    # is added to `totals` directly and has no scene to hang from.
    "n_unload_failed",
)

# Zero recoveries is a perfectly good outcome and never degrades a run. These
# are the counters that mean something did not happen the way it was supposed to.
# `n_deaths_boundary` is deliberately NOT here: a retired track is the CONSEQUENCE
# of a failure already counted by n_windows_failed / n_windows_missing_anchor /
# n_camera_chains_failed, and counting the consequence again would degrade a run
# twice for one defect.
# `n_rejected_oversize` is deliberately NOT here either, for the reason every
# other REJECTION counter (n_below_min_px, n_lost_contest, n_rejected_min_hits) is
# absent: it counts the gate WORKING. A leaked propagated mask is a quality event
# about one object on one keyframe, and the run's output is then exactly what the
# contract says it should be — the leak is out. Degrading on it would flag a
# complete, correct run for containing a defect it contained, and would arm
# --accept-degraded-upstream downstream for a reason no downstream stage can act
# on. It is counted per scene, totalled, and printed, which is what makes the
# leak rate an A/B-visible measurement instead of a run verdict.
DEGRADING_COUNTERS: tuple[str, ...] = (
    "n_windows_failed",
    "n_windows_missing_anchor",
    "n_sessions_unclosed",
    "n_row_cap_drops",
    "n_sweep_images_failed",
    "n_camera_chains_failed",
    "n_unload_failed",
)


def new_counters() -> dict[str, int]:
    return {key: 0 for key in COUNTER_KEYS}


# ---------------------------------------------------------------------------
# rewrite_row — the byte-stability gate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BoxAssignment:
    """What happened to Stage 3 box `j` at this keyframe."""

    track_id: int
    refined_box: list[float] | None = None  # not None == the tight box replaces the detected one


@dataclass(frozen=True)
class Injection:
    """A recovered box appended to a keyframe row."""

    track_id: int
    box_xyxy_px: list[float]
    score: float
    class_name: str
    nuscenes_categories: list[str]
    phrase_char_span: list[int]
    hops: int


def rewrite_row(
    original: dict,
    *,
    assignments: Sequence[BoxAssignment],
    injections: Sequence[Injection],
    window_frames: int,
    n_lost_contest: int = 0,
    n_suppressed_cross_class: int = 0,
) -> dict:
    """Stage 3 row + Stage 3b's additive keys. Never a rebuilt row.

    The existing values are COPIED, not recomputed: re-rounding a score or
    re-serialising a box through float() would make Stage 3b's output differ
    from Stage 3's in the last digit of every untouched number, and no reader
    could then tell a propagation change from a formatting change. A row with no
    refinement and no injection must round-trip json-identical on every original
    key — the self-test asserts exactly that.
    """
    row = dict(original)
    # Shallow copies: `row` shares Stage 3's list objects, so appending in place
    # would mutate the caller's parsed row as well.
    boxes = list(original["boxes_xyxy_px"])
    scores = list(original["scores"])
    class_names = list(original["class_names"])
    categories = list(original["nuscenes_categories"])
    spans = list(original["phrase_char_spans"])

    n_yolo = len(boxes)
    if len(assignments) != n_yolo:
        raise Track2DContractError(
            f"{original.get('sample_data_token')}: {len(assignments)} assignments for {n_yolo} "
            "Stage 3 boxes; track_ids is a parallel array and a mismatch would shift every id"
        )

    track_ids = [int(a.track_id) for a in assignments]
    box_sources = ["yolo"] * n_yolo
    hops = [0] * n_yolo
    refined = [a.refined_box is not None for a in assignments]
    boxes_original: list[list[float] | None] = [None] * n_yolo
    for j, assignment in enumerate(assignments):
        if assignment.refined_box is not None:
            boxes_original[j] = boxes[j]
            boxes[j] = assignment.refined_box

    for injection in injections:
        boxes.append(injection.box_xyxy_px)
        scores.append(injection.score)
        class_names.append(injection.class_name)
        categories.append(list(injection.nuscenes_categories))
        spans.append(list(injection.phrase_char_span))
        track_ids.append(int(injection.track_id))
        box_sources.append("recovered")
        hops.append(int(injection.hops))
        refined.append(False)
        boxes_original.append(None)

    row["boxes_xyxy_px"] = boxes
    row["scores"] = scores
    row["class_names"] = class_names
    row["nuscenes_categories"] = categories
    row["phrase_char_spans"] = spans
    row["n_proposals"] = len(boxes)
    row["track_ids"] = track_ids
    row["box_sources"] = box_sources
    row["n_propagated_hops"] = hops
    row["refined"] = refined
    row["boxes_xyxy_px_original"] = boxes_original
    row["track2d"] = {
        "spec": STAGE_SPEC,
        "n_yolo": n_yolo,
        "n_recovered": len(injections),
        "n_refined": sum(1 for a in assignments if a.refined_box is not None),
        "n_lost_contest": int(n_lost_contest),
        "n_suppressed_cross_class": int(n_suppressed_cross_class),
        "window_frames": int(window_frames),
    }
    return row


# ---------------------------------------------------------------------------
# The registry and the gates
# ---------------------------------------------------------------------------


class CameraTracker:
    """The per-(scene, channel) track registry. Pure bookkeeping: no image, no model.

    Every ordering in here is fixed by construction, because every one of them
    decides an output value: tracks are visited in ascending track_id, boxes in
    row order, injections are appended in ascending track_id, and overflow drops
    resolve on (score, track_id) rather than on dict order.
    """

    def __init__(self, cfg: Track2DConfig, counters: dict[str, int], *,
                 recovered_suppress_iou: float) -> None:
        self.cfg = cfg
        self.counters = counters
        # Stage 3's own same-class dedup IoU, read from the upstream manifest
        # rather than restated here: the recovered-vs-recovered contest is the
        # same question Stage 3 answered for detections, so it uses the same
        # number that run used.
        self.recovered_suppress_iou = recovered_suppress_iou
        self.tracks: dict[int, Track] = {}
        # A track that dies still needs its label for the frames it lived
        # through, so the id -> class map is never pruned.
        self.class_of: dict[int, str] = {}
        # The area of the last box the DETECTOR gave this identity — from its
        # birth, from a keyframe match, or from a Phase-A sweep match. It is the
        # reference `oversize_recovery_reason` measures a propagated box against,
        # and it is never pruned for class_of's reason: the 12 Hz artifact still
        # labels the frames a dead or evicted track lived through.
        self.detected_area_of: dict[int, float] = {}
        self.evicted: set[int] = set()
        self._next_id = 0

    # --- registry primitives ---

    def live(self) -> list[Track]:
        return [self.tracks[track_id] for track_id in sorted(self.tracks)]

    def _birth(
        self,
        *,
        class_name: str,
        phrase_span: Sequence[int],
        categories: Sequence[str],
        score: float,
        box: Sequence[float],
        born_midgap: bool,
    ) -> Track:
        track = Track(
            track_id=self._next_id,
            class_name=str(class_name),
            phrase_span=[int(v) for v in phrase_span],
            nuscenes_categories=[str(v) for v in categories],
            last_yolo_score=float(score),
            evidence_hits=1,
            misses=0,
            seed_box=tuple(float(v) for v in box),  # type: ignore[arg-type]
            hops_since_evidence=0,
            born_midgap=born_midgap,
        )
        self._next_id += 1
        self.tracks[track.track_id] = track
        self.class_of[track.track_id] = track.class_name
        # Every caller of _birth passes a DETECTED box — a keyframe box from
        # Stage 3's row, or a Phase-A sweep detection — so this is detector
        # evidence by construction.
        self.detected_area_of[track.track_id] = _area(track.seed_box)
        self.counters["n_tracks"] += 1
        return track

    def enforce_capacity(self) -> None:
        """Annotation_pipeline.md §7.3's multiplex bound, resolved deterministically."""
        cap = self.cfg.max_active_tracks_per_camera
        if len(self.tracks) <= cap:
            return
        ranked = sorted(self.tracks.values(), key=lambda t: (-t.last_yolo_score, t.track_id))
        for track in ranked[cap:]:
            del self.tracks[track.track_id]
            self.evicted.add(track.track_id)
            self.counters["n_tracks_dropped_overflow"] += 1

    def birth_row(self, row: dict) -> list[BoxAssignment]:
        """Every detected box in `row` births a track. Used at kf_0 and after a failure."""
        assignments: list[BoxAssignment] = []
        for j in range(len(row["boxes_xyxy_px"])):
            track = self._birth(
                class_name=row["class_names"][j],
                phrase_span=row["phrase_char_spans"][j],
                categories=row["nuscenes_categories"][j],
                score=row["scores"][j],
                box=row["boxes_xyxy_px"][j],
                born_midgap=False,
            )
            assignments.append(BoxAssignment(track_id=track.track_id))
        self.enforce_capacity()
        return assignments

    def retire_all(self) -> int:
        """A boundary nothing propagated across: every live track is RETIRED.

        A failed window and an anchor-less keyframe have the same shape — no
        propagation ran, so nothing carries an identity across them — and
        `reinit_at_block_boundary` (Annotation_pipeline.md §7.3) is the vocabulary
        this stage already uses for it. Merely penalizing the tracks, which is
        what this did, cost one DUPLICATE identity per object per boundary: the
        keyframe's own detections birth new tracks (there is nothing to match
        them against) while the old ones stay live coasting on a seed box that is
        now a keyframe stale, and the stale copy can still inject a phantom
        recovery at the NEXT keyframe next to the real detection.

        Counted under its own key, never under n_deaths: an ordinary death is a
        track that ran out of miss tolerance, which is a measurement about
        objects; this is a measurement about the run. `class_of` is not pruned —
        a retired track still needs its label for the frames it lived through.
        """
        n_retired = len(self.tracks)
        self.tracks.clear()
        self.counters["n_deaths_boundary"] += n_retired
        return n_retired

    def reap(self) -> None:
        for track in self.live():
            if track.misses > self.cfg.miss_tolerance_keyframes:
                del self.tracks[track.track_id]
                self.counters["n_deaths"] += 1

    # --- the interior step (sweep frames) ---

    def step_interior(
        self, dets: Sequence[SweepDetection], obs: dict[int, MaskObservation]
    ) -> tuple[list[Track], set[int]]:
        """Match Phase-A detections on one sweep frame; born-here objects are new tracks.

        Returns the tracks that must be prompted into the running session and the
        set of track ids that have detector evidence ON THIS FRAME (which is what
        the 12 Hz artifact labels `evidence: "yolo"`).
        """
        cfg = self.cfg
        candidates = [t for t in self.live() if t.track_id in obs]
        pairs, _ = hungarian_match(
            [obs[t.track_id].box_xyxy_px for t in candidates],
            [t.class_name for t in candidates],
            [d.box_xyxy_px for d in dets],
            [d.class_name for d in dets],
            min_iou=cfg.match_iou,
        )
        evidence_ids: set[int] = set()
        matched_cols: set[int] = set()
        for row_index, col_index in pairs:
            track = candidates[row_index]
            detection = dets[col_index]
            # Sweep evidence re-anchors the score and resets the decay, but does
            # NOT clear `misses`: misses count unmatched KEYFRAMES, which is what
            # miss_tolerance_keyframes is denominated in.
            track.last_yolo_score = detection.score
            # Sweep evidence re-anchors the SIZE reference too, for the reason it
            # re-anchors the score: this is the same detector that produced the
            # keyframe boxes, so a genuinely approaching object's reference stays
            # current and the leak bound does not slowly turn into a growth bound.
            self.detected_area_of[track.track_id] = _area(detection.box_xyxy_px)
            track.evidence_hits += 1
            track.hops_since_evidence = 0
            matched_cols.add(col_index)
            evidence_ids.add(track.track_id)

        born: list[Track] = []
        for col_index, detection in enumerate(dets):
            if col_index in matched_cols:
                continue
            track = self._birth(
                class_name=detection.class_name,
                phrase_span=detection.phrase_char_span,
                categories=detection.nuscenes_categories,
                score=detection.score,
                box=detection.box_xyxy_px,
                born_midgap=True,
            )
            born.append(track)
            self.counters["n_midgap_births"] += 1
        self.enforce_capacity()
        # A birth the capacity rule immediately evicted must not be prompted into
        # the session: it would consume tracker memory for an object no registry
        # entry describes.
        born = [t for t in born if t.track_id in self.tracks]
        evidence_ids.update(t.track_id for t in born)
        return born, evidence_ids

    # --- the keyframe step (the gates) ---

    def step_keyframe(
        self, row: dict, obs: dict[int, MaskObservation]
    ) -> tuple[list[BoxAssignment], list[Injection], int, int]:
        """Contest the propagated tracks against kf_{i+1}'s detections.

        Returns (assignments per detected box, injections, n_lost_contest,
        n_suppressed_cross_class) for this row.
        """
        cfg = self.cfg
        yolo_boxes = [[float(v) for v in box] for box in row["boxes_xyxy_px"]]
        yolo_classes = [str(name) for name in row["class_names"]]
        tracks = self.live()

        # An evidence box exists only where the mask is big enough to mean
        # something: below min_recovered_mask_px it cannot carry 5 LiDAR returns,
        # and below min_recovered_box_side_px it is a degenerate box either way.
        # The UPPER bounds are decided here too, but they do NOT remove the box
        # from `evidence`: a track whose mask leaked may still MATCH this
        # keyframe's detection of the same object, and dropping it here would
        # hand that object a second identity while the first coasted away. What
        # the verdict blocks is recovery and re-seeding, in (c) below.
        evidence: dict[int, tuple[float, float, float, float]] = {}
        oversize: dict[int, str] = {}  # track_id -> the upper bound it broke
        for track in tracks:
            observation = obs.get(track.track_id)
            if observation is None:
                continue
            box = observation.box_xyxy_px
            if (
                observation.n_px < cfg.min_recovered_mask_px
                or (box[2] - box[0]) < cfg.min_recovered_box_side_px
                or (box[3] - box[1]) < cfg.min_recovered_box_side_px
            ):
                self.counters["n_below_min_px"] += 1
                continue
            evidence[track.track_id] = box
            reason = oversize_recovery_reason(
                box,
                last_detected_area_px2=self.detected_area_of.get(track.track_id, 0.0),
                cfg=cfg,
            )
            if reason:
                oversize[track.track_id] = reason

        candidates = [t for t in tracks if t.track_id in evidence]
        pairs, _ = hungarian_match(
            [evidence[t.track_id] for t in candidates],
            [t.class_name for t in candidates],
            yolo_boxes,
            yolo_classes,
            min_iou=cfg.match_iou,
        )
        track_of_box: dict[int, Track] = {c: candidates[r] for r, c in pairs}
        matched_track_ids = {t.track_id for t in track_of_box.values()}

        # (a) MATCHED. The detection wins the box; the track keeps the identity.
        assignments: list[BoxAssignment | None] = [None] * len(yolo_boxes)
        for box_index, track in sorted(track_of_box.items()):
            yolo_box = yolo_boxes[box_index]
            tight = evidence[track.track_id]
            track.misses = 0
            track.hops_since_evidence = 0
            track.evidence_hits += 1
            track.last_yolo_score = float(row["scores"][box_index])
            self.detected_area_of[track.track_id] = _area(yolo_box)
            track.seed_box = tuple(yolo_box)  # type: ignore[assignment]
            refined_box: list[float] | None = None
            # The UPPER bounds are NOT re-applied on this path and no change is
            # needed here: refinement is already bounded from above by
            # refine_area_band's high edge (1.4), which is stricter than
            # max_recovered_area_ratio (2.5) and is measured against this
            # keyframe's own detected box rather than a remembered one. A leaked
            # mask lands far outside [0.6, 1.4] — the trial's leaks are 3.07x and
            # 3.80x — so the detected box is kept, which is exactly the outcome
            # the leak gate wants. The frame-fraction bound would add nothing
            # either: a tight box within 1.4x of its detection can only cover a
            # quarter of the frame if the DETECTION already did, and that is
            # Stage 3's output, not a propagation artifact.
            if cfg.refine_matched_boxes:
                iou = float(pairwise_iou(np.asarray([tight]), np.asarray([yolo_box]))[0, 0])
                yolo_area = _area(yolo_box)
                ratio = _area(tight) / yolo_area if yolo_area > 0.0 else 0.0
                low, high = cfg.refine_area_band
                if iou >= cfg.refine_min_iou and low <= ratio <= high:
                    # Rounded to Stage 3's own box precision so a refined row and
                    # an unrefined one differ only where the refinement happened.
                    refined_box = [round(float(v), 3) for v in tight]
                    self.counters["n_refined"] += 1
            assignments[box_index] = BoxAssignment(track_id=track.track_id, refined_box=refined_box)

        # (b) UNMATCHED DETECTION -> birth, in ascending box index so ids are
        # allocated in the row's own order.
        for box_index in range(len(yolo_boxes)):
            if assignments[box_index] is not None:
                continue
            track = self._birth(
                class_name=row["class_names"][box_index],
                phrase_span=row["phrase_char_spans"][box_index],
                categories=row["nuscenes_categories"][box_index],
                score=row["scores"][box_index],
                box=row["boxes_xyxy_px"][box_index],
                born_midgap=False,
            )
            assignments[box_index] = BoxAssignment(track_id=track.track_id)
        self.enforce_capacity()

        # (c) UNMATCHED TRACK WITH AN EVIDENCE BOX -> recovery candidate.
        n_lost_contest = 0
        n_cross_class = 0
        survivors: list[tuple[Track, tuple[float, float, float, float], float, int]] = []
        for track in candidates:
            if track.track_id in matched_track_ids:
                continue
            if track.track_id not in self.tracks:
                # Evicted by the capacity rule when this row's detections were
                # born: no registry entry describes it any more, so it does not
                # get to inject a box on the way out.
                continue
            tight = evidence[track.track_id]
            # The keyframe is unmatched, so the miss is paid FIRST and paid once,
            # whatever the gates below decide. Deferring it into the individual
            # branches is how a track that is rejected at every keyframe becomes
            # immortal: it would never reach miss_tolerance_keyframes and never
            # die. The propagated box also becomes the next window's prompt,
            # because it is the best knowledge of where this object is — unless
            # it broke an upper bound, immediately below.
            hops = track.hops_since_evidence + 1
            misses = track.misses + 1
            track.misses = misses
            track.hops_since_evidence = hops
            # OVERSIZE. Not one of the numbered injection gates below, and it is
            # ordered before them on purpose: those reject the CANDIDATE and
            # still adopt its propagated box as the next prompt, because the box
            # is real information about where the object is. This one rejects the
            # OBSERVATION — a mask that has leaked off its object says nothing
            # about that object's position — so the track is treated exactly as
            # "no usable evidence this keyframe" (branch (d)): it coasts, misses
            # and hops advance, and `seed_box` is LEFT WHERE IT WAS so the next
            # window re-prompts from the last trustworthy box.
            #
            # Not adopting the seed is the half that stops the CASCADE: on the
            # trial the leaked full-frame box at keyframe 25cd4f36 became the
            # PROMPT that produced [0, 0, 990, 900] at a5a93490 one hop later.
            # Gating only the injection would keep both of those out of the rows
            # and still leave every later window of this track prompted from a box
            # that is not the object — so the truck is never recovered again, and
            # a leak that drifted back under the bounds would then be injected as
            # a detection-shaped proposal for whatever the mask had wandered onto.
            if track.track_id in oversize:
                self.counters["n_rejected_oversize"] += 1
                continue
            track.seed_box = tight
            # (i) a mid-gap birth needs corroboration before it may become an
            # annotated object; one sweep-frame detection is the shape of an FP.
            if track.born_midgap and track.evidence_hits < cfg.sweep_birth_min_hits:
                self.counters["n_rejected_min_hits"] += 1
                continue
            # (ii) a track that dies of this miss (below, (e)) does not get to
            # inject a box on the way out.
            if misses > cfg.miss_tolerance_keyframes:
                continue
            if yolo_boxes:
                ious = pairwise_iou(np.asarray([tight]), np.asarray(yolo_boxes))[0]
                same = [ious[j] for j in range(len(yolo_boxes)) if yolo_classes[j] == track.class_name]
                other = [ious[j] for j in range(len(yolo_boxes)) if yolo_classes[j] != track.class_name]
            else:
                same, other = [], []
            # (iii) it overlaps a same-class detection but did not win the
            # assignment: another track owns that object. Injecting would emit a
            # duplicate under a second identity.
            if same and max(same) >= cfg.match_iou:
                n_lost_contest += 1
                self.counters["n_lost_contest"] += 1
                continue
            # (iv) near-identical to a DIFFERENT-class detection: one object seen
            # under two labels (Stage 3's cross_class_iou, same reasoning).
            if other and max(other) >= cfg.cross_class_suppress_iou:
                n_cross_class += 1
                self.counters["n_suppressed_cross_class"] += 1
                continue
            score = round(track.last_yolo_score * (cfg.recovered_score_decay ** hops), 5)
            survivors.append((track, tight, score, hops))

        # (v) recovered-vs-recovered: two tracks that have drifted onto the same
        # object would otherwise both inject. Greedy, highest decayed score
        # first, ties by lowest track_id.
        survivors.sort(key=lambda item: (-item[2], item[0].track_id))
        kept: list[tuple[Track, tuple[float, float, float, float], float, int]] = []
        for entry in survivors:
            track, tight, _score, _hops = entry
            clash = False
            for other_track, other_box, _s, _h in kept:
                if other_track.class_name != track.class_name:
                    continue
                if float(pairwise_iou(np.asarray([tight]), np.asarray([other_box]))[0, 0]) >= self.recovered_suppress_iou:
                    clash = True
                    break
            if clash:
                self.counters["n_recovered_recovered_suppressed"] += 1
                continue
            kept.append(entry)

        # (vi) Stage 3's per-image cap, honoured post-injection. Only RECOVERED
        # boxes are droppable: a detected box is Stage 3's output and deleting it
        # here would make this stage's row disagree with the stage it copies.
        allowed = max(0, cfg.max_proposals_per_image - len(yolo_boxes))
        if len(kept) > allowed:
            for _track, _tight, _score, _hops in kept[allowed:]:
                self.counters["n_row_cap_drops"] += 1
            kept = kept[:allowed]

        injections: list[Injection] = []
        for track, tight, score, hops in sorted(kept, key=lambda item: item[0].track_id):
            injections.append(
                Injection(
                    track_id=track.track_id,
                    box_xyxy_px=[round(float(v), 3) for v in tight],
                    score=score,
                    class_name=track.class_name,
                    nuscenes_categories=list(track.nuscenes_categories),
                    phrase_char_span=list(track.phrase_span),
                    hops=hops,
                )
            )
            # misses / hops / seed_box were already advanced above, for every
            # unmatched candidate alike — an injected box is not evidence.
            self.counters["n_recovered"] += 1
            if track.born_midgap:
                # The C27 gate question — how many injected boxes descend from a
                # MID-GAP birth rather than from a keyframe seed — is exactly the
                # false-positive channel sweep_birth_min_hits gates, and it needs
                # a field of its own to be answerable: n_midgap_births counts the
                # births, n_rejected_min_hits the ones the gate stopped, and
                # neither says how many got through into a written row.
                self.counters["n_midgap_injected"] += 1

        # (d) UNMATCHED TRACK WITHOUT AN EVIDENCE BOX -> coast. The seed box is
        # left where it was: an absent mask is no information about position.
        for track in tracks:
            if track.track_id in matched_track_ids or track.track_id in evidence:
                continue
            if track.track_id not in self.tracks:  # evicted by the capacity rule
                continue
            track.misses += 1
            track.hops_since_evidence += 1

        # (e) deaths.
        self.reap()
        return [a for a in assignments if a is not None], injections, n_lost_contest, n_cross_class


# ---------------------------------------------------------------------------
# The window loop
# ---------------------------------------------------------------------------


def _observe(masks: dict[int, np.ndarray], presence: dict[int, float]) -> dict[int, MaskObservation]:
    """One propagate_video yield -> tight boxes. An empty mask is simply absent."""
    out: dict[int, MaskObservation] = {}
    for obj_id in sorted(masks):
        mask = masks[obj_id]
        box = tight_box_from_mask(mask)
        if box is None:
            continue
        out[int(obj_id)] = MaskObservation(
            box_xyxy_px=box, n_px=int(np.count_nonzero(mask)),
            presence=float(presence.get(obj_id, 1.0)),
        )
    return out


def track_camera(
    *,
    cfg: Track2DConfig,
    adapter: Any,
    scene_name: str,
    scene_token: str,
    channel: str,
    kf_rows: Sequence[dict],
    windows: Sequence[Window],
    sweep_dets: dict[str, Sequence[SweepDetection]],
    load_frame: Any,
    recovered_suppress_iou: float,
    counters: dict[str, int],
    causes: list[str],
    tracks12: list[dict],
) -> list[dict]:
    """Propagate one camera of one scene. Returns the rewritten rows, in `kf_rows` order.

    The whole stage funnels through here, which is why it takes an adapter, a
    frame loader and a window list rather than a substrate: the self-test drives
    this exact function with a scripted tracker and no filesystem at all.
    """
    tracker = CameraTracker(cfg, counters, recovered_suppress_iou=recovered_suppress_iou)
    index_of_token = {row["sample_data_token"]: i for i, row in enumerate(kf_rows)}
    window_by_end: dict[int, Window] = {}
    for window in windows:
        end_index = index_of_token.get(window.end.sample_data_token)
        if end_index is not None:
            window_by_end[end_index] = window

    out_rows: list[dict] = []
    # WHICH frame tokens the 12 Hz artifact has already emitted, not merely
    # WHETHER some window emitted (F5). "Frame 0 belongs to the previous window's
    # last frame" holds only when that window actually emitted it: after a
    # skipped-empty or a failed window the next successful window used to start
    # at frame 1, and that boundary keyframe was then lost for good even though
    # the window propagated live tracks through it. The set keeps both halves of
    # the invariant — every frame appears, and none appears twice.
    emitted_tokens: set[str] = set()

    for row_index, row in enumerate(kf_rows):
        window = window_by_end.get(row_index)
        if window is None:
            # kf_0 is exactly this case — no window ENDS at the first anchor — and
            # so is any row whose sample_data_token fell off the chain (counted as
            # n_windows_missing_anchor). Nothing propagated across it, so no live
            # track may survive it: an anchor-less keyframe is a BLOCK BOUNDARY
            # (`reinit_at_block_boundary`), the registry is retired and every
            # detected box seeds a fresh identity. Leaving the tracks live gave
            # each object here a SECOND identity while the first kept coasting on
            # a stale seed box (F4). At kf_0 the registry is empty and this is a
            # no-op, which is why the two cases can share the branch.
            tracker.retire_all()
            out_rows.append(
                rewrite_row(row, assignments=tracker.birth_row(row), injections=[], window_frames=1)
            )
            continue

        counters["n_windows"] += 1
        n_frames = len(window.frames)
        interior_has_dets = any(
            len(sweep_dets.get(window.frames[k].sample_data_token, ())) > 0
            for k in range(1, n_frames - 1)
        )
        if not tracker.tracks and not interior_has_dets:
            # Nothing to carry and nothing to find: loading 7 frames into a video
            # session to propagate zero objects is the most expensive no-op in
            # the stage.
            counters["n_windows_skipped_empty"] += 1
            out_rows.append(
                rewrite_row(row, assignments=tracker.birth_row(row), injections=[],
                            window_frames=n_frames)
            )
            continue

        session: Any = None
        window_rows12: list[dict] = []
        # The frames this window is the first to cover; committed to
        # `emitted_tokens` only if the window survives (F5).
        window_tokens12: list[str] = []
        failed = False
        try:
            frames = [load_frame(ref) for ref in window.frames]
            for frame in frames:
                height_px, width_px = int(frame.shape[0]), int(frame.shape[1])
                if (width_px, height_px) != (cfg.image_width_px, cfg.image_height_px):
                    raise Track2DContractError(
                        f"{scene_name}/{channel}: frame is {width_px}x{height_px}; Stage 3b "
                        f"propagates at {cfg.image_width_px}x{cfg.image_height_px} (§1.5 rule 1)"
                    )
            session = adapter.init_video(frames)
            obs_by_frame: dict[int, dict[int, MaskObservation]] = {}
            yolo_evidence: dict[int, set[int]] = {}
            # obj_id -> the LAST frame index of this window for which a track the
            # capacity rule evicted may still emit a 12 Hz row. Eviction is a
            # decision taken at one frame and it must not rewrite the frames the
            # track already lived, was prompted and was observed through (F7).
            evicted_upto: dict[int, int] = {}
            evicted_seen: set[int] = set(tracker.evicted)

            live = tracker.live()
            if live:
                adapter.add_video_boxes(
                    session, 0, [t.track_id for t in live], [list(t.seed_box) for t in live]
                )
                for frame_idx, masks, presence in adapter.propagate_video(session, start_frame_idx=0):
                    obs_by_frame[int(frame_idx)] = _observe(masks, presence)
                # A track whose last evidence IS this keyframe carries detector
                # evidence at frame 0; one that arrived here coasting does not.
                yolo_evidence[0] = {t.track_id for t in live if t.hops_since_evidence == 0}

            for k in range(1, n_frames - 1):
                dets = sweep_dets.get(window.frames[k].sample_data_token, ())
                if not dets:
                    continue
                born, evidence_ids = tracker.step_interior(dets, obs_by_frame.get(k, {}))
                yolo_evidence.setdefault(k, set()).update(evidence_ids)
                # Evicted BY this frame's births: live and observed through frame
                # k, silent from k+1 on. sorted() only because §1.9 admits no
                # set-order iteration, even where the result cannot depend on it.
                for obj_id in sorted(tracker.evicted - evicted_seen):
                    evicted_upto[obj_id] = k
                evicted_seen |= tracker.evicted
                if not born:
                    continue
                new_ids = {t.track_id for t in born}
                adapter.add_video_boxes(
                    session, k, [t.track_id for t in born], [list(t.seed_box) for t in born]
                )
                for frame_idx, masks, presence in adapter.propagate_video(session, start_frame_idx=k):
                    slot = obs_by_frame.setdefault(int(frame_idx), {})
                    fresh = _observe({i: m for i, m in masks.items() if i in new_ids}, presence)
                    for obj_id, observation in fresh.items():
                        # Determinism over freshness: the second pass re-runs the
                        # already-tracked objects too, and their masks may differ
                        # now that the session carries another object's memory.
                        # The first pass's answer is the one that stands, so the
                        # result does not depend on how many births happened.
                        slot.setdefault(obj_id, observation)
                counters["n_repropagations"] += 1

            assignments, injections, n_lost, n_cross = tracker.step_keyframe(
                row, obs_by_frame.get(n_frames - 1, {})
            )
            # The window's END keyframe carries detector evidence too (F6). Every
            # detected box at kf_{i+1} produces an assignment, so a track that
            # appears BOTH in this frame's propagation and in the assignments was
            # confirmed by the detector here. Without this entry the artifact
            # labelled every window-closing keyframe "propagated", and a track
            # YOLO found at every single keyframe read as one the detector had
            # lost. (A track BORN at this keyframe was never prompted into the
            # session, so it has no observation here and cannot be labelled.)
            yolo_evidence[n_frames - 1] = {int(a.track_id) for a in assignments}
            for obj_id in sorted(tracker.evicted - evicted_seen):
                evicted_upto[obj_id] = n_frames - 1
            evicted_seen |= tracker.evicted
            out_rows.append(
                rewrite_row(
                    row,
                    assignments=assignments,
                    injections=injections,
                    window_frames=n_frames,
                    n_lost_contest=n_lost,
                    n_suppressed_cross_class=n_cross,
                )
            )

            if cfg.emit_12hz_tracks:
                # Every frame this window covers that no EARLIER window already
                # emitted. Normally that is frames 1..n-1, because frame 0 is the
                # previous window's last frame — but only when that window
                # actually emitted it (F5).
                for k in range(n_frames):
                    frame_ref = window.frames[k]
                    if frame_ref.sample_data_token in emitted_tokens:
                        continue
                    window_tokens12.append(frame_ref.sample_data_token)
                    for obj_id in sorted(obs_by_frame.get(k, {})):
                        if obj_id not in tracker.class_of:
                            continue
                        if obj_id in tracker.evicted and k > evicted_upto.get(obj_id, -1):
                            # Evicted at evicted_upto[obj_id] and silent after it;
                            # the frames before that are history and eviction does
                            # not rewrite history (F7). A track evicted in an
                            # EARLIER window has no entry here and is silent from
                            # frame 0 — it was never prompted into this session.
                            continue
                        observation = obs_by_frame[k][obj_id]
                        # A mask that broke the upper bounds is RECORDED, with the
                        # bound it broke, rather than dropped. Dropping it was the
                        # other option and it is the wrong one here: this artifact
                        # exists so a human can see what the propagation actually
                        # did between two keyframes, and deleting the leaked
                        # frames would hide the leak in the one file written to
                        # show it — while leaving a gap that reads exactly like a
                        # lost object. It is not recorded as a legitimate track
                        # position either: the field names the failing bound, so
                        # the row is greppable and no reader can mistake it for a
                        # measured position. Nothing downstream consumes this file
                        # (C27), so the flag costs a key and no contract.
                        #
                        # On the window's CLOSING keyframe this verdict is exactly
                        # the one step_keyframe acted on: an unmatched track's
                        # reference area is not touched there, so the same box is
                        # measured against the same reference. On the interior
                        # frames it is the best available reading — the identity's
                        # most recent detector evidence — and it is advisory:
                        # step_keyframe, not this line, decided injection and
                        # re-seeding.
                        oversize = oversize_recovery_reason(
                            observation.box_xyxy_px,
                            last_detected_area_px2=tracker.detected_area_of.get(obj_id, 0.0),
                            cfg=cfg,
                        )
                        window_rows12.append(
                            {
                                "spec": TRACKS12_SPEC,
                                "scene_token": scene_token,
                                "channel": channel,
                                "sample_data_token": frame_ref.sample_data_token,
                                "is_key_frame": bool(frame_ref.is_key_frame),
                                "timestamp_us": int(frame_ref.timestamp_us),
                                "track_id": int(obj_id),
                                "box_xyxy_px": [round(float(v), 2) for v in observation.box_xyxy_px],
                                "class_name": tracker.class_of[obj_id],
                                "presence": round(float(observation.presence), 4),
                                "evidence": "yolo" if obj_id in yolo_evidence.get(k, ()) else "propagated",
                                # "" == within the upper bounds; otherwise the
                                # bound this mask broke (C27 leak gate).
                                "oversize": oversize,
                            }
                        )
        except UpstreamRefusal:
            raise
        except Exception as exc:  # noqa: BLE001 — window isolation is the point
            failed = True
            counters["n_windows_failed"] += 1
            causes.append(
                f"{scene_name}/{channel}: window ending {row['sample_data_token'][:8]} failed "
                f"({type(exc).__name__}: {exc})"
            )
            # The window is abandoned, not the camera — but its propagation did
            # not run, so nothing carried an identity across it. A failed window
            # is a BLOCK BOUNDARY (`reinit_at_block_boundary`): the live tracks
            # are RETIRED, the row is emitted with pure Stage 3 content, and the
            # detected boxes birth fresh identities so tracking resumes at the
            # next window. Penalizing them instead left every live track in the
            # registry on a STALE seed box while this keyframe's detections
            # birthed duplicates of the very same objects (F4).
            tracker.retire_all()
            out_rows.append(
                rewrite_row(row, assignments=tracker.birth_row(row), injections=[],
                            window_frames=n_frames)
            )
        finally:
            if session is not None:
                try:
                    adapter.close_video(session)
                except Exception as exc:  # noqa: BLE001
                    # A session that will not close has leaked device memory. That
                    # is NOT this window's propagation failing, and the window is
                    # no longer marked failed while its output is kept — a state
                    # nothing could act on (F8). The evidence for treating it as
                    # benign is in the adapter above: every mask this window used
                    # was already carried back through post_process_masks and
                    # .detach().cpu().numpy(), which is a hard device sync PER
                    # FRAME, so a fault that only surfaces at close_video's
                    # empty_cache() cannot have corrupted rows that were read back
                    # cleanly before it. And a CUDA context fault is STICKY: if the
                    # device really is poisoned, the next window's init_video
                    # raises and is counted as the window failure it is. Its own
                    # counter, and it degrades the run.
                    counters["n_sessions_unclosed"] += 1
                    causes.append(
                        f"{scene_name}/{channel}: close_video after "
                        f"{row['sample_data_token'][:8]} failed ({type(exc).__name__}: {exc}); the "
                        "window's output stands, this session's device memory is leaked"
                    )

        if not failed:
            emitted_tokens.update(window_tokens12)
            tracks12.extend(window_rows12)
            counters["n_12hz_rows"] += len(window_rows12)

    return out_rows


# ---------------------------------------------------------------------------
# Phase A — Stage 3's detector on the sweep frames
# ---------------------------------------------------------------------------


def rebuild_proposal_config(upstream: dict, cfg: Track2DConfig) -> ProposalConfig:
    """Stage 3's ProposalConfig, read back out of Stage 3's own run manifest.

    Not `ProposalConfig()` with a few overrides: the thresholds, the class map,
    the NMS IoU and the letterbox size are what decided which boxes exist on the
    keyframes, and a sweep detection produced under different ones would be a
    different detector wearing the same checkpoint's name. Every field Stage 3
    echoes is taken; anything a future Stage 3 stops echoing falls back to that
    field's default and the difference is visible in this manifest's config echo.
    """
    echoed = dict(upstream.get("config") or {})
    known = {f.name for f in dataclasses.fields(ProposalConfig)}
    kwargs = {k: v for k, v in echoed.items() if k in known}
    thresholds = kwargs.get("thresholds")
    if isinstance(thresholds, dict):
        # as_dict() flattens the tuple-of-pairs into a mapping; the constructor
        # wants the pairs back, sorted so the rebuild is order-independent.
        kwargs["thresholds"] = tuple(sorted((str(k), float(v)) for k, v in thresholds.items()))
    checkpoint = dict(upstream.get("checkpoint") or {})
    if checkpoint.get("model_id"):
        kwargs["model_id"] = checkpoint["model_id"]
        kwargs["revision"] = checkpoint.get("revision")
    # The detector's IDENTITY is Stage 3's; the DEVICE and the seed are this
    # run's, because they describe the machine and the RNG in front of us.
    kwargs["device"] = cfg.device
    kwargs["global_seed"] = cfg.global_seed
    return ProposalConfig(**kwargs)


def build_sweep_detector(upstream: dict, cfg: Track2DConfig) -> tuple[Any, Any, dict, Any]:
    """Reconstruct Stage 3's detector. Returns (adapter, prompt, checkpoint block, taxonomy)."""
    provider = upstream.get("provider")
    if provider != YOLO_PROVIDER:
        raise UpstreamRefusal(
            f"Stage 3 ran under provider {provider!r}, not {YOLO_PROVIDER!r}. Phase A re-runs "
            "STAGE 3's detector on the sweep frames and has no second choice: a caption-family "
            "provider needs a caption, a token-span map and a different score semantics, and "
            "mixing two detectors' boxes inside one track would make the recovery measurement "
            "meaningless. Re-run with --no-sweep-detection, or re-run Stage 3 with YOLO"
        )
    proposal_cfg = rebuild_proposal_config(upstream, cfg)
    taxonomy_block = dict(upstream.get("taxonomy") or {})
    class_map_block = dict(upstream.get("class_map") or {})
    taxonomy_path = taxonomy_block.get("path") or ""
    if not taxonomy_path:
        raise UpstreamRefusal(
            "Stage 3's manifest records no taxonomy path; Phase A cannot rebuild the class space "
            "its boxes live in. Re-run with --no-sweep-detection"
        )
    taxonomy = load_taxonomy(taxonomy_path)
    if taxonomy_block.get("sha256") and taxonomy.sha256 != taxonomy_block["sha256"]:
        raise UpstreamRefusal(
            f"{taxonomy_path} has changed since Stage 3 ran ({taxonomy.sha256[:12]} != "
            f"{str(taxonomy_block['sha256'])[:12]}): the sweep detections would carry phrases and "
            "character spans from a different class space than the keyframe rows they are matched "
            "against. Restore the file, re-run Stage 3, or re-run with --no-sweep-detection"
        )
    class_map_path = class_map_block.get("path") or proposal_cfg.class_map_path
    class_map = load_class_map(class_map_path, taxonomy)
    if class_map_block.get("sha256") and class_map.sha256 != class_map_block["sha256"]:
        raise UpstreamRefusal(
            f"{class_map_path} has changed since Stage 3 ran ({class_map.sha256[:12]} != "
            f"{str(class_map_block['sha256'])[:12]}): the source class -> phrase bridge is not the "
            "one that produced the keyframe boxes. Re-run with --no-sweep-detection"
        )
    prompt = build_prompt_config(taxonomy, proposal_cfg)
    checkpoint = dict(upstream.get("checkpoint") or {})
    spec = CheckpointSpec(
        role=PROPOSAL_2D,
        provider=provider,
        model_id=checkpoint.get("model_id") or proposal_cfg.model_id,
        revision=checkpoint.get("revision"),
        sha256=checkpoint.get("sha256"),
        provenance=(
            "C27: Stage 3b's sweep detector IS Stage 3's, identity read from that run's manifest "
            "(never re-chosen here) so a recovered box and a detected box descend from the same "
            "weights"
        ),
    )
    errors = spec.validate(prefix="sweep detector: ")
    if errors:
        raise UpstreamRefusal(
            "; ".join(errors) + " — Stage 3's manifest does not carry a usable checkpoint identity"
        )
    adapter = Yolo11Adapter(spec, prompt, proposal_cfg, class_map)
    adapter.load()
    checkpoint = {
        "model_id": spec.model_id,
        "revision": spec.revision,
        "sha256": spec.sha256,
        "provider": provider,
        "class_map_path": class_map.path,
        "class_map_sha256": class_map.sha256,
        "taxonomy_sha256": taxonomy.sha256,
    }
    # The taxonomy travels back with it: a sweep detection needs the same
    # phrase -> nuScenes categories map the keyframe rows were written with, and
    # loading the file a second time would re-open the door the sha check closed.
    return adapter, prompt, checkpoint, taxonomy


def detect_on_sweep_frames(
    *,
    adapter: Any,
    prompt: Any,
    taxonomy_phrase_to_categories: dict,
    cfg: Track2DConfig,
    dataroot: str,
    chains: dict[tuple[str, str], list[FrameRef]],
    scene_tokens: dict[str, str],
    scene_names: Sequence[str],
    out_dir: str,
    checkpoint: dict,
    counters_by_scene: dict[str, dict[str, int]],
    causes_by_scene: dict[str, list[str]],
) -> dict[str, list[SweepDetection]]:
    """Run the rebuilt detector on every NON-keyframe frame; write sweep_proposals.jsonl.

    Keyframe detections are never duplicated here: they exist in Stage 3's rows,
    and a second copy under a second spec would be two answers to one question.
    """
    from PIL import Image  # local: the only place Phase A touches imagery

    detections: dict[str, list[SweepDetection]] = {}
    for scene_name in scene_names:
        scene_token = scene_tokens[scene_name]
        counters = counters_by_scene[scene_name]
        causes = causes_by_scene[scene_name]
        rows: list[dict] = []
        # Channel-major in RING_CAMERAS order, then chain order: a deterministic
        # file order that does not depend on which camera happened to be read first.
        for channel in RING_CAMERAS:
            chain = chains.get((scene_token, channel))
            if chain is None:
                continue
            for frame in chain:
                if frame.is_key_frame:
                    continue
                counters["n_sweep_images"] += 1
                boxes: list[list[float]] = []
                scores: list[float] = []
                class_names: list[str] = []
                frame_dets: list[SweepDetection] = []
                try:
                    with Image.open(os.path.join(dataroot, frame.filename)) as handle:
                        image = np.asarray(handle.convert("RGB"), dtype=np.uint8)
                    height_px, width_px = int(image.shape[0]), int(image.shape[1])
                    if (width_px, height_px) != (cfg.image_width_px, cfg.image_height_px):
                        raise Track2DContractError(
                            f"{frame.filename} is {width_px}x{height_px}; Stage 3b detects at "
                            f"{cfg.image_width_px}x{cfg.image_height_px} (§1.5 rule 1)"
                        )
                    # propose() already applies the per-class thresholds, the
                    # clipping, the degenerate-box filter, the two-threshold
                    # dedup and the per-image cap (finalize_proposals). Nothing
                    # of that is re-implemented here — replicating it would give
                    # sweep boxes a different hygiene than keyframe boxes.
                    proposals = adapter.propose(image, prompt, channel=channel)
                    for index in range(len(proposals)):
                        box = [round(float(v), 3) for v in proposals.boxes_xyxy_px[index]]
                        score = round(float(proposals.scores[index]), 5)
                        name = str(proposals.class_names[index])
                        span = tuple(int(v) for v in proposals.phrase_spans[index])
                        boxes.append(box)
                        scores.append(score)
                        class_names.append(name)
                        frame_dets.append(
                            SweepDetection(
                                box_xyxy_px=(box[0], box[1], box[2], box[3]),
                                score=score,
                                class_name=name,
                                phrase_char_span=(span[0], span[1]),
                                nuscenes_categories=tuple(taxonomy_phrase_to_categories[name]),
                            )
                        )
                except Exception as exc:  # noqa: BLE001 — one image never aborts the stage
                    counters["n_sweep_images_failed"] += 1
                    causes.append(
                        f"{scene_name}/{channel}: sweep frame {frame.sample_data_token[:8]} "
                        f"failed ({type(exc).__name__}: {exc}); counted as zero detections"
                    )
                    boxes, scores, class_names, frame_dets = [], [], [], []
                counters["n_sweep_detections"] += len(frame_dets)
                if frame_dets:
                    detections[frame.sample_data_token] = frame_dets
                rows.append(
                    {
                        "spec": SWEEP_SPEC,
                        "scene_token": scene_token,
                        "channel": channel,
                        "sample_data_token": frame.sample_data_token,
                        "filename": frame.filename,
                        "timestamp_us": int(frame.timestamp_us),
                        "is_key_frame": False,
                        "boxes_xyxy_px": boxes,
                        "scores": scores,
                        "class_names": class_names,
                        "n_proposals": len(boxes),
                        "checkpoint": {
                            "model_id": checkpoint["model_id"],
                            "revision": checkpoint["revision"],
                            "sha256": checkpoint["sha256"],
                        },
                        "seed": cfg.global_seed,
                    }
                )
        write_jsonl_atomic(os.path.join(out_dir, "scenes", scene_name, "sweep_proposals.jsonl"), rows)
        print(
            f"  [A] {scene_name}  {counters['n_sweep_images']:>5} sweep img  "
            f"{counters['n_sweep_detections']:>5} det  {counters['n_sweep_images_failed']:>3} failed"
        )
    return detections


# ---------------------------------------------------------------------------
# Upstream binding
# ---------------------------------------------------------------------------


def read_proposal_rows(path: str) -> list[dict]:
    if not os.path.isfile(path):
        raise UpstreamRefusal(
            f"{path} not found; run `python3 -m pipeline.stage3_proposals.proposals` first"
        )
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


# Prefixed, so a reader of THIS stage's marker can tell a cause Stage 3b
# generated from one it is carrying forward. The two must stay distinguishable:
# "the window ending 25cd4f36 failed" is actionable here, "2 scenes over the
# sector-rejection threshold" is actionable four stages upstream.
INHERITED_CAUSE_PREFIX = "inherited from Stage 3: "


def inherited_degradation_causes(upstream_marker) -> list[str]:
    """Stage 3's degradation, re-stated as THIS stage's run-level causes (C16).

    Stage 3b sits BETWEEN Stage 3 and Stage 4, and Stage 4 sources
    `upstream.degraded` / `degraded_causes` from its IMMEDIATE upstream marker
    (`pipeline.common.manifest.require_upstream`, called from masks.py). So a
    Stage 3b that consumed a DEGRADED Stage 3 under --accept-degraded-upstream
    and then wrote a plain `_SUCCESS` LAUNDERED the chain: the flag C16 exists to
    carry, and the causes that make it auditable, stopped at this stage, and
    every stage after it — and every Results/ run_config quoting them — read as
    built on a clean substrate. Reproduced on the 2026-08-19 trial, whose own
    manifest records `upstream.degraded=true` with all 10 causes next to a clean
    `_SUCCESS`.

    A degraded upstream therefore degrades THIS run: `_SUCCESS.degraded`, causes
    attributed and carried forward, exit 1. No new marker state is invented —
    §1.9's three states already say "complete, quality-flagged, carries the
    causes", which is precisely what this output is. Coupling it to the exit code
    is not decoration either: `scripts/run_stages.sh` cross-examines the two, and
    a stage that exits 0 over a degraded marker is aborted there as BROKEN.

    Empty iff the upstream was clean, so the caller can use the emptiness as the
    decision. A degraded marker that names no cause cannot come out of
    `write_marker` (it refuses one), but it can be read off a hand-written file,
    and a `causes=[]` would make `write_marker` raise AFTER a complete run — the
    F2 failure mode — so it is given a cause of its own instead.
    """
    if not upstream_marker.degraded:
        return []
    causes = [f"{INHERITED_CAUSE_PREFIX}{cause}" for cause in upstream_marker.causes]
    return causes or [
        f"{INHERITED_CAUSE_PREFIX}the Stage 3 marker is degraded and names no cause"
    ]


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
        # A Stage 3 resolution fallback is legal and recorded (§13.1); propagating
        # its boxes at a different resolution is not.
        raise UpstreamRefusal(
            f"Stage 3 ran at {resolution}, Stage 3b runs at {[IMAGE_WIDTH_PX, IMAGE_HEIGHT_PX]}"
        )
    return manifest, marker


def fence_out_dir(out_dir: str, scene_names: Sequence[str], cfg: Track2DConfig,
                  provider: str) -> dict:
    """Refuse a subset re-run that would re-bless another config's scene dirs (C27).

    `--scenes scene-0061` writes one scene and then stands a fresh marker and a
    fresh manifest — whose config echo and whose totals describe that one scene —
    over the WHOLE out dir, and Stage 4 consumes every directory under `scenes/`.
    Yesterday's nine unrefined scenes plus today's one refined one therefore read
    downstream as a single upstream whose manifest says refine_matched_boxes=true.
    Refused here, before clear_markers and before the first write, so the refusal
    costs nothing and the previous run's marker survives it.

    An INCREMENTAL subset re-run under the IDENTICAL config is a different thing
    and is allowed: the tree it leaves behind is one configuration's output
    either way. The manifest then records `scenes_written` next to
    `scenes_in_out_dir`, so a reader never has to infer which is which.

    Returns the manifest block. Raises UpstreamRefusal (rc 2, nothing written).
    """
    root = os.path.join(out_dir, "scenes")
    if not os.path.isdir(root):
        return {"scenes_in_out_dir": [], "foreign_scene_dirs": [], "config_matched_previous": None}
    existing = sorted(n for n in os.listdir(root) if os.path.isdir(os.path.join(root, n)))
    foreign = [n for n in existing if n not in set(scene_names)]
    if not foreign:
        return {"scenes_in_out_dir": existing, "foreign_scene_dirs": [],
                "config_matched_previous": None}
    previous: dict = {}
    manifest_path = os.path.join(out_dir, "run_manifest.json")
    if os.path.isfile(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as handle:
                previous = json.load(handle)
        except (OSError, ValueError):
            # An unreadable manifest is not a matching one; the refusal below
            # says so rather than guessing what produced those directories.
            previous = {}
    # The whole config echo, not a chosen subset: every field in it decided a
    # value in the rows underneath, and the run that wrote them recorded it the
    # same way (as_dict round-trips through JSON byte-comparably by design).
    # `provider` is derived rather than stored in the config, so it joins here.
    same_config = (
        previous.get("spec") == STAGE_SPEC
        and previous.get("provider") == provider
        and previous.get("config") == cfg.as_dict()
    )
    if not same_config:
        raise UpstreamRefusal(
            f"{root} already holds scene directories this run does not write: {foreign}. "
            + (
                "The run_manifest.json standing over them describes a DIFFERENT configuration"
                if previous
                else "No readable run_manifest.json stands over them, so no configuration can be "
                     "quoted for them"
            )
            + f", and this run would put a fresh marker and a fresh manifest — whose config echo "
            f"and totals describe only {sorted(scene_names)} — over all of them. Stage 4 consumes "
            "every directory under scenes/, so the mixture would be read downstream as ONE "
            "upstream produced by this run's config. Write this subset to a separate --out-dir, "
            "or re-run every scene."
        )
    return {"scenes_in_out_dir": existing, "foreign_scene_dirs": foreign,
            "config_matched_previous": True}


def _frame_loader(dataroot: str):
    from PIL import Image  # local: keeps the pure-logic paths import-free

    def load(frame: FrameRef) -> np.ndarray:
        with Image.open(os.path.join(dataroot, frame.filename)) as handle:
            return np.asarray(handle.convert("RGB"), dtype=np.uint8)

    return load


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run(
    paths: Paths,
    upstream: dict,
    upstream_marker,
    cfg: Track2DConfig,
    stage3_dir: str,
    out_dir: str,
    scene_names: Sequence[str] | None,
) -> tuple[dict, int]:
    # Refused BEFORE clear_markers: a refusal that has already taken the previous
    # run's marker down leaves a complete output looking incomplete.
    if cfg.reverse_pass:
        raise UpstreamRefusal(
            "reverse pass is gated to a later revision (C27): a backward pass changes what `hops` "
            "and `miss_tolerance_keyframes` mean and neither has been measured. The flag exists so "
            "the request is refused loudly instead of silently ignored"
        )
    started = time.time()
    provider = cfg.provider or infer_track_provider(cfg.model_id)
    if provider not in _TRACK_ADAPTERS:
        raise UpstreamRefusal(
            f"unknown video-tracker provider {provider!r}; one of {sorted(_TRACK_ADAPTERS)}"
        )
    spec = CheckpointSpec(
        role=MASK_2D,  # Stage 3b registers no role of its own; it borrows mask_2d's
        provider=provider,
        model_id=cfg.model_id,
        revision=cfg.revision,
        provenance=f"C27 (2026-08-19): stage 3b video tracker — {provider}. "
        + _PROVIDER_PROVENANCE.get(provider, ""),
    )
    spec_errors = spec.validate(prefix="checkpoint: ")
    if spec_errors:
        raise UpstreamRefusal(
            "; ".join(spec_errors) + " — pass --revision with the hub commit sha for this checkpoint"
        )

    substrate = Substrate.load(paths)

    root = os.path.join(stage3_dir, "scenes")
    if not os.path.isdir(root):
        raise UpstreamRefusal(f"{root} not found; Stage 3 wrote no scenes")
    names = sorted(n for n in os.listdir(root) if os.path.isdir(os.path.join(root, n)))
    if scene_names:
        missing = sorted(set(scene_names) - set(names))
        if missing:
            raise UpstreamRefusal(f"requested scene(s) not present in Stage 3 output: {missing}")
        names = [n for n in names if n in scene_names]

    rows_by_scene: dict[str, list[dict]] = {}
    scene_tokens: dict[str, str] = {}
    counters_by_scene: dict[str, dict[str, int]] = {}
    causes_by_scene: dict[str, list[str]] = {}
    for scene_name in names:
        rows = read_proposal_rows(os.path.join(root, scene_name, "proposals.jsonl"))
        rows_by_scene[scene_name] = rows
        scene_tokens[scene_name] = rows[0]["scene_token"] if rows else ""
        counters_by_scene[scene_name] = new_counters()
        causes_by_scene[scene_name] = []

    # Nothing above this line has written a byte, and nothing below it writes one
    # until clear_markers: these two are the LAST refusals, which is what keeps
    # "rc 2 == nothing written, previous marker intact" true when either fires.
    out_dir_tree = fence_out_dir(out_dir, names, cfg, provider)
    preflight = preflight_tracker(spec, cfg, provider)

    # The chains are metadata-only and are built ONCE: Phase A walks them for the
    # sweep frames and Phase B cuts the same lists into windows, so a chain that
    # differed between the two phases would be a silent inconsistency.
    chains: dict[tuple[str, str], list[FrameRef]] = {}
    for scene_name in names:
        scene_token = scene_tokens[scene_name]
        if not scene_token:
            continue
        for channel in RING_CAMERAS:
            try:
                chains[(scene_token, channel)] = camera_frame_chain(substrate, scene_token, channel)
            except SequenceError as exc:
                counters_by_scene[scene_name]["n_camera_chains_failed"] += 1
                causes_by_scene[scene_name].append(
                    f"{scene_name}/{channel}: frame chain unusable ({exc}); this camera is emitted "
                    "with detected boxes only"
                )

    # Stage 3's own same-class dedup IoU decides the recovered-vs-recovered
    # contest; read from the run being consumed, not from the working tree.
    recovered_suppress_iou = float(
        (upstream.get("config") or {}).get("same_class_iou", ProposalConfig().same_class_iou)
    )

    # --- Phase A: the sweep detector, resident alone ---
    sweep_dets: dict[str, list[SweepDetection]] = {}
    sweep_checkpoint: dict | None = None
    totals = new_counters()
    degraded = False
    # Run-level causes: an unload runs after the last scene and has no per-scene
    # ledger to hang from, but the marker still has to carry it (F2).
    run_causes: list[str] = []
    # C16: a DEGRADED upstream degrades this run, whatever this run then does.
    # Recorded here, before the first byte, so it cannot be lost to a later
    # branch: main() writes `_SUCCESS.degraded` from `degraded`, and the causes
    # travel in `run_causes` because an upstream flag has no scene to hang from.
    inherited = inherited_degradation_causes(upstream_marker)
    if inherited:
        degraded = True
        run_causes.extend(inherited)
        print(
            f"upstream Stage 3 is DEGRADED ({len(inherited)} cause(s)) and was accepted (C16): "
            "this run's marker is _SUCCESS.degraded and carries them forward, so Stage 4 reads "
            "the flag from the stage in front of it"
        )
    phase_a: tuple[Any, Any, Any] | None = None
    if cfg.detect_on_sweeps:
        # Rebuilt BEFORE clear_markers, for the preflight's reason (F3): every
        # refusal this can raise — a moved taxonomy, a class map that changed
        # under the run, a Stage 3 manifest carrying no usable checkpoint
        # identity — is an upstream contract failure, and a refusal that has
        # already taken the previous run's marker down leaves a complete output
        # looking incomplete.
        detector, prompt, sweep_checkpoint, taxonomy = build_sweep_detector(upstream, cfg)
        phase_a = (detector, prompt, taxonomy)

    # Any marker still standing describes the PREVIOUS run of this stage; it
    # comes down here, immediately before the first write (C16).
    clear_markers(out_dir)

    if phase_a is not None:
        detector, prompt, taxonomy = phase_a
        try:
            sweep_dets = detect_on_sweep_frames(
                adapter=detector,
                prompt=prompt,
                taxonomy_phrase_to_categories=taxonomy.phrase_to_categories,
                cfg=cfg,
                dataroot=paths.dataroot,
                chains=chains,
                scene_tokens=scene_tokens,
                scene_names=names,
                out_dir=out_dir,
                checkpoint=sweep_checkpoint,
                counters_by_scene=counters_by_scene,
                causes_by_scene=causes_by_scene,
            )
        finally:
            # Serial residency (C1): the detector leaves before the tracker
            # arrives, and the cache is emptied so the ceiling is real. Guarded
            # for the reason the Phase B unload is (F2): ten scenes of
            # sweep_proposals.jsonl are on disk by now, and a failure to RELEASE
            # a model must never be what deletes them. If the detector really is
            # still resident, the tracker will not fit under C1's ceiling and
            # every window then fails loudly and is counted as such.
            try:
                detector.unload()
            except Exception as exc:  # noqa: BLE001 — a leaked resident is not a lost run
                totals["n_unload_failed"] += 1
                run_causes.append(
                    f"sweep detector unload failed ({type(exc).__name__}: {exc}); it may still be "
                    "resident, so C1's ceiling is not proved for Phase B"
                )
                degraded = True

    # --- Phase B: the video tracker ---
    # preflight_tracker already exercised this provider's imports, its device
    # requirement and its checkpoint resolution; what remains here is the weight
    # build the preflight deliberately did not pay for twice.
    adapter = _TRACK_ADAPTERS[provider](spec, cfg)
    adapter.load()
    load_frame = _frame_loader(paths.dataroot)

    per_scene: list[dict] = []
    window_lens: dict[int, int] = {}
    n_subsampled = 0

    for scene_name in names:
        rows = rows_by_scene[scene_name]
        scene_token = scene_tokens[scene_name]
        counters = counters_by_scene[scene_name]
        causes = causes_by_scene[scene_name]
        counters["n_camera_rows"] = len(rows)
        counters["n_keyframes"] = len({row["keyframe_token"] for row in rows})
        counters["n_yolo_boxes"] = sum(int(row["n_proposals"]) for row in rows)

        by_channel: dict[str, list[tuple[int, dict]]] = {}
        for index, row in enumerate(rows):
            by_channel.setdefault(row["channel"], []).append((index, row))

        rewritten: dict[int, dict] = {}
        tracks12: list[dict] = []
        for channel in RING_CAMERAS:
            entries = by_channel.get(channel, [])
            if not entries:
                continue
            kf_rows = [row for _, row in entries]
            chain = chains.get((scene_token, channel))
            windows: list[Window] = []
            if chain is not None:
                try:
                    windows, missing = keyframe_windows(
                        chain, [row["sample_data_token"] for row in kf_rows], cfg.window_max_frames
                    )
                except SequenceError as exc:
                    # Stage 3's rows for this camera are not in chain order, so
                    # no window can be cut. One camera's ordering defect must not
                    # end a run that has already written Phase A's artifacts:
                    # counted, named, and this camera falls back to detected
                    # boxes only.
                    counters["n_camera_chains_failed"] += 1
                    causes.append(f"{scene_name}/{channel}: windows unusable ({exc})")
                    windows, missing = [], []
                if missing:
                    counters["n_windows_missing_anchor"] += len(missing)
                    causes.append(
                        f"{scene_name}/{channel}: {len(missing)} Stage 3 keyframe(s) are not on the "
                        f"sample_data chain (first {missing[:3]}); the neighbouring window widens "
                        "over them"
                    )
            for window in windows:
                window_lens[len(window.frames)] = window_lens.get(len(window.frames), 0) + 1
                n_subsampled += 1 if window.subsampled else 0
            out_rows = track_camera(
                cfg=cfg,
                adapter=adapter,
                scene_name=scene_name,
                scene_token=scene_token,
                channel=channel,
                kf_rows=kf_rows,
                windows=windows,
                sweep_dets=sweep_dets,
                load_frame=load_frame,
                recovered_suppress_iou=recovered_suppress_iou,
                counters=counters,
                causes=causes,
                tracks12=tracks12,
            )
            for (index, _), new_row in zip(entries, out_rows):
                rewritten[index] = new_row

        if len(rewritten) != len(rows):
            raise Track2DContractError(
                f"{scene_name}: rewrote {len(rewritten)} of {len(rows)} Stage 3 rows. Every row "
                "must be emitted exactly once, in Stage 3's order — a row on a channel outside "
                f"{list(RING_CAMERAS)} would be silently dropped here"
            )
        out_scene = os.path.join(out_dir, "scenes", scene_name)
        write_jsonl_atomic(
            os.path.join(out_scene, "proposals.jsonl"), [rewritten[i] for i in range(len(rows))]
        )
        if cfg.emit_12hz_tracks:
            write_jsonl_atomic(os.path.join(out_scene, "tracks_12hz.jsonl"), tracks12)

        scene_degraded = any(counters[key] for key in DEGRADING_COUNTERS)
        if scene_degraded and not causes:
            causes = [
                f"{scene_name}: {key}={counters[key]}"
                for key in DEGRADING_COUNTERS
                if counters[key]
            ]
            causes_by_scene[scene_name] = causes
        degraded = degraded or scene_degraded
        per_scene.append(
            {"scene": scene_name, **counters, "degraded": scene_degraded, "causes": list(causes)}
        )
        for key in totals:
            totals[key] += counters[key]
        print(
            f"  [B] {scene_name}  {counters['n_camera_rows']:>4} rows  "
            f"{counters['n_yolo_boxes']:>5} yolo  {counters['n_recovered']:>4} recovered  "
            f"{counters['n_tracks']:>5} tracks  {counters['n_windows']:>4} win  "
            f"{counters['n_windows_failed']:>3} failed" + ("  DEGRADED" if scene_degraded else "")
        )

    try:
        adapter.unload()
    except Exception as exc:  # noqa: BLE001 — a leaked resident is not a lost run
        # The manifest and the three-state marker are written by main() AFTER
        # run() returns, and main() does not catch RuntimeError: an unguarded
        # failure here — a CUDA fault surfacing at empty_cache(), say — discarded
        # a COMPLETE Phase A + Phase B, 40-60 min of work, with no manifest and
        # no marker to show for it (F2). Releasing a model is the last thing this
        # run needs from the device; it is recorded as a degrading cause and the
        # run is DEGRADED, never lost.
        totals["n_unload_failed"] += 1
        run_causes.append(
            f"video tracker unload failed ({type(exc).__name__}: {exc}); device memory may still "
            "be held by this process"
        )
        degraded = True

    manifest = {
        "spec": STAGE_SPEC,
        "stage": STAGE,
        "seed": cfg.global_seed,
        "config": cfg.as_dict(),
        "image_size_px": [IMAGE_WIDTH_PX, IMAGE_HEIGHT_PX],
        # Verbatim: Stage 4 cross-checks prompt.caption_sha256 against the stage
        # it consumes, and Stage 3b changes nothing about the class space.
        "prompt": upstream.get("prompt"),
        "paths": paths.as_dict(),
        "upstream": {
            "metadata_fingerprint": upstream["upstream"]["metadata_fingerprint"],
            "fingerprint_spec": upstream["upstream"]["fingerprint_spec"],
            "stage3_spec": upstream["spec"],
            "stage3_provider": upstream.get("provider"),
            "stage3_checkpoint": upstream.get("checkpoint"),
            # C16: a run built on accepted degradation says so in its provenance.
            "degraded": upstream_marker.degraded,
            "degraded_causes": list(upstream_marker.causes),
            "accepted_degraded_upstream": cfg.accept_degraded_upstream,
        },
        "provider": provider,
        # The sam31 adapter stream-hashes the weights it actually loaded into
        # `checkpoint_sha256`; Stage 4's manifest prefers exactly this expression
        # and Stage 3b threw the hash away, so the A/B cell that ran on those
        # bytes could not be quoted against them (F1, §7.2).
        "checkpoint": {
            "model_id": spec.model_id,
            "revision": spec.revision,
            "sha256": getattr(adapter, "checkpoint_sha256", None) or spec.sha256,
        },
        "preflight": preflight,  # what was proved before the first byte (F3)
        "sweep_detector": sweep_checkpoint,
        "vram_cap": getattr(adapter, "vram_cap", None),  # C1 — ceiling, or its honest absence
        "propagation": {
            "windows": {
                # String keys: an int-keyed dict does not survive the JSON
                # round-trip write_json_atomic verifies.
                "lens_histogram": {str(k): window_lens[k] for k in sorted(window_lens)},
                "max_frames": cfg.window_max_frames,
                "n_subsampled": n_subsampled,
            },
            "reinit_at_block_boundary": True,  # the registry is discarded at every scene
            "reverse_pass": cfg.reverse_pass,
            "video_storage_device": cfg.video_storage_device,
            # Recorded with the answer to "did this provider consume it?" (F10):
            # on sam31_multiplex the flag is inert, and a manifest that recorded
            # it bare described a memory layout that never existed.
            "video_storage_device_consumed": provider in _STORAGE_DEVICE_CONSUMERS,
            "detect_on_sweeps": cfg.detect_on_sweeps,
            "sweep_birth_min_hits": cfg.sweep_birth_min_hits,
            "emit_12hz_tracks": cfg.emit_12hz_tracks,
            "recovered_suppress_iou": recovered_suppress_iou,
            "recovered_suppress_iou_source": "Stage 3 manifest config.same_class_iou",
            "presence_source": getattr(adapter, "presence_source", "unknown"),
        },
        "determinism": (
            "§1.9: fixed iteration orders everywhere (scenes by name, cameras in RING_CAMERAS "
            "order, tracks by ascending track_id, boxes in row order, injections appended by "
            "track_id); the seed is threaded into torch at adapter load; nothing reads the clock "
            "except elapsed_s. CUDA float nondeterminism is acknowledged, never silently reseeded"
        ),
        "numpy_version": np.__version__,
        "python_version": sys.version.split()[0],
        "elapsed_s": round(time.time() - started, 2),
        # `scenes` is the per-scene ledger of what THIS run wrote; the two lists
        # below say what the tree under this marker actually holds, so a
        # legitimate incremental subset re-run cannot be read as a full one and a
        # foreign leftover cannot hide inside it (F11).
        "scenes_written": list(names),
        "scenes_in_out_dir": sorted(set(out_dir_tree["scenes_in_out_dir"]) | set(names)),
        "scene_dirs_not_written_by_this_run": out_dir_tree["foreign_scene_dirs"],
        "scenes": per_scene,
        "totals": totals,
        "run_causes": run_causes,
    }
    return manifest, EXIT_DEGRADED if degraded else EXIT_OK


# ---------------------------------------------------------------------------
# --self-test: the whole window loop, with a scripted tracker and no GPU
# ---------------------------------------------------------------------------


class FakeTracker:
    """A `MaskVideoTracker` over scripted rectangles. No model, no CUDA, no imagery.

    It carries each object's prompted box forward unchanged unless `plan` says
    otherwise for a given (frame_idx, obj_id) — `None` there means "no mask this
    frame", which is the contract's way of saying the object was lost.
    """

    def __init__(self, plan: dict[tuple[int, int], tuple[float, float, float, float] | None] | None = None):
        self.plan = dict(plan or {})
        self.closed = 0

    def init_video(self, frames: Sequence[np.ndarray]) -> Any:
        return {"n_frames": len(frames), "objs": {}}

    def add_video_boxes(
        self, session: Any, frame_idx: int, obj_ids: Sequence[int], boxes_xyxy_px: Any
    ) -> None:
        for obj_id, box in zip(obj_ids, boxes_xyxy_px):
            session["objs"][int(obj_id)] = tuple(float(v) for v in box)

    def propagate_video(
        self, session: Any, *, start_frame_idx: int, max_frames: int | None = None,
        reverse: bool = False,
    ) -> Iterator[tuple[int, dict[int, np.ndarray], dict[int, float]]]:
        for frame_idx in range(int(start_frame_idx), session["n_frames"]):
            masks: dict[int, np.ndarray] = {}
            presence: dict[int, float] = {}
            for obj_id in sorted(session["objs"]):
                box = self.plan.get((frame_idx, obj_id), session["objs"][obj_id])
                if box is None:
                    continue  # absent obj_id == no mask this frame; the driver zero-fills
                mask = np.zeros((IMAGE_HEIGHT_PX, IMAGE_WIDTH_PX), dtype=bool)
                x1, y1, x2, y2 = (int(round(v)) for v in box)
                mask[y1:y2, x1:x2] = True
                masks[obj_id] = mask
                presence[obj_id] = 1.0
            yield frame_idx, masks, presence

    def close_video(self, session: Any) -> None:
        self.closed += 1


class _StFailingTracker(FakeTracker):
    """A FakeTracker that raises on demand — the two failure modes F4 and F8 name.

    `fail_init` raises inside init_video: a window whose propagation never ran.
    `fail_close` raises inside close_video: a window that propagated and then
    leaked its session. Both are indexed by the order in which windows OPENED a
    session, so a skipped-empty window (which never calls init_video) does not
    consume an index, and a window whose init raised is never closed.
    """

    def __init__(self, plan=None, *, fail_init: Sequence[int] = (),
                 fail_close: Sequence[int] = ()) -> None:
        super().__init__(plan)
        self.fail_init = set(fail_init)
        self.fail_close = set(fail_close)
        self.n_opened = 0

    def init_video(self, frames: Sequence[np.ndarray]) -> Any:
        index = self.n_opened
        self.n_opened += 1
        if index in self.fail_init:
            raise RuntimeError(f"scripted init_video failure, window {index}")
        return super().init_video(frames)

    def close_video(self, session: Any) -> None:
        if (self.n_opened - 1) in self.fail_close:
            raise RuntimeError(f"scripted close_video failure, window {self.n_opened - 1}")
        super().close_video(session)


_ST_CHANNEL = "CAM_FRONT"
_ST_SCENE_TOKEN = "scene-token"


def _st_row(index: int, boxes, classes, scores) -> dict:
    """A row shaped like Stage 3's, with the keys rewrite_row touches."""
    return {
        "spec": "dhakascenes-pilot/stage3_proposals/v1",
        "keyframe_token": f"kf-{index}",
        "scene_token": _ST_SCENE_TOKEN,
        "channel": _ST_CHANNEL,
        "sample_data_token": f"kf-{index}",
        "t_ns": 1_500_000_000_000_000_000 + index,
        "time_base": "unix_ns",
        "coverage_config": "R2",
        "image_path": f"samples/{_ST_CHANNEL}/{index}.jpg",
        "image_size_px": [IMAGE_WIDTH_PX, IMAGE_HEIGHT_PX],
        "n_proposals": len(boxes),
        "boxes_xyxy_px": [[float(v) for v in box] for box in boxes],
        "scores": [float(s) for s in scores],
        "class_names": list(classes),
        "nuscenes_categories": [["vehicle.car"] for _ in classes],
        "phrase_char_spans": [[0, 5] for _ in classes],
        "seed": 20260812,
    }


def _st_window(start_index: int, n_interior: int) -> Window:
    """One window kf_i -> kf_{i+1} with `n_interior` sweep frames between them."""
    refs = [FrameRef(f"kf-{start_index}", "samples/x.jpg", 1000 * start_index, True, "s")]
    for k in range(n_interior):
        token = f"sw-{start_index}-{k}"
        refs.append(FrameRef(token, f"sweeps/{token}.jpg", 1000 * start_index + k + 1, False, "s"))
    refs.append(
        FrameRef(f"kf-{start_index + 1}", "samples/y.jpg", 1000 * (start_index + 1), True, "s")
    )
    return Window(start=refs[0], frames=tuple(refs), end=refs[-1], subsampled=False)


_ST_IMAGE = np.zeros((IMAGE_HEIGHT_PX, IMAGE_WIDTH_PX, 3), dtype=np.uint8)


def _st_drive(cfg, rows, windows, tracker, sweep_dets=None):
    counters = new_counters()
    causes: list[str] = []
    tracks12: list[dict] = []
    out = track_camera(
        cfg=cfg,
        adapter=tracker,
        scene_name="scene-self-test",
        scene_token=_ST_SCENE_TOKEN,
        channel=_ST_CHANNEL,
        kf_rows=rows,
        windows=windows,
        sweep_dets=sweep_dets or {},
        # A single shared full-size frame: the driver asserts 1600x900 (§1.5) and
        # never reads a pixel, so one allocation serves every scenario.
        load_frame=lambda ref: _ST_IMAGE,
        recovered_suppress_iou=0.65,  # Stage 3's same_class_iou, passed explicitly here
        counters=counters,
        causes=causes,
        tracks12=tracks12,
    )
    return out, counters, tracks12


def _self_test() -> int:
    results: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        results.append((name, bool(ok), detail))

    def refuses(call) -> str:
        """The refusal text, or '' if the call did not refuse at all."""
        try:
            call()
        except UpstreamRefusal as exc:
            return str(exc)
        return ""

    car, truck = "a car", "a truck"
    box_a = (0.0, 0.0, 100.0, 100.0)

    # (1) continuity: detected, missed, detected -> recovered in the middle.
    cfg = Track2DConfig(revision="self-test")
    rows = [
        _st_row(0, [box_a], [car], [0.8]),
        _st_row(1, [], [], []),
        _st_row(2, [box_a], [car], [0.9]),
    ]
    out, counters, tracks12 = _st_drive(cfg, rows, [_st_window(0, 2), _st_window(1, 2)], FakeTracker())
    ok = (
        [r["box_sources"] for r in out] == [["yolo"], ["recovered"], ["yolo"]]
        and [r["track_ids"] for r in out] == [[0], [0], [0]]
        and out[1]["scores"] == [round(0.8 * cfg.recovered_score_decay, 5)]
        and out[1]["n_propagated_hops"] == [1]
        and out[1]["boxes_xyxy_px"] == [[0.0, 0.0, 100.0, 100.0]]
        and counters["n_recovered"] == 1
        and counters["n_tracks"] == 1
    )
    check("continuity: kf1 recovered from kf0 evidence, id stable, score x decay", ok,
          f"sources={[r['box_sources'] for r in out]} ids={[r['track_ids'] for r in out]} "
          f"score={out[1]['scores']}")

    # (2) lost-contest at IoU 0.40 -> the propagated box MATCHES; nothing injected.
    rows = [_st_row(0, [box_a], [car], [0.9]), _st_row(1, [(43.0, 0.0, 143.0, 100.0)], [car], [0.5])]
    out, counters, _ = _st_drive(cfg, rows, [_st_window(0, 1)], FakeTracker())
    iou_40 = float(pairwise_iou(np.asarray([box_a]), np.asarray([(43.0, 0.0, 143.0, 100.0)]))[0, 0])
    ok = (
        out[1]["n_proposals"] == 1
        and out[1]["box_sources"] == ["yolo"]
        and out[1]["track_ids"] == [0]
        and counters["n_recovered"] == 0
        and counters["n_tracks"] == 1
    )
    check(f"match at IoU {iou_40:.2f} >= match_iou: matched, no injection, no duplicate id", ok,
          f"n={out[1]['n_proposals']} ids={out[1]['track_ids']} recovered={counters['n_recovered']}")

    # (3a) residual window: overlap BELOW match_iou -> injected alongside the detection.
    box_low = (67.0, 0.0, 167.0, 100.0)
    rows = [_st_row(0, [box_a], [car], [0.9]), _st_row(1, [box_low], [car], [0.5])]
    out, counters, _ = _st_drive(cfg, rows, [_st_window(0, 1)], FakeTracker())
    iou_20 = float(pairwise_iou(np.asarray([box_a]), np.asarray([box_low]))[0, 0])
    ok = (
        out[1]["n_proposals"] == 2
        and out[1]["box_sources"] == ["yolo", "recovered"]
        and out[1]["track_ids"] == [1, 0]
        and counters["n_lost_contest"] == 0
    )
    check(f"residual at IoU {iou_20:.2f} < match_iou: injected next to the detection", ok,
          f"sources={out[1]['box_sources']} ids={out[1]['track_ids']}")

    # (3b) the other side: a track that overlaps a same-class detection ANOTHER
    # track won -> lost contest, not injected.
    box_mid = (38.0, 0.0, 138.0, 100.0)
    rows = [_st_row(0, [box_a, box_mid], [car, car], [0.9, 0.8]), _st_row(1, [box_a], [car], [0.7])]
    out, counters, _ = _st_drive(cfg, rows, [_st_window(0, 1)], FakeTracker())
    iou_45 = float(pairwise_iou(np.asarray([box_mid]), np.asarray([box_a]))[0, 0])
    ok = (
        out[1]["n_proposals"] == 1
        and out[1]["track_ids"] == [0]
        and counters["n_lost_contest"] == 1
        and counters["n_recovered"] == 0
    )
    check(f"lost contest at IoU {iou_45:.2f} in [match_iou, 1]: suppressed, counted", ok,
          f"n={out[1]['n_proposals']} lost={counters['n_lost_contest']}")

    # (4) cross-class near-identity -> suppressed.
    box_near = (3.0, 0.0, 103.0, 100.0)
    rows = [_st_row(0, [box_a], [car], [0.9]), _st_row(1, [box_near], [truck], [0.7])]
    out, counters, _ = _st_drive(cfg, rows, [_st_window(0, 1)], FakeTracker())
    iou_95 = float(pairwise_iou(np.asarray([box_a]), np.asarray([box_near]))[0, 0])
    ok = (
        out[1]["n_proposals"] == 1
        and counters["n_suppressed_cross_class"] == 1
        and counters["n_recovered"] == 0
    )
    check(f"cross-class at IoU {iou_95:.2f} >= cross_class_suppress_iou: suppressed", ok,
          f"n={out[1]['n_proposals']} cross={counters['n_suppressed_cross_class']}")

    # (5) death after miss_tolerance + 1 unmatched keyframes. The plan is
    # window-local: the object is visible at each window's prompt frame and lost
    # from the next one on, so every window ends with no evidence box.
    plan: dict[tuple[int, int], tuple[float, float, float, float] | None] = {(1, 0): None, (2, 0): None}
    rows = [_st_row(i, [], [], []) for i in range(4)]
    rows[0] = _st_row(0, [box_a], [car], [0.9])
    windows = [_st_window(0, 1), _st_window(1, 1), _st_window(2, 1)]
    out, counters, _ = _st_drive(cfg, rows, windows, FakeTracker(plan))
    ok = (
        counters["n_deaths"] == 1
        and counters["n_recovered"] == 0
        and all(row["n_proposals"] == 0 for row in out[1:])
    )
    check(f"death after miss_tolerance ({cfg.miss_tolerance_keyframes}) + 1 misses", ok,
          f"deaths={counters['n_deaths']} recovered={counters['n_recovered']}")

    # (6) mid-gap birth needs sweep_birth_min_hits before it may inject.
    det = SweepDetection(box_xyxy_px=box_a, score=0.7, class_name=car,
                         phrase_char_span=(0, 5), nuscenes_categories=("vehicle.car",))
    rows = [_st_row(0, [], [], []), _st_row(1, [], [], [])]
    window = _st_window(0, 2)
    one_hit = {window.frames[1].sample_data_token: [det]}
    out, counters, _ = _st_drive(cfg, rows, [window], FakeTracker(), one_hit)
    ok_one = (
        out[1]["n_proposals"] == 0
        and counters["n_midgap_births"] == 1
        and counters["n_rejected_min_hits"] == 1
        and counters["n_repropagations"] == 1
    )
    two_hits = {window.frames[1].sample_data_token: [det], window.frames[2].sample_data_token: [det]}
    out2, counters2, _ = _st_drive(cfg, rows, [window], FakeTracker(), two_hits)
    ok_two = (
        out2[1]["n_proposals"] == 1
        and out2[1]["box_sources"] == ["recovered"]
        and out2[1]["scores"] == [round(0.7 * cfg.recovered_score_decay, 5)]
        and counters2["n_rejected_min_hits"] == 0
    )
    check(f"mid-gap birth: 1 hit rejected, {cfg.sweep_birth_min_hits} hits injected",
          ok_one and ok_two,
          f"one_hit_n={out[1]['n_proposals']} rejected={counters['n_rejected_min_hits']} "
          f"two_hit_n={out2[1]['n_proposals']} scores={out2[1]['scores']}")

    # (7) refinement: the flag, the IoU gate and the area band.
    tight = (5.0, 5.0, 95.0, 95.0)
    rows = [_st_row(0, [box_a], [car], [0.9]), _st_row(1, [box_a], [car], [0.9])]
    refine_plan = {(2, 0): tight}
    out_off, _, _ = _st_drive(cfg, rows, [_st_window(0, 1)], FakeTracker(refine_plan))
    cfg_on = Track2DConfig(revision="self-test", refine_matched_boxes=True)
    out_on, counters_on, _ = _st_drive(cfg_on, rows, [_st_window(0, 1)], FakeTracker(refine_plan))
    ok = (
        out_off[1]["boxes_xyxy_px"] == [list(box_a)]
        and out_off[1]["refined"] == [False]
        and out_off[1]["boxes_xyxy_px_original"] == [None]
        and out_on[1]["boxes_xyxy_px"] == [[5.0, 5.0, 95.0, 95.0]]
        and out_on[1]["refined"] == [True]
        and out_on[1]["boxes_xyxy_px_original"] == [list(box_a)]
        and counters_on["n_refined"] == 1
    )
    check("refine flag: off leaves the detected box, on swaps it and keeps the original", ok,
          f"off={out_off[1]['boxes_xyxy_px']} on={out_on[1]['boxes_xyxy_px']} "
          f"orig={out_on[1]['boxes_xyxy_px_original']}")

    # (8) byte stability of an untouched row.
    original = _st_row(7, [box_a, box_mid], [car, truck], [0.9, 0.4])
    frozen = json.dumps(original, sort_keys=True)
    rewritten = rewrite_row(
        original, assignments=[BoxAssignment(track_id=3), BoxAssignment(track_id=4)],
        injections=[], window_frames=7,
    )
    same = json.dumps({k: rewritten[k] for k in original}, sort_keys=True)
    ok = (
        same == frozen
        and json.dumps(original, sort_keys=True) == frozen  # rewrite_row did not mutate the input
        and rewritten["track2d"]["window_frames"] == 7
        and rewritten["box_sources"] == ["yolo", "yolo"]
    )
    check("row byte-stability: original keys survive rewrite_row json-identical", ok,
          "mutated" if json.dumps(original, sort_keys=True) != frozen else f"equal={same == frozen}")

    # (9) the per-image cap: a full row leaves no slot for a recovery.
    grid = [
        (200.0 + 10 * (i % 20), 200.0 + 10 * (i // 20), 208.0 + 10 * (i % 20), 208.0 + 10 * (i // 20))
        for i in range(cfg.max_proposals_per_image)
    ]
    rows = [
        _st_row(0, [box_a], [car], [0.9]),
        _st_row(1, grid, [truck] * len(grid), [0.5] * len(grid)),
    ]
    out, counters, _ = _st_drive(cfg, rows, [_st_window(0, 1)], FakeTracker())
    ok = (
        out[1]["n_proposals"] == cfg.max_proposals_per_image
        and counters["n_row_cap_drops"] == 1
        and counters["n_recovered"] == 0
        and out[1]["box_sources"].count("recovered") == 0
    )
    check(f"row cap {cfg.max_proposals_per_image}: the recovery is dropped and counted", ok,
          f"n={out[1]['n_proposals']} drops={counters['n_row_cap_drops']}")

    # (10) the 12 Hz artifact: one row per (frame, live track with a mask), no
    # frame counted twice across two adjacent windows.
    rows = [_st_row(i, [box_a] if i == 0 else [], [car] if i == 0 else [], [0.9] if i == 0 else [])
            for i in range(3)]
    _, counters, tracks12 = _st_drive(cfg, rows, [_st_window(0, 2), _st_window(1, 2)], FakeTracker())
    tokens = [row["sample_data_token"] for row in tracks12]
    ok = (
        len(tokens) == len(set(tokens))
        and counters["n_12hz_rows"] == len(tracks12)
        and tracks12[0]["evidence"] == "yolo"
        and tracks12[0]["sample_data_token"] == "kf-0"
        and all(row["spec"] == TRACKS12_SPEC and row["channel"] == _ST_CHANNEL for row in tracks12)
    )
    check("12 Hz artifact: every frame once, kf_0 emitted with the first window", ok,
          f"rows={len(tracks12)} unique={len(set(tokens))} first={tokens[:1]}")

    # (11) a candidate a gate rejects must still AGE. Deferring the miss into the
    # accept branch is how a track that is rejected at every keyframe becomes
    # immortal — it never reaches miss_tolerance_keyframes and never dies.
    rows = [_st_row(i, [], [], []) for i in range(5)]
    windows = [_st_window(i, 2) for i in range(4)]
    out, counters, _ = _st_drive(
        cfg, rows, windows, FakeTracker(),
        {windows[0].frames[1].sample_data_token: [det]},
    )
    ok_gate_i = (
        counters["n_midgap_births"] == 1
        and counters["n_deaths"] == 1
        and counters["n_recovered"] == 0
        and counters["n_rejected_min_hits"] == cfg.miss_tolerance_keyframes + 1
    )
    rows = [_st_row(0, [box_a, box_mid], [car, car], [0.9, 0.8])] + [
        _st_row(i, [box_a], [car], [0.7]) for i in range(1, 5)
    ]
    _, counters, _ = _st_drive(cfg, rows, [_st_window(i, 1) for i in range(4)], FakeTracker())
    ok_gate_iii = counters["n_deaths"] == 1 and counters["n_lost_contest"] == cfg.miss_tolerance_keyframes
    check("rejected candidates still age: a perpetually gated track dies on schedule",
          ok_gate_i and ok_gate_iii,
          f"min_hits: deaths={counters['n_deaths']} | lost-contest path: "
          f"lost={counters['n_lost_contest']}")

    # (12) F4: a failed window is a BLOCK BOUNDARY. With the live tracks merely
    # penalized, kf_1 births a DUPLICATE of the object that is still coasting on
    # the kf_0 seed box, and at kf_2 that stale copy injects a phantom recovery
    # beside the real detection (reviewer-reproduced with a scripted tracker).
    moving = [(0.0, 0.0, 100.0, 100.0), (40.0, 0.0, 140.0, 100.0), (80.0, 0.0, 180.0, 100.0)]
    rows = [_st_row(i, [moving[i]], [car], [0.9]) for i in range(3)]
    out, counters, _ = _st_drive(
        cfg, rows, [_st_window(0, 1), _st_window(1, 1)], _StFailingTracker(fail_init={0})
    )
    ok = (
        counters["n_windows_failed"] == 1
        and counters["n_deaths_boundary"] == 1
        and counters["n_tracks"] == 2
        and out[2]["n_proposals"] == 1  # no phantom recovery beside the detection
        and out[2]["box_sources"] == ["yolo"]
        and out[2]["track_ids"] == [1]  # the post-boundary identity, and only it
        and counters["n_recovered"] == 0
    )
    check("failed window retires its tracks: no duplicate identity, no phantom recovery", ok,
          f"failed={counters['n_windows_failed']} retired={counters['n_deaths_boundary']} "
          f"kf2_n={out[2]['n_proposals']} ids={out[2]['track_ids']} "
          f"recovered={counters['n_recovered']}")

    # (13) F5: the boundary keyframe of a window that FAILED is still a frame, and
    # the next window is the one that emits it. A camera-global "some window has
    # emitted" boolean starts every later window at frame 1, so kf-2 below used to
    # vanish from the artifact permanently even though window 2 propagated live
    # tracks through it.
    rows = [_st_row(i, [box_a], [car], [0.9]) for i in range(4)]
    windows = [_st_window(0, 2), _st_window(1, 2), _st_window(2, 2)]
    _, counters, tracks12 = _st_drive(cfg, rows, windows, _StFailingTracker(fail_init={1}))
    tokens = [row["sample_data_token"] for row in tracks12]
    expected = ["kf-0", "sw-0-0", "sw-0-1", "kf-1", "kf-2", "sw-2-0", "sw-2-1", "kf-3"]
    ok = (
        sorted(tokens) == sorted(expected)  # every frame of a surviving window, once
        and len(tokens) == len(set(tokens))
        and counters["n_windows_failed"] == 1
        and counters["n_12hz_rows"] == len(tracks12)
    )
    check("12 Hz after a failed window: the boundary keyframe is emitted, still exactly once", ok,
          f"missing={sorted(set(expected) - set(tokens))} extra={sorted(set(tokens) - set(expected))}")

    # (14) F6: the window's CLOSING keyframe carries detector evidence too.
    # Labelling it "propagated" told a human auditor the detector had lost an
    # object it in fact re-detected at every single keyframe.
    rows = [_st_row(i, [box_a], [car], [0.9]) for i in range(3)]
    _, _, tracks12 = _st_drive(cfg, rows, [_st_window(0, 1), _st_window(1, 1)], FakeTracker())
    evidence_of = {row["sample_data_token"]: row["evidence"] for row in tracks12}
    ok = (
        evidence_of.get("kf-0") == "yolo"
        and evidence_of.get("kf-1") == "yolo"  # window 0's closing keyframe
        and evidence_of.get("kf-2") == "yolo"  # window 1's closing keyframe
        and evidence_of.get("sw-0-0") == "propagated"  # nothing was detected there
    )
    check("12 Hz evidence: a keyframe the detector confirmed reads 'yolo', not 'propagated'", ok,
          f"labels={ {k: evidence_of.get(k) for k in ('kf-0', 'sw-0-0', 'kf-1', 'kf-2')} }")

    # (15) F7: a track the capacity rule evicts mid-window keeps the rows for the
    # frames it was live, prompted and observed through. Emission runs at window
    # END and skipped every evicted id outright, so an eviction at frame k
    # retroactively deleted that track's frames 0..k-1.
    cfg_cap = Track2DConfig(revision="self-test", max_active_tracks_per_camera=1)
    stronger = SweepDetection(box_xyxy_px=(400.0, 400.0, 500.0, 500.0), score=0.95,
                              class_name=car, phrase_char_span=(0, 5),
                              nuscenes_categories=("vehicle.car",))
    rows = [_st_row(0, [box_a], [car], [0.5]), _st_row(1, [], [], [])]
    window = _st_window(0, 2)
    _, counters, tracks12 = _st_drive(
        cfg_cap, rows, [window], FakeTracker(), {window.frames[1].sample_data_token: [stronger]}
    )
    seen = {(row["track_id"], row["sample_data_token"]) for row in tracks12}
    ok = (
        counters["n_tracks_dropped_overflow"] == 1
        and (0, "kf-0") in seen  # live, prompted and observed before the eviction
        and (0, "sw-0-0") in seen  # the frame the eviction itself happened on
        and (0, "sw-0-1") not in seen  # silent from the next frame on
        and (0, "kf-1") not in seen
        and (1, "sw-0-0") in seen  # the mid-gap birth that displaced it
    )
    check("capacity eviction does not rewrite history: the evicted track keeps its earlier rows",
          ok, f"evicted={counters['n_tracks_dropped_overflow']} "
              f"track0={sorted(t for i, t in seen if i == 0)}")

    # (16) F8: close_video failing is a LEAKED SESSION, not a failed window. It
    # used to increment n_windows_failed while the window's injections and 12 Hz
    # rows were emitted anyway — a window simultaneously failed and trusted.
    rows = [_st_row(0, [box_a], [car], [0.9]), _st_row(1, [], [], [])]
    out, counters, tracks12 = _st_drive(
        cfg, rows, [_st_window(0, 1)], _StFailingTracker(fail_close={0})
    )
    ok = (
        counters["n_windows_failed"] == 0
        and counters["n_sessions_unclosed"] == 1
        and out[1]["box_sources"] == ["recovered"]  # the window's output stands
        and counters["n_12hz_rows"] == len(tracks12) > 0
    )
    check("close_video failure: counted as an unclosed session, the window is not called failed",
          ok, f"failed={counters['n_windows_failed']} unclosed={counters['n_sessions_unclosed']} "
              f"sources={out[1]['box_sources']} rows12={len(tracks12)}")

    # (17) F11: a --scenes subset re-run must not stand a fresh marker and a fresh
    # config echo over scene dirs another configuration wrote — Stage 4 consumes
    # every directory under scenes/ as one upstream. Refused before any write; an
    # incremental re-run under the IDENTICAL config is legitimate and passes.
    import tempfile  # local: the only place the self-test touches a filesystem

    cfg_refine = Track2DConfig(revision="self-test", refine_matched_boxes=True)
    with tempfile.TemporaryDirectory() as tmp_dir:
        os.makedirs(os.path.join(tmp_dir, "scenes", "scene-0061"))
        os.makedirs(os.path.join(tmp_dir, "scenes", "scene-0103"))
        bare = refuses(lambda: fence_out_dir(tmp_dir, ["scene-0061"], cfg, "sam3_tracker_video"))
        write_json_atomic(
            os.path.join(tmp_dir, "run_manifest.json"),
            {"spec": STAGE_SPEC, "provider": "sam3_tracker_video", "config": cfg.as_dict()},
        )
        differs = refuses(
            lambda: fence_out_dir(tmp_dir, ["scene-0061"], cfg_refine, "sam3_tracker_video")
        )
        same = fence_out_dir(tmp_dir, ["scene-0061"], cfg, "sam3_tracker_video")
        full = fence_out_dir(tmp_dir, ["scene-0061", "scene-0103"], cfg_refine,
                             "sam3_tracker_video")
    ok = (
        "scene-0103" in bare and "--out-dir" in bare  # no manifest can be quoted for it
        and "scene-0103" in differs and "--out-dir" in differs  # refine flag differs
        and same["foreign_scene_dirs"] == ["scene-0103"]  # identical config: allowed
        and same["scenes_in_out_dir"] == ["scene-0061", "scene-0103"]
        and full["foreign_scene_dirs"] == []  # every dir is rewritten by this run
    )
    check("out dir fence: a subset re-run over another config's scenes refuses, an identical "
          "one is allowed and recorded", ok,
          f"bare={bool(bare)} differs={bool(differs)} same={same} full={full}")

    # (18) D1: nothing bounded a propagated box from ABOVE. min_recovered_mask_px
    # and min_recovered_box_side_px are floors, and refine_area_band guards only
    # the refine path (off by default), so a mask that leaked off its object was
    # injected as a detection-shaped proposal carrying full class and score
    # provenance — and then SEEDED the next window, which is what made it
    # cascade. Reproduced on the 2026-08-19 trial, scene-0061/CAM_FRONT_LEFT
    # track 0: a truck leaving frame right became [0, 0, 1600, 900] @ 0.494 at
    # 3.07x its parent's area, then [0, 0, 990, 900] one hop later.
    parent = (700.0, 300.0, 1080.0, 680.0)  # 380 x 380 = 144 400 px^2
    leak = (0.0, 0.0, float(IMAGE_WIDTH_PX), float(IMAGE_HEIGHT_PX))
    rows = [_st_row(0, [parent], [truck], [0.8]), _st_row(1, [], [], []), _st_row(2, [], [], [])]
    # The leak is scripted onto frame 3, which only the FIRST window reaches (4
    # frames against the second's 3), so the second window is left carrying
    # whatever seed the first one left behind — which is the thing under test.
    windows = [_st_window(0, 2), _st_window(1, 1)]
    out, counters, tracks12 = _st_drive(cfg, rows, windows, FakeTracker({(3, 0): leak}))
    flag_of = {(row["track_id"], row["sample_data_token"]): row["oversize"] for row in tracks12}
    ok = (
        out[1]["n_proposals"] == 0  # the leaked box is not injected
        and counters["n_rejected_oversize"] == 1  # it is counted
        and out[2]["box_sources"] == ["recovered"]  # the track lives on and recovers
        and out[2]["boxes_xyxy_px"] == [list(parent)]  # re-prompted from the last DETECTED box
        and out[2]["n_propagated_hops"] == [2]  # it coasted THROUGH the leak, not from it
        and counters["n_recovered"] == 1
        and counters["n_tracks"] == 1  # no second identity anywhere
        and flag_of[(0, "kf-1")].startswith("frame_fraction=")  # the artifact says which bound
        and flag_of[(0, "kf-0")] == ""  # and stays quiet about the plausible frames
    )
    check("oversize leak: rejected, counted, and NOT seeded — the next window re-prompts from the "
          "last detected box", ok,
          f"kf1_n={out[1]['n_proposals']} rejected={counters['n_rejected_oversize']} "
          f"kf2_box={out[2]['boxes_xyxy_px']} hops={out[2]['n_propagated_hops']} "
          f"flag={flag_of.get((0, 'kf-1'))!r}")

    # (19) the other side of the same bound: an object genuinely approaching the
    # camera grows across one 0.5 s hop, and must still be recovered AND still
    # seed the next window. 2.39x its parent's area at 0.24 of the frame — just
    # inside both bounds — so this is what pins the two defaults from below.
    grown = (600.0, 200.0, 1188.0, 788.0)  # 588 x 588 = 345 744 px^2
    out, counters, _ = _st_drive(cfg, rows, windows, FakeTracker({(3, 0): grown}))
    ratio = _area(grown) / _area(parent)
    fraction = _area(grown) / (float(IMAGE_WIDTH_PX) * float(IMAGE_HEIGHT_PX))
    ok = (
        ratio < cfg.max_recovered_area_ratio  # the scenario really is inside the bounds
        and fraction < cfg.max_recovered_frame_fraction
        and counters["n_rejected_oversize"] == 0
        and out[1]["box_sources"] == ["recovered"]
        and out[1]["boxes_xyxy_px"] == [list(grown)]
        and out[1]["n_propagated_hops"] == [1]
        and out[2]["boxes_xyxy_px"] == [list(grown)]  # the plausible box DID become the seed
        and counters["n_recovered"] == 2
    )
    check(f"legitimate growth at {ratio:.2f}x parent / {fraction:.2f} of frame: still recovered, "
          "still seeds the next window", ok,
          f"rejected={counters['n_rejected_oversize']} kf1={out[1]['boxes_xyxy_px']} "
          f"kf2={out[2]['boxes_xyxy_px']} recovered={counters['n_recovered']}")

    # (20) D2: Stage 4 sources `upstream.degraded` from its IMMEDIATE upstream
    # marker, so a Stage 3b that consumed a DEGRADED Stage 3 and wrote a plain
    # _SUCCESS LAUNDERED the chain — every stage after it, and every Results/
    # run_config quoting them, lost the C16 flag. Reproduced on the 2026-08-19
    # trial: its own manifest records upstream.degraded=true with 10 causes, next
    # to a clean _SUCCESS. The round trip below is the real one — this stage's
    # own C16 gate in, `write_marker` out, and the gate masks.py calls reading it
    # back — because the claim is about what the NEXT stage sees, not about a
    # string this stage formatted.
    fingerprint = "self-test-metadata-fingerprint"
    stage3_causes = ["scene-0061: 2 sectors rejected", "scene-0553: 1 sector rejected"]
    with tempfile.TemporaryDirectory() as tmp_dir:
        stage3_dir = os.path.join(tmp_dir, "stage3_proposals")
        write_json_atomic(
            os.path.join(stage3_dir, "run_manifest.json"),
            {"spec": "dhakascenes-pilot/stage3_proposals/v1",
             "image_size_px": [IMAGE_WIDTH_PX, IMAGE_HEIGHT_PX]},
        )
        write_marker(stage3_dir, fingerprint, degraded=True, causes=stage3_causes)
        without_opt_in = refuses(lambda: load_upstream(stage3_dir))
        _, upstream_marker = load_upstream(stage3_dir, accept_degraded=True)
        carried = inherited_degradation_causes(upstream_marker)

        # What run() and main() then do with it. `degraded=bool(carried)` is the
        # case under test — a run whose own windows were all clean — and is
        # exactly run()'s `if inherited: degraded = True`.
        out_3b = os.path.join(tmp_dir, STAGE)
        write_json_atomic(os.path.join(out_3b, "run_manifest.json"), {"spec": STAGE_SPEC})
        marker_path = write_marker(out_3b, fingerprint, degraded=bool(carried), causes=carried)
        state_is_degraded = os.path.basename(marker_path) == MARKER_DEGRADED
        no_clean_marker = not os.path.exists(os.path.join(out_3b, MARKER_CLEAN))
        # Stage 4's gate, called the way masks.py calls it.
        stage4_refusal = refuses(
            lambda: require_upstream(out_3b, stage_name="Stage 3b",
                                     module_hint="pipeline.stage3b_track2d.track2d")
        )
        _, marker_3b = require_upstream(
            out_3b, stage_name="Stage 3b", module_hint="pipeline.stage3b_track2d.track2d",
            accept_degraded=True,
        )
        # A CLEAN upstream must still produce a clean marker: this fix inherits a
        # flag, it does not manufacture one.
        clean_dir = os.path.join(tmp_dir, "stage3_clean")
        write_json_atomic(
            os.path.join(clean_dir, "run_manifest.json"),
            {"spec": "dhakascenes-pilot/stage3_proposals/v1",
             "image_size_px": [IMAGE_WIDTH_PX, IMAGE_HEIGHT_PX]},
        )
        write_marker(clean_dir, fingerprint, degraded=False)
        _, clean_marker = load_upstream(clean_dir)
        inherited_from_clean = inherited_degradation_causes(clean_marker)
    ok = (
        "--accept-degraded-upstream" in without_opt_in  # C16's opt-in is still required
        and carried == [INHERITED_CAUSE_PREFIX + cause for cause in stage3_causes]
        and state_is_degraded and no_clean_marker  # the house state, not a new one
        and marker_3b.degraded  # what masks.py's require_upstream reads
        and list(marker_3b.causes) == carried  # ... with the inherited causes on it
        and "--accept-degraded-upstream" in stage4_refusal  # ... and it refuses without the opt-in
        and inherited_from_clean == []
    )
    check("degraded upstream: Stage 3b's own marker is _SUCCESS.degraded and carries the "
          "inherited causes into Stage 4's gate", ok,
          f"marker={os.path.basename(marker_path)} degraded={marker_3b.degraded} "
          f"causes={list(marker_3b.causes)} clean_upstream_inherits={inherited_from_clean}")

    width = max(len(name) for name, _, _ in results)
    failed = 0
    for name, ok, detail in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name.ljust(width)}" + ("" if ok else f"   [{detail}]"))
        failed += 0 if ok else 1
    print(f"{len(results) - failed}/{len(results)} scenarios passed")
    return EXIT_OK if failed == 0 else EXIT_DEGRADED


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # Env defaults are the .env contract (C17): a declared key either has a
    # reader or is removed — this is the reader.
    parser.add_argument(
        "--paths", default=os.environ.get("DHAKASCENES_PATHS_CONFIG", "configs/paths.yaml")
    )
    parser.add_argument("--stage3-dir", default=None, help="default <work_root>/stage3_proposals")
    parser.add_argument("--out-dir", default=None, help=f"default <work_root>/{STAGE}")
    parser.add_argument("--model-id", default=None, help="video tracker hub id; default facebook/sam3")
    parser.add_argument(
        "--provider", default=None, choices=("", "sam3_tracker_video", "sam31_multiplex"),
        help="adapter override; default inferred from --model-id",
    )
    parser.add_argument("--revision", default=None, help="hub git sha for the checkpoint; required")
    parser.add_argument("--scenes", nargs="*", default=None, help="subset of Stage 3 scene names")
    parser.add_argument("--seed", type=int, default=None, help="override the global seed (recorded)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--video-storage-device", default=None, help="default: --device")
    parser.add_argument(
        "--refine-boxes", action="store_true",
        help="replace a detected box with its propagated mask's tight box when they agree (A/B)",
    )
    parser.add_argument(
        "--reverse-pass", action="store_true",
        help="gated to a later revision (C27); refuses rather than running",
    )
    parser.add_argument(
        "--no-sweep-detection", action="store_true",
        help="skip Phase A: no detector on the sweep frames, no mid-gap births",
    )
    parser.add_argument("--no-12hz-tracks", action="store_true", help="skip tracks_12hz.jsonl")
    parser.add_argument(
        "--accept-degraded-upstream", action="store_true",
        help="consume a DEGRADED (complete, quality-flagged) Stage 3 output; recorded (C16)",
    )
    parser.add_argument(
        "--self-test", action="store_true",
        help="run the pure-logic scenarios (no GPU, no substrate) and exit",
    )
    args = parser.parse_args(argv)

    if args.self_test:
        return _self_test()

    try:
        paths = load_paths(args.paths)
    except PathValidationError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_REFUSED

    stage3_dir = args.stage3_dir or os.path.join(paths.work_root, "stage3_proposals")
    out_dir = args.out_dir or os.path.join(paths.work_root, STAGE)
    assert_dataroot_read_only(paths, out_dir)

    cfg = Track2DConfig(
        device=args.device,
        video_storage_device=args.video_storage_device or args.device,
        revision=args.revision,
        refine_matched_boxes=args.refine_boxes,
        reverse_pass=args.reverse_pass,
        detect_on_sweeps=not args.no_sweep_detection,
        emit_12hz_tracks=not args.no_12hz_tracks,
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
    except (Track2DContractError, RoleContractError, SequenceError, PathValidationError) as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_REFUSED

    write_json_atomic(os.path.join(out_dir, "run_manifest.json"), manifest)
    # Three-state marker (§1.9, C16): clean / degraded-with-causes / absent.
    write_marker(
        out_dir,
        manifest["upstream"]["metadata_fingerprint"],
        degraded=code == EXIT_DEGRADED,
        # Run-level causes have no scene to hang from — an unload that failed
        # after the last scene is the whole reason this run is degraded (F2) —
        # so the marker carries both lists or the degradation has no stated cause.
        causes=[cause for scene in manifest["scenes"] for cause in scene["causes"]]
        + list(manifest.get("run_causes") or []),
    )

    t = manifest["totals"]
    print(f"keyframe rows        : {t['n_camera_rows']}  ({t['n_keyframes']} keyframes)")
    print(f"detected boxes       : {t['n_yolo_boxes']}")
    print(f"recovered boxes      : {t['n_recovered']}  (decay {cfg.recovered_score_decay}^hops, "
          f"{t['n_rejected_oversize']} rejected oversize)")
    print(f"refined boxes        : {t['n_refined']}  (refine_matched_boxes={cfg.refine_matched_boxes})")
    print(f"tracks / deaths      : {t['n_tracks']} / {t['n_deaths']}  "
          f"({t['n_deaths_boundary']} retired at block boundaries)")
    print(f"windows              : {t['n_windows']}  ({t['n_windows_skipped_empty']} skipped, "
          f"{t['n_windows_failed']} failed, {t['n_windows_missing_anchor']} missing anchors, "
          f"{t['n_sessions_unclosed']} unclosed sessions)")
    print(f"contests lost / xclass: {t['n_lost_contest']} / {t['n_suppressed_cross_class']}")
    print(f"sweep images / dets  : {t['n_sweep_images']} / {t['n_sweep_detections']}  "
          f"({t['n_midgap_births']} mid-gap births, {t['n_midgap_injected']} of them injected, "
          f"{t['n_repropagations']} re-propagations)")
    print(f"12 Hz rows           : {t['n_12hz_rows']}")
    print(f"wrote {out_dir}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
