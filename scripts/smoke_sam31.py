#!/usr/bin/env python3
"""C26 GO/NO-GO smoke: does SAM 3.1 multiplex do what Stage 4 needs, on this card?

A gate, not a demo. It measures and asserts; it never assumes. Four things must
hold before C26 can be adopted, and each fails silently if left unchecked:

  1. The checkpoint is the pinned one — `REVISION` is a hub git sha and the file
     is stream-hashed and ASSERTED equal to the adapter's pinned digest, so the
     run is quotable against bytes, not against a repo name that tracks a moving
     branch and not against whatever file `--checkpoint` happened to name (§7.2).
  2. Box prompts arrive as box prompts — SAM 3.1's box prompt is SAM-2's
     corner-pair encoding (labels 2/3), not `bounding_boxes=`. Both spellings
     return masks; only one is per-instance.
  3. Masks come back in the source coordinate space — a rel/abs mix-up yields
     good masks in the wrong place, so every non-empty mask's tight box is
     checked against the box that prompted it (§1.5 rules 2 and 4).
  4. The card comes back — peak VRAM measured two ways, larger believed, and
     teardown asserted rather than hoped for (§7.1).

Modes: `harness` drives `sam3` directly (C26 step A3); `adapter` drives
pipeline.stage4_masks.masks.Sam31MultiplexAdapter through the identical
contracts (C26 step A4) and exits 2 until that adapter lands.

Exit codes: 0 pass. 2 refused (import failure, checkpoint unresolvable, no card,
adapter absent). 3 VRAM peak >= the cap apply_vram_cap() reported, cap-induced
OOM included. 4 API drift / contract failure. 5 teardown failure.

Usage:
    python3 scripts/smoke_sam31.py [--mode harness|adapter] [--checkpoint PATH]
"""

from __future__ import annotations

# --- allocator configuration, before any torch import ----------------------
# The caching allocator parses PYTORCH_CUDA_ALLOC_CONF once, at first CUDA init;
# setting it later is a no-op that leaves no trace. Same contract, enforced the
# same way, as scripts/measure_vram.py — so the two files' numbers compare.
import os as _os
import sys as _sys

ALLOC_CONF = "expandable_segments:True"
_existing = _os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
if "torch" in _sys.modules:
    raise SystemExit("smoke_sam31: torch was imported before PYTORCH_CUDA_ALLOC_CONF could be pinned")
if _existing and _existing != ALLOC_CONF:
    raise SystemExit(f"smoke_sam31: PYTORCH_CUDA_ALLOC_CONF is already {_existing!r}; this smoke is defined "
                     f"at {ALLOC_CONF!r} and will not silently measure another allocator configuration")
_os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ALLOC_CONF

import argparse, gc, hashlib, inspect, itertools, json, os, platform, sys, time  # noqa: E401,E402
from typing import Any, Sequence  # noqa: E402

import numpy as np  # noqa: E402
import yaml  # noqa: E402  (already a hard dependency of pipeline.common.model_interfaces)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.common.model_interfaces import (  # noqa: E402
    MASK_2D, ORIGINAL_SIZE_PX, PER_FRAME_WINDOW, CheckpointSpec, apply_vram_cap)

# The pinned SAM 3.1 release. An unpinned model_id tracks the hub default branch
# and can change under a recorded measurement with no manifest field moving (§7.2).
REVISION = "daa63191845a41281374e725f4c9e51c7a824460"
REPO_ID, CKPT_FILENAME = "facebook/sam3.1", "sam3.1_multiplex.pt"

MIB = 1024.0 * 1024.0
HASH_CHUNK_BYTES = 8 * 1024 * 1024      # the checkpoint is ~3.3 GB; never read whole
IMG_W, IMG_H = ORIGINAL_SIZE_PX

# The deterministic prompt grid: pure arithmetic, no RNG, so two runs prompt the
# same pixels and a change in the masks is a change in the model.
GRID_COLS, GRID_ROWS = 8, 4
BOX_W_PX, BOX_H_PX, MARGIN_PX = 150.0, 120.0, 20.0
VIDEO_BOX_INDICES = (9, 13, 18)         # into that same grid, so both contracts prompt the same pixels
# Forward from frame 0, backward from the last frame; same 3 boxes both ways.
VIDEO_PASSES = (("s_per_frame_video_fwd", "forward", 0), ("s_per_frame_video_rev", "backward", -1))

COORD_IOU_MIN = 0.25                    # tight(mask) vs prompt box: the rel/abs tripwire
QUIESCENT_TOLERANCE_MIB = 50.0          # post-teardown memory_allocated() ceiling


class SmokeFailure(RuntimeError):
    """A gate failed. Carries the exit code the failure maps to."""

    def __init__(self, code: int, reason: str) -> None:
        super().__init__(reason)
        self.code, self.reason = int(code), reason


# --- inputs: the pinned checkpoint, the real substrate, the grid -----------


def _sha256_file(path: str) -> str:
    """Stream-hash, chunked: the identity of the bytes that ran, not of a name."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(HASH_CHUNK_BYTES), b""):
            h.update(chunk)
    return h.hexdigest()


def _pinned_sha256() -> str | None:
    """The pinned digest, read off the adapter — one constant, never a second copy.

    Copying `Sam31MultiplexAdapter._EXPECTED_SHA256` into this file would make a
    second place for the pin to be wrong, and the two would then drift silently in
    exactly the direction this gate exists to catch. Imported LAZILY and PROBED
    rather than assumed, for the two reasons this script probes anything: the class
    lands with C26 step A4 and may not exist in this tree, and the constant is
    private surface in a file this script does not own. `None` means "no pin was
    obtainable" — a caller's decision to make, never a reason to invent one here.
    """
    try:
        from pipeline.stage4_masks.masks import Sam31MultiplexAdapter  # type: ignore
    except (ImportError, AttributeError):
        return None
    pinned = getattr(Sam31MultiplexAdapter, "_EXPECTED_SHA256", None)
    return str(pinned) if pinned else None


def _resolve_checkpoint(override: str | None) -> str:
    """The pinned revision out of the env's HF_HOME cache. No unpinned fetch."""
    if override:
        if not os.path.isfile(override):
            raise SmokeFailure(2, f"--checkpoint {override!r} is not a file")
        return os.path.realpath(override)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SmokeFailure(2, f"huggingface_hub is not importable ({exc})")
    try:
        return hf_hub_download(repo_id=REPO_ID, filename=CKPT_FILENAME, revision=REVISION)
    except Exception as exc:
        raise SmokeFailure(2, f"cannot resolve {REPO_ID}@{REVISION[:12]}/{CKPT_FILENAME}: {exc}. Source .env "
                              "so HF_HOME points at the local cache; this smoke never fetches unpinned")


def _substrate(paths_yaml: str) -> tuple[str, str]:
    """(dataroot, version) off the §1.8 path contract, never re-derived from cwd."""
    with open(paths_yaml, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    root = str(raw.get("dataroot", "")).strip()
    if not root or not os.path.isdir(root):
        raise SmokeFailure(2, f"{paths_yaml}: dataroot={root!r} is not a directory")
    return root, str(raw.get("version", "")).strip()


def _open_1600x900(path: str):
    from PIL import Image

    with Image.open(path) as handle:
        image = handle.convert("RGB")
    if image.size != (IMG_W, IMG_H):
        raise SmokeFailure(2, f"{path} is {image.size[0]}x{image.size[1]}; this smoke is defined at 1600x900")
    return image


def _first_keyframe(dataroot: str):
    """The lexicographically first CAM_FRONT keyframe: one real 1600x900 frame."""
    cam_dir = os.path.join(dataroot, "samples", "CAM_FRONT")
    names = sorted(n for n in os.listdir(cam_dir) if n.lower().endswith(".jpg")) if os.path.isdir(cam_dir) else []
    if not names:
        raise SmokeFailure(2, f"no CAM_FRONT keyframes under {cam_dir}")
    path = os.path.join(cam_dir, names[0])
    return _open_1600x900(path), path


def _consecutive_frames(dataroot: str, version: str, n_frames: int):
    """`n_frames` temporally consecutive CAM_FRONT frames from the middle of the log.

    The channel filter carries a trailing slash deliberately: 'samples/CAM_FRONT'
    without it also matches CAM_FRONT_LEFT and CAM_FRONT_RIGHT, and a "video"
    interleaved from three cameras would fail propagation for a reason that has
    nothing to do with SAM 3.1. samples/ (2 Hz) and sweeps/ (12 Hz) both count:
    consecutive means consecutive in time.
    """
    table = os.path.join(dataroot, version, "sample_data.json")
    if not os.path.isfile(table):
        raise SmokeFailure(2, f"no sample_data.json at {table}")
    with open(table, "r", encoding="utf-8") as fh:
        records = json.load(fh)
    cam = [r for r in records
           if str(r.get("filename", "")).startswith(("samples/CAM_FRONT/", "sweeps/CAM_FRONT/"))]
    cam.sort(key=lambda r: int(r["timestamp"]))
    if len(cam) < n_frames:
        raise SmokeFailure(2, f"only {len(cam)} CAM_FRONT records in {table}; need {n_frames}")
    start = min(len(cam) // 2, len(cam) - n_frames)
    paths = [os.path.join(dataroot, r["filename"]) for r in cam[start : start + n_frames]]
    for path in paths:
        if not os.path.isfile(path):
            raise SmokeFailure(2, f"sample_data.json references a missing blob: {path}")
    return [_open_1600x900(p) for p in paths], paths


def _grid_boxes(n_boxes: int) -> np.ndarray:
    """(n, 4) xyxy in absolute 1600x900 pixels — the §1.5 2D contract."""
    capacity = GRID_COLS * GRID_ROWS
    if not 1 <= n_boxes <= capacity:
        raise SmokeFailure(2, f"--n-boxes {n_boxes} is outside 1..{capacity}, the {GRID_COLS}x{GRID_ROWS} grid")
    step_x = (IMG_W - 2 * MARGIN_PX - BOX_W_PX) / (GRID_COLS - 1)
    step_y = (IMG_H - 2 * MARGIN_PX - BOX_H_PX) / (GRID_ROWS - 1)
    xy = [(MARGIN_PX + (i % GRID_COLS) * step_x, MARGIN_PX + (i // GRID_COLS) * step_y) for i in range(n_boxes)]
    return np.asarray([[x, y, x + BOX_W_PX, y + BOX_H_PX] for x, y in xy], dtype=np.float32)


# --- the contracts: shared by both modes, only the runner differs ----------


def _tight_box(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    iw, ih = max(0.0, min(a[2], b[2]) - max(a[0], b[0])), max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def assert_image_contract(obj_ids: np.ndarray, masks: np.ndarray, boxes: np.ndarray) -> int:
    """One mask per box, bool, at 1600x900, in the coordinate space of the prompt.

    Absent ids are zero-filled and allowed: SAM drops zero-area masks, and a grid
    box over empty sky legitimately segments nothing. An id *outside* the prompted
    range is not — that is the detector path leaking objects nobody asked for.
    """
    n, masks = len(boxes), np.asarray(masks)
    ids = [int(v) for v in np.asarray(obj_ids).reshape(-1).tolist()]
    if masks.dtype != np.bool_:
        raise SmokeFailure(4, f"API drift: out_binary_masks dtype is {masks.dtype}, expected bool")
    if masks.ndim != 3 or masks.shape[1:] != (IMG_H, IMG_W):
        raise SmokeFailure(4, f"API drift: masks shape {masks.shape}, expected (N, {IMG_H}, {IMG_W})")
    if len(ids) != masks.shape[0]:
        raise SmokeFailure(4, f"API drift: {len(ids)} obj ids for {masks.shape[0]} masks")
    if sorted(set(ids) - set(range(n))):
        raise SmokeFailure(4, f"API drift: unprompted obj ids {sorted(set(ids) - set(range(n)))} in the "
                              f"output (prompted 0..{n - 1})")
    by_id = {oid: masks[k] for k, oid in enumerate(ids)}
    n_nonempty = 0
    for i in range(n):
        tight = _tight_box(by_id[i]) if i in by_id else None
        if tight is None:
            continue                    # absent, or present and empty: both allowed
        n_nonempty += 1
        iou = _iou(tight, boxes[i].tolist())
        if iou <= COORD_IOU_MIN:
            # A rel/abs mix-up produces a perfectly good mask somewhere else.
            raise SmokeFailure(4, f"coordinate-space failure at box {i}: IoU(tight, prompt)={iou:.3f} <= "
                                  f"{COORD_IOU_MIN}; tight={[round(v, 1) for v in tight]}, "
                                  f"prompt={[round(v, 1) for v in boxes[i].tolist()]}")
    if n_nonempty < n // 2:
        raise SmokeFailure(4, f"only {n_nonempty}/{n} boxes segmented anything (floor {n // 2}); grid boxes "
                              "over sky and road may legitimately be empty, below half is systematic")
    return n_nonempty


def assert_video_contract(collected: Sequence[tuple[int, Any]], n_frames: int, n_objs: int, label: str) -> None:
    """>= n_frames-1 of n_frames carry every prompted object at 1600x900, bool."""
    expected, good = set(range(n_objs)), 0
    for _, out in collected:
        if not isinstance(out, dict):
            continue
        masks = np.asarray(out["out_binary_masks"])
        ids = {int(v) for v in np.asarray(out["out_obj_ids"]).reshape(-1).tolist()}
        if masks.dtype == np.bool_ and masks.ndim == 3 and masks.shape[1:] == (IMG_H, IMG_W):
            good += expected.issubset(ids)
    if good < n_frames - 1:
        raise SmokeFailure(4, f"{label}: only {good}/{n_frames} frames carry all {n_objs} objects as "
                              f"({IMG_H}, {IMG_W}) bool masks ({len(collected)} yielded); propagation is "
                              "not holding the prompted instances")


# --- VRAM and teardown: same policy as measure_vram.py ---------------------


class VramWatch:
    """`max_memory_reserved()` misses cuBLAS/cuDNN workspaces, kernel modules and
    every other process on the card; the `mem_get_info()` free-delta sees all of
    it. On a capped device the gap between them IS the margin being gated."""

    def __init__(self, torch, index: int) -> None:
        self._torch, self._index = torch, index
        torch.cuda.reset_peak_memory_stats(index)
        self.free_baseline, self.total = torch.cuda.mem_get_info(index)
        self.free_low = self.free_baseline

    def sample(self) -> None:
        self.free_low = min(self.free_low, self._torch.cuda.mem_get_info(self._index)[0])

    @property
    def reserved_mib(self) -> float:
        return self._torch.cuda.max_memory_reserved(self._index) / MIB

    @property
    def mem_get_info_mib(self) -> float:
        return (self.free_baseline - self.free_low) / MIB


def _record_vram(watch: VramWatch, summary: dict) -> None:
    summary["vram_peak_mib_reserved"] = round(watch.reserved_mib, 1)
    summary["vram_peak_mib_mem_get_info"] = round(watch.mem_get_info_mib, 1)
    peak, cap = max(watch.reserved_mib, watch.mem_get_info_mib), summary["vram_cap"]["value_mib"]
    print(f"[vram] reserved={watch.reserved_mib:.1f} MiB  free_delta={watch.mem_get_info_mib:.1f} MiB  cap={cap}")
    if cap is not None and peak >= float(cap):
        raise SmokeFailure(3, f"VRAM peak {peak:.1f} MiB >= cap {cap} MiB ({summary['vram_cap']['enforced']})")


def _autocast_enabled(torch) -> bool:
    # torch >= 2.4 wants a device type; the no-arg form is the legacy spelling.
    try:
        return bool(torch.is_autocast_enabled("cuda"))
    except TypeError:
        return bool(torch.is_autocast_enabled())


def _release(torch, index: int, summary: dict, holder: dict, method: str) -> None:
    """Drop the model, then assert the card is back and nothing global stayed on.

    Called from a `finally`, so the assertions are skipped while another failure
    is already propagating: the first failure in sequence order wins the exit
    code, and teardown itself runs either way.
    """
    pending = sys.exc_info()[0] is not None
    resident = holder.pop("m")
    getattr(resident, method)()
    # C26 compensation, applied here and in Sam31MultiplexAdapter.unload(): upstream
    # fixed the bf16 leak for Sam3VideoPredictor.shutdown (sam3_video_predictor.py:99
    # exits model.tracker.bf16_context) but Sam3MultiplexVideoPredictor inherits the
    # BASE shutdown (sam3_base_predictor.py:482, sessions only) and enters
    # self.bf16_context at __init__ (sam3_multiplex_video_predictor.py:51) — so at
    # sha 8f0b7f4 the multiplex predictor leaks the autocast. Exit it exactly the way
    # upstream's own fix does; in adapter mode the adapter has already done this and
    # the getattr is a no-op.
    # The context is entered TWICE on this stack: once by the model wrapper
    # (Sam3MultiplexTrackingWithInteractivity, sam3_multiplex_base.py:2944) and
    # once by the predictor (sam3_multiplex_video_predictor.py:51). Autocast
    # nests, so every entered context must be exited — walk the chain and exit
    # each object's OWN context (vars(), not getattr: the wrapper proxies
    # attribute lookups to its inner model and would alias them).
    n_exited = 0
    seen_ctx_ids = set()
    obj = resident
    for _ in range(4):
        if obj is None:
            break
        # vars() for the CONTEXT lookup (the wrapper proxies unknown attributes
        # to its inner model and would alias contexts), but getattr for the
        # "model" hop (nn.Module keeps submodules in _modules, not __dict__).
        ctx = vars(obj).get("bf16_context") if hasattr(obj, "__dict__") else None
        if ctx is not None and id(ctx) not in seen_ctx_ids:
            seen_ctx_ids.add(id(ctx))
            try:
                ctx.__exit__(None, None, None)
                n_exited += 1
            finally:
                obj.bf16_context = None
        obj = getattr(obj, "model", None)
    if n_exited:
        print(f"API-NOTE: exited {n_exited} leaked bf16_context(s) past shutdown() "
              "(sam3_base_predictor.py:482 clears sessions only; contexts entered at "
              "sam3_multiplex_base.py:2944 and sam3_multiplex_video_predictor.py:51). "
              "Compensated here as the adapter's unload() does; upstream fixed only "
              "Sam3VideoPredictor (sam3_video_predictor.py:99).")
        summary["bf16_leak_compensated"] = True
        summary["bf16_contexts_exited"] = n_exited
    if _autocast_enabled(torch):
        # Unbalanced enters beyond the discoverable contexts: restore the end
        # state directly. Recorded, never silent.
        try:
            torch.set_autocast_enabled("cuda", False)
        except TypeError:
            torch.set_autocast_enabled(False)
        summary["bf16_forced_off"] = True
        print("API-NOTE: autocast still enabled after exiting all discoverable "
              "bf16_contexts; forced off via torch.set_autocast_enabled.")
    # The autocast weight cache (bf16 copies of fp32 weights, ~1.3 GiB here) is
    # normally dropped when nesting reaches zero — which unbalanced enters
    # prevent. Clear it explicitly or memory_allocated() never quiesces.
    torch.clear_autocast_cache()
    del resident, obj
    gc.collect()
    torch.cuda.synchronize(index)
    torch.cuda.empty_cache()
    allocated_mib = torch.cuda.memory_allocated(index) / MIB
    autocast = _autocast_enabled(torch)
    summary["allocated_after_teardown_mib"] = round(allocated_mib, 1)
    summary["autocast_enabled_after_teardown"] = autocast
    if pending:
        return
    if allocated_mib >= QUIESCENT_TOLERANCE_MIB:
        raise SmokeFailure(5, f"memory_allocated()={allocated_mib:.1f} MiB >= {QUIESCENT_TOLERANCE_MIB} MiB "
                              "after teardown; empty_cache() does not free memory a live Python object holds")
    if autocast:
        # A leaked autocast silently changes the dtype of every later model in
        # this process — including any role measured after this one. Reaching
        # here means the compensation above did NOT restore the state.
        raise SmokeFailure(5, "bf16 autocast still enabled after compensated shutdown")


def _drive(args, torch, summary: dict, build, method: str, image_fn, video_fn, boxes, video_boxes) -> None:
    """Build, run both contracts, gate VRAM, tear down. Identical in both modes.

    `build` returns the resident object and `method` names its teardown call, so
    harness and adapter differ only in those two and in the two runner callables.
    """
    index = torch.cuda.current_device()
    torch.manual_seed(args.seed)
    watch = VramWatch(torch, index)     # created before the build: the load peak counts
    holder: dict[str, Any] = {"m": build()}
    watch.sample()
    try:
        s_image, obj_ids, masks = image_fn(holder["m"], boxes)
        watch.sample()
        summary["s_per_image_32_boxes"] = round(s_image, 4)
        summary["n_nonempty_masks"] = assert_image_contract(obj_ids, masks, boxes)
        print(f"[image] {len(boxes)} boxes in {s_image:.2f} s, {summary['n_nonempty_masks']} non-empty")
        del obj_ids, masks
        for key, direction, prompt_frame in VIDEO_PASSES:
            elapsed, frames_out = video_fn(holder["m"], video_boxes,
                                           prompt_frame % args.video_frames, direction)
            watch.sample()
            assert_video_contract(frames_out, args.video_frames, len(video_boxes), direction)
            summary[key] = round(elapsed / max(1, len(frames_out)), 4)
            print(f"[video/{direction}] {len(frames_out)} frames, {summary[key]:.3f} s/frame")
            del frames_out              # `frame_stats` can carry device tensors
        _record_vram(watch, summary)
    finally:
        _release(torch, index, summary, holder, method)


# --- harness mode: sam3 driven directly ------------------------------------


_SESSION_COUNTER = itertools.count()


def _start_session_compat(predictor, frames: list) -> str:
    """start_session for the multiplex predictor, without the kwarg it rejects.

    API-NOTE (C26): Sam3BasePredictor.start_session passes offload_state_to_cpu
    unconditionally (sam3_base_predictor.py:126-131), but the multiplex model's
    init_state (sam3_multiplex_tracking.py:207-215) does not accept it ->
    TypeError. Mirror start_session's registration while filtering kwargs
    through the model's own signature — the same defensive pattern upstream's
    add_prompt uses (sam3_base_predictor.py:196-201). Session ids are a
    deterministic counter, not uuid4: §1.9.
    """
    init_kwargs = dict(resource_path=frames, offload_video_to_cpu=False,
                       offload_state_to_cpu=False)
    if hasattr(predictor, "async_loading_frames"):
        init_kwargs["async_loading_frames"] = predictor.async_loading_frames
    if hasattr(predictor, "video_loader_type"):
        init_kwargs["video_loader_type"] = predictor.video_loader_type
    valid = set(inspect.signature(predictor.model.init_state).parameters)
    state = predictor.model.init_state(**{k: v for k, v in init_kwargs.items() if k in valid})
    session_id = f"smoke-session-{next(_SESSION_COUNTER)}"
    now = time.time()
    predictor._all_inference_states[session_id] = {
        "state": state, "session_id": session_id, "start_time": now, "last_use_time": now,
    }
    # API-NOTE (C26): _build_sam2_output returns {} for any frame absent from
    # cached_frame_outputs (sam3_multiplex_tracking.py:1244-1245). Text-prompted
    # sessions cache every frame via the detector; an instance-only session
    # caches only the prompted frame, so propagation computes refined tracker
    # masks and then DISCARDS them at that gate (measured: 1/7 frames carried
    # objects). Seeding empty entries passes the gate; refined masks then merge
    # and real outputs overwrite the seeds as frames are visited.
    for _fi in range(state["num_frames"]):
        state["cached_frame_outputs"].setdefault(_fi, {})
    return session_id


def _harness_prompt(predictor, session_id: str, frame_idx: int, boxes: np.ndarray) -> Any:
    """Corner-pair box prompts. NEVER bounding_boxes= / boxes_xywh=.

    Labels 2 and 3 are SAM-2's top-left / bottom-right box encoding. The
    `boxes_xywh=` keyword routes to the semantic/exemplar path, which opens with
    `self.reset_state(inference_state)` (sam3_multiplex_tracking.py:1695) and
    discards every instance prompted so far — while still returning masks.
    """
    outputs = None
    for i, (x1, y1, x2, y2) in enumerate(boxes.tolist()):
        outputs = predictor.add_prompt(
            session_id=session_id, frame_idx=frame_idx, obj_id=i,
            points=[[x1 / IMG_W, y1 / IMG_H], [x2 / IMG_W, y2 / IMG_H]],
            point_labels=[2, 3], rel_coordinates=True)["outputs"]
        if outputs is None:
            raise SmokeFailure(4, f"add_prompt returned outputs=None at box {i} on frame {frame_idx}: "
                                  "max_num_objects exhausted, or the prompt was refused")
    return outputs


def _harness_image(torch, predictor, boxes: np.ndarray, image) -> tuple[float, np.ndarray, np.ndarray]:
    session_id = _start_session_compat(predictor, [image])
    try:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        outputs = _harness_prompt(predictor, session_id, 0, boxes)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    finally:
        predictor.close_session(session_id=session_id)
    return elapsed, np.asarray(outputs["out_obj_ids"]), np.asarray(outputs["out_binary_masks"])


def _harness_video(torch, predictor, boxes: np.ndarray, prompt_frame: int, direction: str,
                   frames) -> tuple[float, list[tuple[int, Any]]]:
    session_id = _start_session_compat(predictor, list(frames))
    try:
        _harness_prompt(predictor, session_id, prompt_frame, boxes)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        collected = [(int(item["frame_index"]), item["outputs"])
                     for item in predictor.handle_stream_request(
                         {"type": "propagate_in_video", "session_id": session_id,
                          "propagation_direction": direction, "start_frame_index": prompt_frame,
                          "max_frame_num_to_track": len(frames)})]
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    finally:
        predictor.close_session(session_id=session_id)
    return elapsed, collected


def run_harness(args, torch, summary: dict, image, frames, boxes, video_boxes, checkpoint: str) -> None:
    from sam3.model_builder import build_sam3_multiplex_video_predictor

    def build():
        # use_fa3=False is mandatory: FA3 kernels are fp8/Hopper-only and this is
        # an RTX 4090 (sm_89). The builder calls .cuda() itself, which is why the
        # VRAM cap is already applied by the time control reaches here.
        predictor = build_sam3_multiplex_video_predictor(
            checkpoint_path=checkpoint, max_num_objects=args.max_num_objects,
            use_fa3=False, compile=False, warm_up=False, async_loading_frames=False)
        # Hotstart buffering delays and de-duplicates newly appearing objects for
        # `hotstart_delay` frames. Right for detector proposals, wrong for box
        # prompts: a prompted instance must appear on the frame it was prompted on.
        if not hasattr(predictor.model, "hotstart_delay"):
            raise SmokeFailure(4, "API drift: hotstart_delay not found")
        predictor.model.hotstart_delay = 0
        # Second instance-mode knob (measured 2026-08-19): the builder ships
        # suppress_unmatched_only_within_hotstart=False, so keep-alive
        # suppression (sam3_multiplex_base.py:2301-2309) hides any tracklet the
        # detector does not re-confirm — and an instance-only session has no
        # detection prompt, so EVERY box-prompted tracklet is "unmatched" and
        # vanishes after a few frames (observed: 1/7 frames carried objects).
        # True limits that suppression to the hotstart window, which
        # hotstart_delay=0 makes empty: pure SAM2-style instance tracking.
        if not hasattr(predictor.model, "suppress_unmatched_only_within_hotstart"):
            raise SmokeFailure(4, "API drift: suppress_unmatched_only_within_hotstart not found")
        predictor.model.suppress_unmatched_only_within_hotstart = True
        summary["hotstart_delay"] = int(predictor.model.hotstart_delay)
        summary["suppress_unmatched_only_within_hotstart"] = bool(
            predictor.model.suppress_unmatched_only_within_hotstart)
        return predictor

    _drive(args, torch, summary, build, "shutdown",
           lambda p, b: _harness_image(torch, p, b, image),
           lambda p, b, f, d: _harness_video(torch, p, b, f, d, frames),
           boxes, video_boxes)


# --- adapter mode: the same contracts through the Stage 4 adapter ----------


def _adapter_image(torch, adapter, boxes: np.ndarray, image_np) -> tuple[float, np.ndarray, np.ndarray]:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = adapter.segment([image_np], boxes, state=None, window=PER_FRAME_WINDOW, channel="CAM_FRONT")
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    masks = np.asarray(result.masks)
    del result                          # MaskResult.state is opaque and may be device-resident
    # The adapter returns one mask per box, in order (§7.1 Mask2D), so the object
    # ids are the box indices and the shared assertions apply verbatim.
    return elapsed, np.arange(len(masks)), masks


def _adapter_video(torch, adapter, boxes: np.ndarray, prompt_frame: int, direction: str,
                   frames_np) -> tuple[float, list[tuple[int, Any]]]:
    """MaskVideoTracker (C27), against the protocol as declared in model_interfaces.py.

    `propagate_video` yields `(frame_idx, {obj_id: mask}, {obj_id: presence})`,
    and an obj_id absent from a yield means "no mask this frame" — the driver
    zero-fills, the adapter never fabricates. Normalising to the harness's
    `{out_obj_ids, out_binary_masks}` shape keeps `assert_video_contract` the
    same assertion in both modes: a missing id simply fails that frame.
    """
    session = adapter.init_video(list(frames_np))
    try:
        adapter.add_video_boxes(session, prompt_frame, list(range(len(boxes))), boxes)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        collected: list[tuple[int, Any]] = []
        for frame_idx, mask_by_id, _presence in adapter.propagate_video(
                session, start_frame_idx=prompt_frame, max_frames=len(frames_np),
                reverse=(direction == "backward")):
            ids = sorted(mask_by_id)
            masks = (np.stack([np.asarray(mask_by_id[i]) for i in ids]) if ids
                     else np.zeros((0, IMG_H, IMG_W), dtype=bool))
            collected.append((int(frame_idx), {"out_obj_ids": np.asarray(ids, dtype=np.int64),
                                               "out_binary_masks": masks}))
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    finally:
        adapter.close_video(session)
    return elapsed, collected


def run_adapter(args, torch, summary: dict, image, frames, boxes, video_boxes, checkpoint: str) -> None:
    try:
        from pipeline.stage4_masks.masks import MaskConfig, Sam31MultiplexAdapter
    except (ImportError, AttributeError) as exc:
        print("adapter not landed yet (C26 step A4); run --mode harness")
        raise SmokeFailure(2, f"Sam31MultiplexAdapter unavailable: {exc}")

    spec = CheckpointSpec(
        role=MASK_2D, provider="sam31_multiplex", model_id=REPO_ID, revision=REVISION,
        provenance=("C26: SAM 3.1 multiplex for mask_2d, gated by scripts/smoke_sam31.py --mode harness; "
                    f"{CKPT_FILENAME} at revision {REVISION}")).assert_valid()
    image_np = np.asarray(image, dtype=np.uint8)
    frames_np = [np.asarray(f, dtype=np.uint8) for f in frames]

    def build():
        adapter = Sam31MultiplexAdapter(
            spec, MaskConfig(model_id=REPO_ID, provider="sam31_multiplex", revision=REVISION,
                             checkpoint_path=checkpoint, device=args.device, global_seed=args.seed))
        adapter.load()
        return adapter

    _drive(args, torch, summary, build, "unload",
           lambda a, b: _adapter_image(torch, a, b, image_np),
           lambda a, b, f, d: _adapter_video(torch, a, b, f, d, frames_np),
           boxes, video_boxes)


# --- driver ----------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("harness", "adapter"), default="harness")
    ap.add_argument("--checkpoint", default=None, help=f"default: {REPO_ID}@{REVISION[:12]}/{CKPT_FILENAME}")
    ap.add_argument("--paths", default=os.environ.get("DHAKASCENES_PATHS_CONFIG", "configs/paths.yaml"))
    ap.add_argument("--n-boxes", type=int, default=32)
    ap.add_argument("--video-frames", type=int, default=7)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=20260812)
    ap.add_argument("--max-num-objects", type=int, default=128)
    args = ap.parse_args()

    summary: dict[str, Any] = {
        "mode": args.mode, "checkpoint_path": None, "checkpoint_sha256": None,
        # The builder prints missing/unexpected keys and discards them
        # (model_builder.py:1223-1230), so they are not obtainable at this call
        # site and are recorded null rather than guessed at.
        "missing_keys_n": None, "unexpected_keys_n": None,
        "vram_peak_mib_reserved": None, "vram_peak_mib_mem_get_info": None,
        "s_per_image_32_boxes": None, "s_per_frame_video_fwd": None, "s_per_frame_video_rev": None,
        "n_nonempty_masks": None, "torch_version": None, "python_version": platform.python_version(),
        "alloc_conf": ALLOC_CONF, "revision": REVISION, "seed": args.seed}
    code, reason = 0, ""
    try:
        # pkg_resources first: sam3.model_builder resolves its BPE vocabulary
        # through it, so a setuptools-less env fails here, not mid-build.
        try:
            import pkg_resources  # noqa: F401
        except ImportError as exc:
            raise SmokeFailure(2, f"import pkg_resources failed: {exc}")
        try:
            import sam3  # noqa: F401
        except ImportError as exc:
            raise SmokeFailure(2, f"import sam3 failed: {exc}")
        try:
            import torch
        except ImportError as exc:
            raise SmokeFailure(2, f"import torch failed: {exc}")
        summary["torch_version"] = torch.__version__

        if not args.device.startswith("cuda") or not torch.cuda.is_available():
            raise SmokeFailure(2, f"--device {args.device!r}: this smoke gates the card and needs CUDA")
        # The builder hard-codes .cuda(), i.e. the *current* device, so --device
        # is honoured by selecting it rather than by passing it down.
        torch.cuda.set_device(int(args.device.split(":", 1)[1]) if ":" in args.device else 0)
        # BEFORE the first allocation: a cap applied after the model loaded caps
        # nothing (C1). The builder's own .cuda() is that first allocation.
        summary["vram_cap"] = apply_vram_cap(torch, args.device)

        checkpoint = _resolve_checkpoint(args.checkpoint)
        summary["checkpoint_path"] = checkpoint
        summary["checkpoint_sha256"] = digest = _sha256_file(checkpoint)
        # Gate 1 is "the checkpoint is the pinned one", and until now this script
        # only MEASURED the digest and printed it. --checkpoint defaults to
        # $MOBILE_SAM_CHECKPOINT on this repo's .env, so ANY file — MobileSAM's
        # 40 MB .pt included — was hashed, reported and driven as if it were SAM
        # 3.1, and the numbers were then quoted against bytes nobody established.
        # The digest stays in the summary; what changes is that it is now a
        # DECISION and not a reading.
        summary["checkpoint_sha256_pinned"] = pinned = _pinned_sha256()
        if pinned is None:
            # No pin obtainable: a pre-A4 tree, or the private constant renamed.
            # The hub branch is still pinned by REPO_ID + REVISION + filename, so
            # harness mode keeps working there exactly as before. An explicit
            # --checkpoint has nothing left to check it against, and measuring it
            # on trust is the hole this gate closes — so it is REFUSED (2: the
            # gate could not be constructed), never quietly measured.
            if args.checkpoint:
                raise SmokeFailure(2, f"--checkpoint {args.checkpoint!r} cannot be verified: "
                                      "Sam31MultiplexAdapter._EXPECTED_SHA256 is not importable, so "
                                      "this smoke has no pinned digest to check it against. Drop "
                                      "--checkpoint and the pinned revision resolves from HF_HOME")
        elif digest != pinned:
            # 4, not 2: the file resolved perfectly well and an ASSERTION about it
            # is false, which is the 3/4/5 half of the taxonomy — 2 is reserved for
            # a gate that could not run at all.
            raise SmokeFailure(4, f"checkpoint {checkpoint!r} hashes to {digest}, not the pinned "
                                  f"{pinned}: that is {REPO_ID}/{CKPT_FILENAME} at revision "
                                  f"{REVISION} (C26), the bytes every C26 number was measured "
                                  "against. This smoke gates SAM 3.1 or it gates nothing")

        dataroot, version = _substrate(args.paths)
        image, summary["image_path"] = _first_keyframe(dataroot)
        frames, summary["video_frame_paths"] = _consecutive_frames(dataroot, version, args.video_frames)
        boxes = _grid_boxes(args.n_boxes)
        video_boxes = _grid_boxes(GRID_COLS * GRID_ROWS)[list(VIDEO_BOX_INDICES)]

        runner = run_harness if args.mode == "harness" else run_adapter
        runner(args, torch, summary, image, frames, boxes, video_boxes, checkpoint)
    except SmokeFailure as exc:
        code, reason = exc.code, exc.reason
    except Exception as exc:                      # noqa: BLE001 - any surprise is API drift
        # An OOM under set_per_process_memory_fraction IS the cap being hit;
        # calling that drift would point the investigation at the wrong file.
        code = 3 if type(exc).__name__ == "OutOfMemoryError" else 4
        reason = f"{type(exc).__name__}: {exc}"

    summary["exit_code"] = code
    print(json.dumps(summary))
    print("SMOKE: PASS" if code == 0 else f"SMOKE: FAIL({reason})")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
