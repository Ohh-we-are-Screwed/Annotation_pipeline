"""Stage 3c — VLM label check: stage-3-shaped tree -> stage3_checked/.

Re-classifies proposal crops with a local vision-language model (Nemotron
3 Nano Omni via llama.cpp's OpenAI-compatible server) and rewrites the label
where the model confidently disagrees. Runs on the ONE tree Stage 4 would
otherwise consume (stage3_merged when the two-arm merge ran) and emits Stage
3's exact schema in Stage 3's row order plus additive keys (the Stage 3b
trick, C27, used a third time), so Stage 4 consumes the output unchanged via
--stage3-dir.

Design decisions, in the pipeline's house style:
  - The VLM is asked BLIND (the current label is never in the prompt): showing
    it the label to "verify" invites yes-bias, and the whole point is an
    independent second opinion.
  - The class space never widens here: the caption must equal the input
    manifest's caption byte-for-byte. Widening is the merge's job (C28).
  - A relabel rewrites class_names / nuscenes_categories / phrase_char_spans
    at that index and NOTHING else: geometry, scores and order ride through
    byte-identical (C27). Scores keep arm semantics — the manifest records
    that a relabeled box's score is the ORIGINAL detector's confidence in the
    ORIGINAL class, because the VLM emits no calibrated score to replace it.
  - Boxes smaller than --min-side-px never reach the model: a 10 px crop is
    noise, and a confident answer on noise is exactly the failure mode this
    stage exists to remove. Skips are recorded per box, not silently.
  - Every box gets an audit verdict in the row's `vlm_check` block; the
    manifest aggregates a `original -> new` confusion table so one glance
    shows what the checker actually did.

Two check modes (--check-mode, default per_box):

  per_box    Today's behavior: one model call per above-floor box.

  per_track  One model call per tracked object, propagated to every box of
             that track. The key is `(channel, int(track_id))` and it is
             SCENE-LOCAL: Stage 3b allocates ids from a counter that restarts
             per (scene, channel), so id 0 on CAM_FRONT and id 0 on CAM_BACK
             are different objects (track2d.py's own docstring says so). The
             representative crop is chosen by a fixed, recorded ranking rule —
             never by scores, which are three incommensurable scales in one
             array (arm A, arm B, and a decayed recovered score).

             Two consequences that are FEATURES, recorded rather than hidden:
             a track's sub-floor boxes inherit the verdict flagged
             `propagated_over_small`, which heals the temporal label flicker a
             per-box floor produces (the same object is "a car" while it is far
             away and "an auto rickshaw" once it is close); and one bad verdict
             now rewrites every frame of its track, so `n_members` rides with
             every propagated verdict and the manifest names the largest
             propagated relabel.

             Per-box `confusion` and `n_errors` are TRACK-LENGTH-WEIGHTED under
             per_track. `confusion_by_track` and `n_error_tracks` are the
             cross-mode-comparable numbers.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import NamedTuple, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pipeline.common.manifest import (  # noqa: E402
    UpstreamRefusal,
    clear_markers,
    require_upstream,
    write_json_atomic,
    write_jsonl_atomic,
    write_marker,
)
from pipeline.common.rowmeta import optional_c27_array  # noqa: E402
from pipeline.stage3_proposals.proposals import build_caption, load_taxonomy  # noqa: E402

STAGE = "stage3c_check"
STAGE_SPEC = "dhakascenes-pilot/stage3c_check/v1"
PROVIDER = "nemotron_vlm_label_check"

VERDICT_CONFIRMED = "confirmed"          # model agrees with the incoming label
VERDICT_RELABELED = "relabeled"          # model names a different caption phrase
VERDICT_UNCLEAR = "unclear"              # model declines; label kept
VERDICT_SKIPPED_SMALL = "skipped_small"  # crop below --min-side-px; never asked
VERDICT_SKIPPED_CLASS = "skipped_class"  # label class is in --skip-class; never asked (C31)
VERDICT_ERROR = "error"                  # unparseable/failed reply; label kept

UNCLEAR = "unclear"

CHECK_MODE_PER_BOX = "per_box"
CHECK_MODE_PER_TRACK = "per_track"
CHECK_MODES = (CHECK_MODE_PER_BOX, CHECK_MODE_PER_TRACK)

# Where a record's verdict came from. `none` is not "unknown": it is the honest
# reading of a skipped_small box that no model ever saw.
SOURCE_VLM = "vlm"      # this exact crop was sent to the model
SOURCE_TRACK = "track"  # propagated from another crop of the same track
SOURCE_NONE = "none"    # no model call decided this box

JPEG_QUALITY = 92  # what NemotronVLM would encode anyway; held crops match it

TRACK_KEY_DEFINITION = (
    "(channel, int(track_id)), scene-local. Stage 3b constructs one CameraTracker per "
    "(scene, channel) with a counter that starts at 0, so the bare integer is NOT a key: "
    "id 0 exists independently in every camera of every scene. The cache is reset at each "
    "scene-directory boundary."
)

REPRESENTATIVE_RANKING_RULE = (
    "among a track's members that clear --min-side-px, ranked ascending by: "
    "(1) NOT (box_sources=='yolo' and n_propagated_hops==0) — a detector-evidenced box "
    "beats a propagated recovered box, whose tight mask box may be drifted or oversized; "
    "(2) NOT fully-inside-image — the margin-expanded crop would clamp, and the prompt "
    "tells the model to answer 'unclear' on a truncated crop, so a clamped crop is a "
    "wasted call; (3) larger MIN SIDE — the quantity --min-side-px itself gates on; "
    "(4) larger area; (5) file order (row index, box index). Scores are never ranked on: "
    "arm-A, arm-B and decayed-recovered scores are three incommensurable scales."
)

TRACK_ACCOUNTING_NOTES = (
    "Under per_track, the per-box `confusion` table and `n_errors` are TRACK-LENGTH-"
    "WEIGHTED: one fresh call decides every box of its track, so a single failure is "
    "amplified to n_members boxes. `confusion_by_track` and `n_error_tracks` count one "
    "entry per decision and are the numbers comparable against a per_box run. A degraded "
    "marker under per_track therefore reads 'K fresh-call failures amplified to N boxes'. "
    "`n_members` rides with every propagated verdict so a relabel's blast radius is "
    "readable per box, and `largest_relabel_track` names the widest one."
)

# Glosses appended to the two Dhaka phrases in the prompt. The phrase spellings
# are the taxonomy's (docs there explain why "cng" itself is never a phrase);
# the glosses carry the local word so the model cannot confuse the pedal and
# the motorized three-wheeler — the exact confusion this stage exists to fix.
PHRASE_GLOSSES = {
    "a rickshaw": "cycle rickshaw: pedal-driven passenger three-wheeler",
    "an auto rickshaw": "CNG: motorized three-wheeler with a metal cage body",
}


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested)
# ---------------------------------------------------------------------------

def crop_box(box_xyxy, image_size_px, *, margin: float = 1.15) -> list[int]:
    """Margin-expanded, image-clamped integer crop window for one box."""
    x0, y0, x1, y1 = (float(v) for v in box_xyxy)
    w, h = image_size_px
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    half_w, half_h = (x1 - x0) * margin / 2.0, (y1 - y0) * margin / 2.0
    return [
        int(round(max(0.0, cx - half_w))),
        int(round(max(0.0, cy - half_h))),
        int(round(min(float(w), cx + half_w))),
        int(round(min(float(h), cy + half_h))),
    ]


def box_min_side(box_xyxy) -> float:
    x0, y0, x1, y1 = (float(v) for v in box_xyxy)
    return min(x1 - x0, y1 - y0)


def crop_is_untruncated(box_xyxy, image_size_px, *, margin: float = 1.15) -> bool:
    """Would `crop_box` return the full margin-expanded window, unclamped?

    Uses the row's DECLARED `image_size_px`. The crop itself is taken against
    the decoded PIL size (which the upstream gate has already checked matches),
    so this is a ranking signal, not a promise about the pixels.
    """
    if not image_size_px:
        return True  # nothing declared: the criterion abstains rather than guesses
    w, h = float(image_size_px[0]), float(image_size_px[1])
    x0, y0, x1, y1 = (float(v) for v in box_xyxy)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    half_w, half_h = (x1 - x0) * margin / 2.0, (y1 - y0) * margin / 2.0
    return (cx - half_w >= 0.0 and cy - half_h >= 0.0
            and cx + half_w <= w and cy + half_h <= h)


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_JSON_RE = re.compile(r"\{[^{}]*\}")


def parse_vlm_reply(text: str, allowed_phrases) -> str | None:
    """The label the model chose, `"unclear"`, or None if nothing parses.

    Reasoning models wrap answers in prose and <think> blocks; the LAST
    well-formed {"label": ...} object in the stripped text is the answer.
    """
    if not text:
        return None
    stripped = _THINK_RE.sub("", text)
    label = None
    for m in _JSON_RE.finditer(stripped):
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("label"), str):
            label = obj["label"]
    if label is None:
        return None
    label = label.strip().lower().rstrip(".")
    # Nemotron often echoes the prompt's own class gloss in parentheses
    # ("a rickshaw (cycle rickshaw: pedal-driven ...)"); strip it before
    # matching against the taxonomy phrases.
    label = re.sub(r"\s*\([^)]*\)", "", label).strip().rstrip(".")
    if label == UNCLEAR:
        return UNCLEAR
    allowed = tuple(allowed_phrases)
    if label in allowed:
        return label
    # The model drops the article often enough to matter ("truck" for
    # "a truck", observed in the smoke run); phrases only ever differ by
    # their article, so the article-stripped form is still unambiguous.
    def _bare(p: str) -> str:
        head, _, tail = p.partition(" ")
        return tail if head in ("a", "an", "the") and tail else p
    bare_map = {}
    for p in allowed:
        bare_map.setdefault(_bare(p), p)
    return bare_map.get(_bare(label))


def verdict_for(original_phrase: str, vlm_phrase: str | None, allowed_phrases) -> str:
    if vlm_phrase is None:
        return VERDICT_ERROR
    if vlm_phrase == UNCLEAR:
        return VERDICT_UNCLEAR
    if vlm_phrase == original_phrase:
        return VERDICT_CONFIRMED
    if vlm_phrase in tuple(allowed_phrases):
        return VERDICT_RELABELED
    return VERDICT_ERROR


class TrackPlan(NamedTuple):
    """What one scene's rows say about which crops have to be asked about.

    members          key -> every (row_index, box_index) carrying that track id
    representatives  key -> the ranked candidates that clear the floor, best first
    untracked        (row_index, box_index) for every box with no usable track id
    all_below_floor  keys whose every member is below the floor: zero calls, and
                     every member is skipped_small exactly as per_box would have it
    """

    members: dict
    representatives: dict
    untracked: list
    all_below_floor: set


def plan_track_checks(rows: Sequence[dict], *, min_side_px: float,
                      margin: float = 1.15,
                      skip_classes: frozenset = frozenset()) -> TrackPlan:
    """Group one SCENE's rows into tracks and rank each track's candidate crops.

    Pure: no IO, no PIL, no model. The cache this plans is scene-local by
    construction — callers must build one plan per scene directory, because
    Stage 3b's ids restart at 0 for every (scene, channel).

    `track_ids` is read through `optional_c27_array`: ABSENT is legal (a plain
    Stage 3 tree, or a merged tree whose arm A was one) and every box then falls
    back to per-box; a `None` entry is legal (an arm-B box appended by the merge)
    and that box falls back to per-box; a PRESENT-but-wrong-length array is an
    upstream contract failure and is REFUSED, never padded.

    Ranking uses the row's DECLARED `image_size_px` for the untruncated-crop
    criterion, while the crop itself is later taken against the decoded PIL size.
    The two agree on every tree the upstream gate accepts (it refuses a manifest
    with no `image_size_px`), and keeping the ranking pure is worth more than
    keeping the two reads identical: a plan must be reproducible without opening
    a single JPEG.

    Ranking rule (deterministic, and recorded verbatim in the run manifest as
    `vlm.representative_ranking_rule` — see REPRESENTATIVE_RANKING_RULE above):
    detector-evidenced first, then untruncated, then largest min-side, then
    largest area, then file order. Never scores.
    """
    floor = float(min_side_px)
    members: dict = {}
    ranked: dict = {}
    untracked: list = []

    for row_i, row in enumerate(rows):
        boxes = row.get("boxes_xyxy_px") or []
        n = len(boxes)
        if not n:
            continue
        names = row.get("class_names") or []
        track_ids = optional_c27_array(row, "track_ids", n)
        if not track_ids:
            untracked.extend((row_i, i) for i in range(n) if names[i] not in skip_classes)
            continue
        sources = optional_c27_array(row, "box_sources", n)
        hops = optional_c27_array(row, "n_propagated_hops", n)
        channel = row.get("channel")
        size = row.get("image_size_px")
        for i, box in enumerate(boxes):
            tid = track_ids[i]
            if tid is None:
                if names[i] not in skip_classes:
                    untracked.append((row_i, i))
                continue
            key = (channel, int(tid))
            members.setdefault(key, []).append((row_i, i))
            if names[i] in skip_classes:
                continue  # never a representative: a skipped class is never asked about
            if box_min_side(box) < floor:
                continue
            source = sources[i] if sources else "yolo"
            hop = int(hops[i]) if hops else 0
            x0, y0, x1, y1 = (float(v) for v in box)
            ranked.setdefault(key, []).append((
                0 if (source == "yolo" and hop == 0) else 1,
                0 if crop_is_untruncated(box, size, margin=margin) else 1,
                -min(x1 - x0, y1 - y0),
                -((x1 - x0) * (y1 - y0)),
                row_i,
                i,
            ))

    representatives = {}
    for key, cands in ranked.items():
        cands.sort()
        representatives[key] = [(c[4], c[5]) for c in cands]
    all_below_floor = {k for k in members if k not in representatives}
    return TrackPlan(members, representatives, untracked, all_below_floor)


# Additive provenance carried from a verdict dict into its audit record. `action`
# and `vlm_phrase` are built directly; everything here is copied only when the
# verdict actually carries it, so a per_box record gains exactly `verdict_source`
# (always) and `error` (only when an ask failed).
_RECORD_EXTRA_KEYS = (
    "verdict_source", "error", "track_id", "checked_at",
    "propagated_over_small", "n_members",
)


def apply_verdicts(row: dict, verdicts: list[dict], *, caption, taxonomy) -> dict:
    """One input row + one verdict per box -> the checked row.

    Only the three label arrays change, and only at relabeled indices; every
    other key — geometry, scores, stage 3b extension arrays, merge ledgers —
    rides through untouched (C27). The input row is never mutated.
    """
    n = len(row["boxes_xyxy_px"])
    if len(verdicts) != n:
        raise ValueError(f"{len(verdicts)} verdicts for {n} boxes")

    span_of = {p: list(s) for p, s in zip(caption.phrases, caption.phrase_char_spans)}
    p2c = taxonomy.phrase_to_categories

    out = dict(row)  # shallow: every list we touch is rebuilt below
    names = list(row["class_names"])
    cats = [list(c) for c in row["nuscenes_categories"]]
    spans = [list(s) for s in row["phrase_char_spans"]]

    records = []
    counts = {VERDICT_CONFIRMED: 0, VERDICT_RELABELED: 0, VERDICT_UNCLEAR: 0,
              VERDICT_SKIPPED_SMALL: 0, VERDICT_SKIPPED_CLASS: 0, VERDICT_ERROR: 0}
    for i, v in enumerate(verdicts):
        action = v["action"]
        counts[action] += 1
        record = {"action": action, "vlm_phrase": v.get("vlm_phrase")}
        for extra in _RECORD_EXTRA_KEYS:
            if v.get(extra) is not None:
                record[extra] = v[extra]
        if action == VERDICT_RELABELED:
            new = v["vlm_phrase"]
            record["original_class_name"] = names[i]
            record["original_nuscenes_categories"] = cats[i]
            record["original_phrase_char_spans"] = spans[i]
            names[i] = new
            cats[i] = list(p2c[new])
            spans[i] = span_of[new]
        records.append(record)

    out["class_names"] = names
    out["nuscenes_categories"] = cats
    out["phrase_char_spans"] = spans
    block = {
        "spec": STAGE_SPEC,
        "verdicts": records,
        "n_relabeled": counts[VERDICT_RELABELED],
        "n_confirmed": counts[VERDICT_CONFIRMED],
        "n_unclear": counts[VERDICT_UNCLEAR],
        "n_skipped_small": counts[VERDICT_SKIPPED_SMALL],
        "n_skipped_class": counts[VERDICT_SKIPPED_CLASS],
        "n_errors": counts[VERDICT_ERROR],
        # The true cost of this row: how many of these verdicts a model produced
        # from THIS row's pixels. Under per_box it equals n_checked for the row.
        "n_fresh_checks": sum(1 for r in records if r.get("verdict_source") == SOURCE_VLM),
    }
    # per_track only, and detected from the records rather than from a mode
    # argument so `apply_verdicts`' signature — and its three tests — stand.
    if any("track_id" in r for r in records):
        block["n_from_track_verdict"] = sum(
            1 for r in records if r.get("verdict_source") == SOURCE_TRACK)
    out["vlm_check"] = block
    return out


# ---------------------------------------------------------------------------
# The Nemotron client and server manager (exercised by the smoke run, not unit
# tests: there is no honest way to mock a 24 GB model)
# ---------------------------------------------------------------------------

def encode_jpeg(crop, quality: int = JPEG_QUALITY) -> bytes:
    """The exact bytes `NemotronVLM` would have encoded for this crop.

    per_track holds a track's runner-up candidates between pass A and the retry
    ladder. Holding them as JPEG bytes rather than decoded PIL images is what
    makes the ladder affordable: n_tracks x (k-1) compressed crops for the
    lifetime of one scene, and the encode is work the client would have done
    anyway if the crop were ever asked about.
    """
    buf = io.BytesIO()
    crop.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _prompt_text(allowed_phrases) -> str:
    lines = []
    for p in allowed_phrases:
        gloss = PHRASE_GLOSSES.get(p)
        lines.append(f"- {p}" + (f"  ({gloss})" if gloss else ""))
    return (
        "The image is one cropped object detection from a street scene in "
        "Dhaka, Bangladesh; the object of interest fills most of the crop "
        "(some background may be visible at the edges).\n"
        "Classify the single main object. Choose EXACTLY one label from this "
        "list, verbatim:\n" + "\n".join(lines) + "\n"
        "Rules:\n"
        "- If the crop is centered on a PERSON — including one riding, driving "
        "or pulling a vehicle — the label is the person's (\"a pedestrian\"), "
        "not the vehicle's. Answer a vehicle label only when the vehicle "
        "itself is the crop's main subject.\n"
        '- If the crop is too blurry, dark or truncated to be sure, answer "unclear" '
        "rather than guessing.\n"
        'If no listed label fits, answer "unclear".\n'
        'Reply with ONLY a JSON object: {"label": "<your choice>"}'
    )


class NemotronVLM:
    """One blind classification question per call against llama-server."""

    def __init__(self, server_url: str, *, temperature: float = 0.0,
                 max_tokens: int = 512, timeout_s: float = 300.0, retries: int = 1):
        self.server_url = server_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self.retries = retries
        self._prompt_cache: dict[tuple, str] = {}

    def describe(self) -> dict:
        return {
            "provider": "llama-server/chat-completions",
            "server_url": self.server_url,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "asked_blind": True,
        }

    def __call__(self, crop, original_phrase: str, allowed_phrases) -> str:
        # original_phrase is part of the call contract for auditability, but is
        # deliberately NOT in the prompt (blind check, see module docstring).
        # `crop` is a PIL image, or the JPEG bytes of one: per_track's retry
        # ladder holds runner-up crops pre-encoded, and re-decoding them just to
        # re-encode them identically would be pure waste.
        key = tuple(allowed_phrases)
        text = self._prompt_cache.get(key)
        if text is None:
            text = self._prompt_cache[key] = _prompt_text(allowed_phrases)
        jpeg = bytes(crop) if isinstance(crop, (bytes, bytearray)) else encode_jpeg(crop)
        data_uri = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")
        payload = json.dumps({
            "messages": [
                {"role": "system",
                 "content": "/no_think You are an exact single-object traffic classifier."},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": data_uri}},
                    {"type": "text", "text": text},
                ]},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }).encode("utf-8")
        req = urllib.request.Request(
            self.server_url + "/v1/chat/completions", data=payload,
            headers={"Content-Type": "application/json"})
        last_exc: Exception | None = None
        for _ in range(self.retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    reply = json.loads(resp.read().decode("utf-8"))
                return reply["choices"][0]["message"]["content"] or ""
            except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
                last_exc = exc
                time.sleep(2.0)
        raise RuntimeError(f"VLM request failed after {self.retries + 1} attempts: {last_exc}")


def assert_gpu_exclusive() -> None:
    """Refuse to load the model while anything else computes on the GPU.

    The operator's constraint, verbatim: when running Nemotron, nothing else
    runs. 24 GB is the whole budget and a co-tenant turns a tuned fit into an
    OOM mid-stage.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"cannot verify the GPU is free (nvidia-smi failed: {exc})")
    if out:
        raise RuntimeError(
            f"GPU is not free — refusing to load the VLM next to:\n{out}\n"
            "Stop those processes or pass --allow-shared-gpu."
        )


class LlamaServer:
    """Spawn/health-check/terminate one llama-server for the stage's lifetime."""

    def __init__(self, *, bin_path: str, gguf: str, mmproj: str, port: int,
                 parallel: int, ctx_per_slot: int, n_cpu_moe: int,
                 n_gpu_layers: int, log_path: str):
        self.url = f"http://127.0.0.1:{port}"
        self.log_path = log_path
        self.args = [
            bin_path, "-m", gguf, "--mmproj", mmproj,
            "--host", "127.0.0.1", "--port", str(port),
            "-np", str(parallel), "-c", str(ctx_per_slot * parallel),
            "--n-cpu-moe", str(n_cpu_moe), "-ngl", str(n_gpu_layers),
            "--no-webui",
        ]
        self.proc: subprocess.Popen | None = None

    def __enter__(self):
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self._log = open(self.log_path, "ab")
        self.proc = subprocess.Popen(self.args, stdout=self._log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 900.0  # 24 GB off NVMe + CUDA warmup
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"llama-server exited with {self.proc.returncode} during load; "
                    f"see {self.log_path}")
            try:
                with urllib.request.urlopen(self.url + "/health", timeout=5) as resp:
                    if resp.status == 200:
                        return self
            except (urllib.error.URLError, TimeoutError):
                pass
            time.sleep(3.0)
        raise RuntimeError(f"llama-server not healthy after 900 s; see {self.log_path}")

    def __exit__(self, *exc):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=30)
        self._log.close()
        return False


# ---------------------------------------------------------------------------
# Preflight: every refusal this stage can raise, before the 24 GB server spawn
# ---------------------------------------------------------------------------

def _read_rows(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def config_echo(*, check_mode: str, min_side_px: float, margin: float,
                track_retry_candidates: int, allow_untracked: bool,
                caption_sha256: str, skip_classes: Sequence[str] = ()) -> dict:
    """The knobs that decided a value in the rows underneath this out dir.

    `--parallel` is deliberately absent: thread count changes throughput, never
    output, and fencing on it would refuse a legitimate incremental re-run for a
    difference no reader can observe in the tree.
    """
    return {
        "check_mode": check_mode,
        "min_side_px": float(min_side_px),
        "margin": float(margin),
        "track_retry_candidates": int(track_retry_candidates),
        "allow_untracked": bool(allow_untracked),
        "caption_sha256": caption_sha256,
        "skip_classes": sorted(skip_classes),
    }


def fence_out_dir(out_dir: str, scene_names: Sequence[str], cfg: dict) -> dict:
    """Refuse a subset re-run that would re-bless another config's scene dirs (C27).

    Ported from track2d.py's fence of the same name, for the same reason:
    `--scenes chunk_0000` writes one scene and then stands a fresh marker and a
    fresh manifest — whose config echo and whose totals describe that one scene —
    over the WHOLE out dir, and Stage 4 consumes every directory under `scenes/`.
    Yesterday's nine per_box scenes plus today's one per_track scene therefore
    read downstream as a single upstream whose manifest says check_mode=per_track.
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
        return {"scenes_in_out_dir": [], "foreign_scene_dirs": [],
                "config_matched_previous": None}
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
    same_config = (
        previous.get("spec") == STAGE_SPEC
        and previous.get("provider") == PROVIDER
        and previous.get("config") == cfg
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


def scan_track_coverage(scenes_root: str, scene_names: Sequence[str],
                        max_rows: int | None = None) -> tuple[int, int]:
    """(boxes carrying a non-null track id, boxes) over the rows this run will read.

    Also the earliest place a short `track_ids` array is refused: the scan reads
    every row through `optional_c27_array`, so a parallel-array contract failure
    costs a file walk instead of a 24 GB model load.
    """
    n_tracked = n_boxes = 0
    budget = max_rows
    for scene in scene_names:
        if budget is not None and budget <= 0:
            break
        path = os.path.join(scenes_root, scene, "proposals.jsonl")
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                if budget is not None:
                    if budget <= 0:
                        break
                    budget -= 1
                row = json.loads(line)
                n = len(row.get("boxes_xyxy_px") or [])
                n_boxes += n
                if not n:
                    continue
                track_ids = optional_c27_array(row, "track_ids", n)
                n_tracked += sum(1 for t in track_ids if t is not None)
    return n_tracked, n_boxes


def preflight(
    input_dir: str,
    out_dir: str,
    taxonomy_path: str,
    *,
    accept_degraded: bool = False,
    check_mode: str = CHECK_MODE_PER_BOX,
    min_side_px: float = 32.0,
    margin: float = 1.15,
    track_retry_candidates: int = 3,
    allow_untracked: bool = False,
    scenes: Sequence[str] | None = None,
    max_rows: int | None = None,
    skip_classes: Sequence[str] | None = None,
) -> dict:
    """Every refusal this stage can raise, evaluated before the first byte.

    `main()` calls this BEFORE spawning llama-server, so a taxonomy typo or a
    stale 3b tree costs a file walk instead of a 24 GB load off NVMe plus up to
    900 s of health polling (Stage 3b's Phase-B preflight, generalised). `run()`
    calls it again as its own first act — the refusals belong to the driver, not
    to one entry point, and calling `run()` directly (as every unit test does)
    must refuse exactly the same things.

    Nothing here writes a byte, so rc 2 still means "nothing written, the
    previous run's marker intact".
    """
    if check_mode not in CHECK_MODES:
        raise UpstreamRefusal(f"--check-mode must be one of {CHECK_MODES}, got {check_mode!r}")

    taxonomy = load_taxonomy(taxonomy_path)
    caption = build_caption(taxonomy.phrases)

    skip = frozenset(skip_classes or ())
    unknown = sorted(skip - set(caption.phrases))
    if unknown:
        raise UpstreamRefusal(
            f"--skip-class names {unknown} which the caption does not hold "
            f"(phrases: {list(caption.phrases)})")

    man, marker = require_upstream(
        input_dir, stage_name="Stage 3 (checker input)",
        module_hint="pipeline.stage3_merge.merge",
        accept_degraded=accept_degraded,
    )
    cap_in = str((man.get("prompt") or {}).get("caption"))
    if cap_in != caption.text:
        raise UpstreamRefusal(
            f"input tree's caption differs from {taxonomy_path}'s caption: the checker "
            "never widens the class space (that is the merge's job, C28) — run 3c with "
            "the SAME taxonomy file its input was produced under"
        )
    image_size = man.get("image_size_px")
    if not image_size:
        raise UpstreamRefusal(
            "input manifest carries no image_size_px; Stage 4 refuses an upstream that "
            "does not state its resolution, so the checked tree could never be consumed"
        )

    scenes_root = os.path.join(input_dir, "scenes")
    available = sorted(os.listdir(scenes_root)) if os.path.isdir(scenes_root) else []
    if not available:
        raise UpstreamRefusal(f"{scenes_root} has no scenes; nothing to check")
    if scenes:
        wanted = list(dict.fromkeys(scenes))
        missing = [s for s in wanted if s not in set(available)]
        if missing:
            raise UpstreamRefusal(
                f"--scenes names {missing} which {scenes_root} does not hold "
                f"(it holds {available})")
        names = [s for s in available if s in set(wanted)]
    else:
        names = available

    if max_rows is not None:
        if int(max_rows) < 1:
            raise UpstreamRefusal(f"--max-rows must be >= 1, got {max_rows}")
        root = os.path.join(out_dir, "scenes")
        if os.path.isdir(root) and any(
                os.path.isdir(os.path.join(root, n)) for n in os.listdir(root)):
            raise UpstreamRefusal(
                f"--max-rows refuses to write into {root}, which already holds scene "
                "directories. A bounded run truncates every scene it touches, so pointed at "
                "a COMPLETE stage3_checked tree it would silently replace it with the first "
                f"{max_rows} rows. --max-rows is a timing probe: give it its own --out-dir."
            )

    cfg = config_echo(check_mode=check_mode, min_side_px=min_side_px, margin=margin,
                      track_retry_candidates=track_retry_candidates,
                      allow_untracked=allow_untracked, caption_sha256=caption.sha256,
                      skip_classes=skip)
    fence = fence_out_dir(out_dir, names, cfg)

    coverage = None
    if check_mode == CHECK_MODE_PER_TRACK:
        n_tracked, n_boxes = scan_track_coverage(scenes_root, names, max_rows)
        coverage = (n_tracked / n_boxes) if n_boxes else None
        if n_boxes and n_tracked == 0 and not allow_untracked:
            raise UpstreamRefusal(
                f"--check-mode per_track over {input_dir}: 0 of {n_boxes} boxes carry a track "
                "id, so every box would fall back to per-box and the mode would buy nothing. "
                "The usual cause is a STALE Stage 3b tree: a chain containing step `3` re-runs "
                "Stage 3 after 3b, `select_stage3_dir_for_4` then hands 3c plain "
                "stage3_proposals (which has no track_ids key at all), and the saving silently "
                "disappears. Run 3b in this chain — `3b 3f 3m 3c` — or pass --allow-untracked "
                "to accept a per-box-priced run under a per_track label."
            )

    return {
        "taxonomy": taxonomy,
        "caption": caption,
        "manifest": man,
        "marker": marker,
        "image_size_px": image_size,
        "scenes": names,
        "scenes_available": available,
        "fence": fence,
        "config": cfg,
        "check_mode": check_mode,
        "track_coverage": coverage,
        "skip_classes": skip,
    }


# ---------------------------------------------------------------------------
# The stage driver
# ---------------------------------------------------------------------------

def _ask(vlm, crop, original_phrase, phrases) -> dict:
    try:
        reply = vlm(crop, original_phrase, phrases)
    except Exception as exc:  # noqa: BLE001 — any failed ask is an audit row, never a crash
        return {"action": VERDICT_ERROR, "vlm_phrase": None, "error": str(exc)[:300]}
    parsed = parse_vlm_reply(reply, phrases)
    return {"action": verdict_for(original_phrase, parsed, phrases), "vlm_phrase": parsed}


def _pass_a(rows, plan, *, pool, vlm, caption, dataroot, image_module,
            min_side_px, margin, parallel, track_retry_candidates) -> dict:
    """Ask the model once per track (plus once per untracked above-floor box).

    Rows are walked in order and an image is opened ONLY when that row owns an
    ask — a row whose every box is a sub-floor member of a checked track costs
    no JPEG decode at all, which today's per-box path cannot avoid.

    Retry ladder: a track's ranked candidates are extracted while the owning
    row's image is open. Candidate #1 is submitted immediately; #2..k are
    retained as JPEG bytes keyed by track. The ladder drains ONCE, after the
    scene's last row — a track's candidates live in DIFFERENT rows by
    construction (3b's Hungarian match is one-to-one, so one box per track per
    row), so an earlier drain would find candidates that had not been extracted
    yet. In-flight futures are bounded by joining every max(parallel*16, 64)
    submissions; pass B runs strictly after every join, so chunking is free.
    """
    k = max(1, int(track_retry_candidates))
    floor = float(min_side_px)

    wanted: dict[int, list] = {}
    for key, ranked in plan.representatives.items():
        for rank, (row_i, box_i) in enumerate(ranked[:k]):
            wanted.setdefault(row_i, []).append(("track", key, rank, box_i))
    for row_i, box_i in plan.untracked:
        if box_min_side(rows[row_i]["boxes_xyxy_px"][box_i]) >= floor:
            wanted.setdefault(row_i, []).append(("untracked", (row_i, box_i), 0, box_i))

    track_verdicts: dict = {}
    checked_at: dict = {}
    untracked_results: dict = {}
    held: dict = {}          # key -> {rank: (jpeg_bytes, row_i, box_i)}
    pending: list = []
    n_calls = 0
    n_retries = 0
    chunk = max(int(parallel) * 16, 64)

    def _join() -> None:
        for kind, ident, row_i, box_i, fut in pending:
            verdict = fut.result()
            if kind == "untracked":
                untracked_results[ident] = verdict
            else:
                track_verdicts[ident] = verdict
                checked_at[ident] = (row_i, box_i)
        pending.clear()

    for row_i in sorted(wanted):
        row = rows[row_i]
        with image_module.open(os.path.join(dataroot, row["image_path"])) as im:
            image = im.convert("RGB")
        names = row["class_names"]
        boxes = row["boxes_xyxy_px"]
        for kind, ident, rank, box_i in wanted[row_i]:
            crop = image.crop(tuple(crop_box(boxes[box_i], image.size, margin=margin)))
            if kind == "untracked" or rank == 0:
                pending.append((kind, ident, row_i, box_i, pool.submit(
                    _ask, vlm, crop, names[box_i], caption.phrases)))
                n_calls += 1
            else:
                held.setdefault(ident, {})[rank] = (encode_jpeg(crop), row_i, box_i)
        if len(pending) >= chunk:
            _join()
    _join()

    for rank in range(1, k):
        round_pending = []
        for key, verdict in track_verdicts.items():
            if verdict["action"] not in (VERDICT_UNCLEAR, VERDICT_ERROR):
                continue  # a confirmed/relabeled verdict is kept; the ladder stops there
            candidate = held.get(key, {}).get(rank)
            if candidate is None:
                continue
            jpeg, row_i, box_i = candidate
            round_pending.append((key, row_i, box_i, pool.submit(
                _ask, vlm, jpeg, rows[row_i]["class_names"][box_i], caption.phrases)))
            n_calls += 1
            n_retries += 1
        if not round_pending:
            break
        for key, row_i, box_i, fut in round_pending:
            verdict = fut.result()
            current = track_verdicts[key]
            if verdict["action"] in (VERDICT_CONFIRMED, VERDICT_RELABELED, VERDICT_UNCLEAR):
                # confirmed/relabeled wins outright; unclear replaces an unclear or
                # an error ("last verdict, preferring unclear over error").
                track_verdicts[key] = verdict
                checked_at[key] = (row_i, box_i)
            elif current["action"] == VERDICT_ERROR:
                track_verdicts[key] = verdict
                checked_at[key] = (row_i, box_i)

    return {
        "track_verdicts": track_verdicts,
        "checked_at": checked_at,
        "untracked_results": untracked_results,
        "n_vlm_calls": n_calls,
        "n_track_retries": n_retries,
        "held_crops": sum(len(d) for d in held.values()),
        "held_bytes": sum(len(j) for d in held.values() for (j, _, _) in d.values()),
    }


def _pass_b_row(row_i, row, *, plan, key_index, pass_a, rows, caption, min_side_px,
                skip_classes: frozenset = frozenset()) -> list[dict]:
    """The verdict list for one row, assembled from the scene's track cache.

    A tracked box's action is recomputed with `verdict_for(this box's own name,
    the cached PHRASE)` rather than copied from the cached ACTION. It costs
    nothing, and if 3b's one-class-per-track invariant is ever broken upstream
    the audit block stays truthful instead of recording "confirmed" on a box
    whose label the model never saw.
    """
    boxes = row["boxes_xyxy_px"]
    names = row["class_names"]
    track_verdicts = pass_a["track_verdicts"]
    checked_at = pass_a["checked_at"]
    out: list[dict] = []
    for box_i in range(len(boxes)):
        key = key_index.get((row_i, box_i))
        if names[box_i] in skip_classes:
            # Class policy, not crop size, is why no model ever saw this box —
            # so the guard sits above the floor check and above track lookup:
            # a skipped class neither receives nor contributes a track verdict.
            record = {"action": VERDICT_SKIPPED_CLASS, "vlm_phrase": None,
                      "verdict_source": SOURCE_NONE}
            if key is not None:
                record["track_id"] = key[1]
                record["n_members"] = len(plan.members[key])
            out.append(record)
            continue
        if key is None:
            fresh = pass_a["untracked_results"].get((row_i, box_i))
            if fresh is None:
                out.append({"action": VERDICT_SKIPPED_SMALL, "vlm_phrase": None,
                            "verdict_source": SOURCE_NONE})
            else:
                record = dict(fresh)
                record["verdict_source"] = SOURCE_VLM
                out.append(record)
            continue
        n_members = len(plan.members[key])
        verdict = track_verdicts.get(key)
        if verdict is None:
            # Every member below the floor: no ask at all, exactly as per_box.
            out.append({"action": VERDICT_SKIPPED_SMALL, "vlm_phrase": None,
                        "verdict_source": SOURCE_NONE, "track_id": key[1],
                        "n_members": n_members})
            continue
        phrase = verdict.get("vlm_phrase")
        src_row, src_box = checked_at[key]
        record = {
            "action": verdict_for(names[box_i], phrase, caption.phrases),
            "vlm_phrase": phrase,
            "verdict_source": SOURCE_VLM if (src_row, src_box) == (row_i, box_i) else SOURCE_TRACK,
            "track_id": key[1],
            "checked_at": {"keyframe_token": rows[src_row].get("keyframe_token"),
                           "box_index": src_box},
            "n_members": n_members,
        }
        if verdict.get("error"):
            record["error"] = verdict["error"]
        if box_min_side(boxes[box_i]) < float(min_side_px):
            # The crop itself was never readable; the label came from a sibling
            # frame of the same object. Flagged, never silent (Decision 5).
            record["propagated_over_small"] = True
        out.append(record)
    return out


def run(
    input_dir: str,
    out_dir: str,
    taxonomy_path: str,
    *,
    dataroot: str,
    vlm,
    min_side_px: float = 32.0,
    margin: float = 1.15,
    parallel: int = 4,
    accept_degraded: bool = False,
    check_mode: str = CHECK_MODE_PER_BOX,
    track_retry_candidates: int = 3,
    allow_untracked: bool = False,
    scenes: Sequence[str] | None = None,
    max_rows: int | None = None,
    skip_classes: Sequence[str] | None = None,
) -> int:
    from PIL import Image  # deferred: keeps pure-helper imports cheap

    pre = preflight(
        input_dir, out_dir, taxonomy_path, accept_degraded=accept_degraded,
        check_mode=check_mode, min_side_px=min_side_px, margin=margin,
        track_retry_candidates=track_retry_candidates, allow_untracked=allow_untracked,
        scenes=scenes, max_rows=max_rows, skip_classes=skip_classes,
    )
    taxonomy, caption = pre["taxonomy"], pre["caption"]
    skip = pre["skip_classes"]
    man, marker = pre["manifest"], pre["marker"]
    image_size = pre["image_size_px"]
    scene_names = pre["scenes"]
    fence = pre["fence"]
    per_track = check_mode == CHECK_MODE_PER_TRACK
    scenes_root = os.path.join(input_dir, "scenes")

    clear_markers(out_dir)
    totals = {"n_rows": 0, "n_boxes": 0, "n_checked": 0, "n_confirmed": 0,
              "n_relabeled": 0, "n_unclear": 0, "n_skipped_small": 0,
              "n_skipped_class": 0, "n_errors": 0}
    confusion: dict[str, int] = {}
    confusion_by_track: dict[str, int] = {}
    assigned: set[str] = set()
    per_scene: dict[str, dict] = {}
    scenes_written: list[str] = []
    n_vlm_calls = 0
    tr = {"n_tracks": 0, "n_tracks_checked": 0, "n_boxes_from_track_verdict": 0,
          "n_boxes_untracked": 0, "n_track_retries": 0, "n_error_tracks": 0,
          "n_boxes_tracked": 0, "held_crops": 0, "held_bytes": 0}
    largest_relabel: dict | None = None
    rows_budget = max_rows

    with ThreadPoolExecutor(max_workers=max(1, parallel)) as pool:
        for scene in scene_names:
            if rows_budget is not None and rows_budget <= 0:
                break
            t0 = time.monotonic()
            rows = _read_rows(os.path.join(scenes_root, scene, "proposals.jsonl"))
            if rows_budget is not None:
                rows = rows[:rows_budget]
                rows_budget -= len(rows)
            out_rows = []
            s_counts = {"n_rows": len(rows), "n_boxes": 0, "n_relabeled": 0,
                        "n_errors": 0, "n_vlm_calls": 0}

            plan = key_index = pass_a = None
            if per_track:
                # Scene-local by construction: a fresh plan and a fresh cache per
                # scene directory, because 3b's ids restart per (scene, channel).
                plan = plan_track_checks(rows, min_side_px=min_side_px, margin=margin,
                                         skip_classes=skip)
                key_index = {rb: key for key, lst in plan.members.items() for rb in lst}
                pass_a = _pass_a(
                    rows, plan, pool=pool, vlm=vlm, caption=caption, dataroot=dataroot,
                    image_module=Image, min_side_px=min_side_px, margin=margin,
                    parallel=parallel, track_retry_candidates=track_retry_candidates)
                n_vlm_calls += pass_a["n_vlm_calls"]
                s_counts["n_vlm_calls"] = pass_a["n_vlm_calls"]
                s_counts["n_tracks"] = len(plan.members)
                s_counts["n_tracks_checked"] = len(pass_a["track_verdicts"])
                tr["n_tracks"] += len(plan.members)
                tr["n_tracks_checked"] += len(pass_a["track_verdicts"])
                tr["n_boxes_untracked"] += len(plan.untracked)
                tr["n_boxes_tracked"] += sum(len(v) for v in plan.members.values())
                tr["n_track_retries"] += pass_a["n_track_retries"]
                tr["held_crops"] = max(tr["held_crops"], pass_a["held_crops"])
                tr["held_bytes"] = max(tr["held_bytes"], pass_a["held_bytes"])
                # One entry per DECISION, not per box: the cross-mode-comparable
                # table (the per-box `confusion` below is track-length-weighted).
                for key, verdict in pass_a["track_verdicts"].items():
                    src_row, src_box = pass_a["checked_at"][key]
                    original = rows[src_row]["class_names"][src_box]
                    action = verdict_for(original, verdict.get("vlm_phrase"), caption.phrases)
                    n_members = len(plan.members[key])
                    if action == VERDICT_ERROR:
                        tr["n_error_tracks"] += 1
                    elif action == VERDICT_RELABELED:
                        ck = f"{original} -> {verdict['vlm_phrase']}"
                        confusion_by_track[ck] = confusion_by_track.get(ck, 0) + 1
                        if largest_relabel is None or n_members > largest_relabel["n_members"]:
                            largest_relabel = {
                                "scene": scene, "channel": key[0], "track_id": key[1],
                                "original": original, "new": verdict["vlm_phrase"],
                                "n_members": n_members,
                                "checked_at": {
                                    "keyframe_token": rows[src_row].get("keyframe_token"),
                                    "box_index": src_box},
                            }

            for row_i, row in enumerate(rows):
                boxes = row["boxes_xyxy_px"]
                if per_track:
                    verdicts = _pass_b_row(
                        row_i, row, plan=plan, key_index=key_index, pass_a=pass_a,
                        rows=rows, caption=caption, min_side_px=min_side_px,
                        skip_classes=skip)
                else:
                    names = row["class_names"]
                    verdicts = [None] * len(boxes)
                    futures = {}
                    if boxes:
                        with Image.open(os.path.join(dataroot, row["image_path"])) as im:
                            image = im.convert("RGB")
                        for i, box in enumerate(boxes):
                            if names[i] in skip:
                                verdicts[i] = {"action": VERDICT_SKIPPED_CLASS,
                                               "vlm_phrase": None,
                                               "verdict_source": SOURCE_NONE}
                                continue
                            if box_min_side(box) < min_side_px:
                                verdicts[i] = {"action": VERDICT_SKIPPED_SMALL,
                                               "vlm_phrase": None,
                                               "verdict_source": SOURCE_NONE}
                                continue
                            window = crop_box(box, image.size, margin=margin)
                            crop = image.crop(tuple(window))
                            futures[i] = pool.submit(_ask, vlm, crop, names[i], caption.phrases)
                            n_vlm_calls += 1
                            s_counts["n_vlm_calls"] += 1
                    for i, fut in futures.items():
                        verdicts[i] = dict(fut.result())
                        verdicts[i]["verdict_source"] = SOURCE_VLM
                checked = apply_verdicts(row, verdicts, caption=caption, taxonomy=taxonomy)
                blk = checked["vlm_check"]
                totals["n_rows"] += 1
                totals["n_boxes"] += len(boxes)
                totals["n_confirmed"] += blk["n_confirmed"]
                totals["n_relabeled"] += blk["n_relabeled"]
                totals["n_unclear"] += blk["n_unclear"]
                totals["n_skipped_small"] += blk["n_skipped_small"]
                totals["n_skipped_class"] += blk["n_skipped_class"]
                totals["n_errors"] += blk["n_errors"]
                tr["n_boxes_from_track_verdict"] += blk.get("n_from_track_verdict", 0)
                s_counts["n_boxes"] += len(boxes)
                s_counts["n_relabeled"] += blk["n_relabeled"]
                s_counts["n_errors"] += blk["n_errors"]
                for rec in blk["verdicts"]:
                    if rec["action"] == VERDICT_RELABELED:
                        key = f"{rec['original_class_name']} -> {rec['vlm_phrase']}"
                        confusion[key] = confusion.get(key, 0) + 1
                        assigned.add(rec["vlm_phrase"])
                out_rows.append(checked)
            write_jsonl_atomic(os.path.join(out_dir, "scenes", scene, "proposals.jsonl"), out_rows)
            scenes_written.append(scene)
            s_counts["seconds"] = round(time.monotonic() - t0, 1)
            per_scene[scene] = s_counts
            print(f"{STAGE}: {scene}: {s_counts['n_boxes']} boxes, "
                  f"{s_counts['n_vlm_calls']} vlm calls, "
                  f"{s_counts['n_relabeled']} relabeled, {s_counts['n_errors']} errors, "
                  f"{s_counts['seconds']}s", flush=True)

    totals["n_checked"] = totals["n_confirmed"] + totals["n_relabeled"] + totals["n_unclear"] + totals["n_errors"]
    totals["n_vlm_calls"] = n_vlm_calls
    totals["check_mode"] = check_mode
    totals["confusion"] = dict(sorted(confusion.items()))
    if per_track:
        totals["n_tracks"] = tr["n_tracks"]
        totals["n_tracks_checked"] = tr["n_tracks_checked"]
        totals["n_boxes_from_track_verdict"] = tr["n_boxes_from_track_verdict"]
        totals["n_boxes_untracked"] = tr["n_boxes_untracked"]
        totals["track_coverage"] = (
            round(tr["n_boxes_tracked"] / totals["n_boxes"], 6) if totals["n_boxes"] else None)
        totals["n_track_retries"] = tr["n_track_retries"]
        totals["n_error_tracks"] = tr["n_error_tracks"]
        totals["confusion_by_track"] = dict(sorted(confusion_by_track.items()))
        totals["max_relabel_track_members"] = (
            largest_relabel["n_members"] if largest_relabel else 0)
        totals["largest_relabel_track"] = largest_relabel

    in_use = list((man.get("class_map") or {}).get("phrases_in_use") or ())
    union = [p for p in caption.phrases if p in set(in_use) | assigned]
    vlm_block = {
        **(vlm.describe() if hasattr(vlm, "describe") else {"provider": type(vlm).__name__}),
        "asked_blind": True,
        "margin": float(margin),
        "min_side_px": float(min_side_px),
        "parallel": int(parallel),
        "check_mode": check_mode,
        "skip_classes": sorted(skip),
        "phrase_glosses": {p: g for p, g in PHRASE_GLOSSES.items() if p in caption.phrases},
    }
    if per_track:
        vlm_block["track_key_definition"] = TRACK_KEY_DEFINITION
        vlm_block["representative_ranking_rule"] = REPRESENTATIVE_RANKING_RULE
        vlm_block["track_retry_candidates"] = int(track_retry_candidates)
        vlm_block["track_retry_held_max_crops"] = tr["held_crops"]
        vlm_block["track_retry_held_max_bytes"] = tr["held_bytes"]
        vlm_block["track_accounting_notes"] = TRACK_ACCOUNTING_NOTES
        vlm_block["allow_untracked"] = bool(allow_untracked)
    manifest = {
        "spec": STAGE_SPEC,
        "stage": STAGE,
        "provider": PROVIDER,
        "config": pre["config"],
        "score_semantics": (
            "scores are the UPSTREAM detector's class confidences, unchanged; a "
            "relabeled box keeps the original arm's confidence in the ORIGINAL "
            "class (the VLM emits no calibrated replacement)"),
        "upstream": {
            "metadata_fingerprint": marker.fingerprint,
            "fingerprint_spec": (man.get("upstream") or {}).get("fingerprint_spec"),
            "input": {"dir": os.path.realpath(input_dir), "spec": man.get("spec"),
                      "checkpoint": man.get("checkpoint"), "degraded": marker.degraded,
                      "degraded_causes": list(marker.causes)},
            "accepted_degraded_upstream": accept_degraded,
        },
        "taxonomy": taxonomy.as_dict(),
        "image_size_px": list(image_size),
        "prompt": {
            "caption": caption.text,
            "caption_sha256": caption.sha256,
            "caption_is_input": False,
            "phrases": list(caption.phrases),
        },
        "class_map": {
            "path": f"vlm_check({(man.get('class_map') or {}).get('path')})",
            "sha256": "",
            "n_mapped": len(union),
            "phrases_in_use": union,
            "unreachable_phrases": [p for p in caption.phrases if p not in set(union)],
            "input_class_map": man.get("class_map"),
        },
        "vlm": vlm_block,
        # What THIS run wrote, next to what the tree under this marker actually
        # holds, so a legitimate incremental subset re-run cannot be read as a
        # full one and a foreign leftover cannot hide inside it.
        "scenes_written": list(scenes_written),
        "scenes_in_out_dir": sorted(set(fence["scenes_in_out_dir"]) | set(scenes_written)),
        "foreign_scene_dirs": fence["foreign_scene_dirs"],
        "config_matched_previous": fence["config_matched_previous"],
        "totals": totals,
        "scenes": per_scene,
    }
    if max_rows is not None:
        # No marker is written below, so nothing downstream can ever adopt this
        # tree; `partial` says why in the artifact itself as well.
        manifest["partial"] = True
        manifest["max_rows"] = int(max_rows)
    write_json_atomic(os.path.join(out_dir, "run_manifest.json"), manifest)

    causes = tuple(f"upstream: {c}" for c in marker.causes)
    if totals["n_errors"]:
        cause = (f"vlm_errors: {totals['n_errors']} boxes got no parseable verdict; "
                 "their labels rode through unchecked")
        if per_track:
            cause += (f" ({tr['n_error_tracks']} fresh-call failures amplified to "
                      f"{totals['n_errors']} boxes)")
        causes += (cause,)
    degraded = marker.degraded or bool(totals["n_errors"])
    if max_rows is None:
        write_marker(out_dir, marker.fingerprint, degraded=degraded, causes=causes)
    else:
        print(f"{STAGE}: --max-rows {max_rows}: PARTIAL tree, no marker written — "
              "no selector will ever adopt it", flush=True)
    summary = (f"{STAGE}: {totals['n_rows']} rows, {totals['n_boxes']} boxes, "
               f"{totals['n_checked']} checked, {totals['n_vlm_calls']} vlm calls, "
               f"{totals['n_relabeled']} relabeled, "
               f"{totals['n_unclear']} unclear, {totals['n_skipped_small']} skipped small, "
               f"{totals['n_skipped_class']} skipped class, "
               f"{totals['n_errors']} errors")
    if per_track:
        summary += (f"; {totals['n_tracks']} tracks, {totals['n_tracks_checked']} checked, "
                    f"track_coverage {totals['track_coverage']}")
    print(summary)
    return 1 if degraded else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage3-dir", required=True,
                        help="the ONE tree Stage 4 would consume (stage3_merged when 3f/3m ran)")
    parser.add_argument("--out-dir", required=True, help="stage3_checked tree to write")
    parser.add_argument("--taxonomy", default="configs/taxonomy_pilot_dhaka.yaml")
    parser.add_argument("--paths",
                        default=os.environ.get("DHAKASCENES_PATHS_CONFIG", "configs/paths.yaml"),
                        help="paths config; dataroot is read from here (§1.8)")
    parser.add_argument("--check-mode", choices=CHECK_MODES, default=CHECK_MODE_PER_BOX,
                        help="per_box: one call per above-floor box (today's behavior, the "
                             "default so a direct CLI run is unchanged). per_track: one call "
                             "per (channel, track_id) propagated to every box of that track "
                             "— needs a tree carrying Stage 3b's track_ids")
    parser.add_argument("--track-retry-candidates", type=int, default=3,
                        help="per_track: ranked crops held per track; #2..k are re-asked only "
                             "when the representative comes back unclear or error")
    parser.add_argument("--allow-untracked", action="store_true",
                        help="per_track: proceed even when NO box carries a track id (the run "
                             "then costs exactly per_box and says so in track_coverage)")
    parser.add_argument("--scenes", nargs="+", default=None,
                        help="check only these scene directories; the out-dir fence refuses a "
                             "subset re-run over another configuration's scenes")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="timing probe: check only the first N ROWS (not boxes) across the "
                             "scene iteration, write NO marker and record partial:true. Refuses "
                             "a non-empty out-dir; give it its own --out-dir")
    parser.add_argument("--skip-class", action="append", default=None, metavar="PHRASE",
                        help="caption phrase whose boxes are never VLM-checked (repeatable); "
                             "they keep their labels and read skipped_class in the audit. "
                             "C31: the VLM systematically flips rider crops against its own "
                             "prompt rule (pedestrian -> motorcycle), so run_stages.sh passes "
                             "'a pedestrian' and 'a motorcycle' here")
    parser.add_argument("--min-side-px", type=float, default=32.0,
                        help="boxes with a smaller side are never asked: below this the "
                             "smoke run showed confident answers on unreadable crops")
    parser.add_argument("--margin", type=float, default=1.15)
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--accept-degraded-upstream", action="store_true")
    parser.add_argument("--server-url", default=None,
                        help="use an already-running llama-server; skips spawn and GPU guard")
    parser.add_argument("--llama-server-bin",
                        default="/home/mt/dhakascenes/tools/llama.cpp/build/bin/llama-server")
    parser.add_argument("--gguf", default="/home/mt/dhakascenes/cache/checkpoints/nemotron-omni/"
                        "NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-UD-Q4_K_XL.gguf")
    parser.add_argument("--mmproj", default="/home/mt/dhakascenes/cache/checkpoints/nemotron-omni/"
                        "mmproj-BF16.gguf")
    parser.add_argument("--port", type=int, default=8092)
    parser.add_argument("--ctx-per-slot", type=int, default=8192)
    parser.add_argument("--n-cpu-moe", type=int, default=8,
                        help="MoE layers whose experts live in system RAM (24 GB fit knob)")
    parser.add_argument("--n-gpu-layers", type=int, default=999)
    parser.add_argument("--allow-shared-gpu", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args(argv)

    from pipeline.common.paths import assert_dataroot_read_only, load_paths
    paths = load_paths(args.paths)
    assert_dataroot_read_only(paths, args.out_dir)

    def _go(url: str) -> int:
        vlm = NemotronVLM(url, temperature=args.temperature, max_tokens=args.max_tokens)
        return run(
            args.stage3_dir, args.out_dir, args.taxonomy,
            dataroot=paths.dataroot, vlm=vlm,
            min_side_px=args.min_side_px, margin=args.margin,
            parallel=args.parallel, accept_degraded=args.accept_degraded_upstream,
            check_mode=args.check_mode,
            track_retry_candidates=args.track_retry_candidates,
            allow_untracked=args.allow_untracked,
            scenes=args.scenes, max_rows=args.max_rows,
            skip_classes=args.skip_class,
        )

    try:
        # Before the spawn, not after it: a taxonomy typo, a stale 3b tree or a
        # fenced out-dir must not cost a 24 GB load off NVMe first.
        preflight(
            args.stage3_dir, args.out_dir, args.taxonomy,
            accept_degraded=args.accept_degraded_upstream, check_mode=args.check_mode,
            min_side_px=args.min_side_px, margin=args.margin,
            track_retry_candidates=args.track_retry_candidates,
            allow_untracked=args.allow_untracked, scenes=args.scenes, max_rows=args.max_rows,
            skip_classes=args.skip_class,
        )
        if args.server_url:
            return _go(args.server_url)
        if not args.allow_shared_gpu:
            assert_gpu_exclusive()
        # The log lives under work_root/logs/, NOT in --out-dir: out_dir must stay
        # byte-free until clear_markers, so that rc 2 means nothing was written.
        log_path = os.path.join(
            paths.work_root, "logs",
            f"llama_server_{time.strftime('%Y%m%dT%H%M%S')}.log")
        print(f"{STAGE}: llama-server log -> {log_path}", flush=True)
        server = LlamaServer(
            bin_path=args.llama_server_bin, gguf=args.gguf, mmproj=args.mmproj,
            port=args.port, parallel=args.parallel, ctx_per_slot=args.ctx_per_slot,
            n_cpu_moe=args.n_cpu_moe, n_gpu_layers=args.n_gpu_layers,
            log_path=log_path,
        )
        with server:
            return _go(server.url)
    except (UpstreamRefusal, RuntimeError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
