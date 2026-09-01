"""Stage 3c — VLM label check: stage-3-shaped tree -> stage3_checked/.

Re-classifies every proposal crop with a local vision-language model (Nemotron
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
from pipeline.stage3_proposals.proposals import build_caption, load_taxonomy  # noqa: E402

STAGE = "stage3c_check"
STAGE_SPEC = "dhakascenes-pilot/stage3c_check/v1"

VERDICT_CONFIRMED = "confirmed"          # model agrees with the incoming label
VERDICT_RELABELED = "relabeled"          # model names a different caption phrase
VERDICT_UNCLEAR = "unclear"              # model declines; label kept
VERDICT_SKIPPED_SMALL = "skipped_small"  # crop below --min-side-px; never asked
VERDICT_ERROR = "error"                  # unparseable/failed reply; label kept

UNCLEAR = "unclear"

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
              VERDICT_SKIPPED_SMALL: 0, VERDICT_ERROR: 0}
    for i, v in enumerate(verdicts):
        action = v["action"]
        counts[action] += 1
        record = {"action": action, "vlm_phrase": v.get("vlm_phrase")}
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
    out["vlm_check"] = {
        "spec": STAGE_SPEC,
        "verdicts": records,
        "n_relabeled": counts[VERDICT_RELABELED],
        "n_confirmed": counts[VERDICT_CONFIRMED],
        "n_unclear": counts[VERDICT_UNCLEAR],
        "n_skipped_small": counts[VERDICT_SKIPPED_SMALL],
        "n_errors": counts[VERDICT_ERROR],
    }
    return out


# ---------------------------------------------------------------------------
# The Nemotron client and server manager (exercised by the smoke run, not unit
# tests: there is no honest way to mock a 24 GB model)
# ---------------------------------------------------------------------------

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
        key = tuple(allowed_phrases)
        text = self._prompt_cache.get(key)
        if text is None:
            text = self._prompt_cache[key] = _prompt_text(allowed_phrases)
        buf = io.BytesIO()
        crop.convert("RGB").save(buf, format="JPEG", quality=92)
        data_uri = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
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
# The stage driver
# ---------------------------------------------------------------------------

def _read_rows(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def run(
    input_dir: str,
    out_dir: str,
    taxonomy_path: str,
    *,
    dataroot: str,
    vlm,
    min_side_px: float = 24.0,
    margin: float = 1.15,
    parallel: int = 4,
    accept_degraded: bool = False,
) -> int:
    from PIL import Image  # deferred: keeps pure-helper imports cheap

    taxonomy = load_taxonomy(taxonomy_path)
    caption = build_caption(taxonomy.phrases)

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
    scenes = sorted(os.listdir(scenes_root)) if os.path.isdir(scenes_root) else []
    if not scenes:
        raise UpstreamRefusal(f"{scenes_root} has no scenes; nothing to check")

    clear_markers(out_dir)
    totals = {"n_rows": 0, "n_boxes": 0, "n_checked": 0, "n_confirmed": 0,
              "n_relabeled": 0, "n_unclear": 0, "n_skipped_small": 0, "n_errors": 0}
    confusion: dict[str, int] = {}
    assigned: set[str] = set()
    per_scene: dict[str, dict] = {}

    with ThreadPoolExecutor(max_workers=max(1, parallel)) as pool:
        for scene in scenes:
            t0 = time.monotonic()
            rows = _read_rows(os.path.join(scenes_root, scene, "proposals.jsonl"))
            out_rows = []
            s_counts = {"n_rows": len(rows), "n_boxes": 0, "n_relabeled": 0, "n_errors": 0}
            for row in rows:
                boxes = row["boxes_xyxy_px"]
                names = row["class_names"]
                verdicts: list[dict | None] = [None] * len(boxes)
                futures = {}
                if boxes:
                    with Image.open(os.path.join(dataroot, row["image_path"])) as im:
                        image = im.convert("RGB")
                    for i, box in enumerate(boxes):
                        if box_min_side(box) < min_side_px:
                            verdicts[i] = {"action": VERDICT_SKIPPED_SMALL, "vlm_phrase": None}
                            continue
                        window = crop_box(box, image.size, margin=margin)
                        crop = image.crop(tuple(window))
                        futures[i] = pool.submit(_ask, vlm, crop, names[i], caption.phrases)
                for i, fut in futures.items():
                    verdicts[i] = fut.result()
                checked = apply_verdicts(row, verdicts, caption=caption, taxonomy=taxonomy)
                blk = checked["vlm_check"]
                totals["n_rows"] += 1
                totals["n_boxes"] += len(boxes)
                totals["n_confirmed"] += blk["n_confirmed"]
                totals["n_relabeled"] += blk["n_relabeled"]
                totals["n_unclear"] += blk["n_unclear"]
                totals["n_skipped_small"] += blk["n_skipped_small"]
                totals["n_errors"] += blk["n_errors"]
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
            s_counts["seconds"] = round(time.monotonic() - t0, 1)
            per_scene[scene] = s_counts
            print(f"{STAGE}: {scene}: {s_counts['n_boxes']} boxes, "
                  f"{s_counts['n_relabeled']} relabeled, {s_counts['n_errors']} errors, "
                  f"{s_counts['seconds']}s", flush=True)

    totals["n_checked"] = totals["n_confirmed"] + totals["n_relabeled"] + totals["n_unclear"] + totals["n_errors"]
    totals["confusion"] = dict(sorted(confusion.items()))

    in_use = list((man.get("class_map") or {}).get("phrases_in_use") or ())
    union = [p for p in caption.phrases if p in set(in_use) | assigned]
    manifest = {
        "spec": STAGE_SPEC,
        "stage": STAGE,
        "provider": "nemotron_vlm_label_check",
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
        "vlm": {
            **(vlm.describe() if hasattr(vlm, "describe") else {"provider": type(vlm).__name__}),
            "asked_blind": True,
            "margin": float(margin),
            "min_side_px": float(min_side_px),
            "parallel": int(parallel),
            "phrase_glosses": {p: g for p, g in PHRASE_GLOSSES.items() if p in caption.phrases},
        },
        "totals": totals,
        "scenes": per_scene,
    }
    write_json_atomic(os.path.join(out_dir, "run_manifest.json"), manifest)

    causes = tuple(f"upstream: {c}" for c in marker.causes)
    if totals["n_errors"]:
        causes += (f"vlm_errors: {totals['n_errors']} boxes got no parseable verdict; "
                   "their labels rode through unchecked",)
    degraded = marker.degraded or bool(totals["n_errors"])
    write_marker(out_dir, marker.fingerprint, degraded=degraded, causes=causes)
    print(f"{STAGE}: {totals['n_rows']} rows, {totals['n_boxes']} boxes, "
          f"{totals['n_checked']} checked, {totals['n_relabeled']} relabeled, "
          f"{totals['n_unclear']} unclear, {totals['n_skipped_small']} skipped small, "
          f"{totals['n_errors']} errors")
    return 1 if degraded else 0


def _ask(vlm, crop, original_phrase, phrases) -> dict:
    try:
        reply = vlm(crop, original_phrase, phrases)
    except Exception as exc:  # noqa: BLE001 — any failed ask is an audit row, never a crash
        return {"action": VERDICT_ERROR, "vlm_phrase": None, "error": str(exc)[:300]}
    parsed = parse_vlm_reply(reply, phrases)
    return {"action": verdict_for(original_phrase, parsed, phrases), "vlm_phrase": parsed}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage3-dir", required=True,
                        help="the ONE tree Stage 4 would consume (stage3_merged when 3f/3m ran)")
    parser.add_argument("--out-dir", required=True, help="stage3_checked tree to write")
    parser.add_argument("--taxonomy", default="configs/taxonomy_pilot_dhaka.yaml")
    parser.add_argument("--paths", default="configs/paths.yaml",
                        help="paths config; dataroot is read from here (§1.8)")
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
        )

    try:
        if args.server_url:
            return _go(args.server_url)
        if not args.allow_shared_gpu:
            assert_gpu_exclusive()
        server = LlamaServer(
            bin_path=args.llama_server_bin, gguf=args.gguf, mmproj=args.mmproj,
            port=args.port, parallel=args.parallel, ctx_per_slot=args.ctx_per_slot,
            n_cpu_moe=args.n_cpu_moe, n_gpu_layers=args.n_gpu_layers,
            log_path=os.path.join(args.out_dir, "llama_server.log"),
        )
        with server:
            return _go(server.url)
    except (UpstreamRefusal, RuntimeError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
