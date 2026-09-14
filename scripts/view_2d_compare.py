#!/usr/bin/env python3
"""Two view_2d exports side by side, one camera at a time (2026-09-14).

view_2d.py exports one directory per run; this serves two of them — the main run and the
Stage 3c VLM-checked one — under /left/ and /right/, with viewer2d/compare.html and
viewer2d/draw.js served at the root. The page joins the two by (keyframe, camera,
proposal_index) and rings every box whose class the VLM moved, so a relabel is judged on
the crop instead of in a diff of proposals.jsonl.

Read-only, like view_2d.py's --serve: it never writes to either export.

    python -m scripts.view_2d_compare \
        --left /mnt/hdd/dhakascenes/viewer_zami/chunk_14_2d \
        --right /mnt/hdd/dhakascenes/viewer_zami/chunk_14_vlm_2d --serve 8769
"""
from __future__ import annotations
import argparse, json, os, sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIEWER = os.path.join(ROOT, "viewer2d")


def make_handler(left: str, right: str, meta: dict, viewer: str = VIEWER):
    """SimpleHTTPRequestHandler over three roots; /meta.json is the only synthetic file."""
    body = json.dumps(meta).encode()

    class Handler(SimpleHTTPRequestHandler):
        def translate_path(self, path):
            rel = urlsplit(path).path
            for prefix, root in (("/left/", left), ("/right/", right)):
                if rel.startswith(prefix):
                    self.directory = root                  # super() sanitises '..' out of the rest
                    return super().translate_path("/" + rel[len(prefix):])
            self.directory = viewer
            return super().translate_path("/compare.html" if rel == "/" else rel)

        def do_GET(self):
            if urlsplit(self.path).path == "/meta.json":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            super().do_GET()

    return Handler


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--left", required=True, help="a view_2d.py --out directory")
    ap.add_argument("--right", required=True, help="the other one, same scene")
    ap.add_argument("--left-label", default="no VLM")
    ap.add_argument("--right-label", default="VLM-checked")
    ap.add_argument("--serve", type=int, default=8769, help="port, bound to 127.0.0.1")
    a = ap.parse_args(argv)
    for side, d in (("--left", a.left), ("--right", a.right)):
        if not os.path.isfile(os.path.join(d, "index.json")):
            print(f"{side} {d}: no index.json — is it a view_2d.py export?", file=sys.stderr)
            return 2
    meta = {"left_label": a.left_label, "right_label": a.right_label,
            "left_dir": os.path.abspath(a.left), "right_dir": os.path.abspath(a.right)}
    handler = make_handler(os.path.abspath(a.left), os.path.abspath(a.right), meta)
    print(f"serving http://127.0.0.1:{a.serve}/  (Ctrl-C to stop)")
    ThreadingHTTPServer(("127.0.0.1", a.serve), handler).serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
