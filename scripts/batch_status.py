#!/usr/bin/env python
"""A browser view of the 38-chunk batch. Read-only, local-only, no dependencies.

It serves ONE page plus two JSON endpoints from whatever scripts/run_all_chunks.py
has written to <ssd>/exports/. It never writes, never runs anything, and never
touches a work root, so it is safe to start, kill and restart mid-batch.

  /            the page (inline CSS and JS; nothing is fetched from anywhere)
  /status.json the runner's status.json, verbatim
  /live.json   {"status": <status.json>, "ssd": {"free","total"}}

  $ python scripts/batch_status.py --port 8766      # then browse 127.0.0.1:8766
"""

from __future__ import annotations

import argparse
import http.server
import json
from pathlib import Path
import shutil
import threading

SSD = "/media/saif/f1b1e65c-6762-4561-b5b1-e7bcb0679ac4"

# Measured on chunk_0010 (668 kf) of the first session: ~2.8 s per keyframe end
# to end. Only used until the batch has finished a chunk of its own.
SECONDS_PER_KEYFRAME = 2.8


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def eta_seconds(done_keyframes, elapsed, remaining_keyframes):
    """Remaining seconds from the throughput observed so far, or None."""
    if not done_keyframes or elapsed <= 0:
        return None
    return remaining_keyframes * elapsed / done_keyframes


PAGE = """<!DOCTYPE html>
<meta charset="utf-8"><title>DhakaScenes batch</title>
<style>
:root { color-scheme: dark; }
body { background:#14161a; color:#dfe3ea; font:13px/1.45 ui-monospace,Menlo,Consolas,monospace;
       margin:0; padding:14px 16px; }
h1 { font-size:15px; margin:0 0 8px; font-weight:600; letter-spacing:.02em }
#head { display:flex; flex-wrap:wrap; gap:6px 18px; margin-bottom:12px; color:#9aa4b2 }
#head b { color:#dfe3ea; font-weight:600 }
table { border-collapse:collapse; width:100%; }
th { text-align:left; font-weight:600; color:#8d96a5; padding:3px 6px; border-bottom:1px solid #2a2f39;
     position:sticky; top:0; background:#14161a }
td { padding:2px 6px; border-bottom:1px solid #1d2129; white-space:nowrap }
tr.row { cursor:pointer } tr.row:hover td { background:#1b1f27 }
td.num { text-align:right }
.cell { display:inline-block; width:15px; height:13px; margin-right:2px; border-radius:2px;
        text-align:center; font-size:9px; line-height:13px; color:#0b0d10 }
.s-pending{background:#2b3039;color:#5d6675} .s-running{background:#3f8cff;color:#06101f}
.s-ok{background:#3fb950} .s-degraded{background:#d29922} .s-refused{background:#f85149}
.s-crashed{background:#a04040} .s-blocked{background:#6e5494} .s-failed{background:#f85149}
.s-done{background:#3fb950} .s-interrupted{background:#8b949e}
.pill { padding:0 6px; border-radius:3px; color:#0b0d10; font-weight:600 }
.detail td { white-space:pre-wrap; background:#0f1116; color:#9aa4b2; padding:8px 12px }
.detail b { color:#dfe3ea }
pre { margin:4px 0 0; max-height:260px; overflow:auto; font:12px/1.4 ui-monospace,monospace; color:#c2c9d4 }
.bar { height:5px; background:#2b3039; border-radius:3px; overflow:hidden; width:100%; margin-top:6px }
.bar i { display:block; height:100%; background:#3fb950 }
</style>
<h1>DhakaScenes &mdash; batch_20260912</h1>
<div id="head"></div><div class="bar"><i id="progress"></i></div>
<table id="grid"><thead></thead><tbody></tbody></table>
<script>
const DEFAULT_STAGES = ["0","1","3","3f","3m","4","5","6s","7","8","9","release"];
const SPK = 2.8;                       // measured seconds per keyframe, fallback only
let openRows = new Set(), stages = DEFAULT_STAGES;

function hb(n){ if(!n) return "0 B"; const u=["B","KB","MB","GB","TB"]; let i=0;
  while(n>=1024 && i<u.length-1){ n/=1024; i++; } return i? n.toFixed(1)+" "+u[i] : n+" B"; }
function hms(s){ if(s==null||!isFinite(s)) return "\u2014"; s=Math.max(0,Math.round(s));
  const h=Math.floor(s/3600), m=Math.floor(s%3600/60); return h? h+"h"+String(m).padStart(2,"0")
  : m? m+"m"+String(s%60).padStart(2,"0") : s+"s"; }
function secs(a,b){ if(!a) return null; return ((b? new Date(b): new Date()) - new Date(a))/1000; }
function esc(t){ return String(t==null?"":t).replace(/[&<>]/g, c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c])); }

function render(live){
  const st = live.status || {}, chunks = st.chunks || [];
  stages = st.stages && st.stages.length ? st.stages : DEFAULT_STAGES;
  const done = chunks.filter(c=>c.state=="done"||c.state=="degraded");
  const running = chunks.filter(c=>c.state=="running");
  const kfDone = done.reduce((a,c)=>a+(c.keyframes||0),0);
  const kfTotal = chunks.reduce((a,c)=>a+(c.keyframes||0),0);
  const elapsed = secs(st.started);
  // Throughput from finished chunks only; wall-clock seconds, so it already
  // accounts for however many workers are actually running.
  const rate = kfDone && elapsed ? kfDone/elapsed : null;      // keyframes per second
  const remain = kfTotal - kfDone;
  const eta = rate ? remain/rate : remain*SPK/Math.max(1, st.workers||1);
  document.getElementById("head").innerHTML = [
    ["chunks", `<b>${done.length}/${chunks.length}</b>`],
    ["running", `<b>${running.length}</b>/${st.workers||"?"}`],
    ["keyframes", `<b>${kfDone.toLocaleString()}</b> / ${kfTotal.toLocaleString()}`],
    ["throughput", `<b>${rate? (rate*60).toFixed(1) : "\u2014"}</b> kf/min`],
    ["started", `<b>${esc((st.started||"").replace("T"," ").slice(0,19))}</b>`],
    ["elapsed", `<b>${hms(elapsed)}</b>`],
    ["eta finish", `<b>${hms(eta)}</b>`],
    ["ssd free", `<b>${hb(live.ssd && live.ssd.free)}</b>`],
    ["updated", `<b>${esc((st.updated||"").slice(11,19))}</b>`],
  ].map(([k,v])=>`<span>${k} ${v}</span>`).join("");
  document.getElementById("progress").style.width =
    (chunks.length? 100*done.length/chunks.length : 0) + "%";

  document.querySelector("#grid thead").innerHTML = "<tr>" +
    ["#","session","scene","kf","state","stage"].map(h=>`<th>${h}</th>`).join("") +
    `<th>stages (${stages.join(" ")})</th><th>elapsed</th><th>eta</th></tr>`;

  const body = chunks.map(c => {
    const el = secs(c.started, c.finished);
    const per = rate ? c.keyframes/rate : c.keyframes*SPK;
    const left = c.state=="running" ? (el!=null? per-el : per) : null;
    const cells = stages.map(s => {
      const e = (c.stage_states||{})[s] || {};
      const state = e.state || "pending";
      const tip = s + (e.seconds!=null ? " " + e.seconds + "s" : "") +
                  (e.causes ? " " + e.causes.join("; ") : "");
      return `<span class="cell s-${esc(state)}" title="${esc(tip)}">${esc(s)}</span>`;
    }).join("");
    const rows = [`<tr class="row" data-n="${c.n}">` +
      `<td class="num">${c.n}</td><td>${esc((c.session||"").replace("dhaka_",""))}</td>` +
      `<td>${esc((c.scene||"").split("_chunk_").pop())}</td>` +
      `<td class="num">${(c.keyframes||0).toLocaleString()}</td>` +
      `<td><span class="pill s-${esc(c.state)}">${esc(c.state)}</span>` +
      (c.blocked? " " + esc(c.blocked) : "") + `</td>` +
      `<td>${esc(c.current_stage||"")}</td><td>${cells}</td>` +
      `<td class="num">${hms(el)}</td><td class="num">${c.state=="running"? hms(left):"\u2014"}</td></tr>`];
    if (openRows.has(c.n)) {
      const causes = Object.entries(c.stage_states||{})
        .filter(([,e])=>e.causes && e.causes.length)
        .map(([s,e])=>`<b>${esc(s)}</b>: ${esc(e.causes.join("; "))}`).join("<br>");
      const ex = c.export ? `${c.export.files} files, ${hb(c.export.bytes)}, ` +
        `<b class="${c.export.symlinks? "s-refused":""}">${c.export.symlinks} symlinks</b>` : "\u2014";
      rows.push(`<tr class="detail"><td colspan="9">` +
        `<b>scene</b> ${esc(c.scene)}   <b>worker</b> ${esc(c.worker)}   ` +
        `<b>started</b> ${esc(c.started)}   <b>finished</b> ${esc(c.finished)}<br>` +
        `<b>export</b> ${ex}<br><b>log</b> ${esc(c.log||"\u2014")}` +
        (causes? `<br><b>degraded</b><br>${causes}` : "") +
        (c.error_tail && c.error_tail.length ? `<pre>${esc(c.error_tail.join("\\n"))}</pre>` : "") +
        `</td></tr>`);
    }
    return rows.join("");
  }).join("");
  document.querySelector("#grid tbody").innerHTML = body;
  document.querySelectorAll("tr.row").forEach(tr => tr.onclick = () => {
    const n = +tr.dataset.n;
    openRows.has(n) ? openRows.delete(n) : openRows.add(n); tick();
  });
}

function tick(){ fetch("/live.json").then(r=>r.json()).then(render).catch(()=>{}); }
tick(); setInterval(tick, 10000);
</script>
"""


def make_handler(exports: Path):
    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send(self, code, body, content_type):
            body = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _status(self):
            try:
                return json.loads((exports / "status.json").read_text())
            except (OSError, ValueError):
                return {"chunks": [], "note": f"no status.json under {exports} yet"}

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                self._send(200, PAGE, "text/html; charset=utf-8")
            elif path == "/status.json":
                self._send(200, json.dumps(self._status()), "application/json")
            elif path == "/live.json":
                usage = shutil.disk_usage(exports if exports.exists() else Path("/"))
                self._send(200, json.dumps({
                    "status": self._status(),
                    "ssd": {"free": usage.free, "total": usage.total, "used": usage.used},
                }), "application/json")
            else:
                self._send(404, "not found\n", "text/plain; charset=utf-8")

        def log_message(self, *args):        # a dashboard should be quiet
            pass

    return Handler


def serve(exports, port=8766, host="127.0.0.1", background=False):
    """Bind and (optionally) run in a thread. Returns (httpd, thread_or_None)."""
    exports = Path(exports)
    httpd = http.server.ThreadingHTTPServer((host, port), make_handler(exports))
    httpd.daemon_threads = True
    thread = None
    if background:
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
    return httpd, thread


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ssd", default=SSD)
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args(argv)
    exports = Path(args.ssd) / "exports"
    httpd, _ = serve(exports, args.port)
    print(f"batch status on 127.0.0.1:{args.port}  (reading {exports})", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
