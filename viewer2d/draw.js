// What a 2D box looks like, for every page that draws one (2026-09-14).
//
// index.html draws six cameras of one run; compare.html draws one camera of two runs
// side by side. Palette, status colours and the canvas painter live here so the two
// cannot disagree about what a box looks like, and so a relabel badge added once shows
// up in both. view_2d.py copies this file next to index.html in every export.

// The same palette viewer/index.html uses, so a car is the same green in the 3D viewer
// and "the 3D one shows something else" is never a colour question.
export const CLASS_COLOR = {
  'a car': '#4cd137', 'a bus': '#fbc531', 'a truck': '#e84118', 'a motorcycle': '#00a8ff',
  'a bicycle': '#9c88ff', 'a pedestrian': '#ff6b81', 'a rickshaw': '#00d2d3',
  'an auto rickshaw': '#badc58', 'a construction vehicle': '#c56cf0', 'a trailer': '#7f8fa6',
  'a traffic cone': '#f5f6fa', 'a road barrier': '#ffa801',
};
export const classColor = (name) => CLASS_COLOR[name] ||
  '#' + ((Math.abs([...String(name)].reduce((h, c) => h * 31 + c.charCodeAt(0) | 0, 7)) % 0xffffff) | 0x606060).toString(16).slice(-6);

// Stage 7 status -> corner marker. out_of_r3 is grey because on the four non-ZED
// cameras it is the EXPECTED answer (no stereo there at all), not a failure.
export const STATUS_COLOR = {
  fit: '#4cd137', out_of_r3: '#7f8fa6',
  too_few_stereo: '#ffa801', beyond_stereo_cap: '#ffa801', no_points: '#ffa801',
};
export const STATUS_FALLBACK = '#e84118';          // channel_disabled, no_prior, no_ground_plane, anything new
export const statusColor = (s) => (s ? STATUS_COLOR[s] || STATUS_FALLBACK : null);
export const DIFF_COLOR = '#ff00ff';               // not in CLASS_COLOR: a diff is never a class

const EDGES = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
const ALL = { mask: true, arm_a: true, arm_b: true, tag: true, wire: true };

/** Draw one camera's boxes over its image. `cam` is a kf JSON camera entry.
 *
 *  opts.view       which overlays are on (see ALL);
 *  opts.isSelected (box) -> white, thicker;
 *  opts.isDiff     (box) -> a thick DIFF_COLOR ring outside the box.
 */
export function paintBoxes(cv, img, cam, opts = {}) {
  const g = cv.getContext('2d');
  const view = opts.view || ALL;
  const isSelected = opts.isSelected || (() => false);
  const isDiff = opts.isDiff || (() => false);
  if (!cam || !img || !img.naturalWidth) { g.clearRect(0, 0, cv.width, cv.height); return; }
  cv.width = img.naturalWidth; cv.height = img.naturalHeight;
  g.setTransform(1, 0, 0, 1, 0, 0);
  g.drawImage(img, 0, 0);
  // Everything below is in ORIGINAL image pixels; one transform makes the exported
  // coordinates and the downscaled JPEG agree without touching a single number.
  const s = cv.width / cam.native[0];
  g.setTransform(s, 0, 0, s, 0, 0);
  g.font = '15px system-ui, sans-serif';
  g.textBaseline = 'bottom';
  g.lineJoin = 'round';
  for (const b of cam.boxes) {
    if (b.arm === 'arm_b' ? !view.arm_b : !view.arm_a) continue;
    const color = classColor(b.cls), sel = isSelected(b);
    const [x0, y0, x1, y1] = b.xyxy;
    if (view.mask && b.poly) {
      g.beginPath();
      g.moveTo(b.poly[0], b.poly[1]);
      for (let i = 2; i < b.poly.length; i += 2) g.lineTo(b.poly[i], b.poly[i + 1]);
      g.closePath();
      g.fillStyle = color; g.globalAlpha = sel ? 0.45 : 0.25; g.fill(); g.globalAlpha = 1;
    }
    if (view.wire && b.wire) {
      g.strokeStyle = color; g.globalAlpha = sel ? 0.9 : 0.4; g.lineWidth = 1; g.setLineDash([]);
      g.beginPath();
      for (const [p, q] of EDGES) {
        // project_corners zeroes a corner that is behind the camera; skip its edges.
        if ((b.wire[p * 2] === 0 && b.wire[p * 2 + 1] === 0) || (b.wire[q * 2] === 0 && b.wire[q * 2 + 1] === 0)) continue;
        g.moveTo(b.wire[p * 2], b.wire[p * 2 + 1]); g.lineTo(b.wire[q * 2], b.wire[q * 2 + 1]);
      }
      g.stroke(); g.globalAlpha = 1;
    }
    if (isDiff(b)) {
      g.strokeStyle = DIFF_COLOR; g.lineWidth = 4; g.setLineDash([]);
      g.strokeRect(x0 - 4, y0 - 4, x1 - x0 + 8, y1 - y0 + 8);
    }
    g.strokeStyle = sel ? '#ffffff' : color;
    g.lineWidth = sel ? 3 : 2;
    g.setLineDash(b.arm === 'arm_b' ? [6, 4] : []);      // dashed = the RSUD20K fine-tune
    g.strokeRect(x0, y0, x1 - x0, y1 - y0);
    g.setLineDash([]);
    // `was ...` only when Stage 3c actually moved the label; an unchecked export has no b.vlm.
    const was = b.vlm && b.vlm.original_class_name ? ` ← ${b.vlm.original_class_name}` : '';
    const label = `${b.cls} ${b.score.toFixed(2)} ${b.arm === 'arm_b' ? 'B' : 'A'}${was}`;
    g.fillStyle = 'rgba(0,0,0,0.6)';
    g.fillRect(x0, y0 - 16, g.measureText(label).width + 6, 16);
    g.fillStyle = sel ? '#ffffff' : color;
    g.fillText(label, x0 + 3, y0 - 2);
    const sc = statusColor(b.status);
    if (sc) {
      g.fillStyle = sc;
      g.fillRect(x1 - 11, y0, 11, 11);                  // corner marker, top-right of the 2D box
      if (view.tag) {
        g.font = '12px ui-monospace, monospace';
        const tw = g.measureText(b.status).width + 6;
        g.fillStyle = 'rgba(0,0,0,0.6)'; g.fillRect(x0, y1 - 14, tw, 14);
        g.fillStyle = sc; g.fillText(b.status, x0 + 3, y1 - 2);
        g.font = '15px system-ui, sans-serif';
      }
    }
  }
  g.setTransform(1, 0, 0, 1, 0, 0);
}

/** The shared legend, as HTML; `extra` spans are prepended (compare.html adds the diff ring). */
export function legendHTML(extra = []) {
  const parts = [...extra];
  for (const [k, c] of Object.entries(STATUS_COLOR)) parts.push(`<span><span class="sw" style="background:${c}"></span>${k}</span>`);
  parts.push(`<span><span class="sw" style="background:${STATUS_FALLBACK}"></span>other status</span>`);
  parts.push('<span class="dim">solid = arm A (COCO YOLO) · dashed = arm B (RSUD20K)</span>');
  for (const [name, c] of Object.entries(CLASS_COLOR)) parts.push(`<span><span class="sw" style="background:${c}"></span>${name}</span>`);
  return parts.join('');
}
