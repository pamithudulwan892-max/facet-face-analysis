import { createLandmarker, createEmbedder } from '../site/js/vision.js';
import { toPx, cropFace, pixelStats, pose } from '../site/js/features.js';

const IMG_DIR = '../data/raw/SCUT-FBP5500_v2/Images/';
const LABELS = '../data/raw/SCUT-FBP5500_v2/train_test_files/All_labels.txt';
const BATCH = 250;

const status = document.getElementById('status');
const logEl = document.getElementById('log');
const log = (m) => { logEl.textContent += m + '\n'; logEl.scrollTop = 1e9; };
const r4 = (v) => Math.round(v * 1e4) / 1e4;

function loadImage(src) {
  return new Promise((res, rej) => {
    const im = new Image();
    im.onload = () => res(im);
    im.onerror = rej;
    im.src = src;
  });
}

async function exists(name) {
  const r = await fetch('../data/landmarks/' + name, { method: 'HEAD' });
  return r.ok;
}

async function main() {
  const text = await (await fetch(LABELS)).text();
  const files = text.trim().split(/\r?\n/).map((l) => l.split(/\s+/)[0]).sort();
  const landmarker = await createLandmarker('IMAGE');
  const embedder = await createEmbedder();
  status.textContent = `Models ready. ${files.length} images.`;
  window.__extract = { done: 0, total: files.length, finished: false, missed: [] };
  const t0 = performance.now();

  for (let b = 0; b * BATCH < files.length; b++) {
    const name = `batch_${String(b).padStart(3, '0')}.json`;
    if (await exists(name)) { log(`${name} already saved, skipping`); window.__extract.done += BATCH; continue; }
    const rows = [];
    for (const file of files.slice(b * BATCH, (b + 1) * BATCH)) {
      const im = await loadImage(IMG_DIR + file);
      const res = landmarker.detect(im);
      if (!res.faceLandmarks.length) { window.__extract.missed.push(file); log('no face: ' + file); continue; }
      const P = toPx(res.faceLandmarks[0], im.naturalWidth, im.naturalHeight);
      const crop = cropFace(im, P);
      const emb = embedder.embed(crop.canvas).embeddings[0].floatEmbedding;
      rows.push({
        file,
        lm: P.map((p) => [r4(p[0]), r4(p[1]), r4(p[2])]),
        pose: Object.fromEntries(Object.entries(pose(P)).map(([k, v]) => [k, r4(v)])),
        px: Object.fromEntries(Object.entries(pixelStats(crop.ctx, crop.lm)).map(([k, v]) => [k, r4(v)])),
        emb: Array.from(emb, (v) => Math.round(v * 1e5) / 1e5),
      });
      window.__extract.done++;
    }
    await fetch('/save', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ file: name, rows }) });
    const secs = ((performance.now() - t0) / 1000).toFixed(0);
    log(`${name}: ${rows.length} faces (${window.__extract.done}/${files.length}, ${secs}s)`);
    status.textContent = `Working... ${window.__extract.done}/${files.length}`;
  }
  window.__extract.finished = true;
  status.textContent = `Finished. Missed ${window.__extract.missed.length} images.`;
}

main().catch((e) => { status.textContent = 'Error: ' + e.message; console.error(e); });
