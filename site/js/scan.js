import { createLandmarker, createEmbedder } from './vision.js';
import { toPx, cropFace, pixelStats, geometry, shapeVector, pose, alignParams } from './features.js';
import { loadModel, analyse } from './scoring.js';

const $ = (id) => document.getElementById(id);
const video = $('video'), overlay = $('overlay'), still = $('still');
const octx = overlay.getContext('2d');
const views = ['intro', 'live', 'working', 'error'];
const show = (id) => views.forEach((v) => $(v).classList.toggle('hidden', v !== id));

const FRAMES = 8;           // frames averaged per scan
const HOLD_MS = 900;        // how long all checks must stay green before auto scan

let landmarkerVideo, landmarkerImage, embedder, model;
let stream = null, running = false, lastTs = -1;
let readySince = 0, capturing = null, prevPts = null, lightLevel = 0, frameNo = 0;

function msg(text) {
  $('msg').textContent = text;
  $('msg').classList.toggle('hidden', !text);
}

function fail(text) {
  stopCamera();
  $('errorText').innerHTML = text;
  show('error');
}

async function loadAll(needVideo) {
  $('loadState').textContent = 'Loading the analysis models (about 15 MB, only the first time)...';
  const jobs = [loadModel().then((m) => (model = m)), embedder ? null : createEmbedder().then((e) => (embedder = e))];
  if (needVideo && !landmarkerVideo) jobs.push(createLandmarker('VIDEO').then((l) => (landmarkerVideo = l)));
  if (!needVideo && !landmarkerImage) jobs.push(createLandmarker('IMAGE').then((l) => (landmarkerImage = l)));
  await Promise.all(jobs);
  $('loadState').textContent = '';
}

// ---------- measurement ----------

function smileOf(res) {
  const cats = res.faceBlendshapes?.[0]?.categories;
  if (!cats) return null;
  const get = (n) => cats.find((c) => c.categoryName === n)?.score ?? 0;
  return (get('mouthSmileLeft') + get('mouthSmileRight')) / 2;
}

function measure(source, P, smile) {
  const crop = cropFace(source, P);
  const emb = embedder.embed(crop.canvas).embeddings[0].floatEmbedding;
  return {
    geo: geometry(P, model.pairs),
    px: pixelStats(crop.ctx, crop.lm),
    shape: shapeVector(P),
    emb: Array.from(emb),
    pose: pose(P),
    smile,
  };
}

function averageSamples(list) {
  const avgObj = (key) => {
    const out = {};
    for (const k of Object.keys(list[0][key])) out[k] = list.reduce((s, x) => s + x[key][k], 0) / list.length;
    return out;
  };
  const avgArr = (key) => list[0][key].map((_, i) => list.reduce((s, x) => s + x[key][i], 0) / list.length);
  const smiles = list.map((x) => x.smile).filter((v) => v !== null);
  return {
    geo: avgObj('geo'), px: avgObj('px'), pose: avgObj('pose'),
    shape: avgArr('shape'), emb: avgArr('emb'),
    smile: smiles.length ? smiles.reduce((a, b) => a + b, 0) / smiles.length : null,
    frames: list.length,
  };
}

function finish(sample, source) {
  sample.source = source;
  const result = analyse(model, sample);
  try {
    sessionStorage.setItem('facet:result', JSON.stringify(result));
  } catch (e) {
    console.warn(e);
  }
  stopCamera();
  location.href = 'results.html';
}

// ---------- drawing ----------

function sizeOverlay(w, h) {
  if (overlay.width !== w || overlay.height !== h) { overlay.width = w; overlay.height = h; }
}

function drawGuide(ok) {
  const w = overlay.width, h = overlay.height;
  octx.save();
  octx.fillStyle = 'rgba(0,0,0,.35)';
  octx.beginPath();
  octx.rect(0, 0, w, h);
  octx.ellipse(w / 2, h * 0.48, h * 0.27, h * 0.36, 0, 0, Math.PI * 2);
  octx.fill('evenodd');
  octx.lineWidth = Math.max(2, h / 200);
  octx.strokeStyle = ok ? '#6cc0a5' : 'rgba(255,255,255,.8)';
  octx.beginPath();
  octx.ellipse(w / 2, h * 0.48, h * 0.27, h * 0.36, 0, 0, Math.PI * 2);
  octx.stroke();
  octx.restore();
}

function drawPoints(P, color = 'rgba(108,192,165,.85)') {
  const r = Math.max(1, overlay.height / 380);
  octx.fillStyle = color;
  for (let i = 0; i < 468; i += 2) {
    octx.beginPath();
    octx.arc(P[i][0], P[i][1], r, 0, Math.PI * 2);
    octx.fill();
  }
}

// ---------- camera loop ----------

function checkFrame(res, w, h) {
  const c = { face: false, size: false, center: false, straight: false, light: false, still: false };
  const n = res.faceLandmarks.length;
  if (n !== 1) return { c, P: null, hint: n > 1 ? 'Only one face please' : 'Looking for your face...' };
  c.face = true;
  const P = toPx(res.faceLandmarks[0], w, h);
  const a = alignParams(P);
  const rel = a.faceH / h;
  c.size = rel > 0.36 && rel < 0.85;
  c.center = Math.abs(a.cx / w - 0.5) < 0.12 && Math.abs(a.cy / h - 0.5) < 0.14;
  const pz = pose(P);
  const ref = model?.pose_reference ?? { noseRatio: 0.64, noseRatio_p90: 0.08, yaw_p90: 15 };
  const pitchOff = Math.abs(pz.noseRatio - ref.noseRatio);
  c.straight = Math.abs(pz.yaw) < Math.max(8, ref.yaw_p90) && Math.abs(pz.roll) < 6 && pitchOff < Math.max(0.06, ref.noseRatio_p90 * 1.2);

  if (frameNo % 6 === 0) {
    const small = cropFace(video, P, 64);
    const d = small.ctx.getImageData(16, 20, 32, 28).data;
    let s = 0;
    for (let i = 0; i < d.length; i += 4) s += 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
    lightLevel = s / (d.length / 4);
  }
  c.light = lightLevel > 75;

  const key = [1, 33, 263, 61, 291, 152];
  if (prevPts) {
    const mv = key.reduce((s, i, j) => s + Math.hypot(P[i][0] - prevPts[j][0], P[i][1] - prevPts[j][1]), 0) / key.length;
    c.still = mv / a.faceH < 0.012;
  }
  prevPts = key.map((i) => P[i]);

  let hint = '';
  if (!c.size) hint = rel <= 0.36 ? 'Move a little closer' : 'Move back a little';
  else if (!c.center) hint = 'Centre your face in the oval';
  else if (!c.straight) hint = Math.abs(pz.roll) >= 6 ? 'Keep your head level' : 'Look straight at the camera';
  else if (!c.light) hint = 'Find more light, facing a window or lamp';
  else if (!c.still) hint = 'Hold still';
  return { c, P, hint, smile: smileOf(res) };
}

function setChecks(c) {
  for (const li of $('checks').children) li.classList.toggle('ok', !!c[li.dataset.check]);
}

function setRing(fraction) {
  $('ring').classList.toggle('hidden', fraction <= 0);
  $('ringArc').setAttribute('stroke-dashoffset', String(119.4 * (1 - fraction)));
}

function loop() {
  if (!running) return;
  if (video.readyState >= 2 && video.currentTime !== lastTs) {
    lastTs = video.currentTime;
    frameNo++;
    const w = video.videoWidth, h = video.videoHeight;
    sizeOverlay(w, h);
    const res = landmarkerVideo.detectForVideo(video, performance.now());
    const { c, P, hint, smile } = checkFrame(res, w, h);
    const allOk = Object.values(c).every(Boolean);
    octx.clearRect(0, 0, w, h);
    drawGuide(allOk);
    if (P) drawPoints(P);
    setChecks(c);
    $('scanNow').disabled = !(c.face && c.size);

    if (capturing) {
      if (P && c.size) {
        capturing.samples.push(measure(video, P, smile));
        setRing(capturing.samples.length / FRAMES);
        msg('Scanning, hold still...');
        if (capturing.samples.length >= FRAMES) {
          const sample = averageSamples(capturing.samples);
          capturing = null;
          running = false;
          show('working');
          setTimeout(() => finish(sample, 'camera'), 50);
          return;
        }
      } else {
        msg('Lost your face, hold still in the oval');
      }
    } else {
      msg(hint || 'Perfect, hold still');
      if (allOk) {
        readySince ||= performance.now();
        const t = (performance.now() - readySince) / HOLD_MS;
        setRing(Math.min(1, t) * 0.25);
        if (t >= 1) capturing = { samples: [] };
      } else {
        readySince = 0;
        setRing(0);
      }
    }
  }
  requestAnimationFrame(loop);
}

async function startCamera() {
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
    fail('Camera access needs a secure connection (https) and a modern browser. You can still <strong>use a photo instead</strong>.');
    return;
  }
  $('startBtn').disabled = true;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
  } catch (e) {
    $('startBtn').disabled = false;
    if (e.name === 'NotAllowedError') fail('Camera permission was blocked. Allow camera access for this site in your browser settings (the camera icon in the address bar), then try again. Or <strong>use a photo instead</strong>.');
    else if (e.name === 'NotFoundError' || e.name === 'OverconstrainedError') fail('No camera was found on this device. You can <strong>use a photo instead</strong>.');
    else fail('Could not start the camera: ' + e.message);
    return;
  }
  try {
    await loadAll(true);
  } catch (e) {
    fail('The analysis models failed to load. Check your connection and refresh the page.');
    console.error(e);
    return;
  }
  video.srcObject = stream;
  await video.play();
  $('placeholder').classList.add('hidden');
  video.classList.remove('hidden');
  still.classList.add('hidden');
  overlay.classList.add('mirror');
  $('stage').classList.remove('contain');
  show('live');
  running = true;
  readySince = 0; capturing = null; prevPts = null;
  requestAnimationFrame(loop);
}

function stopCamera() {
  running = false;
  capturing = null;
  if (stream) stream.getTracks().forEach((t) => t.stop());
  stream = null;
  setRing(0);
  msg('');
}

async function usePhoto(file) {
  if (!file) return;
  stopCamera();
  video.classList.add('hidden');
  const url = URL.createObjectURL(file);
  try {
    await new Promise((resolve, reject) => {
      still.onload = resolve;
      still.onerror = reject;
      still.src = url;
    });
  } catch {
    fail('That file could not be opened as an image.');
    return;
  }
  $('placeholder').classList.add('hidden');
  still.classList.remove('hidden');
  overlay.classList.remove('mirror');
  $('stage').classList.add('contain');
  show('working');
  $('workingText').textContent = 'Loading models and measuring your photo...';
  try {
    await loadAll(false);
  } catch (e) {
    fail('The analysis models failed to load. Check your connection and refresh the page.');
    return;
  }
  const w = still.naturalWidth, h = still.naturalHeight;
  sizeOverlay(w, h);
  const res = landmarkerImage.detect(still);
  if (res.faceLandmarks.length === 0) { fail('No face was found in that photo. Try a clear, front-facing photo with good light.'); return; }
  if (res.faceLandmarks.length > 1) { fail('That photo has more than one face. Please use a photo of just you.'); return; }
  const P = toPx(res.faceLandmarks[0], w, h);
  octx.clearRect(0, 0, w, h);
  drawPoints(P);
  const sample = measure(still, P, smileOf(res));
  sample.frames = 1;
  URL.revokeObjectURL(url);
  setTimeout(() => finish(sample, 'photo'), 400);
}

$('startBtn').addEventListener('click', startCamera);
$('stopBtn').addEventListener('click', () => {
  stopCamera();
  video.classList.add('hidden');
  octx.clearRect(0, 0, overlay.width, overlay.height);
  $('placeholder').classList.remove('hidden');
  $('startBtn').disabled = false;
  show('intro');
});
$('scanNow').addEventListener('click', () => { if (running && !capturing) capturing = { samples: [] }; });
$('fileInput').addEventListener('change', (e) => usePhoto(e.target.files[0]));
$('retryBtn').addEventListener('click', () => {
  $('startBtn').disabled = false;
  $('placeholder').classList.remove('hidden');
  still.classList.add('hidden');
  octx.clearRect(0, 0, overlay.width, overlay.height);
  $('fileInput').value = '';
  show('intro');
});
window.addEventListener('pagehide', stopCamera);

// Start fetching the scoring model early; it is small.
loadModel().catch(() => {});
