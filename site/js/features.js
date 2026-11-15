// Feature extraction shared by the website and the dataset builder.
// Everything here works on MediaPipe Face Landmarker output (478 points).
// Geometry functions are pure so they can also run in Node for training.

export const MIDLINE = [10, 151, 9, 8, 168, 6, 197, 195, 5, 4, 1, 2, 164, 0, 17, 18, 200, 199, 175, 152];

// Left/right landmark pairs (subject's right side first). Checked against the
// dataset mean shape in tools/train.py.
export const PAIRS = [
  [33, 263], [133, 362], [159, 386], [145, 374], [160, 387], [144, 373],
  [70, 300], [63, 293], [105, 334], [66, 296], [107, 336], [55, 285],
  [234, 454], [127, 356], [93, 323], [132, 361], [58, 288], [172, 397],
  [136, 365], [150, 379], [149, 378], [176, 400], [148, 377],
  [21, 251], [54, 284], [103, 332], [67, 297], [109, 338],
  [61, 291], [40, 270], [37, 267], [39, 269], [185, 409], [91, 321], [84, 314],
  [64, 294], [98, 327], [129, 358], [48, 278], [115, 344],
  [50, 280], [205, 425], [187, 411], [116, 345], [117, 346], [118, 347], [119, 348],
  [101, 330], [36, 266], [142, 371], [203, 423], [57, 287], [43, 273],
  [230, 450], [231, 451], [468, 473],
];

const mean2 = (a, b) => [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);

// MediaPipe gives normalised coordinates; convert to pixels.
export function toPx(landmarks, w, h) {
  return landmarks.map((p) => [p.x * w, p.y * h, p.z * w]);
}

export function alignParams(P) {
  const r = mean2(P[33], P[133]);
  const l = mean2(P[362], P[263]);
  const angle = Math.atan2(l[1] - r[1], l[0] - r[0]);
  const faceH = dist(P[10], P[152]);
  return {
    angle,
    iod: dist(r, l),
    faceH,
    cx: (P[10][0] + P[152][0]) / 2,
    cy: (P[10][1] + P[152][1]) / 2,
    side: 2.0 * faceH,
    eyeMid: mean2(r, l),
  };
}

// Rotate so the eyes are level, centre on the eye midpoint and scale so the
// distance between the eye centres is 1.
export function normalizeShape(P) {
  const a = alignParams(P);
  const c = Math.cos(-a.angle), s = Math.sin(-a.angle);
  return P.map(([x, y, z]) => {
    const dx = x - a.eyeMid[0], dy = y - a.eyeMid[1];
    return [(dx * c - dy * s) / a.iod, (dx * s + dy * c) / a.iod, (z || 0) / a.iod];
  });
}

// Head pose from landmarks. yaw/pitch are rough ratios turned into degrees.
export function pose(P) {
  const N = normalizeShape(P);
  const dr = dist(N[1], N[234]), dl = dist(N[1], N[454]);
  const yaw = Math.asin(Math.max(-1, Math.min(1, (dr - dl) / (dr + dl) * 2))) * 180 / Math.PI;
  const eyeY = (N[159][1] + N[386][1] + N[145][1] + N[374][1]) / 4;
  const mouthY = (N[13][1] + N[14][1]) / 2;
  const noseRatio = (N[1][1] - eyeY) / (mouthY - eyeY);
  const a = alignParams(P);
  return { yaw, noseRatio, roll: a.angle * 180 / Math.PI };
}

export const GEOMETRY_NAMES = [
  'face_ratio', 'jaw_ratio', 'eye_width', 'eye_spacing', 'eye_open', 'canthal_tilt',
  'nose_width', 'nose_length', 'mouth_width', 'lip_fullness', 'upper_lower_lip',
  'brow_height', 'thirds_ratio', 'chin_length',
  'symmetry_err', 'midline_dev', 'eye_size_asym', 'brow_asym',
];

export function geometry(P, pairs = PAIRS) {
  const N = normalizeShape(P);
  const d = (i, j) => dist(N[i], N[j]);
  const faceW = d(234, 454), faceH = d(10, 152);
  const eyeWR = d(33, 133), eyeWL = d(362, 263), eyeW = (eyeWR + eyeWL) / 2;
  const eyeHR = d(159, 145), eyeHL = d(386, 374);
  const tiltR = Math.atan2(N[133][1] - N[33][1], N[133][0] - N[33][0]);
  const tiltL = Math.atan2(N[362][1] - N[263][1], N[263][0] - N[362][0]);
  const mouthW = d(61, 291);
  const browR = N[105][1], browL = N[334][1];
  const eyeTopR = N[159][1], eyeTopL = N[386][1];
  const mid = d(9, 2), low = d(2, 152);

  // Symmetry: reflect each right-side point across the fitted midline.
  const x0 = MIDLINE.reduce((s, i) => s + N[i][0], 0) / MIDLINE.length;
  let err = 0;
  for (const [a, b] of pairs) {
    err += Math.hypot(N[a][0] - (2 * x0 - N[b][0]), N[a][1] - N[b][1]);
  }
  const midDev = Math.sqrt(MIDLINE.reduce((s, i) => s + (N[i][0] - x0) ** 2, 0) / MIDLINE.length);

  return {
    face_ratio: faceH / faceW,
    jaw_ratio: d(172, 397) / faceW,
    eye_width: eyeW / faceW,
    eye_spacing: d(133, 362) / eyeW,
    eye_open: (eyeHR / eyeWR + eyeHL / eyeWL) / 2,
    canthal_tilt: ((tiltR + tiltL) / 2) * 180 / Math.PI,
    nose_width: d(64, 294) / d(133, 362),
    nose_length: d(168, 2) / faceH,
    mouth_width: mouthW / d(64, 294),
    lip_fullness: d(0, 17) / mouthW,
    upper_lower_lip: d(0, 13) / Math.max(1e-6, d(14, 17)),
    brow_height: ((eyeTopR - browR) + (eyeTopL - browL)) / 2,
    thirds_ratio: low / mid,
    chin_length: d(17, 152) / low,
    symmetry_err: err / pairs.length,
    midline_dev: midDev,
    eye_size_asym: Math.abs(eyeWR * eyeHR - eyeWL * eyeHL) / (eyeW * (eyeHR + eyeHL) / 2),
    brow_asym: Math.abs((eyeTopR - browR) - (eyeTopL - browL)),
  };
}

// Flattened normalised x,y of every landmark: the "shape vector" the model uses.
export function shapeVector(P) {
  const N = normalizeShape(P);
  const out = new Array(N.length * 2);
  for (let i = 0; i < N.length; i++) { out[2 * i] = N[i][0]; out[2 * i + 1] = N[i][1]; }
  return out;
}

// ---------- browser-only helpers below ----------

export const CROP = 256;

function makeCanvas(w, h) {
  if (typeof OffscreenCanvas !== 'undefined') return new OffscreenCanvas(w, h);
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  return c;
}

// Rotated, scaled square crop around the face, plus landmarks in crop space.
export function cropFace(source, P, size = CROP) {
  const a = alignParams(P);
  const k = size / a.side;
  const canvas = makeCanvas(size, size);
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.fillStyle = '#808080';
  ctx.fillRect(0, 0, size, size);
  ctx.save();
  ctx.translate(size / 2, size / 2);
  ctx.scale(k, k);
  ctx.rotate(-a.angle);
  ctx.translate(-a.cx, -a.cy);
  ctx.drawImage(source, 0, 0);
  ctx.restore();
  const c = Math.cos(-a.angle), s = Math.sin(-a.angle);
  const lm = P.map(([x, y, z]) => {
    const dx = x - a.cx, dy = y - a.cy;
    return [size / 2 + k * (dx * c - dy * s), size / 2 + k * (dx * s + dy * c), z * k];
  });
  return { canvas, ctx, lm };
}

function boxBlur(src, w, h, r) {
  const tmp = new Float32Array(w * h), out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let s = 0, n = 0;
      for (let k = -r; k <= r; k++) { const xx = x + k; if (xx >= 0 && xx < w) { s += src[y * w + xx]; n++; } }
      tmp[y * w + x] = s / n;
    }
  }
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let s = 0, n = 0;
      for (let k = -r; k <= r; k++) { const yy = y + k; if (yy >= 0 && yy < h) { s += tmp[yy * w + x]; n++; } }
      out[y * w + x] = s / n;
    }
  }
  return out;
}

export const PIXEL_NAMES = [
  'skin_texture', 'skin_tone_var', 'skin_redness', 'redness_var', 'under_eye_dark',
  'brightness', 'contrast', 'sharpness', 'light_balance',
];

// Skin and photo-quality measurements on the aligned crop.
export function pixelStats(ctx, lm, size = CROP) {
  const img = ctx.getImageData(0, 0, size, size).data;
  const n = size * size;
  const L = new Float32Array(n), red = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const r = img[4 * i], g = img[4 * i + 1], b = img[4 * i + 2];
    L[i] = 0.299 * r + 0.587 * g + 0.114 * b;
    red[i] = (r - g) / (r + g + b + 1);
  }
  const b1 = boxBlur(L, size, size, 1);
  const b2 = boxBlur(L, size, size, 2);
  // Band-pass (small blotches) = fine blur minus coarse blur, so smooth
  // lighting gradients across the face do not count as uneven skin.
  const b7 = boxBlur(L, size, size, 7);
  const band = new Float32Array(n);
  for (let i = 0; i < n; i++) band[i] = b2[i] - b7[i];
  const iod = Math.hypot(lm[263][0] - lm[33][0], lm[263][1] - lm[33][1]) * 0.6;

  const patch = (pt, rad) => {
    const idx = [];
    const r = Math.max(2, Math.round(rad));
    const cx = Math.round(pt[0]), cy = Math.round(pt[1]);
    for (let y = cy - r; y <= cy + r; y++) {
      for (let x = cx - r; x <= cx + r; x++) {
        if (x < 1 || y < 1 || x >= size - 1 || y >= size - 1) continue;
        if ((x - cx) ** 2 + (y - cy) ** 2 <= r * r) idx.push(y * size + x);
      }
    }
    return idx;
  };
  const avg = (arr, idx) => idx.reduce((s, i) => s + arr[i], 0) / Math.max(1, idx.length);
  const sd = (arr, idx) => {
    const m = avg(arr, idx);
    return Math.sqrt(idx.reduce((s, i) => s + (arr[i] - m) ** 2, 0) / Math.max(1, idx.length));
  };

  const rad = 0.2 * iod;
  const cheekR = patch(lm[50], rad), cheekL = patch(lm[280], rad);
  const fh = [(lm[151][0] + lm[9][0]) / 2, (lm[151][1] + lm[9][1]) / 2];
  const forehead = patch(fh, rad * 0.8), chin = patch(lm[199], rad * 0.8);
  const skin = [cheekR, cheekL, forehead, chin];

  let tex = 0, tone = 0, rv = 0, rm = 0;
  for (const p of skin) {
    tex += p.reduce((s, i) => s + Math.abs(L[i] - b1[i]), 0) / Math.max(1, p.length);
    tone += sd(band, p);
    rv += sd(red, p);
    rm += avg(red, p);
  }
  const skinL = skin.reduce((s, p) => s + avg(L, p), 0) / skin.length;

  // Under-eye compared with the upper cheek just below it.
  const ueR = patch(lm[230], rad * 0.6), ueL = patch(lm[450], rad * 0.6);
  const ue = (avg(L, ueR) + avg(L, ueL)) / 2;
  const cheek = (avg(L, cheekR) + avg(L, cheekL)) / 2;

  // Sharpness: variance of the Laplacian over the eye-nose-mouth area.
  const x0 = Math.max(1, Math.floor(lm[33][0])), x1 = Math.min(size - 2, Math.ceil(lm[263][0]));
  const y0 = Math.max(1, Math.floor(lm[9][1])), y1 = Math.min(size - 2, Math.ceil(lm[17][1]));
  let lsum = 0, lsq = 0, ln = 0, fsum = 0, fsq = 0;
  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      const i = y * size + x;
      const lap = 4 * L[i] - L[i - 1] - L[i + 1] - L[i - size] - L[i + size];
      lsum += lap; lsq += lap * lap; ln++;
      fsum += L[i]; fsq += L[i] * L[i];
    }
  }
  const lapVar = ln ? lsq / ln - (lsum / ln) ** 2 : 0;
  const fMean = ln ? fsum / ln : 0;
  const contrast = ln ? Math.sqrt(Math.max(0, fsq / ln - fMean ** 2)) : 0;

  const lR = avg(L, cheekR), lL = avg(L, cheekL);
  return {
    skin_texture: tex / skin.length,
    skin_tone_var: tone / skin.length,
    skin_redness: rm / skin.length,
    redness_var: rv / skin.length,
    under_eye_dark: (cheek - ue) / Math.max(1, cheek),
    brightness: skinL,
    contrast,
    sharpness: Math.log1p(lapVar),
    light_balance: Math.abs(lR - lL) / Math.max(1, (lR + lL) / 2),
  };
}
