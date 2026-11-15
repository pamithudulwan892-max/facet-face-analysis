// Turns measured features into the score, the factor breakdown and tips.

let modelPromise;
export function loadModel() {
  modelPromise ??= fetch(new URL('../models/model.json', import.meta.url)).then((r) => {
    if (!r.ok) throw new Error('Could not load the scoring model');
    return r.json();
  });
  return modelPromise;
}

// Where a value sits within the dataset (0-100), using stored quantiles.
export function percentile(q, v) {
  if (v <= q[0]) return 0;
  if (v >= q[q.length - 1]) return 100;
  let lo = 0, hi = q.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (q[mid] <= v) lo = mid; else hi = mid;
  }
  const span = q[hi] - q[lo];
  const t = span > 0 ? (v - q[lo]) / span : 0.5;
  return ((lo + t) / (q.length - 1)) * 100;
}

const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const lowerIsBetter = (model, name, v) => 100 - percentile(model.feature_quantiles[name], v);

export function predictRating(model, sample) {
  const parts = {
    embedding: sample.emb,
    shape: sample.shape,
    geometry: model.geometry_names.map((k) => sample.geo[k]),
    skin: model.skin_names.map((k) => sample.px[k]),
  };
  let y = model.bias, j = 0;
  for (const name of model.inputs) {
    const v = parts[name];
    if (!v || v.length !== model.sizes[name]) throw new Error('Feature size mismatch for ' + name);
    for (let i = 0; i < v.length; i++) y += model.weights[j++] * v[i];
  }
  return y;
}

export const toTen = (r) => 1 + ((r - 1) * 9) / 4;

function photoScore(model, s) {
  const notes = [];
  const b = s.px.brightness;
  let light = 100;
  if (b < 95) { light = clamp(100 - (95 - b) * 1.6, 0, 100); notes.push('dark'); }
  if (b > 215) { light = clamp(100 - (b - 215) * 2.5, 0, 100); notes.push('bright'); }
  const sharp = clamp(percentile(model.feature_quantiles.sharpness, s.px.sharpness) * 1.6, 0, 100);
  if (sharp < 45) notes.push('blurry');
  const balance = lowerIsBetter(model, 'light_balance', s.px.light_balance);
  if (balance < 30) notes.push('side-light');
  const ref = model.pose_reference;
  const yaw = Math.abs(s.pose.yaw), roll = Math.abs(s.pose.roll);
  const pitch = Math.abs(s.pose.noseRatio - ref.noseRatio);
  const poseScore = clamp(100
    - Math.max(0, yaw - ref.yaw_p50) * (40 / Math.max(1, ref.yaw_p90 - ref.yaw_p50))
    - Math.max(0, roll - 2) * 6
    - Math.max(0, pitch - ref.noseRatio_p90 / 2) * (40 / Math.max(0.01, ref.noseRatio_p90 / 2)), 0, 100);
  if (poseScore < 70) notes.push('pose');
  return { score: 0.3 * light + 0.25 * sharp + 0.2 * balance + 0.25 * poseScore, notes };
}

export function analyse(model, sample) {
  const rating = clamp(predictRating(model, sample), 1, 5);
  const score = toTen(rating);
  const mae10 = (model.metrics.test_mae * 9) / 4;
  const pct = percentile(model.rating_quantiles, rating);

  const g = sample.geo, p = sample.px;
  const symmetry = 0.55 * lowerIsBetter(model, 'symmetry_err', g.symmetry_err)
    + 0.15 * lowerIsBetter(model, 'midline_dev', g.midline_dev)
    + 0.15 * lowerIsBetter(model, 'eye_size_asym', g.eye_size_asym)
    + 0.15 * lowerIsBetter(model, 'brow_asym', g.brow_asym);

  const bal = model.balance;
  const z = bal.names.map((k, i) => (g[k] - bal.mean[i]) / bal.std[i]);
  const dist = Math.sqrt(z.reduce((s, v) => s + v * v, 0) / z.length);
  const balance = 100 - percentile(bal.quantiles, dist);

  const skin = (lowerIsBetter(model, 'skin_texture', p.skin_texture)
    + lowerIsBetter(model, 'skin_tone_var', p.skin_tone_var)
    + lowerIsBetter(model, 'redness_var', p.redness_var)) / 3;
  const eyes = lowerIsBetter(model, 'under_eye_dark', p.under_eye_dark);
  const photo = photoScore(model, sample);

  const factors = [
    { id: 'symmetry', name: 'Symmetry', score: symmetry, changeable: false,
      text: 'How closely the left and right sides of your face mirror each other, measured from 478 points. Almost nobody is perfectly symmetric, and small differences are normal.' },
    { id: 'balance', name: 'Facial balance', score: balance, changeable: false,
      text: 'How close your face proportions (face shape, eye spacing, nose and mouth width, facial thirds) are to the dataset average. Research links "average" proportions with higher ratings.' },
    { id: 'skin', name: 'Skin clarity', score: skin, changeable: true,
      text: 'Fine texture, blotchiness and uneven redness on your cheeks, forehead and chin. Camera noise and lighting also affect this one.' },
    { id: 'eyes', name: 'Eye area', score: eyes, changeable: true,
      text: 'How much darker the skin under your eyes is than your cheeks. Sleep, hydration and allergies often show up here.' },
    { id: 'photo', name: 'Photo conditions', score: photo.score, changeable: true,
      text: 'Lighting, focus and head angle during the scan. These do not describe your face, but they change how faces come across in photos.' },
  ].map((f) => ({ ...f, score: Math.round(clamp(f.score, 0, 100)) }));

  const measurements = describeMeasurements(model, g);
  const tips = buildTips(factors, sample, photo.notes, g);

  return {
    created: new Date().toISOString(),
    rating: +rating.toFixed(3),
    score: +score.toFixed(1),
    range: [+clamp(score - mae10, 1, 10).toFixed(1), +clamp(score + mae10, 1, 10).toFixed(1)],
    percentile: Math.round(pct),
    factors,
    measurements,
    tips,
    frames: sample.frames || 1,
    mesh: sample.shape,
    smile: sample.smile ?? null,
    source: sample.source,
  };
}

const MEASURE_INFO = {
  face_ratio: ['Face length to width', 'Higher means a longer face shape, lower a rounder or wider one.'],
  jaw_ratio: ['Jaw width to cheek width', 'How wide the jaw is compared with the cheekbones.'],
  eye_spacing: ['Eye spacing', 'Gap between the eyes divided by one eye width.'],
  eye_open: ['Eye openness', 'Eye height divided by eye width.'],
  canthal_tilt: ['Eye tilt (degrees)', 'Angle from the inner to the outer eye corner. Positive means the outer corner sits higher.'],
  nose_width: ['Nose width', 'Nose width divided by the gap between the eyes.'],
  mouth_width: ['Mouth width', 'Mouth width divided by nose width.'],
  lip_fullness: ['Lip fullness', 'Lip height divided by mouth width.'],
  thirds_ratio: ['Lower to middle third', 'Nose-to-chin length divided by brow-to-nose length.'],
  brow_height: ['Brow height', 'Distance from brow to upper eyelid, relative to eye spacing.'],
};

function describeMeasurements(model, g) {
  return Object.entries(MEASURE_INFO).map(([k, [label, help]]) => {
    const q = model.feature_quantiles[k];
    return { key: k, label, help, value: +g[k].toFixed(3), typical: [+q[25].toFixed(3), +q[75].toFixed(3)], pct: Math.round(percentile(q, g[k])) };
  });
}

function buildTips(factors, sample, photoNotes, g) {
  const f = Object.fromEntries(factors.map((x) => [x.id, x.score]));
  const tips = [];
  const add = (priority, icon, title, body) => tips.push({ priority, icon, title, body });

  if (photoNotes.includes('dark')) add(95, 'light', 'Use brighter, softer light', 'Face a window or a lamp placed in front of you at eye level. Dim light hides detail and adds camera noise, which can lower skin and overall scores.');
  if (photoNotes.includes('bright')) add(90, 'light', 'Soften harsh light', 'Very bright or direct light washes out features. Step back from the light source or diffuse it with a thin curtain.');
  if (photoNotes.includes('side-light')) add(85, 'light', 'Even out the lighting', 'One side of your face is noticeably brighter. Light from the front, or a second light or white wall on the darker side, gives a more even look.');
  if (photoNotes.includes('blurry')) add(80, 'camera', 'Get a sharper image', 'Wipe the camera lens, hold still, and make sure your face is in good light so the camera does not blur fine detail.');
  if (photoNotes.includes('pose')) add(75, 'camera', 'Keep the camera at eye level', 'Look straight at the lens with your head level. A camera below the face makes the jaw and nose look larger; above it shortens the chin.');

  if (f.skin < 60) {
    add(70 - f.skin / 4, 'skin', 'Build a simple skincare routine', 'A gentle cleanser twice a day, a fragrance-free moisturiser, and sunscreen (SPF 30 or more) every morning are the basics that help texture and evenness most over a few weeks.');
    if (sample.px.redness_var > 0) add(55, 'skin', 'Calm redness', 'Avoid very hot water and harsh scrubs, and patch-test new products. If redness, acne or flaking does not settle, a dermatologist can help.');
  } else {
    add(30, 'skin', 'Keep protecting your skin', 'Your skin reads as fairly even. Daily sunscreen is the single most effective habit for keeping it that way.');
  }
  if (f.eyes < 55) add(65 - f.eyes / 4, 'sleep', 'Look after the under-eye area', 'Aim for 7 to 9 hours of sleep, drink enough water, and cut screens before bed. A cold compress in the morning reduces puffiness. Persistent dark circles can also come from allergies or genetics.');

  add(40, 'water', 'Hydration and diet', 'Drinking enough water and eating plenty of fruit and vegetables supports skin over time. Very salty food and alcohol the night before often show up as puffiness.');

  if (g.brow_asym > 0 && f.symmetry < 50) add(45, 'groom', 'Tidy and balance the brows', 'Brows strongly frame the face. Light grooming so both brows have a similar shape and height makes the face read as more symmetric.');
  if (f.balance < 50) {
    const long = g.face_ratio > 1.2;
    add(42, 'groom', 'Use hair and grooming to frame your face', long
      ? 'Your face is longer than average in this dataset. Styles with some volume at the sides, or a fringe, balance a longer face shape.'
      : 'Your face is wider than average in this dataset. Styles with some height on top and less volume at the sides lengthen the look of the face.');
  }
  if (sample.smile !== null && sample.smile < 0.15) add(38, 'smile', 'Try a relaxed, natural smile', 'Studies of face ratings consistently find that a slight genuine smile is rated as more attractive and approachable than a neutral face in photos.');
  add(20, 'posture', 'Posture for photos', 'Push your chin slightly forward and down, relax your shoulders, and take a slow breath before the photo. It sharpens the jawline and relaxes the face.');

  return tips.sort((a, b) => b.priority - a.priority).slice(0, 7);
}
