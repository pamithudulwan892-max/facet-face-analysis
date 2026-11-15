import { loadModel } from './scoring.js';

const LABELS = {
  geometry: 'Geometric ratios only',
  skin: 'Skin measurements only',
  'geometry+skin': 'Geometry and skin',
  shape: 'Landmark shape (956 coordinates)',
  'shape+geometry+skin': 'Shape, geometry and skin',
  embedding: 'Image embedding only',
  'embedding+shape+geometry+skin': 'Everything combined',
};

loadModel().then((m) => {
  const x = m.metrics;
  const kpi = (v, l) => `<div class="kpi"><div class="v">${v}</div><div class="l">${l}</div></div>`;
  document.getElementById('kpis').innerHTML = [
    kpi(x.test_pcc.toFixed(2), 'Correlation with raters'),
    kpi(x.test_mae.toFixed(2), 'Average error (1-5 scale)'),
    kpi(((x.test_mae * 9) / 4).toFixed(1), 'Average error (1-10 scale)'),
    kpi(x.n_faces.toLocaleString(), 'Faces in dataset'),
  ].join('');
  document.getElementById('groups').innerHTML = Object.entries(x.groups).map(([g, v]) =>
    `<tr><td>${g}</td><td class="num">${v.n}</td><td class="num">${v.pcc.toFixed(2)}</td><td class="num">${v.mae.toFixed(2)}</td></tr>`).join('');
  document.getElementById('baselines').innerHTML = Object.entries(x.baselines).map(([k, v]) =>
    `<tr${k === x.chosen ? ' style="font-weight:600"' : ''}><td>${LABELS[k] || k}${k === x.chosen ? ' (used)' : ''}</td><td class="num">${v.n_features.toLocaleString()}</td><td class="num">${v.test_pcc.toFixed(2)}</td></tr>`).join('');
}).catch(() => {
  document.getElementById('kpis').innerHTML = '<p class="muted">Model details could not be loaded.</p>';
});
