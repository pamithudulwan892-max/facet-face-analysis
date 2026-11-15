const $ = (id) => document.getElementById(id);

const ICONS = {
  light: '<path d="M12 3v2M12 19v2M4.2 4.2l1.4 1.4M18.4 18.4l1.4 1.4M3 12h2M19 12h2M4.2 19.8l1.4-1.4M18.4 5.6l1.4-1.4"/><circle cx="12" cy="12" r="4"/>',
  camera: '<path d="M4 8h3l2-2h6l2 2h3v11H4z"/><circle cx="12" cy="13" r="3.5"/>',
  skin: '<path d="M12 3c3 4 6 7 6 11a6 6 0 0 1-12 0c0-4 3-7 6-11z"/>',
  sleep: '<path d="M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5z"/>',
  water: '<path d="M7 4h10l-1.5 16h-7z"/><path d="M8 9h8"/>',
  groom: '<path d="M4 9c3-3 13-3 16 0M6 15c2 1.5 10 1.5 12 0"/>',
  smile: '<circle cx="12" cy="12" r="9"/><path d="M8 14c1 1.5 2.4 2.2 4 2.2s3-.7 4-2.2M9 9.5h.01M15 9.5h.01"/>',
  posture: '<circle cx="12" cy="5" r="2"/><path d="M12 7v7M8 10h8M12 14l-3 6M12 14l3 6"/>',
};

const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const color = (s) => (s >= 65 ? 'var(--good)' : s >= 40 ? 'var(--mid)' : 'var(--low)');

function headline(p) {
  if (p >= 80) return ['Rated above most faces in the dataset', 'A strong result.'];
  if (p >= 55) return ['Above the dataset average', 'A solid result.'];
  if (p >= 45) return ['Right around the dataset average', 'Most people land in this range.'];
  if (p >= 20) return ['A little below the dataset average', 'Photo conditions often account for part of this.'];
  return ['Below the dataset average for this scan', 'Lighting, angle and camera quality can pull this down a lot, so try the photo tips and scan again.'];
}

function drawMesh(shape) {
  const svg = $('mesh');
  const xs = [], ys = [];
  for (let i = 0; i < shape.length; i += 2) { xs.push(shape[i]); ys.push(shape[i + 1]); }
  const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
  const pad = 0.12, size = Math.max(maxX - minX, maxY - minY) + 2 * pad;
  const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
  svg.setAttribute('viewBox', `${cx - size / 2} ${cy - size / 2} ${size} ${size}`);
  const r = size / 260;
  let out = '';
  for (let i = 0; i < xs.length && i < 468; i++) out += `<circle cx="${xs[i].toFixed(4)}" cy="${ys[i].toFixed(4)}" r="${r.toFixed(4)}"/>`;
  svg.innerHTML = out;
}

function render(r) {
  $('report').classList.remove('hidden');
  $('scoreNum').textContent = r.score.toFixed(1);
  requestAnimationFrame(() => {
    $('scoreArc').setAttribute('stroke-dashoffset', String(326.7 * (1 - r.score / 10)));
  });
  const [h, extra] = headline(r.percentile);
  $('headline').textContent = h;
  $('subline').textContent = `Likely range ${r.range[0].toFixed(1)} to ${r.range[1].toFixed(1)}. That is higher than about ${r.percentile}% of faces in the research dataset. ${extra}`;
  $('mae').textContent = ((r.range[1] - r.range[0]) / 2).toFixed(1);

  $('factors').innerHTML = r.factors.map((f) => `
    <div class="card factor">
      <div class="factor-head">
        <h3 style="margin:0">${esc(f.name)}</h3>
        <span class="factor-score" style="color:${color(f.score)}">${f.score}</span>
      </div>
      <div class="bar"><span data-w="${f.score}" style="background:${color(f.score)}"></span></div>
      <div><span class="tag ${f.changeable ? 'yes' : ''}">${f.changeable ? 'You can change this' : 'Mostly fixed by bone structure'}</span></div>
      <p class="muted small" style="margin:0">${esc(f.text)}</p>
    </div>`).join('');
  requestAnimationFrame(() => document.querySelectorAll('.bar > span').forEach((s) => (s.style.width = s.dataset.w + '%')));

  $('tips').innerHTML = r.tips.map((t) => `
    <div class="tip">
      <div class="tip-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">${ICONS[t.icon] || ICONS.skin}</svg></div>
      <div><h3>${esc(t.title)}</h3><p>${esc(t.body)}</p></div>
    </div>`).join('');

  $('measures').innerHTML = r.measurements.map((m) => `
    <tr><td>${esc(m.label)}</td><td class="num">${m.value.toFixed(2)}</td><td class="num">${m.typical[0].toFixed(2)} to ${m.typical[1].toFixed(2)}</td><td class="muted">${esc(m.help)}</td></tr>`).join('');

  if (r.mesh) drawMesh(r.mesh);
  const when = new Date(r.created).toLocaleString();
  $('meta').textContent = `Scanned ${when} from ${r.source === 'photo' ? 'an uploaded photo' : `${r.frames} camera frames`}. Stored only in this tab.`;
}

$('clearBtn').addEventListener('click', () => {
  sessionStorage.removeItem('facet:result');
  location.href = 'index.html';
});

let result = null;
try { result = JSON.parse(sessionStorage.getItem('facet:result')); } catch { result = null; }
if (result && result.factors) render(result);
else $('empty').classList.remove('hidden');
