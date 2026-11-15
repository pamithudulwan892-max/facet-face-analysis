// Runs the same geometry code the website uses (site/js/features.js) over the
// saved dataset landmarks, so training and the live site measure faces identically.
// Usage: node tools/build_geometry.mjs [pairs.json]
import fs from 'node:fs';
import path from 'node:path';
import { geometry, shapeVector, GEOMETRY_NAMES, PAIRS } from '../site/js/features.js';

const root = path.resolve(path.dirname(new URL(import.meta.url).pathname.replace(/^\/(\w:)/, '$1')), '..');
const dir = path.join(root, 'data', 'landmarks');
const pairsFile = process.argv[2];
const pairs = pairsFile ? JSON.parse(fs.readFileSync(pairsFile, 'utf8')) : PAIRS;

const out = [];
const header = ['file', ...GEOMETRY_NAMES];
const shapeRows = [];
for (const f of fs.readdirSync(dir).filter((x) => x.startsWith('batch_')).sort()) {
  for (const row of JSON.parse(fs.readFileSync(path.join(dir, f), 'utf8'))) {
    const g = geometry(row.lm, pairs);
    out.push([row.file, ...GEOMETRY_NAMES.map((k) => g[k].toFixed(6))].join(','));
    shapeRows.push([row.file, ...shapeVector(row.lm).map((v) => v.toFixed(5))].join(','));
  }
}
fs.writeFileSync(path.join(root, 'data', 'geometry.csv'), header.join(',') + '\n' + out.join('\n') + '\n');
fs.writeFileSync(path.join(root, 'data', 'shape.csv'), shapeRows.join('\n') + '\n');
console.log(`wrote ${out.length} rows using ${pairs.length} pairs`);
