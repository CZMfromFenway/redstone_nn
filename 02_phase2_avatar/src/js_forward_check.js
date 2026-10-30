// Numeric equivalence check: the page's JavaScript forward pass vs the Python model.
// We read pca22_model.json, run the JS math, and print reference vectors that Python
// then has to reproduce. If the two disagree, the web page is showing a different model.
const fs = require('fs');
const path = require('path');

const p = path.join(__dirname, '..', 'webapp', 'pca22_model.json');
const j = JSON.parse(fs.readFileSync(p, 'utf8'));
const W1 = j.gen.W1, b1 = j.gen.b1, W2 = j.gen.W2, b2 = j.gen.b2, W3 = j.gen.W3, b3 = j.gen.b3;

function forward(pca, z) {
  const h1 = new Float64Array(256), h2 = new Float64Array(256), out = new Float64Array(512);
  for (let jj = 0; jj < 256; jj++) {
    let s = b1[jj]; const row = W1[jj];
    for (let i = 0; i < 22; i++) s += row[i] * pca[i];
    for (let i = 0; i < 64; i++) s += row[22 + i] * z[i];
    h1[jj] = s > 0 ? s : 0;
  }
  for (let jj = 0; jj < 256; jj++) {
    let s = b2[jj]; const row = W2[jj];
    for (let i = 0; i < 256; i++) s += row[i] * h1[i];
    h2[jj] = s > 0 ? s : 0;
  }
  for (let jj = 0; jj < 512; jj++) {
    let s = b3[jj]; const row = W3[jj];
    for (let i = 0; i < 256; i++) s += row[i] * h2[i];
    out[jj] = s;
  }
  return out;
}

// three fixed test cases: all-zero pca, gallery[0]'s pca, and a deterministic ramp
const cases = [];
cases.push({ name: 'pca_all_zero', pca: new Array(22).fill(0) });
cases.push({ name: 'pca_gallery0', pca: j.gallery[0].pca });
cases.push({ name: 'pca_ramp', pca: Array.from({ length: 22 }, (_, i) => ((i % 7) - 3) * 0.37) });

// deterministic z: same formula the page uses with seed 12345 (mulberry32 + Box-Muller)
function rng(s) { let a = s >>> 0; return function () { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
function gauss(r) { let u = 0, v = 0; while (u === 0) u = r(); while (v === 0) v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
const r = rng(12345); const z = new Float64Array(64); for (let i = 0; i < 64; i++) z[i] = gauss(r);

const out = { z_seed12345: Array.from(z), cases: [] };
for (const c of cases) {
  const lg = forward(c.pca, z);
  out.cases.push({
    name: c.name,
    pca: c.pca,
    logits_first8: Array.from(lg.slice(0, 8)),
    logits_sum: Array.from(lg).reduce((a, b) => a + b, 0),
    logits_at_pos0: Array.from(lg.slice(0, 16)),
    argmax_pos0: Array.from(lg.slice(0, 16)).indexOf(Math.max(...Array.from(lg.slice(0, 16)))),
  });
}
fs.writeFileSync(path.join(__dirname, 'js_forward_reference.json'), JSON.stringify(out, null, 2));
console.log('JS forward pass done. summary:');
out.cases.forEach(c => console.log(`  ${c.name}: logits[0:8]=${c.logits_first8.map(v => v.toFixed(5)).join(',')}  sum=${c.logits_sum.toFixed(5)}  argmax@pos0=${c.argmax_pos0}`));
console.log('wrote js_forward_reference.json');
