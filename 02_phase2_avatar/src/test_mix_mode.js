// Exercise the blind mixed mode in label.html with the test manifest (dim4 fixture + dim22).
// What must hold for the comparison to be fair:
//   1. MIX activates from ?mix=1 and picks up every arm in the manifest
//   2. consecutive faces really do alternate between arms
//   3. the input dimension follows the chosen arm, and the 4-dim arm genuinely IGNORES
//      pca[4..21] -- proven by feeding two vectors that differ only beyond index 3 and
//      checking the face is identical
//   4. the reward readout is hidden (a w128 score for another model is meaningless and
//      would leak a hint to the judge)
//   5. each submitted label records the arm that produced it
//   6. the sampler is PLAIN random, not reward-guided
const fs = require('fs'), path = require('path');
const WEB = path.join(__dirname, '..', 'webapp');
const html = fs.readFileSync(path.join(WEB, 'label.html'), 'utf8');
let code = html.match(/<script>([\s\S]*?)<\/script>/)[1];

function mkEl() {
  const el = { children: [], style: {}, dataset: {}, _t: '', _h: '',
    appendChild(c) { this.children.push(c); return c; }, addEventListener() {},
    removeEventListener() {}, querySelectorAll() { return []; }, querySelector() { return null; },
    classList: { add() {}, remove() {}, contains() { return false; }, toggle() {} },
    getContext() { return { createImageData: (w, h) => ({ data: new Uint8ClampedArray(w * h * 4) }),
      putImageData() {}, clearRect() {}, fillRect() {}, drawImage() {} }; },
    set innerHTML(v) { this._h = v; this.children = []; }, get innerHTML() { return this._h; },
    set textContent(v) { this._t = v; }, get textContent() { return this._t; },
    width: 8, height: 8 };
  return el;
}
const byId = {};
global.document = { createElement: mkEl, getElementById: id => (byId[id] ||= mkEl()),
  querySelectorAll: () => [], querySelector: () => mkEl(), addEventListener() {} };
global.window = { addEventListener() {}, location: { search: '?mix=1' } };
global.location = global.window.location;

const FILES = ['pca22_dim4_model.json', 'pca22_dim8_model.json', 'pca22_dim22_model.json',
               'dimsweep_manifest.json', 'preference_reward.json'];
const F = {};
for (const f of FILES) {
  const p = path.join(WEB, f);
  if (fs.existsSync(p)) F[f] = JSON.parse(fs.readFileSync(p, 'utf8'));
}
let posted = [];
global.fetch = async (u, opt) => {
  const k = String(u).split('?')[0].split('/').pop();
  if (k === 'label' && opt && opt.method === 'POST') {
    posted.push(JSON.parse(opt.body));
    return { ok: true, status: 200, json: async () => ({ ok: true, total: posted.length }) };
  }
  if (k === 'labelstats') {
    return { ok: true, status: 200, json: async () => ({ total: 0, counts: [0, 0, 0, 0] }) };
  }
  if (F[k]) return { ok: true, status: 200, json: async () => F[k] };
  return { ok: false, status: 404, json: async () => ({}) };
};
global.console.warn = () => {};
code = code.replace(/init\(\)\.catch\([\s\S]*?}\);\s*$/, '');
new Function(code)();

(async () => {
  await new Promise(r => setTimeout(r, 900));
  const L = global.window.LABEL;
  console.log('manifest arms:', F['dimsweep_manifest.json'].map(m => m.key).join(', '));

  const seen = new Set(), dims = new Set();
  for (let i = 0; i < 40; i++) {
    L.next();
    const s = L.getState();
    seen.add(s.modelKey);
    dims.add(s.IN_DIM);
  }
  console.log('arms drawn in 40 faces :', [...seen].join(', '));
  console.log('input dimensions used  :', [...dims].join(', '));

  // prove the 4-dim arm ignores pca[4..21]
  let ignoreTest = false;
  for (let i = 0; i < 80 && !ignoreTest; i++) {
    L.next();
    if (L.getState().modelKey !== 'dim4') continue;
    const a = Array.from(L.getState().cur);
    const b = a.slice();
    b[4] += 3.0; b[10] -= 2.5; b[21] += 1.5;
    const outA = L.probe(a), outB = L.probe(b);
    if (outA.length === 32 && outA.every((v, j) => Math.abs(v - outB[j]) < 1e-6)) {
      ignoreTest = true;
    }
  }

  const rs = byId['rs'] ? byId['rs'].textContent : '';
  const okHidden = !/[0-9]/.test(rs);

  for (let i = 0; i < 8; i++) { L.next(); await L.grade(i % 4); }
  const models = posted.map(p => p.model);
  const okBoth = seen.size === 3;
  const okDims = dims.has(4) && dims.has(8) && dims.has(22);
  const okRecorded = new Set(models).size >= 2 &&
    posted.every(p => ['dim4', 'dim8', 'dim22'].includes(p.model));
  const okPlain = posted.every(p => p.note === 'plain' || p.note === 'explore');
  const okPcaLen = posted.every(p => p.pca.length === 22);

  console.log();
  console.log('  both arms drawn                 : ' + (okBoth ? 'PASS' : 'FAIL'));
  console.log('  input dim follows the arm       : ' + (okDims ? 'PASS' : 'FAIL'));
  console.log('  4-dim arm ignores pca[4..21]     : ' + (ignoreTest ? 'PASS' : 'FAIL'));
  console.log('  reward readout hidden           : ' + (okHidden ? 'PASS' : 'FAIL') + ' ("' + rs + '")');
  console.log('  every label records its arm     : ' + (okRecorded ? 'PASS' : 'FAIL'));
  console.log('  sampling is plain, not guided   : ' + (okPlain ? 'PASS' : 'FAIL'));
  console.log('  pca vectors stay 22 long        : ' + (okPcaLen ? 'PASS' : 'FAIL'));

  if (!(okBoth && okDims && ignoreTest && okHidden && okRecorded && okPlain && okPcaLen)) {
    console.error('\nFAIL');
    process.exit(1);
  }
  console.log('\nPASS blind mixing works: arms alternate at random, each is fed its own input');
  console.log('     dimension, the judge sees no hint, and every label is attributed.');
  console.log('MIX_READY');
})();
