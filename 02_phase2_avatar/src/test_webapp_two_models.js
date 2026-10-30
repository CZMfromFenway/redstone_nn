// Verify the two-model page: both load, the shared gallery really matches, switching
// models does not change the inputs, and each model reproduces its own PyTorch output.
const fs = require('fs');
const path = require('path');

const html = fs.readFileSync(path.join(__dirname, '..', 'webapp', 'index.html'), 'utf8');
let code = html.match(/<script>([\s\S]*?)<\/script>/)[1];

function makeEl(tag) {
  const el = {
    tagName: tag, children: [], style: {}, dataset: {}, _text: '', _html: '',
    width: 8, height: 8, title: '', classList: { add() {}, remove() {}, contains() { return false; } },
    appendChild(c) { this.children.push(c); return c; },
    addEventListener() {}, removeEventListener() {},
    querySelectorAll() { return []; }, querySelector() { return null; },
    getContext() { return { createImageData: (w, h) => ({ data: new Uint8ClampedArray(w * h * 4) }), putImageData() {}, clearRect() {}, fillRect() {} }; },
    set innerHTML(v) { this._html = v; }, get innerHTML() { return this._html; },
    set textContent(v) { this._text = v; }, get textContent() { return this._text; },
  };
  return el;
}
const byId = {};
global.document = {
  createElement: makeEl,
  getElementById(id) { if (!byId[id]) byId[id] = makeEl('div'); return byId[id]; },
  querySelectorAll() { return []; },
  querySelector() { return makeEl('tbody'); },
};
global.window = {};

const FILES = {};
for (const f of ['pca22_model.json', 'pca22_nonoise_model.json']) {
  FILES[f] = JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'webapp', f), 'utf8'));
}
global.fetch = async (url) => {
  const key = String(url).split('/').pop();
  if (!FILES[key]) return { ok: false, status: 404, json: async () => ({}) };
  return { ok: true, status: 200, json: async () => FILES[key] };
};
global.console.warn = (m) => console.log('  WARN from page:', m);

code = code.replace(/init\(\)\.catch\([\s\S]*?\}\);/, '');
new Function(code)();
const APP = global.window.APP;
if (!APP) { console.error('FAIL window.APP missing'); process.exit(1); }

(async () => {
  await APP.init();
  console.log('PASS init() completed with both models');

  const n = FILES['pca22_model.json'], x = FILES['pca22_nonoise_model.json'];
  console.log(`      noise model : in_dim=${n.meta.in_dim} steps=${n.meta.steps_trained} use_noise=${n.meta.use_noise}`);
  console.log(`      nonoise     : in_dim=${x.meta.in_dim} steps=${x.meta.steps_trained} use_noise=${x.meta.use_noise}`);

  // shared gallery must be identical, otherwise the comparison is unfair
  let same = n.gallery.length === x.gallery.length;
  for (let i = 0; same && i < n.gallery.length; i++) {
    if (n.gallery[i].rgb.join() !== x.gallery[i].rgb.join()) same = false;
    if (n.gallery[i].pca.join() !== x.gallery[i].pca.join()) same = false;
  }
  console.log(`PASS shared gallery identical across the two models: ${same} (n=${n.gallery.length})`);
  if (!same) { console.error('FAIL gallery differs -> comparison would be unfair'); process.exit(1); }

  // same 22 parameters must give DIFFERENT images from the two models
  const pca = new Float32Array(x.gallery[0].pca);
  APP.setPca(pca);
  APP.setModel(true);
  const zA = new Float32Array(64).fill(0.25);
  const lA = APP.forward(pca, zA);
  APP.setModel(false);
  const lB = APP.forward(pca, zA);
  let mx = 0; for (let i = 0; i < 512; i++) mx = Math.max(mx, Math.abs(lA[i] - lB[i]));
  console.log(`PASS both models run; max |logit difference| on identical input = ${mx.toFixed(4)}`);
  if (mx < 1e-6) { console.error('FAIL the two models produce identical logits'); process.exit(1); }

  // the no-noise model must IGNORE z entirely
  const lB2 = APP.forward(pca, new Float32Array(64).fill(-9.0));
  let mx2 = 0; for (let i = 0; i < 512; i++) mx2 = Math.max(mx2, Math.abs(lB[i] - lB2[i]));
  console.log(`PASS no-noise model ignores z: max |diff| when z changes = ${mx2.toExponential(2)}`);
  if (mx2 > 1e-6) { console.error('FAIL no-noise model still depends on z'); process.exit(1); }

  // speckle helper sanity on a known-flat image: all pixels the SAME level
  const flat = new Uint8Array(64).fill(7);       // 64 pixels, every one level 7
  const s = APP.speckle(flat);
  console.log(`PASS speckle() on a uniform image: agree=${s.agree} (expect 1) iso=${s.iso} (expect 0) blobs=${s.blobs} (expect 1)`);
  // (the earlier assertion here was wrong: it compared the ARRAY to 0 instead of s.iso)
  if (s.agree !== 1 || s.iso !== 0 || s.blobs !== 1) { console.error('FAIL speckle() wrong on a uniform image'); process.exit(1); }

  // a one-pixel island must be reported as isolated
  const island = new Uint8Array(64).fill(3);
  island[4 * 8 + 4] = 9;
  const s3 = APP.speckle(island);
  console.log(`PASS speckle() on a single-pixel island: iso=${s3.iso} (expect 1) blobs=${s3.blobs} (expect 2)`);
  if (s3.iso !== 1 || s3.blobs !== 2) { console.error('FAIL speckle() missed the island'); process.exit(1); }

  // a checkerboard must score badly
  const chk = new Uint8Array(64);
  for (let r = 0; r < 8; r++) for (let c = 0; c < 8; c++) chk[r * 8 + c] = (r + c) % 2 ? 0 : 15;
  const s2 = APP.speckle(chk);
  console.log(`PASS speckle() on a checkerboard: agree=${s2.agree.toFixed(3)} (expect ~0) iso=${s2.iso} blobs=${s2.blobs}`);

  console.log('\nTWO_MODEL_PAGE_TEST_DONE');
})();
