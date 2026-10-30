// Guard against the decode bug that produced all-black faces with two bright corners.
//
// ROOT CAUSE OF THAT BUG
// label.html chooses the decoder from meta.output_mode. The ladder exports only carried
// meta.mode, so OUT_MODE fell back to 'logits_gray' and decode() read raw[p*16+c] up to
// index 511 on a 32-element array. Everything past the first two positions therefore became
// level 0 -- black -- while the first two picked an arbitrary argmax. The models were fine;
// the decoding was wrong.
//
// WHAT THIS PINS DOWN
// For every arm, the page must produce faces that actually use the grey range: several
// distinct levels, and not almost entirely black. A regression to logits_gray on a 32-wide
// output trips this immediately.
const fs = require('fs'), path = require('path');
const WEB = path.join(__dirname, '..', 'webapp');
const html = fs.readFileSync(path.join(WEB, 'label.html'), 'utf8');
let code = html.match(/<script>([\s\S]*?)<\/script>/)[1];

function mkEl(id) {
  const el = { id, children: [], style: {}, dataset: {}, _t: '', _h: '',
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
global.document = {
  createElement: () => mkEl('new'),
  getElementById: id => (byId[id] ||= mkEl(id)),
  querySelectorAll: () => [], querySelector: () => mkEl('q'), addEventListener() {}
};
global.window = { addEventListener() {}, location: { search: '?mix=1' } };
global.location = global.window.location;

const FILES = ['pca22_dim4_model.json', 'pca22_dim8_model.json', 'pca22_dim22_model.json',
               'dimsweep_manifest.json', 'preference_reward.json'];
const F = {};
for (const f of FILES) {
  const p = path.join(WEB, f);
  if (fs.existsSync(p)) F[f] = JSON.parse(fs.readFileSync(p, 'utf8'));
}
global.fetch = async (u, opt) => {
  const k = String(u).split('?')[0].split('/').pop();
  if (k === 'label') return { ok: true, status: 200, json: async () => ({ ok: true, total: 1 }) };
  if (k === 'labelstats') {
    return { ok: true, status: 200, json: async () => ({ total: 0, counts: [0, 0, 0, 0], byModel: {} }) };
  }
  if (F[k]) return { ok: true, status: 200, json: async () => F[k] };
  return { ok: false, status: 404, json: async () => ({}) };
};
global.console.warn = () => {};
code = code.replace(/init\(\)\.catch\([\s\S]*?}\)\.;\s*$/, '');
new Function(code)();

(async () => {
  await new Promise(r => setTimeout(r, 1000));
  const L = global.window.LABEL;
  const per = {};
  for (let i = 0; i < 60; i++) {
    L.next();
    const st = L.getState();
    const key = st.modelKey;
    const lv = Array.from(global.window._lastLevels || []);
    if (!lv.length) { console.error('no levels recorded'); process.exit(1); }
    const zeros = lv.filter(v => v < 0.5).length;
    const distinct = new Set(lv.map(v => Math.round(v))).size;
    (per[key] ||= { n: 0, blackFrac: 0, distinct: [], allZero: 0 });
    per[key].n++;
    per[key].blackFrac += zeros / lv.length;
    per[key].distinct.push(distinct);
    if (distinct <= 1) per[key].allZero++;
  }
  console.log('arm      n   mean black%  mean distinct levels  single-level faces');
  let ok = true;
  for (const k of Object.keys(per)) {
    const s = per[k];
    const black = 100 * s.blackFrac / s.n;
    const dis = s.distinct.reduce((a, b) => a + b, 0) / s.n;
    console.log(`  ${k.padEnd(7)} ${String(s.n).padStart(3)}  ${black.toFixed(1).padStart(10)}%  ` +
                `${dis.toFixed(1).padStart(19)}  ${s.allZero}`);
    if (black > 60) { console.log(`    FAIL ${k}: ${black.toFixed(0)}% of pixels are black`); ok = false; }
    if (dis < 3) { console.log(`    FAIL ${k}: only ${dis.toFixed(1)} distinct levels on average`); ok = false; }
    if (s.allZero > 0) { console.log(`    FAIL ${k}: ${s.allZero} face(s) used a single level`); ok = false; }
  }
  if (!ok) { console.error('\nFAIL decode is not producing real faces'); process.exit(1); }
  console.log('\nPASS every arm decodes to faces that use the grey range.');
  console.log('DECODE_OK');
})();
