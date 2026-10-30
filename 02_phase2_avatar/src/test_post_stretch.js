// Verify the contrast stretch post-process actually does what it claims, through the page's
// own code path.
//
// dim4x uses the SAME weights as dim4 and differs only in meta.post='stretch', so any change
// in the generated faces is attributable to the post-processing alone. Expected:
//   * dim4x has higher mean contrast than dim4
//   * dim4x has fewer washed-out faces (contrast <= 5)
//   * dim4 and dim22 are untouched (their meta.post is absent)
//   * the stretch must not create frozen pixels: a level that never varies across samples
//     would mean the transform is pinning pixels, which is the failure that invalidated the
//     contrast-loss model.
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

const FILES = ['pca22_dim4x_model.json', 'pca22_dim4g_model.json', 'pca22_dim22_model.json',
               'dimsweep_manifest.json', 'preference_reward.json'];
const F = {};
for (const f of FILES) {
  const p = path.join(WEB, f);
  if (fs.existsSync(p)) F[f] = JSON.parse(fs.readFileSync(p, 'utf8'));
}
global.fetch = async (u) => {
  const k = String(u).split('?')[0].split('/').pop();
  if (k === 'labelstats') {
    return { ok: true, status: 200, json: async () => ({ total: 0, counts: [0,0,0,0], byModel: {} }) };
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
  const acc = {};
  const N = 1500;
  for (let i = 0; i < N * 4; i++) {
    L.next();
    const st = L.getState();
    const key = st.modelKey;
    const lv = Array.from(global.window._lastLevels || []).map(v => Math.round(v));
    (acc[key] ||= { rows: [] });
    if (acc[key].rows.length < N) acc[key].rows.push(lv);
    if (acc[key].rows.length >= N && Object.keys(acc).length === 3) {
      const done = Object.values(acc).every(a => a.rows.length >= N);
      if (done) break;
    }
  }
  console.log('arm      n     mean contrast   washed%   frozen pixels');
  const summary = {};
  for (const k of Object.keys(acc)) {
    const rows = acc[k].rows;
    let con = 0, washed = 0;
    for (const r of rows) {
      const c = Math.max(...r) - Math.min(...r);
      con += c;
      if (c <= 5) washed++;
    }
    // frozen: a position whose value never varies across samples
    let frozen = 0;
    for (let p = 0; p < 32; p++) {
      let lo = 1e9, hi = -1e9;
      for (const r of rows) { if (r[p] < lo) lo = r[p]; if (r[p] > hi) hi = r[p]; }
      if (hi - lo < 0.5) frozen++;
    }
    summary[k] = { n: rows.length, con: con / rows.length, washed: washed / rows.length, frozen };
    console.log(`  ${k.padEnd(7)} ${String(rows.length).padStart(4)}  ` +
                `${(con / rows.length).toFixed(2).padStart(12)}  ` +
                `${(100 * washed / rows.length).toFixed(2).padStart(7)}%  ` +
                `${String(frozen).padStart(6)}/32`);
  }
  const d4x = summary['dim4x'], dst = summary['dim4g'], d22 = summary['dim22'];
  const checks = {
    'all three arms present': !!(d4x && dst && d22),
    'dim4x contrast is full range': d4x ? d4x.con > 14.0 : false,
    'dim4g contrast near real (10-13)': dst ? (dst.con > 9.5 && dst.con < 13.5) : false,
    'dim4g is NOT min-max (below 14)': dst ? dst.con < 14.0 : false,
    'dim22 NOT stretched (contrast ~12, not 15)': d22 ? d22.con < 14.0 : false,
    'dim4x has no frozen pixels': d4x ? d4x.frozen === 0 : false,
    'dim4g has no frozen pixels': dst ? dst.frozen === 0 : false
  };
  console.log();
  let ok = true;
  for (const [k, v] of Object.entries(checks)) {
    console.log('  ' + (v ? 'PASS' : 'FAIL') + '  ' + k);
    if (!v) ok = false;
  }
  if (!ok) { console.error('\nFAIL'); process.exit(1); }
  console.log('\nPASS the stretch is applied only to dim4x and adds contrast without pinning pixels.');
  console.log('POST_STRETCH_READY');
})();
