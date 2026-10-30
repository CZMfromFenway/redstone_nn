// Verify the 3-arm blind round: all three models load, each gets the same stretch, arms are
// assigned randomly, and the recorded model key identifies which arm produced the face.
const fs = require('fs'), path = require('path');
const WEB = path.join(__dirname, '..', 'webapp');
const html = fs.readFileSync(path.join(WEB, 'label.html'), 'utf8');
let code = html.match(/<script>([\s\S]*?)<\/script>/)[1];

function mkEl(id) {
  const el = { id, children: [], style: { cssText: '' }, dataset: {}, _t: '', _h: '',
    appendChild(c) { this.children.push(c); return c; }, addEventListener() {},
    removeEventListener() {}, querySelectorAll() { return []; }, querySelector() { return null; },
    classList: { add() {}, remove() {}, contains() { return false; }, toggle() {} },
    getContext() { return { createImageData: (w, h) => ({ data: new Uint8ClampedArray(w*h*4) }),
      putImageData() {}, clearRect() {}, fillRect() {}, drawImage() {} }; },
    set innerHTML(v) { this._h = v; this.children = []; }, get innerHTML() { return this._h; },
    set textContent(v) { this._t = v; }, get textContent() { return this._t; },
    width: 8, height: 8 };
  return el;
}
const byId = {};
global.document = { createElement: () => mkEl('new'),
  getElementById: id => (byId[id] ||= mkEl(id)),
  querySelector: () => mkEl('q'), querySelectorAll: () => [], addEventListener() {} };
global.window = { addEventListener() {}, location: { search: '?mix=1&round=7' } };
global.location = global.window.location;

const man = JSON.parse(fs.readFileSync(path.join(WEB, 'dimsweep_manifest.json'), 'utf8'));
const MODELS = {};
for (const a of man) MODELS[a.file] = JSON.parse(fs.readFileSync(path.join(WEB, a.file), 'utf8'));
let posted = [];
global.fetch = async (u, opt) => {
  const k = String(u).split('?')[0].split('/').pop();
  if (k === 'label' && opt && opt.method === 'POST') {
    posted.push(JSON.parse(opt.body));
    return { ok: true, status: 200, json: async () => ({ ok: true, total: posted.length }) };
  }
  if (k.startsWith('labelstats'))
    return { ok: true, status: 200, json: async () => ({ total: 0, counts: [0,0,0,0,0], byModel: {} }) };
  if (k === 'dimsweep_manifest.json') return { ok: true, status: 200, json: async () => man };
  if (MODELS[k]) return { ok: true, status: 200, json: async () => MODELS[k] };
  return { ok: false, status: 404, json: async () => ({}) };
};

code = code.replace(/init\(\)\.catch\([\s\S]*?}\)\.;\s*$/, '');
new Function(code)();

(async () => {
  await new Promise(r => setTimeout(r, 1200));
  const L = window.LABEL;
  const st = L.getState();
  console.log('arms in manifest :', man.map(a => a.key + '/' + a.multiplies).join(', '));
  console.log('default model    :', st.modelKey);

  // sample many faces and record which arm each came from plus its pixel statistics
  const seen = {};
  for (let i = 0; i < 90; i++) {
    L.next();
    const s2 = L.getState();
    seen[s2.modelKey] = (seen[s2.modelKey] || 0) + 1;
  }
  console.log('arm draw counts  :', JSON.stringify(seen));

  // submit one label per arm so the recorded key can be checked
  const perArm = {};
  for (let i = 0; i < 120 && Object.keys(perArm).length < 3; i++) {
    L.next();
    const s2 = L.getState();
    if (!perArm[s2.modelKey]) {
      await L.grade(4);
      perArm[s2.modelKey] = true;
    }
  }
  const recs = posted.map(p => p.model);
  console.log('posted models    :', JSON.stringify(recs));

  // the three arms must produce DIFFERENT pixels for the same seed, otherwise the manifest is
  // silently pointing at the same weights
  const sig = {};
  for (const key of ['base', 'w96', 'w64']) {
    const f = man.find(a => a.key === key).file;
    const j = MODELS[f];
    const W0 = j.gen.layers[0].W, b0 = j.gen.layers[0].b;
    const z = [0.3, -0.7, 1.1, -0.2];
    const h = [];
    for (let o = 0; o < 4; o++) {
      let s = b0[o];
      for (let i = 0; i < 4; i++) s += W0[o][i] * z[i];
      h.push(s);
    }
    sig[key] = h.map(v => v.toFixed(4)).join(',');
  }
  console.log('first-layer output per arm (must all differ):');
  for (const [k, v] of Object.entries(sig)) console.log('  ' + k.padEnd(5) + ' ' + v);

  const checks = {
    'three arms in manifest': man.length === 3,
    'all three arms get drawn': Object.keys(seen).length === 3,
    'a label can be recorded per arm': new Set(recs).size === 3,
    'post is stretch on every arm': man.every(a => a.post === 'stretch'),
    'output_mode is continuous32 on every arm':
      man.every(a => MODELS[a.file].meta.output_mode === 'continuous32'),
    'the three arms have different weights': new Set(Object.values(sig)).size === 3,
    'multiply counts differ (not the same file)':
      new Set(man.map(a => a.multiplies)).size === 3
  };
  console.log();
  let ok = true;
  for (const [k, v] of Object.entries(checks)) {
    console.log('  ' + (v ? 'PASS' : 'FAIL') + '  ' + k);
    if (!v) ok = false;
  }
  if (!ok) { console.error('\nFAIL'); process.exit(1); }
  console.log('\nPASS the 3-arm blind round is ready.');
  console.log('BLIND3_READY');
})();
