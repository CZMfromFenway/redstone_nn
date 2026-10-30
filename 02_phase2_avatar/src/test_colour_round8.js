// Verify the colour blind round: three palette arms load, each produces discrete colours,
// all three get drawn, and each label records which arm produced the face.
const fs = require('fs'), path = require('path');
const WEB = path.join(__dirname, '..', 'webapp');
const html = fs.readFileSync(path.join(WEB, 'label_colour.html'), 'utf8');
const code = html.match(/<script>([\s\S]*?)<\/script>/)[1];

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
global.window = { addEventListener() {}, location: { search: '?round=8' } };
global.location = global.window.location;
global.Math.random = (() => {
  let s = 987654321;
  return () => { s = (s * 1103515245 + 12345) & 0x7fffffff; return s / 0x7fffffff; };
})();

const man = JSON.parse(fs.readFileSync(path.join(WEB, 'colour_manifest.json'), 'utf8'));
const MODELS = {};
for (const a of man) MODELS[a.file] = JSON.parse(fs.readFileSync(path.join(WEB, a.file), 'utf8'));
const posted = [];
global.fetch = async (u, opt) => {
  const k = String(u).split('?')[0].split('/').pop();
  if (k === 'label' && opt && opt.method === 'POST') {
    posted.push(JSON.parse(opt.body));
    return { ok: true, status: 200, json: async () => ({ ok: true, total: posted.length }) };
  }
  if (k.startsWith('labelstats'))
    return { ok: true, status: 200, json: async () => ({ total: 0, counts: [0,0,0,0,0], byModel: {} }) };
  if (k === 'colour_manifest.json') return { ok: true, status: 200, json: async () => man };
  if (MODELS[k]) return { ok: true, status: 200, json: async () => MODELS[k] };
  return { ok: false, status: 404, json: async () => ({}) };
};

new Function(code)();

(async () => {
  await new Promise(r => setTimeout(r, 1500));
  const L = window.LABEL;
  if (!L) { console.error('page never initialised'); process.exit(1); }
  const st = L.getState();
  console.log('arms         :', st.arms.join(', '));
  console.log('kinds        :', st.kinds.join(', '));
  console.log('input dim    :', st.kin);
  console.log('face values  :', st.cur.length);

  const counts = {}, draws = {};
  for (let i = 0; i < 120; i++) {
    L.next();
    const s2 = L.getState();
    draws[s2.model] = (draws[s2.model] || 0) + 1;
    const uniq = new Set();
    for (let p = 0; p < 32; p++)
      uniq.add(s2.cur[p*3] + ',' + s2.cur[p*3+1] + ',' + s2.cur[p*3+2]);
    counts[s2.model] = Math.max(counts[s2.model] || 0, uniq.size);
  }
  console.log('arm draws    :', JSON.stringify(draws));
  for (const k of Object.keys(counts).sort())
    console.log('  ' + k.padEnd(6) + ' max distinct colours in one face: ' + counts[k]);

  const perArm = {};
  for (let i = 0; i < 200 && Object.keys(perArm).length < 3; i++) {
    L.next();
    const s2 = L.getState();
    if (!perArm[s2.model]) { await L.grade(4); perArm[s2.model] = true; }
  }
  console.log('posted models:', JSON.stringify(posted.map(p => p.model)));

  const K = { pal8: 8, pal16: 16, pal32: 32 };
  const checks = {
    'three arms in the manifest': man.length === 3,
    'all arms are palette type': st.kinds.every(k => k === 'palette'),
    'palette sizes are 8/16/32': man.every(a => MODELS[a.file].palette.length === K[a.key]),
    'faces are 96 values': st.cur.length === 96,
    'colours are 0..15': st.cur.every(v => v >= 0 && v <= 15),
    'all three arms get drawn': Object.keys(draws).length === 3,
    'a label records each arm': new Set(posted.map(p => p.model)).size === 3,
    'palette output stays discrete (<= its K)':
      Object.keys(counts).every(k => counts[k] <= K[k]),
    'the arms use different palettes':
      JSON.stringify(MODELS['colour_pal8_model.json'].palette) !==
      JSON.stringify(MODELS['colour_pal32_model.json'].palette)
  };
  console.log();
  let ok = true;
  for (const [k, v] of Object.entries(checks)) {
    console.log('  ' + (v ? 'PASS' : 'FAIL') + '  ' + k);
    if (!v) ok = false;
  }
  if (!ok) { console.error('\nFAIL'); process.exit(1); }
  console.log('\nPASS the colour blind round is ready.');
  console.log('COLOUR_ROUND8_READY');
})();
