// Verify the round-11 colour round: three K=8 arms that differ only in the palette.
const fs = require('fs'), path = require('path');
const WEB = path.join(__dirname, '..', 'webapp');
const html = fs.readFileSync(path.join(WEB, 'label_colour.html'), 'utf8');
const code = html.match(/<script>([\s\S]*?)<\/script>/)[1];

function mkEl(id) {
  const el = { id, children: [], style: { cssText: '' }, dataset: {}, _h: '', _t: '',
    appendChild(c) { this.children.push(c); return c; }, addEventListener() {},
    querySelectorAll() { return []; }, querySelector() { return null; },
    classList: { add() {}, remove() {}, contains() { return false; } },
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
global.window = { addEventListener() {}, location: {
  search: '?man=colour_k8_manifest.json&round=11' } };
global.location = global.window.location;
global.URLSearchParams = URLSearchParams;

const man = JSON.parse(fs.readFileSync(path.join(WEB, 'colour_k8_manifest.json'), 'utf8'));
const FILES = {};
for (const a of man) FILES[a.file] = JSON.parse(fs.readFileSync(path.join(WEB, a.file), 'utf8'));
const posted = [];
global.fetch = async (u, opt) => {
  const k = String(u).split('?')[0].split('/').pop();
  if (k === 'label' && opt && opt.method === 'POST') {
    posted.push(JSON.parse(opt.body));
    return { ok: true, status: 200, json: async () => ({ ok: true, total: posted.length }) };
  }
  if (k.startsWith('labelstats'))
    return { ok: true, json: async () => ({ total: 0, counts: [], byModel: {} }) };
  if (k === 'colour_k8_manifest.json') return { ok: true, json: async () => man };
  if (FILES[k]) return { ok: true, json: async () => FILES[k] };
  return { ok: false, status: 404, json: async () => ({}) };
};

new Function(code)();

(async () => {
  await new Promise(r => setTimeout(r, 1500));
  const L = window.LABEL;
  if (!L) { console.error('page never initialised'); process.exit(1); }
  const s0 = L.getState();
  console.log('arms      :', s0.arms.join(', '));
  console.log('kinds     :', s0.kinds.join(', '));

  const drawn = {};
  const paletteOf = {};
  for (let i = 0; i < 400; i++) {
    L.next();
    const s = L.getState();
    drawn[s.model] = (drawn[s.model] || 0) + 1;
    paletteOf[s.model] = s.cur ? true : true;
  }
  console.log('drawn     :', JSON.stringify(drawn));

  for (let i = 0; i < 300 && Object.keys(posted.reduce((a, p) => (a[p.model] = 1, a), {})).length < 3; i++) {
    L.next();
    const s = L.getState();
    if (!posted.some(p => p.model === s.model)) await L.grade(2);
  }
  console.log('posted    :', JSON.stringify(posted.map(p => p.model)));

  // the two snapped arms must share weights but have different palettes
  const free = FILES['colour_pal8_model.json'], snap = FILES['colour_k8snap_model.json'],
        mc = FILES['colour_k8mc_model.json'];
  const sameWeights = JSON.stringify(free.body) === JSON.stringify(snap.body) &&
                      JSON.stringify(free.head) === JSON.stringify(snap.head);
  const diffPal = JSON.stringify(free.palette) !== JSON.stringify(snap.palette);
  const allDiff = new Set([JSON.stringify(free.palette), JSON.stringify(snap.palette),
                           JSON.stringify(mc.palette)]).size === 3;

  const checks = {
    'three arms in the manifest': man.length === 3,
    'all three drawn': Object.keys(drawn).length === 3,
    'a label per arm': new Set(posted.map(p => p.model)).size === 3,
    'k8free and k8snap share weights': sameWeights,
    'k8free and k8snap differ in palette': diffPal,
    'all three palettes are distinct': allDiff,
    'all three are K=8': [free, snap, mc].every(j => j.palette.length === 8),
    'k8mc is a separately trained model': JSON.stringify(free.body) !== JSON.stringify(mc.body)
  };
  console.log();
  let ok = true;
  for (const [k, v] of Object.entries(checks)) {
    console.log('  ' + (v ? 'PASS' : 'FAIL') + '  ' + k);
    if (!v) ok = false;
  }
  if (!ok) { console.error('\nFAIL'); process.exit(1); }
  console.log('\nPASS the round-11 colour round is ready.');
  console.log('ROUND11_READY');
})();
