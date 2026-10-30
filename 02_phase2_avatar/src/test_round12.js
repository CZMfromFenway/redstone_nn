// Verify the C1/C2 blind round: three arms, same palette, different weights.
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
  search: '?man=colour_c1c2_manifest.json&round=12' } };
global.location = global.window.location;

const man = JSON.parse(fs.readFileSync(path.join(WEB, 'colour_c1c2_manifest.json'), 'utf8'));
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
  if (k === 'colour_c1c2_manifest.json') return { ok: true, json: async () => man };
  if (FILES[k]) return { ok: true, json: async () => FILES[k] };
  return { ok: false, status: 404, json: async () => ({}) };
};

new Function(code)();

(async () => {
  await new Promise(r => setTimeout(r, 1500));
  const L = window.LABEL;
  if (!L) { console.error('page never initialised'); process.exit(1); }
  console.log('arms   :', L.getState().arms.join(', '));

  const drawn = {};
  for (let i = 0; i < 400; i++) {
    L.next();
    const s = L.getState();
    drawn[s.model] = (drawn[s.model] || 0) + 1;
  }
  console.log('drawn  :', JSON.stringify(drawn));
  for (let i = 0; i < 300 && new Set(posted.map(p => p.model)).size < 3; i++) {
    L.next(); const s = L.getState();
    if (!posted.some(p => p.model === s.model)) await L.grade(2);
  }
  console.log('posted :', JSON.stringify([...new Set(posted.map(p => p.model))]));

  const palOf = {};
  for (const a of man) palOf[a.key] = JSON.stringify(FILES[a.file].palette);
  const bodyOf = {};
  for (const a of man) bodyOf[a.key] = JSON.stringify(FILES[a.file].body);

  const checks = {
    'three arms': man.length === 3,
    'all three drawn': Object.keys(drawn).length === 3,
    'a label per arm': new Set(posted.map(p => p.model)).size === 3,
    'all three share the same palette': new Set(Object.values(palOf)).size === 1,
    'all three have different weights': new Set(Object.values(bodyOf)).size === 3,
    'all are K=8': man.every(a => FILES[a.file].palette.length === 8)
  };
  console.log();
  let ok = true;
  for (const [k, v] of Object.entries(checks)) {
    console.log('  ' + (v ? 'PASS' : 'FAIL') + '  ' + k);
    if (!v) ok = false;
  }
  if (!ok) { console.error('\nFAIL'); process.exit(1); }
  console.log('\nPASS the C1/C2 round is ready.');
  console.log('ROUND12_READY');
})();
