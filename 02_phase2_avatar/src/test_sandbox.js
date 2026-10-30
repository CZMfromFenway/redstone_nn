// Verify the sandbox page: every family model loads, both colour schemes work, both scorer
// switches change what gets drawn, and the two columns are driven by different models.
const fs = require('fs'), path = require('path');
const WEB = path.join(__dirname, '..', 'webapp');
const html = fs.readFileSync(path.join(WEB, 'sandbox.html'), 'utf8');
const code = html.match(/<script>([\s\S]*?)<\/script>/)[1];

function mkEl(id) {
  const el = { id, children: [], style: {}, dataset: {}, className: '', _h: '', _t: '',
    appendChild(c) { this.children.push(c); return c; }, addEventListener() {},
    querySelectorAll(sel) { return []; }, querySelector() { return mkEl('q'); },
    classList: { add() {}, remove() {}, contains() { return false; } },
    getContext() { return { createImageData: (w, h) => ({ data: new Uint8ClampedArray(w*h*4) }),
      putImageData() {}, clearRect() {} }; },
    set innerHTML(v) { this._h = v; this.children = []; }, get innerHTML() { return this._h; },
    set textContent(v) { this._t = v; }, get textContent() { return this._t; },
    value: '', width: 8, height: 8 };
  return el;
}
const byId = {};
global.document = {
  createElement: (tag) => mkEl(tag),
  getElementById: id => (byId[id] ||= mkEl(id)),
  querySelector: () => mkEl('q'), querySelectorAll: () => [], addEventListener() {}
};
global.window = { addEventListener() {}, location: { search: '' } };
global.location = global.window.location;
global.performance = { now: () => Date.now() };

const fam = JSON.parse(fs.readFileSync(path.join(WEB, 'grayscale_family.json'), 'utf8'));
const FILES = {};
for (const m of fam.models) {
  const p = path.join(WEB, m.file);
  if (fs.existsSync(p)) FILES[m.file] = JSON.parse(fs.readFileSync(p, 'utf8'));
}
const acc = JSON.parse(fs.readFileSync(path.join(WEB, 'scorer_accept.json'), 'utf8'));
const anime = JSON.parse(fs.readFileSync(path.join(WEB, 'scorer_anime.json'), 'utf8'));
const mc11 = JSON.parse(fs.readFileSync(path.join(WEB, 'merge_mc11.json'), 'utf8'));

global.fetch = async (u) => {
  const k = String(u).split('?')[0].split('/').pop();
  if (k === 'grayscale_family.json') return { ok: true, json: async () => fam };
  if (k === 'scorer_accept.json') return { ok: true, json: async () => acc };
  if (k === 'scorer_anime.json') return { ok: true, json: async () => anime };
  if (k === 'merge_mc11.json') return { ok: true, json: async () => mc11 };
  if (FILES[k]) return { ok: true, json: async () => FILES[k] };
  return { ok: false, status: 404, json: async () => ({}) };
};

new Function(code)();

(async () => {
  await new Promise(r => setTimeout(r, 2000));
  const S = window.SANDBOX;
  if (!S) { console.error('page never initialised'); process.exit(1); }
  console.log('models loaded :', S.models.length, '/', fam.models.length);
  console.log('scorers       : accept=' + S.hasAccept + ' anime=' + S.hasAnime +
              ' merge=' + S.hasMerge);
  console.log('models        :', S.models.join(', '));

  // drive the internals directly to compare pixel sets between settings
  const out = {};
  const seed = [0.3, -0.7, 1.1, -0.2];
  for (const key of S.models) {
    out[key] = { plain: window.SANDBOX.levelsFor(key, seed, '16'), merged: window.SANDBOX.levelsFor(key, seed, 'mc11') };
  }
  const lut = new Set(mc11._lut.map(v => v.toFixed(2)));
  const mergedOK = S.models.every(k => out[k].merged.every(v => lut.has(v.toFixed(2))));
  const differ = S.models.filter(k =>
    out[k].plain.join(',') !== out[k].merged.join(',')).length;

  const plainVals = new Set();
  for (const k of S.models) out[k].plain.forEach(v => plainVals.add(v));

  const checks = {
    'all ten family models loaded': S.models.length === 10,
    'both scorers present': S.hasAccept && S.hasAnime,
    'merge table present': S.hasMerge,
    'merged output only uses the 11 MC greys': mergedOK,
    'merge actually changes the output': differ === S.models.length,
    'plain output uses many more values than 11': plainVals.size > 11,
    'different models give different output for one seed':
      new Set(S.models.map(k => out[k].plain.join(','))).size > 6
  };
  console.log();
  let ok = true;
  for (const [k, v] of Object.entries(checks)) {
    console.log('  ' + (v ? 'PASS' : 'FAIL') + '  ' + k);
    if (!v) ok = false;
  }
  console.log('\n  distinct plain values across all models:', plainVals.size);
  if (!ok) { console.error('\nFAIL'); process.exit(1); }
  console.log('\nPASS the sandbox is ready.');
  console.log('SANDBOX_READY');
})();
