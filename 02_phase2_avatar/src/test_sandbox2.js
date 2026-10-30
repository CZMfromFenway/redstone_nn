// Verify the rewritten sandbox: one model at a time, single-face knob mode and batch mode.
const fs = require('fs'), path = require('path');
const WEB = path.join(__dirname, '..', 'webapp');
const html = fs.readFileSync(path.join(WEB, 'sandbox.html'), 'utf8');
const code = html.match(/<script>([\s\S]*?)<\/script>/)[1];

function mkEl(id) {
  const el = { id, children: [], style: {}, dataset: {}, className: '', _h: '', _t: '',
    value: '', width: 8, height: 8, type: '', min: '', max: '', step: '', oninput: null,
    appendChild(c) { this.children.push(c); return c; }, addEventListener() {},
    querySelectorAll(sel) { return []; }, querySelector() { return mkEl('q'); },
    classList: { add() {}, remove() {}, contains() { return false; },
                 toggle() { return false; } },
    getContext() { return { createImageData: (w, h) => ({ data: new Uint8ClampedArray(w*h*4) }),
      putImageData() {}, clearRect() {} }; },
    set innerHTML(v) { this._h = v; this.children = []; }, get innerHTML() { return this._h; },
    set textContent(v) { this._t = v; }, get textContent() { return this._t; } };
  return el;
}
const byId = {};
global.document = {
  createElement: () => mkEl('new'),
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
  await new Promise(r => setTimeout(r, 2200));
  const S = window.SANDBOX;
  if (!S) { console.error('page never initialised'); process.exit(1); }
  console.log('models    :', S.models.length, '/', fam.models.length);
  console.log('default   :', S.state.key, ' scheme=' + S.state.scheme,
              ' mode=' + S.state.mode);
  console.log('scorers   : accept=' + S.hasAccept + ' anime=' + S.hasAnime);

  // knob mode: moving a knob must change the face; zero must be stable
  S.state.knob = [0, 0, 0, 0];
  const f0 = S.levelsFor(new Float32Array([0,0,0,0]), '16');
  const f1 = S.levelsFor(new Float32Array([1.5,0,0,0]), '16');
  const f2 = S.levelsFor(new Float32Array([0,-1.5,0,0]), '16');
  console.log('zero face levels :', f0.join(','));
  console.log('PC0=+1.5 mean    :', (f1.reduce((a,b)=>a+b,0)/32).toFixed(2));
  console.log('PC1=-1.5 mean    :', (f2.reduce((a,b)=>a+b,0)/32).toFixed(2));

  // the scorers must respond to the knobs too, otherwise the page teaches nothing
  const s0 = S.scoreAcc(new Float32Array([0,0,0,0]));
  const s1 = S.scoreAcc(new Float32Array([0,-2.5,0,0]));
  console.log('accept score at 0        :', s0.toFixed(3));
  console.log('accept score at PC1=-2.5 :', s1.toFixed(3));

  // merge changes output, and only into the 11 greys
  const merged = S.levelsFor(new Float32Array([0.3,-0.7,1.1,-0.2]), 'mc11');
  const lutSet = new Set(mc11._lut.map(v => v.toFixed(2)));
  const mergedOK = merged.every(v => lutSet.has(v.toFixed(2)));

  // switching models must change the face for the same knobs
  const perModel = {};
  for (const k of S.models) { S.state.key = k; perModel[k] = S.levelsFor(new Float32Array([0.3,-0.7,1.1,-0.2]), '16').join(','); }
  S.state.key = 'w64l1';

  const checks = {
    'ten models': S.models.length === 10,
    'both scorers': S.hasAccept && S.hasAnime,
    'defaults to the single-layer model': S.state.key === 'w64l1',
    'starts in single-face mode': S.state.mode === 'single',
    'PC0 brightens': (f1.reduce((a,b)=>a+b,0)/32) > (f0.reduce((a,b)=>a+b,0)/32),
    'PC1 changes the face': f2.join(',') !== f0.join(','),
    'the accept scorer responds to knobs': Math.abs(s1 - s0) > 0.2,
    'merge emits only the 11 MC greys': mergedOK,
    'different models give different faces': new Set(Object.values(perModel)).size > 6
  };
  console.log();
  let ok = true;
  for (const [k, v] of Object.entries(checks)) {
    console.log('  ' + (v ? 'PASS' : 'FAIL') + '  ' + k);
    if (!v) ok = false;
  }
  if (!ok) { console.error('\nFAIL'); process.exit(1); }
  console.log('\nPASS the sandbox is ready.');
  console.log('SANDBOX2_READY');
})();
