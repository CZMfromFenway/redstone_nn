// Verify the quadratic scorer and box restriction inside the sandbox.
const fs = require('fs'), path = require('path');
const WEB = path.join(__dirname, '..', 'webapp');
const html = fs.readFileSync(path.join(WEB, 'sandbox.html'), 'utf8');
const code = html.match(/<script>([\s\S]*?)<\/script>/)[1];

function mkEl(id) {
  const el = { id, children: [], style: { cssText: '' }, dataset: {}, className: '',
    _h: '', _t: '', value: '', width: 8, height: 8, type: '', min: '', max: '', step: '',
    oninput: null, onchange: null,
    appendChild(c) { this.children.push(c); return c; }, addEventListener() {},
    querySelectorAll() { return []; }, querySelector() { return null; },
    classList: { add() {}, remove() {}, contains() { return false; }, toggle() { return false; } },
    getContext() { return { createImageData: (w, h) => ({ data: new Uint8ClampedArray(w*h*4) }),
      putImageData() {}, clearRect() {} }; },
    set innerHTML(v) { this._h = v; this.children = []; }, get innerHTML() { return this._h; },
    set textContent(v) { this._t = v; }, get textContent() { return this._t; } };
  return el;
}
const byId = {};
global.document = { createElement: (t) => mkEl(t),
  getElementById: id => (byId[id] ||= mkEl(id)),
  querySelector: () => mkEl('q'), querySelectorAll: () => [], addEventListener() {} };
global.window = { addEventListener() {}, location: { search: '' } };
global.location = global.window.location;
global.performance = { now: () => Date.now() };

const fam = JSON.parse(fs.readFileSync(path.join(WEB, 'grayscale_family.json'), 'utf8'));
const FILES = {};
for (const m of fam.models) {
  const p = path.join(WEB, m.file);
  if (fs.existsSync(p)) FILES[m.file] = JSON.parse(fs.readFileSync(p, 'utf8'));
}
const quad = JSON.parse(fs.readFileSync(path.join(WEB, 'scorer_quad.json'), 'utf8'));
const acc = JSON.parse(fs.readFileSync(path.join(WEB, 'scorer_accept.json'), 'utf8'));
const anime = JSON.parse(fs.readFileSync(path.join(WEB, 'scorer_anime.json'), 'utf8'));
const mc11 = JSON.parse(fs.readFileSync(path.join(WEB, 'merge_mc11.json'), 'utf8'));

global.fetch = async (u) => {
  const k = String(u).split('?')[0].split('/').pop();
  if (k === 'grayscale_family.json') return { ok: true, json: async () => fam };
  if (k === 'scorer_quad.json') return { ok: true, json: async () => quad };
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
  console.log('models :', S.models.length);
  console.log('quad   :', S.hasQuad);
  console.log('scorer :', S.state.scorer);

  // the scorer must rank the known-bad corner above the known-good one.
  // From the marginals: bad at low PC0, high PC1, low PC2; good at low PC1, high PC2.
  const badPt  = new Float32Array([-3.5, 0.5, -3.5, -0.5]);
  const goodPt = new Float32Array([-0.5, -3.5, 3.5, -1.5]);
  const bBad = S.badnessQuad(badPt), bGood = S.badnessQuad(goodPt);
  console.log('badness at a bad corner :', bBad.toFixed(4));
  console.log('badness at a good corner:', bGood.toFixed(4));

  // box restriction must be respected by drawSeed
  S.state.scorer = '';
  S.state.pc1Max = -1.5;
  S.state.pc2Min = -0.5;
  let viol = 0;
  for (let i = 0; i < 600; i++) {
    const c = S.drawSeed2();
    if (c[1] > -1.5 + 1e-6 || c[2] < -0.5 - 1e-6) viol++;
  }
  console.log('box violations out of 600:', viol);

  // with no box every draw must still be inside the clip range
  S.state.pc1Max = null; S.state.pc2Min = null;
  let outside = 0;
  for (let i = 0; i < 400; i++) {
    const c = S.drawSeed2();
    for (let k = 0; k < 4; k++) if (Math.abs(c[k]) > 3.5 + 1e-6) outside++;
  }
  console.log('out-of-range components:', outside);

  // selection must raise the mean quad score relative to raw sampling
  S.state.scorer = 'quad'; S.state.times = 4;
  const sel = [];
  for (let i = 0; i < 300; i++) sel.push(S.quadScore(S.drawSeed2()));
  S.state.scorer = '';
  const raw = [];
  for (let i = 0; i < 300; i++) raw.push(S.quadScore(S.drawSeed2()));
  const mean = a => a.reduce((x, y) => x + y, 0) / a.length;
  console.log('mean quad score, no filter :', mean(raw).toFixed(4));
  console.log('mean quad score, 4-to-1    :', mean(sel).toFixed(4));

  const checks = {
    'ten models': S.models.length === 10,
    'quadratic scorer loaded': S.hasQuad,
    'window bias matches the export': Math.abs(quad.coef.c - 0.33045) < 1e-4,
    'scorer treats the known-bad corner as worse': bBad > bGood,
    'box restriction is honoured (0 violations)': viol === 0,
    'draws stay inside the clip range': outside === 0,
    'selection raises the score': mean(sel) > mean(raw) + 0.05,
    'default scorer is the quadratic': quad.coef.linear.length === 4 &&
      Object.keys(quad.coef.cross).length === 6
  };
  console.log();
  let ok = true;
  for (const [k, v] of Object.entries(checks)) {
    console.log('  ' + (v ? 'PASS' : 'FAIL') + '  ' + k);
    if (!v) ok = false;
  }
  if (!ok) { console.error('\nFAIL'); process.exit(1); }
  console.log('\nPASS the quadratic scorer and the style box work in the sandbox.');
  console.log('QUAD_SANDBOX_READY');
})();
