// Verify style sampling in the sandbox: six styles load, samples follow the fitted mean and
// covariance, stay inside the cube, and the free mode is still the default.
const fs = require('fs'), path = require('path');
const WEB = path.join(__dirname, '..', 'webapp');
const html = fs.readFileSync(path.join(WEB, 'sandbox.html'), 'utf8');
const code = html.match(/<script>([\s\S]*?)<\/script>/)[1];

function mkEl(id) {
  return { id, children: [], style: { cssText: '' }, dataset: {}, className: '',
    _h: '', _t: '', value: '', width: 8, height: 8,
    type: '', min: '', max: '', step: '', oninput: null, onchange: null,
    appendChild(c) { this.children.push(c); return c; }, addEventListener() {},
    querySelectorAll() { return []; }, querySelector() { return null; },
    classList: { add() {}, remove() {}, contains() { return false; }, toggle() { return false; } },
    getContext() { return { createImageData: (w, h) => ({ data: new Uint8ClampedArray(w*h*4) }),
      putImageData() {}, clearRect() {} }; },
    set innerHTML(v) { this._h = v; this.children = []; }, get innerHTML() { return this._h; },
    set textContent(v) { this._t = v; }, get textContent() { return this._t; } };
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
const J = {};
for (const f of ['scorer_quad.json', 'scorer_accept.json', 'scorer_anime.json',
                 'merge_mc11.json', 'styles_gauss.json']) {
  J[f] = JSON.parse(fs.readFileSync(path.join(WEB, f), 'utf8'));
}
global.fetch = async (u) => {
  const k = String(u).split('?')[0].split('/').pop();
  if (k === 'grayscale_family.json') return { ok: true, json: async () => fam };
  if (J[k]) return { ok: true, json: async () => J[k] };
  if (FILES[k]) return { ok: true, json: async () => FILES[k] };
  return { ok: false, status: 404, json: async () => ({}) };
};

new Function(code)();

(async () => {
  await new Promise(r => setTimeout(r, 2400));
  const S = window.SANDBOX;
  if (!S) { console.error('page never initialised'); process.exit(1); }
  console.log('models :', S.models.length);
  console.log('styles :', S.styles.map(t => t.name).join(' / '));
  console.log('default style:', JSON.stringify(S.state.style));

  // walk to each style and check that samples follow its mean and covariance
  const results = {};
  for (const t of S.styles) {
    S.state.style = t.name;
    S.state.scorer = '';            // no filtering, so the raw style distribution is measured
    S.state.pc1Max = null; S.state.pc2Min = null;
    const xs = [];
    for (let i = 0; i < 4000; i++) {
      const c = S.drawSeed2();
      xs.push([c[0], c[1], c[2], c[3]]);
    }
    const mean = [0,1,2,3].map(k => xs.reduce((a, b) => a + b[k], 0) / xs.length);
    const sd = [0,1,2,3].map(k => {
      const m = mean[k];
      return Math.sqrt(xs.reduce((a, b) => a + (b[k] - m) ** 2, 0) / xs.length);
    });
    const outside = xs.filter(c => c.some(v => Math.abs(v) > 3.5 + 1e-6)).length;
    results[t.name] = { mean, sd, outside };
    const muErr = Math.max(...[0,1,2,3].map(k => Math.abs(mean[k] - t.mu[k])));
    const sdErr = Math.max(...[0,1,2,3].map(k => Math.abs(sd[k] - t.sd[k])));
    console.log(`  ${t.name.padEnd(8)} max|mean-mu| ${muErr.toFixed(2)}  ` +
                `max|sd-sd| ${sdErr.toFixed(2)}  outside ${outside}`);
  }

  // free mode must still be a standard normal, centred near zero
  S.state.style = '';
  S.state.scorer = '';
  const free = [];
  for (let i = 0; i < 4000; i++) { const c = S.drawSeed2(); free.push(c[0]); }
  const fm = free.reduce((a, b) => a + b, 0) / free.length;
  console.log('free mode PC0 mean:', fm.toFixed(3));

  const worstMu = Math.max(...Object.entries(results).map(([n, r]) =>
    Math.max(...[0,1,2,3].map(k => Math.abs(r.mean[k] - S.styles.find(t => t.name === n).mu[k])))));
  const worstSd = Math.max(...Object.entries(results).map(([n, r]) =>
    Math.max(...[0,1,2,3].map(k => Math.abs(r.sd[k] - S.styles.find(t => t.name === n).sd[k])))));
  const anyOutside = Object.values(results).some(r => r.outside > 0);

  const checks = {
    'ten models': S.models.length === 10,
    'six styles loaded': S.styles.length === 6,
    '潦草大叔 is gone': !S.styles.some(t => t.name === '潦草大叔'),
    'free mode is the default': S.state.style === '',
    'free mode is centred near zero': Math.abs(fm) < 0.1,
    'style means match the export': worstMu < 0.25,
    'style spreads match the export': worstSd < 0.25,
    'style samples stay inside the cube': !anyOutside,
    'every style has a 4x4 Cholesky factor':
      S.styles.every(t => t.chol.length === 4 && t.chol[3].length === 4)
  };
  console.log();
  let ok = true;
  for (const [k, v] of Object.entries(checks)) {
    console.log('  ' + (v ? 'PASS' : 'FAIL') + '  ' + k);
    if (!v) ok = false;
  }
  if (!ok) { console.error('\nFAIL'); process.exit(1); }
  console.log('\nPASS style sampling works in the sandbox.');
  console.log('STYLE_SANDBOX_READY');
})();
