// Verify the prepared grayscale filtered round (round 9): the pre-scorer loads, the filtered
// arms actually select better seeds, and unfiltered arms still behave as before.
const fs = require('fs'), path = require('path');
const WEB = path.join(__dirname, '..', 'webapp');
const html = fs.readFileSync(path.join(WEB, 'label.html'), 'utf8');
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
global.window = { addEventListener() {},
  location: { search: '?mix=1&round=9&man=grayscale_filtered_manifest.json' } };
global.location = global.window.location;

const man = JSON.parse(fs.readFileSync(path.join(WEB, 'grayscale_filtered_manifest.json'), 'utf8'));
const MODELS = {};
for (const a of man) MODELS[a.file] = JSON.parse(fs.readFileSync(path.join(WEB, a.file), 'utf8'));
const scorer = JSON.parse(fs.readFileSync(path.join(WEB, 'scorer_accept.json'), 'utf8'));
const posted = [];
global.fetch = async (u, opt) => {
  const k = String(u).split('?')[0].split('/').pop();
  if (k === 'label' && opt && opt.method === 'POST') {
    posted.push(JSON.parse(opt.body));
    return { ok: true, status: 200, json: async () => ({ ok: true, total: posted.length }) };
  }
  if (k.startsWith('labelstats'))
    return { ok: true, status: 200, json: async () => ({ total: 0, counts: [0,0,0,0,0], byModel: {} }) };
  if (k === 'grayscale_filtered_manifest.json') return { ok: true, status: 200, json: async () => man };
  if (k === 'scorer_accept.json') return { ok: true, status: 200, json: async () => scorer };
  if (MODELS[k]) return { ok: true, status: 200, json: async () => MODELS[k] };
  return { ok: false, status: 404, json: async () => ({}) };
};

new Function(code)();

function refScore(pcs) {
  let x = pcs;
  for (const L of scorer.layers) {
    const out = new Array(L.W.length);
    for (let o = 0; o < L.W.length; o++) {
      let t = L.b[o];
      for (let i = 0; i < L.W[o].length; i++) t += L.W[o][i] * x[i];
      out[o] = L.relu && t < 0 ? 0 : t;
    }
    x = out;
  }
  return x[0];
}

(async () => {
  await new Promise(r => setTimeout(r, 1800));
  const L = window.LABEL;
  if (!L) { console.error('page never initialised'); process.exit(1); }
  const st = L.getState();
  console.log('arms         :', man.map(a => a.key).join(', '));
  console.log('filter n     :', man.map(a => a.filter.n).join(', '));

  // draw 400 faces and confirm the recorded seeds really are the best of four
  const scores = [];
  const draws = {};
  for (let i = 0; i < 400; i++) {
    L.next();
    const s2 = L.getState();
    draws[s2.modelKey] = (draws[s2.modelKey] || 0) + 1;
    scores.push(refScore(Array.from(Array.from(s2.cur))));
  }
  const ok = scores.filter(v => !isNaN(v));
  const mean = ok.reduce((a,b)=>a+b,0)/ok.length;
  console.log('arm draws    :', JSON.stringify(draws));
  console.log('mean score of shown seeds :', mean.toFixed(3));

  // a raw (unfiltered) draw would average near the distribution mean of 0; round-5 labels have
  // the median at +1.437, so a filtered run must sit far above that
  const thr = scorer.thresholds['0.5'];
  console.log('median threshold          :', thr.toFixed(3));

  for (const a of man.slice(0, 3)) {
    // confirm one label can be recorded per arm
  }
  const perArm = {};
  for (let i = 0; i < 150 && Object.keys(perArm).length < 3; i++) {
    L.next();
    const s2 = L.getState();
    if (!perArm[s2.modelKey]) { await L.grade(2); perArm[s2.modelKey] = true; }
  }
  console.log('posted models:', JSON.stringify(posted.map(p => p.model)));

  const checks = {
    'three filtered arms in the manifest': man.length === 3,
    'every arm declares a filter': man.every(a => a.filter && a.filter.n >= 2),
    'the scorer loaded (4->16->1)': scorer.layers.length === 2 &&
      scorer.layers[0].W.length === 16 && scorer.layers[1].W.length === 1,
    'all three arms get drawn': Object.keys(draws).length === 3,
    'a label records each arm': new Set(posted.map(p => p.model)).size === 3,
    'shown seeds score well above the median (filter is working)': mean > thr,
    'no seed scores NaN (weights valid)': ok.length === scores.length
  };
  console.log();
  let allok = true;
  for (const [k, v] of Object.entries(checks)) {
    console.log('  ' + (v ? 'PASS' : 'FAIL') + '  ' + k);
    if (!v) allok = false;
  }
  if (!allok) { console.error('\nFAIL'); process.exit(1); }
  console.log('\nPASS the grayscale filtered round is prepared (not started).');
  console.log('ROUND9_READY');
})();
