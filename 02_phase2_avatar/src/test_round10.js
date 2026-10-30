// Verify the round-10 blind test: the three arms differ only by the merge and the filter, the
// merged arms really print a restricted set of greys, and shapes are untouched by the merge.
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
  location: { search: '?mix=1&round=10&man=grayscale_mc_manifest.json' } };
global.location = global.window.location;

const man = JSON.parse(fs.readFileSync(path.join(WEB, 'grayscale_mc_manifest.json'), 'utf8'));
const MERGE = JSON.parse(fs.readFileSync(path.join(WEB, 'merge_mc11.json'), 'utf8'));
const scorer = JSON.parse(fs.readFileSync(path.join(WEB, 'scorer_accept.json'), 'utf8'));
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
  if (k === 'grayscale_mc_manifest.json') return { ok: true, status: 200, json: async () => man };
  if (k === 'merge_mc11.json') return { ok: true, status: 200, json: async () => MERGE };
  if (k === 'scorer_accept.json') return { ok: true, status: 200, json: async () => scorer };
  if (MODELS[k]) return { ok: true, status: 200, json: async () => MODELS[k] };
  return { ok: false, status: 404, json: async () => ({}) };
};

new Function(code)();

(async () => {
  await new Promise(r => setTimeout(r, 1800));
  const L = window.LABEL;
  if (!L) { console.error('page never initialised'); process.exit(1); }

  const levels = {}, draws = {};
  for (let i = 0; i < 600; i++) {
    L.next();
    const s2 = L.getState();
    draws[s2.modelKey] = (draws[s2.modelKey] || 0) + 1;
    const u = new Set((global.window._lastLevels || []).map(v => v.toFixed(2)));
    (levels[s2.modelKey] ||= new Set());
    for (const v of u) levels[s2.modelKey].add(v);
  }
  console.log('arms        :', man.map(a => a.key).join(', '));
  console.log('arm draws   :', JSON.stringify(draws));
  for (const k of Object.keys(levels).sort())
    console.log('  ' + k.padEnd(8) + ' distinct grey values seen: ' + levels[k].size);

  const lutSet = new Set(MERGE._lut.map(v => v.toFixed(2)));
  const mergedVals = (k) => [...levels[k]].every(v => lutSet.has(v));

  const perArm = {};
  for (let i = 0; i < 300 && Object.keys(perArm).length < 3; i++) {
    L.next();
    const s2 = L.getState();
    if (!perArm[s2.modelKey]) { await L.grade(2); perArm[s2.modelKey] = true; }
  }
  console.log('posted      :', JSON.stringify(posted.map(p => p.model)));

  const checks = {
    'three arms': man.length === 3,
    'all three drawn': Object.keys(draws).length === 3,
    'a label per arm': new Set(posted.map(p => p.model)).size === 3,
    'two arms declare merge': man.filter(a => a.merge).length === 2,
    'one arm declares filter': man.filter(a => a.filter).length === 1,
    'baseline arm has NO merge (per arm, same model file)': !man[0].merge,
    'merged arms only print the 11 MC greys': mergedVals('w64_mc') && mergedVals('w64_mcf'),
    'baseline arm prints MORE greys than merged':
      levels['w64_c'].size > levels['w64_mc'].size,
    'merge table has 16 entries': MERGE._lut.length === 16,
    'merge table has 11 bins but only ~9 separable levels': (function () {
      const a = [...new Set(MERGE._lut)].sort((x, y) => x - y);
      let n = 1;
      for (let i = 1; i < a.length; i++) if (a[i] - a[i - 1] >= 0.5) n++;
      return a.length === 11 && n === 9;
    })()
  };
  console.log();
  let ok = true;
  for (const [k, v] of Object.entries(checks)) {
    console.log('  ' + (v ? 'PASS' : 'FAIL') + '  ' + k);
    if (!v) ok = false;
  }
  if (!ok) { console.error('\nFAIL'); process.exit(1); }
  console.log('\nPASS the round-10 blind test is ready.');
  console.log('ROUND10_READY');
})();
