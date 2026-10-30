// Verify the UI actually TELLS the truth in mix mode.
//
// The first version set the "blind mix" banner and then immediately overwrote it with the
// plain per-model line, so the page displayed "模型 w128" while secretly mixing three
// models. The function was right and the screen lied -- exactly the sort of thing that makes
// a fresh experiment look contaminated. These assertions pin the display down.
const fs = require('fs'), path = require('path');
const WEB = path.join(__dirname, '..', 'webapp');
const html = fs.readFileSync(path.join(WEB, 'label.html'), 'utf8');
let code = html.match(/<script>([\s\S]*?)<\/script>/)[1];

function mkEl(id) {
  const el = { id, children: [], style: {}, dataset: {}, _t: '', _h: '',
    appendChild(c) { this.children.push(c); return c; }, addEventListener() {},
    removeEventListener() {}, querySelectorAll() { return []; }, querySelector() { return null; },
    classList: { add() {}, remove() {}, contains() { return false; }, toggle() {} },
    getContext() { return { createImageData: (w, h) => ({ data: new Uint8ClampedArray(w * h * 4) }),
      putImageData() {}, clearRect() {}, fillRect() {}, drawImage() {} }; },
    set innerHTML(v) { this._h = v; this.children = []; }, get innerHTML() { return this._h; },
    set textContent(v) { this._t = v; }, get textContent() { return this._t; },
    width: 8, height: 8 };
  return el;
}
const byId = {};
global.document = {
  createElement: () => mkEl('new'),
  getElementById: id => (byId[id] ||= mkEl(id)),
  querySelectorAll: () => [], querySelector: () => mkEl('q'), addEventListener() {}
};
global.window = { addEventListener() {}, location: { search: '?mix=1' } };
global.location = global.window.location;

const FILES = ['pca22_dim4_model.json', 'pca22_dim8_model.json', 'pca22_dim22_model.json',
               'dimsweep_manifest.json', 'preference_reward.json'];
const F = {};
for (const f of FILES) {
  const p = path.join(WEB, f);
  if (fs.existsSync(p)) F[f] = JSON.parse(fs.readFileSync(p, 'utf8'));
}
const fakeStats = {
  total: 428, counts: [71, 132, 198, 27],
  byModel: { lam5: [10, 42, 86, 21], w128: [61, 90, 112, 6] }
};
let posted = [];
global.fetch = async (u, opt) => {
  const k = String(u).split('?')[0].split('/').pop();
  if (k === 'label' && opt && opt.method === 'POST') {
    posted.push(JSON.parse(opt.body));
    return { ok: true, status: 200, json: async () => ({ ok: true, total: 1 }) };
  }
  if (k === 'labelstats') return { ok: true, status: 200, json: async () => fakeStats };
  if (F[k]) return { ok: true, status: 200, json: async () => F[k] };
  return { ok: false, status: 404, json: async () => ({}) };
};
global.console.warn = () => {};
code = code.replace(/init\(\)\.catch\([\s\S]*?}\);\s*$/, '');
new Function(code)();

(async () => {
  await new Promise(r => setTimeout(r, 1000));
  const meta = byId['meta'] ? byId['meta']._h : '';
  const sampler = byId['sampler'] ? byId['sampler']._h : '';
  const stats = byId['stats'] ? byId['stats']._h : '';
  const mbtnsHidden = byId['mbtns'] && byId['mbtns'].style.display === 'none';

  console.log('meta    :', meta.replace(/\s+/g, ' ').slice(0, 90));
  console.log('sampler :', sampler.replace(/\s+/g, ' ').slice(0, 80));
  console.log('stats   :', stats.replace(/\s+/g, ' ').slice(0, 110));

  // one label so the progress panel refreshes with the per-model aggregation
  const L = global.window.LABEL;
  L.next(); await L.grade(2);
  await new Promise(r => setTimeout(r, 400));
  const stats2 = byId['stats'] ? byId['stats']._h : '';
  console.log('stats after 1 label:', stats2.replace(/\s+/g, ' ').slice(0, 120));

  const checks = {
    'banner says blind mix': /盲测混合/.test(meta),
    'banner does NOT name a single model': !/模型\s*<b>(w128|dim)/.test(meta),
    'sampler says plain random': /纯随机/.test(sampler),
    'sampler does not mention the reward model': !/奖励模型/.test(sampler),
    'model buttons hidden': !!mbtnsHidden,
    'progress shows only the ladder (0/1), not 428': !/>\s*428\s*</.test(stats2) &&
      /仅本次阶梯实验/.test(stats2),
    'progress labels the scope': /dim4|dim8|dim22/.test(stats2)
  };
  console.log();
  let ok = true;
  for (const [k, v] of Object.entries(checks)) {
    console.log('  ' + (v ? 'PASS' : 'FAIL') + '  ' + k);
    if (!v) ok = false;
  }
  if (!ok) { console.error('\nFAIL'); process.exit(1); }
  console.log('\nPASS the page now states what it is doing, and counts only the ladder.');
  console.log('MIX_UI_READY');
})();
