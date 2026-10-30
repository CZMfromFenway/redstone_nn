// Verify the style labelling scheme: five buttons, grade 4 accepted, scheme recorded,
// progress scoped to the round, and the model buttons hidden.
const fs = require('fs'), path = require('path');
const WEB = path.join(__dirname, '..', 'webapp');
const html = fs.readFileSync(path.join(WEB, 'label.html'), 'utf8');
let code = html.match(/<script>([\s\S]*?)<\/script>/)[1];

function mkEl(id) {
  const el = { id, children: [], style: { cssText: '' }, dataset: {}, _t: '', _h: '',
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
const gradesBox = mkEl('gradesbox');
global.document = {
  createElement: () => mkEl('new'),
  getElementById: id => (byId[id] ||= mkEl(id)),
  querySelector: (sel) => (sel === '.grades' ? gradesBox : mkEl('q')),
  querySelectorAll: () => [], addEventListener() {}
};
global.window = { addEventListener() {}, location: { search: '?mix=1&round=5&scheme=style' } };
global.location = global.window.location;

const FILES = ['pca22_dim4x_model.json', 'dimsweep_manifest.json', 'preference_reward.json'];
const F = {};
for (const f of FILES) {
  const p = path.join(WEB, f);
  if (fs.existsSync(p)) F[f] = JSON.parse(fs.readFileSync(p, 'utf8'));
}
const fakeStats = { total: 0, counts: [0, 0, 0, 0, 0], byModel: {} };
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
code = code.replace(/init\(\)\.catch\([\s\S]*?}\)\.;\s*$/, '');
new Function(code)();

(async () => {
  await new Promise(r => setTimeout(r, 900));
  const L = global.window.LABEL;

  const labels = gradesBox.children.map(b => (b._h || '').replace(/\s*<kbd>.*$/, '').trim());
  console.log('buttons rendered :', JSON.stringify(labels));

  // submit one of each of the five grades
  for (const g of [4, 3, 2, 1, 0]) { L.next(); await L.grade(g); }
  const gs = posted.map(p => p.grade);
  const schemes = [...new Set(posted.map(p => p.scheme))];
  const rounds = [...new Set(posted.map(p => p.round))];
  const stats = byId['stats'] ? byId['stats']._h : '';

  console.log('grades posted    :', JSON.stringify(gs));
  console.log('scheme recorded  :', JSON.stringify(schemes));
  console.log('round recorded   :', JSON.stringify(rounds));
  console.log('stats text       :', stats.replace(/\s+/g, ' ').slice(0, 130));

  const want = ['类二次元', '类游戏', '类三次元', '中', '差'];
  const checks = {
    'five buttons in scheme order': JSON.stringify(labels) === JSON.stringify(want),
    'grade 4 accepted by the page': gs.includes(4),
    'all five grades posted': gs.length === 5 && new Set(gs).size === 5,
    'scheme=style is recorded': schemes.length === 1 && schemes[0] === 'style',
    'round=5 is recorded': rounds.length === 1 && rounds[0] === '5',
    'progress names the style categories': /类二次元/.test(stats),
    'progress counts only this round (0)': /已标注<\/span><b>0 张/.test(stats)
  };
  console.log();
  let ok = true;
  for (const [k, v] of Object.entries(checks)) {
    console.log('  ' + (v ? 'PASS' : 'FAIL') + '  ' + k);
    if (!v) ok = false;
  }
  if (!ok) { console.error('\nFAIL'); process.exit(1); }
  console.log('\nPASS the style scheme works end to end. Ready for labelling.');
  console.log('STYLE_SCHEME_READY');
})();
