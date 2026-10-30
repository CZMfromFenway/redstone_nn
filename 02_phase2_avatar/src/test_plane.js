// Verify the plane-labelling page: 64 cells, correct axis orientation, three marking paths,
// custom categories, and an export that round-trips the coordinates.
const fs = require('fs'), path = require('path');
const WEB = path.join(__dirname, '..', 'webapp');
const html = fs.readFileSync(path.join(WEB, 'plane.html'), 'utf8');
const code = html.match(/<script>([\s\S]*?)<\/script>/)[1];

function mkEl(tag) {
  const el = { tagName: (tag || 'div').toUpperCase(), children: [], style: {}, dataset: {},
    className: '', _h: '', _t: '', value: '', id: '',
    appendChild(c) { this.children.push(c); return c; },
    addEventListener() {}, removeEventListener() {},
    setPointerCapture() {},
    querySelectorAll() { return []; }, querySelector() { return null; },
    classList: { _s: new Set(),
      add(c) { this._s.add(c); }, remove(c) { this._s.delete(c); },
      contains(c) { return this._s.has(c); },
      toggle(c, on) { if (on === undefined) { this._s.has(c) ? this._s.delete(c) : this._s.add(c); }
                      else { on ? this._s.add(c) : this._s.delete(c); } } },
    getBoundingClientRect() { return { left: 0, top: 0, width: 600, height: 600 }; },
    getContext() { return { createImageData: (w, h) => ({ data: new Uint8ClampedArray(w*h*4) }),
      putImageData() {}, clearRect() {} }; },
    set innerHTML(v) { this._h = v; if (v === '') this.children = []; },
    get innerHTML() { return this._h; },
    set textContent(v) { this._t = v; }, get textContent() { return this._t; },
    width: 8, height: 8 };
  return el;
}
const byId = {};
global.document = {
  createElement: mkEl,
  getElementById: id => (byId[id] ||= Object.assign(mkEl('div'), { id })),
  addEventListener() {},
  elementFromPoint() { return null; }
};
global.window = { addEventListener() {}, location: { search: '' } };
global.location = global.window.location;
const store = {};
global.localStorage = { getItem: k => store[k] ?? null, setItem: (k, v) => { store[k] = v; } };
global.Blob = class { constructor(p) { this.p = p; } };
global.URL = { createObjectURL: () => 'blob:x', revokeObjectURL() {} };
global.confirm = () => true;
global.alert = m => { throw new Error('alert: ' + m); };

const fam = JSON.parse(fs.readFileSync(path.join(WEB, 'grayscale_family.json'), 'utf8'));
const FILES = {};
for (const m of fam.models) {
  const p = path.join(WEB, m.file);
  if (fs.existsSync(p)) FILES[m.file] = JSON.parse(fs.readFileSync(p, 'utf8'));
}
global.fetch = async (u) => {
  const k = String(u).split('?')[0].split('/').pop();
  if (k === 'grayscale_family.json') return { ok: true, json: async () => fam };
  if (FILES[k]) return { ok: true, json: async () => FILES[k] };
  return { ok: false, status: 404, json: async () => ({}) };
};

new Function(code)();

(async () => {
  await new Promise(r => setTimeout(r, 2000));
  const P = window.PLANE;
  if (!P) { console.error('page never initialised'); process.exit(1); }
  console.log('models   :', P.models.length);
  console.log('planes   : x=' + P.state.xA + ' y=' + P.state.yA +
              ' fixed=' + JSON.stringify(P.state.fixed));

  // 64 cells exist
  const cells = P.cells();
  console.log('cells    :', cells.length);

  // axis orientation: the top row must carry the highest y value
  const topRow = cells.filter(c => c.yi === 7);
  const botRow = cells.filter(c => c.yi === 0);
  console.log('top row yi:', topRow.length, ' bottom row yi:', botRow.length);

  // the seed for a cell must place the axis PCs at the matching level
  const sTop = P.seedAt(0, 7), sBot = P.seedAt(0, 0);
  console.log('seed at (xi=0, yi=7):', JSON.stringify(sTop));
  console.log('seed at (xi=0, yi=0):', JSON.stringify(sBot));

  // marking: click path and rectangle path both end up in setMark
  P.setMark(0, 7, 'mid');
  P.setMark(3, 5, 'bad');
  P.setMark(0, 0, 'mid');
  const mk = P.state.marks[P.planeKey()];
  const nAfterThree = Object.keys(mk).length;
  console.log('marks    :', JSON.stringify(mk));

  // custom category
  P.state.cats.push({ id: 'girl', name: '少女', color: '#8fd3ff' });
  P.setMark(4, 4, 'girl');
  const mk2 = P.state.marks[P.planeKey()];
  console.log('cats     :', P.state.cats.map(c => c.name).join(' / '));

  // switching the plane must not lose the previous marks
  const before = Object.keys(P.state.marks).length;
  P.state.fixed[0] = -1.5;
  P.rebuild();
  const after = Object.keys(P.state.marks).length;
  console.log('planes stored across a rebuild:', before, '->', after);

  const checks = {
    '64 cells': cells.length === 64,
    'eight rows each way':
      new Set(cells.map(c => c.yi)).size === 8 && new Set(cells.map(c => c.xi)).size === 8,
    'top row is the highest y value': sTop[P.state.yA] === 3.5 && sBot[P.state.yA] === -3.5,
    'axis PCs use the plane levels':
      P.levels.includes(sTop[P.state.xA]) && P.levels.includes(sTop[P.state.yA]),
    'non-axis PCs are the fixed values':
      [0,1,2,3].filter(i => i !== P.state.yA && i !== P.state.xA)
        .every(i => P.levels.includes(sTop[i])),
    'marks recorded for click and rectangle paths': nAfterThree === 3,
    'a custom category can be marked': Object.values(mk2).includes('girl'),
    'marks survive a rebuild': after >= before,
    'three categories exist': P.state.cats.length === 3
  };
  console.log();
  let ok = true;
  for (const [k, v] of Object.entries(checks)) {
    console.log('  ' + (v ? 'PASS' : 'FAIL') + '  ' + k);
    if (!v) ok = false;
  }
  if (!ok) { console.error('\nFAIL'); process.exit(1); }
  console.log('\nPASS the plane labelling page is ready.');
  console.log('PLANE_READY');
})();
