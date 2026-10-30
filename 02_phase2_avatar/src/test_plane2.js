// Verify the reworked plane page: the click path actually reaches a cell, hover does nothing
// without the button, arrow keys step the fixed axes, and only one category exists by default.
const fs = require('fs'), path = require('path');
const WEB = path.join(__dirname, '..', 'webapp');
const html = fs.readFileSync(path.join(WEB, 'plane.html'), 'utf8');
const code = html.match(/<script>([\s\S]*?)<\/script>/)[1];

function mkEl(tag) {
  const el = {
    tagName: (tag || 'div').toUpperCase(), children: [], style: {}, dataset: {},
    className: '', parentNode: null, _h: '', _t: '', value: '', id: '', onclick: null,
    appendChild(c) { c.parentNode = this; this.children.push(c); return c; },
    set innerHTML(v) { this._h = v; if (v === '') this.children.forEach(c => c.parentNode = null),
                       this.children = v === '' ? [] : this.children; },
    get innerHTML() { return this._h; },
    set textContent(v) { this._t = v; }, get textContent() { return this._t; },
    setPointerCapture() {}, releasePointerCapture() {},
    getBoundingClientRect() { return { left:0, top:0, width:600, height:600 }; },
    getContext() { return { createImageData: (w,h) => ({ data: new Uint8ClampedArray(w*h*4) }),
      putImageData() {}, clearRect() {} }; },
    width: 8, height: 8,
    classList: { _s: new Set(['cell']),
      add(c) { this._s.add(c); }, remove(c) { this._s.delete(c); },
      contains(c) { return this._s.has(c); },
      toggle(c, o) { o === undefined ? (this._s.has(c) ? this._s.delete(c) : this._s.add(c))
                                     : (o ? this._s.add(c) : this._s.delete(c)); } }
  };
  return el;
}
const byId = {};
const listeners = {};
function addL(el, type, fn) {
  const key = (el && el.id) || 'grid';
  (listeners[key] = listeners[key] || {});
  (listeners[key][type] = listeners[key][type] || []).push(fn);
}
global.document = {
  createElement: (t) => { const e = mkEl(t); e.addEventListener = (ty, f) => addL(e, ty, f); return e; },
  getElementById: (id) => {
    if (!byId[id]) {
      const e = Object.assign(mkEl('div'), { id });
      e.addEventListener = (ty, f) => addL(e, ty, f);
      byId[id] = e;
    }
    return byId[id];
  },
  addEventListener: (ty, f) => addL({ id: '__doc' }, ty, f),
  elementFromPoint: () => null
};
global.window = { addEventListener() {}, location: { search: '' } };
global.location = global.window.location;
const store = {};
global.localStorage = { getItem: k => store[k] ?? null, setItem: (k, v) => { store[k] = v; } };
global.Blob = class { constructor(p) { this.p = p; } };
global.URL = { createObjectURL: () => 'blob:x', revokeObjectURL() {} };
global.confirm = () => true;
global.alert = m => { throw new Error('alert: ' + m); };

// the grid element must exist before the page script runs
byId['grid'] = Object.assign(mkEl('div'), { id: 'grid' });
byId['grid'].addEventListener = (ty, f) => addL({ id: 'grid' }, ty, f);

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

const fire = (type, ev) => (listeners['grid'][type] || []).forEach(f => f(ev));

(async () => {
  await new Promise(r => setTimeout(r, 2000));
  const P = window.PLANE;
  if (!P) { console.error('page never initialised'); process.exit(1); }
  console.log('models   :', P.models.length);
  console.log('categories:', P.state.cats.map(c => c.name).join(' / '));

  const cells = byId['grid'].children.filter(c => c.classList.contains('cell'));
  console.log('cells    :', cells.length);

  // find the cell that represents xi=2, yi=5
  const target = cells.find(c => +c.dataset.xi === 2 && +c.dataset.yi === 5);
  const target2 = cells.find(c => +c.dataset.xi === 3 && +c.dataset.yi === 5);
  if (!target) { console.error('no target cell'); process.exit(1); }

  // ---- click path: pointerdown then pointerup on the same cell with no movement
  const ev = (el, x, y, btn) => ({ target: el, clientX: x, clientY: y, button: btn || 0,
                                   pointerId: 1, preventDefault() {} });

  fire('pointerdown', ev(target, 100, 100));
  fire('pointerup', ev(target, 100, 100));
  const after1 = P.getMark(2, 5);
  console.log('after click       :', after1);

  // clicking the same cell again must clear it
  fire('pointerdown', ev(target, 100, 100));
  fire('pointerup', ev(target, 100, 100));
  const after2 = P.getMark(2, 5);
  console.log('after second click:', after2);

  // ---- hover with no button must do nothing
  fire('pointermove', ev(target2, 200, 200));
  const after3 = P.getMark(3, 5);
  console.log('after bare hover  :', after3);

  // ---- hold and sweep: down on one cell, move to another, up
  fire('pointerdown', ev(target, 100, 100));
  fire('pointermove', ev(target2, 300, 300));
  fire('pointerup', ev(target2, 300, 300));
  const swept = [P.getMark(2, 5), P.getMark(3, 5)];
  console.log('after sweep       :', JSON.stringify(swept));

  // ---- arrow keys step the fixed axes
  const beforeFixed = JSON.stringify(P.state.fixed);
  const arrows = (listeners['__doc'] && listeners['__doc']['keydown']) || [];
  const kd = (k) => arrows.forEach(f => f({ key: k, target: { tagName: 'BODY' },
                                            preventDefault() {} }));
  kd('ArrowRight');
  const afterRight = JSON.stringify(P.state.fixed);
  kd('ArrowUp');
  const afterUp = JSON.stringify(P.state.fixed);
  console.log('fixed before      :', beforeFixed);
  console.log('fixed after Right :', afterRight);
  console.log('fixed after Up    :', afterUp);

  const checks = {
    'ten models': P.models.length === 10,
    'only one category by default (中)':
      P.state.cats.length === 1 && P.state.cats[0].name === '中',
    '64 cells': cells.length === 64,
    'a click marks the cell': after1 === 'mid',
    'a second click clears it': after2 === null,
    'a bare hover does nothing': after3 === null,
    'hold and sweep marks both cells': swept[0] === 'mid' && swept[1] === 'mid',
    'right arrow changes a fixed axis': afterRight !== beforeFixed,
    'up arrow changes the other axis': afterUp !== afterRight
  };
  console.log();
  let ok = true;
  for (const [k, v] of Object.entries(checks)) {
    console.log('  ' + (v ? 'PASS' : 'FAIL') + '  ' + k);
    if (!v) ok = false;
  }
  if (!ok) { console.error('\nFAIL'); process.exit(1); }
  console.log('\nPASS the reworked plane page behaves as asked.');
  console.log('PLANE2_READY');
})();
