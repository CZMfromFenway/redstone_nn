// Node 回归：跑 palette_editor.html（v2，多候选）的 JS。
// 用法：node tools/_palette_editor_selftest.js
const fs = require('fs');
const path = require('path');

const html = fs.readFileSync(path.join(__dirname, '..', 'webapp', 'palette_editor.html'), 'utf8');
const m = html.match(/<script>([\s\S]*)<\/script>/);
if (!m) throw new Error('no script');
const js = m[1].replace(/<\/script>[\s\S]*$/, '');

function mkEl(tag) {
  return {
    tagName: (tag || 'div').toUpperCase(),
    children: [], style: {}, dataset: {}, value: '', checked: false,
    textContent: '', title: '', draggable: false, width: 8, height: 8, className: '',
    classList: { add(){}, remove(){}, toggle(){} },
    appendChild(c) { this.children.push(c); return c; },
    addEventListener() {}, click() {},
    getContext() { return { createImageData: (w, h) => ({ data: new Uint8ClampedArray(w*h*4) }),
                            putImageData() {} }; },
    querySelector() { return mkEl('div'); },
    querySelectorAll() { return []; },
    set innerHTML(v) { this._html = v; }, get innerHTML() { return this._html || ''; },
  };
}
const els = {};
global.document = {
  getElementById: (id) => (els[id] = els[id] || mkEl('div')),
  createElement: (t) => mkEl(t),
  addEventListener() {},
};
global.window = global;
global.navigator = { clipboard: { writeText: async () => {} } };
global.Blob = function () {};
global.URL = { createObjectURL: () => 'blob:x' };

const boot = new Function('document', 'window', 'navigator', 'Blob', 'URL',
  js + '\nreturn {S:S, D:D, addAlt:addAlt, applyPreset:applyPreset, ' +
        'schemeJSON:schemeJSON, paletteOnce:paletteOnce};');
const api = boot(global.document, global.window, global.navigator, global.Blob, global.URL);
const S = api.S, D = api.D;

let ok = true;
const check = (n, c, x) => { console.log(`  ${c ? 'OK  ' : 'FAIL'} ${n}${x ? ' — ' + x : ''}`); if (!c) ok = false; };

check('数据', D.slots.length === 16 && D.high.length === 61 && D.faces.length === 24
             && Object.keys(D.prefill).length > 0,
      'slots=16 high=61 faces=24 prefill=' + Object.keys(D.prefill).length);
check('初始 s30：3 槽各 1 候选',
      Object.keys(S.alts).length === 3 && Object.values(S.alts).every(a => a.length === 1),
      'slots=' + Object.keys(S.alts).join(','));

const iFire = D.high.findIndex(h => h.name === 'FIRE' && Math.abs(h.L - 4.5) < 0.2);
const iRed  = D.high.findIndex(h => h.name === 'RED' && Math.abs(h.h - 0) < 5);
api.addAlt(4, iFire);
api.addAlt(4, iRed >= 0 ? iRed : D.high.findIndex(h => h.name === 'TERRACOTTA_RED'));
check('一个槽可以有多个候选', S.alts[4].length === 2, S.alts[4].map(a => a.name).join(','));
check('默认权重 1 / 概率 100%',
      S.alts[4].every(a => a.weight === 1) && S.p[4] === 1);

S.alts[4][0].weight = 3;
S.alts[4][1].weight = 1;
let nFire = 0, nRed = 0, nOrig = 0;
for (let t = 0; t < 4000; t++) {
  const c = api.paletteOnce()[4];
  const d = (a) => Math.abs(a[0]-c[0]) + Math.abs(a[1]-c[1]) + Math.abs(a[2]-c[2]);
  if (d(S.alts[4][0].rgb) < 0.5) nFire++;
  else if (d(S.alts[4][1].rgb) < 0.5) nRed++;
  else nOrig++;
}
check('权重 3:1 抽样生效', nFire > nRed * 2 && nFire > 2000,
      `候选1=${nFire} 候选2=${nRed} 原色=${nOrig}`);

S.p[4] = 0;
let allOrig = true;
for (let t = 0; t < 500; t++) {
  const c = api.paletteOnce()[4];
  if (Math.abs(c[0] - D.slots[4].rgb[0]) > 0.5) { allOrig = false; break; }
}
check('p=0 时该槽不被替换', allOrig);
S.p[4] = 1;

const j = api.schemeJSON();
check('导出 v2', j.kind === 'palette_scheme' && j.version === 2);
const s4 = j.slots.find(x => x.slot === 4);
check('导出含多候选与权重',
      !!s4 && s4.alts.length === 2 && s4.alts[0].weight === 3 && typeof s4.p === 'number',
      s4 ? JSON.stringify(s4.alts.map(a => [a.name, a.weight, a.dL])) : 'null');
check('palette_first_alt 16 项',
      Array.isArray(j.palette_first_alt) && j.palette_first_alt.length === 16);
check('summary 影响面积', typeof j.summary.affected_pixel_share === 'number',
      'share=' + j.summary.affected_pixel_share);

const pf = Object.values(D.prefill)[0];
check('prefill 条目完整', pf.every(a => a.rgb && a.name && typeof a.L === 'number'),
      'n=' + pf.length);

console.log(ok ? 'PALETTE_EDITOR_SELFTEST_OK' : 'PALETTE_EDITOR_SELFTEST_FAIL');
process.exit(ok ? 0 : 1);
