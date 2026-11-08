// Node 回归：跑 config_ranker.html 的 JS（DOM 桩），检查枚举、打分、导出。
// 用法：node tools/_config_ranker_selftest.js
const fs = require('fs');
const path = require('path');

const html = fs.readFileSync(path.join(__dirname, '..', 'webapp', 'config_ranker.html'), 'utf8');
const js = html.slice(html.indexOf('<script>') + 8, html.indexOf('</script>'));

function mkEl(tag) {
  const el = {
    tagName: (tag || 'div').toUpperCase(), children: [], style: {}, dataset: {},
    className: '', textContent: '', _html: '', value: 'all', width: 8, height: 8,
    classList: { add(){}, remove(){}, toggle(){} },
    appendChild(c) { this.children.push(c); return c; },
    addEventListener() {}, scrollIntoView() {},
    querySelectorAll() { return []; },
    querySelector() { return mkEl('i'); },
    getContext() { return { createImageData: (w, h) => ({ data: new Uint8ClampedArray(w*h*4) }),
                            putImageData() {} }; },
    set innerHTML(v) { this._html = v; if (v === '') this.children = []; },
    get innerHTML() { return this._html; },
  };
  return el;
}
const els = {};
const store = {};
const doc = {
  getElementById: (id) => (els[id] = els[id] || mkEl('div')),
  createElement: (t) => mkEl(t),
  querySelector: () => mkEl('i'),
  addEventListener() {},
};
const boot = new Function('document', 'localStorage', 'URL', 'Blob', 'confirm',
  js + '\nreturn {D:D, S:S, cards:cards, setScore:setScore, paletteOf:paletteOf};');
const api = boot(doc,
  { getItem: (k) => store[k] || null, setItem: (k, v) => { store[k] = v; } },
  { createObjectURL: () => 'blob:x' }, function () {}, () => true);

let ok = true;
const check = (n, c, x) => { console.log(`  ${c ? 'OK  ' : 'FAIL'} ${n}${x ? ' — ' + x : ''}`); if (!c) ok = false; };

check('枚举 144 种组合', api.D.combos.length === 144, 'n=' + api.D.combos.length);
check('可见槽位 = 4/7/14', JSON.stringify(api.D.vis) === '[4,7,14]', api.D.vis.join(','));
check('每卡 8 张脸', api.D.faces.length === 8 && api.D.faces.every(f => f.length === 32));
check('脸的可替换面积 ≥60%', (api.D.face_cov || []).every(c => c >= 0.6),
      (api.D.face_cov || []).map(c => Math.round(c*100) + '%').join(','));
check('渲染出 144 张卡', api.cards.length === 144, 'cards=' + api.cards.length);

const pal = api.paletteOf({ '4': [15, 0, 0] });
check('paletteOf 覆盖指定槽位', pal[4].join(',') === '255,0,0', pal[4].join(','));
check('未指定槽位保持原色', pal[0].length === 3, JSON.stringify(pal[0]));

api.setScore(3, 5);
api.setScore(7, 2);
check('打分记录', api.S.scores[3] === 5 && api.S.scores[7] === 2,
      JSON.stringify(api.S.scores));
check('打分写进 localStorage', !!store['config_ranker_v1'], store['config_ranker_v1'].slice(0, 40));

const cfg = api.D.combos[0].cfg;
check('组合结构可读', Object.keys(cfg).length === 3,
      JSON.stringify(Object.keys(cfg)));

console.log(ok ? 'CONFIG_RANKER_SELFTEST_OK' : 'CONFIG_RANKER_SELFTEST_FAIL');
process.exit(ok ? 0 : 1);
