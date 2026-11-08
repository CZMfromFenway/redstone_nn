// 冒烟：第 14 轮清单（现役臂**现场生成** + 预计算池）能不能正常装载与出图。
// 用法：node tools/_manifest_smoke.js [清单文件名]
const fs = require('fs');
const path = require('path');

const HERE = __dirname;
const MAN = process.argv[2] || 'label_manifest_diffusion14.json';
const N = +(process.argv[3] || 12);

const html = fs.readFileSync(path.join(HERE, 'label.html'), 'utf8');
const m = html.match(/<script>([\s\S]*)<\/script>/);
let js = m[1].replace(/boot\(\);\s*$/, '');
js += '\nglobalThis.__api = { boot, next, grade, get arms(){return arms;},' +
      ' get item(){return item;}, get counts(){return counts;} };\n';

const mkEl = () => ({
  style: {}, textContent: '', innerHTML: '',
  classList: { add() {}, remove() {}, toggle() {}, contains() { return false; } },
  firstElementChild: { style: {} },
  getContext: () => ({ createImageData: (w, h) => ({ data: new Uint8ClampedArray(w * h * 4) }),
                       putImageData: () => {} }),
  addEventListener: () => {}, setAttribute: () => {},
});
const els = {};
const posted = [];
const sandbox = {
  document: { getElementById: (id) => (els[id] = els[id] || mkEl()),
              querySelectorAll: () => [], addEventListener: () => {} },
  location: { search: '?n=' + N + '&man=' + MAN + '&round=14' },
  window: { addEventListener: () => {} },
  fetch: async (url) => {
    if (url === '/label') return { ok: true, json: async () => ({ ok: true }) };
    const p = path.join(HERE, url);
    if (!fs.existsSync(p)) throw new Error('404 ' + url);
    return { ok: true, json: async () => JSON.parse(fs.readFileSync(p, 'utf8')) };
  },
  URLSearchParams, Date, Math, console, parseInt, parseFloat, isNaN, Infinity,
  Float64Array, Uint8Array, Uint8ClampedArray, JSON, setTimeout, clearTimeout, Promise,
};
sandbox.globalThis = sandbox;
sandbox.window.document = sandbox.document;
new Function(...Object.keys(sandbox), js)(...Object.values(sandbox));

(async () => {
  const api = sandbox.__api;
  await api.boot();
  console.log('臂：', api.arms.map(a => a.__name + (a.__pool ? '(池 ' + a.__pool.length + ')' : '(现场生成)')).join(' · '));
  const seen = [];
  for (let i = 0; i < N * api.arms.length; i++) {
    if (!api.item) { console.log('第 ' + i + ' 张没有 item'); break; }
    const it = api.item;
    const sig = it.cols ? it.cols.slice(0, 6).map(v => v.toFixed(2)).join(',') : 'sample#' + it.sample;
    seen.push(it.arm + ':' + sig + ':' + (it.z ? it.z.map(v => v.toFixed(2)).join('/') : ''));
    await api.grade(3);
  }
  const per = {};
  for (const s of seen) { const a = s.split(':')[0]; per[a] = (per[a] || 0) + 1; }
  console.log('每臂出图张数：', JSON.stringify(per));
  const uniq = new Set(seen).size;
  console.log('不同的 (臂, 图, 旋钮) 组合：%d / %d', uniq, seen.length);
  console.log('前 4 张：'); seen.slice(0, 4).forEach(s => console.log('   ' + s));
})();
