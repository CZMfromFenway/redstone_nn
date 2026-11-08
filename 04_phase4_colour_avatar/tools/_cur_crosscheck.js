// 把盲测页里那条「现役两阶段 + 换调色板 + 4选1」的管线原样跑出来，供 Python 对账。
// 用法：node tools/_cur_crosscheck.js > out.json
// 输出：若干条 { z, palette, colours }，colours 是页面算出的 96 个 0..15 值。
const fs = require('fs');
const path = require('path');

const HERE = __dirname;
const html = fs.readFileSync(path.join(HERE, 'label.html'), 'utf8');
const m = html.match(/<script>([\s\S]*)<\/script>/);
if (!m) throw new Error('no script');
let js = m[1].replace(/boot\(\);\s*$/, '');
// 让调色板可被外部指定，隔离「前向」与「调色板抽样」两件事
js = js.replace('const pal = samplePalette(model);',
                'const pal = (globalThis.__forcePal || samplePalette(model));');
// 记录灰度段量化后的 32 个数（coloursOf 内部变量 x），用于比对「量化」这一步
js = js.replace("const mode = (arm.meta && arm.meta.output_mode) || 'palette';",
                "globalThis.__lastX = Array.from(x);\n  const mode = (arm.meta && arm.meta.output_mode) || 'palette';");
// 记录头层的 512 个 logits，用来定位 argmax 的差异
js = js.replace('  const idx = new Uint8Array(32);',
                '  globalThis.__lastLg = Array.from(lg);\n  const idx = new Uint8Array(32);');
js += '\nglobalThis.__api = { coloursOf, samplePalette, boot, paletteColours, fwdLayers,' +
      ' get arms(){return arms;} };\n';

const mkEl = () => ({
  style: {}, textContent: '', innerHTML: '',
  classList: { add() {}, remove() {}, toggle() {}, contains() { return false; } },
  firstElementChild: { style: {} },
  getContext: () => ({ createImageData: (w, h) => ({ data: new Uint8ClampedArray(w * h * 4) }),
                       putImageData: () => {} }),
  addEventListener: () => {}, setAttribute: () => {},
});
const els = {};
const sandbox = {
  document: { getElementById: (id) => (els[id] = els[id] || mkEl()),
              querySelectorAll: () => [], addEventListener: () => {} },
  location: { search: '?n=1&man=label_manifest.json' },
  window: { addEventListener: () => {} },
  fetch: async (url) => {
    const p = path.join(HERE, url);
    if (!fs.existsSync(p)) throw new Error('404 ' + url);
    const txt = fs.readFileSync(p, 'utf8');
    return { ok: true, json: async () => JSON.parse(txt) };
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

  const arm = api.arms.find(a => a.__name === 'cur64_rule');
  if (!arm) throw new Error('清单里没有 cur64_rule');
  const model = arm;

  // 固定 z 序列（与 Python 侧一致）
  const ZS = [];
  let s = 12345;
  const rnd = () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
  const gauss = () => {
    let u = 0, v = 0;
    while (u === 0) u = rnd();
    while (v === 0) v = rnd();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
  for (let i = 0; i < 8; i++) {
    const z = [];
    for (let j = 0; j < 4; j++) z.push(Math.max(-3.5, Math.min(3.5, gauss())));
    ZS.push(z);
  }

  // 三种调色板：base、以及两个「人工指定」的（含 curated config 里出现过的槽值）
  const pals = [
    { name: 'base', pal: model.palette.map(c => c.slice()) },
    { name: 'forceA', pal: model.palette.map((c, i) => (i === 4 ? [0.0, 0.0, 0.0] : c.slice())) },
  ];

  const out = { z: ZS, cases: [], gray: [] };
  sandbox.__forcePal = model.palette.map(c => c.slice());
  for (const z of ZS) {
    const g = Array.from(api.fwdLayers(model.__gray.layers, z));
    out.gray.push({ z, g: g.map(v => Math.round(v * 1e6) / 1e6) });
  }
  for (const p of pals) {
    sandbox.__forcePal = p.pal;
    for (const z of ZS) {
      const cols = Array.from(api.coloursOf(model, z));
      out.cases.push({ pal: p.name, z, x32: sandbox.__lastX, lg: sandbox.__lastLg,
                       cols: cols.map(v => Math.round(v * 10000) / 10000) });
    }
  }
  // 再抽 20 次「页面自己的调色板抽样」，看抽样本身（不固定调色板）
  sandbox.__forcePal = null;
  const draws = [];
  for (let i = 0; i < 20; i++) draws.push(api.samplePalette(model).map(c => c.map(v => Math.round(v * 1000) / 1000)));
  out.sampled_palettes = draws;
  out.palette_base = model.palette;
  out.configs_n = (model.configs || []).length;
  out.slot_p = model.slot_p;
  out.set_lens = (model.palette_sets || []).map(s => s.length);
  console.log(JSON.stringify(out));
})();



