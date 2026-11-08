// 盲测页的「随机性」自检：种子是否每轮都变、臂顺序是否打乱、预计算臂是否抽得动。
// 用法：node tools/_label_seed_selftest.js
// 注意：只读页面与样本 JSON，不写任何文件。
const fs = require('fs');
const path = require('path');

const HERE = __dirname;
const html = fs.readFileSync(path.join(HERE, 'label.html'), 'utf8');
const m = html.match(/<script>([\s\S]*)<\/script>/);
if (!m) throw new Error('没找到脚本块');
let js = m[1].replace(/boot\(\);\s*$/, '');
js += '\nglobalThis.__api = { boot, next, grade, skipOne, get schedule(){return schedule;},' +
      ' get arms(){return arms;}, get item(){return item;}, get seed0(){return seed0;},' +
      ' get ptr(){return ptr;}, get counts(){return counts;} };\n';

/** 把页面脚本在受控环境里跑一遍。nowMs / random 决定种子。 */
function loadPage({ search = '', nowMs = 1700000000000, random = 0.123456 } = {}) {
  const mkEl = () => ({
    style: {}, textContent: '', innerHTML: '', classList: { add() {}, remove() {}, toggle() {}, contains() { return false; } },
    firstElementChild: { style: {} },
    getContext: () => ({
      createImageData: (w, h) => ({ data: new Uint8ClampedArray(w * h * 4) }),
      putImageData: () => {},
    }),
    addEventListener: () => {},
    setAttribute: () => {},
  });
  const els = {};
  const FakeDate = class extends Date { static now() { return nowMs; } };
  const FakeMath = Object.create(Math);
  FakeMath.random = () => random;

  const sandbox = {
    document: {
      getElementById: (id) => (els[id] = els[id] || mkEl()),
      querySelectorAll: () => [],
      addEventListener: () => {},
    },
    location: { search },
    window: { addEventListener: () => {}, matchMedia: undefined },
    fetch: async (url) => {
      const p = path.join(HERE, url);
      if (!fs.existsSync(p)) throw new Error('404 ' + url);
      const txt = fs.readFileSync(p, 'utf8');
      return { ok: true, json: async () => JSON.parse(txt) };
    },
    URLSearchParams, Date: FakeDate, Math: FakeMath, console,
    parseInt, parseFloat, isNaN, Infinity, Float64Array, Uint8Array, Uint8ClampedArray, JSON,
    setTimeout, clearTimeout, Promise,
  };
  sandbox.globalThis = sandbox;
  sandbox.window.document = sandbox.document;

  const fn = new Function(...Object.keys(sandbox), js);
  fn(...Object.values(sandbox));
  return sandbox.__api;
}

function sig(api) {
  return {
    seed0: api.seed0,
    firstArms: api.schedule.slice(0, 8),
    distinctArms: [...new Set(api.schedule)].length,
    len: api.schedule.length,
  };
}

let fail = 0;
function check(name, ok, extra) {
  console.log((ok ? '  ✓ ' : '  ✗ ') + name + (extra === undefined ? '' : '  ' + extra));
  if (!ok) fail++;
}

(async () => {
  console.log('== 1. 同一秒、同一 Math.random → 种子应完全相同（可复现） ==');
  const a1 = loadPage({ search: '?n=2&man=label_manifest.json', nowMs: 1700000000000, random: 0.5 });
  await a1.boot();
  const a2 = loadPage({ search: '?n=2&man=label_manifest.json', nowMs: 1700000000000, random: 0.5 });
  await a2.boot();
  check('两次装载种子相同', a1.seed0 === a2.seed0, a1.seed0);
  check('两次装载臂顺序相同', JSON.stringify(a1.schedule.slice(0, 8)) === JSON.stringify(a2.schedule.slice(0, 8)));

  console.log('== 2. 换一秒 → 种子必须变，臂顺序也必须变 ==');
  const b1 = loadPage({ search: '?n=2&man=label_manifest.json', nowMs: 1700000000000, random: 0.5 });
  await b1.boot();
  const b2 = loadPage({ search: '?n=2&man=label_manifest.json', nowMs: 1700000001000, random: 0.5 });
  await b2.boot();
  check('隔一秒种子不同', b1.seed0 !== b2.seed0, b1.seed0 + ' vs ' + b2.seed0);
  check('隔一秒臂顺序不同', JSON.stringify(b1.schedule.slice(0, 8)) !== JSON.stringify(b2.schedule.slice(0, 8)));

  console.log('== 3. 配额与顺序：每臂 QUOTA 张、整轮打乱 ==');
  const c = loadPage({ search: '?n=3&man=label_manifest.json', nowMs: 1700000000000, random: 0.5 });
  await c.boot();
  const cnt = {};
  for (const n of c.schedule) cnt[n] = (cnt[n] || 0) + 1;
  check('臂数 × 3 = 总配额', c.schedule.length === 3 * c.arms.length, c.schedule.length + ' 张');
  check('每臂恰好 3 张', Object.values(cnt).every(v => v === 3), JSON.stringify(cnt));
  const runs = [];
  for (let t = 0; t < 5; t++) {
    const p = loadPage({ search: '?n=3&man=label_manifest.json', nowMs: 1700000000000 + t * 137, random: 0.5 });
    await p.boot();
    runs.push(p.schedule[0]);
  }
  check('前 5 轮的首张臂不全相同', new Set(runs).size > 1, runs.join(','));

  console.log('== 4. 预计算臂（扩散/真实数据）能抽到不同的图，且**一轮内不重复** ==');
  const d = loadPage({ search: '?n=4&man=label_manifest_diffusion.json', nowMs: 1700000000000, random: 0.5 });
  await d.boot();
  check('清单里含预计算臂', d.arms.some(x => x.__pool && x.__pool.length > 0));
  const picked = [];
  for (let i = 0; i < 12; i++) { d.next(); picked.push(d.item && d.item.sample); }
  check('连续 12 次抽到的样本下标不全相同', new Set(picked.filter(v => v != null)).size > 1, picked.join(','));

  // 关键：每个臂抽满 n 张后不能有重复（有放回时 200 池抽 40 张平均会有 ~4 次重复）
  const NQ = 40;
  const e2 = loadPage({ search: '?n=' + NQ + '&man=label_manifest_diffusion13.json', nowMs: 1700000000000, random: 0.5 });
  await e2.boot();
  const per = {};
  for (let i = 0; i < NQ * e2.arms.length; i++) { e2.next(); per[e2.item.arm] = (per[e2.item.arm] || []).concat([e2.item.sample]); }
  let dupArms = [];
  for (const k in per) { if (new Set(per[k]).size !== per[k].length) dupArms.push(k + '(' + per[k].length + '->' + new Set(per[k]).size + ')'); }
  check('每臂抽 ' + NQ + ' 张全部互不相同', dupArms.length === 0, dupArms.join(' '));

  console.log('== 5. 「跳过」不消耗配额 ==');
  const e = loadPage({ search: '?n=2&man=label_manifest.json', nowMs: 1700000000000, random: 0.5 });
  await e.boot();
  const before = e.ptr;
  e.skipOne();
  check('跳过之后 ptr 不变', e.ptr === before, before + ' -> ' + e.ptr);

  console.log(fail === 0 ? '\n全部通过' : '\n有 ' + fail + ' 项失败');
  process.exit(fail === 0 ? 0 : 1);
})();
