// 导出层1 全部 64 个通道的板（每通道一片），供 Python 堆叠
const fs = require('fs');
const ctxProxy = new Proxy({}, { get: (t, k) => (k === 'canvas' ? { width: 0, height: 0 } : () => {}), set: () => true });
const fakeEl = () => ({ value: '8', checked: true, innerHTML: '', getContext: () => ctxProxy, style: {}, width: 0, height: 0 });
global.document = { getElementById: () => fakeEl(), querySelectorAll: () => [] };
global.window = { devicePixelRatio: 1 };
eval(fs.readFileSync('E:/redstone_nn/03_phase3_quantized_network/viewer/plane_sim.html', 'utf8')
  .match(/<script>([\s\S]*)<\/script>/)[1].replace(/document\.querySelectorAll[\s\S]*$/, ''));

const mj = JSON.parse(fs.readFileSync('E:/redstone_nn/02_phase2_avatar/webapp/pca22_dim4w64L1_model.json', 'utf8'));
const L1 = mj.gen.layers[0].W;
const b = 8;
const U = compile(L1, b);
const OPT = { greyUp: true, whiteUp: true, inRev: false, bitRev: false, parSwap: false };
const X = [0, 0, 0, 0];          // 火把塔抽掉后，取合法输入 X=0

const planes = [];
let badTotal = 0;
let maxBarrel = 0;
for (let ch = 0; ch < U.length; ch++) {
  const B = build(4, b, U, ch, OPT);
  // 输入 X=0：火把亮 ⟺ X_bit != [W>0] ⟺ (W>0)
  B.cells.forEach(c => {
    if (c.ik && c.t === 'blk' && c.c.startsWith('col')) {
      const [i, k] = c.ik, W = B.WS[i][k];
      c.torch = ((((X[i] >> k) & 1) === 1) !== (W > 0)) ? 15 : 0;
    }
  });
  evalBoard(B);
  // 导出门槛：物理不合法就不许写文件
  const bad = legalCheck(B);
  if (bad.length) {
    badTotal += bad.length;
    console.log(`  ✗ ch=${ch} 非法格 ${bad.length} 个：` +
      bad.slice(0, 3).map(v => `${v.why}@(x${v.x},z${v.z})=${v.v}`).join('; '));
  }
  const arr = [];
  B.cells.forEach(c => {
    if (c.t === 'barrel') maxBarrel = Math.max(maxBarrel, c.val | 0);
    arr.push({
      x: c.x, z: c.z, t: c.t, c: c.c || null, out: c.out || null,
      mode: c.mode || null, st: c.state === undefined ? null : c.state,
      v: c.val === undefined ? null : c.val, ik: c.ik || null
    });
  });
  planes.push({ ch, W: B.W, L: B.L, cells: arr });
  if (ch % 16 === 0) console.log(`  ch=${ch} ${B.W}x${B.L} cells=${arr.length}`);
}
if (badTotal) {
  console.log(`✗ 物理合法性检查未通过（${badTotal} 个非法格），已放弃写 planes64.json`);
  process.exit(1);
}
console.log(`物理合法性 ✓（0 违规），最大容器信号 ${maxBarrel} ≤ 897`);
const W = planes[0].W, L = planes[0].L;
const bad = planes.filter(p => p.W !== W || p.L !== L);
console.log(`共 ${planes.length} 片，尺寸 ${W}x${L}，尺寸不一致的 ${bad.length} 片`);
fs.writeFileSync('E:/redstone_nn/03_phase3_quantized_network/results/planes64.json', JSON.stringify({ W, L, b, planes }));
console.log('已写 planes64.json');
