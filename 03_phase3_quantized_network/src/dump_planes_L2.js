// 导出层2（64 → 32）全部 32 个通道的板，供 Python 堆叠。
// 层2 的输入是层1 的 64 个输出（各 0..255），所以 delta 逐通道来自 layer1DeltaOut。
// 用法：node src/dump_planes_L2.js
const fs = require('fs');
const HTML = 'E:/redstone_nn/03_phase3_quantized_network/viewer/plane_sim.html';
const ctxProxy = new Proxy({}, { get: (t, k) => (k === 'canvas' ? { width: 0, height: 0 } : () => {}), set: () => true });
const fakeEl = () => ({ value: '8', checked: true, innerHTML: '', getContext: () => ctxProxy, style: {}, width: 0, height: 0 });
global.document = { getElementById: () => fakeEl(), querySelectorAll: () => [] };
global.window = { devicePixelRatio: 1 };
eval(fs.readFileSync(HTML, 'utf8').match(/<script>([\s\S]*)<\/script>/)[1]
  .replace(/document\.querySelectorAll[\s\S]*$/, ''));

const mj = JSON.parse(fs.readFileSync('E:/redstone_nn/02_phase2_avatar/webapp/pca22_dim4w64L1_model.json', 'utf8'));
const L1 = mj.gen.layers[0].W, L2 = mj.gen.layers[1].W, TH1 = mj.gen.layers[0].b;
const b = 8;
const U2 = compile(L2, 8, layer1DeltaOut(L1, TH1, 8, 8), 8);
const n = U2[0].length;
console.log(`层2：${U2.length} 个通道 × ${n} 个输入 × b=${b}`);
const OPT = { greyUp: true, whiteUp: true, inRev: false, bitRev: false, parSwap: false };
const X = new Array(n).fill(0);          // 输入全 0

const planes = [];
let badTotal = 0, maxBarrel = 0;
for (let ch = 0; ch < U2.length; ch++) {
  const B = build(n, b, U2, ch, OPT);
  B.cells.forEach(c => {
    if (c.ik && c.t === 'blk' && c.c.startsWith('col')) {
      const [i, k] = c.ik, W = B.WS[i][k];
      c.torch = ((((X[i] >> k) & 1) === 1) !== (W > 0)) ? 15 : 0;
    }
  });
  evalBoard(B);
  const bad = legalCheck(B);
  if (bad.length) {
    badTotal += bad.length;
    console.log(`  ✗ ch=${ch} 非法格 ${bad.length}：` +
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
  // 头部输出（白总线端口）
  const rr = []; for (let j = 0; j < B.J; j++) rr.push(B.originZ + (OPT.bitRev ? 6 * j : -6 * j));
  const pz = (OPT.whiteUp !== OPT.bitRev) ? Math.min(...rr) : Math.max(...rr);
  const head = B.cells.get(B.xw + ',' + pz);
  let N = 0;
  for (let i = 0; i < n; i++) for (let k = 0; k < b; k++)
    N += B.WS[i][k] * ((X[i] >> k) & 1) - Math.min(0, B.WS[i][k]);
  planes.push({ ch, W: B.W, L: B.L, K: B.K, W0: B.W0, head: head ? head.state : null, N,
                xw: B.xw, pz, cells: arr });
  if (ch % 8 === 0) console.log(`  ch=${ch} ${B.W}x${B.L} cells=${arr.length} head=${head ? head.state : null} N=${N}`);
}
if (badTotal) { console.log(`✗ 物理合法性未通过（${badTotal}），放弃写文件`); process.exit(1); }
const W = planes[0].W, L = planes[0].L;
const mism = planes.filter(p => p.W !== W || p.L !== L);
const hbad = planes.filter(p => p.head !== p.N);
console.log(`共 ${planes.length} 片，尺寸 ${W}x${L}，尺寸不一致 ${mism.length} 片`);
console.log(`头部 = N：${planes.length - hbad.length}/${planes.length}` +
  (hbad.length ? `  ✗ ${hbad.slice(0, 5).map(p => `ch${p.ch}:${p.head}≠${p.N}`).join(' ')}` : '  ✓'));
console.log(`最大容器信号 ${maxBarrel} ≤ 897`);
fs.writeFileSync('E:/redstone_nn/03_phase3_quantized_network/results/planes_L2.json',
  JSON.stringify({ W, L, b, n, planes }));
console.log('已写 results/planes_L2.json');
