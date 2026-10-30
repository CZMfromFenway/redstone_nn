// 层1 全 64 通道 × 32 选项 × 多组 8 位输入 的逐片验收
//   A. 头部输出 = 整数前向 N（按真实位展开，不是"全 0/全 1"那种特例）
//   B. 物理合法性：每个容器信号 ≤ 897、每个比较器信号 ≤ 2^31-1、全部解析
// 用法：node src/verify_planes.js
const fs = require('fs');
const HTML = 'E:/redstone_nn/03_phase3_quantized_network/viewer/plane_sim.html';
const ctxProxy = new Proxy({}, { get: (t, k) => (k === 'canvas' ? { width: 0, height: 0 } : () => {}), set: () => true });
const fakeEl = () => ({ value: '8', checked: true, innerHTML: '', getContext: () => ctxProxy, style: {}, width: 0, height: 0 });
global.document = { getElementById: () => fakeEl(), querySelectorAll: () => [] };
global.window = { devicePixelRatio: 1 };
eval(fs.readFileSync(HTML, 'utf8').match(/<script>([\s\S]*)<\/script>/)[1]
  .replace(/document\.querySelectorAll[\s\S]*$/, ''));

const mj = JSON.parse(fs.readFileSync('E:/redstone_nn/02_phase2_avatar/webapp/pca22_dim4w64L1_model.json', 'utf8'));
const L1 = mj.gen.layers[0].W;
const b = 8;
const U = compile(L1, b);
const NCH = U.length;
const keys = ['greyUp', 'whiteUp', 'inRev', 'bitRev', 'parSwap'];

// 输入组：极端 + 单比特 + 伪随机
const XSET = [
  [0, 0, 0, 0], [255, 255, 255, 255],
  [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
  [128, 0, 0, 0], [0, 128, 0, 0], [0, 0, 0, 128],
  [170, 85, 170, 85], [255, 0, 255, 0], [0, 255, 0, 255],
  [7, 63, 200, 17], [254, 253, 251, 247], [31, 129, 64, 192], [1, 2, 4, 8]
];

function expectN(WS, X, n) {
  let N = 0;
  for (let i = 0; i < n; i++) for (let k = 0; k < b; k++) {
    N += WS[i][k] * ((X[i] >> k) & 1) - Math.min(0, WS[i][k]);
  }
  return N;
}
function headOf(B, o) {
  const rr = []; for (let j = 0; j < B.J; j++) rr.push(B.originZ + (o.bitRev ? 6 * j : -6 * j));
  const pz = (o.whiteUp !== o.bitRev) ? Math.min(...rr) : Math.max(...rr);
  return B.cells.get(B.xw + ',' + pz);
}

const t0 = Date.now();
let headOk = 0, headTot = 0, legalOk = 0, badCell = 0;
const fails = [];
const sizes = new Map();

for (let m = 0; m < 32; m++) {
  const o = {}; keys.forEach((k, i) => o[k] = !!(m & (1 << i)));
  for (let ch = 0; ch < NCH; ch++) {
    for (const X of XSET) {
      const B = build(4, b, U, ch, o);
      B.cells.forEach(c => {
        if (c.ik && c.t === 'blk' && c.c.startsWith('col')) {
          const [i, k] = c.ik, W = B.WS[i][k];
          c.torch = ((((X[i] >> k) & 1) === 1) !== (W > 0)) ? 15 : 0;
        }
      });
      evalBoard(B);
      const head = headOf(B, o);
      const N = expectN(B.WS, X, 4);
      const hGood = !!head && head.state === N;
      const bad = legalCheck(B);
      headTot++; if (hGood) headOk++;
      if (hGood && bad.length === 0) legalOk++;
      badCell += bad.length;
      if (!hGood && fails.length < 6) {
        fails.push(`选项${m} ch=${ch} X=${X.join(',')} head=${head ? head.state : 'null'} 期望=${N}`);
      }
      if (bad.length && fails.length < 12) {
        fails.push(`选项${m} ch=${ch} X=${X.join(',')} 非法格 ${bad.length} 个：` +
          bad.slice(0, 2).map(v => `${v.why}@(x${v.x},z${v.z})=${v.v}`).join('; '));
      }
      if (m === 0) sizes.set(ch, B.W + 'x' + B.L);
    }
  }
}

console.log(`A. 头部 = 整数前向 N：${headOk} / ${headTot}  ${headOk === headTot ? '✓' : '✗'}`);
console.log(`B. 物理合法性：      ${legalOk} / ${headTot}  ${badCell === 0 ? '✓' : '✗ 违规格 ' + badCell}`);
const dims = [...new Set(sizes.values())];
console.log(`C. 默认选项下尺寸：${dims.join(' / ')}  ${dims.length === 1 ? '✓ 64 片同尺寸' : '✗ 尺寸不一致'}`);
if (fails.length) { console.log('  失败样例：'); fails.forEach(f => console.log('    ' + f)); }
console.log(`耗时 ${((Date.now() - t0) / 1000).toFixed(1)} s`);
const pass = headOk === headTot && badCell === 0 && dims.length === 1;
console.log(pass ? '全部通过 ✓' : '有失败项 ✗');
process.exitCode = pass ? 0 : 1;
