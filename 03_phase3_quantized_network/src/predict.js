// 随机输入测试：给定 X=(X0..X3)，仿真 64 个通道，输出游戏里可对照的检查点。
//   火把：亮 ⟺ 该位的 X_bit ≠ [W>0]
//   检查点（绝对坐标，y = 2ch+3）：
//     灰环 K          (x=1, z=21)
//     白环 W0         (x=11, z=23)
//     白总线沿途      (x=11, z=21/15/9)  + 头部 (x=11, z=3) = N
// 用法：node src/predict.js [X0 X1 X2 X3] [ch,ch,...]
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
const b = 8, n = 4, NCH = L1.length;
const U = compile(L1, b);
const OPT = { greyUp: true, whiteUp: true, inRev: false, bitRev: false, parSwap: false };

let X = process.argv.slice(2, 6).map(Number);
if (X.length < 4) X = [186, 37, 204, 91];
const which = (process.argv[6] || '0,3,20,63').split(',').map(Number);

function run(ch) {
  const B = build(n, b, U, ch, OPT);
  B.cells.forEach(c => {
    if (c.ik && c.t === 'blk' && c.c.startsWith('col')) {
      const [i, k] = c.ik, W = B.WS[i][k];
      c.torch = ((((X[i] >> k) & 1) === 1) !== (W > 0)) ? 15 : 0;
    }
  });
  evalBoard(B);
  const g = (x, z) => { const c = B.cells.get(x + ',' + z); return c ? c.state : null; };
  let N = 0;
  for (let i = 0; i < n; i++) for (let k = 0; k < b; k++)
    N += B.WS[i][k] * ((X[i] >> k) & 1) - Math.min(0, B.WS[i][k]);
  return { ch, K: B.K, W0: B.W0, bus: [g(11, 21), g(11, 15), g(11, 9)], head: g(11, 3), N,
           legal: legalCheck(B).length };
}

console.log('X = (' + X.join(', ') + ')   二进制 LSB 在前：');
for (let i = 0; i < 4; i++) {
  const bits = []; for (let k = 0; k < 8; k++) bits.push((X[i] >> k) & 1);
  console.log(`  X${i} = ${String(X[i]).padStart(3)} = 0b` +
    [7, 6, 5, 4, 3, 2, 1, 0].map(k => (X[i] >> k) & 1).join('') +
    `   (位0..7 = ${bits.join(' ')})`);
}
console.log('\n32 条总线（塔 (i,k) 送 X_i 的第 k 位，1 = 给塔底块充能）：');
for (let i = 0; i < 4; i++) {
  const bits = []; for (let k = 7; k >= 0; k--) bits.push((X[i] >> k) & 1);
  console.log(`  输入${i}: 位7..0 = ${bits.join(' ')}`);
}

console.log('\n全部 64 通道头部 N：');
const all = [];
for (let ch = 0; ch < NCH; ch++) all.push(run(ch));
const bad = all.filter(r => r.head !== r.N || r.legal);
console.log('  ' + all.map(r => r.head).join(' '));
console.log(bad.length ? `  ✗ ${bad.length} 个通道不一致` : '  ✓ 64/64 与整数前向 N 一致，且物理合法');

console.log('\n抽查通道（坐标 y = 2ch+3）：');
const hdr = 'ch | 灰环K  白环W0 | x11,z21  x11,z15  x11,z9 | 头部 x11,z3 | 公式N';
console.log(hdr);
console.log('-'.repeat(hdr.length));
for (const ch of which) {
  const r = all[ch];
  console.log(`${String(ch).padStart(2)} | ${String(r.K).padStart(5)}  ${String(r.W0).padStart(5)} | ` +
    `${String(r.bus[0]).padStart(7)}  ${String(r.bus[1]).padStart(7)}  ${String(r.bus[2]).padStart(6)} | ` +
    `${String(r.head).padStart(10)} | ${r.N}`);
}

const out = { X, options: OPT, all: all.map(r => ({ ch: r.ch, K: r.K, W0: r.W0, bus: r.bus, head: r.head })) };
fs.writeFileSync('E:/redstone_nn/03_phase3_quantized_network/results/predict_X.json', JSON.stringify(out, null, 1));
console.log('\n已写 results/predict_X.json');
