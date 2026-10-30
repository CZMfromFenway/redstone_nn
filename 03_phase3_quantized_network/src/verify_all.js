// 总验收：仿真器 vs Python 编译模型
const fs = require('fs');
const ctxProxy = new Proxy({}, { get: (t, k) => (k === 'canvas' ? { width: 0, height: 0 } : () => {}), set: () => true });
const fakeEl = () => ({ value: '8', checked: true, innerHTML: '', getContext: () => ctxProxy, style: {}, width: 0, height: 0 });
global.document = { getElementById: () => fakeEl(), querySelectorAll: () => [] };
global.window = { devicePixelRatio: 1 };
eval(fs.readFileSync('E:/redstone_nn/03_phase3_quantized_network/viewer/plane_sim.html', 'utf8')
  .match(/<script>([\s\S]*)<\/script>/)[1].replace(/document\.querySelectorAll[\s\S]*$/, ''));

const REF = JSON.parse(fs.readFileSync('E:/redstone_nn/03_phase3_quantized_network/results/py_compile_reference.json', 'utf8'));
const FC = JSON.parse(fs.readFileSync('E:/redstone_nn/03_phase3_quantized_network/results/py_forward_cases.json', 'utf8'));
const mj = JSON.parse(fs.readFileSync('E:/redstone_nn/02_phase2_avatar/webapp/pca22_dim4w64L1_model.json', 'utf8'));
const L1 = mj.gen.layers[0].W, L2 = mj.gen.layers[1].W;
const TH1 = mj.gen.layers[0].b;

console.log('━━━ A. 权重编译一致性 ━━━');
let aOk = true;
for (const bs of Object.keys(REF.layers.L1)) {
  const b = +bs, py = REF.layers.L1[bs].W, js = compile(L1, b);
  let d = 0;
  for (let i = 0; i < py.length; i++) for (let j = 0; j < py[i].length; j++) for (let k = 0; k < b; k++) if (py[i][j][k] !== js[i][j][k]) d++;
  console.log(`  层1 b=${b}: ${d === 0 ? '完全一致 ✓' : d + ' 处不同 ✗'}`); if (d) aOk = false;
}
{
  const py = REF.layers.L2['8'].W, js = compile(L2, 8, REF.layers.L2['8'].delta_in, 8);
  let d = 0;
  for (let i = 0; i < py.length; i++) for (let j = 0; j < py[i].length; j++) for (let k = 0; k < 8; k++) if (py[i][j][k] !== js[i][j][k]) d++;
  console.log(`  层2 b=8: ${d === 0 ? '完全一致 ✓' : d + ' 处不同 ✗'}`); if (d) aOk = false;
}
{
  const djs = layer1DeltaOut(L1, TH1, 8, 8), dpy = REF.layers.L2['8'].delta_in;
  let md = 0; for (let i = 0; i < dpy.length; i++) md = Math.max(md, Math.abs(djs[i] - dpy[i]));
  console.log(`  层1 δ_out: 最大差 ${md.toExponential(2)} ${md < 1e-9 ? '✓' : '✗'}`); if (md > 1e-9) aOk = false;
}

console.log('\n━━━ B. 头部输出 = Python 整数前向 N ━━━');
function runCase(ch, xs, orientation) {
  const B = build(4, 8, compile(L1, 8), ch, orientation);
  B.cells.forEach(k => {
    if (k.ik && k.t === 'blk' && k.c.startsWith('col')) {
      const [i, bit] = k.ik, W = B.WS[i][bit];
      k.torch = (((xs[i] === 1) !== (W > 0))) ? 15 : 0;
    }
  });
  evalBoard(B);
  return B;
}
let bOk = 0, bTot = 0, bBad = [];
for (const c of FC.cases) {
  for (const [gu, wu] of [[1, 1], [1, 0], [0, 1], [0, 0]]) {
    const o = { greyUp: !!gu, whiteUp: !!wu, inRev: false, bitRev: false, parSwap: false };
    const B = runCase(c.ch, c.x, o);
    const rr = []; for (let j = 0; j < B.J; j++) rr.push(B.originZ + (o.bitRev ? 6 * j : -6 * j));
    const pz = o.whiteUp ? Math.min(...rr) : Math.max(...rr);
    const head = B.cells.get(B.xw + ',' + pz);
    const v = head ? head.state : null;
    bTot++;
    if (v === c.N) bOk++;
    else if (bBad.length < 5) bBad.push(`ch=${c.ch} x=${c.x} g${gu}w${wu} head=${v} pyN=${c.N}`);
  }
}
console.log(`  ${bOk}/${bTot} 一致 ${bOk === bTot ? '✓' : '✗'}`);
bBad.forEach(x => console.log('    ' + x));

console.log('\n━━━ C. 环与总线的常数 ━━━');
{
  const B = runCase(0, [0, 0, 0, 0], { greyUp: true, whiteUp: true, inRev: false, bitRev: false, parSwap: false });
  const S = B.S.flat();
  const Kpy = S.reduce((a, v) => a + v, 0);
  console.log(`  K  = Σ S           仿真器 ${B.K}   Python ${Kpy}  ${B.K === Kpy ? '✓' : '✗'}`);
  console.log(`  W₀ = J·K − 15nb    仿真器 ${B.W0}   手算 ${B.J * Kpy - 15 * 4 * 8}  ${B.W0 === B.J * Kpy - 15 * 4 * 8 ? '✓' : '✗'}`);
  // 每个环的实际值
  const oX = B.originX, oZ = B.originZ;
  const g = (rx, Z) => B.cells.get((rx + oX) + ',' + (Z + oZ));
  const gr = [g(-2, -2), g(-1, -1), g(-2, 0), g(-3, -1)].map(c => c && c.state);
  const rw = B.xw, wz = Zmsb_white(B);
  console.log(`  灰环四值 ${gr.join('/')}  ${gr.every(v => v === B.K) ? '✓' : '✗'}`);
  const wr = [g(rw - oX + 1, wz - oZ), g(rw - oX + 2, wz + 1 - oZ), g(rw - oX + 1, wz + 2 - oZ), g(rw - oX, wz + 1 - oZ)].map(c => c && c.state);
  console.log(`  白环四值 ${wr.join('/')}  ${wr.every(v => v === B.W0) ? '✓' : '✗'}`);
}
function Zmsb_white(B) { return B.originZ + 1; }

console.log(`\n${aOk && bOk === bTot ? '全部通过 ✓' : '有失败项 ✗'}`);
