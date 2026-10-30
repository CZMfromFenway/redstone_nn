// 确认 b≠8 的分歧只来自 "in_bits 的定义"，公式本身相同
const fs = require('fs');
const ctxProxy = new Proxy({}, { get: (t, k) => (k === 'canvas' ? { width: 0, height: 0 } : () => {}), set: () => true });
const fakeEl = () => ({ value: '8', checked: true, innerHTML: '', getContext: () => ctxProxy, style: {}, width: 0, height: 0 });
global.document = { getElementById: () => fakeEl(), querySelectorAll: () => [] };
global.window = { devicePixelRatio: 1 };
eval(fs.readFileSync('E:/redstone_nn/03_phase3_quantized_network/viewer/plane_sim.html', 'utf8')
  .match(/<script>([\s\S]*)<\/script>/)[1].replace(/document\.querySelectorAll[\s\S]*$/, ''));

const REF = JSON.parse(fs.readFileSync('E:/redstone_nn/03_phase3_quantized_network/results/py_compile_reference.json', 'utf8'));
const mj = JSON.parse(fs.readFileSync('E:/redstone_nn/02_phase2_avatar/webapp/pca22_dim4w64L1_model.json', 'utf8'));
const L1 = mj.gen.layers[0].W;

// 把 in_bits 也当作 b 的版本（= compiler.py 的本意：输入位宽=权重位宽）
function compileTied(Wm, b) {
  const M = Wm.length, N = Wm[0].length;
  const d = new Array(N).fill(7 / (1 << b));          // delta 跟着 b 变
  const sig = [];
  for (let i = 0; i < M; i++) {
    let g = 0;
    for (let j = 0; j < N; j++) g = Math.max(g, Math.abs(Wm[i][j]) * d[j]);
    sig.push(882 / (g * (1 << (b - 2))));
  }
  const U = [];
  for (let i = 0; i < M; i++) {
    const row = [];
    for (let j = 0; j < N; j++) {
      const r = [];
      for (let k = 0; k < b; k++) r.push(rint(sig[i] * Wm[i][j] * d[j] * (1 << k)));
      row.push(r);
    }
    U.push(row);
  }
  return U;
}

console.log('用"输入位宽=权重位宽"的写法复算，看能否与 Python 的 b=9..12 对上：');
for (const bs of ['9', '10', '11', '12']) {
  const b = +bs, py = REF.layers.L1[bs].W;
  const js = compileTied(L1, b);
  let diff = 0, worst = 0;
  for (let i = 0; i < py.length; i++) for (let j = 0; j < py[i].length; j++) for (let k = 0; k < b; k++) {
    if (py[i][j][k] !== js[i][j][k]) { diff++; worst = Math.max(worst, Math.abs(py[i][j][k] - js[i][j][k])); }
  }
  const total = py.length * py[0].length * b;
  console.log(`  b=${b}: ${diff === 0 ? '完全一致 ✓' : `${diff}/${total} 不同，最大差 ${worst}`}`);
}

// 现在这款（IN_BITS=8 固定、b 独立）的 MSB 权重规模
console.log('\n本仿真器（输入 8 位固定，b 可调）各 b 的 |W| 上界：');
for (const b of [8, 9, 10, 11, 12]) {
  const U = compile(L1, b);
  let top = 0, non = 0;
  for (const row of U) for (const c of row) {
    top = Math.max(top, Math.abs(c[b - 1]));
    for (let k = 0; k < b - 1; k++) non = Math.max(non, Math.abs(c[k]));
  }
  console.log(`  b=${b}: 非最高位 ${String(non).padStart(6)}  最高位 ${String(top).padStart(6)}` +
    (non > 882 ? '   ← 超容器上限 882 ✗' : ''));
}
