// 诊断：① 层1 的 32 处差异是不是 .5 平局；② 层2 的 delta 分布
const fs = require('fs');
const REF = JSON.parse(fs.readFileSync('E:/redstone_nn/03_phase3_quantized_network/results/py_compile_reference.json', 'utf8'));
const mj = JSON.parse(fs.readFileSync('E:/redstone_nn/02_phase2_avatar/webapp/pca22_dim4w64L1_model.json', 'utf8'));
const L1 = mj.gen.layers[0].W;
const L2 = mj.gen.layers[1].W;
const XMIN = -3.5, XMAX = 3.5, INB = 8, LIMIT = 882;

const delta1 = new Array(4).fill((XMAX - XMIN) / (1 << INB));

// ---- ① 层1 的 32 处
{
  const py = REF.layers.L1['8'].W;
  const b = 8;
  let ties = 0, nonties = 0, samples = [];
  for (let i = 0; i < L1.length; i++) {
    const maxAbs = Math.max(...L1[i].map(Math.abs)) || 1;
    const al = LIMIT / (maxAbs * 7 / 4);            // JS 的写法
    for (let j = 0; j < L1[i].length; j++) {
      for (let k = 0; k < b; k++) {
        const raw = al * L1[i][j] * (7 / (1 << b)) * (1 << k);
        const js = Math.round(raw);
        if (js !== py[i][j][k]) {
          const frac = Math.abs(raw - Math.trunc(raw));
          if (Math.abs(frac - 0.5) < 1e-12) ties++;
          else { nonties++; if (samples.length < 5) samples.push({ i, j, k, raw, py: py[i][j][k], js }); }
        }
      }
    }
  }
  console.log(`层1 b=8 的差异：.5 平局 ${ties} 处，非平局 ${nonties} 处`);
  samples.forEach(s => console.log('   非平局样例', JSON.stringify(s)));

  // numpy rint = 四舍六入五取偶
  const rint = v => { const f = Math.floor(v), d = v - f;
    if (d > 0.5) return f + 1; if (d < 0.5) return f;
    return (f % 2 === 0) ? f : f + 1; };
  let fixed = 0;
  for (let i = 0; i < L1.length; i++) {
    const maxAbs = Math.max(...L1[i].map(Math.abs)) || 1;
    const al = LIMIT / (maxAbs * 7 / 4);
    for (let j = 0; j < L1[i].length; j++) for (let k = 0; k < b; k++) {
      const raw = al * L1[i][j] * (7 / (1 << b)) * (1 << k);
      if (rint(raw) !== py[i][j][k]) fixed++;
    }
  }
  console.log(`  改用"四舍六入五取偶"后，剩余不一致 ${fixed} 处`);
}

// ---- ② 层2 的 delta 分布
{
  const d = REF.layers.L2['8'].delta_in;
  const mn = Math.min(...d), mx = Math.max(...d), mean = d.reduce((a, x) => a + x, 0) / d.length;
  const span = mx / mn;
  console.log(`\n层2 的输入步长 delta（64 通道）：`);
  console.log(`   min ${mn.toExponential(4)}   max ${mx.toExponential(4)}   均值 ${mean.toExponential(4)}`);
  console.log(`   最大/最小 = ${span.toFixed(2)} 倍   ← JS 用统一的 7，误差可达 ${(span).toFixed(0)} 倍`);
  // 按 Python 的正确做法重算，看 JS 用统一 delta 会差多少
  const py = REF.layers.L2['8'].W;
  const b = 8;
  let worst = 0, cnt = 0;
  for (let i = 0; i < L2.length; i++) {
    const gmax = Math.max(...L2[i].map((w, j) => Math.abs(w) * d[j]));
    const sig = LIMIT / (gmax * (1 << (INB - 2)));
    for (let j = 0; j < L2[i].length; j++) for (let k = 0; k < b; k++) {
      const raw = sig * L2[i][j] * d[j] * (1 << k);   // 正确（逐通道 delta）
      const a = Math.round(raw);
      worst = Math.max(worst, Math.abs(a - py[i][j][k]));
      if (a !== py[i][j][k]) cnt++;
    }
  }
  console.log(`   用"逐通道 delta + Math.round"复算：与 Python 不一致 ${cnt} 处，最大差 ${worst}（应只剩 .5 平局）`);
}
