import io
p = r'E:\redstone_nn\03_phase3_quantized_network\viewer\plane_sim.html'
s = io.open(p, encoding='utf-8').read()

old = """/* ---------- 权重 → 整数 U ---------- */
function compile(Wm,b){
  const M=Wm.length,N=Wm[0].length;
  let maxAbs=[]; for(let i=0;i<M;i++){ let m=0; for(let j=0;j<N;j++) m=Math.max(m,Math.abs(Wm[i][j])); maxAbs.push(m||1); }
  const WREF=882, DELTA=7;
  const U=[];
  for(let i=0;i<M;i++){
    const al=WREF/(maxAbs[i]*DELTA/4);
    const row=[];
    for(let j=0;j<N;j++){
      const r=[];
      for(let k=0;k<b;k++) r.push(Math.round(al*Wm[i][j]*(DELTA/(1<<b))*(1<<k)));
      row.push(r);
    }
    U.push(row);
  }
  return U;
}"""

new = """/* ---------- 权重 → 整数 U ----------
   严格对应 03_phase3_quantized_network/src/compiler.py：
       delta_j = (xmax-xmin)/2^in_bits        （层1 均匀；层2 逐通道）
       g_i     = max_j |w_ij| * delta_j
       sig_i   = w_ref / (g_i * 2^(in_bits-2))
       W_ijk   = rint( sig_i * w_ij * delta_j * 2^k )      ← rint = 四舍六入五取偶
   注意 b 是"权重位宽"，in_bits 是"输入端位宽"，两者不同。
   ------------------------------------------------------------- */
const IN_BITS = 8;          // 模型输入端 8 位
const W_REF   = 882;

function rint(v){           // numpy.rint：.5 取偶
  const f = Math.floor(v), d = v - f;
  if (d > 0.5) return f + 1;
  if (d < 0.5) return f;
  return (f % 2 === 0) ? f : f + 1;
}

function compile(Wm, b, delta){
  const M = Wm.length, N = Wm[0].length;
  const d = delta || new Array(N).fill(7 / (1 << IN_BITS));   // 层1 默认
  const sig = [];
  for (let i = 0; i < M; i++) {
    let g = 0;
    for (let j = 0; j < N; j++) g = Math.max(g, Math.abs(Wm[i][j]) * d[j]);
    sig.push(W_REF / (g * (1 << (IN_BITS - 2))));
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

/* 层1 的输出步长（供层2 当 delta 用），对应 compiler.py 的 d_out = c/sig */
function layer1Delta(Wm, b){
  const U = compile(Wm, b);
  const m = U.length, n = U[0].length;
  const out = [];
  for (let i = 0; i < m; i++) {
    let Cmin = 0, Cmax = 0;
    for (let j = 0; j < n; j++) for (let k = 0; k < b; k++) {
      const v = U[i][j][k];
      if (v < 0) Cmin += v; else Cmax += v;
    }
    const Nmax = Cmax - Cmin;
    // kappa 与 P 只为求 c：c = max(ceil((Nmax-P)/255), 1)
    // P 需要 theta 与 a0，这里用 0 作下界估算 c 的上界并不安全，故直接用 Nmax/255 近似不成立。
    // 改为：把 c 视为 1（当 Nmax ≤ 255）或按 Nmax/255 取整 —— 与 compiler.py 一致需要 P。
    out.push({ Cmin, Cmax, Nmax });
  }
  return out;
}"""
assert old in s
s = s.replace(old, new)
io.open(p, 'w', encoding='utf-8').write(s)
print('compile() rewritten: per-channel delta + banker rounding')
