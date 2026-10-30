// 层1 平面，白环预先减去 (P−1)：头部输出直接 = max(0, N−P+1)，即 RELU 除法器的输入 V。
// 导出 → results/planes64_relu.json（供 schem_full.py 堆叠）
// 用法：node src/dump_planes_relu.js
const fs = require('fs');
const HTML = 'E:/redstone_nn/03_phase3_quantized_network/viewer/plane_sim.html';
const ctxProxy = new Proxy({}, { get: (t, k) => (k === 'canvas' ? { width: 0, height: 0 } : () => {}), set: () => true });
const fakeEl = () => ({ value: '8', checked: true, innerHTML: '', getContext: () => ctxProxy, style: {}, width: 0, height: 0 });
global.document = { getElementById: () => fakeEl(), querySelectorAll: () => [] };
global.window = { devicePixelRatio: 1 };
eval(fs.readFileSync(HTML, 'utf8').match(/<script>([\s\S]*)<\/script>/)[1]
  .replace(/document\.querySelectorAll[\s\S]*$/, ''));

const mj = JSON.parse(fs.readFileSync('E:/redstone_nn/02_phase2_avatar/webapp/pca22_dim4w64L1_model.json', 'utf8'));
const NL = JSON.parse(fs.readFileSync('E:/redstone_nn/03_phase3_quantized_network/results/netlist_grayscale.json', 'utf8'));
const L1 = mj.gen.layers[0].W;
const b = 8, n = 4, NCH = L1.length;
const U = compile(L1, b);
const P = NL.layers[0].P, CVAL = NL.layers[0].c;
const XSET = [[0, 0, 0, 0], [255, 255, 255, 255], [186, 37, 204, 91],
              [7, 63, 200, 17], [128, 0, 0, 128], [1, 2, 4, 8]];

function expectN(WS, X) {
  let N = 0;
  for (let i = 0; i < n; i++) for (let k = 0; k < b; k++)
    N += WS[i][k] * ((X[i] >> k) & 1) - Math.min(0, WS[i][k]);
  return N;
}

let ok = 0, tot = 0, legBad = 0;
const rows = [];
for (let ch = 0; ch < NCH; ch++) {
  const o = { greyUp: true, whiteUp: true, inRev: false, bitRev: false, parSwap: false,
              subP: P[ch] - 1 };
  let cellDump = null, headAt0 = null;
  for (const X of XSET) {
    const B = build(n, b, U, ch, o);
    B.cells.forEach(c => {
      if (c.ik && c.t === 'blk' && c.c.startsWith('col')) {
        const [i, k] = c.ik, W = B.WS[i][k];
        c.torch = ((((X[i] >> k) & 1) === 1) !== (W > 0)) ? 15 : 0;
      }
    });
    evalBoard(B);
    const rr = []; for (let j = 0; j < B.J; j++) rr.push(B.originZ + (o.bitRev ? 6 * j : -6 * j));
    const pz = (o.whiteUp !== o.bitRev) ? Math.min(...rr) : Math.max(...rr);
    const head = B.cells.get(B.xw + ',' + pz);
    const N = expectN(B.WS, X);
    const want = Math.max(0, N - P[ch] + 1);
    tot++; if (head && head.state === want) ok++;
    else if (tot - ok <= 4) console.log(`  ✗ ch${ch} X=${X} head=${head && head.state} 期望 ${want}`);
    if (legalCheck(B).length) legBad++;
    if (X === XSET[0]) {
      headAt0 = head ? head.state : null;
      const arr = [];
      B.cells.forEach(c => arr.push({
        x: c.x, z: c.z, t: c.t, c: c.c || null, out: c.out || null,
        mode: c.mode || null, st: c.state === undefined ? null : c.state,
        v: c.val === undefined ? null : c.val, ik: c.ik || null
      }));
      cellDump = { ch, W: B.W, L: B.L, K: B.K, W0: B.W0, head: headAt0, xw: B.xw, pz,
                   P: P[ch], c: CVAL[ch], subP: P[ch] - 1, cells: arr };
    }
  }
  rows.push(cellDump);
}

console.log(`白环 = J·K − 15nb − (P−1)；头部应等于 max(0, N−P+1)`);
console.log(`  头部校验：${ok}/${tot}  ${ok === tot ? '✓' : '✗'}`);
console.log(`  物理合法性：${legBad === 0 ? '✓ 全部通过' : '✗ ' + legBad + ' 组有问题'}`);
/**
 * 头部的范围：V = max(0, N−P+1)
 *   N ∈ [0, Nmax]  ⇒  V 最大 = Nmax−P+1，最小 0
 */
const W = rows[0].W, L = rows[0].L;
const bad = rows.filter(r => r.W !== W || r.L !== L);
console.log(`  ${rows.length} 片，尺寸 ${W}x${L}，不一致 ${bad.length} 片`);
fs.writeFileSync('E:/redstone_nn/03_phase3_quantized_network/results/planes64_relu.json',
  JSON.stringify({ W, L, b, n, note: 'white ring = J*K-15nb-(P-1)', planes: rows }));
console.log('已写 results/planes64_relu.json');
