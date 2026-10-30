// 导出整块板（含数值）为 JSON，供 Python 写 .schem 与测试样例
const fs = require('fs');
const ctxProxy = new Proxy({}, { get: (t, k) => (k === 'canvas' ? { width: 0, height: 0 } : () => {}), set: () => true });
const fakeEl = () => ({ value: '8', checked: true, innerHTML: '', getContext: () => ctxProxy, style: {}, width: 0, height: 0 });
global.document = { getElementById: () => fakeEl(), querySelectorAll: () => [] };
global.window = { devicePixelRatio: 1 };
eval(fs.readFileSync('E:/redstone_nn/03_phase3_quantized_network/viewer/plane_sim.html', 'utf8')
  .match(/<script>([\s\S]*)<\/script>/)[1].replace(/document\.querySelectorAll[\s\S]*$/, ''));

const mj = JSON.parse(fs.readFileSync('E:/redstone_nn/02_phase2_avatar/webapp/pca22_dim4w64L1_model.json', 'utf8'));
const B8 = compile(mj.gen.layers[0].W, 8);
const CH = 0, b = 8;
const OPT = { greyUp: true, whiteUp: true, inRev: false, bitRev: false, parSwap: false };

// 测试用例：每维的火把方案（x=0/1）
const CASES = [
  { name: 'C0_全0', x: [0, 0, 0, 0] },
  { name: 'C1_全1', x: [1, 1, 1, 1] },
  { name: 'C2_1010', x: [1, 0, 1, 0] },
  { name: 'C3_0101', x: [0, 1, 0, 1] },
];

const out = { meta: { ch: CH, b, opt: OPT, model: 'pca22_dim4w64L1' }, cases: [] };

for (const c of CASES) {
  const B = build(4, b, B8, CH, OPT);
  // 火把：on ⟺ (x==1) != (W>0)
  B.cells.forEach(k => {
    if (k.ik && k.t === 'blk' && k.c.startsWith('col')) {
      const [i, bit] = k.ik, W = B.WS[i][bit];
      k.torch = (((c.x[i] === 1) !== (W > 0))) ? 15 : 0;
    }
  });
  evalBoard(B);
  const rr = []; for (let j = 0; j < B.J; j++) rr.push(B.originZ + (OPT.bitRev ? 6 * j : -6 * j));
  const pz = (OPT.whiteUp !== OPT.bitRev) ? Math.min(...rr) : Math.max(...rr);
  const headCell = B.cells.get(B.xw + ',' + pz);
  let N = 0; for (let i = 0; i < 4; i++) for (let k = 0; k < b; k++) N += B.WS[i][k] * c.x[i] - Math.min(0, B.WS[i][k]);

  const cells = [];
  B.cells.forEach(k => cells.push({
    x: k.x, z: k.z, t: k.t, c: k.c || null, out: k.out || null,
    mode: k.mode || null, st: k.state === undefined ? null : k.state,
    v: k.val === undefined ? null : k.val, ik: k.ik || null, src: k.src ? 1 : 0,
    torch: k.torch || 0
  }));

  out.cases.push({
    name: c.name, x: c.x, N, head: headCell ? headCell.state : null,
    headPos: [B.xw, pz], W: B.W, L: B.L, J: B.J, K: B.K, W0: B.W0,
    originX: B.originX, originZ: B.originZ, chainRows: rr,
    S: B.S.map(r => r.slice()), WS: B.WS.map(r => r.slice()),
    cells
  });
}

// 只把第一个用例的方块布局写进 schem（布局与用例无关，只有火把不同）
fs.writeFileSync('E:/redstone_nn/03_phase3_quantized_network/results/board_dump.json', JSON.stringify(out));
console.log(`导出 ${out.cases.length} 个用例`);
for (const c of out.cases) console.log(`  ${c.name}  x=${c.x.join('')}  head=${c.head}  N=${c.N}  ${c.head === c.N ? '✓' : '✗'}`);
