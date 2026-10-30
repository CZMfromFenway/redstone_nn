// 参数化扫描：node _sweep.js <html路径>
const fs = require('fs');
const path = process.argv[2] || 'E:/redstone_nn/03_phase3_quantized_network/viewer/plane_sim.html';
const ctxProxy = new Proxy({}, { get: (t, k) => (k === 'canvas' ? { width: 0, height: 0 } : () => {}), set: () => true });
const fakeEl = () => ({ value: '8', checked: true, innerHTML: '', getContext: () => ctxProxy, style: {}, width: 0, height: 0 });
global.document = { getElementById: () => fakeEl(), querySelectorAll: () => [] };
global.window = { devicePixelRatio: 1 };
eval(fs.readFileSync(path, 'utf8')
  .match(/<script>([\s\S]*)<\/script>/)[1].replace(/document\.querySelectorAll[\s\S]*$/, ''));

const mj = JSON.parse(fs.readFileSync('E:/redstone_nn/02_phase2_avatar/webapp/pca22_dim4w64L1_model.json', 'utf8'));
const L1 = mj.gen.layers[0].W;
const keys = ['greyUp', 'whiteUp', 'inRev', 'bitRev', 'parSwap'];
let ok = 0, tot = 0;
let legal = 0, badCell = 0;
const byBad = {};
const by = {};
for (const b of [8, 9, 10, 11]) {
  const U = compile(L1, b);
  for (const ch of [0, 3, 8, 20, 31, 63]) for (let m = 0; m < 32; m++) {
    const o = {}; keys.forEach((k, i) => o[k] = !!(m & (1 << i)));
    for (const xs of [[0, 0, 0, 0], [1, 0, 1, 0]]) {
      const B = build(4, b, U, ch, o);
      B.cells.forEach(c => {
        if (c.ik && c.t === 'blk' && c.c.startsWith('col')) {
          const [i, k] = c.ik, W = B.WS[i][k];
          c.torch = (((xs[i] === 1) !== (W > 0))) ? 15 : 0;
        }
      });
      evalBoard(B);
      const rr = []; for (let j = 0; j < B.J; j++) rr.push(B.originZ + (o.bitRev ? 6 * j : -6 * j));
      const pz = (o.whiteUp !== o.bitRev) ? Math.min(...rr) : Math.max(...rr);
      const head = B.cells.get(B.xw + ',' + pz);
      let N = 0; for (let i = 0; i < 4; i++) for (let k = 0; k < b; k++) N += B.WS[i][k] * xs[i] - Math.min(0, B.WS[i][k]);
      const good = head && head.state === N;
      tot++; if (good) ok++;
      const bad = legalCheck(B);
      if (good && bad.length === 0) legal++;
      badCell += bad.length;
      const tag = (o.inRev ? 'i1' : 'i0') + (o.bitRev ? 'b1' : 'b0');
      by[tag] = by[tag] || [0, 0]; by[tag][0] += good ? 1 : 0; by[tag][1]++;
      if (bad.length) {
        byBad[tag] = byBad[tag] || [0, 0];
        byBad[tag][0] += bad.length; byBad[tag][1]++;
      }
    }
  }
}
console.log(`${ok} / ${tot}`);
for (const k of Object.keys(by).sort()) console.log(`  ${k}: ${by[k][0]}/${by[k][1]}`);
console.log(`物理合法性（容器≤897、比较器≤2^31-1、全部解析）: ${legal} / ${tot}`);
if (badCell) {
  console.log(`  ✗ 违规格 ${badCell} 个，出问题的组合分布（违规数/组合数）：`);
  for (const k of Object.keys(byBad).sort()) console.log(`    ${k}: ${byBad[k][0]}/${byBad[k][1]}`);
  process.exitCode = 1;
} else {
  console.log('  ✓ 全部可建造');
}
