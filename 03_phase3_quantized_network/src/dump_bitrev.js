// 为「位权镜像」出对比图：bitRev=0 vs bitRev=1
const fs = require('fs');
const ctxProxy = new Proxy({}, { get: (t, k) => (k === 'canvas' ? { width: 0, height: 0 } : () => {}), set: () => true });
const fakeEl = () => ({ value: '8', checked: true, innerHTML: '', getContext: () => ctxProxy, style: {}, width: 0, height: 0 });
global.document = { getElementById: () => fakeEl(), querySelectorAll: () => [] };
global.window = { devicePixelRatio: 1 };
eval(fs.readFileSync('E:/redstone_nn/03_phase3_quantized_network/viewer/plane_sim.html', 'utf8')
  .match(/<script>([\s\S]*)<\/script>/)[1].replace(/document\.querySelectorAll[\s\S]*$/, ''));

const mj = JSON.parse(fs.readFileSync('E:/redstone_nn/02_phase2_avatar/webapp/pca22_dim4w64L1_model.json', 'utf8'));
const U = compile(mj.gen.layers[0].W, 8);
const xs = [1, 0, 1, 0], ch = 1;
const out = {};
for (const br of [0, 1]) {
  const o = { greyUp: true, whiteUp: true, inRev: false, bitRev: !!br, parSwap: false };
  const B = build(4, 8, U, ch, o);
  B.cells.forEach(c => {
    if (c.ik && c.t === 'blk' && c.c.startsWith('col')) {
      const [i, k] = c.ik, W = B.WS[i][k];
      c.torch = (((xs[i] === 1) !== (W > 0))) ? 15 : 0;
    }
  });
  evalBoard(B);
  const oZ = B.originZ;
  const rr = []; for (let j = 0; j < B.J; j++) rr.push(oZ + (o.bitRev ? 6 * j : -6 * j));
  const pz = (o.whiteUp !== o.bitRev) ? Math.min(...rr) : Math.max(...rr);
  const head = B.cells.get(B.xw + ',' + pz);
  let N = 0; for (let i = 0; i < 4; i++) for (let k = 0; k < 8; k++) N += B.WS[i][k] * xs[i] - Math.min(0, B.WS[i][k]);
  const arr = []; B.cells.forEach(c => arr.push({ x: c.x, z: c.z, t: c.t, c: c.c, out: c.out, mode: c.mode,
    st: c.state, v: c.val, src: c.src ? 1 : 0, ik: c.ik || null, torch: c.torch || 0 }));
  out['ir' + br] = { W: B.W, L: B.L, cells: arr, oX: B.originX, oZ: B.originZ, xw: B.xw,
    head: head ? head.state : null, N, S: B.S.map(r => r[7]), xwRow: pz };
  console.log(`bitRev=${br}: 链行=${rr.join('/')} 端口=${pz} head=${out['ir'+br].head} N=${N} ${out['ir'+br].head===N?'✓':'✗'}`);
}
fs.writeFileSync('E:/redstone_nn/03_phase3_quantized_network/results/sim_cmp_ch1.json', JSON.stringify(out));
