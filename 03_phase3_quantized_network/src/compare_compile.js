// 对比 JS 的 compile() 与 Python compiler.py 的逐元素结果
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
const L2 = mj.gen.layers[1].W;

function cmpLayer(name, Wm, refSet, tag) {
  for (const bStr of Object.keys(refSet)) {
    const b = +bStr;
    const py = refSet[bStr].W;
    const js = compile(Wm, b);
    let diff = 0, maxAbs = 0, firstFew = [];
    for (let i = 0; i < py.length; i++) {
      for (let j = 0; j < py[i].length; j++) {
        for (let k = 0; k < b; k++) {
          const a = py[i][j][k], c = js[i][j][k];
          if (a !== c) {
            diff++;
            maxAbs = Math.max(maxAbs, Math.abs(a - c));
            if (firstFew.length < 4) firstFew.push(`ch${i} in${j} bit${k}: py=${a} js=${c}`);
          }
        }
      }
    }
    const total = py.length * py[0].length * b;
    const ok = diff === 0;
    console.log(`  ${name} b=${b}: ${ok ? '一致 ✓' : `不一致 ✗  ${diff}/${total} 个不同，最大差 ${maxAbs}`}`);
    firstFew.forEach(f => console.log('      ' + f));
    // 容器上限
    const maxTop = Math.max(...js.map(r => Math.max(...r.map(c => Math.abs(c[b - 1])))));
    const maxNon = b > 1 ? Math.max(...js.map(r => Math.max(...r.map(c => Math.max(...c.slice(0, b - 1).map(Math.abs)))))) : 0;
    console.log(`      |W| 非最高位最大 ${maxNon}  最高位最大 ${maxTop}` +
      (maxNon > 882 ? '  ← 超容器上限!' : ''));
  }
}

console.log('=== 层1（输入 4 维，delta 均匀 = 7/256）===');
cmpLayer('L1', L1, REF.layers.L1);

console.log('\n=== 层2（输入 64 维，delta 逐通道不等）===');
cmpLayer('L2', L2, REF.layers.L2);
