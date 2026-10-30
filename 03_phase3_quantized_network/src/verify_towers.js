// 火把塔规则在 inRev / bitRev / parSwap 全 32 种组合下的适配性验证。
//   塔的占地不是查表算的，而是从"建好的平面"里读出来的：
//     输入块 = ik=[i,k] 的彩色块 -> (slotCol, blockRow)
//     读入比较器 = rear 正指到这个块的比较器 -> (slotCol, readRow)
//     塔列 = 同列对里的另一列（rx 异或 1）
//   所以 inRev/bitRev/parSwap 改什么，塔就跟着变什么。
// 同时检查：塔新增的每个位置在各通道里都是空气，且塔与塔不重叠。
const fs = require('fs');
const HTML = 'E:/redstone_nn/03_phase3_quantized_network/viewer/plane_sim.html';
const ctxProxy = new Proxy({}, { get: (t, k) => (k === 'canvas' ? { width: 0, height: 0 } : () => {}), set: () => true });
const fakeEl = () => ({ value: '8', checked: true, innerHTML: '', getContext: () => ctxProxy, style: {}, width: 0, height: 0 });
global.document = { getElementById: () => fakeEl(), querySelectorAll: () => [] };
global.window = { devicePixelRatio: 1 };
eval(fs.readFileSync(HTML, 'utf8').match(/<script>([\s\S]*)<\/script>/)[1]
  .replace(/document\.querySelectorAll[\s\S]*$/, ''));

const mj = JSON.parse(fs.readFileSync('E:/redstone_nn/02_phase2_avatar/webapp/pca22_dim4w64L1_model.json', 'utf8'));
const DV = { N: [0, -1], S: [0, 1], E: [1, 0], W: [-1, 0] };   // eval 里的 const 不外泄，自己来一份
const L1 = mj.gen.layers[0].W;
const b = 8, n = 4, NCH = L1.length;
const U = compile(L1, b);
const keys = ['greyUp', 'whiteUp', 'inRev', 'bitRev', 'parSwap'];
const XBIT = 0;

// u 的真实符号：取绝对值最大的那个位（同一输入 8 个位共用）
function trueSign(i) {
  const s = [];
  for (let ch = 0; ch < NCH; ch++) {
    const row = U[ch][i];
    let best = 0;
    for (let k = 1; k < b; k++) if (Math.abs(row[k]) > Math.abs(row[best])) best = k;
    s.push(row[best] > 0 ? 1 : 0);
  }
  return s;
}

let cases = 0, bad = [];
const flipSets = {};
for (let m = 0; m < 32; m++) {
  const o = {}; keys.forEach((k, i) => o[k] = !!(m & (1 << i)));
  const boards = [];
  for (let ch = 0; ch < NCH; ch++) {
    const B = build(n, b, U, ch, o);
    if (m === 0) {
      B.cells.forEach(c => {
        if (c.ik && c.t === 'blk' && c.c.startsWith('col')) {
          const [i, k] = c.ik, W = B.WS[i][k];
          c.torch = ((XBIT >> k) & 1) === 1 !== (W > 0) ? 15 : 0;
        }
      });
      evalBoard(B);
    }
    boards.push(B);
  }
  const B0 = boards[0];
  const occOdd = ch => new Set([...boards[ch].cells.values()].map(c => c.x + ',' + c.z));
  const occEven = ch => new Set([...boards[ch].cells.values()].filter(c => c.t === 'cmp').map(c => c.x + ',' + c.z));

  const towers = [];
  for (let i = 0; i < n; i++) for (let k = 0; k < b; k++) {
    let blk = null;
    B0.cells.forEach(c => { if (c.ik && c.ik[0] === i && c.ik[1] === k && c.t === 'blk') blk = c; });
    if (!blk) { bad.push(`opt${m} (i${i},k${k}) 找不到输入块`); continue; }
    let rd = null;
    B0.cells.forEach(c => {
      if (c.t !== 'cmp') return;
      const d = DV[c.out];
      if (c.x - d[0] === blk.x && c.z - d[1] === blk.z) rd = c;
    });
    if (!rd) { bad.push(`opt${m} (i${i},k${k}) 找不到读入比较器`); continue; }
    if (rd.x !== blk.x) bad.push(`opt${m} (i${i},k${k}) 读入比较器不在同列`);
    const rx = blk.x - B0.originX;
    const otherRx = rx ^ 1;
    const otherX = otherRx + B0.originX;
    towers.push({ i, k, slotCol: blk.x, otherCol: otherX, blockRow: blk.z, readRow: rd.z });
  }

  // 检查
  const seen = new Map();
  for (const t of towers) {
    for (const ch of [0, 1, Math.floor(NCH / 2), NCH - 1]) {
      const od = occOdd(ch), ev = occEven(ch);
      const tap = t.slotCol + ',' + t.blockRow;
      if (ev.has(tap)) bad.push(`opt${m} ch${ch} (i${t.i},k${t.k}) tap 位置偶层非空`);
      for (const p of [t.otherCol + ',' + t.blockRow, t.otherCol + ',' + t.readRow]) {
        if (od.has(p)) bad.push(`opt${m} ch${ch} (i${t.i},k${t.k}) ${p} 奇层非空`);
        if (ev.has(p)) bad.push(`opt${m} ch${ch} (i${t.i},k${t.k}) ${p} 偶层非空`);
      }
    }
    for (const p of [t.slotCol + ',' + t.blockRow, t.otherCol + ',' + t.blockRow,
                     t.otherCol + ',' + t.readRow]) {
      const key = p;
      if (seen.has(key) && seen.get(key) !== t.i + ',' + t.k)
        bad.push(`opt${m} 塔重叠 ${key}: (${seen.get(key)}) 与 (${t.i},${t.k})`);
      seen.set(key, t.i + ',' + t.k);
    }
    if (Math.abs(t.readRow - t.blockRow) !== 1)
      bad.push(`opt${m} (i${t.i},k${t.k}) 行距 ${Math.abs(t.readRow - t.blockRow)} ≠ 1`);
  }
  // 翻相集合是否与位序无关（同一输入的 8 个位应完全一致）
  for (let i = 0; i < n; i++) {
    const s = trueSign(i);
    const f = []; for (let ch = 0; ch < NCH - 1; ch++) if (s[ch + 1] !== s[ch]) f.push(ch);
    flipSets[i] = f.join(',');
  }
  cases++;
}

console.log(`覆盖 ${cases} 种选项组合（greyUp×whiteUp×inRev×bitRev×parSwap）`);
if (bad.length) {
  console.log(`✗ ${bad.length} 个问题，前 10：`);
  bad.slice(0, 10).forEach(x => console.log('   ' + x));
  process.exitCode = 1;
} else {
  console.log('✓ 全部组合：塔的 3 个格子都是空气、行距=1、塔与塔不重叠');
}
console.log('同一输入 8 个位的翻相集合（应与位序无关）：');
for (let i = 0; i < 4; i++) {
  const f = flipSets[i].split(',').filter(Boolean);
  console.log(`  输入${i}: ${f.length} 处  ${flipSets[i]}`);
}
