// 用 Node 跑 sandbox.html 里的数学，和 Python 参考对账（DOM 用桩）。
// 用法：node tools/_sandbox_selftest.js
const fs = require('fs');
const path = require('path');

const htmlPath = path.join(__dirname, '..', 'webapp', 'sandbox.html');
const html = fs.readFileSync(htmlPath, 'utf8');
const m = html.match(/<script>([\s\S]*)<\/script>/);
if (!m) throw new Error('no script');
const js = m[1].replace(/<\/script>[\s\S]*$/, '');

const mkCtx = () => ({ fillRect: () => {}, clearRect: () => {}, createImageData: (w, h) => ({ data: new Uint8ClampedArray(w * h * 4) }), putImageData: () => {} });
const mkEl = (id) => ({
  id, value: '0', checked: true, textContent: '', innerHTML: '',
  style: {}, dataset: {}, width: 8, height: 8,
  addEventListener: () => {}, appendChild: () => {},
  getContext: () => mkCtx(),
  firstElementChild: { style: {} },
});
const els = {};
const sliderDefaults = { p0: '0.5', p1: '-1.0', p2: '0.25', p3: '1.5', cands: '4', bn: '64' };
global.document = {
  getElementById: (id) => {
    if (!els[id]) { els[id] = mkEl(id); if (sliderDefaults[id]) els[id].value = sliderDefaults[id]; }
    return els[id];
  },
  createElement: () => mkEl('tmp'),
  addEventListener: () => {},
  querySelectorAll: () => [],
};
global.window = global;
global.Math.random = () => 0.5;   // 确定性，避免随机影响自检

const apiSrc = js + '\nglobalThis.__st = selfTest();\nglobalThis.__z = [0.5,-1.0,0.25,1.5];';
eval(apiSrc);

const st = globalThis.__st;
const ref = JSON.parse(fs.readFileSync(path.join(__dirname, 'python_reference.json'), 'utf8'));
console.log('JS  selfTest:', JSON.stringify(st).slice(0, 260));
console.log('参考:', JSON.stringify(ref).slice(0, 260));

function cmp(name, a, b, tol) {
  if (Array.isArray(a)) {
    let bad = 0;
    for (let i = 0; i < b.length; i++) if (Math.abs(a[i] - b[i]) > tol) bad++;
    console.log(`  ${name}: ${bad === 0 ? 'OK' : 'MISMATCH ' + bad + ' 项'}`);
    return bad === 0;
  }
  const ok = Math.abs(a - b) <= tol;
  console.log(`  ${name}: ${ok ? 'OK' : 'MISMATCH'} (JS ${a} vs PY ${b})`);
  return ok;
}
console.log('对账：');
let ok = true;
ok = cmp('灰度级', st.levels, ref.levels, 0) && ok;
ok = cmp('调色板索引', st.idx, ref.idx, 0) && ok;
ok = cmp('打分器', st.score, ref.score, 1e-4) && ok;
console.log(ok ? 'SANDBOX_SELFTEST_OK' : 'SANDBOX_SELFTEST_FAIL');
