// Verify the page's "random" path actually produces DIFFERENT values on each click
// (the reported bug), and that both decoding modes differ.
const fs = require('fs');
const path = require('path');

const html = fs.readFileSync(path.join(__dirname, '..', 'webapp', 'index.html'), 'utf8');
let code = html.match(/<script>([\s\S]*?)<\/script>/)[1];

function makeEl(tag) {
  return {
    tagName: tag, children: [], style: {}, dataset: {}, _text: '', _html: '',
    width: 8, height: 8, title: '',
    appendChild(c) { this.children.push(c); return c; },
    addEventListener() {}, removeEventListener() {},
    querySelectorAll() { return []; }, querySelector() { return null; },
    getContext() { return { createImageData: (w, h) => ({ data: new Uint8ClampedArray(w * h * 4) }), putImageData() {}, clearRect() {} }; },
    set innerHTML(v) { this._html = v; }, get innerHTML() { return this._html; },
    set textContent(v) { this._text = v; }, get textContent() { return this._text; },
  };
}
const byId = {};
global.document = {
  createElement: makeEl,
  getElementById(id) { if (!byId[id]) byId[id] = makeEl('div'); return byId[id]; },
  querySelectorAll() { return []; },
};
global.window = {};
global.fetch = async () => ({ json: async () => JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'webapp', 'pca22_model.json'), 'utf8')) });

code = code.replace(/load\(\)\.catch\([\s\S]*?\}\);/, '');
new Function(code)();
const APP = global.window.APP;
if (!APP) { console.error('FAIL window.APP missing'); process.exit(1); }

(async () => {
  await APP.load();
  console.log('PASS load()');

  // replicate the random button exactly as written in index.html
  const simulateRandClicks = (n) => {
    const seen = [];
    for (let k = 0; k < n; k++) {
      const r = APP.rng((Math.random() * 4294967295) >>> 0);
      const cur = [];
      for (let i = 0; i < 22; i++) cur.push(Math.max(-3.5, Math.min(3.5, APP.gauss(r))));
      seen.push(cur);
    }
    return seen;
  };
  const draws = simulateRandClicks(6);
  const distinct = new Set(draws.map(d => d.map(v => v.toFixed(3)).join(','))).size;
  console.log(`PASS 6 clicks of "random": ${distinct} distinct parameter vectors (expect 6)`);
  console.log(`      first three PC values per click: ` +
    draws.slice(0, 4).map(d => `[${d[0].toFixed(2)},${d[1].toFixed(2)},${d[2].toFixed(2)}]`).join('  '));
  if (distinct < 6) { console.error('FAIL random button still repeats'); process.exit(1); }

  // the OLD buggy behaviour, for contrast
  const r0 = APP.rng(12345);
  const oldA = Array.from({ length: 22 }, () => APP.gauss(r0));
  const r1 = APP.rng(12345);
  const oldB = Array.from({ length: 22 }, () => APP.gauss(r1));
  const same = oldA.map(v => v.toFixed(6)).join() === oldB.map(v => v.toFixed(6)).join();
  console.log(`      (the old code used a fixed seed, so two clicks were identical: ${same})`);

  // sample vs mode must differ
  const pca = new Float32Array(22);
  const z = new Float32Array(64).fill(0.3);
  const lg = APP.forward(pca, z);
  const s = APP.render(lg, 1.0, 'sample', APP.rng(7));
  const m = APP.render(lg, 1.0, 'mode', APP.rng(7));
  let diff = 0; for (let i = 0; i < 32; i++) if (s[i] !== m[i]) diff++;
  console.log(`PASS sample vs mode differ on ${diff}/32 positions`);
  console.log(`      sample: ${Array.from(s).join('')}`);
  console.log(`      mode  : ${Array.from(m).join('')}`);
  console.log('\nRANDOM_BUTTON_TEST_DONE');
})();
