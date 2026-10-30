// Verify the standalone build: no network, all models present, and it runs from the inlined data.
const fs = require('fs'), path = require('path');
const WEB = path.join(__dirname, '..', 'webapp');
const P = path.join(WEB, 'sandbox_standalone.html');
const html = fs.readFileSync(P, 'utf8');

// the file must contain the inlined data and must NOT depend on a server
const hasShim = html.includes('window.fetch = async function(u)');
const hasData = html.includes('const __DATA =');
const scriptCount = (html.match(/<script>/g) || []).length;
console.log('file size      :', (fs.statSync(P).size / 1e6).toFixed(2), 'MB');
console.log('inlined data   :', hasData);
console.log('fetch shim     :', hasShim);
console.log('script blocks  :', scriptCount);

// extract the last <script> block (the page logic) and run it with the shim, no fetch mock
const blocks = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => m[1]);
console.log('blocks found   :', blocks.length);
const shimCode = blocks.find(b => b.includes('const __DATA ='));
const pageCode = blocks.find(b => b.includes("'use strict';"));

function mkEl(id) {
  const el = { id, children: [], style: {}, dataset: {}, className: '', _h: '', _t: '',
    value: '', width: 8, height: 8, type: '', min: '', max: '', step: '', oninput: null,
    appendChild(c) { this.children.push(c); return c; }, addEventListener() {},
    querySelectorAll() { return []; }, querySelector() { return mkEl('q'); },
    classList: { add() {}, remove() {}, contains() { return false; }, toggle() { return false; } },
    getContext() { return { createImageData: (w, h) => ({ data: new Uint8ClampedArray(w*h*4) }),
      putImageData() {}, clearRect() {} }; },
    set innerHTML(v) { this._h = v; this.children = []; }, get innerHTML() { return this._h; },
    set textContent(v) { this._t = v; }, get textContent() { return this._t; } };
  return el;
}
const byId = {};
global.document = { createElement: () => mkEl('new'),
  getElementById: id => (byId[id] ||= mkEl(id)),
  querySelector: () => mkEl('q'), querySelectorAll: () => [], addEventListener() {} };
global.window = { addEventListener() {}, location: { search: '' } };
global.location = global.window.location;
global.performance = { now: () => Date.now() };
// NO fetch mock on purpose: the shim inside the file must be the only source of data
delete global.fetch;

new Function(shimCode)();
if (typeof global.window.fetch !== 'function') throw new Error('shim did not install');
global.fetch = global.window.fetch;
new Function(pageCode)();

(async () => {
  await new Promise(r => setTimeout(r, 2500));
  const S = window.SANDBOX;
  if (!S) { console.error('page never initialised from inlined data'); process.exit(1); }
  console.log();
  console.log('models loaded  :', S.models.length);
  console.log('scorers        : accept=' + S.hasAccept + ' anime=' + S.hasAnime +
              ' merge=' + S.hasMerge);
  const lv = S.levelsFor(new Float32Array([0.3, -0.7, 1.1, -0.2]), '16');
  const mc = S.levelsFor(new Float32Array([0.3, -0.7, 1.1, -0.2]), 'mc11');
  console.log('sample levels  :', lv.join(','));
  console.log('merged levels  :', mc.join(','));

  const checks = {
    'weights are inlined': hasData,
    'fetch shim is present': hasShim,
    'no network needed (init worked without a fetch mock)': S.models.length > 0,
    'all ten models': S.models.length === 10,
    'both scorers inlined': S.hasAccept && S.hasAnime,
    'merge table inlined': S.hasMerge,
    'inference produces 32 levels': lv.length === 32,
    'merge produces a restricted set':
      new Set(mc.map(v => v.toFixed(2))).size <= 11
  };
  console.log();
  let ok = true;
  for (const [k, v] of Object.entries(checks)) {
    console.log('  ' + (v ? 'PASS' : 'FAIL') + '  ' + k);
    if (!v) ok = false;
  }
  if (!ok) { console.error('\nFAIL'); process.exit(1); }
  console.log('\nPASS the single file works with no server and no network.');
  console.log('STANDALONE_READY');
})();
