// Render exactly what the page will show, using the page's own decode path, and emit the
// levels as JSON so Python can turn them into a PNG. This is the end-to-end proof: not the
// PyTorch checkpoint, not a reimplementation -- the actual label.html code.
const fs = require('fs'), path = require('path');
const WEB = path.join(__dirname, '..', 'webapp');
const html = fs.readFileSync(path.join(WEB, 'label.html'), 'utf8');
let code = html.match(/<script>([\s\S]*?)<\/script>/)[1];

function mkEl(id) {
  const el = { id, children: [], style: {}, dataset: {}, _t: '', _h: '',
    appendChild(c) { this.children.push(c); return c; }, addEventListener() {},
    removeEventListener() {}, querySelectorAll() { return []; }, querySelector() { return null; },
    classList: { add() {}, remove() {}, contains() { return false; }, toggle() {} },
    getContext() { return { createImageData: (w, h) => ({ data: new Uint8ClampedArray(w * h * 4) }),
      putImageData() {}, clearRect() {}, fillRect() {}, drawImage() {} }; },
    set innerHTML(v) { this._h = v; this.children = []; }, get innerHTML() { return this._h; },
    set textContent(v) { this._t = v; }, get textContent() { return this._t; },
    width: 8, height: 8 };
  return el;
}
const byId = {};
global.document = {
  createElement: () => mkEl('new'),
  getElementById: id => (byId[id] ||= mkEl(id)),
  querySelectorAll: () => [], querySelector: () => mkEl('q'), addEventListener() {}
};
global.window = { addEventListener() {}, location: { search: '?mix=1' } };
global.location = global.window.location;

const FILES = ['pca22_dim4_model.json', 'pca22_dim4x_model.json', 'pca22_dim22_model.json',
               'dimsweep_manifest.json', 'preference_reward.json'];
const F = {};
for (const f of FILES) {
  const p = path.join(WEB, f);
  if (fs.existsSync(p)) F[f] = JSON.parse(fs.readFileSync(p, 'utf8'));
}
global.fetch = async (u) => {
  const k = String(u).split('?')[0].split('/').pop();
  if (k === 'labelstats') {
    return { ok: true, status: 200, json: async () => ({ total: 0, counts: [0,0,0,0], byModel: {} }) };
  }
  if (F[k]) return { ok: true, status: 200, json: async () => F[k] };
  return { ok: false, status: 404, json: async () => ({}) };
};
global.console.warn = () => {};
code = code.replace(/init\(\)\.catch\([\s\S]*?}\)\.;\s*$/, '');
new Function(code)();

(async () => {
  await new Promise(r => setTimeout(r, 1000));
  const L = global.window.LABEL;
  const out = {};
  const PER = 24;
  for (let i = 0; i < PER * 3 + 30; i++) {
    L.next();
    const st = L.getState();
    const key = st.modelKey;
    (out[key] ||= []);
    if (out[key].length < PER) {
      out[key].push(Array.from(global.window._lastLevels).map(v => Math.round(v)));
    }
  }
  fs.writeFileSync(path.join(__dirname, '..', 'results', 'dimsweep', 'page_render.json'),
                   JSON.stringify(out));
  console.log('captured per arm:', Object.keys(out).map(k => k + '=' + out[k].length).join('  '));
  console.log('PAGE_RENDER_DONE');
})();
