// Verify the colour labelling page end to end: model loads, faces render, all five grades post
// with the right scheme and round, and the payload satisfies the server's constraints.
const fs = require('fs'), path = require('path');
const WEB = path.join(__dirname, '..', 'webapp');
const html = fs.readFileSync(path.join(WEB, 'label_colour.html'), 'utf8');
let code = html.match(/<script>([\s\S]*?)<\/script>/)[1];

function mkEl(id) {
  const el = { id, children: [], style: { cssText: '' }, dataset: {}, _t: '', _h: '',
    appendChild(c) { this.children.push(c); return c; },
    addEventListener() {}, removeEventListener() {},
    querySelectorAll() { return []; }, querySelector() { return null; },
    classList: { add() {}, remove() {}, contains() { return false; }, toggle() {} },
    getContext() {
      return {
        createImageData: (w, h) => ({ data: new Uint8ClampedArray(w * h * 4), width: w, height: h }),
        putImageData() {}, clearRect() {}, drawImage() {}, fillRect() {},
        imageSmoothingEnabled: false
      };
    },
    set innerHTML(v) { this._h = v; this.children = []; }, get innerHTML() { return this._h; },
    set textContent(v) { this._t = v; }, get textContent() { return this._t; },
    width: 8, height: 8 };
  return el;
}
const byId = {};
const gradesBox = mkEl('gradesbox');
global.document = {
  createElement: () => mkEl('new'),
  getElementById: id => (byId[id] ||= mkEl(id)),
  querySelector: sel => (sel === '.grades' ? gradesBox : mkEl('q')),
  querySelectorAll: () => [], addEventListener() {}
};
global.window = { addEventListener() {}, location: { search: '?round=6' } };
global.location = global.window.location;
global.Math.random = (() => { let s = 12345; return () => { s = (s * 1103515245 + 12345) & 0x7fffffff; return s / 0x7fffffff; }; })();

const model = JSON.parse(fs.readFileSync(path.join(WEB, 'colour_e6_model.json'), 'utf8'));
let posted = [];
global.fetch = async (u, opt) => {
  const k = String(u).split('?')[0].split('/').pop();
  if (k === 'label' && opt && opt.method === 'POST') {
    posted.push(JSON.parse(opt.body));
    return { ok: true, status: 200, json: async () => ({ ok: true, total: posted.length }) };
  }
  if (k.startsWith('labelstats'))
    return { ok: true, status: 200, json: async () => ({ total: 0, counts: [0,0,0,0,0] }) };
  if (k === 'colour_e6_model.json')
    return { ok: true, status: 200, json: async () => model };
  return { ok: false, status: 404, json: async () => ({}) };
};

code = code.replace(/\(async function init\(\)\{[\s\S]*$/,
  '(async function init(){\n' +
  '  const r = await fetch("colour_e6_model.json");\n' +
  '  loadModel(await r.json());\n' +
  '  nextFace(); renderStats();\n' +
  '  window.LABEL = {getState:()=>({cur,curPcs,stats,kin:KIN,model:MODEL_KEY}),' +
  'grade:g=>grade(g), next:()=>nextFace()};\n' +
  '})();\n');
new Function(code)();

(async () => {
  await new Promise(r => setTimeout(r, 600));
  const L = window.LABEL;
  const st = L.getState();
  const labels = gradesBox.children.map(b => (b._h || '').replace(/\s*<kbd>.*$/, '').trim());
  console.log('buttons        :', JSON.stringify(labels));
  console.log('input dim      :', st.kin);
  console.log('faces rendered :', st.cur ? st.cur.length + ' RGB values' : 'NONE');

  const sample = st.cur ? st.cur.slice(0, 6).map(v => +v.toFixed(2)) : [];
  console.log('sample output  :', JSON.stringify(sample));

  for (const g of [4, 3, 2, 1, 0]) { L.next(); await L.grade(g); }
  const p = posted[0];
  const checks = {
    'model has 8 inputs': st.kin === 8,
    'produces 96 RGB values': st.cur && st.cur.length === 96,
    'outputs are 0..15': st.cur && st.cur.every(v => v >= 0 && v <= 15),
    'five buttons in the right order':
      JSON.stringify(labels) === JSON.stringify(['类二次元','类游戏','类三次元','中','差']),
    'posts 5 grades': posted.length === 5,
    'pca is exactly 22 numbers (server constraint)': p && p.pca.length === 22,
    'scheme=style recorded': p && p.scheme === 'style',
    'round=6 recorded': p && p.round === '6',
    'model key is colour_e6': p && p.model === 'colour_e6',
    'predicted_levels keeps the 96 RGB values': p && p.predicted_levels.length === 96,
    'pca carries the 8 colour components': p && p.pca.slice(0, 8).some(v => v !== 0)
  };
  console.log();
  let ok = true;
  for (const [k, v] of Object.entries(checks)) {
    console.log('  ' + (v ? 'PASS' : 'FAIL') + '  ' + k);
    if (!v) ok = false;
  }
  if (!ok) { console.error('\nFAIL'); process.exit(1); }
  console.log('\nPASS colour labelling works end to end.');
  console.log('COLOUR_LABEL_READY');
})();
