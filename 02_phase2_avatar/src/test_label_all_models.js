// Does the labelling page load and switch between all six architectures, default to the
// 22->128->128->32 model, and produce valid samples from each?
const fs = require('fs'), path = require('path');
const WEB = path.join(__dirname, '..', 'webapp');
const html = fs.readFileSync(path.join(WEB, 'label.html'), 'utf8');
let code = html.match(/<script>([\s\S]*?)<\/script>/)[1];

function mk() {
  return { children: [], style: {}, dataset: {}, _t: '', _h: '',
    appendChild(c){this.children.push(c);return c;}, addEventListener(){}, removeEventListener(){},
    querySelectorAll(){return[];}, querySelector(){return null;},
    classList:{add(){},remove(){},contains(){return false;},toggle(){}},
    getContext(){return{createImageData:(w,h)=>({data:new Uint8ClampedArray(w*h*4)}),putImageData(){},clearRect(){}};},
    set innerHTML(v){this._h=v;this.children=[];}, get innerHTML(){return this._h;},
    set textContent(v){this._t=v;}, get textContent(){return this._t;}, width:8, height:8 };
}
const byId = {};
global.document = { createElement: mk, getElementById: id => (byId[id] ||= mk()),
  querySelectorAll: () => [], querySelector: () => mk() };
global.window = { addEventListener(){} };

const NAMES = ['pca22_lam5_model.json','pca22_gray32_model.json','pca22_color16_model.json',
               'pca22_w128_128_model.json','pca22_best128_model.json','pca22_w32_model.json'];
const F = {};
for (const f of NAMES) {
  const p = path.join(WEB, f);
  if (fs.existsSync(p)) F[f] = JSON.parse(fs.readFileSync(p, 'utf8'));
}
global.fetch = async u => {
  const k = String(u).split('?')[0].split('/').pop();
  if (F[k]) return {ok:true,status:200,json:async()=>F[k]};
  if (k === 'labelstats') return {ok:true,status:200,json:async()=>({total:159,counts:[10,42,86,21]})};
  return {ok:false,status:404,json:async()=>({})};
};
global.console.warn = () => {};

code = code.replace(/init\(\)\.catch\([\s\S]*?\}\);\s*$/,'');
new Function(code)();

(async () => {
  await new Promise(r => setTimeout(r, 700));
  const st = global.window.LABEL.getState();
  console.log('default architecture :', st.modelKey);
  if (st.modelKey !== 'w128') { console.error('FAIL default should be w128'); process.exit(1); }
  console.log('loaded architectures :', st.available.join(', '));
  if (st.available.length !== NAMES.length) {
    console.error(`FAIL expected ${NAMES.length}, got ${st.available.length}`);
    process.exit(1);
  }
  for (const k of st.available) {
    global.window.LABEL.switchModel(k);
    const s2 = global.window.LABEL.getState();
    if (s2.modelKey !== k) { console.error('FAIL switch to', k); process.exit(1); }
    const mx = Math.max(...Array.from(s2.cur, Math.abs));
    if (!(mx <= 3.5001)) { console.error('FAIL out of range on', k, mx); process.exit(1); }
  }
  console.log('PASS all six architectures load, switch, and sample in range');
  console.log('W128_LABEL_READY');
})();
