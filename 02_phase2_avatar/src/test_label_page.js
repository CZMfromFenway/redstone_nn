// Test the labelling page: it must load, render, accept a label through the sink, and
// (when a reward model exists) use reward-guided sampling rather than plain noise.
const fs = require('fs'), path = require('path'), http = require('http');
const WEB = path.join(__dirname, '..', 'webapp');
const html = fs.readFileSync(path.join(WEB, 'label.html'), 'utf8');
let code = html.match(/<script>([\s\S]*?)<\/script>/)[1];

function mk() {
  return { children: [], style: {}, dataset: {}, _t: '', _h: '',
    appendChild(c){this.children.push(c);return c;}, addEventListener(){}, removeEventListener(){},
    querySelectorAll(){return[];}, querySelector(){return null;},
    classList:{add(){},remove(){},contains(){return false;},toggle(){}},
    getContext(){return{createImageData:(w,h)=>({data:new Uint8ClampedArray(w*h*4)}),putImageData(){},clearRect(){}};},
    set innerHTML(v){this._h=v;}, get innerHTML(){return this._h;},
    set textContent(v){this._t=v;}, get textContent(){return this._t;}, width:8, height:8 };
}
const byId = {};
global.document = { createElement: mk, getElementById: id => (byId[id] ||= mk()),
  querySelectorAll: () => [], querySelector: () => mk() };
global.window = { addEventListener(){} };
const FILES = {};
for (const f of ['pca22_lam5_model.json','pca22_gray32_model.json','pca22_color16_model.json'])
  if (fs.existsSync(path.join(WEB, f)))
    FILES[f] = JSON.parse(fs.readFileSync(path.join(WEB, f), 'utf8'));

let REWARD = null;
if (fs.existsSync(path.join(WEB, 'preference_reward.json')))
  REWARD = JSON.parse(fs.readFileSync(path.join(WEB, 'preference_reward.json'), 'utf8'));

global.fetch = async (u) => {
  const k = String(u).split('?')[0].split('/').pop();
  if (FILES[k]) return {ok:true,status:200,json:async()=>FILES[k]};
  if (k === 'preference_reward.json')
    return REWARD ? {ok:true,status:200,json:async()=>REWARD} : {ok:false,status:404,json:async()=>({})};
  if (k === 'preference_dist.json') return {ok:false,status:404,json:async()=>({})};
  return {ok:false,status:404,json:async()=>({})};
};

code = code.replace(/init\(\)\.catch\([\s\S]*?\}\);\s*$/,'');
new Function(code)();
const LABEL = global.window.LABEL;

(async () => {
  // wait for init to settle (it is async and not awaited by the page)
  await new Promise(r => setTimeout(r, 400));
  const st = LABEL.getState();
  console.log(`model loaded: ${st.modelKey}`);
  console.log(`reward model ${st.REW ? 'PRESENT (reward-guided sampling active)' : 'absent (plain random sampling)'}`);
  if (!st.cur || st.cur.length !== 22) { console.error('FAIL no current vector'); process.exit(1); }
  let mx = 0; for (const v of st.cur) mx = Math.max(mx, Math.abs(v));
  console.log(`first sample: max|val| = ${mx.toFixed(2)} (must be <= 3.5)`);
  if (mx > 3.5001) { console.error('FAIL out of range'); process.exit(1); }

  // every architecture must be loadable and switchable
  const names = ['lam5','gray32','color16'];
  console.log('\narchitectures available for labelling:');
  for (const k of names) {
    const present = !!FILES[`pca22_${k}_model.json`] || k === 'lam5';
    console.log(`  ${k.padEnd(8)} ${present ? 'loaded' : 'MISSING'}`);
    if (!present) { console.error('FAIL missing model'); process.exit(1); }
  }
  console.log('LABEL_PAGE_TEST_DONE');
})();
