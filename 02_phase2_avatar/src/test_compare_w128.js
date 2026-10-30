// Verify the compare page against the w128 reward model.
const fs = require('fs'), path = require('path');
const WEB = path.join(__dirname, '..', 'webapp');
const html = fs.readFileSync(path.join(WEB, 'compare.html'), 'utf8');
let code = html.match(/<script>([\s\S]*?)<\/script>/)[1];

function mk() {
  return { children: [], style: {cssText:''}, dataset: {}, _t: '', _h: '',
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
global.window = { addEventListener(){}, location: { search: '?model=w128' } };
global.location = global.window.location;

const F = {};
for (const f of ['pca22_w128_128_model.json','pca22_lam5_model.json','preference_reward.json'])
  if (fs.existsSync(path.join(WEB, f))) F[f] = JSON.parse(fs.readFileSync(path.join(WEB, f), 'utf8'));
global.fetch = async u => { const k = String(u).split('?')[0].split('/').pop();
  return F[k] ? {ok:true,status:200,json:async()=>F[k]} : {ok:false,status:404,json:async()=>({})}; };
const warnings = [];
global.console.warn = m => { warnings.push(String(m)); };

code = code.replace(/init\(\)\.catch\([\s\S]*?\}\);\s*$/,'');
new Function(code)();

(async () => {
  await new Promise(r => setTimeout(r, 900));
  const st = global.window.CMP.getState();
  console.log('model loaded         :', st.MODEL && st.MODEL.meta.source_run,
              '| multiplies:', st.MODEL && st.MODEL.meta.multiplies);
  if (!st.MODEL) { console.error('FAIL model did not load'); process.exit(1); }
  console.log('reward model present :', !!st.REW,
              st.REW ? `(fitted on ${st.REW.model}, ${st.REW.n_labels} labels, AUC ${st.REW.cv_auc.toFixed(3)})` : '');
  if (!st.REW) { console.error('FAIL reward model should be accepted for w128'); process.exit(1); }
  if (st.REW.model !== 'w128') { console.error('FAIL wrong reward model'); process.exit(1); }

  // the page must render two grids of faces
  global.window.CMP.run();
  const nr = byId['gRand'].children.length, ng = byId['gGuid'].children.length;
  console.log(`grids rendered       : random ${nr}, guided ${ng}`);
  if (nr !== 24 || ng !== 24) { console.error('FAIL grid count'); process.exit(1); }

  const sR = parseFloat(byId['sRand'].textContent);
  const sG = parseFloat(byId['sGuid'].textContent);
  console.log(`mean predicted score : random ${sR.toFixed(3)}, guided ${sG.toFixed(3)}, gain ${(sG-sR).toFixed(3)}`);
  if (!(sG > sR)) { console.error('FAIL guided should score higher'); process.exit(1); }

  // a reward model from a different architecture must be refused
  if (warnings.some(w => w.includes('reward model is for')))
    console.log('note: a mismatched reward model was refused (correct)');

  console.log('\nPASS compare page shows w128 with its own reward model');
  console.log('COMPARE_W128_DONE');
})();
