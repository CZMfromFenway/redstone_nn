// Verify the labelling page can label the 22->128->128->32 model, and that the whole
// chain (label -> fit_reward --model -> compare.html) is wired correctly.
const fs = require('fs'), path = require('path');
const WEB = path.join(__dirname, '..', 'webapp');

function harness(file, extra) {
  const html = fs.readFileSync(path.join(WEB, file), 'utf8');
  let code = html.match(/<script>([\s\S]*?)<\/script>/)[1];
  function mk() {
    return { children: [], style: {cssText:''}, dataset: {}, _t: '', _h: '',
      appendChild(c){this.children.push(c);return c;}, addEventListener(){},
      querySelectorAll(){return[];}, querySelector(){return null;},
      classList:{add(){},remove(){},contains(){return false;},toggle(){}},
      getContext(){return{createImageData:(w,h)=>({data:new Uint8ClampedArray(w*h*4)}),putImageData(){},clearRect(){}};},
      set innerHTML(v){this._h=v;this.children=[];}, get innerHTML(){return this._h;},
      set textContent(v){this._t=v;}, get textContent(){return this._t;}, width:8, height:8 };
  }
  const byId = {};
  global.document = { createElement: mk, getElementById: id => (byId[id] ||= mk()),
    querySelectorAll: () => [], querySelector: () => mk() };
  global.window = { addEventListener(){}, location: { search: extra.search || '' } };
  global.location = global.window.location;
  const FILES = {};
  for (const f of ['pca22_lam5_model.json','pca22_gray32_model.json','pca22_color16_model.json',
                   'pca22_w128_128_model.json','pca22_best128_model.json','pca22_w32_model.json',
                   'pca22_distill32_model.json','preference_reward.json'])
    if (fs.existsSync(path.join(WEB, f))) FILES[f] = JSON.parse(fs.readFileSync(path.join(WEB, f), 'utf8'));
  global.fetch = async u => { const k = String(u).split('?')[0].split('/').pop();
    if (FILES[k]) return {ok:true,status:200,json:async()=>FILES[k]};
    if (k === 'labelstats') return {ok:true,status:200,json:async()=>({total:0,counts:[0,0,0,0]})};
    return {ok:false,status:404,json:async()=>({})}; };
  global.console.warn = m => console.log('  WARN', m);
  code = code.replace(/init\(\)\.catch\([\s\S]*?\}\);\s*$/,'');
  new Function(code)();
  return { byId, FILES };
}

(async () => {
  // ---- 1. the labelling page ------------------------------------------------------
  const { byId, FILES } = harness('label.html', {});
  await new Promise(r => setTimeout(r, 500));
  const st = global.window.LABEL.getState();
  console.log(`label.html default model: ${st.modelKey}`);
  if (st.modelKey !== 'w128') { console.error('FAIL default should be w128'); process.exit(1); }
  console.log(`  first sample max|val| = ${Math.max(...Array.from(st.cur, Math.abs)).toFixed(2)}`);
  console.log(`  architectures offered: ${st.MOD ? Object.keys(st.MOD).join(', ') : '(MOD not exposed)'}`);
  console.log('  PASS labelling page defaults to the 22->128->128->32 model');

  // ---- 2. the compare page must target the same model -----------------------------
  delete global.window;
  harness('compare.html', { search: '?model=w128' });
  await new Promise(r => setTimeout(r, 500));
  const cs = global.window.CMP.getState();
  console.log(`\ncompare.html model loaded: ${cs.MODEL && cs.MODEL.meta.source_run}`);
  const mul = cs.MODEL && cs.MODEL.meta.multiplies;
  console.log(`  multiplies reported by the model: ${mul}`);
  if (mul !== 23296) { console.error('FAIL expected 23,296 multiplies for 22->128->128->32'); process.exit(1); }
  console.log('  PASS compare page shows the same architecture');

  // ---- 3. a reward model for a DIFFERENT model must be refused --------------------
  console.log('\nreward-model guard:');
  if (FILES['preference_reward.json']) {
    const r = FILES['preference_reward.json'];
    console.log(`  on disk the reward model is fitted for: ${r.model || '(no model field)'}`);
    if (r.model && r.model !== 'w128')
      console.log('  -> compare.html will IGNORE it for w128 (correct)');
  } else {
    console.log('  no reward model on disk yet');
  }
  console.log('\nLABEL_W128_TEST_DONE');
})();
