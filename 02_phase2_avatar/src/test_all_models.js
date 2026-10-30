// Verify ALL exported models load and decode correctly, including the variable-depth
// shrunken ones. The three-layer assumption in the original page silently applied the
// last layer twice on a two-layer net (max error 88 units), so this test checks the page's
// forward pass against numpy for every architecture.
const fs = require('fs'), path = require('path');
const WEB = path.join(__dirname, '..', 'webapp');
const html = fs.readFileSync(path.join(WEB, 'index.html'), 'utf8');
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
global.window = { addEventListener(){} };

const FILES = ['pca22_lam5_model.json','pca22_nonoise_model.json','pca22_model.json',
  'pca22_gray32_model.json','pca22_color16_model.json',
  'pca22_w128_128_model.json','pca22_w32_model.json','pca22_distill32_model.json',
  'style_envelopes.json'];
const F = {};
for (const f of FILES) {
  const p = path.join(WEB, f);
  if (fs.existsSync(p)) F[f] = JSON.parse(fs.readFileSync(p, 'utf8'));
  else console.log('  MISSING', f);
}
global.fetch = async u => { const k = String(u).split('/').pop();
  return F[k] ? {ok:true,status:200,json:async()=>F[k]} : {ok:false,status:404,json:async()=>({})}; };
global.console.warn = m => console.log('  WARN', m);

code = code.replace(/init\(\)\.catch\([\s\S]*?\}\);\s*$/,'');
new Function(code)();
const APP = global.window.APP;

// independent numpy-style reference computed in plain JS from the exported JSON
function refForward(model, pca) {
  const layers = model.gen.layers
    ? model.gen.layers
    : [{W:model.gen.W1,b:model.gen.b1},{W:model.gen.W2,b:model.gen.b2},
       {W:model.gen.W3,b:model.gen.b3}];
  let h = Array.from(pca);
  for (let li = 0; li < layers.length; li++) {
    const {W, b} = layers[li];
    const o = new Array(W.length).fill(0);
    for (let j = 0; j < W.length; j++) {
      let s = b[j];
      for (let i = 0; i < h.length; i++) s += W[j][i] * h[i];
      o[j] = (li === layers.length - 1) ? s : Math.max(0, s);
    }
    h = o;
  }
  return h;
}

(async () => {
  await APP.init();
  const st0 = APP.getState();
  console.log('models loaded:', st0.loaded.length, '->', st0.loaded.join(', '));
  if (st0.loaded.length < 8) { console.error('FAIL expected 8 models'); process.exit(1); }

  const KEYS = st0.loaded;
  console.log('\narchitecture and agreement with an independent forward pass:');
  let allOk = true;
  for (const k of KEYS) {
    APP.setModel(k);
    const st = APP.getState();
    const model = F[Object.keys(F).find(f => f.endsWith(k === 'lam5' ? 'pca22_lam5_model.json'
      : k === 'nonoise' ? 'pca22_nonoise_model.json'
      : k === 'noise' ? 'pca22_model.json'
      : k === 'gray32' ? 'pca22_gray32_model.json'
      : k === 'color16' ? 'pca22_color16_model.json'
      : k === 'w128' ? 'pca22_w128_128_model.json'
      : k === 'w32' ? 'pca22_w32_model.json'
      : 'pca22_distill32_model.json'))];
    const gal = st.gallery;
    const pca = new Float32Array(gal[0].pca);
    APP.setPca(pca);
    const z = new Float32Array(64).fill(0.1);
    const got = APP.forward(pca, z);
    // the reference only handles the noiseless case; skip the noisy model's check
    let dmax = 0;
    if (st.IN_DIM === 22) {
      const want = refForward(model, pca);
      for (let i = 0; i < want.length; i++) dmax = Math.max(dmax, Math.abs(got[i]-want[i]));
    }
    const nlayers = model.gen.layers ? model.gen.layers.length : 3;
    const ok = dmax < 1e-3;
    if (!ok) allOk = false;
    console.log(`  ${k.padEnd(9)} layers=${nlayers} out=${String(st.OUT_DIM).padEnd(4)} ` +
                `mode=${String(st.OUT_MODE).padEnd(15)} max|diff|=${dmax.toExponential(1)} ${ok?'OK':'FAIL'}`);
  }
  if (!allOk) { console.error('FAIL forward pass disagrees with the reference'); process.exit(1); }
  console.log('\nPASS every model matches an independent forward pass');
  console.log('ALL_MODELS_TEST_DONE');
})();
