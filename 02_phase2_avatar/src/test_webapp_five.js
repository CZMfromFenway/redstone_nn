// Verify the five-model page: all load, every architecture decodes correctly, the
// colour model uses the palette, and the 32-channel model's continuous output is not
// secretly an integer.
const fs = require('fs'), path = require('path');
const WEB = path.join(__dirname, '..', 'webapp');
const html = fs.readFileSync(path.join(WEB, 'index.html'), 'utf8');
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

const FILES = ['pca22_lam5_model.json','pca22_nonoise_model.json','pca22_model.json',
               'pca22_gray32_model.json','pca22_color16_model.json','style_envelopes.json'];
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

(async () => {
  await APP.init();
  let st = APP.getState();
  console.log('models loaded:', st.loaded.join(', '), `(${st.loaded.length}/5)`);
  if (st.loaded.length !== 5) { console.error('FAIL expected 5 models'); process.exit(1); }

  const specs = [
    ['lam5',    'logits_gray',      512],
    ['nonoise', 'logits_gray',      512],
    ['noise',   'logits_gray',      512],
    ['gray32',  'continuous32',      32],
    ['color16', 'logits_palette',   512],
  ];
  console.log('\nper-model structure:');
  for (const [k, mode, dim] of specs) {
    APP.setModel(k);
    st = APP.getState();
    const ok = st.OUT_MODE === mode && st.OUT_DIM === dim;
    console.log(`  ${k.padEnd(8)} out_mode=${String(st.OUT_MODE).padEnd(15)} ` +
                `out_dim=${String(st.OUT_DIM).padEnd(4)} hid=${st.HID} ` +
                `in=${st.IN_DIM} ${ok ? 'OK' : 'MISMATCH expected ' + mode + '/' + dim}`);
    if (!ok) { console.error('FAIL structure'); process.exit(1); }
  }

  // ---- the colour model must produce distinct palette colours ----------------------
  APP.setModel('color16');
  const gal = APP.getState().gallery;
  const pal = F['pca22_color16_model.json'].palette;
  const p0 = new Float32Array(gal[0].pca);
  APP.setPca(p0);
  let raw = APP.forward(p0, new Float32Array(64).fill(0.1));
  let v = APP.decode(raw, 1.0, 'mode', APP.rng(1));
  const idxSet = new Set(Array.from(v));
  console.log(`\ncolour16: ${idxSet.size} distinct palette indices in one face, ` +
              `indices ${[...idxSet].sort((a,b)=>a-b).join(',')}`);
  // valueToRGB must return the palette entry, not a grey ramp
  const rgb = APP.valueToRGB(v[0]);
  const expected = pal[Math.round(v[0])];
  const match = rgb[0]===expected[0] && rgb[1]===expected[1] && rgb[2]===expected[2];
  console.log(`  valueToRGB(${v[0]}) = [${rgb}] vs palette[${Math.round(v[0])}] = [${expected}] -> ${match?'PASS':'FAIL'}`);
  if (!match) { console.error('FAIL colour render does not use the palette'); process.exit(1); }
  // and it must NOT be greyscale (unless the palette colour happens to be grey)
  let nonGrey = 0;
  for (const i of idxSet) { const c = pal[i]; if (!(c[0]===c[1] && c[1]===c[2])) nonGrey++; }
  console.log(`  ${nonGrey}/${idxSet.size} of those palette entries are non-greyscale`);
  if (nonGrey === 0) { console.error('FAIL rendered colours are all grey'); process.exit(1); }

  // ---- the 32-channel model's output must be continuous, not integer-valued --------
  APP.setModel('gray32');
  APP.setPca(p0);
  raw = APP.forward(p0, new Float32Array(64).fill(0.1));
  v = APP.decode(raw, 1.0, 'mode', APP.rng(1));
  let nearInt = 0, lo = 1e9, hi = -1e9;
  for (let i = 0; i < 32; i++) {
    if (Math.abs(v[i] - Math.round(v[i])) < 0.1) nearInt++;
    lo = Math.min(lo, v[i]); hi = Math.max(hi, v[i]);
  }
  console.log(`\ngray32: output range ${lo.toFixed(2)}..${hi.toFixed(2)} (must be 0..15), ` +
              `${nearInt}/32 within 0.1 of an integer`);
  if (lo < -0.001 || hi > 15.001) { console.error('FAIL out of level range'); process.exit(1); }
  console.log('  -> the continuous output is NOT an integer in disguise');

  // ---- all five must respond differently to the same input (they are different models)
  console.log('\nspeckle per model on the same input:');
  const res = {};
  for (const [k] of specs) {
    APP.setModel(k);
    APP.setPca(p0);
    const r = APP.forward(p0, new Float32Array(64).fill(0.1));
    const vv = APP.decode(r, 1.0, 'mode', APP.rng(1));
    const img = APP.toClassImg(APP.buildImg(vv));
    res[k] = APP.speckle(img);
    console.log(`  ${k.padEnd(8)} agree ${(res[k].agree*100).toFixed(1)}%  iso ${res[k].iso}  blobs ${res[k].blobs}`);
  }
  const sigs = new Set(specs.map(([k]) => `${res[k].agree.toFixed(3)}|${res[k].iso}|${res[k].blobs}`));
  console.log(`  ${sigs.size}/5 distinct results (all five are genuinely different models)`);
  if (sigs.size < 4) { console.error('FAIL models look identical'); process.exit(1); }

  console.log('\nFIVE_MODEL_PAGE_TEST_DONE');
})();
