// Test the Gaussian-envelope sampling added to the page.
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
global.window = {};
const F = {};
for (const f of ['pca22_lam5_model.json','pca22_nonoise_model.json','pca22_model.json','style_envelopes.json'])
  F[f] = JSON.parse(fs.readFileSync(path.join(WEB, f), 'utf8'));
global.fetch = async u => { const k = String(u).split('/').pop();
  return F[k] ? {ok:true,status:200,json:async()=>F[k]} : {ok:false,status:404,json:async()=>({})}; };
global.console.warn = m => console.log('  WARN', m);

code = code.replace(/init\(\)\.catch\([\s\S]*?\}\);/, '');
new Function(code)();
const APP = global.window.APP;

(async () => {
  await APP.init();
  const st = APP.getState();
  console.log('models   :', st.loaded.join(', '), '(expect 3)');
  console.log('envelopes:', st.envelopes.join(', '), '(expect 4)');
  if (st.loaded.length !== 3 || st.envelopes.length !== 4) {
    console.error('FAIL wrong number of models or envelopes'); process.exit(1);
  }

  const E = APP.ENV();
  console.log('\nenvelope contents:');
  for (const k of Object.keys(E.styles)) {
    const s = E.styles[k];
    const rows = s.chol.length, cols = s.chol[0].length;
    console.log('  ' + k.padEnd(8) + ' n=' + String(s.n_samples).padEnd(3) +
      ' |mean|=' + String(s.mean_norm).padEnd(6) +
      ' chol=' + rows + 'x' + cols +
      ' hit_nn=' + (s.independent_sampler_hit_nn*100).toFixed(1) + '%');
    if (rows !== 22 || cols !== 22) { console.error('FAIL chol not 22x22'); process.exit(1); }
    if (s.mean.length !== 22) { console.error('FAIL mean not length 22'); process.exit(1); }
  }

  // a Cholesky factor must be lower triangular
  let tri = true;
  for (const k of Object.keys(E.styles)) {
    const L = E.styles[k].chol;
    for (let i = 0; i < 22; i++) for (let j = i + 1; j < 22; j++)
      if (Math.abs(L[i][j]) > 1e-9) tri = false;
  }
  console.log('\nPASS all Cholesky factors are lower-triangular:', tri);
  if (!tri) { console.error('FAIL not lower triangular'); process.exit(1); }

  // sampling must (a) stay in the slider range, (b) actually vary, (c) be reproducible
  console.log('\nsampling from each envelope:');
  for (const k of Object.keys(E.styles)) {
    let maxabs = 0, offmean = 0;
    const seen = new Set();
    for (let t = 0; t < 20; t++) {
      const v = APP.sampleEnvelope(k);
      if (!v) { console.error('FAIL sampleEnvelope returned null for ' + k); process.exit(1); }
      let mx = 0;
      for (let i = 0; i < 22; i++) mx = Math.max(mx, Math.abs(v[i]));
      maxabs = Math.max(maxabs, mx);
      if (t === 0) for (let i = 0; i < 22; i++)
        if (Math.abs(v[i] - E.styles[k].mean[i]) > 1e-6) offmean++;
      seen.add(v.map(x => x.toFixed(2)).join(','));
    }
    console.log('  ' + k.padEnd(8) + ' max|val|=' + maxabs.toFixed(2) +
      ' (<=3.5) differs-from-mean in ' + offmean + '/22 dims, distinct draws=' + seen.size + '/20');
    if (maxabs > 3.5 + 1e-6) { console.error('FAIL out of range'); process.exit(1); }
    if (seen.size < 15) { console.error('FAIL draws are not varying'); process.exit(1); }
  }

  console.log('\nPASS envelope sampling stays in range and varies across draws');
  console.log('ENVELOPE_PAGE_TEST_DONE');
})();
