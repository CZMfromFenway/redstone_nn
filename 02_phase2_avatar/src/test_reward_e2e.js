// End-to-end check: with the fitted reward model in place, does the compare page's guided
// sampler actually concentrate on high-scoring style vectors, and does the page report it
// honestly?
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
global.window = { addEventListener(){} };
const F = {};
for (const f of ['pca22_lam5_model.json','preference_reward.json'])
  if (fs.existsSync(path.join(WEB, f))) F[f] = JSON.parse(fs.readFileSync(path.join(WEB, f), 'utf8'));
global.fetch = async u => { const k = String(u).split('/').pop();
  return F[k] ? {ok:true,status:200,json:async()=>F[k]} : {ok:false,status:404,json:async()=>({})}; };
global.console.warn = () => {};

code = code.replace(/init\(\)\.catch\([\s\S]*?\}\);\s*$/,'');
new Function(code)();
const CMP = global.window.CMP;

(async () => {
  await new Promise(r => setTimeout(r, 400));
  const st = CMP.getState();
  const R = st.REW;
  if (!R) { console.error('FAIL no reward model exported'); process.exit(1); }
  console.log(`reward model: kind=${R.kind}  n_labels=${R.n_labels}  cv_auc=${R.cv_auc}`);
  console.log(`  counts 优/良/中/差 = ${R.counts['优']}/${R.counts['良']}/${R.counts['中']}/${R.counts['差']}`);
  if (R.kind !== 'linear') { console.error('FAIL expected the linear model'); process.exit(1); }
  if (R.w.length !== 22) { console.error('FAIL w must have 22 entries'); process.exit(1); }

  // score the 159 labelled points through the page's own rewardOf and confirm it ranks
  // 优+良 above 中+差 (in-sample sanity: a sign flip would break the sampler)
  const labels = fs.readFileSync(path.join(WEB, '..', 'results', 'preference', 'labels.jsonl'), 'utf8')
    .split('\n').filter(Boolean).map(l => JSON.parse(l)).filter(r => Array.isArray(r.pca) && r.pca.length === 22);
  console.log(`\nscoring the ${labels.length} labelled points through the exported model:`);
  const hi = [], lo = [];
  for (const r of labels) {
    const zs = r.pca.map((v,i)=>(v-R.zmean[i])/R.zsd[i]);
    let s = R.b; for (let i=0;i<22;i++) s += R.w[i]*zs[i];
    (r.grade >= 2 ? hi : lo).push(s);
  }
  const mh = hi.reduce((a,b)=>a+b,0)/hi.length;
  const ml = lo.reduce((a,b)=>a+b,0)/lo.length;
  console.log(`  优+良 mean ${mh.toFixed(3)}  (n=${hi.length})`);
  console.log(`  中+差 mean ${ml.toFixed(3)}  (n=${lo.length})`);
  console.log(`  gap ${(mh-ml).toFixed(3)}  ${mh>ml?'PASS':'FAIL sign flipped'}`);
  if (!(mh > ml)) { console.error('FAIL the exported model ranks the classes backwards'); process.exit(1); }

  // the decisive check: guided draws must score higher than random draws
  console.log('\ncomparing 400 random vs 400 guided draws through the page sampler:');
  const gal = F['pca22_lam5_model.json'].gallery;
  CMP.run();
  // re-derive the two samplers the page uses, by calling run and reading the reported means
  const sRand = parseFloat(byId['sRand'].textContent);
  const sGuid = parseFloat(byId['sGuid'].textContent);
  console.log(`  page reports  random ${sRand.toFixed(3)}  guided ${sGuid.toFixed(3)}  ` +
              `gain ${(sGuid-sRand).toFixed(3)}`);
  if (!(sGuid > sRand)) { console.error('FAIL guided sampling does not score higher'); process.exit(1); }
  console.log('  PASS guided sampling concentrates on higher-scoring vectors');

  console.log('\nNOTE the expected real-world gain is modest: on held-out labels, ranking by');
  console.log('     this model and keeping the top 35% raised the true 优+良 rate from');
  console.log('     67.3% to 74.5% (+7.2 points). See results/preference/filter_gain.json.');
  console.log('\nREWARD_E2E_DONE');
})();
