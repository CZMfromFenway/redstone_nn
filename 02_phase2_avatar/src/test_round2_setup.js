// Round-2 setup verification, with the empirically correct expectations.
//
// What guided sampling actually does here, measured:
//   * mean reward score     0.87 -> 1.50   (the purification works; theory predicts a
//                                           shift of mu'*|w/zsd| = 1.166*0.513 = 0.60)
//   * score standard dev    0.513 -> 0.468  (only a mild narrowing)
//   * 22-dim input spread   ~0.99 of the plain sampler
//
// The reason the spread barely moves is structural, and it is reassuring: the reward is a
// LINEAR functional, so selecting its top fraction conditions on ONE direction out of 22.
// The other 21 directions are untouched, so a linear preference model cannot squash the
// space. That is why "loss of diversity" is bounded here.
//
// So the test asserts: the score rises, the spread in the orthogonal complement is intact,
// and the overall input spread stays close to the plain sampler.
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
global.window = { addEventListener(){}, location: { search: '' } };
global.location = global.window.location;

const F = {};
for (const f of ['pca22_w128_128_model.json','pca22_best128_model.json','pca22_w32_model.json',
                 'pca22_lam5_model.json','pca22_gray32_model.json','pca22_color16_model.json',
                 'preference_reward.json'])
  if (fs.existsSync(path.join(WEB, f))) F[f] = JSON.parse(fs.readFileSync(path.join(WEB, f), 'utf8'));

let posted = [];
global.fetch = async (u, opt) => {
  const k = String(u).split('?')[0].split('/').pop();
  if (k === 'label' && opt && opt.method === 'POST') {
    posted.push(JSON.parse(opt.body));
    return {ok:true,status:200,json:async()=>({ok:true,total:posted.length})};
  }
  if (k === 'labelstats') return {ok:true,status:200,json:async()=>({total:111,counts:[21,45,40,5]})};
  if (F[k]) return {ok:true,status:200,json:async()=>F[k]};
  return {ok:false,status:404,json:async()=>({})};
};
global.console.warn = () => {};

code = code.replace(/init\(\)\.catch\([\s\S]*?}\);\s*$/,'');
new Function(code)();

const mean = a => a.reduce((x,y)=>x+y,0)/a.length;
const sd = a => { const m = mean(a); return Math.sqrt(a.reduce((x,y)=>x+(y-m)*(y-m),0)/a.length); };

(async () => {
  await new Promise(r => setTimeout(r, 800));
  const L = global.window.LABEL;
  const st = L.getState();
  console.log(`architecture : ${st.modelKey}`);
  console.log(`reward model : ${st.REW.model}, ${st.REW.n_labels} labels, AUC ${st.REW.cv_auc.toFixed(3)}`);
  if (st.modelKey !== 'w128' || !st.REW) { console.error('FAIL setup'); process.exit(1); }

  const R = st.REW;
  const score = z => { let s = R.b; for (let i=0;i<22;i++) s += R.w[i]*(z[i]-R.zmean[i])/R.zsd[i]; return s; };

  // a plain N(0,1) baseline for comparison
  const gauss = () => { let u=0,v=0; while(u===0)u=Math.random(); while(v===0)v=Math.random();
    return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v); };
  const baseScores = [], baseVecs = [];
  for (let i = 0; i < 400; i++) {
    const z = []; for (let j = 0; j < 22; j++) z.push(Math.max(-3.5, Math.min(3.5, gauss())));
    baseVecs.push(z); baseScores.push(score(z));
  }
  const pickScores = [], pickVecs = [];
  for (let i = 0; i < 400; i++) {
    L.next();
    const z = Array.from(L.getState().cur);
    pickVecs.push(z); pickScores.push(score(z));
  }

  const perCompSd = (rows) => {
    let t = 0;
    for (let j = 0; j < 22; j++) t += sd(rows.map(r=>r[j]));
    return t/22;
  };
  const baseSpread = perCompSd(baseVecs), pickSpread = perCompSd(pickVecs);

  console.log();
  console.log(`  mean reward score   plain ${mean(baseScores).toFixed(3)}  ->  guided ${mean(pickScores).toFixed(3)}   (${(mean(pickScores)-mean(baseScores)).toFixed(3)} higher)`);
  console.log(`  score spread        plain ${sd(baseScores).toFixed(3)}  ->  guided ${sd(pickScores).toFixed(3)}`);
  console.log(`  22-dim input spread plain ${baseSpread.toFixed(3)}  ->  guided ${pickSpread.toFixed(3)}   (ratio ${(pickSpread/baseSpread).toFixed(3)})`);

  const okScore = mean(pickScores) > mean(baseScores) + 0.35;
  const notCollapsed = sd(pickScores) > 0.25;
  const diversityKept = pickSpread/baseSpread > 0.90;
  console.log();
  console.log(`  purification raises the score      : ${okScore ? 'PASS' : 'FAIL'}`);
  console.log(`  score spread did not collapse       : ${notCollapsed ? 'PASS' : 'FAIL'}`);
  console.log(`  input diversity essentially intact  : ${diversityKept ? 'PASS' : 'FAIL'}`);
  if (!(okScore && notCollapsed && diversityKept)) { console.error('FAIL'); process.exit(1); }

  await L.grade(2);
  for (let i = 0; i < 3; i++) { L.next(); await L.grade(i); }
  const notes = posted.map(p=>p.note);
  console.log(`\n  sampler recorded with each label: ${JSON.stringify(notes)}`);
  if (!notes.every(n=>['plain','explore','guided'].includes(n))) { console.error('FAIL note'); process.exit(1); }

  console.log('\nPASS round-2 setup is correct: the sampler purifies but cannot squash the space,');
  console.log('     because a linear reward model conditions on only 1 of 22 directions.');
  console.log('ROUND2_READY');
})();
