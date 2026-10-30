const fs=require('fs'),path=require('path');
const WEB=path.join('E:/redstone_nn/02_phase2_avatar','webapp');
const html=fs.readFileSync(path.join(WEB,'compare.html'),'utf8');
let code=html.match(/<script>([\s\S]*?)<\/script>/)[1];
function mk(){return{children:[],style:{cssText:''},dataset:{},_t:'',_h:'',appendChild(c){this.children.push(c);return c;},
 addEventListener(){},querySelectorAll(){return[];},querySelector(){return null;},
 classList:{add(){},remove(){},contains(){return false;},toggle(){}},
 getContext(){return{createImageData:(w,h)=>({data:new Uint8ClampedArray(w*h*4)}),putImageData(){},clearRect(){}};},
 set innerHTML(v){this._h=v;this.children=[];},get innerHTML(){return this._h;},
 set textContent(v){this._t=v;},get textContent(){return this._t;},width:8,height:8};}
const byId={};
global.document={createElement:mk,getElementById:id=>(byId[id]||=mk()),querySelectorAll:()=>[],querySelector:()=>mk()};
global.window={addEventListener(){}};
const F={};
for(const f of ['pca22_lam5_model.json','preference_reward.json'])
  if(fs.existsSync(path.join(WEB,f))) F[f]=JSON.parse(fs.readFileSync(path.join(WEB,f),'utf8'));
global.fetch=async u=>{const k=String(u).split('/').pop();return F[k]?{ok:true,status:200,json:async()=>F[k]}:{ok:false,status:404,json:async()=>({})};};
global.console.warn=()=>{};
code=code.replace(/init\(\)\.catch\([\s\S]*?\}\);\s*$/,'');
new Function(code)();
(async()=>{
  await new Promise(r=>setTimeout(r,400));
  const st=global.window.CMP.getState();
  console.log('model:', st.MODEL&&st.MODEL.meta.steps_trained,'steps  out_mode:',st.OUT_MODE);
  console.log('reward model:', st.REW?'PRESENT ('+st.REW.n_labels+' labels)':'absent');
  global.window.CMP.run();
  console.log('random grid tiles:', byId['gRand'].children.length);
  console.log('guided grid tiles:', byId['gGuid'].children.length);
  const okTiles = byId['gRand'].children.length===24 && byId['gGuid'].children.length===24;
  console.log(okTiles?'PASS both grids rendered 24 faces':'FAIL grid count');
  console.log(okTiles?'COMPARE_PAGE_TEST_DONE':'COMPARE_FAILED');
  if(!okTiles) process.exit(1);
})();
