const fs=require('fs'),path=require('path');
const WEB=path.join('E:/redstone_nn/02_phase2_avatar','webapp');
const html=fs.readFileSync(path.join(WEB,'index.html'),'utf8');
let code=html.match(/<script>([\s\S]*?)<\/script>/)[1];
function mk(){return{children:[],style:{},dataset:{},_t:'',_h:'',appendChild(c){this.children.push(c);return c;},
 addEventListener(){},querySelectorAll(){return[];},querySelector(){return null;},
 classList:{add(){},remove(){},contains(){return false;},toggle(){}},
 getContext(){return{createImageData:(w,h)=>({data:new Uint8ClampedArray(w*h*4)}),putImageData(){},clearRect(){}};},
 set innerHTML(v){this._h=v;},get innerHTML(){return this._h;},
 set textContent(v){this._t=v;},get textContent(){return this._t;},width:8,height:8};}
const byId={};
global.document={createElement:mk,getElementById:id=>(byId[id]||=mk()),querySelectorAll:()=>[],querySelector:()=>mk()};
global.window={addEventListener(){}};
const F={};
for(const f of ['pca22_lam5_model.json','pca22_nonoise_model.json','pca22_model.json','pca22_gray32_model.json','pca22_color16_model.json','style_envelopes.json'])
  F[f]=JSON.parse(fs.readFileSync(path.join(WEB,f),'utf8'));
global.fetch=async u=>{const k=String(u).split('/').pop();return F[k]?{ok:true,status:200,json:async()=>F[k]}:{ok:false,status:404,json:async()=>({})};};
global.console.warn=()=>{};
code=code.replace(/init\(\)\.catch\([\s\S]*?\}\);\s*$/,'');
new Function(code)();
const APP=global.window.APP;
(async()=>{
  await APP.init();
  const gal=APP.getState().gallery;
  let samePairs={};
  const keys=['lam5','nonoise','noise','gray32','color16'];
  for(let gi=0;gi<12;gi++){
    const p=new Float32Array(gal[gi].pca);
    const sigs={};
    for(const k of keys){
      APP.setModel(k); APP.setPca(p);
      const r=APP.forward(p,new Float32Array(64).fill(0.1));
      const v=APP.decode(r,1.0,'mode',APP.rng(1));
      sigs[k]=Array.from(APP.toClassImg(APP.buildImg(v))).join(',');
    }
    for(let i=0;i<keys.length;i++)for(let j=i+1;j<keys.length;j++){
      const a=keys[i],b=keys[j];
      if(sigs[a]===sigs[b]){const key=a+'='+b;samePairs[key]=(samePairs[key]||0)+1;}
    }
  }
  console.log('identical-output counts over 12 inputs (0 means genuinely different):');
  const all=[];
  for(let i=0;i<keys.length;i++)for(let j=i+1;j<keys.length;j++)all.push(keys[i]+'='+keys[j]);
  for(const k of all) console.log('  '+k.padEnd(20)+' '+(samePairs[k]||0)+'/12');
  const bad=Object.entries(samePairs).filter(([k,v])=>v>=10);
  if(bad.length){console.error('FAIL these pairs are effectively the same model:',bad);process.exit(1);}
  console.log('\nPASS all five models produce distinct outputs');
})();
