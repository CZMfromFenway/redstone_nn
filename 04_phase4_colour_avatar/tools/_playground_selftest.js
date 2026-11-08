// Node 冒烟测试：跑 playground.html 的数学，检查各开关是否真的生效。
// 用法：node tools/_playground_selftest.js
const fs = require('fs');
const path = require('path');

const html = fs.readFileSync(path.join(__dirname, '..', 'webapp', 'playground.html'), 'utf8');
const js = html.slice(html.indexOf('<script>') + 8, html.indexOf('</script>'));

function mkEl(tag) {
  return {
    tagName: (tag||'div').toUpperCase(), children: [], style: {}, dataset: {},
    value: '0', checked: true, textContent: '', _html: '', width: 8, height: 8,
    classList: { add(){}, remove(){}, toggle(){} },
    appendChild(c){ this.children.push(c); return c; },
    addEventListener(){}, scrollIntoView(){}, querySelector(){ return mkEl('i'); },
    querySelectorAll(){ return []; },
    getContext(){ return { fillRect(){}, clearRect(){},
      createImageData:(w,h)=>({data:new Uint8ClampedArray(w*h*4)}), putImageData(){} }; },
    set innerHTML(v){ this._html=v; if (v==='') this.children=[]; },
    get innerHTML(){ return this._html; },
  };
}
const els = {};
const doc = { getElementById:(id)=>(els[id]=els[id]||mkEl('div')),
              createElement:(t)=>mkEl(t), addEventListener(){} };

// 让 UI 读到的开关可控
function setUI(o){
  const map = { colour:'colour', revive:'revive', wool:'wool', selOn:'selOn' };
  for (const k in map) els[k].checked = o[k];
  els.variant.value = o.variant;
  els.alpha.value = String(o.alpha);
  els.cands.value = String(o.cands);
  els.bn.value = String(o.batchN || 16);
  for (let i=0;i<4;i++) els['p'+i].value = '0';
}

// 先给桩元素设初始值，避免页面初始化时读到空值
els.variant = mkEl('select'); els.variant.value = 'curated';
els.alpha = mkEl('input'); els.alpha.value = '1';
els.cands = mkEl('input'); els.cands.value = '4';
els.bn = mkEl('select'); els.bn.value = '16';
for (let i=0;i<4;i++){ els['p'+i] = mkEl('input'); els['p'+i].value = '0';
  els['v'+i] = mkEl('span'); }
['colour','revive','wool','selOn'].forEach(k => { els[k] = mkEl('input'); els[k].checked = true; });
els.wool.checked = false;

const boot = new Function('document','Math',
  js + '\nreturn {D:D,S:S,readUI:readUI,grayLevels:grayLevels,colourIdx:colourIdx,' +
       'samplePalette:samplePalette,scoreOf:scoreOf,pickBest:pickBest,stats:stats};');
const api = boot(doc, Math);
let ok = true;
const check = (n,c,x)=>{ console.log(`  ${c?'OK  ':'FAIL'} ${n}${x?' — '+x:''}`); if(!c) ok=false; };

check('数据完整', !!api.D.gray && !!api.D.colour && !!api.D.variants &&
      Object.keys(api.D.variants).length === 3,
      'variants=' + Object.keys(api.D.variants).join(','));

setUI({ colour:true, revive:true, wool:false, selOn:true, variant:'curated',
        alpha:1, cands:4, batchN:16 });
api.readUI();
const z = [0.0,0.0,0.0,0.0];
const lv = api.grayLevels(Float64Array.from(z));
check('灰度级在 0..15', Array.from(lv).every(v => v>=0 && v<=15),
      'range=' + Math.min(...lv) + '..' + Math.max(...lv));

// 归一化 α 影响灰度
const a1 = Array.from(lv);
api.S.alpha = 0; const a0 = Array.from(api.grayLevels(Float64Array.from(z)));
api.S.alpha = 1;
check('α=0 与 α=1 明显不同', a0.join(',') !== a1.join(','),
      'std α1=' + (Math.std || 0) + ' 差=' + a0.filter((v,i)=>v!==a1[i]).length + ' 位');

// 调色板变体差异
setUI({ colour:true, revive:true, wool:false, selOn:true, variant:'base',
        alpha:1, cands:4 });
api.readUI();
const pBase = api.samplePalette();
setUI({ colour:true, revive:true, wool:false, selOn:true, variant:'curated',
        alpha:1, cands:4 });
api.readUI();
let differs = 0;
for (let t=0;t<200;t++){
  const p = api.samplePalette();
  if (p.map(c=>c.join(',')).join('|') !== pBase.map(c=>c.join(',')).join('|')) differs++;
}
check('精选变体与原色板不同', differs > 0, differs + '/200 次不同');

// 先验校正开/关
api.S.revive = true;  const lvA = api.grayLevels(Float64Array.from(z));
const pal = api.samplePalette();
const iA = api.colourIdx(lvA, pal);
api.S.revive = false;
const iB = api.colourIdx(api.grayLevels(Float64Array.from(z)), pal);
api.S.revive = true;
check('先验校正改变输出', Array.from(iA).join(',') !== Array.from(iB).join(','),
      '差异位 ' + Array.from(iA).filter((v,i)=>v!==iB[i]).length);

// WOOL→QUARTZ 规则：只在没有 QUARTZ 且有 WOOL 时生效
api.S.wool = false;
const iNo = api.colourIdx(api.grayLevels(Float64Array.from(z)), pal);
api.S.wool = true;
const iYes = api.colourIdx(api.grayLevels(Float64Array.from(z)), pal);
api.S.wool = false;
const hasQ = Array.from(iNo).includes(15), hasW = Array.from(iNo).includes(5);
const fired = Array.from(iYes).includes(5) === false && hasW && !hasQ;
check('规则条件正确', (hasQ || !hasW) ? Array.from(iNo).join()===Array.from(iYes).join() : fired,
      `hasQ=${hasQ} hasW=${hasW}`);

// 筛选：挑出来的分数应高于单抽平均
const single = [], best = [];
for (let t=0;t<300;t++){ single.push(api.scoreOf(api.pickBest(1).z)); best.push(api.pickBest(4).score); }
const mean = a => a.reduce((x,y)=>x+y,0)/a.length;
check('留 4 选 1 抬高分', mean(best) > mean(single) + 0.1,
      `单抽 ${mean(single).toFixed(2)} → 留4 ${mean(best).toFixed(2)}`);

// 统计量
const st = api.stats(iA, pal);
check('统计量合理', st.expH > 1 && st.expH <= 16 && st.ncol >= 1,
      `expH=${st.expH.toFixed(2)} ncol=${st.ncol}`);

console.log(ok ? 'PLAYGROUND_SELFTEST_OK' : 'PLAYGROUND_SELFTEST_FAIL');
process.exit(ok ? 0 : 1);
