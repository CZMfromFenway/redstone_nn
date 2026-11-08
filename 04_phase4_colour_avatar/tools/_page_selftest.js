// 用 Node 直接跑 label.html 里的 JS，定位「评了一张之后按键失效」的原因。
// 只做本地复现，不参与生产；跑完即可删。
const fs = require('fs');
const path = require('path');

const P4 = path.resolve(__dirname, '..');
const htmlPath = path.join(__dirname, 'label.html');
const html = fs.readFileSync(htmlPath, 'utf8');
const m = html.match(/<script>([\s\S]*)<\/script>/);
if (!m) throw new Error('no script block');
let js = m[1];

// 暴露内部状态，并把最后的 boot() 换成挂到 global
js = js.replace(/boot\(\);\s*$/, '');
js += '\nglobalThis.__api = {boot, next, grade, pull:()=>({item, done, totalQuota, counts:JSON.parse(JSON.stringify(counts)), nArms:arms.length})};\n';

// ---- 最小 DOM 桩
const mkEl = () => ({
  style: {}, textContent: '', innerHTML: '', firstElementChild: { style: {} },
  getContext: () => ({
    createImageData: (w, h) => ({ data: new Uint8ClampedArray(w * h * 4) }),
    putImageData: () => {},
  }),
  addEventListener: () => {},
});
const els = {};
global.document = {
  getElementById: (id) => (els[id] = els[id] || mkEl()),
  querySelectorAll: () => [],
  addEventListener: () => {},
};
global.location = { search: '?round=dbg&n=3' };
global.window = global;
global.window.addEventListener = () => {};
global.URLSearchParams = URLSearchParams;

// ---- fetch 桩：相对路径读磁盘，/label 记账
const posted = [];
global.fetch = async (url, opt) => {
  if (url === '/label') {
    posted.push(JSON.parse(opt.body));
    return { ok: true, json: async () => ({ ok: true }) };
  }
  const p = path.join(__dirname, url);
  if (!fs.existsSync(p)) throw new Error('404 ' + url);
  const txt = fs.readFileSync(p, 'utf8');
  return { ok: true, json: async () => JSON.parse(txt) };
};

(async () => {
  eval(js);
  await globalThis.__api.boot();
  let st = globalThis.__api.pull();
  console.log('boot: arms=', st.nArms, 'item=', !!st.item, 'totalQuota=', st.totalQuota);
  console.log('页面提示:', (els.msg && els.msg.textContent) || '(空)', '| class=', (els.msg && els.msg.className) || '');
  for (let i = 1; i <= 40; i++) {
    try {
      await globalThis.__api.grade(2);
      st = globalThis.__api.pull();
      if (st.done >= st.totalQuota) { console.log(`完成：done=${st.done}/${st.totalQuota} counts=${JSON.stringify(st.counts)}`); break; }
      if (i % 5 === 0) console.log(`after grade#${i}: done=${st.done}/${st.totalQuota} counts=${JSON.stringify(st.counts)}`);
    } catch (e) {
      console.log(`grade#${i} THREW:`, e && e.message);
      break;
    }
  }
  const sel = posted.filter(r => r.candidates > 0);
  const last = sel[sel.length-1] || {};
  console.log('选择臂记录 candidates/score:', last.candidates, last.score,
              '| z[0:3]=', (last.z||[]).slice(0,3));
  const ctl = posted.filter(r => r.model === 'snap_sel4' || r.model === 'snap_sel');
  console.log('所有选择臂记录数:', sel.length, ' 其中 candidates=4 的:', sel.filter(r=>r.candidates===4).length);
  console.log('pca 长度分布:', JSON.stringify(posted.reduce((a,r)=>{a[r.pca.length]=(a[r.pca.length]||0)+1;return a;},{})),
              ' z 长度分布:', JSON.stringify(posted.reduce((a,r)=>{a[(r.z||[]).length]=(a[(r.z||[]).length]||0)+1;return a;},{})));
  // 再试 skip 路径
  try {
    globalThis.__api.next();
    st = globalThis.__api.pull();
    console.log('after skip: item=', !!st.item);
  } catch (e) {
    console.log('skip THREW:', e && e.message);
  }
})();
