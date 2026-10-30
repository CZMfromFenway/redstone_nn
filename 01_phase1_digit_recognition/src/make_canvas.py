#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
"""
生成画板程序 draw.html（单文件，双击即用）
=============================================
功能:
  * 15x15 画板，鼠标/触屏绘制
  * 笔刷粗细 1~4 格
  * 橡皮、撤回、清空
  * 实时显示: 识别数字、10 个通道的 z / F 值、柱状图
  * 显示每个通道的容器值列表（可切换到"容器值"视图）
  * 内置 10 个预设样本（每个数字一个，识别正确的）
权重从 results/Wint_model.npz 读出后直接嵌入 HTML，保证与模型逐位一致。
"""
import os, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, 'results')
OUT = os.path.join(ROOT, 'draw.html')
TARGET = 128


def load_idx(p):
    import struct
    with open(p, 'rb') as f:
        mg = struct.unpack('>I', f.read(4))[0]; nd = mg & 0xFF
        dims = [struct.unpack('>I', f.read(4))[0] for _ in range(nd)]
        d = np.frombuffer(f.read(), dtype=np.uint8)
    return d.reshape(dims)


def to15(im):
    n = len(im)
    a = im.reshape(n, 28, 28).astype(np.float32).reshape(n, 14, 2, 14, 2).mean(axis=(2, 4))
    o = np.zeros((n, 15, 15), dtype=np.float32)
    o[:, :14, :14] = a; o[:, 14, :14] = a[:, 13, :]
    o[:, :14, 14] = a[:, :, 13]; o[:, 14, 14] = a[:, 13, 13]
    return o


def main():
    d = np.load(os.path.join(RES, 'Wint_model.npz'))
    W = d['W'].astype(int); bias = d['b'].astype(int)

    # 各层容器值
    cvals = {}
    for k in range(10):
        par = 'even' if (k + 1) % 2 == 0 else 'odd'
        cvals[k] = (15 - W[:, k]) if par == 'even' else (15 + W[:, k])

    # 预设样本
    RAW = r'D:\czm我的世界\文件\Comparator Network\tiny_cnn\data\MNIST\raw'
    Xte = load_idx(os.path.join(RAW, 't10k-images-idx3-ubyte'))
    yte = load_idx(os.path.join(RAW, 't10k-labels-idx1-ubyte'))
    thr = to15(load_idx(os.path.join(RAW, 'train-images-idx3-ubyte'))).mean()
    B = (to15(Xte) > thr)
    Z = B.reshape(len(B), -1).astype(np.int64) @ W + bias
    presets = {}
    for digit in range(10):
        cand = np.flatnonzero((yte == digit) & (Z.argmax(1) == digit))
        if len(cand):
            i = int(cand[0])
            presets[str(digit)] = [int(v) for v in B[i].reshape(-1)]

    data = {
        'W': W.tolist(),                 # 225 x 10
        'bias': bias.tolist(),
        'cvals': {str(k): cvals[k].tolist() for k in range(10)},
        'presets': presets,
        'target': TARGET,
    }

    html = HTML_TEMPLATE.replace('__DATA__', json.dumps(data, separators=(',', ':')))
    with open(OUT, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"已生成: {OUT}")
    print(f"  权重 {W.shape}, 预设样本 {len(presets)} 个")


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>红石神经网络 画板调试器</title>
<style>
  :root { --red:#c82828; --blue:#1e5ab4; --grey:#888; }
  * { box-sizing: border-box; }
  body { margin:0; font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
         background:#f4f5f7; color:#1c1e21; }
  .wrap { display:flex; gap:18px; padding:16px; align-items:flex-start;
          flex-wrap:wrap; }
  .left { background:#fff; padding:14px; border-radius:10px;
          box-shadow:0 1px 4px rgba(0,0,0,.12); }
  .right { flex:1; min-width:420px; }
  h2 { margin:0 0 10px; font-size:16px; font-weight:600; }
  #board { border:1px solid #c9ccd1; border-radius:4px; cursor:crosshair;
           touch-action:none; display:block; background:#fff; }
  .btns { margin-top:10px; display:flex; gap:8px; flex-wrap:wrap;
          align-items:center; }
  button { padding:7px 14px; font-size:14px; border:1px solid #c9ccd1;
           background:#fff; border-radius:6px; cursor:pointer; }
  button:hover { background:#eef1f5; }
  button.on { background:#2d6cdf; color:#fff; border-color:#2d6cdf; }
  select { padding:6px 8px; border:1px solid #c9ccd1; border-radius:6px;
           font-size:14px; }
  .hint { color:#666; font-size:12px; margin-top:8px; line-height:1.6; }
  .card { background:#fff; padding:14px; border-radius:10px;
          box-shadow:0 1px 4px rgba(0,0,0,.12); margin-bottom:14px; }
  .big { font-size:44px; font-weight:700; line-height:1; }
  .big small { font-size:14px; font-weight:400; color:#666; }
  table { border-collapse:collapse; width:100%; font-variant-numeric:tabular-nums; }
  th,td { padding:4px 8px; text-align:right; font-size:13px;
          border-bottom:1px solid #eceff2; }
  th:first-child, td:first-child { text-align:left; }
  tr.win { background:#fff5d6; }
  tr.win td { font-weight:700; }
  .bar { height:14px; background:#e8ebef; border-radius:3px; overflow:hidden;
         position:relative; }
  .bar i { display:block; height:100%; background:#2d6cdf; }
  .bar i.neg { background:#c82828; }
  .mono { font-family: Consolas, monospace; font-size:12px; }
  .grid2 { display:grid; grid-template-columns:1fr 1fr; gap:14px; }
  .note { font-size:12px; color:#666; margin-top:6px; }
</style>
</head>
<body>
<div class="wrap">
  <div class="left">
    <h2>画板（15×15）</h2>
    <canvas id="board" width="450" height="450"></canvas>
    <div class="btns">
      <button id="btnDraw" class="on">✏️ 画笔</button>
      <button id="btnErase">🧽 橡皮</button>
      <button id="btnUndo">↶ 撤回</button>
      <button id="btnClear">🗑 清空</button>
    </div>
    <div class="btns">
      <span style="font-size:13px">笔刷</span>
      <button class="sz" data-sz="1">1</button>
      <button class="sz on" data-sz="2">2</button>
      <button class="sz" data-sz="3">3</button>
      <button class="sz" data-sz="4">4</button>
      <span style="font-size:13px;margin-left:8px">预设</span>
      <select id="preset"><option value="">— 选择 —</option></select>
    </div>
    <div class="hint">
      左键拖动绘制 · <b>Ctrl+Z</b> 撤回 · 点亮格数 <b id="nlit">0</b>
    </div>
  </div>

  <div class="right">
    <div class="card">
      <h2>识别结果</h2>
      <div class="big"><span id="pred">—</span>
        <small id="predInfo"></small></div>
      <div class="note" id="marginInfo"></div>
    </div>

    <div class="card">
      <h2>10 个通道输出（F = z + 128，可直接比较大小）</h2>
      <table id="tbl">
        <thead><tr>
          <th>层(数字)</th><th>y</th><th>ΣG<sub>i</sub></th><th>F<sub>0</sub></th>
          <th>z</th><th>F</th><th>排序</th><th style="width:150px">F</th>
        </tr></thead>
        <tbody></tbody>
      </table>
      <div class="note">
        识别到的数字 = <b>F 最大的层</b>。
        若两个通道 F 相同（margin=0），本程序取层号小者，实机可能不同。
      </div>
    </div>

    <div class="card">
      <h2>容器值（该数字层需要放置的信号）</h2>
      <div class="btns">
        <span style="font-size:13px">查看数字</span>
        <select id="viewDigit"></select>
        <label style="font-size:13px">
          <input type="checkbox" id="onlyLit"> 只显示点亮的格</label>
      </div>
      <div id="cbox" class="mono" style="margin-top:10px"></div>
    </div>
  </div>
</div>

<script>
const DATA = __DATA__;
const W = DATA.W, BIAS = DATA.bias, CVALS = DATA.cvals;
const TARGET = DATA.target;
const N = 15, SC = 30;

let grid = new Array(N*N).fill(0);
let history = [];
let tool = 'draw', brush = 2;
let drawing = false;

const cv = document.getElementById('board');
const ctx = cv.getContext('2d');

function snapshot(){ history.push(grid.slice()); if(history.length>200) history.shift(); }

function drawBoard(){
  ctx.clearRect(0,0,cv.width,cv.height);
  ctx.fillStyle = '#fff'; ctx.fillRect(0,0,cv.width,cv.height);
  // 网格
  ctx.strokeStyle = '#dfe3e8'; ctx.lineWidth = 1;
  for(let i=0;i<=N;i++){
    ctx.beginPath(); ctx.moveTo(i*SC+.5,0); ctx.lineTo(i*SC+.5,N*SC);
    ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0,i*SC+.5); ctx.lineTo(N*SC,i*SC+.5);
    ctx.stroke();
  }
  // 点亮
  ctx.fillStyle = '#1c1e21';
  for(let r=0;r<N;r++) for(let c=0;c<N;c++)
    if(grid[r*N+c]) ctx.fillRect(c*SC, r*SC, SC, SC);
  // 外框
  ctx.strokeStyle = '#9aa0a6'; ctx.lineWidth = 2;
  ctx.strokeRect(1,1,cv.width-2,cv.height-2);
}

function paintAt(px, py){
  const c = Math.floor(px/SC), r = Math.floor(py/SC);
  if(r<0||r>=N||c<0||c>=N) return;
  const half = Math.floor((brush-1)/2);
  let changed = false;
  for(let dy=0; dy<brush; dy++) for(let dx=0; dx<brush; dx++){
    const rr = r - half + dy, cc = c - half + dx;
    if(rr<0||rr>=N||cc<0||cc>=N) continue;
    const v = (tool==='erase') ? 0 : 1;
    if(grid[rr*N+cc] !== v){ grid[rr*N+cc] = v; changed = true; }
  }
  if(changed){ drawBoard(); evaluate(); }
}

cv.addEventListener('pointerdown', e=>{
  snapshot(); drawing = true; cv.setPointerCapture(e.pointerId);
  const rc = cv.getBoundingClientRect();
  paintAt(e.clientX-rc.left, e.clientY-rc.top);
});
cv.addEventListener('pointermove', e=>{
  if(!drawing) return;
  const rc = cv.getBoundingClientRect();
  paintAt(e.clientX-rc.left, e.clientY-rc.top);
});
cv.addEventListener('pointerup', ()=>{ drawing = false; });
cv.addEventListener('pointercancel', ()=>{ drawing = false; });

function evaluate(){
  const nlit = grid.reduce((a,b)=>a+b,0);
  document.getElementById('nlit').textContent = nlit;
  const Fs = [], zs = [], Gsums = [];
  for(let k=0;k<10;k++){
    let z = BIAS[k], gs = 0;
    // G_i 用于展示: G_i0 - sum(输出)
    for(let p=0;p<225;p++){
      if(grid[p]) z += W[p][k];
    }
    zs.push(z);
    Fs.push(z + TARGET);
  }
  // 排序
  const order = Fs.map((v,i)=>[v,i]).sort((a,b)=> b[0]-a[0] || a[1]-b[1]);
  const pred = order[0][1];
  const margin = order[0][0] - order[1][0];
  document.getElementById('pred').textContent = pred;
  document.getElementById('predInfo').textContent =
     `  (第 ${pred+1} 层, y=${yOf(pred+1)})`;
  document.getElementById('marginInfo').textContent =
     `F = ${order[0][0]}，次大 ${order[1][0]}（数字 ${order[1][1]}），` +
     `margin = ${margin}` + (margin===0 ? '  ⚠ 平局' : '');

  const tb = document.querySelector('#tbl tbody');
  tb.innerHTML = '';
  for(let k=0;k<10;k++){
    const y = yOf(k+1);
    const par = ((k+1)%2===0) ? '偶' : '奇';
    const rank = order.findIndex(o=>o[1]===k)+1;
    const tr = document.createElement('tr');
    if(k===pred) tr.className='win';
    // F 柱: 以最小 F 为 0，最大 F 为满格
    const fmin = Math.min(...Fs), fmax = Math.max(...Fs);
    const w = fmax>fmin ? (Fs[k]-fmin)/(fmax-fmin) : 0.5;
    tr.innerHTML = `<td>${k+1} (${k}) ${par}</td><td>${y}</td>` +
      `<td></td><td></td><td>${zs[k]}</td><td><b>${Fs[k]}</b></td>` +
      `<td>#${rank}</td>` +
      `<td><div class="bar"><i style="width:${(w*100).toFixed(1)}%"></i></div></td>`;
    tb.appendChild(tr);
  }
  renderContainers();
}

function yOf(layer){ return -(1 + 2*(10-layer)); }

function renderContainers(){
  const k = +document.getElementById('viewDigit').value;
  const onlyLit = document.getElementById('onlyLit').checked;
  const cv2 = CVALS[String(k)];
  let html = '<table><thead><tr><th>行</th><th>列</th>' +
             '<th>点亮</th><th>容器信号</th><th>权重 W</th></tr></thead><tbody>';
  for(let r=0;r<N;r++) for(let c=0;c<N;c++){
    const p = r*N+c;
    if(onlyLit && !grid[p]) continue;
    const lit = grid[p];
    html += `<tr${lit?' class="win"':''}><td>${r}</td><td>${c}</td>` +
            `<td>${lit?'●':'·'}</td><td>${cv2[p]}</td><td>${W[p][k]}</td></tr>`;
  }
  html += '</tbody></table>';
  document.getElementById('cbox').innerHTML = html;
}

// 控件
document.getElementById('btnDraw').onclick = ()=>{
  tool='draw';
  document.getElementById('btnDraw').classList.add('on');
  document.getElementById('btnErase').classList.remove('on');
};
document.getElementById('btnErase').onclick = ()=>{
  tool='erase';
  document.getElementById('btnErase').classList.add('on');
  document.getElementById('btnDraw').classList.remove('on');
};
document.getElementById('btnUndo').onclick = ()=>{
  if(history.length){ grid = history.pop(); drawBoard(); evaluate(); }
};
document.getElementById('btnClear').onclick = ()=>{
  snapshot(); grid = new Array(N*N).fill(0); drawBoard(); evaluate();
};
document.querySelectorAll('.sz').forEach(b=>{
  b.onclick = ()=>{
    brush = +b.dataset.sz;
    document.querySelectorAll('.sz').forEach(x=>x.classList.remove('on'));
    b.classList.add('on');
  };
});
window.addEventListener('keydown', e=>{
  if((e.ctrlKey||e.metaKey) && e.key.toLowerCase()==='z'){
    e.preventDefault(); document.getElementById('btnUndo').click();
  }
});

// 视图选择
const vd = document.getElementById('viewDigit');
for(let k=0;k<10;k++){
  const o=document.createElement('option'); o.value=k; o.textContent=`${k}（第${k+1}层）`;
  vd.appendChild(o);
}
vd.value = 9;
vd.onchange = renderContainers;
document.getElementById('onlyLit').onchange = renderContainers;

// 预设
const ps = document.getElementById('preset');
Object.keys(DATA.presets).sort((a,b)=>a-b).forEach(d=>{
  const o=document.createElement('option'); o.value=d; o.textContent=`数字 ${d}`;
  ps.appendChild(o);
});
ps.onchange = ()=>{
  const d = ps.value; if(d==='') return;
  snapshot();
  grid = DATA.presets[d].slice();
  drawBoard(); evaluate();
};

// 初始：加载数字 0 的预设
grid = DATA.presets['0'].slice();
drawBoard(); evaluate();
</script>
</body>
</html>
"""

if __name__ == '__main__':
    main()
