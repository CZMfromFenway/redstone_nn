/*
 * 直接跑 Jim_Sama 的 8x8 diffusion 权重（零改动地用他自带的 DDPM6 推理内核）。
 *
 * 为什么这样做：他的 HTML 是一个 7.74 MB 的单文件，模型源码与 fp16 权重都在里面。
 * 我们不需要复现他的训练，只需要**把成品当黑箱用**，先量出「8x8 上正统 diffusion
 * 能出什么样的脸」——这是四期 diffusion 上限计划的 L1。
 *
 * 用法（node 不在 PATH 上，用绝对路径）：
 *   node tools/jim_diffusion_run.js --n 8 --steps 10 --seed 34 --w 1.2
 *   node tools/jim_diffusion_run.js --n 200 --out results/diffusion/jim_samples.json
 *
 * 产物：每张脸的 64 个 #rrggbb（吸附到 MC 官方 234 色之前的连续色 + 吸附后的色号），
 *       JSON 里还带参数与耗时，便于与现役管线同轮盲测。
 */
'use strict';

const fs = require('fs');
const path = require('path');

function arg(name, def) {
  const i = process.argv.indexOf('--' + name);
  if (i < 0) return def;
  const v = process.argv[i + 1];
  return v === undefined ? true : (isNaN(+v) ? v : +v);
}

const ROOT = path.resolve(__dirname, '..', '..');
const HTML = arg('html', path.join(ROOT, 'shared_by_jim_sama', 'index_v6.3u_diffusion.html'));
const N = arg('n', 8);
const STEPS = arg('steps', 10);
const SEED0 = arg('seed', 34);
const W = arg('w', 1.2);
const QB = arg('qb', 7);          // 0..7，他的「拟人化」档位（null = 8）
const ALPHA = arg('alpha', 0.08); // 镜像引导
const K = arg('k', 12);           // 逐张 k-means 色数
const SYM = arg('sym', 16);       // 对称吸附阈值 τ
const OUT = arg('out', '');

/* ---------- 1. 从 HTML 里挖出 payload 与模型脚本 ---------- */
const html = fs.readFileSync(HTML, 'utf8');

function grab(re, what) {
  const m = html.match(re);
  if (!m) throw new Error('没找到 ' + what);
  return m[1];
}

// payload：<script ... id="payload" type="application/json">{...}</script>
const payloadTxt = grab(/id="payload"[^>]*>([\s\S]*?)<\/script>/, 'payload');
const payload = JSON.parse(payloadTxt);

// 模型脚本：紧跟在 payload 之后、含 ddpm6.js 注释的那个 <script>
const modelSrc = grab(/<script[^>]*>(\/\* =+\s*\n\s*ddpm6\.js[\s\S]*?)<\/script>/, 'ddpm6.js');

// 让模型脚本以为自己跑在浏览器（它检测 module / global）
const mod = { exports: {} };
const fn = new Function('module', 'exports', 'globalThis', modelSrc + '\nreturn DDPM6;');
const DDPM6 = fn(mod, mod.exports, globalThis);
if (!DDPM6) throw new Error('DDPM6 没挂上');

const PAL = payload.palette;
const MODEL = DDPM6.decode(payload.meta, payload.w);
const META = payload.meta;

console.error('[jim] params=%s T=%s pred=%s palette=%d',
  META.params, META.T, META.pred, PAL.length);

/* ---------- 2. 采样 ---------- */
function sampleOne(seed) {
  const t0 = Date.now();
  const st = MODEL.ddimInit({ steps: STEPS, seed: seed, cb: QB, w: W, alpha: ALPHA });
  let rec = null;
  while (!st.done) rec = MODEL.ddimStep(st);
  const x0 = st.cur;                       // 最后一次的第 0 步输出
  const px = DDPM6.toRGB(x0);              // 连续 RGB（0..255）
  const q = DDPM6.quantizeSnap(px, PAL, K, 24, seed);   // 逐张 k-means → 吸附 234 色
  const s = DDPM6.symSnap(q, PAL, SYM);                 // 对称吸附
  const ms = Date.now() - t0;
  return {
    seed: seed,
    ms: ms,
    raw: DDPM6.toHex(x0),                  // 吸附前的连续色
    snapped: DDPM6.idxToHex(s, PAL),       // 吸附后的成品色
    sym: DDPM6.symScore(s),
    ncol: new Set(s).size,
  };
}

const rows = [];
const tAll = Date.now();
for (let i = 0; i < N; i++) {
  const r = sampleOne(SEED0 + i);
  rows.push(r);
  if (!OUT) {
    console.log('seed=' + r.seed + '  ' + r.ms.toFixed(0) + 'ms  ncol=' + r.ncol +
      '  sym=' + r.sym.toFixed(2) + '  ' + r.snapped.slice(0, 4).join(' '));
  }
}
const total = Date.now() - tAll;

const summary = {
  source: path.relative(ROOT, HTML),
  params: META.params,
  T: META.T,
  pred: META.pred,
  steps: STEPS,
  cfg_w: W,
  qb: QB,
  alpha: ALPHA,
  k: K,
  sym_tau: SYM,
  palette_size: PAL.length,
  n: rows.length,
  ms_mean: total / Math.max(1, rows.length),
  ms_total: total,
  samples: rows,
};

if (OUT) {
  fs.mkdirSync(path.dirname(path.resolve(ROOT, OUT)), { recursive: true });
  fs.writeFileSync(path.resolve(ROOT, OUT), JSON.stringify(summary, null, 1), 'utf8');
  console.error('[jim] 写了 ' + OUT + '（' + rows.length + ' 张，平均 ' +
    summary.ms_mean.toFixed(0) + ' ms/张）');
}
