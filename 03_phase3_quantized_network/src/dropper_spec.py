#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
绿色方案（INDEP）的完整参数表。

结构
    每风格 4 个通道，各一套独立的 13 个投掷器（参数完全相同），得到 S_i ∈ {0..256}
    S_i 代表 z_i = (S_i − 128)/36.466
    对角变换 x_i = mu'_i + sigma'_i · z_i
    量化        X_i = clip( floor( (256/7)(x_i + 3.5) ), 0, 255 )
                = clip( floor( alpha_i·S_i + beta_i ), 0, 255 )
    整数实现    X_i = clip( (n_i·S_i + B_i) >> 8, 0, 255 )
                n_i = round(256 alpha_i),  B_i = round(256 beta_i)
"""
import json
import os
import sys
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
P2 = os.path.join(os.path.dirname(ROOT), "02_phase2_avatar")
RESULTS = os.path.join(ROOT, "results")
styles = json.load(open(os.path.join(P2, "webapp", "styles_gauss.json"),
                        encoding="utf-8"))["styles"]

DELTA = 7.0
NB = 256
SIG_DROP = 36.466
WDROP = [1, 6, 7, 7, 7, 9, 10, 11, 11, 12, 14, 15, 18]
PV = np.array([4, 1, 4]) / 9.0

rng = np.random.default_rng(31)
NG = 300000
spec = {}

print("=" * 104)
print("投掷器（每通道一套，13 个，参数相同）")
print("=" * 104)
print(f"  权重 {WDROP}   Σ={sum(WDROP)}  Σ²={sum(v*v for v in WDROP)}")
print(f"  每个 9 格，{{0,1,2}} 各占 4/1/4 格；单投掷器 E[v]=1, Var=0.8889")
print(f"  求和 S ∈ [0, 256]，代表 z = (S − 128)/{SIG_DROP}")
print(f"  实测 σ_S = {SIG_DROP:.3f}（目标 36.460）")

for e in styles:
    nm = e["name"]
    mu = np.array(e["mu"], dtype=float)
    L = np.array(e["chol"], dtype=float)
    x = rng.standard_normal((NG, 4)) @ L.T + mu
    xa = x[np.all(np.abs(x) <= 3.5, axis=1)]
    mup = xa.mean(0)
    sdp = xa.std(0)
    alpha = (NB / DELTA) * sdp / SIG_DROP
    beta = (NB / DELTA) * (mup + 3.5) - 128.0 * alpha
    n = np.round(NB * alpha).astype(np.int64)
    B = np.round(NB * beta).astype(np.int64)
    spec[nm] = dict(mu=[float(v) for v in mup], sigma=[float(v) for v in sdp],
                    alpha=[float(v) for v in alpha], beta=[float(v) for v in beta],
                    n=[int(v) for v in n], B=[int(v) for v in B],
                    mean_note=float(np.abs(mu).max()))

print()
print("=" * 104)
print("各风格四通道参数")
print("=" * 104)
for nm, s in spec.items():
    print(f"\n  {nm}")
    print(f"    {'通道':>4s} {'mu′':>8s} {'sigma′':>8s} {'alpha':>8s} {'beta':>9s} "
          f"{'n=256a':>7s} {'B=256b':>8s}")
    for i in range(4):
        print(f"    {i+1:4d} {s['mu'][i]:8.4f} {s['sigma'][i]:8.4f} "
              f"{s['alpha'][i]:8.4f} {s['beta'][i]:9.3f} "
              f"{s['n'][i]:7d} {s['B'][i]:8d}")

# ---------------- 验证：整数常数 vs 精确浮点
print()
print("=" * 104)
print("验证：用整数常数 (n, B) 与精确浮点 (alpha, beta) 的 X 分布差异")
print("=" * 104)
print(f"  {'风格':10s} {'通道':>4s} {'均值差':>8s} {'std 比':>8s} {'最大CDF差':>10s}")
print("  " + "-" * 46)

# 复现 S 的分布
pos = np.array([0])
pr = np.array([1.0])
for wi in WDROP:
    v = np.arange(3) * wi
    np_ = (pos[:, None] + v[None, :]).ravel()
    npr = (pr[:, None] * PV[None, :]).ravel()
    up, inv = np.unique(np_, return_inverse=True)
    pos = up
    pr = np.bincount(inv, weights=npr)
Spdf = np.zeros(257)
Spdf[pos] = pr

worst = 0.0
for nm, s in spec.items():
    for i in range(4):
        # 精确：X = floor(alpha S + beta)，在 S 的格点上取值
        Xa = np.clip(np.floor(s["alpha"][i] * pos + s["beta"][i]), 0, NB - 1).astype(int)
        Xb = np.clip((s["n"][i] * pos + s["B"][i]) >> 8, 0, NB - 1).astype(int)
        pa = np.zeros(NB)
        pb = np.zeros(NB)
        np.add.at(pa, Xa, pr)
        np.add.at(pb, Xb, pr)
        cdf = float(np.abs(np.cumsum(pa) - np.cumsum(pb)).max())
        worst = max(worst, cdf)
        print(f"  {nm if i == 0 else '':10s} {i+1:4d} {pb.mean()-pa.mean():+8.3f} "
              f"{pb.std()/pa.std() if pa.std() > 0 else 0:8.4f} {cdf:10.5f}")
print(f"\n  四通道 24 组，整数常数与浮点的最大 CDF 差 = {worst:.5f}")

outp = os.path.join(RESULTS, "dropper_spec.json")
json.dump({"dropper_weights": WDROP, "dropper_slots": {"0": 4, "1": 1, "2": 4},
           "sigma_dropper": SIG_DROP, "S_range": [0, 256],
           "formula": "X = clip((n*S + B) >> 8, 0, 255)",
           "styles": spec}, open(outp, "w", encoding="utf-8"), indent=1)
print(f"\n  已写入 {outp}")
print("SPEC_DONE")
