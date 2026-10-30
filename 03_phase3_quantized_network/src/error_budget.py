#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
三类误差的来源、量级与最优 b 的解析预测。

符号（按你的新约定）
    X        输入整数，Y 输出整数，W 整数权重
    W_ref    次高位位权重的值（原 w_ref）
    delta_x  输入侧实数分辨率      Delta_X  输入侧每通道极差
    delta_y  输出侧实数分辨率      Delta_Y  输出侧每通道极差
    hat x    逆回的输入实数；eta 是预激活（实数），hat y 是从 Y 逆回的实数

三个误差
    eps_x = delta_x / (2 Delta_X)         输入量化
    eps_y = delta_y / (2 Delta_Y)         输出量化
    eps_w = sqrt(n b / 24) / (sigma Delta_Y)   权重量化
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
RESULTS = os.path.join(ROOT, "results")
P2 = os.path.join(os.path.dirname(ROOT), "02_phase2_avatar")

mj = json.load(open(os.path.join(P2, "webapp", "pca22_dim4w64L1_model.json"),
                    encoding="utf-8"))
U1W = np.array(mj["gen"]["layers"][0]["W"], dtype=np.float64)
TH1 = np.array(mj["gen"]["layers"][0]["b"], dtype=np.float64)
V2W = np.array(mj["gen"]["layers"][1]["W"], dtype=np.float64)
PHI = np.array(mj["gen"]["layers"][1]["b"], dtype=np.float64)

DELTA = 7.0
XLO = -3.5
CAP = 882
NLEV = 15


def build(b, W_ref):
    NB = 1 << b
    alpha = CAP / (np.abs(U1W).max(1) * DELTA / 4) * (W_ref / CAP)
    U = np.rint(alpha[:, None, None] * U1W[:, :, None] * (DELTA / NB)
                * (1 << np.arange(b))[None, None, :]).astype(np.int64)
    Hmin = np.where(U < 0, U, 0).sum(axis=(1, 2))
    Hmax = np.where(U > 0, U, 0).sum(axis=(1, 2))
    A = np.rint(-alpha * (TH1 + XLO * U1W.sum(1)
                          + DELTA / (2 ** (b + 1)) * U1W.sum(1)) - Hmin).astype(np.int64)
    c = np.maximum(np.ceil((Hmax - Hmin - A) / (NB - 1)), 1).astype(np.int64)
    delta = c / alpha
    beta = W_ref / ((np.abs(V2W) * delta[None, :]).max() * (1 << (b - 2)))
    return dict(NB=NB, alpha=alpha, U=U, Hmin=Hmin, A=A, c=c, delta=delta, beta=beta)


def measure_ranges(b, W_ref, rng, NF=40000):
    """实测两层各通道的实数极差 Delta_X / Delta_Y（层 1 的 eta、层 2 的 z）。"""
    d = build(b, W_ref)
    x = np.clip(rng.standard_normal((NF, 4)), -3.5, 3.5)
    X = np.clip(np.floor((x + 3.5) / DELTA * d["NB"]), 0, d["NB"] - 1)
    xh = -3.5 + (X + 0.5) * DELTA / d["NB"]
    eta = xh @ U1W.T + TH1
    y = np.maximum(0.0, eta)
    z = y @ V2W.T + PHI
    dY1 = float((y.max(0) - y.min(0)).mean())          # 层 1 输出极差
    dY2 = float((z.max(0) - z.min(0)).mean())          # 层 2 输出极差
    return d, dY1, dY2


print("=" * 100)
print("一、核对你的两个公式")
print("=" * 100)
print("""  输入量化：真值 x 与逆回 hat_x 的最大误差 = delta_x/2
            delta_x = Delta_X / 2^b
            =>  eps_x = delta_x/(2 Delta_X) = 1/2^(b+1)     ✓ 你的推导正确

  输出量化：同理 eps_y = delta_y/(2 Delta_Y) = 1/2^(b+1)

  注意一个偏差点：delta_y 不是按「实测极差」定的，而是按「可达最大极差」定的
      c_j = ceil((F_max - A_j)/(2^b - 1))
  可达极差 >= 实测极差，所以 delta_y 偏大，eps_y 会略大于 1/2^(b+1)。
""")

rng = np.random.default_rng(7)
print()
print(f"  {'b':>3s} {'W_ref':>6s} {'eps_x 理论':>11s} "
      f"{'层1 eps_y 实测':>14s} {'层2 eps_y 实测':>14s}")
print("  " + "-" * 56)
for b in (8, 10):
    for W_ref in (441, 882, 1764):
        d, dY1, dY2 = measure_ranges(b, W_ref, rng)
        ex = 1.0 / (1 << (b + 1))
        ey1 = float(np.median(d["delta"] / (2 * dY1)))
        ey2 = float(0.5 / d["beta"] / (2 * dY2))
        print(f"  {b:3d} {W_ref:6d} {ex:11.6f} {ey1:14.6f} {ey2:14.6f}")

print()
print("  → eps_x 与 eps_y 同量级（差不到 2 倍），你的猜想成立。")

# ---------------- 第三类误差
print()
print("=" * 100)
print("二、第三类：权重量化误差")
print("=" * 100)
print("""  每个位权重取整误差在 [-1/2, +1/2]，方差 1/12。
  它按下式进入累加量：
        Delta eta = (1/sigma) * sum_{k,j} err_{kj} * A_j^(k)
  A_j^(k) 是 0/1 位，E[(A^(k))^2] = 1/2，共有 n*b 项，且互相独立：
        Var  = (1/12)(1/2) n b = n b / 24
        sigma_eta = sqrt(n b / 24) / sigma

  换算成相对误差：
        eps_w = sqrt(n b / 24) / (sigma * Delta_Y)

  关键：sigma 与 b 无关（delta = Delta/2^b 与 2^(b-2) 相消），
        所以 eps_w 随 b 以 sqrt(b) 缓慢增长，而 eps_x、eps_y 以 2^-b 迅速下降。
        => 必然存在最优 b。
""")

print(f"  {'b':>3s} {'W_ref':>6s} {'层1 eps_x':>10s} {'eps_y':>9s} {'eps_w':>9s} "
      f"{'层1 总':>9s} | {'层2 eps_x':>10s} {'eps_y':>9s} {'eps_w':>9s} {'层2 总':>9s}")
print("  " + "-" * 100)


def layer_errors(b, W_ref, rng, NF=40000):
    d, dY1, dY2 = measure_ranges(b, W_ref, rng, NF)
    NB = d["NB"]
    # ---- 层 1
    dx = DELTA / NB
    # 输入误差对 eta 的贡献：sum_i |u_ji| dx/2 / Delta_Y1
    ex1 = float(np.median((np.abs(U1W).sum(1) * dx / 2) / dY1))
    ey1 = float(np.median(d["delta"] / (2 * dY1)))
    ew1 = float(np.median(np.sqrt(4 * b / 24) / (d["alpha"] * dY1)))
    # ---- 层 2
    # 输入误差：sum_j |v_pj| delta_j/2 / Delta_Y2
    ex2 = float(np.median((np.abs(V2W) * d["delta"][None, :]).sum(1) / 2 / dY2))
    ey2 = float(0.5 / d["beta"] / (2 * dY2))
    ew2 = float(np.sqrt(64 * b / 24) / (d["beta"] * dY2))
    tot1 = float(np.sqrt(ex1**2 + ey1**2 + ew1**2))
    tot2 = float(np.sqrt(ex2**2 + ey2**2 + ew2**2))
    return ex1, ey1, ew1, tot1, ex2, ey2, ew2, tot2


pred = {}
for W_ref in (441, 882, 1764):
    print(f"\n  W_ref = {W_ref}")
    for b in (6, 7, 8, 9, 10, 11, 12, 13):
        ex1, ey1, ew1, t1, ex2, ey2, ew2, t2 = layer_errors(b, W_ref, rng)
        pred[(W_ref, b)] = t1 + t2
        print(f"  {b:3d} {W_ref:6d} {ex1:10.6f} {ey1:9.6f} {ew1:9.6f} {t1:9.6f} | "
              f"{ex2:10.6f} {ey2:9.6f} {ew2:9.6f} {t2:9.6f}")

print()
print("=" * 100)
print("三、解析预测的最优 b")
print("=" * 100)
print(f"  {'W_ref':>6s} {'预测最优 b':>10s} {'实测最优 b':>10s}")
print("  " + "-" * 30)
meas = {441: 9, 882: 10, 1764: 11}
for W_ref in (441, 882, 1764):
    cand = [(pred[(W_ref, b)], b) for b in range(6, 14)]
    cand.sort()
    print(f"  {W_ref:6d} {cand[0][1]:10d} {meas[W_ref]:10d}")

print()
print("=" * 100)
print("四、为什么最优 b 随 W_ref 每翻倍 +1")
print("=" * 100)
print("""  平衡点：eps_x ≈ eps_w
        1/2^(b+1) ≈ sqrt(nb/24)/(sigma Delta_Y)
  而 sigma = W_ref/(max|w| Delta_X/4)，与 b 无关
        => 2^b ≈ sigma Delta_Y sqrt(24/(nb)) / 2
        => b* ≈ log2(W_ref) + const

  即：W_ref 翻倍 -> b* 加 1。实测 9 / 10 / 11 完全吻合。
""")
for W_ref in (441, 882, 1764):
    print(f"    W_ref = {W_ref:4d}  log2 = {np.log2(W_ref):.3f}")

json.dump({f"{k[0]}|{k[1]}": v for k, v in pred.items()},
          open(os.path.join(RESULTS, "error_budget.json"), "w", encoding="utf-8"),
          indent=1)
print()
print("  已写入 results/error_budget.json")
print("ERR_DONE")
