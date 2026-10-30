#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
按层独立的 W_ref 与位宽。

    b_x  输入量化位数（x → X）
    b_y  Y 的位数（层1 输出 / 层2 输入）
    W_ref1  层1 的次高位
    W_ref2  层2 的次高位

容器约束（位权重当 LUT 阶跃量，单容器上限 882）
    层1: 2·W_ref1 需环(1 个)，W_ref1 ≤ 882
    层2: 2·W_ref2 与 W_ref2 需环(2 个)，W_ref2/2 ≤ 882 → W_ref2 ≤ 1764

环数 = 64·[层1 有环] + 2·32·[层2 双环]
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

DELTA, XLO, CAP, NLEV = 7.0, -3.5, 882, 15


def compile2(bx, by, W1, W2):
    NX, NY = 1 << bx, 1 << by
    alpha = W1 / (np.abs(U1W).max(1) * DELTA / 4)
    U = np.rint(alpha[:, None, None] * U1W[:, :, None] * (DELTA / NX)
                * (1 << np.arange(bx))[None, None, :]).astype(np.int64)
    Hmin = np.where(U < 0, U, 0).sum(axis=(1, 2))
    Hmax = np.where(U > 0, U, 0).sum(axis=(1, 2))
    kappa = TH1 + XLO * U1W.sum(1) + DELTA / (2 ** (bx + 1)) * U1W.sum(1)
    P = np.rint(-alpha * kappa - Hmin).astype(np.int64)
    c = np.maximum(np.ceil((Hmax - Hmin - P) / (NY - 1)), 1).astype(np.int64)
    delta = c / alpha
    beta = W2 / ((np.abs(V2W) * delta[None, :]).max() * (1 << (by - 2)))
    V = np.rint(beta * V2W[:, :, None] * delta[None, :, None]
                * (1 << np.arange(by))[None, None, :]).astype(np.int64)
    Gmin = np.where(V < 0, V, 0).sum(axis=(1, 2))
    Bp = np.rint(-beta * (PHI + 0.5 * (V2W * delta[None, :]).sum(1)) - Gmin).astype(np.int64)
    T = int(round(NLEV * beta))
    # 环数：逐位看该位的最大权重是否超过单容器上限 882
    rings1 = sum(64 for k in range(bx) if int(np.abs(U[:, :, k]).max()) > CAP)
    rings2 = sum(32 for k in range(by) if int(np.abs(V[:, :, k]).max()) > CAP)
    ok1 = ok2 = True
    return dict(bx=bx, by=by, W1=W1, W2=W2, alpha=alpha, U=U, Hmin=Hmin, P=P, c=c,
                delta=delta, beta=beta, V=V, Gmin=Gmin, Bp=Bp, T=T,
                ok1=ok1, ok2=ok2, rings=rings1 + rings2,
                rings1=rings1, rings2=rings2,
                bits_weights=U.size + V.size)


def forward(d, x):
    NX, NY = 1 << d["bx"], 1 << d["by"]
    X = np.clip(np.floor((x - XLO) / DELTA * NX), 0, NX - 1).astype(np.int64)
    b1 = ((X[:, :, None] >> np.arange(d["bx"])[None, None, :]) & 1).astype(np.int64)
    N = np.einsum("nik,jik->nj", b1, d["U"]) - d["Hmin"][None, :]
    Y = np.clip(np.floor_divide(N - d["P"][None, :], d["c"][None, :]), 0, NY - 1)
    b2 = ((Y[:, :, None] >> np.arange(d["by"])[None, None, :]) & 1).astype(np.int64)
    G = np.einsum("njk,pjk->np", b2, d["V"])
    # 层2 输出截断（显示范围 [0,15β]）
    Z = np.clip(G - d["Gmin"][None, :] - d["Bp"][None, :], 0, int(round(NLEV * d["beta"])))
    return Z


def levels(Z):
    m, n = Z.max(1, keepdims=True), Z.min(1, keepdims=True)
    dZ = np.maximum(m - n, 1)
    return np.clip(np.floor_divide((Z - n) * NLEV + (dZ >> 1), dZ), 0, NLEV)


rng = np.random.default_rng(41)
NF = 6000
xs = np.clip(rng.standard_normal((NF, 4)), -3.5, 3.5)
Xq = np.clip(np.floor((xs + 3.5) / DELTA * 256), 0, 255)
zref = (np.maximum(0.0, (-3.5 + (Xq + 0.5) * DELTA / 256) @ U1W.T + TH1) @ V2W.T + PHI)
v = np.clip(zref, 0, 15)
lo, hi = v.min(1, keepdims=True), v.max(1, keepdims=True)
Lref = np.clip(np.rint((v - lo) / np.maximum(hi - lo, 1e-9) * 15), 0, 15).astype(int)

print("=" * 100)
print("按层独立 W_ref：层1 单环(882) + 层2 双环(1764)   vs   全局均匀")
print("=" * 100)

CONFIGS = [
    ("均匀 882",   882, 882),
    ("均匀 1764", 1764, 1764),
    ("层1=882 层2=1764", 882, 1764),
    ("层1=441 层2=1764", 441, 1764),
]
for nm, W1, W2 in CONFIGS:
    print(f"\n  【{nm}】 W_ref1={W1}, W_ref2={W2}")
    print(f"  {'b_x':>4s} {'b_y':>4s} {'环数':>5s} {'位权重数':>9s} "
          f"{'不同像素':>9s} {'平均|差|':>8s}")
    print("  " + "-" * 50)
    best = None
    for bx in (8, 9, 10, 11, 12):
        for by in (8, 9, 10, 11, 12, 13):
            d = compile2(bx, by, W1, W2)
            if not (d["ok1"] and d["ok2"]):
                continue
            L = levels(forward(d, xs))
            diff = 100 * float((L != Lref).mean())
            mean = float(np.abs(L - Lref).mean())
            flag = ""
            if best is None or diff < best[0]:
                best = (diff, bx, by, d["rings"])
                flag = " ←"
            if by in (8, 10, 12) or flag:
                print(f"  {bx:4d} {by:4d} {d['rings']:5d} {d['bits_weights']:9d} "
                      f"{diff:8.2f}% {mean:8.4f}{flag}")
    print(f"  → 最优 b_x={best[1]}, b_y={best[2]}, {best[3]} 环, {best[0]:.2f}%")

print()
print("=" * 100)
print("结论对照")
print("=" * 100)
rows = []
for nm, W1, W2 in CONFIGS:
    bb = None
    for bx in (8, 9, 10, 11, 12):
        for by in (8, 9, 10, 11, 12, 13):
            d = compile2(bx, by, W1, W2)
            if not (d["ok1"] and d["ok2"]):
                continue
            diff = 100 * float((levels(forward(d, xs)) != Lref).mean())
            if bb is None or diff < bb[0]:
                bb = (diff, bx, by, d["rings"], d["bits_weights"])
    rows.append((nm, W1, W2, bb))
    print(f"  {nm:22s} 最优 b_x={bb[1]}, b_y={bb[2]}  环 {bb[3]:3d}  "
          f"权重 {bb[4]:6d}  误差 {bb[0]:5.2f}%")
print()
print("E[1/r_p] 随通道数的标度（用子集抽样）")
d = compile2(10, 10, 882, 882)
q = (np.abs(V2W) * d["delta"][None, :]).max(axis=1)
sub = np.random.default_rng(1)
print(f"  {'通道数 m':>8s} {'E[1/r_p]':>10s} {'r 最小':>8s}")
print("  " + "-" * 30)
for m in (4, 8, 16, 32):
    vals = []
    for _ in range(200):
        idx = sub.choice(32, size=m, replace=False)
        qq = q[idx]
        vals.append(float((qq.max() / qq).mean()))
    print(f"  {m:8d} {np.mean(vals):10.3f} {np.mean([r.min() for r in [q[idx]/q[idx].max() for idx in [sub.choice(32, size=m, replace=False)]]]):8.3f}")

json.dump({nm: {"bx": bb[1], "by": bb[2], "rings": bb[3],
                "weights": bb[4], "diff": bb[0]}
           for nm, _, _, bb in rows},
          open(os.path.join(RESULTS, "mixed_wref.json"), "w", encoding="utf-8"),
          indent=1)
print()
print("  已写入 results/mixed_wref.json")
print("MIXED_DONE")
