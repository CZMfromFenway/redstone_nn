#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
两件事
    1 逐元素核对：编译器产物 与 之前手算的公式 是否完全一致
    2 测 w_ref = 882 与 w_ref = 441
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
sys.path.insert(0, HERE)
from compiler import Layer, compile_network, integer_forward  # noqa: E402

P2 = os.path.join(os.path.dirname(ROOT), "02_phase2_avatar")
mj = json.load(open(os.path.join(P2, "webapp", "pca22_dim4w64L1_model.json"),
                    encoding="utf-8"))
U1W = np.array(mj["gen"]["layers"][0]["W"], dtype=np.float64)
TH1 = np.array(mj["gen"]["layers"][0]["b"], dtype=np.float64)
V2W = np.array(mj["gen"]["layers"][1]["W"], dtype=np.float64)
PHI = np.array(mj["gen"]["layers"][1]["b"], dtype=np.float64)

B = 8
NB = 1 << B
XLO, XHI = -3.5, 3.5
DELTA = XHI - XLO


def build(w_ref=882):
    return [
        Layer(U1W, TH1, "relu", comparable=False, out_bits=8, name="层1 4→64"),
        Layer(V2W, PHI, "clip", comparable=True, lam=np.full(32, 15.0), name="层2 64→32"),
    ], w_ref


# ======================================================================
# 1  手算（此前那套脚本里的公式，原样重写一遍）
# ======================================================================
def hand(w_ref=882):
    LIM = w_ref
    alpha = LIM / (np.abs(U1W).max(1) * DELTA / 4)
    U = np.rint(alpha[:, None, None] * U1W[:, :, None] * (DELTA / NB)
                * (1 << np.arange(B))[None, None, :]).astype(np.int64)
    Hmin = np.where(U < 0, U, 0).sum(axis=(1, 2))
    Hmax = np.where(U > 0, U, 0).sum(axis=(1, 2))
    A = np.rint(-alpha * (TH1 + XLO * U1W.sum(1)
                          + DELTA / (2 ** (B + 1)) * U1W.sum(1)) - Hmin).astype(np.int64)
    c = np.maximum(np.ceil((Hmax - Hmin - A) / (NB - 1)), 1).astype(np.int64)
    delta = c / alpha
    beta = LIM / ((np.abs(V2W) * delta[None, :]).max() * (1 << (B - 2)))
    V = np.rint(beta * V2W[:, :, None] * delta[None, :, None]
                * (1 << np.arange(B))[None, None, :]).astype(np.int64)
    Gmin = np.where(V < 0, V, 0).sum(axis=(1, 2))
    Bp = np.rint(-beta * (PHI + 0.5 * (V2W * delta[None, :]).sum(1)) - Gmin).astype(np.int64)
    T = int(round(15 * beta))
    return dict(alpha=alpha, U=U, Hmin=Hmin, A=A, c=c, delta=delta, beta=beta,
                V=V, Gmin=Gmin, Bp=Bp, T=T)


print("=" * 100)
print("1  逐元素核对：编译器  vs  手算")
print("=" * 100)
for w_ref in (882, 441):
    layers, wr = build(w_ref)
    comp = compile_network(layers, XLO, XHI, B, verbose=False, w_ref=wr)
    Hd = hand(w_ref)
    L1, L2 = comp[0], comp[1]
    checks = [
        ("σ_1 (α_j)", L1["sigma"], Hd["alpha"]),
        ("W_1 位权重", L1["W"], Hd["U"]),
        ("C_1,min (H_j,min)", L1["Cmin"], Hd["Hmin"]),
        ("P_1 (A_j)", L1["P"], Hd["A"]),
        ("c_1 (c_j)", L1["c"], Hd["c"]),
        ("δ_out (δ_j)", L1["delta_out"], Hd["delta"]),
        ("σ_2 (β)", L2["sigma"], np.full(32, Hd["beta"])),
        ("W_2 位权重", L2["W"], Hd["V"]),
        ("C_2,min (G_p,min)", L2["Cmin"], Hd["Gmin"]),
        ("P_2 (B_p)", L2["P"], Hd["Bp"]),
        ("T_2 (15β)", L2["T"], np.full(32, Hd["T"])),
    ]
    print(f"\n  w_ref = {w_ref}")
    ok = True
    for nm, got, ref in checks:
        same = np.array_equal(np.asarray(got), np.asarray(ref))
        ok &= same
        d = np.max(np.abs(np.asarray(got, float) - np.asarray(ref, float)))
        print(f"    {nm:20s} {'完全一致' if same else '不一致'}   最大差 {d:g}")
    print(f"    → {'全部一致 ✓' if ok else '有不一致 ✗'}")

# ======================================================================
# 2  w_ref 对比
# ======================================================================
print()
print("=" * 100)
print("2  w_ref = 882 与 w_ref = 441")
print("=" * 100)
rng = np.random.default_rng(41)
NF = 20000
xs = np.clip(rng.standard_normal((NF, 4)), XLO, XHI)
Xq = np.clip(np.floor((xs - XLO) / DELTA * NB), 0, NB - 1).astype(np.int64)

zref = np.maximum(0.0, (XLO + (Xq + 0.5) * DELTA / NB) @ U1W.T + TH1) @ V2W.T + PHI
v = np.clip(zref, 0, 15)
lo, hi = v.min(1, keepdims=True), v.max(1, keepdims=True)
refL = np.clip(np.rint((v - lo) / np.maximum(hi - lo, 1e-9) * 15), 0, 15).astype(int)

print(f"  {'w_ref':>7s} {'最高位|W|':>10s} {'σ_1 中位':>10s} {'σ_2':>8s} {'T':>6s} "
      f"{'N 上界':>8s} {'端到端误差 平均':>14s} {'最大':>8s} {'16级像素差':>10s}")
print("  " + "-" * 92)
for w_ref in (882, 441):
    layers, wr = build(w_ref)
    comp = compile_network(layers, XLO, XHI, B, verbose=False, w_ref=wr)
    L2 = comp[1]
    Zi = integer_forward(comp, Xq)
    phys = (Zi + 0.5) / L2["sigma"][0]
    e = phys - np.clip(zref, 0, 15)
    Li = Zi * 15 // np.maximum(L2["T"][0], 1)
    # 真正的 16 级要走后处理，这里只比物理误差
    print(f"  {w_ref:7d} {int(np.abs(comp[0]['W']).max()):10d} "
          f"{np.median(comp[0]['sigma']):10.1f} {L2['sigma'][0]:8.3f} "
          f"{int(L2['T'][0]):6d} {int(L2['Nmax'].max()):8,} "
          f"{np.abs(e).mean():14.4f} {np.abs(e).max():8.4f} {'—':>10s}")

print()
print("  说明：w_ref=882 让最高位到 1764（需许可），w_ref=441 让全部位不超过 882。")
print("        w_ref 减半 → σ 减半 → 位权重减半 → 累加量程减半 → 有效级数减半。")
print("VERIFY_COMPILER_DONE")
