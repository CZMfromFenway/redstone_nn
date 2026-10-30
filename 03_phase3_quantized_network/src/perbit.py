#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""逐位诊断：保留最高 keep 位时残差 std，判定「等比数列」还是「每位等权」。"""
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
mj = json.load(open(os.path.join(P2, "webapp", "pca22_dim4w64L1_model.json"),
                    encoding="utf-8"))
U1W = np.array(mj["gen"]["layers"][0]["W"], dtype=np.float64)
TH1 = np.array(mj["gen"]["layers"][0]["b"], dtype=np.float64)
DELTA, XLO, CAP, N = 7.0, -3.5, 882, 4
rng = np.random.default_rng(3)


def run(b, W_ref):
    NB = 1 << b
    alpha = CAP / (np.abs(U1W).max(1) * DELTA / 4) * (W_ref / CAP)
    U = np.rint(alpha[:, None, None] * U1W[:, :, None] * (DELTA / NB)
                * (1 << np.arange(b))[None, None, :]).astype(np.int64)
    kappa = TH1 + XLO * U1W.sum(1) + DELTA / (2 ** (b + 1)) * U1W.sum(1)
    X = rng.integers(0, NB, size=(20000, N))
    xh = XLO + (X + 0.5) * DELTA / NB
    eta_exact = xh @ U1W.T + TH1
    bits = ((X[:, :, None] >> np.arange(b)[None, None, :]) & 1).astype(np.int64)
    out = []
    for keep in range(1, b + 1):
        U2 = U.copy()
        U2[:, :, :b - keep] = 0
        C = np.einsum("nik,jik->nj", bits, U2)
        Hmin2 = np.where(U2 < 0, U2, 0).sum(axis=(1, 2))
        P2 = np.rint(-alpha * kappa - Hmin2).astype(np.int64)
        eta_int = (C - Hmin2[None, :] - P2[None, :]) / alpha[None, :]
        out.append((keep, float((eta_int - eta_exact).std(0).mean()),
                    int(np.abs(U[:, :, b - keep]).max())))
    return out


for b, W_ref in ((10, 882), (8, 882)):
    res = run(b, W_ref)
    full = res[-1][1]
    print("=" * 92)
    print(f"保留最高 keep 位时的残差 std   (W_ref={W_ref}, b={b})")
    print("=" * 92)
    print(f"  {'keep':>5s} {'该位最大权重':>12s} {'std(eta)':>11s} {'占比':>8s} "
          f"{'本层增量':>10s} {'增量的平方':>11s}")
    print("  " + "-" * 62)
    prev, incs = 0.0, []
    for keep, s, wtop in res:
        inc = s - prev
        incs.append(inc)
        print(f"  {keep:5d} {wtop:12d} {s:11.6f} {s/full*100:7.1f}% "
              f"{inc:10.6f} {inc*inc:11.8f}")
        prev = s
    print()
    print("  判读：")
    print("   · 若「低位可忽略」（等比），本层增量应迅速趋零")
    print("   · 若「每位等权」，增量的平方应大致恒定")
    print()
    # 从第 2 位起看增量的趋势
    tail = [i for i in incs[1:] if i > 0]
    print(f"  第 2 位以后每层增量：{[f'{v:.5f}' for v in tail]}")
    if tail:
        print(f"  最大/最小 = {max(tail)/min(tail):.2f}"
              f"   （等比 1/2 会迅速趋近于 0；等权约为 1）")
    print()

print("=" * 92)
print("结论")
print("=" * 92)
print("  每一位的取整误差乘的是 0/1 的位值，不是 2^k——")
print("  所以它在累加器里贡献的绝对误差与位数无关，各位等权。")
print("  总误差 ∝ sqrt(位数)，不是等比收敛。")
print("PERBIT_DONE")
