#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
干净地分解权重量化误差：逐位算取整残差的贡献。

位 k 的取整残差
    err_k = W^(k) - sigma * w * delta * 2^k          （在 [-0.5, 0.5]）
它对 eta 的贡献
    d_eta_k = (1/sigma) * sum_i err_k(i) * A_i^(k)
因为 A^(k) 是 0/1，各项等权，所以各位贡献的方差应当相同 —— 这是要验证的。
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
mj = json.load(open(os.path.join(P2, "webapp", "pca22_dim4w64L1_model.json"),
                    encoding="utf-8"))
U1W = np.array(mj["gen"]["layers"][0]["W"], dtype=np.float64)
TH1 = np.array(mj["gen"]["layers"][0]["b"], dtype=np.float64)
DELTA, XLO, CAP, N = 7.0, -3.5, 882, 4
rng = np.random.default_rng(11)

for b, W_ref in ((10, 882), (8, 882)):
    NB = 1 << b
    alpha = CAP / (np.abs(U1W).max(1) * DELTA / 4) * (W_ref / CAP)
    exact = alpha[:, None, None] * U1W[:, :, None] * (DELTA / NB) \
        * (1 << np.arange(b))[None, None, :]          # sigma*w*delta*2^k
    U = np.rint(exact).astype(np.int64)
    err = U - exact                                    # 取整残差，每项在 ±0.5
    X = rng.integers(0, NB, size=(40000, N))
    bits = ((X[:, :, None] >> np.arange(b)[None, None, :]) & 1).astype(np.int64)

    print("=" * 90)
    print(f"逐位取整残差对 eta 的贡献   (W_ref={W_ref}, b={b})")
    print("=" * 90)
    print(f"  {'位 k':>4s} {'该位最大权重':>12s} {'残差 std':>10s} "
          f"{'贡献 std':>11s} {'贡献方差':>11s} {'理论 sqrt(n/24)/sigma':>22s}")
    print("  " + "-" * 80)
    tot_var = 0.0
    for k in range(b):
        d_eta = np.einsum("ni,ji->nj", bits[:, :, k], err[:, :, k]) / alpha[None, :]
        sv = float(d_eta.std(0).mean())
        var = sv ** 2
        tot_var += var
        theo = float(np.median(np.sqrt(N / 24) / alpha))
        print(f"  {k:4d} {int(np.abs(U[:, :, k]).max()):12d} "
              f"{float(err[:, :, k].std()):10.4f} {sv:11.6f} {var:11.8f} "
              f"{theo:22.6f}")
    print(f"  {'合计':>4s} {'':>12s} {'':>10s} {'':>11s} {tot_var:11.8f}")
    print(f"  实测总 std = {np.sqrt(tot_var):.6f}")
    print(f"  若各位独立等权，理论总 std = {float(np.median(np.sqrt(N*b/24)/alpha)):.6f}")
    print()

print("=" * 90)
print("结论")
print("=" * 90)
print("  每一位的取整残差都乘同量级的 0/1 位值，所以：")
print("    各位贡献方差相同  =>  总方差 ∝ b  =>  总 std ∝ sqrt(b)")
print("  不是等比收敛，低位并不会「无关紧要」。")
print("  但它们也不比高位更重要——每位的量级完全一样。")
print("PERBIT2_DONE")
