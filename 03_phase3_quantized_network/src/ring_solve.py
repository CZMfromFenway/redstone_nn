#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
用模拟器反解两个环的烧录值。

约束（用户给的）
  1. 常数尽量小
  2. 白环的最后输出 = 该通道的 N

做法：固定灰环 = K = ΣS，扫描白环初值 W0，
      看右总线顶端 (11,4) 等于 N 时 W0 是多少。
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
from cmp_sim import build, DIRV  # noqa: E402

P2 = os.path.join(os.path.dirname(ROOT), "02_phase2_avatar")
mj = json.load(open(os.path.join(P2, "webapp", "pca22_dim4w64L1_model.json"),
                    encoding="utf-8"))
W1 = np.array(mj["gen"]["layers"][0]["W"], dtype=np.float64)
B1 = np.array(mj["gen"]["layers"][0]["b"], dtype=np.float64)
DELTA, WREF, B, N = 7.0, 882.0, 8, 4
alpha = WREF / (np.abs(W1).max(1) * DELTA / 4)
U = np.rint(alpha[:, None, None] * W1[:, :, None] * (DELTA / (1 << B))
            * (1 << np.arange(B))[None, None, :]).astype(np.int64)
J = 8
S = (15 + np.abs(U[J])).astype(int)
msb = np.abs(U[J, :, B - 1]).tolist()

np.random.seed(1)
inp = (np.random.randint(0, 2, size=(N, B)))
K = int(S.sum())
Ntrue = int((U[J] * inp).sum() - np.minimum(0, U[J]).sum())
print(f"通道 {J}")
print(f"  K = ΣS = {K}")
print(f"  输入 {inp.tolist()}")
print(f"  理论 N = ΣWx − Σmin(0,W) = {Ntrue}")

OUT = (11, 4)
print()
print("=" * 88)
print("扫描白环初值 W0，看右总线顶端 (11,4) = N ?")
print("=" * 88)
print(f"  {'W0':>10s} {'(11,4)':>10s} {'(11,22)':>10s} {'(10,4)':>10s} "
      f"{'(12,23)':>10s} {'差':>10s}")
print("  " + "-" * 66)
best = []
for W0 in (0, K // 4, K // 2, K, 2 * K, Ntrue, K + Ntrue):
    b = build(msb, S, K, W0, inp)
    it, ok = b.run()
    v_out = b.cmp[OUT]["state"] if OUT in b.cmp else -1
    v1 = b.cmp[(11, 22)]["state"]
    v2 = b.cmp[(10, 4)]["state"]
    v3 = b.cmp[(12, 23)]["state"]
    print(f"  {W0:10d} {v_out:10d} {v1:10d} {v2:10d} {v3:10d} "
          f"{v_out - Ntrue:10d}")
    best.append((abs(v_out - Ntrue), W0, v_out))

print()
best.sort()
print(f"  最接近 N 的 W0 = {best[0][1]}，输出 {best[0][2]}，N = {Ntrue}")
print()
print("=" * 88)
print("对照：如果 K 仍用老的（多算 15n）")
print("=" * 88)
Kold = 15 * N * B + int(S.sum())
print(f"  K_old = {Kold}   K_correct = {K}   差 {Kold-K}")
print("SIM2_DONE")
