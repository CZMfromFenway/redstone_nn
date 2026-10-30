# -*- coding: utf-8 -*-
"""端到端：Python 的整数前向 N 值，供与仿真器头部输出比对。"""
import json
import os
import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = r"E:\redstone_nn"
MODEL = os.path.join(ROOT, "02_phase2_avatar", "webapp",
                     "pca22_dim4w64L1_model.json")
XMIN, XMAX, INB = -3.5, 3.5, 8
LIMIT = 882

mj = json.load(open(MODEL, encoding="utf-8"))
L1 = np.array(mj["gen"]["layers"][0]["W"], dtype=np.float64)
delta = np.full(4, (XMAX - XMIN) / (1 << INB))
g = np.abs(L1) * delta[None, :]
sig = LIMIT / (g.max(axis=1) * (1 << (INB - 2)))
bits = np.arange(INB)
W = np.rint(sig[:, None, None] * L1[:, :, None] * delta[None, :, None]
            * (1 << bits)[None, None, :]).astype(np.int64)
Cmin = np.where(W < 0, W, 0).sum(axis=(1, 2))
Cmax = np.where(W > 0, W, 0).sum(axis=(1, 2))
Nmax = Cmax - Cmin

rng = np.random.default_rng(2024)
cases = []
for ch in (0, 1, 3, 8, 20, 31, 63):
    for _ in range(6):
        xs = rng.integers(0, 2, 4)                 # 8 位输入的低 4 位（每维 0/1）
        A = np.zeros((1, 4), dtype=np.int64)
        for j in range(4):
            A[0, j] = 0xFF if int(xs[j]) else 0     # 该维全部位相同（对应仿真器的单一 x）
        xbit = [int(v) for v in xs]                 # 每维的最低位 = 我们的 x
        bts = ((A[:, :, None] >> np.arange(INB)[None, None, :]) & 1).astype(np.int64)
        C = np.einsum("njk,ijk->ni", bts, W)
        N = int((C - Cmin[None, :])[0, ch])
        cases.append(dict(ch=ch, x=xbit, N=N,
                          S=[int(15 + abs(W[ch, j, INB - 1])) for j in range(4)],
                          W_msb=[int(W[ch, j, INB - 1]) for j in range(4)]))

out = {"cases": cases,
       "Cmin": Cmin.tolist(), "Nmax": Nmax.tolist()}
p = os.path.join(ROOT, "03_phase3_quantized_network", "results",
                 "py_forward_cases.json")
json.dump(out, open(p, "w", encoding="utf-8"))
print(f"  写出 {p}  共 {len(cases)} 个用例")
for c in cases[:4]:
    print(f"    ch={c['ch']} x={c['x']} N={c['N']} S_msb={c['S']}")
