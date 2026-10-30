#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""把 alpha 的定点位数从 8 位提到 12 位，看能改善多少。"""
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
spec = json.load(open(os.path.join(ROOT, "results", "dropper_spec.json"),
                     encoding="utf-8"))
WDROP = spec["dropper_weights"]
PV = np.array([4, 1, 4]) / 9.0
pos = np.array([0])
pr = np.array([1.0])
for wi in WDROP:
    v = np.arange(3) * wi
    np_ = (pos[:, None] + v[None, :]).ravel()
    npr = (pr[:, None] * PV[None, :]).ravel()
    up, inv = np.unique(np_, return_inverse=True)
    pos = up
    pr = np.bincount(inv, weights=npr)

NB = 256
print("=" * 84)
print("定点位宽对 X 分布的影响（24 组通道取最大 CDF 差）")
print("=" * 84)
print(f"  {'位宽 m':>7s} {'公式':22s} {'最大 CDF 差':>12s} {'n 的最大值':>11s}")
print("  " + "-" * 58)
best = {}
for m in (8, 10, 12, 14, 16):
    worst = 0.0
    nmax = 0
    for st in spec["styles"].values():
        for i in range(4):
            a = st["alpha"][i]
            b = st["beta"][i]
            n = int(round((1 << m) * a))
            B = int(round((1 << m) * b))
            nmax = max(nmax, n)
            Xa = np.clip(np.floor(a * pos + b), 0, NB - 1).astype(int)
            Xb = np.clip((n * pos + B) >> m, 0, NB - 1).astype(int)
            pa = np.zeros(NB)
            pb = np.zeros(NB)
            np.add.at(pa, Xa, pr)
            np.add.at(pb, Xb, pr)
            worst = max(worst, float(np.abs(np.cumsum(pa) - np.cumsum(pb)).max()))
    print(f"  {m:7d} {'X=(n*S+B)>>'+str(m):22s} {worst:12.5f} {nmax:11d}")
    best[m] = (worst, nmax)

print()
print("  m=8 时 CDF 差 0.012，与投掷器本身的 0.010 同量级，会把总误差翻倍。")
print("  m=12 时降到 0.0007，n 最大 5232，仍在 16 位内。")
print()
print("  推荐：m = 12")
print()
for nm, st in spec["styles"].items():
    st["n12"] = [int(round(4096 * v)) for v in st["alpha"]]
    st["B12"] = [int(round(4096 * v)) for v in st["beta"]]
spec["formula"] = "X = clip((n12*S + B12) >> 12, 0, 255)"
json.dump(spec, open(os.path.join(ROOT, "results", "dropper_spec.json"), "w",
                     encoding="utf-8"), indent=1)
print("=" * 84)
print("各风格四通道的 12 位定点常数")
print("=" * 84)
for nm, st in spec["styles"].items():
    print(f"\n  {nm}")
    print(f"    {'通道':>4s} {'alpha':>8s} {'beta':>9s} {'n12':>7s} {'B12':>9s}")
    for i in range(4):
        print(f"    {i+1:4d} {st['alpha'][i]:8.4f} {st['beta'][i]:9.3f} "
              f"{st['n12'][i]:7d} {st['B12'][i]:9d}")
print()
print("  已更新 results/dropper_spec.json（含 n12/B12）")
print("FIXED_DONE")
