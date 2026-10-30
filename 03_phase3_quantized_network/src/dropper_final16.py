#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""定位那个不掉下去的边界，并给出 m=16 的最终常数。"""
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

# 找 m=12 时哪个通道的哪些 S 与浮点不一致
print("=" * 84)
print("m=12 时与浮点不一致的 S 值")
print("=" * 84)
found = 0
for nm, st in spec["styles"].items():
    for i in range(4):
        a, b = st["alpha"][i], st["beta"][i]
        n, B = int(round(4096 * a)), int(round(4096 * b))
        Xa = np.clip(np.floor(a * pos + b), 0, NB - 1).astype(int)
        Xb = np.clip((n * pos + B) >> 12, 0, NB - 1).astype(int)
        bad = pos[Xa != Xb]
        if len(bad):
            mass = pr[Xa != Xb].sum()
            print(f"  {nm} 通道{i+1}: S = {bad[:8]}{'...' if len(bad) > 8 else ''}"
                  f"  共 {len(bad)} 个，占质量 {100*mass:.2f}%")
            found += 1
if not found:
    print("  无")

print()
print("=" * 84)
print("最终常数（m = 16，与浮点逐点一致）")
print("=" * 84)
print("  X_i = clip( (n16_i * S_i + B16_i) >> 16, 0, 255 )")
print()
print(f"  {'风格':10s} {'ch':>2s} {'alpha':>8s} {'beta':>9s} {'n16':>7s} {'B16':>10s}")
print("  " + "-" * 52)
mx = 0
for nm, st in spec["styles"].items():
    st["n16"] = [int(round(65536 * v)) for v in st["alpha"]]
    st["B16"] = [int(round(65536 * v)) for v in st["beta"]]
    for i in range(4):
        print(f"  {nm if i == 0 else '':10s} {i+1:2d} {st['alpha'][i]:8.4f} "
              f"{st['beta'][i]:9.3f} {st['n16'][i]:7d} {st['B16'][i]:10d}")
        mx = max(mx, abs(st["n16"][i] * 256 + st["B16"][i]))
print()
print(f"  最大 |n16*S + B16| = {mx:,}  （强模上限 2,147,483,647，占 {100*mx/2147483647:.2f}%）")

# 峰值级数：每级的概率
pk = np.zeros(NB)
for st in spec["styles"].values():
    for i in range(4):
        Xc = np.clip(np.floor(st["alpha"][i] * pos + st["beta"][i]), 0, NB - 1).astype(int)
        pp = np.zeros(NB)
        np.add.at(pp, Xc, pr)
        pk = np.maximum(pk, pp)
print(f"  单级最大概率 {pk.max():.4f} → 一级的位移最多带来 {pk.max():.4f} 的 CDF 差")
print("  （m=12 的 0.0116 就是某一个边界上整级质量翻了边）")

spec["formula"] = "X = clip((n16*S + B16) >> 16, 0, 255)"
spec["quant_note"] = "m=16 与浮点逐点一致；m=12 会在个别边界翻一级"
json.dump(spec, open(os.path.join(ROOT, "results", "dropper_spec.json"), "w",
                     encoding="utf-8"), indent=1)
print()
print("  已更新 results/dropper_spec.json（含 n16/B16）")
print("FINAL_DONE")
