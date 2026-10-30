#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""复刻官方 verify_teachers.py 的前向，定位我的实现差异"""
import numpy as np, json, os
from pathlib import Path

ROOT = Path(r"D:\czm我的世界\文件\Comparator Network\DT_Redstone_Research_Archive_20261001\DT_baseline_recovery")
X_official = np.indices((16,) * 4, dtype=np.int8).reshape(4, -1).T
print("官方 X shape:", X_official.shape)
print("X[0] =", X_official[0].tolist(), " (应为 0,0,0,0)")
print("X[1] =", X_official[1].tolist())
print("X[15] =", X_official[15].tolist())
print("X[16] =", X_official[16].tolist())

p = ROOT / "teachers" / "N4_MLP_T1"
y = np.load(str(p) + ".npy")
m = json.loads(Path(str(p) + ".json").read_text(encoding="utf-8"))
W1 = np.array(m["W1"]); b1 = np.array(m["b1"]); W2 = np.array(m["W2"]); b2 = m["b2"]
print("\nW1 shape:", W1.shape, " W2 shape:", W2.shape, " y shape:", y.shape)

yp = np.tanh(((X_official.astype(float) - 7.5) / 7.5) @ W1.T + b1) @ W2 + b2 >= 0
print("\n官方前向:")
print("  yp 正类数 =", int(yp.sum()), "  y 正类数 =", int(y.sum()))
print("  不一致 =", int(np.sum(y != yp)))
print("  yp 与 y 的一致率 =", float((yp == y).mean()))

# 我的口径
X_mine = np.indices((16,) * 4, dtype=np.int8).reshape(4, -1).T
print("\n我的 X 与官方 X 完全相同:", np.array_equal(X_mine, X_official))
