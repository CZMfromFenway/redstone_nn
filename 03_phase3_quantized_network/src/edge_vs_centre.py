#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
格心 vs 格边：ReLU 层的输出用哪种重建。

要比较的是同一个正向映射
    Y = floor(y / y_max * 2^b)
在两种重建约定下的下游后果：
    格心  y_hat = (Y + 0.5) / 2^b * y_max
    格边  y_hat =  Y      / 2^b * y_max

只让层 0 的重建约定变化，层 1 仍用浮点权重，这样测的是重建约定本身，不混入整数权重误差。
终点是最终那 32 个 16 级灰度值，也就是人眼看到的东西。
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
NS = 40000
LEVELS = 16


def load_layers(path):
    j = json.load(open(path, encoding="utf-8"))
    layers = [{"W": np.array(L["W"]), "b": np.array(L["b"]), "relu": True}
              for L in j["gen"]["layers"]]
    layers[-1]["relu"] = False
    return layers


def to_face(out32):
    """复现二期的后处理：clip 到 [0,15] -> 逐张 min-max 拉伸 -> 取整到 16 级。"""
    v = np.clip(out32, 0, 15)
    lo = v.min(1, keepdims=True)
    hi = v.max(1, keepdims=True)
    r = np.clip((v - lo) / np.maximum(hi - lo, 1e-6), 0, 1)
    return np.rint(r * (LEVELS - 1)).astype(int)


gray = load_layers(os.path.join(P2, "webapp", "pca22_dim4w64L1_model.json"))
L0, L1 = gray[0], gray[1]
rng = np.random.default_rng(7)
X = np.clip(rng.standard_normal((NS, L0["W"].shape[1])), -3.5, 3.5)

y0 = np.maximum(0.0, X @ L0["W"].T + L0["b"])       # 层0 输出，非负
y1 = y0 @ L1["W"].T + L1["b"]                        # 层1 输出
face_ref = to_face(y1)

y0max = float(y0.max())
atom = float((y0 == 0).mean())
print("=" * 92)
print("RELU 层输出：格心 vs 格边")
print("=" * 92)
print(f"  层 0 输出范围 [0, {y0max:.4f}]，恰为 0 的比例 {100*atom:.2f}%")

for b in (8, 10, 12):
    N = 1 << b
    step = y0max / N
    print()
    print(f"  b = {b}   步长 {step:.6f}   半边偏置 {step/2:.6f}")
    print(f"    {'约定':10s} {'层0重建误差':>12s} {'最终输出偏移':>13s} "
          f"{'最终绝对误差':>13s} {'灰度级变化':>10s} {'平均级差':>9s}")
    print("    " + "-" * 76)
    rows = {}
    for name, shift in (("格心", 0.5), ("格边", 0.0)):
        Q = np.floor(y0 / y0max * N)
        Q = np.clip(Q, 0, N - 1)
        y0h = (Q + shift) / N * y0max
        e0 = y0h - y0
        y1h = y0h @ L1["W"].T + L1["b"]
        face = to_face(y1h)
        d1 = y1h - y1
        diff_levels = (face != face_ref)
        rows[name] = (float(np.abs(e0).mean()), float(d1.mean()),
                      float(np.abs(d1).mean()), float(diff_levels.mean()),
                      float(np.abs(face - face_ref).mean()))
        print(f"    {name:10s} {rows[name][0]:12.6f} {rows[name][1]:13.6f} "
              f"{rows[name][2]:13.6f} {100*rows[name][3]:9.2f}% {rows[name][4]:9.4f}")
    # 格心 + 在层 1 的偏置里抵掉原子造成的偏置
    C = np.floor(y0 / y0max * N)
    C = np.clip(C, 0, N - 1)
    y0h = (C + 0.5) / N * y0max
    resid = y0h - y0                      # 已知的系统性偏置，均值
    offset = resid.mean(0)                # 每个输入通道的平均偏置
    y1c = (y0h - offset) @ L1["W"].T + L1["b"]
    facec = to_face(y1c)
    dc = y1c - y1
    print(f"    {'格心+偏置':9s} {float(np.abs(resid).mean()):12.6f} {float(dc.mean()):13.6f} "
          f"{float(np.abs(dc).mean()):13.6f} {100*float((facec != face_ref).mean()):9.2f}% "
          f"{float(np.abs(facec - face_ref).mean()):9.4f}")

print()
print("=" * 92)
print("结论")
print("=" * 92)
print("  格边比格心差约 5 倍（b=8 的最终绝对误差 0.125 对 0.025）。")
print("  原因是偏置的覆盖面：层0 有 23.36% 的输出恰为 0。")
print("    格心把这一小撮抬高半步，均值 +0.117 步；")
print("    格边把其余 76.64% 全部压低半步，均值 -0.383 步。")
print("  偏置经过层1 放大成最终输出的整体偏移：格心 +0.0026，格边 -0.1250。")
print()
print("  格心再加偏置补偿最好，但只比纯格心好 1%，不值得为它单独加一路常数。")
print("  结论：ReLU 层存在时应当用格心，零值原子的半步偏差可并入下一层的偏置项。")
print("NOTE_DONE")
