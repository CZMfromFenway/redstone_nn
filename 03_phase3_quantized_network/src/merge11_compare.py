#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
16 级 vs MC 11 档：差异像素数会不会下降。

11 档是 16 级之后的一张归并表，把部分相邻级并成同一个灰度：
    0,1,2 -> BLACK      3,4 -> GRAY       10,11 -> 9.92       12,13 -> 11.71
所以两个像素在 16 级下差 1 级，若恰好落在同一组，归并后就一样了。
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
from intpipe import compile_layer, run, to_float_face  # noqa: E402

P2 = os.path.join(os.path.dirname(ROOT), "02_phase2_avatar")
XMIN, XMAX, SPAN = -3.5, 3.5, 7.0
NLEV = 15

mj = json.load(open(os.path.join(P2, "webapp", "pca22_dim4w64L1_model.json"),
                    encoding="utf-8"))
W1 = np.array(mj["gen"]["layers"][0]["W"], dtype=np.float64)
T1 = np.array(mj["gen"]["layers"][0]["b"], dtype=np.float64)
W2 = np.array(mj["gen"]["layers"][1]["W"], dtype=np.float64)
T2 = np.array(mj["gen"]["layers"][1]["b"], dtype=np.float64)

mg = json.load(open(os.path.join(P2, "webapp", "merge_mc11.json"), encoding="utf-8"))
LUT = np.array(mg["_lut"], dtype=np.float64)          # 16 级 -> 亮度值


def post_C(Y):
    ymax, ymin = Y.max(1, keepdims=True), Y.min(1, keepdims=True)
    span = np.maximum(ymax - ymin, 1)
    return np.clip(np.floor_divide((Y - ymin) * NLEV + (span >> 1), span), 0, NLEV)


NF = 4000
rng = np.random.default_rng(31)
print("=" * 100)
print(f"16 级  vs  MC 11 档      {NF} 张脸，{NF*32:,} 个像素")
print("=" * 100)
print(f"  归并表把 16 级映到 {len(np.unique(LUT))} 个不同亮度，其中相邻可分的约 9 个")
groups = {}
for i, v in enumerate(LUT):
    groups.setdefault(round(float(v), 2), []).append(i)
print("  被并成同一灰度的级组：" + "、".join(
    "{" + ",".join(str(x) for x in g) + "}" for g in groups.values() if len(g) > 1))

for b1, b2 in ((8, 8), (11, 11)):
    N1 = 1 << b1
    L1 = compile_layer(W1, T1, SPAN / N1, b1, relu=True, xbase=XMIN)
    step1 = L1["c"] / L1["s"]
    y1m = np.clip(np.floor_divide(L1["Fhi"] - L1["A"], L1["c"]), 0, N1 - 1)
    L2 = compile_layer(W2, T2, step1, b2, relu=False, xbase=0.0, ymax_in=y1m,
                       ztop=15.0, shared_scale=True)
    N2 = 1 << b2

    xt = np.clip(rng.standard_normal((NF, 4)), XMIN, XMAX)
    X1 = np.clip(np.floor((xt - XMIN) / SPAN * N1), 0, N1 - 1).astype(np.int64)
    xr = XMIN + (X1 + 0.5) * SPAN / N1
    z2f = np.maximum(0.0, xr @ W1.T + T1) @ W2.T + T2

    ref16 = to_float_face(z2f)
    _, _, Y1 = run(X1, L1)
    _, _, Y2 = run(Y1, L2)
    int16 = post_C(Y2)

    d16 = int16 - ref16
    lf11, li11 = LUT[ref16], LUT[int16]
    d11 = li11 - lf11

    print()
    print("=" * 100)
    print(f"b = {b1}/{b2}")
    print("=" * 100)
    print(f"  {'表示':16s} {'不同像素':>10s} {'平均|差|':>10s} {'≥2 级的脸':>10s} {'完全相同的脸':>12s}")
    print("  " + "-" * 64)
    for name, r, d in (("16 级（0..15）", ref16, d16), ("MC 11 档（亮度）", lf11, d11)):
        face_max = np.abs(d).max(1)
        print(f"  {name:16s} {100*(d!=0).mean():9.2f}% {np.abs(d).mean():10.4f} "
              f"{100*(face_max>=2).mean():9.2f}% {100*(face_max==0).mean():11.2f}%")

    print()
    print("  16 级下的差异像素，归并后有 %d 个变成相同（占原差异的 %.1f%%）"
          % (int(((d16 != 0) & (d11 == 0)).sum()),
             100 * ((d16 != 0) & (d11 == 0)).sum() / max((d16 != 0).sum(), 1)))
    # 按 16 级差异的级值统计
    print()
    print(f"  {'16级差的级值':>12s} {'出现像素':>10s} {'归并后仍不同':>13s} {'归并后相同':>11s}")
    print("  " + "-" * 50)
    for k in range(0, 4):
        m = np.abs(d16) == k
        if m.sum() == 0:
            continue
        still = (m & (d11 != 0)).sum()
        print(f"  {k:12d} {int(m.sum()):10d} {int(still):13d} {int(m.sum()-still):11d}")
    # 哪些级差会塌
    print()
    collapse = {}
    for a_ in range(16):
        for b_ in range(16):
            if a_ != b_ and LUT[a_] == LUT[b_]:
                collapse[(a_, b_)] = True
    print(f"  会塌掉的相邻级对：{[(a_,b_) for (a_,b_) in collapse if abs(a_-b_)==1]}")

print()
print("MERGE_DONE")
