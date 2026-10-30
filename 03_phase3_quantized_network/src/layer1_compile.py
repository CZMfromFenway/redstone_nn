#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
编译 4→64 层1（b=8, W_ref=882），把结果落到平面图上。

环放在品红（MSB）侧：
    · 配对环（一组 2 个 MSB 权重共用一个）
    · 灰环 K、白环 P 也搬到这里
    ⇒ 省掉左上角 3 列的独立环区，W 从 2n+6 降到 2n+4
"""
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
P2 = os.path.join(os.path.dirname(ROOT), "02_phase2_avatar")

mj = json.load(open(os.path.join(P2, "webapp", "pca22_dim4w64L1_model.json"),
                    encoding="utf-8"))
W1 = np.array(mj["gen"]["layers"][0]["W"], dtype=np.float64)
B1 = np.array(mj["gen"]["layers"][0]["b"], dtype=np.float64)
V2W = np.array(mj["gen"]["layers"][1]["W"], dtype=np.float64)
PHI = np.array(mj["gen"]["layers"][1]["b"], dtype=np.float64)

DELTA, XLO, WREF, B = 7.0, -3.5, 882.0, 8
NX = 1 << B
M, N = W1.shape

alpha = WREF / (np.abs(W1).max(1) * DELTA / 4)
U = np.rint(alpha[:, None, None] * W1[:, :, None] * (DELTA / NX)
            * (1 << np.arange(B))[None, None, :]).astype(np.int64)   # (64,4,8)

print("=" * 96)
print("一、64 个通道的 MSB 配对分类")
print("=" * 96)
print("  情形 Ⅰ：两路都 >882  → 1 环 + 1 木桶(存差值)")
print("  情形 Ⅱ：一路 >882     → 1 环 + 1 木桶(存另一路)")
print("  情形 Ⅲ：两路都 ≤882   → 0 环 + 2 木桶")
print()

cnt = {"I": 0, "II": 0, "III": 0}
bad = 0
maxdiff = 0
cls = []
for j in range(M):
    w = np.abs(U[j, :, B - 1])
    for p in range(N // 2):
        a, b2 = w[2 * p], w[2 * p + 1]
        hi = max(a, b2)
        lo = min(a, b2)
        if lo > 882:
            k = "I"
            d = hi - lo
            maxdiff = max(maxdiff, d)
            if d > 881:
                bad += 1
        elif hi > 882:
            k = "II"
        else:
            k = "III"
        cnt[k] += 1
        cls.append((j, p, k, int(a), int(b2)))

tot = sum(cnt.values())
print(f"  {'情形':>4s} {'对数':>6s} {'占比':>7s}")
print("  " + "-" * 22)
for k in ("I", "II", "III"):
    print(f"  {k:>4s} {cnt[k]:6d} {100*cnt[k]/tot:6.1f}%")
print(f"\n  合计 {tot} 对 = 64 通道 × 2 对")
print(f"  情形 Ⅰ 的最大 |差值| = {maxdiff}  （约束 ≤881）"
      f"  {'✓ 全部通过' if bad == 0 else f'✗ {bad} 对越界'}")
print(f"  需要环的对数 = Ⅰ+Ⅱ = {cnt['I']+cnt['II']}")

print()
print("  每个通道的明细（前 12 个）：")
print(f"  {'通道':>4s} {'对0':>16s} {'对1':>16s} {'环数':>4s}")
print("  " + "-" * 48)
for j in range(12):
    e = [c for c in cls if c[0] == j]
    s = "".join(f"{c[2]}({c[3]},{c[4]}) " for c in e)
    nr = sum(1 for c in e if c[2] in "I II")
    print(f"  {j:4d} {e[0][2]+'('+str(e[0][3])+','+str(e[0][4])+')':>16s} "
          f"{e[1][2]+'('+str(e[1][3])+','+str(e[1][4])+')':>16s} {nr:4d}")

print()
print("=" * 96)
print("二、整个模型的环数")
print("=" * 96)


def layer_rings(Wm, b):
    M2, N2 = Wm.shape
    al = WREF / (np.abs(Wm).max(1) * DELTA / 4)
    UU = np.rint(al[:, None, None] * Wm[:, :, None] * (DELTA / (1 << b))
                 * (1 << np.arange(b))[None, None, :]).astype(np.int64)
    c = {"I": 0, "II": 0, "III": 0}
    md = 0
    for j in range(M2):
        w = np.abs(UU[j, :, b - 1])
        for p in range(N2 // 2):
            a, bb = w[2 * p], w[2 * p + 1]
            hi, lo = max(a, bb), min(a, bb)
            if lo > 882:
                c["I"] += 1
                md = max(md, hi - lo)
            elif hi > 882:
                c["II"] += 1
            else:
                c["III"] += 1
    return c, md, UU


c1, md1, U1 = layer_rings(W1, 8)
# 层2：64→32，b_y=10
V2 = np.array(mj["gen"]["layers"][1]["W"], dtype=np.float64)
c2, md2, U2 = layer_rings(V2.T, 10)   # (32,64)，输入是 64

n1 = c1["I"] + c1["II"]
n2 = c2["I"] + c2["II"]
print(f"  层1 (4→64,  b=8) :  Ⅰ={c1['I']:3d}  Ⅱ={c1['II']:3d}  Ⅲ={c1['III']:3d} "
      f"→ 配对环 {n1} 个,  最大差 {md1}")
print(f"  层2 (64→32, b=10):  Ⅰ={c2['I']:4d}  Ⅱ={c2['II']:4d}  Ⅲ={c2['III']:4d} "
      f"→ 配对环 {n2} 个,  最大差 {md2}")
print(f"  全局环：每平面 2 个（K 和 P）")
print(f"    层1: {M}×2 = {M*2}   层2: 32×2 = 64")
print(f"  总环数 = {n1}+{M*2} + {n2}+64 = {n1+M*2+n2+64}")

print()
print("=" * 96)
print("三、体积（环全放品红侧）")
print("=" * 96)
L1 = 3 * 8 + 8
L2 = 3 * 10 + 8
W1d = 2 * 4 + 4
W2d = 2 * 64 + 4
v1 = 2 * W1d * L1 * 64
v2 = 2 * W2d * L2 * 32
print(f"  层1 平面 {W1d}×{L1} = {W1d*L1}  ×64 平面 ×2 高 = {v1:,}")
print(f"  层2 平面 {W2d}×{L2} = {W2d*L2}  ×32 平面 ×2 高 = {v2:,}")
print(f"  合计 = {v1+v2:,}")
print()
print(f"  对照（环不合并，W=2n+6, L=3b+5）：")
v1b = 2 * (2*4+6) * (3*8+5) * 64
v2b = 2 * (2*64+6) * (3*10+5) * 32
print(f"    层1 {v1b:,}   层2 {v2b:,}   合计 {v1b+v2b:,}")
print(f"  节省 = {(v1b+v2b)-(v1+v2):,}  ({(1-(v1+v2)/(v1b+v2b))*100:.1f}%)")

json.dump({"pairs": {k: cnt[k] for k in cnt}, "max_diff": int(maxdiff),
           "rings_layer1": n1, "rings_layer2": n2,
           "vol_magenta_side": v1 + v2, "vol_split": v1b + v2b},
          open(os.path.join(RESULTS, "layer1_compile.json"), "w", encoding="utf-8"),
          indent=1)

# ---------------- 出图：一个代表性通道的平面
j0 = 0
print()
print("=" * 96)
print(f"四、出图：通道 {j0} 的平面")
print("=" * 96)
S = 15 + np.abs(U[j0])
print(f"  8 个位权重 × 4 输入，MSB 行 = {np.abs(U[j0,:,7]).tolist()}")
print(f"  配对 0: ({abs(U[j0,0,7])}, {abs(U[j0,1,7])})  "
      f"配对 1: ({abs(U[j0,2,7])}, {abs(U[j0,3,7])})")

n = N
Wd, Ld = 2 * n + 4, 3 * B + 8
CELL, PADL, PADT, LEG = 30, 60, 70, 330
img = Image.new("RGB", (PADL + Wd * CELL + LEG, PADT + Ld * CELL + 16), (22, 22, 28))
dr = ImageDraw.Draw(img)
PAL = [(200, 45, 45), (235, 140, 35), (240, 225, 55), (105, 215, 55),
       (35, 170, 190), (50, 70, 200), (135, 50, 195), (215, 80, 200)]
BASE, WHITE, CMPF, SUBF = (120, 120, 130), (240, 240, 240), (56, 56, 66), (56, 50, 62)
BAR = (150, 110, 70)


def cell(x, z, kind, col=None, lab="", txtcol=(10, 10, 10)):
    x0, y0 = PADL + x * CELL, PADT + z * CELL
    if kind == "bar":
        dr.rectangle([x0+1, y0+1, x0+CELL-2, y0+CELL-2], fill=col or BAR,
                     outline=(20, 20, 20))
        if lab:
            dr.text((x0 + CELL//2, y0 + CELL//2), lab, txtcol, anchor="mm")
    elif kind == "col":
        dr.rectangle([x0+1, y0+1, x0+CELL-2, y0+CELL-2], fill=col,
                     outline=(90, 90, 90))
    elif kind in ("cmp", "sub"):
        dr.rectangle([x0+1, y0+1, x0+CELL-2, y0+CELL-2],
                     fill=SUBF if kind == "sub" else CMPF,
                     outline=(160, 90, 90) if kind == "sub" else (110, 110, 125))
        c = (255, 110, 110) if kind == "sub" else (255, 205, 95)
        dr.text((x0 + CELL//2, y0 + CELL//2), lab, c, anchor="mm")
    elif kind == "white":
        dr.rectangle([x0+1, y0+1, x0+CELL-2, y0+CELL-2], fill=WHITE)
    else:
        dr.rectangle([x0+1, y0+1, x0+CELL-2, y0+CELL-2], fill=BASE)


# 主体：位 0..6 的彩色行 + 木桶
for k in range(B - 1):
    col = [2 * i + 3 for i in range(n)] if k % 2 == 0 else [2 * i + 4 for i in range(n)]
    zc = 3 * k + 2 if k % 2 == 0 else 3 * k + 3
    zb = 3 * k + 1 if k % 2 == 0 else 3 * k + 4
    for i in range(n):
        cell(col[i], zc, "col", PAL[k])
        cell(col[i], zb, "bar", PAL[k], str(S[i, k]))
# 减法链
for jj in range((B - 1 + 1) // 2):
    z4 = 6 * jj + 4
    if z4 < 3 * (B - 1) + 2:
        for x in range(3, 2 * n + 3):
            cell(x, z4, "sub", lab="S")
# MSB 行
zc7 = 3 * (B - 1) + 3
for i in range(n):
    cell(2 * i + 4, zc7, "col", PAL[B - 1])
    cell(2 * i + 3, zc7 - 1, "cmp", lab="N")
# 环区（MSB 之下）
zr = zc7 + 1
for p in range(n // 2):
    xa, xb = 2 * (2 * p) + 4, 2 * (2 * p + 1) + 4
    for zz in range(4):
        cell(xa, zr + zz, "col", PAL[B - 1])
        cell(xb, zr + zz, "col", PAL[B - 1])
    cell(xa + 1, zr + 1, "cmp", lab="E")
    cell(xa + 1, zr + 2, "sub", lab="S")
    cell(xa + 1, zr + 3, "cmp", lab="W")
    cell(xa + 1, zr + 0, "cmp", lab="N")
    cell(xb, zr + 2, "bar", BAR, "Δ" + str(p))
# 两个全局环放右侧
for gi, (lb, val) in enumerate((("K", "K"), ("P", "P"))):
    gx = 2 * n + 3
    for zz in range(3):
        cell(gx, zr + gi * 3 + zz, "cmp", lab=lb if zz == 1 else "")
    cell(gx, zr + gi * 3 + 1, "white")

for x in range(Wd):
    dr.text((PADL + x * CELL + CELL // 2, PADT - 24), str(x), (195, 195, 205), anchor="ma")
for z in range(Ld):
    dr.text((8, PADT + z * CELL + CELL // 2), f"z{z}", (195, 195, 205), anchor="lm")

info = [f"层1 通道 {j0}   b=8 n=4", f"平面 {Wd}×{Ld} = {Wd*Ld}",
        "", "数字 = S=15+|W|", "S = 减法比较器", "Δp = 差值木桶",
        "K/P = 两个全局环", "", "环全放品红侧"]
for t, s in enumerate(info):
    dr.text((PADL + Wd * CELL + 14, PADT + t * 20), s, (228, 228, 238))
out = os.path.join(RESULTS, "layer1_plane.png")
img.save(out)
print(f"  wrote {out}  ({os.path.getsize(out)//1024} KB)")
print("L1_DONE")
