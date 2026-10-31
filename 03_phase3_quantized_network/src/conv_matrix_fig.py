# -*- coding: utf-8 -*-
"""8×8 图、3×3 核：把卷积写成 64×64 矩阵，按最优（行优先）顺序填进去，出图。

约定
    像素 (r,c)，r=行、c=列，0..7；行优先序号 p = 8r + c。
    矩阵 M[p][q] = a_{i,j}，其中 i = 1 + r_p - r_q，j = 1 + c_p - c_q（0 <= i,j <= 2 才算非零）。
    等价：M[p][q] 落在偏移 Delta = q - p = 9 - 8i - j 那条对角线上。

出图：results/conv_matrix_8x8.png
用法：python src/conv_matrix_fig.py
"""
import os
import sys

from PIL import Image, ImageDraw, ImageFont

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
N = 8
NN = N * N

C_BG = (16, 16, 20)
C_ZERO = (30, 30, 38)
C_GRID = (58, 58, 70)
C_GRID2 = (96, 96, 116)
C_TXT = (228, 228, 236)
C_DIM = (150, 150, 165)

# a_ij → 颜色（9 个）
KCOL = {
    (0, 0): (0, 196, 255), (0, 1): (52, 108, 235), (0, 2): (146, 74, 230),
    (1, 0): (0, 210, 130), (1, 1): (250, 226, 60), (1, 2): (250, 150, 46),
    (2, 0): (250, 84, 128), (2, 1): (196, 52, 60), (2, 2): (130, 88, 58),
}
DCOL = {(0, 0): 9, (0, 1): 8, (0, 2): 7,
        (1, 0): 1, (1, 1): 0, (1, 2): -1,
        (2, 0): -7, (2, 1): -8, (2, 2): -9}


def fonts():
    fp = r"C:\Windows\Fonts"

    def f(name, size):
        p = os.path.join(fp, name)
        return ImageFont.truetype(p, size) if os.path.exists(p) else ImageFont.load_default()
    return {"t": f("msyh.ttc", 22), "h": f("msyh.ttc", 16), "cn": f("msyh.ttc", 13),
            "cn_s": f("msyh.ttc", 11), "mono": f("consola.ttf", 11),
            "mono_s": f("consola.ttf", 9), "big": f("consola.ttf", 14)}


def kernel_at(p, q):
    rp, cp = divmod(p, N)
    rq, cq = divmod(q, N)
    i, j = 1 + rp - rq, 1 + cp - cq
    return (i, j) if (0 <= i <= 2 and 0 <= j <= 2) else None


def main():
    F = fonts()
    nz = [(p, q) for p in range(NN) for q in range(NN) if kernel_at(p, q)]

    # ---------------- 画布 ----------------
    W_, H_ = 2140, 1350
    img = Image.new("RGB", (W_, H_), C_BG)
    dr = ImageDraw.Draw(img)

    dr.text((30, 16), "8×8 图 + 3×3 核 → 64×64 卷积矩阵：最优填法（行优先，带宽 9）",
            font=F["t"], fill=C_TXT)
    dr.text((30, 48), "只有 484 个非零（内部 9 / 边 6 / 角 4）；0 不画。九个 a_ij 各占一条对角线。",
            font=F["cn"], fill=C_DIM)

    # ================= Panel A：整张矩阵 =================
    CELL = 9
    ax, ay = 100, 112
    dr.text((ax, ay - 50), "A  整张 64×64 矩阵（行 = 输出 p，列 = 输入 q）",
            font=F["h"], fill=C_TXT)
    for p in range(NN):
        for q in range(NN):
            k = kernel_at(p, q)
            x0 = ax + q * CELL
            y0 = ay + p * CELL
            dr.rectangle([x0, y0, x0 + CELL - 1, y0 + CELL - 1],
                         fill=KCOL[k] if k else C_ZERO)
    # 8×8 分块线（每 8 个像素一条）
    for t in range(0, NN + 1, N):
        x = ax + t * CELL
        y = ay + t * CELL
        dr.line([x, ay, x, ay + NN * CELL], fill=C_GRID2 if t % NN else C_TXT, width=1)
        dr.line([ax, y, ax + NN * CELL, y], fill=C_GRID2 if t % NN else C_TXT, width=1)
        if t < NN:
            r = t // N
            dr.text((ax - 52, ay + t * CELL + 1), "p=%2d" % t, font=F["mono_s"], fill=C_DIM)
            dr.text((ax + t * CELL + 1, ay - 20), "q=%2d" % t, font=F["mono_s"], fill=C_DIM)

    dr.text((ax + NN * CELL - 20, ay - 15), "q", font=F["mono"], fill=C_DIM)

    # 图例
    lx, ly = ax + NN * CELL + 34, ay - 6
    dr.text((lx, ly - 22), "a_ij  →  对角线偏移 Δ = q−p", font=F["cn"], fill=C_TXT)
    order = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
    for k, (i, j) in enumerate(order):
        y = ly + k * 26
        dr.rectangle([lx, y, lx + 18, y + 18], fill=KCOL[(i, j)])
        dr.text((lx + 26, y + 2), "a%d%d" % (i, j), font=F["big"], fill=C_TXT)
        dr.text((lx + 66, y + 3), "Δ = %+d" % DCOL[(i, j)], font=F["mono"], fill=C_DIM)
    dr.text((lx, ly + 9 * 26 + 8), "带宽 W = 9（= n+1，可证最小）", font=F["cn"], fill=C_TXT)
    dr.text((lx, ly + 9 * 26 + 28), "带内占 19/64 列；非零仅 484 个", font=F["cn_s"], fill=C_DIM)

    # ================= Panel B：放大 =================
    RW, CW, CB = 14, 25, 26
    bx, by = 1060, 118
    dr.text((bx, by - 46), "B  左上角放大（带缺口：越过图边就不是邻居）", font=F["h"], fill=C_TXT)
    for p in range(RW):
        for q in range(CW):
            k = kernel_at(p, q)
            x0 = bx + q * CB
            y0 = by + p * CB
            dr.rectangle([x0, y0, x0 + CB - 1, y0 + CB - 1],
                         fill=KCOL[k] if k else C_ZERO, outline=C_GRID)
            if k:
                dr.text((x0 + 6, y0 + 6), "a%d%d" % k, font=F["mono_s"],
                        fill=(10, 10, 14) if k == (1, 1) else (245, 245, 250))
    for p in range(RW):
        dr.text((bx - 40, by + p * CB + 7), "p=%2d" % p, font=F["mono_s"], fill=C_DIM)
    for q in range(0, CW, 4):
        dr.text((bx + q * CB + 4, by + RW * CB + 4), "q=%d" % q, font=F["mono_s"], fill=C_DIM)
    # 标出 Δ=+9 与 Δ=-1 两端的含义
    dr.text((bx, by + RW * CB + 22),
            "Δ=+9 那条是 a00（上一行左一列）；Δ=−1 那条是 a12（同一行右一列）。",
            font=F["cn_s"], fill=C_DIM)

    # ================= Panel C：真实平面俯视 =================
    S = 4.2
    cx, cy = 70, 742
    dr.text((cx, cy - 30), "C  真实平面俯视（x = 3+2q 输入列，y = 3+2p 输出层；主方向 z 垂直纸面）",
            font=F["h"], fill=C_TXT)
    # 背景：整方阵的范围
    dr.rectangle([cx, cy, cx + int(134 * S), cy + int(130 * S)], outline=C_GRID)
    for p in range(NN):
        for q in range(NN):
            k = kernel_at(p, q)
            if not k:
                continue
            x0 = cx + int((3 + 2 * q) * S)
            y0 = cy + int((3 + 2 * p) * S)
            dr.rectangle([x0, y0, x0 + int(2 * S) - 1, y0 + int(2 * S) - 1], fill=KCOL[k])
    for t in range(0, N + 1):
        x = cx + int((3 + 2 * N * t) * S) if t < N else cx + int(132 * S)
        y = cy + int((3 + 2 * N * t) * S) if t < N else cy + int(132 * S)
        if t < N:
            dr.line([x, cy, x, cy + int(130 * S)], fill=C_GRID)
            dr.line([cx, y, cx + int(134 * S), y], fill=C_GRID)
    dr.text((cx, cy + int(130 * S) + 6), "134 格宽（x）× 130 格高（y）；z 方向再乘 3b+7 = 31",
            font=F["cn_s"], fill=C_DIM)

    # ================= 文字说明 =================
    tx, ty = 740, 706
    lines = [
        ("行 p 与列 q 是什么", "h"),
        ("行 p = 第 p 个输出像素，位置 y = 2p+3（每通道 2 层）", "cn"),
        ("列 q = 第 q 个输入像素，位置 x = 3+2q（每输入 2 列）", "cn"),
        ("p = 8r+c、q = 8r'+c'，都按行优先（从左到右、从上到下）", "cn"),
        ("", "cn"),
        ("为什么行优先是最优", "h"),
        ("非零元只在 |r−r'|<=1 且 |c−c'|<=1 处 → 标签差 ≤ 8+1 = 9", "cn"),
        ("下界：对角两端像素的标签差 / 直径 = (64−1)/7 = 9", "cn"),
        ("⇒ 带宽恰为 9，任何同序排序都压不窄；独立重排搜索也没更好", "cn"),
        ("", "cn"),
        ("九个 a_ij 的偏移", "h"),
        ("Δ = 9 − 8i − j  →  +9,+8,+7, +1,0,−1, −7,−8,−9", "mono"),
        ("缺口 = 越过图边（例如 c=7 时 a12 不是邻居）", "cn"),
        ("", "cn"),
        ("体积口径（与 docs/44 一致）", "h"),
        ("包围盒 = (2n_in+6) × (2m_out+2) × (3b+7)", "mono"),
        ("= 134 × 130 × 31 = 540,020 格（n=8、b=8）", "mono"),
        ("稀疏只减少非空方块（484 个容器），不缩小包围盒", "cn"),
    ]
    y = ty
    for text, kind in lines:
        if not text:
            y += 10
            continue
        f = F["h"] if kind == "h" else (F["mono"] if kind == "mono" else F["cn"])
        dr.text((tx, y), text, font=f, fill=C_TXT if kind == "h" else C_DIM)
        y += 24 if kind == "h" else 20

    out = os.path.join(ROOT, "results", "conv_matrix_8x8.png")
    img.save(out)
    print("已写出", out, img.size)
    print()
    print("非零个数:", len(nz), "/", NN * NN)
    from collections import Counter
    cnt = Counter(kernel_at(p, q) for p, q in nz)
    for (i, j) in order:
        print("  a%d%d  Δ=%+2d  个数 %d" % (i, j, DCOL[(i, j)], cnt[(i, j)]))
    band = NN * (2 * 9 + 1)
    print("带宽 9 的带内位置:", band, " 占整方阵 %.1f%%" % (100.0 * band / (NN * NN)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
