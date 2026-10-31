# -*- coding: utf-8 -*-
"""K 个卷积核的斜带堆叠图（给 kb 用）。

左：单核的 64×64 矩阵，标出一条行的 19 列窗。
右：K=3 个核按"下移 跨度 行"密堆，总高度 = n^2 + (K-1)*跨度 行。

用法：python src/conv_band_stack_fig.py
出图：kb/figures/conv_band_stacking.png
"""
import os
import sys

from PIL import Image, ImageDraw, ImageFont

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))   # 仓库根
N = 8
NN = N * N
SPAN = 2 * N + 3          # 19
HALF = (SPAN - 1) // 2    # 9

C_BG = (16, 16, 20)
C_ZERO = (30, 30, 38)
C_GRID = (60, 60, 72)
C_TXT = (228, 228, 236)
C_DIM = (152, 152, 168)
KCOL = [(0, 196, 255), (250, 226, 60), (250, 84, 128)]


def fonts():
    fp = r"C:\Windows\Fonts"

    def f(name, size):
        p = os.path.join(fp, name)
        return ImageFont.truetype(p, size) if os.path.exists(p) else ImageFont.load_default()
    return {"t": f("msyh.ttc", 20), "h": f("msyh.ttc", 15), "cn": f("msyh.ttc", 13),
            "cn_s": f("msyh.ttc", 11), "mono": f("consola.ttf", 11),
            "mono_s": f("consola.ttf", 9)}


def cell(img, dr, x0, y0, w, h, col):
    dr.rectangle([x0, y0, x0 + w - 1, y0 + h - 1], fill=col)


def main():
    F = fonts()
    W_, H_ = 1420, 760
    img = Image.new("RGB", (W_, H_), C_BG)
    dr = ImageDraw.Draw(img)
    dr.text((28, 14), "把 K 个卷积核的斜带上下密堆：宽不变，高按“跨度”线性涨", font=F["t"], fill=C_TXT)
    dr.text((28, 44), f"n={N}，3×3 核，跨度 S = 2n+3 = {SPAN}（每行触及 {SPAN} 个输入列）",
            font=F["cn"], fill=C_DIM)

    # ---------------- 左：单核 ----------------
    CELL, bx, by = 8, 172, 104
    dr.text((172, 76), "A  一个核：64×64 斜带", font=F["h"], fill=C_TXT)
    for p in range(NN):
        for q in range(NN):
            x0 = bx + q * CELL
            y0 = by + p * CELL
            on = abs(q - p) <= HALF
            cell(img, dr, x0, y0, CELL, CELL, (70, 90, 120) if on else C_ZERO)
    for t in range(0, NN + 1, N):
        dr.line([bx + t * CELL, by, bx + t * CELL, by + NN * CELL], fill=C_GRID)
        dr.line([bx, by + t * CELL, bx + NN * CELL, by + t * CELL], fill=C_GRID)
    # 标两条行的窗（标签写在左侧空白处）
    for prow, col in ((20, KCOL[0]), (52, KCOL[1])):
        qlo, qhi = max(0, prow - HALF), min(NN - 1, prow + HALF)
        y0 = by + prow * CELL
        dr.rectangle([bx + qlo * CELL, y0, bx + (qhi + 1) * CELL - 1, y0 + CELL - 1],
                     outline=col, width=2)
        dr.line([bx - 8, y0 + CELL // 2, bx + qlo * CELL, y0 + CELL // 2], fill=col)
        dr.text((44, y0 - 4), f"行 {prow}：{qhi-qlo+1} 列", font=F["cn_s"], fill=col)

    dr.text((172, by + NN * CELL + 8), "宽 n² = 64 列，高 n² = 64 行；每行只有 19 列非零",
            font=F["cn_s"], fill=C_DIM)

    # ---------------- 右：K=3 堆叠 ----------------
    K = 3
    CW, CWx, CWy = 5, 800, 104
    rows = NN + (K - 1) * SPAN
    dr.text((CWx - 60, 76), f"B  K={K} 个核各自下移 {SPAN} 行 → 高 {NN}+{SPAN}×{K-1}={rows} 行",
            font=F["h"], fill=C_TXT)
    for k in range(K):
        off = k * SPAN
        for p in range(NN):
            for q in range(NN):
                if abs(q - p) > HALF:
                    continue
                x0 = CWx + q * CW
                y0 = CWy + (p + off) * CW
                cell(img, dr, x0, y0, CW, CW, KCOL[k])
    # 每 16 行一条浅网格
    for r in range(0, rows + 1, 16):
        dr.line([CWx, CWy + r * CW, CWx + NN * CW, CWy + r * CW], fill=(44, 44, 54))
    for q in range(0, NN + 1, 16):
        dr.line([CWx + q * CW, CWy, CWx + q * CW, CWy + rows * CW], fill=(44, 44, 54))
    for k in range(K):
        y = CWy + (k * SPAN + NN // 2) * CW
        dr.text((CWx + NN * CW + 8, y - 6), f"核{k+1}", font=F["cn"], fill=KCOL[k])
    # 高度标注（右对齐到轴线左侧的窄缝里）
    def rtext(x_right, y, s, font, fill):
        dr.text((x_right - dr.textlength(s, font=font), y), s, font=font, fill=fill)

    dr.line([CWx - 20, CWy, CWx - 20, CWy + NN * CW], fill=C_TXT, width=2)
    rtext(CWx - 26, CWy + NN * CW // 2 - 8, f"2n²={2*NN}", F["mono_s"], C_TXT)
    dr.line([CWx - 20, CWy + NN * CW, CWx - 20, CWy + rows * CW], fill=(250, 226, 60), width=2)
    rtext(CWx - 26, CWy + (NN + (K - 1) * SPAN // 2) * CW,
          f"2(K−1)S={2*(K-1)*SPAN}", F["mono_s"], (250, 226, 60))
    # 下移量箭头（第 1→2 核的起点之间）
    for k in range(K - 1):
        y1 = CWy + (k * SPAN) * CW
        y2 = CWy + ((k + 1) * SPAN) * CW
        dr.line([CWx + NN * CW + 30, y1, CWx + NN * CW + 30, y2], fill=C_DIM)
        dr.text((CWx + NN * CW + 36, (y1 + y2) // 2 - 6), f"S={SPAN} 行", font=F["mono_s"], fill=C_DIM)
    dr.text((CWx - 60, CWy + rows * CW + 8),
            f"高 = 2n² + 2(K−1)S = {2*NN} + {2*(K-1)*SPAN} = {2*NN + 2*(K-1)*SPAN} 格；宽仍为 2n²+6",
            font=F["cn"], fill=C_DIM)

    # ---------------- 结论一行 ----------------
    dr.text((172, 690),
            "为什么 S 是最小间隔：下移量 < S 时，两个核的行区间 [s, s+n²] 必定重叠（因为 S < n²），",
            font=F["cn"], fill=C_DIM)
    dr.text((172, 712),
            "于是在同一行上两个列窗相交 → 撞车。所以任意两个核至少隔 S 行，等距堆叠即达到下界。",
            font=F["cn"], fill=C_DIM)

    out = os.path.join(ROOT, "kb", "figures", "conv_band_stacking.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    img.save(out)
    print("已写出", out, img.size)
    print(f"K=1 高 {2*NN}，K={K} 高 {2*NN + 2*(K-1)*SPAN}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
