# -*- coding: utf-8 -*-
"""画层1 一片平面的俯视平面图（供火把塔选址用）。

A 通道 0 的实际版图：方块类型 + 比较器朝向 + 容器信号
B 64 片一致性：该格在 64 个通道里是不是同一个结构
    #  64 片全同结构、非空   → 改这里对所有通道等价
    ~  64 片都非空但结构变   → 环/桶随情形切换
    o  只有部分通道存在      → 别的通道那里是空气（塔踩上去会新增方块）
    .  64 片全空
C 支撑层（y = 2ch，白色混凝土）块数 / 64：该格在多少个通道里是比较器

坐标：x 绝对列 / rx 相对列（rx = x − 3）；z 绝对行 / Z 相对行（Z = z − 21）
      平面方块在 y = 2ch+1，支撑方块在 y = 2ch

用法：python src/plane_plan.py [输出路径]
"""
import json
import os
import sys
from collections import defaultdict

from PIL import Image, ImageDraw, ImageFont

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PAL = [(200, 45, 45), (235, 140, 35), (240, 225, 55), (105, 215, 55),
       (35, 170, 190), (50, 70, 200), (135, 50, 195), (215, 80, 200)]
C_CMP = (58, 58, 72)
C_SUB = (90, 58, 82)
C_BAR = (150, 110, 70)
C_WIRE = (176, 74, 154)
C_WHITE = (200, 200, 200)
C_BASE = (110, 110, 120)
C_AIR = (24, 24, 30)
C_GRID = (60, 60, 72)
C_SAFE = (47, 107, 70)
C_VARY = (138, 106, 32)
C_PART = (122, 48, 48)

CELL = 27
MARG_L, MARG_T = 104, 74
PAD_MID = 44


def load_fonts():
    fp = r"C:\Windows\Fonts"

    def f(name, size):
        p = os.path.join(fp, name)
        return ImageFont.truetype(p, size) if os.path.exists(p) else ImageFont.load_default()
    return {"cn": f("msyh.ttc", 14), "cn_s": f("msyh.ttc", 12),
            "num": f("consola.ttf", 10), "num_s": f("consola.ttf", 8),
            "num_l": f("consola.ttf", 12)}


def cell_fill(c):
    if c is None:
        return C_AIR
    t = c["t"]
    if t == "cmp":
        return C_SUB if c["mode"] == "subtract" else C_CMP
    if t == "barrel":
        return C_BAR
    cc = c["c"] or ""
    if cc.startswith("col"):
        return PAL[int(cc[3:]) % 8]
    if cc == "white":
        return C_WHITE
    if cc == "wire":
        return C_WIRE
    return C_BASE


def main():
    d = json.load(open(os.path.join(ROOT, "results", "planes64.json"), encoding="utf-8"))
    W, L = d["W"], d["L"]
    planes = d["planes"]
    originX, originZ = 3, 21

    ch0 = {(c["x"], c["z"]): c for c in planes[0]["cells"]}

    sigs = defaultdict(dict)
    sup = defaultdict(int)
    for p in planes:
        for c in p["cells"]:
            sigs[(c["x"], c["z"])].setdefault(
                (c["t"], c["mode"] or "", c["out"] or "", c["c"] or ""), set()).add(p["ch"])
            if c["t"] == "cmp":
                sup[(c["x"], c["z"])] += 1

    def kind(x, z):
        m = sigs.get((x, z))
        if not m:
            return "."
        n = sum(len(v) for v in m.values())
        if n == len(planes):
            return "#" if len(m) == 1 else "~"
        return "o"

    F = load_fonts()
    paW, paH = W * CELL, L * CELL
    imgW = MARG_L + 3 * paW + 2 * PAD_MID + 26
    imgH = MARG_T + paH + 128
    img = Image.new("RGB", (imgW, imgH), (14, 14, 18))
    dr = ImageDraw.Draw(img)

    def grid(ox):
        for z in range(L):
            for x in range(W):
                X0, Y0 = ox + x * CELL, MARG_T + z * CELL
                dr.rectangle([X0, Y0, X0 + CELL - 1, Y0 + CELL - 1], fill=C_AIR, outline=C_GRID)

    def panelA(ox):
        grid(ox)
        for z in range(L):
            for x in range(W):
                X0, Y0 = ox + x * CELL, MARG_T + z * CELL
                c = ch0.get((x, z))
                if c is None:
                    continue
                dr.rectangle([X0, Y0, X0 + CELL - 1, Y0 + CELL - 1],
                             fill=cell_fill(c), outline=C_GRID)
                if c["t"] == "cmp":
                    dr.text((X0 + CELL / 2, Y0 + CELL / 2 - 1), c["out"], font=F["num"],
                            fill=(255, 214, 110), anchor="mm")
                    if c["mode"] == "subtract":
                        dr.rectangle([X0 + 2, Y0 + 2, X0 + 7, Y0 + 7], fill=(255, 90, 90))
                elif c["t"] == "barrel":
                    dr.text((X0 + CELL / 2, Y0 + CELL / 2), str(c["v"] | 0), font=F["num_s"],
                            fill=(255, 245, 215), anchor="mm")
                elif (c["c"] or "").startswith("col"):
                    dr.text((X0 + CELL / 2, Y0 + CELL / 2), str(c["c"][3:]), font=F["num_s"],
                            fill=(20, 20, 24), anchor="mm")

    def panelB(ox):
        grid(ox)
        for z in range(L):
            for x in range(W):
                k = kind(x, z)
                col = {"#": C_SAFE, "~": C_VARY, "o": C_PART}.get(k)
                if not col:
                    continue
                X0, Y0 = ox + x * CELL, MARG_T + z * CELL
                dr.rectangle([X0, Y0, X0 + CELL - 1, Y0 + CELL - 1], fill=col, outline=C_GRID)
                dr.text((X0 + CELL / 2, Y0 + CELL / 2), k, font=F["num_l"],
                        fill=(240, 240, 240), anchor="mm")

    def panelC(ox):
        grid(ox)
        for z in range(L):
            for x in range(W):
                n = sup.get((x, z), 0)
                if not n:
                    continue
                X0, Y0 = ox + x * CELL, MARG_T + z * CELL
                col = (47, 107, 70) if n == len(planes) else (138, 106, 32)
                dr.rectangle([X0, Y0, X0 + CELL - 1, Y0 + CELL - 1], fill=col, outline=C_GRID)
                dr.text((X0 + CELL / 2, Y0 + CELL / 2), str(n), font=F["num_s"],
                        fill=(240, 240, 240), anchor="mm")

    # 坐标标签
    titles = ["A. 通道 0 实际版图", "B. 64 片一致性（塔的安全区）", "C. 支撑层块数 / 64"]
    for i, t in enumerate(titles):
        ox = MARG_L + i * (paW + PAD_MID)
        dr.text((ox, 16), t, font=F["cn"], fill=(150, 215, 255))
        for x in range(W):
            X0 = ox + x * CELL + CELL / 2
            dr.text((X0, 40), str(x), font=F["num_s"], fill=(170, 170, 180), anchor="mm")
            dr.text((X0, 56), f"{x - originX:+d}", font=F["num_s"], fill=(90, 200, 230), anchor="mm")
    dr.text((MARG_L - 92, 40), "x", font=F["num_s"], fill=(170, 170, 180))
    dr.text((MARG_L - 92, 56), "rx", font=F["num_s"], fill=(90, 200, 230))
    for z in range(L):
        Y0 = MARG_T + z * CELL + CELL / 2
        dr.text((38, Y0), str(z), font=F["num_s"], fill=(170, 170, 180), anchor="mm")
        dr.text((74, Y0), f"{z - originZ:+d}", font=F["num_s"], fill=(90, 200, 230), anchor="mm")
    dr.text((38, MARG_T - 12), "z", font=F["num_s"], fill=(170, 170, 180), anchor="mm")
    dr.text((74, MARG_T - 12), "Z", font=F["num_s"], fill=(90, 200, 230), anchor="mm")

    panelA(MARG_L)
    panelB(MARG_L + paW + PAD_MID)
    panelC(MARG_L + 2 * (paW + PAD_MID))

    # 区域括号（画在 A 栏下方）
    by = MARG_T + paH + 6
    regions = [(0, 2, "灰环 / 灰总线"), (3, 10, "位阵列 + 减法链"),
               (11, 11, "白总线"), (12, 13, "白环")]
    for x0, x1, name in regions:
        X0 = MARG_L + x0 * CELL
        X1 = MARG_L + (x1 + 1) * CELL - 1
        dr.line([X0, by, X1, by], fill=(120, 120, 140))
        dr.line([X0, by, X0, by + 5], fill=(120, 120, 140))
        dr.line([X1, by, X1, by + 5], fill=(120, 120, 140))
        dr.text(((X0 + X1) / 2, by + 8), name, font=F["cn_s"], fill=(150, 150, 165), anchor="ma")

    # 图例
    ly = by + 34
    lx = MARG_L
    for name, col in (("比较器", C_CMP), ("减法器", C_SUB), ("容器", C_BAR),
                      ("导线", C_WIRE), ("白块", C_WHITE), ("基底", C_BASE)):
        dr.rectangle([lx, ly + 2, lx + 12, ly + 14], fill=col, outline=C_GRID)
        dr.text((lx + 16, ly + 8), name, font=F["cn_s"], fill=(200, 200, 210), anchor="lm")
        lx += 16 + dr.textlength(name, font=F["cn_s"]) + 14
    lx += 10
    for i in range(8):
        dr.rectangle([lx, ly + 2, lx + 12, ly + 14], fill=PAL[i], outline=C_GRID)
        dr.text((lx + 14, ly + 8), f"位{i}", font=F["cn_s"], fill=(200, 200, 210), anchor="lm")
        lx += 14 + dr.textlength(f"位{i}", font=F["cn_s"]) + 9

    ly += 24
    lx = MARG_L
    for k, col, txt in (("#", C_SAFE, "64 片同结构（改这里对所有通道等价）"),
                        ("~", C_VARY, "64 片都非空、结构随情形变"),
                        ("o", C_PART, "只在部分通道存在"),
                        (".", (40, 40, 48), "64 片全空")):
        dr.rectangle([lx, ly + 1, lx + 14, ly + 15], fill=col, outline=C_GRID)
        dr.text((lx + 19, ly + 8), txt, font=F["cn_s"], fill=(200, 200, 210), anchor="lm")
        lx += 19 + dr.textlength(txt, font=F["cn_s"]) + 18

    ly += 24
    dr.text((MARG_L, ly),
            "红角标 = 减法比较器；容器上的数字 = 信号强度；彩色块上的数字 = 位号。"
            "平面方块在 y=2ch+1（64 层），支撑方块在 y=2ch。", 
            font=F["cn_s"], fill=(150, 150, 162))

    out = (sys.argv[1] if len(sys.argv) > 1
           else os.path.join(ROOT, "results", "plane_plan.png"))
    img.save(out)
    print(f"写出 {out}  {imgW}x{imgH}")

    safe = [(x, z) for z in range(L) for x in range(W) if kind(x, z) == "#"]
    vary = [(x, z) for z in range(L) for x in range(W) if kind(x, z) == "~"]
    part = [(x, z) for z in range(L) for x in range(W) if kind(x, z) == "o"]
    air = [(x, z) for z in range(L) for x in range(W) if kind(x, z) == "."]
    print(f"  64 片同结构 {len(safe)}   结构随情形 {len(vary)}   部分通道存在 {len(part)}   "
          f"全空 {len(air)}   （共 {W*L}）")
    print(f"  结构随情形：{vary}")
    print(f"  部分通道存在：{part}")


if __name__ == "__main__":
    main()
