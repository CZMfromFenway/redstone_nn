# -*- coding: utf-8 -*-
"""画一张带完整标注的样图：原点、j、相对/绝对坐标、各结构区域。"""
import json
import os

from PIL import Image, ImageDraw, ImageFont

ROOT = r"E:\redstone_nn\03_phase3_quantized_network"
D = json.load(open(os.path.join(ROOT, "results", "board_dump.json"), encoding="utf-8"))
C = D["cases"][0]

C_C = 34
PADL, PADT, PADR, PADB = 210, 104, 150, 70
W, L = C["W"], C["L"]
oX, oZ = C["originX"], C["originZ"]

FONT = None
for fp in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf",
           r"C:\Windows\Fonts\simsun.ttc"):
    try:
        FONT = ImageFont.truetype(fp, 12); break
    except Exception:
        pass
if FONT is None:
    FONT = ImageFont.load_default()

img = Image.new("RGB", (PADL + W * C_C + PADR, PADT + L * C_C + PADB), (14, 14, 18))
dr = ImageDraw.Draw(img)
T = lambda xy, txt, fill, anchor='la': dr.text(xy, txt, fill=fill, anchor=anchor, font=FONT)

PAL = [(200, 45, 45), (235, 140, 35), (240, 225, 55), (105, 215, 55),
       (35, 170, 190), (50, 70, 200), (135, 50, 195), (215, 80, 200)]
DV = {"N": (0, -1), "S": (0, 1), "E": (1, 0), "W": (-1, 0)}

cells = {(c["x"], c["z"]): c for c in C["cells"]}

ax = lambda x: PADL + x * C_C
ay = lambda z: PADT + z * C_C


def region(x0, z0, x1, z1, color, label=None, lx=0, ly=0):
    x0, x1 = min(x0, x1), max(x0, x1)
    z0, z1 = min(z0, z1), max(z0, z1)
    dr.rectangle([ax(x0) - 2, ay(z0) - 2, ax(x1) + C_C + 1, ay(z1) + C_C + 1],
                 outline=color, width=2)
    if label:
        lxp = max(6, ax(x0) + lx)
        lxp = min(lxp, img.width - 8 * len(label) // 2)
        T((lxp, ay(z0) + ly), label, fill=color, anchor="lb")


# ---------- 格子 ----------
for z in range(L):
    for x in range(W):
        c = cells.get((x, z))
        X, Y = ax(x), ay(z)
        if not c:
            continue
        t = c["t"]
        if t == "cmp":
            sub = c["mode"] == "subtract"
            dr.rectangle([X + 1, Y + 1, X + C_C - 2, Y + C_C - 2],
                         fill=(74, 58, 80) if sub else (44, 44, 56),
                         outline=(215, 96, 96) if sub else (110, 110, 130))
            dx, dz = DV[c["out"]]
            dr.line([X + C_C / 2 - dx * 5, Y + C_C / 2 - dz * 5 - 3,
                     X + C_C / 2 + dx * 5, Y + C_C / 2 + dz * 5 - 3],
                    fill=(255, 110, 110) if sub else (255, 200, 90), width=2)
            T((X + C_C / 2, Y + C_C / 2 + 8), str(c["st"]),
                    fill=(190, 255, 200), anchor="mm")
        elif t == "barrel":
            dr.rectangle([X + 1, Y + 1, X + C_C - 2, Y + C_C - 2],
                         fill=(150, 110, 70), outline=(90, 60, 30))
            T((X + C_C / 2, Y + C_C / 2), str(c["v"]),
                    fill=(255, 246, 220), anchor="mm")
        else:
            cc = c["c"] or ""
            if cc.startswith("col"):
                dr.rectangle([X + 1, Y + 1, X + C_C - 2, Y + C_C - 2],
                             fill=PAL[int(cc[3:]) % 8], outline=(235, 235, 235))
                if c["torch"]:
                    dr.ellipse([X + C_C - 11, Y + 2, X + C_C - 3, Y + 10],
                               fill=(255, 230, 120))
            elif cc == "white":
                dr.rectangle([X + 1, Y + 1, X + C_C - 2, Y + C_C - 2],
                             fill=(200, 200, 200))
            elif cc == "wire":
                dr.rectangle([X + 1, Y + 1, X + C_C - 2, Y + C_C - 2],
                             fill=(170, 70, 150))
            else:
                dr.rectangle([X + 1, Y + 1, X + C_C - 2, Y + C_C - 2],
                             fill=(100, 100, 112))

# ---------- 区域框（由 board_dump 实测坐标）----------
ch = C["chainRows"]
Zc = [z - oZ for z in ch]
def RZ(v): return v + oZ
region(oX - 3, RZ(-2), oX - 1, RZ(0),  (90, 210, 255),  "灰环 3x3  rx=-3..-1, Z=-2..0", -300, -4)
region(oX - 1, RZ(-18), oX - 1, RZ(0), (60, 150, 220),  "灰总线 rx=-1", -300, 2)
region(oX + 0, RZ(-21), oX + 2 * 4 - 1, RZ(0), (150, 220, 150), "位阵列+链 rx=0..2n-1", -300, 8)
region(oX + 2 * 4, RZ(-18), oX + 2 * 4, RZ(3), (220, 220, 220), "白总线 rx=2n", 2 * 4 + 2, 0)
region(oX + 2 * 4, RZ(1), oX + 2 * 4 + 2, RZ(3), (255, 255, 160), "白环 3x3", 2 * 4 + 2, -4)
region(oX + 1, RZ(2), oX + 7, RZ(2), (215, 80, 200), "MSB 行 Z=+2", -260, -12)
region(oX + 1, RZ(3), oX + 10, RZ(9), (255, 140, 60), "配对区 Z=+3..+9", 0, L * 0 + 14)

# ---------- 链行标注 j ----------
for j, z in enumerate(ch):
    Y = ay(z) + C_C / 2
    dr.line([14 + 78, Y, ax(0) - 4, Y], fill=(120, 200, 255), width=1)
    T((14, Y), f"j={j}  Z={Zc[j]}", fill=(120, 200, 255), anchor="lm")

# ---------- 原点 ----------
ox_, oy_ = ax(oX), ay(oZ)
dr.line([ox_ - 14, oy_ + C_C / 2, ox_ + C_C + 14, oy_ + C_C / 2],
        fill=(255, 80, 80), width=2)
dr.line([ox_ + C_C / 2, oy_ - 14, ox_ + C_C / 2, oy_ + C_C + 14],
        fill=(255, 80, 80), width=2)
dr.ellipse([ox_ + C_C / 2 - 5, oy_ + C_C / 2 - 5, ox_ + C_C / 2 + 5, oy_ + C_C / 2 + 5],
           outline=(255, 80, 80), width=2)
T((ox_ + C_C / 2, oy_ - 20), "原点 O (rx=0, Z=0)", fill=(255, 120, 120), anchor="mb")

# ---------- 坐标轴 ----------
for x in range(W):
    T((ax(x) + C_C / 2, PADT - 34), str(x), fill=(150, 150, 165), anchor="mm")
    T((ax(x) + C_C / 2, PADT - 16), str(x - oX), fill=(120, 200, 255), anchor="mm")
T((ax(0) - 90, PADT - 34), "绝对 x", fill=(150, 150, 165), anchor="lm")
T((ax(0) - 90, PADT - 16), "相对 rx", fill=(120, 200, 255), anchor="lm")

for z in range(0, L, 1):
    if (z - oZ) % 3 == 0 or z == oZ:
        T((PADL - 100, ay(z) + C_C / 2), str(z), fill=(150, 150, 165), anchor="rm")
        T((PADL - 52, ay(z) + C_C / 2), str(z - oZ), fill=(120, 200, 255), anchor="rm")
T((PADL - 100, PADT - 6), "绝对 z", fill=(150, 150, 165), anchor="rm")
T((PADL - 52, PADT - 6), "相对 Z", fill=(120, 200, 255), anchor="rm")

# ---------- 标题 ----------
T((PADL, 22), f"层1 ch0  b=8  x=(0,0,0,0)   {W}×{L}   "
                    f"K={C['K']}  W0={C['W0']}  head={C['head']}",
        fill=(240, 240, 250), anchor="la")
T((PADL, 44), "灰=实心块  红框=减法比较器  黄框=普通比较器  棕=容器  "
                    "亮角=火把已亮  品红=环内导线",
        fill=(150, 150, 165), anchor="la")

out = os.path.join(ROOT, "results", "annotated_layout.png")
img.save(out)
print("wrote", out, os.path.getsize(out) // 1024, "KB",
      f"{img.width}x{img.height}")
