# -*- coding: utf-8 -*-
"""出对比图：inRev=0（对） vs inRev=1（错），同一通道。"""
import json
import os

from PIL import Image, ImageDraw

ROOT = r"E:\redstone_nn\03_phase3_quantized_network"
J = json.load(open(os.path.join(ROOT, "results", "sim_cmp_ch1.json"), encoding="utf-8"))

PAL = [(200, 45, 45), (235, 140, 35), (240, 225, 55), (105, 215, 55),
       (35, 170, 190), (50, 70, 200), (135, 50, 195), (215, 80, 200)]
DV = {"N": (0, -1), "S": (0, 1), "E": (1, 0), "W": (-1, 0)}

C = 34
PADL, PADT = 46, 46
GAP = 40


def panel(dr, ox0, oy0, d, title):
    cells = {}
    for c in d["cells"]:
        cells[(c["x"], c["z"])] = c
    W, L = d["W"], d["L"]
    for z in range(L):
        for x in range(W):
            c = cells.get((x, z))
            X, Y = ox0 + x * C, oy0 + z * C
            if not c:
                continue
            t = c["t"]
            if t == "cmp":
                sub = c.get("mode") == "subtract"
                fill = (74, 58, 80) if sub else (44, 44, 56)
                dr.rectangle([X + 1, Y + 1, X + C - 2, Y + C - 2], fill=fill,
                             outline=(215, 96, 96) if sub else (106, 106, 128))
                if c.get("src"):
                    dr.rectangle([X + 1, Y + 1, X + C - 2, Y + C - 2],
                                 outline=(90, 210, 255), width=2)
                dx, dz = DV[c["out"]]
                cx, cy = X + C // 2, Y + C // 2
                dr.line([cx - dx * 5, cy - dz * 5 - 3, cx + dx * 5, cy + dz * 5 - 3],
                        fill=(255, 96, 96) if sub else (255, 196, 86), width=2)
                dr.text((cx, cy + 7), str(c.get("st", "")), fill=(180, 255, 190), anchor="mm")
            elif t == "barrel":
                dr.rectangle([X + 2, Y + 2, X + C - 3, Y + C - 3],
                             fill=(150, 110, 70), outline=(90, 60, 30))
                dr.text((X + C // 2, Y + C // 2), str(c.get("v", "")),
                        fill=(255, 246, 220), anchor="mm")
            else:
                col = (110, 110, 120)
                cc = c.get("c") or ""
                if cc.startswith("col"):
                    col = PAL[int(cc[3:]) % 8]
                elif cc == "white":
                    col = (200, 200, 200)
                elif cc == "wire":
                    col = (176, 74, 154)
                dr.rectangle([X + 2, Y + 2, X + C - 3, Y + C - 3], fill=col,
                             outline=(220, 220, 220))
                if cc.startswith("col"):
                    dr.text((X + C // 2, Y + C // 2), str(c.get("torch", 0)),
                            fill=(10, 10, 10), anchor="mm")
    # 坐标
    for x in range(W):
        dr.text((ox0 + x * C + C // 2, oy0 - 14), str(x - d["oX"]), fill=(150, 150, 162), anchor="mm")
    for z in range(L):
        dz = z - d["oZ"]
        dr.text((ox0 - 16, oy0 + z * C + C // 2), "Z0" if dz == 0 else str(dz),
                fill=(90, 210, 255) if dz == 0 else (150, 150, 162), anchor="mm")
    # 头部高亮
    hz = d["xwRow"]
    dr.rectangle([ox0 + d["xw"] * C, oy0 + hz * C,
                  ox0 + (d["xw"] + 1) * C - 1, oy0 + (hz + 1) * C - 1],
                 outline=(127, 255, 159), width=3)
    ok = d["head"] == d["N"]
    dr.text((ox0, oy0 - 34), title, fill=(240, 240, 250), anchor="la")
    dr.text((ox0, oy0 + L * C + 16),
            f"head = {d['head']}   N = {d['N']}   {'OK' if ok else 'WRONG'}",
            fill=(127, 224, 160) if ok else (255, 120, 120), anchor="la")


d0, d1 = J["ir0"], J["ir1"]
w = PADL + max(d0["W"], d1["W"]) * C + 20
h = PADT + max(d0["L"], d1["L"]) * C + 46
img = Image.new("RGB", (w * 2 + GAP, h), (14, 14, 18))
dr = ImageDraw.Draw(img)
panel(dr, PADL, PADT, d0, "位权镜像 = 关   （低位在前）")
panel(dr, PADL + w + GAP, PADT, d1, "位权镜像 = 开   （Z → −Z 整体翻转）")
out = os.path.join(ROOT, "results", "sim_compare_bitrev.png")
img.save(out)
print("wrote", out, os.path.getsize(out) // 1024, "KB")
print("MSB 信号 (按输入):", d0["S"], " 两图相同")
