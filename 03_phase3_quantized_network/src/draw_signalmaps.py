# -*- coding: utf-8 -*-
"""四个测试输入下的"比较器全局信号图"。"""
import json
import os

from PIL import Image, ImageDraw

ROOT = r"E:\redstone_nn\03_phase3_quantized_network"
D = json.load(open(os.path.join(ROOT, "results", "board_dump.json"), encoding="utf-8"))

C = 26
PADL, PADT = 44, 34
GAP = 26
PAL = [(200, 45, 45), (235, 140, 35), (240, 225, 55), (105, 215, 55),
       (35, 170, 190), (50, 70, 200), (135, 50, 195), (215, 80, 200)]
DV = {"N": (0, -1), "S": (0, 1), "E": (1, 0), "W": (-1, 0)}

cases = D["cases"]
W = max(c["W"] for c in cases)
L = max(c["L"] for c in cases)
cols = 2
rows = (len(cases) + cols - 1) // cols

pw = PADL + W * C + 8
ph = PADT + L * C + 40
img = Image.new("RGB", (pw * cols + GAP, ph * rows + GAP), (12, 12, 16))
dr = ImageDraw.Draw(img)


def panel(ox, oy, case):
    cells = {}
    for c in case["cells"]:
        cells[(c["x"], c["z"])] = c
    oX, oZ = case["originX"], case["originZ"]
    # 信号热图：比较器
    for z in range(case["L"]):
        for x in range(case["W"]):
            c = cells.get((x, z))
            X, Y = ox + x * C, oy + z * C
            if not c:
                continue
            t = c["t"]
            if t == "cmp":
                st = c["st"] or 0
                k = min(1.0, (st % 1000) / 1000.0)
                if c["mode"] == "subtract":
                    fill = (int(70 + 90 * k), int(40 + 30 * k), int(60 + 90 * k))
                else:
                    fill = (int(28 + 40 * k), int(34 + 90 * k), int(44 + 60 * k))
                dr.rectangle([X + 1, Y + 1, X + C - 2, Y + C - 2], fill=fill,
                             outline=(90, 90, 110))
                dr.text((X + C // 2, Y + C // 2), str(st),
                        fill=(230, 255, 235), anchor="mm")
            elif t == "barrel":
                dr.rectangle([X + 1, Y + 1, X + C - 2, Y + C - 2],
                             fill=(96, 72, 46), outline=(60, 40, 20))
                dr.text((X + C // 2, Y + C // 2), str(c["v"]),
                        fill=(255, 240, 210), anchor="mm")
            else:
                cc = c["c"] or ""
                if cc.startswith("col"):
                    col = PAL[int(cc[3:]) % 8]
                    dr.rectangle([X + 1, Y + 1, X + C - 2, Y + C - 2], fill=col,
                                 outline=(230, 230, 230))
                    dr.text((X + C // 2, Y + C // 2),
                            "on" if c["torch"] > 0 else "off",
                            fill=(10, 10, 10), anchor="mm")
                elif cc == "white":
                    dr.rectangle([X + 1, Y + 1, X + C - 2, Y + C - 2],
                                 fill=(190, 190, 190))
                elif cc == "wire":
                    dr.rectangle([X + 1, Y + 1, X + C - 2, Y + C - 2],
                                 fill=(150, 60, 130))
                else:
                    dr.rectangle([X + 1, Y + 1, X + C - 2, Y + C - 2],
                                 fill=(96, 96, 106))
    # 坐标
    for x in range(case["W"]):
        dr.text((ox + x * C + C // 2, oy - 11), str(x - oX),
                fill=(120, 120, 135), anchor="mm")
    for z in range(case["L"]):
        rel = z - oZ
        dr.text((ox - 14, oy + z * C + C // 2), str(rel),
                fill=(90, 200, 250) if rel == 0 else (120, 120, 135), anchor="mm")
    # 头部
    hx, hz = case["headPos"]
    dr.rectangle([ox + hx * C, oy + hz * C, ox + (hx + 1) * C - 1, oy + (hz + 1) * C - 1],
                 outline=(120, 255, 150), width=2)
    ok = case["head"] == case["N"]
    dr.text((ox, oy - 26), f"{case['name']}   x=({','.join(map(str, case['x']))})",
            fill=(240, 240, 250), anchor="la")
    dr.text((ox, oy + case["L"] * C + 8),
            f"head={case['head']}  N={case['N']}  {'OK' if ok else 'WRONG'}",
            fill=(130, 225, 165) if ok else (255, 120, 120), anchor="la")


for i, case in enumerate(cases):
    r, c = divmod(i, cols)
    panel(GAP + c * pw, GAP + r * ph, case)

out = os.path.join(ROOT, "results", "signal_maps_4cases.png")
img.save(out)
print("wrote", out, os.path.getsize(out) // 1024, "KB")
for case in cases:
    print(f"  {case['name']:10s} head={case['head']} N={case['N']}")
