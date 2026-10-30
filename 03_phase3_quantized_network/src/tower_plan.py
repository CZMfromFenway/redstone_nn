# -*- coding: utf-8 -*-
"""n×b 个火把塔的 2×2 占地计算 + 与现有 64 片平面的冲突校验 + 平面图。

从 data/torch_tower.schem 反推出的规则（位0=红 / 位1=橙 / 位7=品红 三例都吻合）：

    每条链 j 有两个位：上槽位（槽行 = Zch(j)-2，读入行 = Zch(j)-1）
                      下槽位（槽行 = Zch(j)+2，读入行 = Zch(j)+1）
    读入行永远紧贴链行，槽行在链行 ±2。

    槽列 slotCol = CE(i)=2i（上槽）或 CO(i)=2i+1（下槽）——输入块与读入比较器都在这一列
    塔列 otherCol = 另一列

    每个通道，塔在这 3 个平面格上加 5 个方块：
        (slotCol, 槽行)   偶层：tap 火把     ← 给输入块供电
        (otherCol, 槽行)  偶层：支撑块 / 奇层：carrier 火把
        (otherCol, 读入行) 奇层：支撑块 / 偶层：relay 火把
    平面原有方块（输入块、读入比较器、其白色支撑）不动。

用法：python src/tower_plan.py
"""
import json
import os
import sys

from PIL import Image, ImageDraw, ImageFont

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PAL = [(200, 45, 45), (235, 140, 35), (240, 225, 55), (105, 215, 55),
       (35, 170, 190), (50, 70, 200), (135, 50, 195), (215, 80, 200)]
CELL = 30
MARG_L, MARG_T = 104, 78
PAD = 40


def slots(b):
    """与 plane_sim.html 的 build() 完全一致的槽位表。"""
    J = (b + 1) // 2
    out = []
    for j in range(J):
        out.append({"c": j, "Z": -6 * j + 2, "up": False})
        out.append({"c": j, "Z": -6 * j - 2, "up": True})
    out.sort(key=lambda s: (abs(s["Z"]), -s["Z"]))
    SLOT = [None] * b
    for i in range(b):
        SLOT[b - 1 - i] = out[i]
    return SLOT


def towers(n, b):
    """返回 [(i, k, dict)]，dict 给出 3 个平面格（rx）与两个行。"""
    SLOT = slots(b)
    res = []
    for i in range(n):
        for k in range(b):
            s = SLOT[k]
            chain = -6 * s["c"]
            if s["up"]:
                blockRow, readRow = chain - 2, chain - 1
                slotCol, otherCol = 2 * i, 2 * i + 1
            else:
                blockRow, readRow = chain + 2, chain + 1
                slotCol, otherCol = 2 * i + 1, 2 * i
            res.append({
                "i": i, "k": k, "up": s["up"], "chain": chain,
                "blockRow": blockRow, "readRow": readRow,
                "slotCol": slotCol, "otherCol": otherCol,
                # 3 个平面格（相对列 rx, 相对行 Z）
                "tap": (slotCol, blockRow),      # 偶层加火把
                "mid": (otherCol, blockRow),     # 偶层支撑块 + 奇层 carrier 火把
                "far": (otherCol, readRow),      # 奇层支撑块 + 偶层 relay 火把
            })
    return res


def main():
    d = json.load(open(os.path.join(ROOT, "results", "planes64.json"), encoding="utf-8"))
    W, L, b = d["W"], d["L"], d["b"]
    planes = d["planes"]
    originX, originZ = 3, 21
    n = 4

    occ = []          # 每通道：奇层占位集合、偶层占位集合
    for p in planes:
        odd, even = set(), set()
        for c in p["cells"]:
            odd.add((c["x"], c["z"]))
            if c["t"] == "cmp":
                even.add((c["x"], c["z"]))     # 支撑块在比较器正下方
        occ.append((odd, even))

    tw = towers(n, b)
    print(f"塔数 n×b = {n}×{b} = {len(tw)}")
    print("i  k  槽  链行  槽行/读入行  槽列 塔列 | 3 个平面格 (rx,Z) -> (x,z)")
    for t in tw:
        def ab(p):
            return (p[0] + originX, p[1] + originZ)
        print(f"{t['i']}  {t['k']}  {'上' if t['up'] else '下'}  {t['chain']:3d}  "
              f"{t['blockRow']:3d}/{t['readRow']:3d}      {t['slotCol']:2d}  {t['otherCol']:2d} | "
              f"{t['tap']}->{ab(t['tap'])}  {t['mid']}->{ab(t['mid'])}  {t['far']}->{ab(t['far'])}")

    # ---- 冲突检查：塔新增的位置在现有 64 片里必须全是空气 ----
    bad = []
    for idx, (odd, even) in enumerate(occ):
        for t in tw:
            for p, par in ((t["tap"], "even"), (t["mid"], "both"), (t["far"], "both")):
                x, z = p[0] + originX, p[1] + originZ
                if par in ("even", "both") and (x, z) in even:
                    bad.append((idx, t["i"], t["k"], p, "偶层已有方块"))
                if par in ("odd", "both") and (x, z) in odd:
                    # 奇层已有方块：只有"平面自己的输入块/比较器"允许，但那两格不在塔的 3 格里
                    bad.append((idx, t["i"], t["k"], p, "奇层已有方块"))
    print()
    if bad:
        print(f"✗ 冲突 {len(bad)} 处：")
        for r in bad[:12]:
            print("   ch=%d i=%d k=%d 格=%s  %s" % r)
    else:
        print(f"✓ 零冲突：32 个塔 × 64 通道 × 5 个方块 = {32*64*5} 个新增位置全部是空气，"
              f"不需要改动任何既有方块")

    # ---- 平面图：奇层（平面层）与偶层（塔层）两张 ----
    fp = r"C:\Windows\Fonts"
    def F(name, size):
        p = os.path.join(fp, name)
        return ImageFont.truetype(p, size) if os.path.exists(p) else ImageFont.load_default()
    fcn, fnum, fsm = F("msyh.ttc", 13), F("consola.ttf", 9), F("msyh.ttc", 11)

    ch0 = {(c["x"], c["z"]): c for c in planes[0]["cells"]}
    paW, paH = W * CELL, L * CELL
    img = Image.new("RGB", (MARG_L + 2 * paW + PAD + 30, MARG_T + paH + 120), (14, 14, 18))
    dr = ImageDraw.Draw(img)
    C_AIR, C_GRID = (24, 24, 30), (58, 58, 68)

    oddT, evenT = {}, {}
    for t in tw:
        oddT.setdefault((t["mid"][0] + originX, t["mid"][1] + originZ), []).append((t["k"], "carrier"))
        oddT.setdefault((t["far"][0] + originX, t["far"][1] + originZ), []).append((t["k"], "块"))
        evenT.setdefault((t["tap"][0] + originX, t["tap"][1] + originZ), []).append((t["k"], "tap"))
        evenT.setdefault((t["mid"][0] + originX, t["mid"][1] + originZ), []).append((t["k"], "块"))
        evenT.setdefault((t["far"][0] + originX, t["far"][1] + originZ), []).append((t["k"], "relay"))

    def draw(ox, title, layerT, show_plane):
        dr.text((ox, 16), title, font=fcn, fill=(150, 215, 255))
        for x in range(W):
            X0 = ox + x * CELL + CELL / 2
            dr.text((X0, 40), str(x), font=fnum, fill=(170, 170, 180), anchor="mm")
            dr.text((X0, 56), f"{x - originX:+d}", font=fnum, fill=(90, 200, 230), anchor="mm")
        for z in range(L):
            for x in range(W):
                X0, Y0 = ox + x * CELL, MARG_T + z * CELL
                dr.rectangle([X0, Y0, X0 + CELL - 1, Y0 + CELL - 1], fill=C_AIR, outline=C_GRID)
                c = ch0.get((x, z))
                if show_plane and c:
                    if c["t"] == "cmp":
                        dr.rectangle([X0 + 6, Y0 + 6, X0 + CELL - 7, Y0 + CELL - 7],
                                     fill=(72, 72, 88) if c["mode"] != "subtract" else (96, 62, 88))
                        dr.text((X0 + CELL / 2, Y0 + CELL / 2), c["out"], font=fnum,
                                fill=(255, 214, 110), anchor="mm")
                    elif c["t"] == "barrel":
                        dr.rectangle([X0 + 3, Y0 + 3, X0 + CELL - 4, Y0 + CELL - 4], fill=(96, 72, 46))
                elif (not show_plane) and c and c["t"] == "cmp":
                    dr.rectangle([X0 + 3, Y0 + 3, X0 + CELL - 4, Y0 + CELL - 4],
                                 fill=(60, 60, 60), outline=(110, 110, 120))
                marks = layerT.get((x, z))
                if marks:
                    k, kind = marks[0]
                    dr.rectangle([X0 + 1, Y0 + 1, X0 + CELL - 2, Y0 + CELL - 2],
                                 outline=PAL[k % 8], width=2)
                    dr.text((X0 + CELL / 2, Y0 + CELL / 2), f"k{k}", font=fsm,
                            fill=PAL[k % 8], anchor="mm")

    draw(MARG_L, "A. 平面层（y = 2ch+1）：塔在此层加 carrier 火把 + 支撑块", oddT, True)
    draw(MARG_L + paW + PAD, "B. 塔层（y = 2ch）：塔在此层加 tap/relay 火把 + 支撑块", evenT, False)

    for z in range(L):
        Y0 = MARG_T + z * CELL + CELL / 2
        dr.text((38, Y0), str(z), font=fnum, fill=(170, 170, 180), anchor="mm")
        dr.text((74, Y0), f"{z - originZ:+d}", font=fnum, fill=(90, 200, 230), anchor="mm")
    dr.text((38, MARG_T - 12), "z", font=fnum, fill=(170, 170, 180), anchor="mm")
    dr.text((74, MARG_T - 12), "Z", font=fnum, fill=(90, 200, 230), anchor="mm")

    ly = MARG_T + paH + 14
    dr.text((MARG_L, ly), "彩色框 = 该格的塔（框色 = 位号 k，数字 = k）。A 栏灰块/棕块是平面原有方块；"
                          "B 栏灰块是平面原有的比较器支撑（白色混凝土）。", font=fsm, fill=(160, 160, 172))
    dr.text((MARG_L, ly + 20), "塔的 3 个平面格 = (槽列,槽行)、(塔列,槽行)、(塔列,读入行)。"
                               "第 4 格 (槽列,读入行) 是平面自己的读入比较器，塔不碰。", font=fsm, fill=(160, 160, 172))

    out = os.path.join(ROOT, "results", "tower_plan.png")
    img.save(out)
    print(f"\n写出 {out}")


if __name__ == "__main__":
    main()
