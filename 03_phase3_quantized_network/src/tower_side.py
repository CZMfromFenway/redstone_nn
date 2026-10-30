# -*- coding: utf-8 -*-
"""一根火把塔的侧视图（y 竖直；2×2 截面展开成 4 列）。

电学闭环（逐条对上 data/torch_tower.schem 与用户口述）：

  A(ch) = (塔列, 槽行, 偶层 y=2ch)      塔加的彩色块
  B(ch) = (塔列, 读入行, 奇层 y=2ch+1)  塔加的彩色块
  T_tap(ch)   = (槽列, 槽行, 偶层)      搭在 A(ch) 上
  T_relay(ch) = (塔列, 读入行, 偶层)    搭在 A(ch) 上  ⇒ 与 T_tap 同状态
  T_car(ch)   = (塔列, 槽行, 奇层)      壁面：搭在 B(ch)；直立：搭在 A(ch)

  A(ch) 只被 T_car(ch-1) 充能；B(ch) 只被 T_relay(ch) 充能；
  输入块只被 T_tap(ch) 充能 ⇒ 比较器读 15（充能）或读容器 S=15+|W|（不充能）。

  壁面 carrier：T_car(ch) = ¬B(ch) = T_tap(ch)     —— 链不变号
  直立 carrier：T_car(ch) = ¬A(ch) = ¬T_tap(ch)    —— 链在此格变号（翻相）

  要求 T_tap(ch) = X_bit ⊕ s(ch)，s(ch)=[W>0]
  ⇒ carrier 格改直立 ⟺ s(ch+1) ≠ s(ch)
  ⇒ 塔底 A(0) 由总线直接驱动；s(0)=0 时塔底需再加一级反相（块+直立火把）

用法：python src/tower_side.py [i] [k] [ch数]
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
COLOR_NAME = {0: "红", 1: "橙", 2: "黄", 3: "绿", 4: "青", 5: "蓝", 6: "紫", 7: "品红"}
RGB = {"red": (176, 46, 46), "orange": (198, 118, 40), "yellow": (196, 182, 52),
       "lime": (92, 168, 52), "cyan": (44, 140, 152), "blue": (48, 72, 170),
       "purple": (110, 52, 158), "magenta": (170, 68, 158),
       "white": (208, 208, 208), "cmp": (62, 62, 76), "air": (22, 22, 28),
       "grid": (62, 62, 74), "lit": (255, 206, 96), "unlit": (86, 78, 76),
       "bus": (120, 200, 255)}
COLORS = ["red", "orange", "yellow", "lime", "cyan", "blue", "purple", "magenta"]


def slots(b):
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


def main():
    i_in = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    k_in = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    NCH = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    XBIT = 0

    ref = json.load(open(os.path.join(ROOT, "results", "py_compile_reference.json"),
                         encoding="utf-8"))
    Wm = ref["layers"]["L1"]["8"]["W"]
    b, n = 8, 4
    SLOT = slots(b)
    s = SLOT[k_in]
    chain = -6 * s["c"]
    if s["up"]:
        blockRow, readRow = chain - 2, chain - 1
        slotCol, otherCol = 2 * i_in, 2 * i_in + 1
    else:
        blockRow, readRow = chain + 2, chain + 1
        slotCol, otherCol = 2 * i_in + 1, 2 * i_in

    wcol = [Wm[ch][i_in][k_in] for ch in range(64)]
    sign = [1 if w > 0 else 0 for w in wcol]
    N = len(wcol)
    stand = [False] * N
    for ch in range(N - 1):
        stand[ch] = (sign[ch + 1] != sign[ch])
    tap = [1 if XBIT != sign[ch] else 0 for ch in range(N)]
    carrier = [(tap[ch] if stand[ch] else 1 - tap[ch]) for ch in range(N)]
    bottom_inv = (sign[0] == 0)
    # facing（箭头指向它搭的块；真正写进 schem 的 facing 是箭头反方向）
    tap_face = "W" if otherCol > slotCol else "E"
    car_face = "N" if readRow > blockRow else "S"
    rel_face = "S" if readRow > blockRow else "N"

    print(f"塔 (i={i_in}, k={k_in})  {'上槽' if s['up'] else '下槽'}  链行 Z={chain}")
    print(f"  槽列 rx={slotCol} (x={slotCol+3})   塔列 rx={otherCol} (x={otherCol+3})")
    print(f"  槽行 Z={blockRow} (z={blockRow+21})   读入行 Z={readRow} (z={readRow+21})")
    print(f"  facing: tap={tap_face}  carrier={car_face}  relay={rel_face}")
    print(f"  s(0)={sign[0]} -> 塔底{'需要' if bottom_inv else '不需要'}反相级")
    print(f"  ch    W   s  tap car  火把  比较器读")
    for ch in range(NCH):
        print(f"  {ch:2d} {wcol[ch]:5d}  {sign[ch]}   {tap[ch]}   {carrier[ch]}   "
              f"{'直立' if stand[ch] else '壁面'}  {'15' if tap[ch] else 'S=%d' % (15+abs(wcol[ch]))}")
    print(f"  全 64 通道翻相位置: {[c for c in range(N) if stand[c]]}")

    # ---------------- 画图 ----------------
    fp = r"C:\Windows\Fonts"
    def F(name, size):
        p = os.path.join(fp, name)
        return ImageFont.truetype(p, size) if os.path.exists(p) else ImageFont.load_default()
    fcn, fnum, fnum_s = F("msyh.ttc", 14), F("msyh.ttc", 11), F("msyh.ttc", 10)
    fmono = F("consola.ttf", 10)

    CW, CH = 74, 36
    colHdr = [(slotCol + 3, blockRow + 21, "平面·槽列"), (slotCol + 3, readRow + 21, "平面·槽列"),
              (otherCol + 3, blockRow + 21, "塔列"), (otherCol + 3, readRow + 21, "塔列")]
    ylo, yhi = -2, 2 * NCH - 1
    rows = yhi - ylo + 1
    M_L, M_T, M_R = 120, 104, 330
    img = Image.new("RGB", (M_L + CW * 4 + M_R, M_T + CH * rows + 150), (14, 14, 18))
    dr = ImageDraw.Draw(img)

    def cellrect(c, y):
        return M_L + c * CW, M_T + (yhi - y) * CH

    def torch(cx, cy, lit, standing, face, arrow):
        r = 9
        col = RGB["lit"] if lit else RGB["unlit"]
        if standing:
            dr.ellipse([cx - r, cy - r + 5, cx + r, cy + r + 5], fill=col, outline=(245, 245, 245))
        else:
            dr.ellipse([cx - r + 1, cy - r + 5, cx + r - 1, cy + r + 3], fill=col,
                       outline=(150, 150, 160))
        ax, ay = arrow
        ex, ey = cx + ax * 19, cy + 3 + ay * 15
        dr.line([cx + ax * 4, cy + 3 + ay * 4, ex, ey], fill=RGB["bus"], width=2)
        if ax:
            dr.polygon([(ex, ey), (ex - ax * 5, ey - 3), (ex - ax * 5, ey + 3)],
                       fill=RGB["bus"])
        else:
            dr.polygon([(ex, ey), (ex - 3, ey - ay * 5), (ex + 3, ey - ay * 5)],
                       fill=RGB["bus"])
        dr.text((cx, cy - 9), face, font=fmono, fill=(150, 205, 255), anchor="mm") if face else None

    def block(cx, cy, col, txt, sub=None):
        dr.rectangle([cx - 24, cy - 12, cx + 24, cy + 12], fill=col, outline=(155, 155, 165))
        dr.text((cx, cy - 3), txt, font=fnum, fill=(16, 16, 20), anchor="mm")
        if sub:
            dr.text((cx, cy + 7), sub, font=fnum_s, fill=(235, 235, 235), anchor="mm")

    for c, (xc, zc, tag) in enumerate(colHdr):
        X0 = M_L + c * CW
        dr.text((X0 + CW / 2, 30), tag, font=fnum, fill=(150, 215, 255), anchor="mm")
        dr.text((X0 + CW / 2, 50), "x=%d" % xc, font=fnum, fill=(190, 190, 200), anchor="mm")
        dr.text((X0 + CW / 2, 68), "z=%d" % zc, font=fnum, fill=(90, 200, 230), anchor="mm")
    dr.text((M_L + CW, 88), "← 平面原有格子", font=fnum_s, fill=(150, 150, 165), anchor="mm")
    dr.text((M_L + 3 * CW, 88), "塔的之字形 →", font=fnum_s, fill=(150, 150, 165), anchor="mm")

    for y in range(ylo, yhi + 1):
        odd = (y % 2 == 1)
        ch = (y - 1) // 2 if odd else y // 2
        for c in range(4):
            X0, Y0 = cellrect(c, y)
            dr.rectangle([X0 + 1, Y0 + 1, X0 + CW - 2, Y0 + CH - 2],
                         fill=RGB["air"], outline=RGB["grid"])
        cx = [cellrect(c, y)[0] + CW / 2 for c in range(4)]
        cy = cellrect(0, y)[1] + CH / 2
        # y 轴标签
        dr.text((M_L - 12, cy), "%d" % y, font=fmono, fill=(170, 170, 180), anchor="rm")
        if 0 <= ch < NCH and odd:
            dr.text((M_L - 44, cy), "ch%d" % ch, font=fnum, fill=(120, 200, 230), anchor="rm")
        if not (0 <= ch < NCH):
            continue
        colr = RGB[COLORS[k_in]]
        if not odd:
            # tap：搭在 A(ch)（画在右方）→ 箭头向右
            torch(cx[0], cy, tap[ch], False, tap_face, (1, 0))
            block(cx[1], cy, RGB["white"], "支撑")
            block(cx[2], cy, colr, "A(ch%d)" % ch)
            # relay：搭在 A(ch)（画在左方）→ 箭头向左
            torch(cx[3], cy, tap[ch], False, rel_face, (-1, 0))
        else:
            block(cx[0], cy, colr, "输入块", "W=%d" % wcol[ch])
            block(cx[1], cy, RGB["cmp"], "cmp", "→链")
            if stand[ch]:
                # 直立：搭在正下方的 A(ch) → 箭头向下
                torch(cx[2], cy, carrier[ch], True, "", (0, 1))
                dr.text((cx[2], cy + 15), "直立", font=fnum_s, fill=(255, 210, 90), anchor="mm")
                dr.rectangle([cellrect(2, y)[0] + 1, cellrect(2, y)[1] + 1,
                              cellrect(2, y)[0] + CW - 2, cellrect(2, y)[1] + CH - 2],
                             outline=(255, 210, 90), width=2)
            else:
                # 壁面：搭在同行的 B(ch)（画在右方）→ 箭头向右
                torch(cx[2], cy, carrier[ch], False, car_face, (1, 0))
            block(cx[3], cy, colr, "B(ch%d)" % ch)
            X1 = M_L + 4 * CW + 10
            dr.text((X1, cy - 7), "ch%-2d W=%d s=%d%s" % (ch, wcol[ch], sign[ch],
                                                          "  翻相" if stand[ch] else ""),
                    font=fnum_s, fill=(255, 210, 90) if stand[ch] else (185, 185, 195), anchor="lm")
            dr.text((X1, cy + 7), "tap=%d car=%d → 读%s"
                    % (tap[ch], carrier[ch], "15" if tap[ch] else "S"),
                    font=fnum_s, fill=(160, 205, 170) if tap[ch] else (165, 165, 180), anchor="lm")

    # 塔底两行：接口 / 可选反相级
    X0, Y0 = cellrect(2, -1)
    dr.rectangle([X0 + 1, Y0 + 1, X0 + 2 * CW - 2, Y0 + 2 * CH - 2], outline=(90, 90, 105))
    if bottom_inv:
        torch(X0 + CW / 2, Y0 + CH / 2, 0, True, "直立", (0, 1))
        block(X0 + CW / 2, Y0 + 3 * CH / 2, RGB["bus"], "总线", "x")
        dr.text((X0 + 2 * CW + 4, Y0 + CH), "塔底反相级（仅当 s(0)=0）：块 + 直立火把",
                font=fnum_s, fill=(255, 210, 90), anchor="lm")
    else:
        dr.text((X0 + 2 * CW + 4, Y0 + CH), "本塔 s(0)=1：总线直接驱动 A(0)，此处无额外方块",
                font=fnum_s, fill=(150, 150, 165), anchor="lm")

    ty = M_T + CH * rows + 14
    lines = [
        "圆点 = 火把（亮=黄，灭=灰）；蓝箭头 = 它搭在哪个块上（schem 里写的 facing 是箭头反方向）；白边 = 直立火把。",
        "A(ch)/B(ch) 是塔加的彩色块。T_tap 与 T_relay 搭同一个 A(ch) ⇒ 状态恒等；T_car 壁面搭 B(ch)、直立搭 A(ch)。",
        "链：T_car(ch-1) 充能 A(ch) → A 上的两个火把灭 → B(ch) 不充能 → T_car(ch) 亮。壁面时链不变号。",
        "翻相：carrier 格换直立 ⇒ T_car(ch) = 非 A(ch) = 非 T_tap(ch)，链在此格变号。规则：s(ch+1) 与 s(ch) 不同就换。",
        "本塔：s(0)=%d，塔底%s反相级；X=(0,0,0,0) 时 tap 只在 W<0 的通道亮（=比读器读 15）。"
        % (sign[0], "需要" if bottom_inv else "不需要"),
    ]
    for j, ln in enumerate(lines):
        dr.text((M_L - 104, ty + j * 21), ln, font=fnum_s, fill=(165, 165, 178))

    out = os.path.join(ROOT, "results", "tower_side_i%d_k%d.png" % (i_in, k_in))
    img.save(out)
    print(f"\n写出 {out}")


if __name__ == "__main__":
    main()
