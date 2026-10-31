#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成地图基色表（62 项）与配图。数据源：Minecraft Wiki「地图存储格式」§地图颜色的算法（1.21）。

修饰效果（Wiki 原文）：
    M=0 LOW     × 180/255
    M=1 NORMAL  × 220/255
    M=2 HIGH    × 255/255（不变，即基色本身）
    M=3 LOWEST  × 135/255
取整为向下取整（由 Wiki 表格的十六进制反推：如 GRASS 127×220/255=109.57 → 0x6D=109）。
"""
import os
from PIL import Image, ImageDraw

# (id, 枚举名, HIGH 档颜色 = 基色)
BASE = [
    (0, "NONE", None), (1, "GRASS", "#7FB238"), (2, "SAND", "#F7E9A3"),
    (3, "WOOL", "#C7C7C7"), (4, "FIRE", "#FF0000"), (5, "ICE", "#A0A0FF"),
    (6, "METAL", "#A7A7A7"), (7, "PLANT", "#007C00"), (8, "SNOW", "#FFFFFF"),
    (9, "CLAY", "#A4A8B8"), (10, "DIRT", "#976D4D"), (11, "STONE", "#707070"),
    (12, "WATER", "#4040FF"), (13, "WOOD", "#8F7748"), (14, "QUARTZ", "#FFFCF5"),
    (15, "COLOR_ORANGE", "#D87F33"), (16, "COLOR_MAGENTA", "#B24CD8"),
    (17, "COLOR_LIGHT_BLUE", "#6699D8"), (18, "COLOR_YELLOW", "#E5E533"),
    (19, "COLOR_LIGHT_GREEN", "#7FCC19"), (20, "COLOR_PINK", "#F27FA5"),
    (21, "COLOR_GRAY", "#4C4C4C"), (22, "COLOR_LIGHT_GRAY", "#999999"),
    (23, "COLOR_CYAN", "#4C7F99"), (24, "COLOR_PURPLE", "#7F3FB2"),
    (25, "COLOR_BLUE", "#334CB2"), (26, "COLOR_BROWN", "#664C33"),
    (27, "COLOR_GREEN", "#667F33"), (28, "COLOR_RED", "#993333"),
    (29, "COLOR_BLACK", "#191919"), (30, "GOLD", "#FAEE4D"),
    (31, "DIAMOND", "#5CDBD5"), (32, "LAPIS", "#4A80FF"),
    (33, "EMERALD", "#00D93A"), (34, "PODZOL", "#815631"),
    (35, "NETHER", "#700200"), (36, "TERRACOTTA_WHITE", "#D1B1A1"),
    (37, "TERRACOTTA_ORANGE", "#9F5224"), (38, "TERRACOTTA_MAGENTA", "#95576C"),
    (39, "TERRACOTTA_LIGHT_BLUE", "#706C8A"), (40, "TERRACOTTA_YELLOW", "#BA8524"),
    (41, "TERRACOTTA_LIGHT_GREEN", "#677535"), (42, "TERRACOTTA_PINK", "#A04D4E"),
    (43, "TERRACOTTA_GRAY", "#392923"), (44, "TERRACOTTA_LIGHT_GRAY", "#876B62"),
    (45, "TERRACOTTA_CYAN", "#575C5C"), (46, "TERRACOTTA_PURPLE", "#7A4958"),
    (47, "TERRACOTTA_BLUE", "#4C3E5C"), (48, "TERRACOTTA_BROWN", "#4C3223"),
    (49, "TERRACOTTA_GREEN", "#4C522A"), (50, "TERRACOTTA_RED", "#8E3C2E"),
    (51, "TERRACOTTA_BLACK", "#251610"), (52, "CRIMSON_NYLIUM", "#BD3031"),
    (53, "CRIMSON_STEM", "#943F61"), (54, "CRIMSON_HYPHAE", "#5C191D"),
    (55, "WARPED_NYLIUM", "#167E86"), (56, "WARPED_STEM", "#3A8E8C"),
    (57, "WARPED_HYPHAE", "#562C3E"), (58, "WARPED_WART_BLOCK", "#14B485"),
    (59, "DEEPSLATE", "#646464"), (60, "RAW_IRON", "#D8AF93"),
    (61, "GLOW_LICHEN", "#7FA796"),
]
MUL = {0: 180 / 255, 1: 220 / 255, 2: 1.0, 3: 135 / 255}
MNAME = {0: "LOW", 1: "NORMAL", 2: "HIGH", 3: "LOWEST"}


def hx(s):
    s = s.lstrip('#')
    return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))


def shade(rgb, m):
    return tuple(int(c * MUL[m]) for c in rgb)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    fig_dir = os.path.join(here, "..", "..", "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # ---- 图：62 行 × 4 修饰 ----
    SW, SH, LBL, HDR = 58, 22, 250, 54
    W = LBL + SW * 4 + 10
    H = HDR + SH * len(BASE) + 10
    img = Image.new("RGB", (W, H), (250, 250, 248))
    d = ImageDraw.Draw(img)
    d.text((8, 6), "Minecraft map colours (1.21): 62 bases x 4 modifiers", fill=(20, 20, 20))
    for i, m in enumerate([0, 1, 2, 3]):
        x = LBL + i * SW + 6
        d.text((x, 22), MNAME[m], fill=(20, 20, 20))
        d.text((x, 38), f"x{int(MUL[m]*255)}/255", fill=(90, 90, 90))
    for row, (idx, name, h) in enumerate(BASE):
        y = HDR + row * SH
        label = f"{idx:2d} {name}"
        d.text((8, y + 5), label[:34], fill=(20, 20, 20))
        if h is None:
            for i in range(4):
                x = LBL + i * SW
                for k in range(0, SW - 2, 8):
                    d.rectangle([x + k, y, x + min(k + 4, SW - 2), y + SH - 2],
                                fill=(210, 210, 210))
            continue
        rgb = hx(h)
        for i, m in enumerate([0, 1, 2, 3]):
            x = LBL + i * SW
            d.rectangle([x, y, x + SW - 3, y + SH - 2], fill=shade(rgb, m))
    out = os.path.join(fig_dir, "map_colors_62x4.png")
    img.save(out)
    print("PNG:", os.path.normpath(out), img.size)

    # ---- 表：id / 枚举名 / 基色(HIGH) ----
    lines = ["| id | 枚举名 | 基色（= HIGH） | LOW | NORMAL | LOWEST |",
             "|---:|---|---|---|---|---|"]
    for idx, name, h in BASE:
        if h is None:
            lines.append(f"| 0 | `NONE` | 透明 | — | — | — |")
            continue
        rgb = hx(h)
        f = lambda m: "#%02X%02X%02X" % shade(rgb, m)
        lines.append(f"| {idx} | `{name}` | `{h}` | `{f(0)}` | `{f(1)}` | `{f(3)}` |")
    tbl = os.path.join(here, "map_colors_table.md")
    open(tbl, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("TABLE:", os.path.normpath(tbl), len(lines) - 2, "rows")


if __name__ == "__main__":
    main()
