#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""b=9 两种布局方案的蓝图对比（不看数值，只看行/链/桶/配对区的占位）。"""
import os

from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")

PAL = [(200, 45, 45), (235, 140, 35), (240, 225, 55), (105, 215, 55),
       (35, 170, 190), (50, 70, 200), (135, 50, 195), (215, 80, 200)]
ODD = [3, 5, 7, 9]
EVEN = [4, 6, 8, 10]

SCHEMES = {
    "A": dict(
        title="方案 A：MSB 锚定在 z=30（≡0 mod 6），4 条链，顶部 5 行空着",
        bits={0: 6, 1: 8, 2: 12, 3: 14, 4: 18, 5: 20, 6: 24, 7: 26},
        msb=30,
        chains=[10, 16, 22, 28],
        barrels={0: 5, 1: 7, 2: 7, 3: 13, 4: 13, 5: 19, 6: 19, 7: 25},
        pair_z0=31, L=38,
        note=["bit0 在 z=6，桶在 z=5（朝上）",
              "MSB z=30，链 28 从上方读它 ✓",
              "配对区 31..37，紧贴 MSB 下方 ✓",
              "链只有 4 条（10/16/22/28）",
              "顶部 z=0..4 整片浪费"],
    ),
    "B": dict(
        title="方案 B：保留自然行号（bit0 在 z=2），补第 5 条链在 z=28",
        bits={0: 2, 1: 6, 2: 8, 3: 12, 4: 14, 5: 18, 6: 20, 7: 24},
        msb=26,
        chains=[4, 10, 16, 22, 28],
        barrels={0: 1, 1: 7, 2: 7, 3: 13, 4: 13, 5: 19, 6: 19, 7: 25},
        pair_z0=27, L=34,
        note=["bit0 在 z=2，桶在 z=1（朝上）✓ 最省",
              "MSB z=26，链 28 在它下方（读 z=27 侧）",
              "配对区 27..33 与链 27/28/29 同行 ✗ 冲突",
              "链 5 条，但它和配对区抢 z=27..29"],
    ),
}


def draw(tag, sc):
    W, L = 14, sc["L"]
    C, PADL, PADT, LEG = 34, 60, 54, 320
    img = Image.new("RGB", (PADL + W*C + LEG, PADT + L*C + 12), (16, 16, 20))
    dr = ImageDraw.Draw(img)

    def T(xy, s, fill=(235, 235, 245), anchor="mm"):
        dr.text(xy, s, fill=fill, anchor=anchor)

    # 行底色：链行 / 配对区
    for zc in sc["chains"]:
        for dz in (-1, 0, 1):
            z = zc + dz
            if 0 <= z < L:
                dr.rectangle([PADL, PADT+z*C+1, PADL+W*C-1, PADT+(z+1)*C-2],
                             fill=(58, 34, 44) if dz else (78, 42, 56),
                             outline=(150, 80, 90))
    for z in range(sc["pair_z0"], min(sc["pair_z0"] + 7, L)):
        dr.rectangle([PADL, PADT+z*C+1, PADL+W*C-1, PADT+(z+1)*C-2],
                     fill=(48, 30, 52), outline=(150, 90, 170))

    # 桶
    for k, zb in sc["barrels"].items():
        for x in (ODD if k % 2 == 0 else EVEN):
            dr.rectangle([PADL+x*C+2, PADT+zb*C+2,
                          PADL+(x+1)*C-3, PADT+(zb+1)*C-3],
                         fill=(150, 110, 70), outline=(90, 60, 30))
    # 色行
    for k, zc in sc["bits"].items():
        col = ODD if k % 2 == 0 else EVEN
        for x in col:
            dr.rectangle([PADL+x*C+2, PADT+zc*C+2,
                          PADL+(x+1)*C-3, PADT+(zc+1)*C-3],
                         fill=PAL[k % 8], outline=(215, 215, 215))
    # MSB 行
    for x in EVEN:
        dr.rectangle([PADL+x*C+2, PADT+sc["msb"]*C+2,
                      PADL+(x+1)*C-3, PADT+(sc["msb"]+1)*C-3],
                     fill=PAL[7], outline=(255, 255, 255))
    # 配对区的两个环 + Δ桶示意
    zr = sc["pair_z0"]
    for z in range(zr, min(zr+7, L)):
        for x in (4, 6, 8, 10):
            dr.rectangle([PADL+x*C+8, PADT+z*C+8,
                          PADL+(x+1)*C-9, PADT+(z+1)*C-9],
                         outline=(200, 120, 220), width=1)

    for x in range(W):
        T((PADL+x*C+C//2, PADT-14), str(x), (185, 185, 195))
    for z in range(L):
        T((2, PADT+z*C+C//2), f"z{z}", (185, 185, 195), anchor="lm")
    T((PADL+W*C+10, PADT-18), sc["title"][:48], (240, 240, 250), anchor="la")
    for i, s in enumerate(sc["note"]):
        T((PADL+W*C+10, PADT+6+i*16), s, (215, 220, 230), anchor="la")
    T((PADL+W*C+10, PADT+6+len(sc["note"])*16+10),
      f"L = {L}", (255, 240, 160), anchor="la")
    fn = os.path.join(RESULTS, f"scheme_{tag}.png")
    img.save(fn)
    return fn, os.path.getsize(fn)


if __name__ == "__main__":
    for tag, sc in SCHEMES.items():
        fn, kb = draw(tag, sc)
        print(f"方案 {tag}: {fn}  {kb//1024}KB  L={sc['L']} 链数={len(sc['chains'])}")
    print("SCHEME_DONE")
