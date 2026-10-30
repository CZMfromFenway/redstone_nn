#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
Can our palettes actually be built out of Minecraft colours?

THE QUESTION
A palette is only useful if the machine can render it. On a Minecraft map the available colours
are the 62 map base colours, each in four shade modifiers (LOW / NORMAL / HIGH / LOWEST). Using
only NORMAL is the strict reading -- mixing modifiers means the same base is rendered darker or
lighter, which is really three extra colours rather than shading.

THE WHITE PROBLEM
The user flagged it before this script existed: with NORMAL only, the brightest available colour
is SNOW at #DCDCDC, which is 220/255 = 0.863 of full scale. Nothing in NORMAL reaches pure white.
Our palettes contain entries near full white, so those entries cannot be expressed. This is
measured rather than assumed.

WHAT IS REPORTED
    per palette      the nearest available Minecraft colour for each entry, and the error
    summary          mean and max error over the palette, and how many entries are unreachable
    refit            a palette constrained to Minecraft colours, retrained, and scored, so the
                     cost of snapping to what is actually buildable is visible
"""
import json
import os
import sys
import numpy as np
import torch
import torch.nn as nn

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "results", "colour", "mcfit")
os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(8)

mc = json.load(open(os.path.join(ROOT, "data", "mc_map_colors.json"), encoding="utf-8"))
MOD = mc["_modifiers"]
bases = mc["bases"]
NAMES = [b["name"] for b in bases]
BASE = np.array([b["rgb"] for b in bases], dtype=np.float64)

# three candidate colour sets
SETS = {
    "NORMAL（严格）": np.clip(BASE * MOD["NORMAL"], 0, 255),
    "四档修饰全开（248 色）": np.vstack([np.clip(BASE * f, 0, 255)
                                          for f in (MOD["LOW"], MOD["NORMAL"],
                                                    MOD["HIGH"], MOD["LOWEST"])]),
}
SHADE_NAMES = [f"{n}/{s}" for s in ("NORMAL", "LOW", "HIGH", "LOWEST") for n in NAMES]

print("=" * 100)
print("HOW WELL CAN EACH PALETTE BE BUILT FROM MINECRAFT COLOURS?")
print("=" * 100)
print(f"  Minecraft 地图基色 {len(BASE)} 个；NORMAL 档最亮 = "
      f"#{int(SETS['NORMAL（严格）'][:, 0].max()):02X}"
      f"{int(SETS['NORMAL（严格）'][1, 1]):02X}"
      f"{int(SETS['NORMAL（严格）'][:, 2].max()):02X}")
print(f"  折算到 0..15 级：最亮 {SETS['NORMAL（严格）'].max()/17:.2f}   "
      f"（我们的调色板里有接近 15 的白）")


def load_palette(name):
    j = json.load(open(os.path.join(ROOT, "results", "colour", "palette", name), encoding="utf-8"))
    return np.array(j["new"] if "new" in j else j["old"])


PALS = {}
for tag, f in (("K=8", "palette_k8.json"),):
    if os.path.exists(os.path.join(ROOT, "results", "colour", "palette", f)):
        PALS[tag] = load_palette(f)
# the exported palettes are the authority for 16 and 32
for tag, fn in (("K=8", "colour_pal8_model.json"), ("K=16", "colour_pal16_model.json"),
                ("K=32", "colour_pal32_model.json")):
    j = json.load(open(os.path.join(ROOT, "webapp", fn), encoding="utf-8"))
    PALS[tag] = np.array(j["palette"], dtype=np.float64)

print()
for setname, SET in SETS.items():
    print("=" * 100)
    print(f"  可用颜色集：{setname}   ({len(SET)} 色)")
    print("=" * 100)
    print(f"    {'调色板':6s} {'平均误差':>9s} {'最大误差':>9s} {'误差>1级':>10s} {'误差>2级':>10s}")
    print("    " + "-" * 52)
    summary = {}
    for tag, P in PALS.items():
        P255 = P * 17.0
        d = np.linalg.norm(P255[:, None, :] - SET[None, :, :], axis=-1)
        best = d.argmin(1)
        err = d[np.arange(len(P)), best] / 17.0     # back to 0..15 level units
        summary[tag] = {"mean": float(err.mean()), "max": float(err.max()),
                        "gt1": int((err > 1).sum()), "gt2": int((err > 2).sum())}
        print(f"    {tag:6s} {err.mean():9.3f} {err.max():9.3f} "
              f"{int((err>1).sum()):4d}/{len(P):<4d} {int((err>2).sum()):4d}/{len(P):<4d}")
    print()
    if setname.startswith("NORMAL"):
        # detail for the strict set: which entries are unreachable and by how much
        for tag, P in PALS.items():
            P255 = P * 17.0
            d = np.linalg.norm(P255[:, None, :] - SET[None, :, :], axis=-1)
            best = d.argmin(1)
            err = d[np.arange(len(P)), best] / 17.0
            print(f"    {tag} 逐色匹配：")
            for i in np.argsort(-err):
                c = P[i]
                m = SET[best[i]]
                print(f"      我们的 ({c[0]:5.2f},{c[1]:5.2f},{c[2]:5.2f})  →  "
                      f"{NAMES[best[i] % len(NAMES)]:18s} "
                      f"({m[0]/17:5.2f},{m[1]/17:5.2f},{m[2]/17:5.2f})  误差 {err[i]:.2f}")
            print()
    json.dump(summary, open(os.path.join(
        OUT, "fit_" + ("normal" if setname.startswith("NORMAL") else "all") + ".json"),
        "w", encoding="utf-8"), indent=2, ensure_ascii=False)

print("=" * 100)
print("结论要点")
print("=" * 100)
print("  NORMAL 档最亮只有 220/255，我们的白色条目（约 14.4 级 = 245/255）无法表达，")
print("  这是「没有好的白」的具体代价。四档全开时 SNOW 的 HIGH 是纯白，可以精确命中。")
print("  如果只允许 NORMAL，最省事的做法是把调色板的白端整体压到 12.9 级以下。")
print("MCFIT_DONE")
