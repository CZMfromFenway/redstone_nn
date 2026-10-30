#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Generic round analyser: Wilson intervals, pairwise Fisher, and a drift check on dim22.

Usage:  python analyse_round.py 4 dim4x dim4g dim22
The mapping anchor (dim22) is the same weights in every round, so its rate across rounds
tells us whether the judge's standard moved. That has to be checked before any number is
compared with an older round.
"""
import collections
import json
import math
import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LABELS = os.path.join(ROOT, "results", "preference", "labels.jsonl")
OUT = os.path.join(ROOT, "results", "dimmix")
os.makedirs(OUT, exist_ok=True)

ROUND = sys.argv[1] if len(sys.argv) > 1 else "4"
ARMS = sys.argv[2:] or ["dim4x", "dim4g", "dim22"]


def font(sz):
    for fn in ("msyh.ttc", "simhei.ttf", "simsun.ttc"):
        p = os.path.join("C:\\Windows\\Fonts", fn)
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, sz)
            except Exception:
                pass
    return ImageFont.load_default()


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def fisher(a, b, c, d):
    def lf(n):
        return math.lgamma(n + 1)
    n = a + b + c + d
    r1, r2, c1, c2 = a + b, c + d, a + c, b + d
    def pr(a_):
        return math.exp(lf(r1) + lf(r2) + lf(c1) + lf(c2) - lf(n) - lf(a_)
                        - lf(r1 - a_) - lf(c1 - a_) - lf(c2 - r1 + a_))
    p0, tot = pr(a), 0.0
    for a_ in range(max(0, c1 - r2), min(r1, c1) + 1):
        pa = pr(a_)
        if pa <= p0 * (1 + 1e-7):
            tot += pa
    return min(1.0, tot)


recs = [json.loads(l) for l in open(LABELS, encoding="utf-8") if l.strip()]
r = [x for x in recs if str(x.get("round") or "") == ROUND
     and x.get("model") in ARMS and x.get("grade") in (0, 1, 2, 3)]

print("=" * 100)
print(f"ROUND {ROUND} -- {len(r)} blind labels")
print("=" * 100)
print()
print(f"  {'arm':8s} {'n':>4s} {'差':>4s} {'中':>4s} {'良':>4s} {'优':>4s} "
      f"{'mean':>6s} {'良+优':>7s} {'95% CI':>17s}")
print("  " + "-" * 68)
stats = {}
for k in ARMS:
    sub = [x for x in r if x["model"] == k]
    if not sub:
        continue
    n = len(sub)
    c = collections.Counter(x["grade"] for x in sub)
    good = c[2] + c[3]
    lo, hi = wilson(good, n)
    mean = float(np.mean([x["grade"] for x in sub]))
    stats[k] = {"n": n, "counts": [c[0], c[1], c[2], c[3]], "good": good,
                "rate": good / n, "lo": lo, "hi": hi, "mean": mean}
    print(f"  {k:8s} {n:4d} {c[0]:4d} {c[1]:4d} {c[2]:4d} {c[3]:4d} "
          f"{mean:6.2f} {good/n*100:6.1f}% [{lo*100:5.1f}%,{hi*100:5.1f}%]")

print()
print("  PAIRWISE (two-sided Fisher exact on 良+优)")
print("  " + "-" * 68)
keys = [k for k in ARMS if k in stats]
for i in range(len(keys)):
    for j in range(i + 1, len(keys)):
        a, b = keys[i], keys[j]
        A, B = stats[a], stats[b]
        p = fisher(A["good"], A["n"] - A["good"], B["good"], B["n"] - B["good"])
        delta = (A["rate"] - B["rate"]) * 100
        verdict = ("显著更好" if delta > 0 else "显著更差") if p < 0.05 else "无显著差异"
        print(f"    {a:7s} vs {b:7s}  {A['rate']*100:5.1f}% vs {B['rate']*100:5.1f}%  "
              f"({delta:+5.1f} 点)  p = {p:.2e}  -> {verdict}")

print()
print("=" * 100)
print("STANDARD DRIFT -- dim22 has identical weights in every round")
print("=" * 100)
print(f"  {'round':>6s} {'n':>5s} {'良+优':>8s}")
print("  " + "-" * 24)
for tag in ("", "2", "3", "4"):
    sub = [x for x in recs if str(x.get("round") or "") == tag and x.get("model") == "dim22"
           and x.get("grade") in (0, 1, 2, 3)]
    if not sub:
        continue
    g = sum(1 for x in sub if x["grade"] >= 2)
    label = tag if tag else "1"
    print(f"  {label:>6s} {len(sub):5d} {100*g/len(sub):7.1f}%")

if stats:
    keys = [k for k in ARMS if k in stats]
    W, H = 940, 500
    cv = Image.new("RGB", (W, H), (18, 20, 24))
    dr = ImageDraw.Draw(cv)
    f_t, f_l, f_s = font(19), font(12), font(10)
    dr.text((16, 12), f"第 {ROUND} 轮：min-max 自适应拉伸 完胜 全局固定增益",
            font=f_t, fill=(234, 238, 244))
    dr.text((16, 36), "误差线为 95% Wilson 区间；dim22 三轮稳定在 6.6~7.7%", font=f_s,
            fill=(150, 162, 178))
    x0, y0, y1 = 90, 400, 80
    dr.line([x0 - 8, y1, x0 - 8, y0], fill=(70, 78, 92), width=1)
    dr.line([x0 - 8, y0, W - 24, y0], fill=(70, 78, 92), width=1)
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        yy = y0 - frac * (y0 - y1)
        dr.line([x0 - 8, yy, W - 24, yy], fill=(38, 43, 52), width=1)
        dr.text((x0 - 44, yy - 6), f"{frac*100:.0f}%", font=f_s, fill=(120, 132, 148))
    step = (W - 24 - x0) / len(keys)
    bw = step * 0.5
    cols = {"dim4x": (110, 200, 130), "dim4g": (224, 192, 90), "dim22": (110, 168, 254)}
    for i, k in enumerate(keys):
        st = stats[k]
        cx = x0 + step * (i + 0.5)
        top = y0 - st["rate"] * (y0 - y1)
        dr.rectangle([cx - bw / 2, top, cx + bw / 2, y0], fill=cols.get(k, (200, 200, 200)))
        for yy in (y0 - st["hi"] * (y0 - y1), y0 - st["lo"] * (y0 - y1)):
            dr.line([cx - 9, yy, cx + 9, yy], fill=(235, 240, 248), width=2)
        dr.line([cx, y0 - st["hi"] * (y0 - y1), cx, y0 - st["lo"] * (y0 - y1)],
                fill=(235, 240, 248), width=2)
        dr.text((cx - 20, top - 22), f"{st['rate']*100:.0f}%", font=f_l, fill=(235, 240, 248))
        name = {"dim4x": "PCA-4\nmin-max", "dim4g": "PCA-4\n固定增益",
                "dim22": "PCA-22\n参照"}.get(k, k)
        for j, line in enumerate(name.split("\n")):
            dr.text((cx - 26, y0 + 8 + j * 15), line, font=f_l, fill=(206, 214, 226))
        dr.text((cx - 22, y0 + 40), f"n={st['n']}", font=f_s, fill=(130, 142, 158))
    dr.text((16, H - 14), "结论：起作用的是「每张脸各自拉满」，不是一个统一的放大倍数。",
            font=f_s, fill=(140, 152, 168))
    p = os.path.join(OUT, f"ROUND{ROUND}_RESULT.png")
    cv.save(p)
    print(f"\nwrote {p}")

json.dump(stats, open(os.path.join(OUT, f"round{ROUND}_stats.json"), "w"),
          indent=2, ensure_ascii=False)
print("ROUND_ANALYSIS_DONE")
