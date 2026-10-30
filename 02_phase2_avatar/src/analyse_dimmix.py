#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Analyse the blind mixed labelling over the PCA dimension ladder.

THE QUESTION
Not "which model reproduces a target face most accurately" but "which input dimension
produces the most usable RANDOM avatars". Accuracy was always the wrong metric for that:
a k=4 model scores only 0.21 on accuracy yet produces smoother output, and real faces are
smoother than most models' output.

THE DESIGN
label.html?mix=1 draws every face from a randomly chosen model of the ladder with FRESH
random parameters and never displays the model. So the labels are a controlled comparison:
one judge, one sitting, no knowledge of the source. The reference arm is PCA-22, the same
32-output recipe, un-purified.

STATISTICS
A good-rate from a finite sample deserves an interval, so each arm gets a Wilson score
interval rather than a bare percentage, and each arm is compared against the PCA-22 arm
with a two-sided Fisher exact test.
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
GRADE = {0: "差", 1: "中", 2: "良", 3: "优"}
ORDER = ["dim1", "dim2", "dim4", "dim8", "dim16", "dim22"]


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
    """Wilson score interval, which behaves far better than normal-approx at small n."""
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
    def p(a_):
        return math.exp(lf(r1) + lf(r2) + lf(c1) + lf(c2) - lf(n) - lf(a_)
                        - lf(r1 - a_) - lf(c1 - a_) - lf(c2 - r1 + a_))
    p0, tot = p(a), 0.0
    for a_ in range(max(0, c1 - r2), min(r1, c1) + 1):
        pa = p(a_)
        if pa <= p0 * (1 + 1e-7):
            tot += pa
    return min(1.0, tot)


recs = []
for line in open(LABELS, encoding="utf-8"):
    line = line.strip()
    if not line:
        continue
    try:
        r = json.loads(line)
    except Exception:
        continue
    if r.get("model") in ORDER and r.get("grade") in (0, 1, 2, 3):
        recs.append(r)

print("=" * 100)
print(f"BLIND MIXED LABELLING -- {len(recs)} labels across the dimension ladder")
print("=" * 100)
if not recs:
    print("  no ladder labels yet; run label.html?mix=1 and label some faces")
    raise SystemExit(0)

by = collections.defaultdict(list)
for r in recs:
    by[r["model"]].append(r)

print()
print(f"  {'model':7s} {'n':>4s} {'差':>4s} {'中':>4s} {'良':>4s} {'优':>4s} "
      f"{'mean':>6s} {'良+优':>7s} {'95% 区间':>16s}")
print("  " + "-" * 74)
stats = {}
for key in ORDER:
    rs = by.get(key, [])
    if not rs:
        continue
    n = len(rs)
    c = collections.Counter(r["grade"] for r in rs)
    good = c[2] + c[3]
    lo, hi = wilson(good, n)
    mean = float(np.mean([r["grade"] for r in rs]))
    stats[key] = {"n": n, "counts": [c[0], c[1], c[2], c[3]], "good": good,
                  "rate": good / n, "lo": lo, "hi": hi, "mean": mean}
    print(f"  {key:7s} {n:4d} {c[0]:4d} {c[1]:4d} {c[2]:4d} {c[3]:4d} "
          f"{mean:6.2f} {good/n*100:6.1f}% [{lo*100:5.1f}%,{hi*100:5.1f}%]")

ref = stats.get("dim22")
if ref:
    print()
    print("  each arm against the PCA-22 reference (two-sided Fisher exact on 良+优):")
    for key in ORDER:
        if key == "dim22" or key not in stats:
            continue
        st = stats[key]
        p = fisher(st["good"], st["n"] - st["good"],
                   ref["good"], ref["n"] - ref["good"])
        delta = (st["rate"] - ref["rate"]) * 100
        verdict = ("更好" if delta > 0 else "更差") if p < 0.05 else "无显著差异"
        print(f"    {key:7s} {st['rate']*100:5.1f}% vs {ref['rate']*100:5.1f}%  "
              f"({delta:+5.1f} 点)  p = {p:.4f}  -> {verdict}")

# ---------------------------------------------------------------- chart
keys = [k for k in ORDER if k in stats]
if keys:
    W, H = 900, 460
    cv = Image.new("RGB", (W, H), (18, 20, 24))
    dr = ImageDraw.Draw(cv)
    f_t, f_l, f_s = font(19), font(12), font(10)
    dr.text((16, 12), "盲测混合：输入维数 vs 随机头像良率", font=f_t, fill=(234, 238, 244))
    dr.text((16, 36), "每根柱＝该模型的「良+优」比例，误差线是 95% Wilson 区间；"
            "横线是 PCA-22 参照", font=f_s, fill=(150, 162, 178))
    x0, y0, y1 = 90, 400, 80
    dr.line([x0 - 8, y1, x0 - 8, y0], fill=(70, 78, 92), width=1)
    dr.line([x0 - 8, y0, W - 24, y0], fill=(70, 78, 92), width=1)
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        yy = y0 - frac * (y0 - y1)
        dr.line([x0 - 8, yy, W - 24, yy], fill=(38, 43, 52), width=1)
        dr.text((x0 - 44, yy - 6), f"{frac*100:.0f}%", font=f_s, fill=(120, 132, 148))
    bw = (W - 24 - x0) / max(len(keys), 1) * 0.55
    step = (W - 24 - x0) / max(len(keys), 1)
    for i, k in enumerate(keys):
        st = stats[k]
        cx = x0 + step * (i + 0.5)
        top = y0 - st["rate"] * (y0 - y1)
        col = (110, 168, 254) if k == "dim22" else (110, 200, 130)
        dr.rectangle([cx - bw / 2, top, cx + bw / 2, y0], fill=col)
        lo, hi = st["lo"], st["hi"]
        dr.line([cx, y0 - hi * (y0 - y1), cx, y0 - lo * (y0 - y1)],
                fill=(235, 240, 248), width=2)
        dr.line([cx - 7, y0 - hi * (y0 - y1), cx + 7, y0 - hi * (y0 - y1)],
                fill=(235, 240, 248), width=2)
        dr.line([cx - 7, y0 - lo * (y0 - y1), cx + 7, y0 - lo * (y0 - y1)],
                fill=(235, 240, 248), width=2)
        dr.text((cx - 18, top - 20), f"{st['rate']*100:.0f}%", font=f_l,
                fill=(235, 240, 248))
        dr.text((cx - 14, y0 + 8), k.replace("dim", "PCA-"), font=f_l, fill=(206, 214, 226))
        dr.text((cx - 12, y0 + 24), f"n={st['n']}", font=f_s, fill=(130, 142, 158))
    if ref:
        ry = y0 - ref["rate"] * (y0 - y1)
        dr.line([x0 - 8, ry, W - 24, ry], fill=(240, 200, 110), width=1)
    dr.text((16, H - 22), "柱子越高越好。若低维柱显著高于 PCA-22，说明「能生成合理随机头像」"
            "并不需要那么多输入。", font=f_s, fill=(140, 152, 168))
    p = os.path.join(OUT, "DIM_YIELD.png")
    cv.save(p)
    print(f"\nwrote {p}")
    json.dump({"stats": stats, "order": keys},
              open(os.path.join(OUT, "dim_yield.json"), "w"), indent=2, ensure_ascii=False)
    print(f"wrote {os.path.join(OUT, 'dim_yield.json')}")
print("DIMMIX_ANALYSIS_DONE")
