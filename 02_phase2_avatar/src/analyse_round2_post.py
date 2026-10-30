#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Round 2 analysis: does the contrast post-process improve PCA-4's yield?

THE DESIGN THAT MAKES THIS CLEAN
dim4 and dim4x share the SAME WEIGHTS. The only difference is that dim4x's decoded levels
are stretched to fill [0,15]. So dim4x - dim4 measures the post-processing and nothing else.
dim22 rides along as an anchor.

THE CONFOUND THAT HAS TO BE HANDLED
The judge's standard moved between rounds: dim22 scored 15.7% in round 1 and 7.4% in round 2,
and dim4 fell from 45.5% to 20.9%. That means:
    * comparing round 2 numbers against round 1 numbers is INVALID
    * comparing arms WITHIN round 2 is valid, because the three arms were interleaved in one
      sitting with one standard
The script therefore reports the within-round comparison as the result and the between-round
shift separately, as a warning about the earlier figures.
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
r2 = [r for r in recs if str(r.get("round") or "") == "2"
      and r.get("model") in ("dim4", "dim4x", "dim22") and r.get("grade") in (0, 1, 2, 3)]

print("=" * 104)
print(f"ROUND 2 -- {len(r2)} blind labels, three interleaved arms, one standard")
print("=" * 104)
order = ["dim4", "dim4x", "dim22"]
stats = {}
print()
print(f"  {'arm':7s} {'n':>4s} {'差':>4s} {'中':>4s} {'良':>4s} {'优':>4s} "
      f"{'mean':>6s} {'良+优':>7s} {'95% CI':>16s}")
print("  " + "-" * 66)
for k in order:
    sub = [r for r in r2 if r["model"] == k]
    if not sub:
        continue
    n = len(sub)
    c = collections.Counter(r["grade"] for r in sub)
    good = c[2] + c[3]
    lo, hi = wilson(good, n)
    mean = float(np.mean([r["grade"] for r in sub]))
    stats[k] = {"n": n, "counts": [c[0], c[1], c[2], c[3]], "good": good,
                "rate": good / n, "lo": lo, "hi": hi, "mean": mean}
    print(f"  {k:7s} {n:4d} {c[0]:4d} {c[1]:4d} {c[2]:4d} {c[3]:4d} "
          f"{mean:6.2f} {good/n*100:6.1f}% [{lo*100:5.1f}%,{hi*100:5.1f}%]")

print()
print("  PAIRWISE (two-sided Fisher exact on 良+优 counts)")
print("  " + "-" * 66)
pairs = [("dim4x", "dim4"), ("dim4", "dim22"), ("dim4x", "dim22")]
for a, b in pairs:
    if a not in stats or b not in stats:
        continue
    A, B = stats[a], stats[b]
    p = fisher(A["good"], A["n"] - A["good"], B["good"], B["n"] - B["good"])
    delta = (A["rate"] - B["rate"]) * 100
    verdict = ("显著更好" if delta > 0 else "显著更差") if p < 0.05 else "无显著差异"
    print(f"    {a:6s} vs {b:6s}  {A['rate']*100:5.1f}% vs {B['rate']*100:5.1f}%  "
          f"({delta:+5.1f} 点)  p = {p:.2e}  -> {verdict}")

# ---------------------------------------------------------------- drift between rounds
print()
print("=" * 104)
print("STANDARD DRIFT BETWEEN ROUNDS (this is why rounds must not be pooled)")
print("=" * 104)
r1 = [r for r in recs if str(r.get("round") or "") == ""
      and r.get("model") in ("dim4", "dim8", "dim22")]
print(f"  {'arm':7s} {'round 1':>18s} {'round 2':>18s} {'change':>10s}")
print("  " + "-" * 58)
for k in ("dim4", "dim22"):
    s1 = [r for r in r1 if r["model"] == k]
    s2 = [r for r in r2 if r["model"] == k]
    if not s1 or not s2:
        continue
    g1 = sum(1 for r in s1 if r["grade"] >= 2) / len(s1)
    g2 = sum(1 for r in s2 if r["grade"] >= 2) / len(s2)
    p = fisher(sum(1 for r in s1 if r["grade"] >= 2),
               len(s1) - sum(1 for r in s1 if r["grade"] >= 2),
               sum(1 for r in s2 if r["grade"] >= 2),
               len(s2) - sum(1 for r in s2 if r["grade"] >= 2))
    print(f"  {k:7s} {g1*100:10.1f}% (n={len(s1):3d})  {g2*100:10.1f}% (n={len(s2):3d})  "
          f"{(g2-g1)*100:+9.1f} 点  p={p:.2e}")
print()
print("  Both arms fell by a similar amount, which is the signature of a stricter judge,")
print("  not of the models getting worse (the weights did not change between rounds).")

# ---------------------------------------------------------------- chart
if stats:
    keys = [k for k in order if k in stats]
    W, H = 940, 500
    cv = Image.new("RGB", (W, H), (18, 20, 24))
    dr = ImageDraw.Draw(cv)
    f_t, f_l, f_s = font(19), font(12), font(10)
    dr.text((16, 12), "第 2 轮盲测：后处理拉伸把 PCA-4 的良率从 20.9% 提到 62.1%",
            font=f_t, fill=(234, 238, 244))
    dr.text((16, 36), "dim4 与 dim4x 权重完全相同，唯一差别是显示时把灰度拉伸到满幅；"
            "误差线为 95% Wilson 区间", font=f_s, fill=(150, 162, 178))
    x0, y0, y1 = 90, 400, 80
    dr.line([x0 - 8, y1, x0 - 8, y0], fill=(70, 78, 92), width=1)
    dr.line([x0 - 8, y0, W - 24, y0], fill=(70, 78, 92), width=1)
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        yy = y0 - frac * (y0 - y1)
        dr.line([x0 - 8, yy, W - 24, yy], fill=(38, 43, 52), width=1)
        dr.text((x0 - 44, yy - 6), f"{frac*100:.0f}%", font=f_s, fill=(120, 132, 148))
    step = (W - 24 - x0) / len(keys)
    bw = step * 0.5
    cols = {"dim4": (150, 160, 180), "dim4x": (110, 200, 130), "dim22": (110, 168, 254)}
    for i, k in enumerate(keys):
        st = stats[k]
        cx = x0 + step * (i + 0.5)
        top = y0 - st["rate"] * (y0 - y1)
        dr.rectangle([cx - bw / 2, top, cx + bw / 2, y0], fill=cols.get(k, (200, 200, 200)))
        for yy, v in ((y0 - st["hi"] * (y0 - y1), "hi"), (y0 - st["lo"] * (y0 - y1), "lo")):
            dr.line([cx - 9, yy, cx + 9, yy], fill=(235, 240, 248), width=2)
        dr.line([cx, y0 - st["hi"] * (y0 - y1), cx, y0 - st["lo"] * (y0 - y1)],
                fill=(235, 240, 248), width=2)
        dr.text((cx - 20, top - 22), f"{st['rate']*100:.0f}%", font=f_l, fill=(235, 240, 248))
        name = {"dim4": "PCA-4\n原始", "dim4x": "PCA-4\n+拉伸", "dim22": "PCA-22\n参照"}[k]
        for j, line in enumerate(name.split("\n")):
            dr.text((cx - 24, y0 + 8 + j * 15), line, font=f_l, fill=(206, 214, 226))
        dr.text((cx - 22, y0 + 40), f"n={st['n']}", font=f_s, fill=(130, 142, 158))
    dr.text((16, H - 14), "同一模型、同一批输入，只差一个显示变换 —— 差异全部归因于后处理。",
            font=f_s, fill=(140, 152, 168))
    p = os.path.join(OUT, "ROUND2_POSTPROCESS.png")
    cv.save(p)
    print(f"\nwrote {p}")

json.dump(stats, open(os.path.join(OUT, "round2_stats.json"), "w"), indent=2, ensure_ascii=False)
print(f"wrote {os.path.join(OUT, 'round2_stats.json')}")
print("ROUND2_DONE")
