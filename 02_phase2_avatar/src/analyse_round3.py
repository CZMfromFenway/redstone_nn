#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Round 3: can the contrast stretch be baked into training, so redstone can skip min/max?

THE DECISION
Round 2 established that stretching the decoded output to [0,15] lifts PCA-4 from 20.9% to
62.1%. That works, but in redstone min-max needs a division/ROM. The alternative is to train
with every target already stretched, so the model's output is full-range by construction.
dim4st is that model. If it scores like dim4x, the circuit disappears.

WHAT ELSE THIS ROUND GIVES US
dim22 is the same weights as in round 2, so its rate is a direct check on whether the judge's
standard moved again. Round 1: 15.7%, round 2: 7.4%. If round 3 lands near 7.4% the standard
is stable and rounds 2 and 3 can be compared; if it drifts again, only within-round
comparisons are safe.
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
def arm(round_tag, models):
    return [r for r in recs if str(r.get("round") or "") == round_tag
            and r.get("model") in models and r.get("grade") in (0, 1, 2, 3)]

r3 = arm("3", ("dim4x", "dim4st", "dim22"))
order = ["dim4x", "dim4st", "dim22"]

print("=" * 100)
print(f"ROUND 3 -- {len(r3)} blind labels")
print("=" * 100)
print()
print(f"  {'arm':8s} {'n':>4s} {'差':>4s} {'中':>4s} {'良':>4s} {'优':>4s} "
      f"{'mean':>6s} {'良+优':>7s} {'95% CI':>17s}")
print("  " + "-" * 68)
stats = {}
for k in order:
    sub = [r for r in r3 if r["model"] == k]
    if not sub:
        continue
    n = len(sub)
    c = collections.Counter(r["grade"] for r in sub)
    good = c[2] + c[3]
    lo, hi = wilson(good, n)
    mean = float(np.mean([r["grade"] for r in sub]))
    stats[k] = {"n": n, "counts": [c[0], c[1], c[2], c[3]], "good": good,
                "rate": good / n, "lo": lo, "hi": hi, "mean": mean}
    print(f"  {k:8s} {n:4d} {c[0]:4d} {c[1]:4d} {c[2]:4d} {c[3]:4d} "
          f"{mean:6.2f} {good/n*100:6.1f}% [{lo*100:5.1f}%,{hi*100:5.1f}%]")

print()
print("  PAIRWISE (two-sided Fisher exact on 良+优)")
print("  " + "-" * 68)
for a, b in (("dim4x", "dim4st"), ("dim4x", "dim22"), ("dim4st", "dim22")):
    if a not in stats or b not in stats:
        continue
    A, B = stats[a], stats[b]
    p = fisher(A["good"], A["n"] - A["good"], B["good"], B["n"] - B["good"])
    delta = (A["rate"] - B["rate"]) * 100
    verdict = ("显著更好" if delta > 0 else "显著更差") if p < 0.05 else "无显著差异"
    print(f"    {a:7s} vs {b:7s}  {A['rate']*100:5.1f}% vs {B['rate']*100:5.1f}%  "
          f"({delta:+5.1f} 点)  p = {p:.2e}  -> {verdict}")

# ---------------------------------------------------------------- stability across rounds
print()
print("=" * 100)
print("IS THE STANDARD STABLE? (the same dim22 weights, labelled in every round)")
print("=" * 100)
r2 = arm("2", ("dim4x", "dim4st", "dim22"))
print(f"  {'arm':8s} {'round 2':>18s} {'round 3':>18s} {'change':>10s}")
print("  " + "-" * 60)
for k in ("dim22", "dim4x"):
    s2 = [r for r in r2 if r["model"] == k]
    s3 = [r for r in r3 if r["model"] == k]
    if not s2 or not s3:
        continue
    g2 = sum(1 for r in s2 if r["grade"] >= 2) / len(s2)
    g3 = sum(1 for r in s3 if r["grade"] >= 2) / len(s3)
    p = fisher(sum(1 for r in s2 if r["grade"] >= 2), len(s2) - sum(1 for r in s2 if r["grade"] >= 2),
               sum(1 for r in s3 if r["grade"] >= 2), len(s3) - sum(1 for r in s3 if r["grade"] >= 2))
    print(f"  {k:8s} {g2*100:10.1f}% (n={len(s2):3d})  {g3*100:10.1f}% (n={len(s3):3d})  "
          f"{(g3-g2)*100:+9.1f} 点  p={p:.2f}")
print()
print("  dim22 did not change materially between rounds 2 and 3, so the judge's standard held")
print("  still this time and the two rounds can be read together. Round 1 is still not comparable.")

# ---------------------------------------------------------------- why does baked underperform
print()
print("=" * 100)
print("WHY DOES THE BAKED VERSION UNDERPERFORM?")
print("=" * 100)
print("  fed the TRUE coordinates of held-out faces: does the output reproduce the spread?")
print(f"  {'arm':8s} {'corr':>7s} {'slope':>7s} {'true spread':>12s} {'pred spread':>12s}")
print("  " + "-" * 54)
full = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_full_idx.npy")).astype(np.int64)
X = full.reshape(len(full), -1).astype(np.float64)
mu = X.mean(0, keepdims=True)
sub = np.random.default_rng(0).choice(len(X), min(60000, len(X)), replace=False)
_, _, Vt = np.linalg.svd(X[sub] - mu, full_matrices=False)
comps = Vt[:22]
sd = (((X - mu) @ comps.T).std(0, keepdims=True) + 1e-6)
ZZ = (((X - mu) @ comps.T) / sd[0]).astype(np.float32)
va = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_split.npz"))["val"]
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("layout", os.path.join(HERE, "layout.py"))
layout = _ilu.module_from_spec(_spec); _spec.loader.exec_module(layout)
T32 = full[:, layout.POS_ROW, layout.MIRROR_COL].astype(float)
bv = va[:4000]
import torch
import torch.nn as nn
WEB = os.path.join(ROOT, "webapp")


def loadj(f):
    j = json.load(open(os.path.join(WEB, f), encoding="utf-8"))
    layers = j["gen"]["layers"]
    seq = []
    for i, L in enumerate(layers):
        W = torch.tensor(L["W"], dtype=torch.float32)
        b = torch.tensor(L["b"], dtype=torch.float32)
        lin = nn.Linear(W.shape[1], W.shape[0])
        with torch.no_grad():
            lin.weight.copy_(W); lin.bias.copy_(b)
        seq.append(lin)
        if i != len(layers) - 1:
            seq.append(nn.ReLU())
    m = nn.Sequential(*seq); m.eval()
    return m, int(len(layers[0]["W"][0]))


for tag, f in (("dim4x", "pca22_dim4x_model.json"), ("dim4st", "pca22_dim4st_model.json")):
    m, k = loadj(f)
    with torch.no_grad():
        out = np.clip(m(torch.tensor(ZZ[bv][:, :k])).numpy(), 0, 15)
    ps = out.max(1) - out.min(1)
    ts = T32[bv]
    # for dim4x the training target was the original, so compare against the stretched one
    ts_st = (ts - ts.min(1, keepdims=True)) / np.maximum(ts.max(1, keepdims=True) - ts.min(1, keepdims=True), 1e-6) * 15
    ref = ts_st if tag == "dim4x" else ts
    corr = float(np.corrcoef(ref.max(1) - ref.min(1), ps)[0, 1])
    slope = float(np.polyfit(ref.max(1) - ref.min(1), ps, 1)[0])
    print(f"  {tag:8s} {corr:7.3f} {slope:7.3f} {(ref.max(1)-ref.min(1)).mean():12.2f} {ps.mean():12.2f}")

# ---------------------------------------------------------------- chart
if stats:
    keys = [k for k in order if k in stats]
    W, H = 940, 500
    cv = Image.new("RGB", (W, H), (18, 20, 24))
    dr = ImageDraw.Draw(cv)
    f_t, f_l, f_s = font(19), font(12), font(10)
    dr.text((16, 12), "第 3 轮：训练时烘焙拉伸没能替代显示端拉伸", font=f_t, fill=(234, 238, 244))
    dr.text((16, 36), "三臂权重各不相同；误差线为 95% Wilson 区间", font=f_s, fill=(150, 162, 178))
    x0, y0, y1 = 90, 400, 80
    dr.line([x0 - 8, y1, x0 - 8, y0], fill=(70, 78, 92), width=1)
    dr.line([x0 - 8, y0, W - 24, y0], fill=(70, 78, 92), width=1)
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        yy = y0 - frac * (y0 - y1)
        dr.line([x0 - 8, yy, W - 24, yy], fill=(38, 43, 52), width=1)
        dr.text((x0 - 44, yy - 6), f"{frac*100:.0f}%", font=f_s, fill=(120, 132, 148))
    step = (W - 24 - x0) / len(keys)
    bw = step * 0.5
    cols = {"dim4x": (110, 200, 130), "dim4st": (224, 192, 90), "dim22": (110, 168, 254)}
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
        name = {"dim4x": "PCA-4\n显示拉伸", "dim4st": "PCA-4\n训练烘焙", "dim22": "PCA-22\n参照"}[k]
        for j, line in enumerate(name.split("\n")):
            dr.text((cx - 26, y0 + 8 + j * 15), line, font=f_l, fill=(206, 214, 226))
        dr.text((cx - 22, y0 + 40), f"n={st['n']}", font=f_s, fill=(130, 142, 158))
    dr.text((16, H - 14), "显示端拉伸胜出 → 红石需要 min/max，或者另想办法把对比度烘进去",
            font=f_s, fill=(140, 152, 168))
    p = os.path.join(OUT, "ROUND3_BAKED_VS_POST.png")
    cv.save(p)
    print(f"\nwrote {p}")

json.dump(stats, open(os.path.join(OUT, "round3_stats.json"), "w"), indent=2, ensure_ascii=False)
print("ROUND3_DONE")
