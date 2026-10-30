#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
What separates the user's BEST colour faces from merely acceptable and from 中/差?

THE CORRECTION
The user clarified that 7:2:1 is the AREA RATIO OF THE THREE MAIN COLOURS in a face, not the
share of pixels in saturation bands. That is a different measurement, so it is redone here:
cluster the 32 positions into three colour groups, sort the groups by area, and compare the
resulting (a1, a2, a3) with (0.7, 0.2, 0.1).

They also described the intent of that ratio: 铺垫 / 主体 / 点睛 -- backing, body, accent. That
predicts an ORDERING as well as a ratio: the largest area should be the least saturated, and
the smallest area the most saturated. That ordering is measured too, by the rank correlation
between a group's area and its saturation.

THREE GROUPS
    best      the 16 faces the user named as 优中之优
    good      acceptable but not named (67 - 16 = 51)
    bad       中 + 差 (137)
"""
import json
import math
import os
import sys
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "results", "colour")
LABELS = os.path.join(ROOT, "results", "preference", "labels.jsonl")

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("layout", os.path.join(HERE, "layout.py"))
layout = _ilu.module_from_spec(_spec); _spec.loader.exec_module(layout)

BEST = [58, 6, 204, 105, 38, 41, 157, 166, 29, 55, 48, 189, 25, 158, 56, 28]

recs = [json.loads(l) for l in open(LABELS, encoding="utf-8") if l.strip()]
r6 = [r for r in recs if str(r.get("round") or "") == "6"
      and len(r.get("predicted_levels") or []) == 96]
for i, r in enumerate(r6):
    r["_idx"] = i + 1
print(f"colour labels: {len(r6)}   named best: {len(BEST)}")


def to_img(r):
    v = np.clip(np.array(r["predicted_levels"], dtype=np.float64), 0, 15).reshape(32, 3)
    img = np.zeros((8, 8, 3), dtype=np.float64)
    img[layout.POS_ROW, layout.POS_COL] = v
    img[layout.POS_ROW, layout.MIRROR_COL] = v
    return img


def sat_of(x):
    mx = x.max(-1); mn = x.min(-1)
    return np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)


def analyse(img):
    x = img.reshape(-1, 3)                       # 64 pixels (mirrored), each 3 channels
    s = sat_of(x)
    # three colour groups, initialised from the saturation order so the result is deterministic
    o = np.argsort(s)
    C = x[[o[0], o[len(o) // 2], o[-1]]].copy()
    lab = np.zeros(len(x), dtype=int)
    for _ in range(20):
        d = ((x[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        lab = d.argmin(1)
        for j in range(3):
            m = lab == j
            if m.any():
                C[j] = x[m].mean(0)
    area = np.array([(lab == j).mean() for j in range(3)])
    csat = np.array([sat_of(C[j]) for j in range(3)])
    clum = np.array([C[j].mean() / 15.0 for j in range(3)])
    o2 = np.argsort(-area)                       # largest first
    a = area[o2]; sa = csat[o2]; lu = clum[o2]
    # the 7:2:1 claim, on the three areas
    d721 = float(np.abs(a - np.array([0.7, 0.2, 0.1])).sum())
    # 铺垫/主体/点睛: saturation should RISE as area falls -> rank correlation of a vs sa is -1
    ra = np.argsort(np.argsort(-a)); rs = np.argsort(np.argsort(-sa))
    order_ok = float(1 - np.abs(ra - rs).sum() / 6.0)      # 1 = perfect accent ordering
    # how distinct are the three colours from each other?
    sep = float(np.mean([np.linalg.norm(C[i] - C[j]) for i in range(3)
                         for j in range(i + 1, 3)]))
    # edges
    dh = np.linalg.norm(img[:, 1:] - img[:, :-1], axis=-1)
    dv = np.linalg.norm(img[1:] - img[:-1], axis=-1)
    ed = np.concatenate([dh.ravel(), dv.ravel()])
    ncol = len({tuple(np.rint(c).astype(int)) for c in x})
    lum = x @ np.array([0.299, 0.587, 0.114])
    return {
        "a1": float(a[0]), "a2": float(a[1]), "a3": float(a[2]),
        "d721": d721,
        "sat1": float(sa[0]), "sat2": float(sa[1]), "sat3": float(sa[2]),
        "order_ok": order_ok,
        "sep": sep,
        "sat_mean": float(s.mean()),
        "sat_lo": float((s < 0.15).mean()),
        "sat_hi": float((s > 0.45).mean()),
        "edge_mean": float(ed.mean()),
        "edge_p90": float(np.percentile(ed, 90)),
        "n_colours": ncol,
        "lum_mean": float(lum.mean() / 15.0),
        "lum_sd": float(lum.std() / 15.0),
    }


rows, grp = [], []
for r in r6:
    rows.append(analyse(to_img(r)))
    if r["_idx"] in BEST:
        grp.append("best")
    elif r["grade"] >= 2:
        grp.append("good")
    else:
        grp.append("bad")
grp = np.array(grp)
KEYS = list(rows[0].keys())
V = {k: np.array([r[k] for r in rows]) for k in KEYS}

print()
print("=" * 104)
print("THREE-WAY COMPARISON: 优中之优 / 可接受 / 中差")
print("=" * 104)
print(f"  best {int((grp=='best').sum())}   good {int((grp=='good').sum())}   "
      f"bad {int((grp=='bad').sum())}")
print()
print(f"  {'feature':12s} {'best':>9s} {'good':>9s} {'bad':>9s} {'best vs bad':>13s}")
print("  " + "-" * 58)
order = ["a1", "a2", "a3", "d721", "sat1", "sat2", "sat3", "order_ok", "sep",
         "sat_mean", "sat_lo", "sat_hi", "edge_mean", "edge_p90", "n_colours",
         "lum_mean", "lum_sd"]
for k in order:
    b = V[k][grp == "best"].mean()
    g = V[k][grp == "good"].mean()
    d = V[k][grp == "bad"].mean()
    tag = ""
    if b > g > d:
        tag = "  best>good>bad"
    elif b < g < d:
        tag = "  best<good<bad"
    print(f"  {k:12s} {b:9.3f} {g:9.3f} {d:9.3f}{tag}")

print()
print("=" * 104)
print("THE 7:2:1 CLAIM, AS AREA RATIO OF THE THREE MAIN COLOURS")
print("=" * 104)
print(f"  {'group':10s} {'最大面积':>9s} {'其次':>8s} {'最小':>8s} {'离 7:2:1':>10s}")
print("  " + "-" * 50)
for g_ in ("best", "good", "bad"):
    m = grp == g_
    print(f"  {g_:10s} {V['a1'][m].mean():9.3f} {V['a2'][m].mean():8.3f} "
          f"{V['a3'][m].mean():8.3f} {V['d721'][m].mean():10.3f}")
print(f"  {'7:2:1':10s} {0.700:9.3f} {0.200:8.3f} {0.100:8.3f} {0.0:10.3f}")


def auc(s, t):
    hi = s[t == 1][:, None]; lo = s[t == 0][None, :]
    return float((hi > lo).mean() + 0.5 * (hi == lo).mean())


print()
print("=" * 104)
print("WHICH FEATURE SEPARATES BEST FROM THE REST?  (AUC, best=1 vs everything else=0)")
print("=" * 104)
isbest = (grp == "best").astype(float)
scored = []
for k in KEYS:
    a = auc(V[k], isbest)
    scored.append((abs(a - 0.5), k, a))
scored.sort(reverse=True)
for _, k, a in scored[:10]:
    d = "越大越好" if a > 0.5 else "越小越好"
    print(f"  {k:12s} AUC {a:6.3f}   {d}   best {V[k][grp=='best'].mean():.3f} vs "
          f"bad {V[k][grp=='bad'].mean():.3f}")

json.dump({"groups": {g_: int((grp == g_).sum()) for g_ in ("best", "good", "bad")},
           "means": {k: {g_: float(V[k][grp == g_].mean()) for g_ in ("best", "good", "bad")}
                     for k in KEYS}},
          open(os.path.join(OUT, "three_way.json"), "w", encoding="utf-8"),
          indent=2, ensure_ascii=False)
print(f"\nwrote {os.path.join(OUT, 'three_way.json')}")
print("THREE_WAY_DONE")
