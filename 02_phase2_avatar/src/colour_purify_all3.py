#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
All three palette models, before and after a gentle filter (keep 50%), on one sheet.

WHY KEEP 50% AND NOT 25%
The earlier run showed filtering at 25% pushed edge sharpness to 19.6 while the real data sits at
15.6 -- the filter rewards sharpness, so a hard filter picks faces harsher than the source art.
At 50% the same measurement was 17.3, close enough to be plausible.

HOW THE NUMBERS ARE MADE HONEST
    - two disjoint scorers per repetition: A ranks and keeps, B judges, fitted on different
      halves of the 260 labels, ten splits averaged
    - the decision threshold is calibrated on the held-out LABELS, so the reported rate is
      comparable to the rate the user actually gave
    - the faces drawn are a RANDOM sample of the survivors, never the top of the ranking, which
      is the extreme tail and makes every filtered row look degenerate

LAYOUT
A 2x3 block grid: rows are the three models, columns are before / after. Each block is a
rectangle of 8 x 3 = 24 faces rather than a single long row, so more faces fit at a readable size.
"""
import json
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
WEB = os.path.join(ROOT, "webapp")
OUT = os.path.join(ROOT, "results", "colour", "round8")

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("layout", os.path.join(HERE, "layout.py"))
layout = _ilu.module_from_spec(_spec); _spec.loader.exec_module(layout)

LUM = np.array([0.299, 0.587, 0.114])
KEEP = 0.50
NB, BW, BH = 24, 8, 3          # faces per block, block columns, block rows


def font(sz):
    for fn in ("msyh.ttc", "simhei.ttf", "simsun.ttc"):
        p = os.path.join("C:\\Windows\\Fonts", fn)
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, sz)
            except Exception:
                pass
    return ImageFont.load_default()


def load_palette(fname):
    j = json.load(open(os.path.join(WEB, fname), encoding="utf-8"))
    pal = np.array(j["palette"]); K = len(pal)
    all = [{"W": np.array(L["W"]), "b": np.array(L["b"]), "relu": True} for L in j["body"]]
    all.append({"W": np.array(j["head"]["W"]), "b": np.array(j["head"]["b"]), "relu": False})

    def gen(Z):
        v = Z
        for L in all:
            v = v @ L["W"].T + L["b"]
            if L["relu"]:
                v = np.maximum(0, v)
        return pal[v.reshape(len(Z), 32, K).argmax(-1)].reshape(len(Z), 96)
    return gen, K


def to_img(flat):
    a = np.clip(flat, 0, 15).reshape(len(flat), 32, 3)
    img = np.zeros((len(a), 8, 8, 3))
    img[:, layout.POS_ROW, layout.POS_COL] = a
    img[:, layout.POS_ROW, layout.MIRROR_COL] = a
    return img


def feats(flat):
    flat = np.asarray(flat, dtype=np.float64)
    img = to_img(flat)
    a = np.clip(flat, 0, 15).reshape(len(flat), 32, 3)
    x = img.reshape(len(img), 64, 3)
    mx = x.max(-1); mn = x.min(-1)
    sat = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    d = np.linalg.norm(a[:, :, None, :] - a[:, None, :, :], axis=-1)
    iu = np.triu_indices(32, 1)
    sep = d[:, iu[0], iu[1]].mean(1)
    dh = np.linalg.norm(img[:, :, 1:] - img[:, :, :-1], axis=-1)
    dv = np.linalg.norm(img[:, 1:] - img[:, :-1], axis=-1)
    ed = np.concatenate([dh.reshape(len(img), -1), dv.reshape(len(img), -1)], 1)
    ncol = np.array([len({tuple(np.rint(c).astype(int)) for c in a[i]}) for i in range(len(a))])
    o = np.argsort(sat, 1)
    q = np.array([[sat[i, o[i, 0]], sat[i, o[i, -1]]] for i in range(len(a))])
    return np.c_[sat.mean(1), sep, np.percentile(ed, 90, axis=1), ed.mean(1), ncol, q[:, 0], q[:, 1]]


def fit(F, Y, lam=3.0):
    mu = F.mean(0); sd = F.std(0) + 1e-9
    A = np.c_[(F - mu) / sd, np.ones(len(F))]
    w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ Y)
    return (mu, sd, w)


def apply(m, F):
    mu, sd, w = m
    return np.c_[(F - mu) / sd, np.ones(len(F))] @ w


# ---- labels
_recs = [json.loads(l) for l in open(os.path.join(ROOT, "results", "preference", "labels.jsonl"),
                                     encoding="utf-8") if l.strip()]
FL, YL = [], []
for r in _recs:
    if str(r.get("round") or "") != "8":
        continue
    pl = r.get("predicted_levels")
    if not pl or len(pl) != 96:
        continue
    FL.append(feats([pl])[0]); YL.append(1.0 if r["grade"] >= 2 else 0.0)
FL = np.array(FL); YL = np.array(YL)
print(f"colour labels: {len(FL)}")

N = 8000
rng = np.random.default_rng(11)
Z = np.clip(rng.standard_normal((N, 8)), -3.5, 3.5).astype(np.float32)

print()
print("=" * 100)
print(f"ALL THREE MODELS, BEFORE / AFTER  (keep {int(KEEP*100)}%, judged out of sample)")
print("=" * 100)
print(f"  {'model':7s} {'K':>3s} {'原始':>8s} {'筛选后':>8s} {'提升':>8s} "
      f"{'边缘前':>7s} {'边缘后':>7s} {'分离前':>7s} {'分离后':>7s} {'中前':>7s}")
print("  " + "-" * 84)
results = {}
for arm, fname, K in (("pal8", "colour_pal8_model.json", 8),
                      ("pal16", "colour_pal16_model.json", 16),
                      ("pal32", "colour_pal32_model.json", 32)):
    gen, _ = load_palette(fname)
    faces = gen(Z)
    F = feats(faces)
    reps = {1.0: [], KEEP: []}
    for sp in range(10):
        perm = np.random.default_rng(300 + sp).permutation(len(FL))
        A_, B_ = perm[:len(perm)//2], perm[len(perm)//2:]
        if len(np.unique(YL[A_])) < 2 or len(np.unique(YL[B_])) < 2:
            continue
        mA = fit(FL[A_], YL[A_]); mB = fit(FL[B_], YL[B_])
        sA = apply(mA, F)
        rate_b = float(YL[B_].mean())
        t = float(np.quantile(apply(mB, FL[B_]), 1.0 - rate_b))
        sB = apply(mB, F)
        reps[1.0].append(float((sB > t).mean()))
        sel = np.argsort(-sA)[:int(N * KEEP)]
        reps[KEEP].append(float((sB[sel] > t).mean()))
    r0 = float(np.mean(reps[1.0])); r1 = float(np.mean(reps[KEEP]))
    sel = np.argsort(-apply(fit(FL, YL), F))[:int(N * KEEP)]
    results[arm] = {"K": K, "raw": r0, "filtered": r1,
                    "edge_raw": float(F[:, 2].mean()), "edge_f": float(F[sel][:, 2].mean()),
                    "sep_raw": float(F[:, 1].mean()), "sep_f": float(F[sel][:, 1].mean()),
                    "show_raw": faces[np.random.default_rng(7).choice(N, NB, replace=False)],
                    "show_f": faces[sel[np.random.default_rng(5).choice(len(sel), NB, replace=False)]]}
    print(f"  {arm:7s} {K:3d} {100*r0:7.1f}% {100*r1:7.1f}% {100*(r1-r0):+7.1f} "
          f"{F[:,2].mean():7.2f} {F[sel][:,2].mean():7.2f} "
          f"{F[:,1].mean():7.2f} {F[sel][:,1].mean():7.2f} "
          f"{100*(1-r0):7.1f}%")
print()
print("  真实数据参照：边缘p90 15.56，分离度 6.67")
print("  '中前' 是原始行的预期不可接受比例，用来判断哪个模型本来的问题更严重")

# ---- figure: 3 rows (models) x 2 columns (before/after), each block 8x3 faces
TILE, PAD = 46, 4
BLOCKW = BW * (TILE + PAD)
BLOCKH = BH * (TILE + PAD)
LEFT, TOP = 210, 104
COLGAP, ROWGAP = 40, 34
W = LEFT + 2 * BLOCKW + COLGAP + 20
H = TOP + 3 * (BLOCKH + 30 + ROWGAP) + 26
cv = Image.new("RGB", (W, H), (18, 20, 24))
dr = ImageDraw.Draw(cv)
f_t, f_l, f_s = font(21), font(13), font(10)
dr.text((12, 12), "彩色三个模型：筛选前 / 筛选后（保留 50%）", font=f_t, fill=(234, 238, 244))
dr.text((12, 40), "每块 8×3 = 24 张随机脸。筛选＝抽 2 倍种子、按颜色特征打分、留一半。",
        font=f_s, fill=(150, 162, 178))
dr.text((12, 56), "判定用独立打分器（标签不相交、10 次划分平均），图里是存活者的随机样本。",
        font=f_s, fill=(130, 142, 158))
dr.text((12, 72), "真实数据：边缘 15.6，分离度 6.7。", font=f_s, fill=(110, 200, 130))

for row, arm in enumerate(("pal8", "pal16", "pal32")):
    r = results[arm]
    y = TOP + row * (BLOCKH + 30 + ROWGAP)
    dr.text((12, y + 8), f"{arm}（K={r['K']}）", font=f_l, fill=(110, 168, 254))
    dr.text((12, y + 28), f"原始 {100*r['raw']:.0f}%", font=f_s, fill=(180, 190, 205))
    dr.text((12, y + 42), f"筛选后 {100*r['filtered']:.0f}%", font=f_s, fill=(110, 200, 130))
    dr.text((12, y + 56), f"提升 {100*(r['filtered']-r['raw']):+.0f} 点", font=f_s,
            fill=(224, 192, 90))
    for col, (key, title) in enumerate((("show_raw", "筛选前"), ("show_f", "筛选后"))):
        x0 = LEFT + col * (BLOCKW + COLGAP)
        dr.text((x0, y - 15), title, font=f_s,
                fill=(150, 162, 178) if col == 0 else (110, 200, 130))
        imgs = to_img(r[key])
        for i in range(NB):
            rr, cc = divmod(i, BW)
            t = np.kron(np.clip(imgs[i] / 15.0 * 255, 0, 255).astype(np.uint8),
                        np.ones((TILE // 8, TILE // 8, 1))).astype(np.uint8)
            cv.paste(Image.fromarray(t), (x0 + cc * (TILE + PAD), y + rr * (TILE + PAD)))
dr.text((12, H - 20), "pal8 的「中」最多：8 种颜色不够用。pal32 原始就更好，筛选后提升也仍然明显。",
        font=f_s, fill=(140, 152, 168))
p = os.path.join(OUT, "ALL3_PURIFY.png")
cv.save(p)
print(f"\nwrote {p}  {cv.size}")
json.dump({k: {kk: vv for kk, vv in v.items() if not kk.startswith("show")}
           for k, v in results.items()},
          open(os.path.join(OUT, "all3_purify.json"), "w", encoding="utf-8"),
          indent=2, ensure_ascii=False)
print(f"wrote {os.path.join(OUT, 'all3_purify.json')}")
print("ALL3_DONE")
