#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
pal32 before and after purification.

"Purification" here means the same thing it meant for grayscale: draw many seeds, score the face
each would produce, and keep the best fraction. The scorer is the one fitted on the 260 round-8
labels (7 colour features, cross-validated AUC 0.632). It is weaker than the grayscale scorer
(0.82 on w64), so the gain is expected to be smaller -- and that is worth seeing rather than
hiding.

Three rows, identical drawing and size:
    raw           every draw kept
    keep 50%      twice as many drawn, half kept
    keep 25%      four times as many drawn, a quarter kept

Reported: the scorer's predicted acceptable share for each row, plus the measured feature values,
so the rows can be compared on the dimensions the user actually judges.
"""
import json
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WEB = os.path.join(ROOT, "webapp")
OUT = os.path.join(ROOT, "results", "colour", "round8")
torch.set_num_threads(8)

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("layout", os.path.join(HERE, "layout.py"))
layout = _ilu.module_from_spec(_spec); _spec.loader.exec_module(layout)


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
    pal = np.array(j["palette"])
    K = len(pal)
    all = [{"W": np.array(L["W"]), "b": np.array(L["b"]), "relu": True} for L in j["body"]]
    all.append({"W": np.array(j["head"]["W"]), "b": np.array(j["head"]["b"]), "relu": False})

    def gen(Z):
        v = Z
        for L in all:
            v = v @ L["W"].T + L["b"]
            if L["relu"]:
                v = np.maximum(0, v)
        idx = v.reshape(len(Z), 32, K).argmax(-1)
        return pal[idx].reshape(len(Z), 96)      # (n,96) RGB in 0..15
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
    order = np.argsort(sat, 1)
    q = np.array([[sat[i, order[i, 0]], sat[i, order[i, -1]]] for i in range(len(a))])
    return np.c_[sat.mean(1), sep, np.percentile(ed, 90, axis=1), ed.mean(1),
                 ncol, q[:, 0], q[:, 1]]


sc = json.load(open(os.path.join(OUT, "colour_scorer.json"), encoding="utf-8"))


def fit_scorer(F, Y, lam=3.0):
    mu = F.mean(0); sd = F.std(0) + 1e-9
    A = np.c_[(F - mu) / sd, np.ones(len(F))]
    w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ Y)
    return mu, sd, w


def apply_scorer(m, F):
    mu, sd, w = m
    return np.c_[(F - mu) / sd, np.ones(len(F))] @ w


# the labels, for fitting the two disjoint scorers
FL, YL = [], []
_recs = [json.loads(l) for l in open(
    os.path.join(ROOT, "results", "preference", "labels.jsonl"), encoding="utf-8") if l.strip()]
for r in _recs:
    if str(r.get("round") or "") != "8":
        continue
    pl = r.get("predicted_levels")
    if not pl or len(pl) != 96:
        continue
    FL.append(feats([pl])[0])
    YL.append(1.0 if r["grade"] >= 2 else 0.0)
FL = np.array(FL); YL = np.array(YL)

gen, K = load_palette("colour_pal32_model.json")
N = 8000
rng = np.random.default_rng(11)
Z = np.clip(rng.standard_normal((N, 8)), -3.5, 3.5).astype(np.float32)
faces = gen(Z)
F = feats(faces)

# ten splits: A filters, B judges, and B is fitted on labels A never saw
SPLITS = 10
judged = {1.0: [], 0.50: [], 0.25: []}
for sp in range(SPLITS):
    perm = np.random.default_rng(200 + sp).permutation(len(FL))
    A_idx, B_idx = perm[:len(perm)//2], perm[len(perm)//2:]
    if len(np.unique(YL[A_idx])) < 2 or len(np.unique(YL[B_idx])) < 2:
        continue
    mA = fit_scorer(FL[A_idx], YL[A_idx])
    mB = fit_scorer(FL[B_idx], YL[B_idx])
    sA = apply_scorer(mA, F)
    sB = apply_scorer(mB, F)
    # CALIBRATION: the raw score has no absolute meaning, so the decision threshold is taken
    # from the held-out LABELS -- choose t so that the share of labelled faces above it equals
    # the share the user actually called acceptable. Applying 0 instead made almost every
    # generated face acceptable, which is why the previous numbers were 98%.
    rate_b = float(YL[B_idx].mean())
    sB_lab = apply_scorer(mB, FL[B_idx])
    t = float(np.quantile(sB_lab, 1.0 - rate_b))
    for keep in (1.0, 0.50, 0.25):
        sel = np.arange(N) if keep >= 1.0 else np.argsort(-sA)[:int(N * keep)]
        judged[keep].append(float((sB[sel] > t).mean()))
# for reporting, use one weighted scorer for the feature table only
sc = fit_scorer(FL, YL)
s = apply_scorer(sc, F)
print("=" * 100)
print(f"pal32 BEFORE / AFTER  ({N:,} faces scored, K={K})")
print("=" * 100)
print(f"  scorer fitted on {len(FL)} labels; the filtering gain below is measured out of sample")
print()
print(f"  {'row':12s} {'预期可接受':>10s} {'饱和':>7s} {'分离度':>7s} {'边缘p90':>8s} "
      f"{'颜色数':>7s}")
print("  " + "-" * 58)
rows = []
for name, keep in (("原始（不筛）", 1.0), ("筛选 保留50%", 0.50), ("筛选 保留25%", 0.25)):
    sel = np.arange(N) if keep >= 1.0 else np.argsort(-s)[:int(N * keep)]
    # judged with the HELD-OUT scorer, averaged over splits
    rate = float(np.mean(judged[keep])) if judged[keep] else float("nan")
    ff = F[sel].mean(0)
    # show a RANDOM sample of the survivors, not the top of the ranking: the top is the extreme
    # tail (the highest-scoring faces are the ones the scorer's own quirks push to the limit) and
    # makes every filtered row look degenerate. The mean statistics above tell the real story.
    show = sel[np.random.default_rng(5).choice(len(sel), min(32, len(sel)), replace=False)]
    rows.append((name, faces[show], rate, ff))
    print(f"  {name:12s} {100*rate:9.1f}% {ff[0]:7.3f} {ff[1]:7.2f} {ff[2]:8.2f} {ff[4]:7.1f}")
print()
print("  '预期可接受' 用一个独立的打分器判定（筛选用 A，判定用 B，两者标签不相交，10 次划分平均），")
print("  所以不再是循环论证。但它仍然是模型判断，不是你的真实评分；打分器 AUC 0.63，"
      "只能当方向看。")

# ---- sheet
COLS, TILE, PAD = 32, 50, 5
W = 250 + COLS * (TILE + PAD) + PAD
H = 100 + len(rows) * (TILE + 30) + 26
cv = Image.new("RGB", (W, H), (18, 20, 24))
dr = ImageDraw.Draw(cv)
f_t, f_l, f_s = font(20), font(12), font(10)
dr.text((12, 12), "彩色最好模型 pal32：纯化前后", font=f_t, fill=(234, 238, 244))
dr.text((12, 38), f"调色板 K={K}，每像素 5 bit。每行 32 张随机脸，"
        f"筛选是「抽多个种子、打分、留最好的」。", font=f_s, fill=(150, 162, 178))
dr.text((12, 54), "打分器只用颜色特征，AUC 0.63，所以提升幅度有限——这是真实情况。",
        font=f_s, fill=(130, 142, 158))
y = 78
for name, faces_r, rate, ff in rows:
    dr.text((12, y + 4), name, font=f_l, fill=(110, 168, 254))
    dr.text((12, y + 20), f"预期 {100*rate:.0f}%", font=f_s, fill=(110, 200, 130))
    dr.text((12, y + 32), f"边缘 {ff[2]:.1f}", font=f_s, fill=(150, 162, 178))
    imgs = to_img(faces_r[:COLS])
    for i in range(COLS):
        t = np.kron(np.clip(imgs[i] / 15.0 * 255, 0, 255).astype(np.uint8),
                    np.ones((TILE // 8, TILE // 8, 1))).astype(np.uint8)
        cv.paste(Image.fromarray(t), (250 + i * (TILE + PAD), y))
    y += TILE + 30
dr.text((12, H - 20), "三行用的是同一批随机数生成器，只是筛选强度不同。", font=f_s,
        fill=(140, 152, 168))
p = os.path.join(OUT, "PAL32_PURIFY.png")
cv.save(p)
print(f"\nwrote {p}  {cv.size}")
json.dump([{"row": n, "pred": r, "feats": f.tolist()} for n, _, r, f in rows],
          open(os.path.join(OUT, "purify.json"), "w", encoding="utf-8"),
          indent=2, ensure_ascii=False)
print(f"wrote {os.path.join(OUT, 'purify.json')}")
print("PURIFY_DONE")
