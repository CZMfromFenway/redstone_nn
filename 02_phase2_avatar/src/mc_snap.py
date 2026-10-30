#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
Snap the palettes onto real Minecraft colours, then retrain and measure the cost.

WHY
Every entry of every palette is at least one level away from the nearest Minecraft map colour,
so a picture generated from them cannot be built. Snapping each entry to its nearest available
Minecraft colour makes the palette exactly buildable -- the question is what that costs in
image quality.

TWO CONSTRAINT LEVELS
    normal   61 colours, the same brightness for every base. Strict: one block per colour.
    all      all four shade modifiers, 244 colours. Not really "shading" here, just more colours,
             and it includes pure white via SNOW/HIGH.

MEASURED
    RGB MAE against the real data, the user's colour criteria, and the predicted acceptable rate
    from a scorer fitted on the round-8 labels and judged out of sample.
"""
import argparse
import json
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "results", "colour", "mcfit")
WEB = os.path.join(ROOT, "webapp")
os.makedirs(OUT, exist_ok=True)

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("layout", os.path.join(HERE, "layout.py"))
layout = _ilu.module_from_spec(_spec); _spec.loader.exec_module(layout)
torch.set_num_threads(8)

A = argparse.ArgumentParser()
A.add_argument("--steps", type=int, default=12000)
A.add_argument("--tag", default="")
A = A.parse_args()

mc = json.load(open(os.path.join(ROOT, "data", "mc_map_colors.json"), encoding="utf-8"))
MOD = mc["_modifiers"]
BASE = np.array([b["rgb"] for b in mc["bases"]], dtype=np.float64)
NAMES = [b["name"] for b in mc["bases"]]
SETS = {
    "normal": np.clip(BASE * MOD["NORMAL"], 0, 255),
    "all": np.vstack([np.clip(BASE * f, 0, 255)
                      for f in (MOD["LOW"], MOD["NORMAL"], MOD["HIGH"], MOD["LOWEST"])]),
}
SHADE = [f"{n}/{s}" for s in ("NORMAL", "LOW", "HIGH", "LOWEST") for n in NAMES]

z = np.load(os.path.join(ROOT, "results", "colour", "cache.npz"))
rgb96 = z["rgb96"].astype(np.float64)
cpc = z["cpc"]
N = len(rgb96)
sp = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_split.npz"))
tr, va = sp["train"], sp["val"]

# ---- labels and a colour scorer, for the predicted rate
_recs = [json.loads(l) for l in open(os.path.join(ROOT, "results", "preference", "labels.jsonl"),
                                     encoding="utf-8") if l.strip()]


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


FL, YL, ARM = [], [], []
for r in _recs:
    if str(r.get("round") or "") != "8":
        continue
    pl = r.get("predicted_levels")
    if not pl or len(pl) != 96:
        continue
    FL.append(feats([pl])[0]); YL.append(1.0 if r["grade"] >= 2 else 0.0)
    ARM.append(r.get("model"))
FL = np.array(FL); YL = np.array(YL)
_tab = {a: (np.array([f for f, m in zip(FL, ARM) if m == a]),
            np.array([y for y, m in zip(YL, ARM) if m == a])) for a in set(ARM)}


def fit(X, Y, lam=3.0):
    mu = X.mean(0); sd = X.std(0) + 1e-9
    A_ = np.c_[(X - mu) / sd, np.ones(len(X))]
    w = np.linalg.solve(A_.T @ A_ + lam * np.eye(A_.shape[1]), A_.T @ Y)
    return (mu, sd, w)


def apply(m, X):
    mu, sd, w = m
    return np.c_[(X - mu) / sd, np.ones(len(X))] @ w


def load_palette(fn):
    j = json.load(open(os.path.join(WEB, fn), encoding="utf-8"))
    return np.array(j["palette"], dtype=np.float64)


PALS = {"K=8": load_palette("colour_pal8_model.json"),
        "K=16": load_palette("colour_pal16_model.json"),
        "K=32": load_palette("colour_pal32_model.json")}

print("=" * 104)
print("MC-CONSTRAINED PALETTES: snap every entry to a real Minecraft colour, retrain, measure")
print("=" * 104)

results = {}
for setname, SET in SETS.items():
    snames = NAMES if setname == "normal" else SHADE
    print()
    print(f"  约束集：{setname}（{len(SET)} 色）")
    print(f"    {'调色板':6s} {'原平均误差':>10s} {'吸附后':>8s} {'RGB MAE':>9s} "
          f"{'边缘p90':>8s} {'颜色数':>7s} {'预测可接受':>10s}")
    print("    " + "-" * 70)
    for tag, P in PALS.items():
        P255 = P * 17.0
        d = np.linalg.norm(P255[:, None, :] - SET[None, :, :], axis=-1)
        err0 = float(d.min(1).mean() / 17.0)
        # snap
        snap = SET[d.argmin(1)] / 17.0
        K = len(snap)
        # rebuild index targets with the snapped palette
        pos = rgb96.reshape(N, 32, 3)
        dd = ((pos[:, :, None, :] - snap[None, None, :, :]) ** 2).sum(-1)
        Y = torch.tensor(dd.argmin(-1).reshape(N, 32), dtype=torch.long)
        CT = torch.tensor(cpc[:, :8])
        hid = 128
        torch.manual_seed(0)
        net = nn.Sequential(nn.Linear(8, hid), nn.ReLU(), nn.Linear(hid, hid), nn.ReLU(),
                            nn.Linear(hid, 32 * K))
        opt = torch.optim.Adam(net.parameters(), lr=3e-4)
        tr_t = torch.tensor(tr, dtype=torch.long)
        rng = np.random.default_rng(0)
        for step in range(A.steps):
            bi = tr_t[torch.from_numpy(rng.integers(0, len(tr_t), 256))]
            lg = net(CT[bi]).view(-1, 32, K)
            loss = F.cross_entropy(lg.reshape(-1, K), Y[bi].reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            idx = net(CT).view(-1, 32, K).argmax(-1).numpy()
        faces = snap[idx].reshape(N, 96)
        m = feats(faces)
        # predicted acceptable: judge with a scorer fitted on a DIFFERENT half of the labels
        rates = []
        for s in range(10):
            perm = np.random.default_rng(400 + s).permutation(len(FL))
            a_, b_ = perm[:len(perm)//2], perm[len(perm)//2:]
            if len(np.unique(YL[a_])) < 2 or len(np.unique(YL[b_])) < 2:
                continue
            mA = fit(FL[a_], YL[a_]); mB = fit(FL[b_], YL[b_])
            t = float(np.quantile(apply(mB, FL[b_]), 1.0 - float(YL[b_].mean())))
            rates.append(float((apply(mB, m) > t).mean()))
        rate = float(np.mean(rates))
        results[f"{setname}|{tag}"] = {"snap": snap.tolist(), "err0": err0, "rate": rate,
                                       "edge": float(m[:, 2].mean()),
                                       "ncol": float(m[:, 4].mean()),
                                       "names": [snames[i] for i in d.argmin(1)]}
        print(f"    {tag:6s} {err0:10.3f} {0.0:8.3f} "
              f"{np.abs(faces - rgb96).mean():9.3f} {m[:,2].mean():8.2f} "
              f"{m[:,4].mean():7.1f} {100*rate:9.1f}%")
    print(f"    真实数据参照  边缘 15.56  颜色数 7.2")

json.dump({k: {kk: vv for kk, vv in v.items()} for k, v in results.items()},
          open(os.path.join(OUT, "snapped.json"), "w", encoding="utf-8"),
          indent=2, ensure_ascii=False)
print()
print("=" * 104)
print("每个调色板吸附后用到哪些 MC 颜色")
print("=" * 104)
for setname in SETS:
    for tag in PALS:
        key = f"{setname}|{tag}"
        if key not in results:
            continue
        print(f"  {setname:7s} {tag:6s} → " + ", ".join(results[key]["names"]))
print(f"\nwrote {os.path.join(OUT, 'snapped.json')}")
print("SNAP_DONE")
