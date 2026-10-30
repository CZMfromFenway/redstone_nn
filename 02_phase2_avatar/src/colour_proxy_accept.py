#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
A proxy acceptability score for generated COLOUR faces, using the existing grayscale scorer.

THE PROBLEM
Every yield number in this project (67.3%, 92.1%, 94.7%) was measured on GRAYSCALE faces,
labelled by the user. Generated colour faces have no labels at all, so there is no direct
number. But there is an indirect one worth computing:

    1. generate colour faces
    2. take their luminance (the same 0.299/0.587/0.114 weights)
    3. project that luminance onto the grayscale PCA basis to get the 4 knobs
    4. score those knobs with the 4->16->1 network trained on the style labels

That answers "is the STRUCTURE of a colour face as good as the structure of a grayscale face",
which is the part the existing model can speak to. It says nothing about whether the colours
themselves look right -- only the user can judge that.

THE CONTROL
The same procedure applied to the grayscale generator should reproduce the known ~67%
acceptable rate. If it does not, the projection is wrong and the colour number means nothing.
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
OUT = os.path.join(ROOT, "results", "colour")
WEB = os.path.join(ROOT, "webapp")
LABELS = os.path.join(ROOT, "results", "preference", "labels.jsonl")
torch.set_num_threads(8)

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("layout", os.path.join(HERE, "layout.py"))
layout = _ilu.module_from_spec(_spec); _spec.loader.exec_module(layout)
LUM = np.array([0.299, 0.587, 0.114])

# ---------------------------------------------------------------- rebuild the gray PCA basis
full = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_full_idx.npy")).astype(np.int64)
N = len(full)
Xg = full.reshape(N, -1).astype(np.float64)
mu_g = Xg.mean(0, keepdims=True)
sub = np.random.default_rng(0).choice(N, 60000, replace=False)
_, _, Vt = np.linalg.svd(Xg[sub] - mu_g, full_matrices=False)
comps = Vt[:22]
c = (Xg - mu_g) @ comps.T
sd_g = (c.std(0, keepdims=True) + 1e-6)

# ---------------------------------------------------------------- the style scorer
recs = [json.loads(l) for l in open(LABELS, encoding="utf-8") if l.strip()]
lab = [r for r in recs if str(r.get("round") or "") == "5" and r.get("model") == "dim4x"
       and r.get("grade") in (0, 1, 2, 3, 4)]
Xl = np.array([r["pca"][:4] for r in lab], dtype=np.float32)
g = np.array([r["grade"] for r in lab])
mu_l = Xl.mean(0); sd_l = Xl.std(0) + 1e-9


def train_scorer(target, seed=0, steps=900):
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 1))
    opt = torch.optim.Adam(net.parameters(), lr=0.02, weight_decay=1e-3)
    t = torch.tensor((Xl - mu_l) / sd_l)
    v = torch.tensor(target, dtype=torch.float32)[:, None]
    for _ in range(steps):
        opt.zero_grad()
        nn.functional.binary_cross_entropy_with_logits(net(t), v).backward()
        opt.step()
    net.eval()
    return net


sc_acc = train_scorer((g >= 2).astype(float), seed=1)


def knobs_from_lum(lum64):
    """(n,64) 0..15 luminance -> standardised first 4 PCA coordinates."""
    x = lum64.astype(np.float64)
    cc = (x - mu_g) @ comps.T
    return (cc / sd_g[0])[:, :4].astype(np.float32)


def score(knobs):
    with torch.no_grad():
        s = sc_acc(torch.tensor((knobs - mu_l) / sd_l)).numpy().ravel()
    return s


# ---------------------------------------------------------------- CONTROL: grayscale pipeline
gj = json.load(open(os.path.join(WEB, "pca22_dim4x_model.json"), encoding="utf-8"))
seq = []
Lj = gj["gen"]["layers"]
for i, L in enumerate(Lj):
    W = torch.tensor(L["W"], dtype=torch.float32)
    b = torch.tensor(L["b"], dtype=torch.float32)
    lin = nn.Linear(W.shape[1], W.shape[0])
    with torch.no_grad():
        lin.weight.copy_(W); lin.bias.copy_(b)
    seq.append(lin)
    if i != len(Lj) - 1:
        seq.append(nn.ReLU())
G1 = nn.Sequential(*seq).eval()


def gray_faces(n, seed):
    rng = np.random.default_rng(seed)
    Z = np.clip(rng.standard_normal((n, 4)), -3.5, 3.5)
    with torch.no_grad():
        v = torch.clamp(G1(torch.tensor(Z, dtype=torch.float32)), 0, 15)
        lo = v.min(1, keepdim=True)[0]; hi = v.max(1, keepdim=True)[0]
        v = torch.clamp((v - lo) / (hi - lo).clamp_min(1e-6) * 15, 0, 15)
        v = torch.round(v).numpy()
    img = layout.image_from_left_half(np.rint(v).astype(int))
    return img.reshape(n, 64).astype(np.float32)


# ---------------------------------------------------------------- colour pipeline
z = np.load(os.path.join(OUT, "cache.npz"))
cpc = z["cpc"]


def colour_model(tag):
    m = torch.load(os.path.join(OUT, f"stage2_{tag}.pt"), weights_only=False)
    net = nn.Sequential(nn.Linear(m["d_in"], m["hid"]), nn.ReLU(),
                        nn.Linear(m["hid"], m["hid"]), nn.ReLU(), nn.Linear(m["hid"], 96))
    net.load_state_dict(m["state"]); net.eval()
    return net, m["k"]


def colour_faces(net, k, n, seed):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, k)).astype(np.float32)
    with torch.no_grad():
        rgbv = np.clip(net(torch.tensor(Z)).numpy() * 15.0, 0, 15)
    a = rgbv.reshape(n, 32, 3)
    img = np.zeros((n, 8, 8, 3), dtype=np.float32)
    img[:, layout.POS_ROW, layout.POS_COL] = a
    img[:, layout.POS_ROW, layout.MIRROR_COL] = a
    lum = img @ LUM
    return (img, lum.reshape(n, 64).astype(np.float32))


M = 3000
print("=" * 100)
print("PROXY ACCEPTABILITY: does the STRUCTURE of a colour face score as well as grayscale?")
print("=" * 100)
print("  the scorer only sees the 4 luminance knobs; colour itself is not judged here")
print()

gi = gray_faces(M, 11)
kn = knobs_from_lum(gi)
s = score(kn)
print(f"  {'pipeline':34s} {'predicted acceptable':>21s} {'mean knob |z|':>14s}")
print("  " + "-" * 72)
print(f"  {'灰度（对照组，已知应约 67%）':34s} {100*np.mean(s>0):20.1f}% "
      f"{np.abs(kn).mean():14.2f}")

# which knob is off? the aggregate score hides this, and the fix depends on the answer
print()
print("  per-knob mean (real faces are 0 by construction):")
print(f"  {'pipeline':20s} {'PC0 明度':>9s} {'PC1 发肤':>9s} {'PC2 眼型':>9s} {'PC3 脸型':>9s}")
print("  " + "-" * 62)
print(f"  {'灰度':20s} " + " ".join(f"{kn[:, i].mean():+9.3f}" for i in range(4)))
print(f"  {'真实数据':20s} " + " ".join(f"{0.0:+9.3f}" for _ in range(4)))

out = {"gray_control": float(np.mean(s > 0)), "gray_knobs": kn.mean(0).tolist()}
for tag in ("E6", "E8"):
    fp = os.path.join(OUT, f"stage2_{tag}.pt")
    if not os.path.exists(fp):
        continue
    net, k = colour_model(tag)
    img, lum = colour_faces(net, k, M, 11)
    kn2 = knobs_from_lum(lum)
    s2 = score(kn2)
    out[tag] = float(np.mean(s2 > 0))
    out[tag + "_knobs"] = kn2.mean(0).tolist()
    print(f"  {('彩色 ' + tag + f'（颜色{k}维）'):20s} "
          + " ".join(f"{kn2[:, i].mean():+9.3f}" for i in range(4)))
    print(f"  {'  → predicted acceptable':20s} {100*np.mean(s2>0):20.1f}% "
          f"{np.abs(kn2).mean():14.2f}")

print()
print("  'mean knob |z|' shows how far the generated faces sit from the average face.")
print("  A large value means the structure is unusual, which the scorer penalises.")

json.dump(out, open(os.path.join(OUT, "proxy_accept.json"), "w", encoding="utf-8"),
          indent=2, ensure_ascii=False)
print(f"\nwrote {os.path.join(OUT, 'proxy_accept.json')}")
print("PROXY_ACCEPT_DONE")
