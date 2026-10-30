#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Export the BEST model (final_lam5_20k: 22-dim condition, coherence penalty lam=5,
20000 steps) for the web page, and add a style-clustering visualisation that lets the
user CHECK whether the clusters match their own visual categories.

Model differences from the earlier exports that the page must know about:
  * in_dim = 22 (condition only, no latent noise)
  * it was trained with the TV coherence penalty, so its argmax output is far less
    speckled than the 1500-step model already on the page

Style visualisation: cluster ALL validation faces into k style groups using the PCA-22
space (the space that scored best against the user's labels), then draw an INDEXED
contact sheet per cluster so the grouping can be judged by eye, plus a large sheet of
random samples per cluster.
"""
import argparse, json, os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw

import importlib.util as _ilu
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
_lspec = _ilu.spec_from_file_location("layout", os.path.join(HERE, "layout.py"))
layout = _ilu.module_from_spec(_lspec); _lspec.loader.exec_module(layout)
NLEV = 16
LEVELS = np.linspace(0, 255, 16)

AP = argparse.ArgumentParser()
AP.add_argument("--model-tag", default="final_lam5_20k")
AP.add_argument("--coherence-dir", default="coherence")
AP.add_argument("--out-name", default="pca22_lam5_model.json")
AP.add_argument("--k", type=int, default=4)
AP.add_argument("--per-cluster", type=int, default=24)
A = AP.parse_args()

WEB = os.path.join(ROOT, "webapp")
os.makedirs(WEB, exist_ok=True)
OUT = os.path.join(ROOT, "results", "style")
os.makedirs(OUT, exist_ok=True)


class Gen(nn.Module):
    def __init__(self, hid=256, cdim=22):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(cdim, hid), nn.ReLU(),
                                 nn.Linear(hid, hid), nn.ReLU(),
                                 nn.Linear(hid, 32 * NLEV))
    def logits(self, cond):
        return self.net(cond).view(-1, 32, NLEV)


# ---------------------------------------------------------------- shared PCA basis
rgb = np.load(os.path.join(ROOT, "data", "faces8_rgb_v2.npy"))
LUM = np.array([0.299, 0.587, 0.114])
gray = (rgb.astype(np.float64) * LUM).sum(-1)
gyi = np.abs(gray[:, :, :, None] - LEVELS[None, None, None, :]).argmin(-1)
kn = (gyi[:, layout.POS_ROW, layout.POS_COL] ==
      gyi[:, layout.POS_ROW, layout.MIRROR_COL]).mean(1) >= 0.95
Gg = gray[kn]
full = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_full_idx.npy"))
X = Gg.reshape(len(Gg), -1)
mu = X.mean(0, keepdims=True)
sub = np.random.default_rng(0).choice(len(X), min(60000, len(X)), replace=False)
U, S_, Vt = np.linalg.svd(X[sub] - mu, full_matrices=False)
comps = Vt[:22]
sd = (((X - mu) @ comps.T).std(0, keepdims=True) + 1e-6)
STYLE = ((X - mu) @ comps.T) / sd[0]
ev = (S_ ** 2) / (S_ ** 2).sum()
print(f"shared PCA basis: 22 comps, {100*ev[:22].sum():.2f}% variance")
va = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_split.npz"))["val"]

# ---------------------------------------------------------------- export the model
run = os.path.join(ROOT, "results", A.coherence_dir, A.model_tag)
ck = torch.load(os.path.join(run, "final.pt"), weights_only=False)
a = ck["args"]
Gm = Gen(a["hid"], 22); Gm.load_state_dict(ck["G"]); Gm.eval()
W1 = Gm.net[0].weight.detach().numpy(); b1 = Gm.net[0].bias.detach().numpy()
W2 = Gm.net[2].weight.detach().numpy(); b2 = Gm.net[2].bias.detach().numpy()
W3 = Gm.net[4].weight.detach().numpy(); b3 = Gm.net[4].bias.detach().numpy()
print(f"\nmodel {A.model_tag}: steps={a['steps']} lam={a['lam']} decode={a['decode']} "
      f"adv_weight={a.get('adv_weight')}")
print(f"  in_dim={W1.shape[1]} (no latent noise)  W1{W1.shape} W2{W2.shape} W3{W3.shape}")

# the shared gallery must be the SAME 24 rows the other exports used
shared_rows = va[np.random.default_rng(11).choice(len(va), 24, replace=False)]
gallery = []
for r in shared_rows:
    pca = ((X[r] - mu[0]) @ comps.T) / sd[0]
    gallery.append({"rgb": full[r].reshape(-1).tolist(),
                    "pca": [round(float(v), 6) for v in pca]})

out = {
    "meta": {
        "source_run": A.model_tag,
        "use_noise": False,
        "in_dim": int(W1.shape[1]),
        "steps_trained": int(a["steps"]),
        "batch": int(a.get("batch", 256)),
        "lr_g": float(a.get("lr_g", 3e-4)),
        "lr_d": float(a.get("lr_d", 2e-4)),
        "ce_weight": 1.0,
        "coherence_penalty": "TV on the softmax probabilities",
        "coherence_lambda": float(a["lam"]),
        "adversarial_weight": float(a.get("adv_weight", 1.0)),
        "zdim": 0,
        "hid": int(a["hid"]),
        "pca_k": 22,
        "pca_variance_explained": float(ev[:22].sum()),
        "seed": int(a.get("seed", 0)),
        "layout": "canonical: p = col*8 + row over the left half; mirror within each row",
        "note": ("condition only, no latent z; trained with the spatial-coherence penalty "
                 "so the argmax output is much less speckled than the 1500-step model"),
    },
    "levels": [float(v) for v in LEVELS],
    "pca": {"mean": [float(v) for v in mu[0]],
            "components": [[float(v) for v in row] for row in comps],
            "sd": [float(v) for v in sd[0]]},
    "gen": {"W1": W1.tolist(), "b1": b1.tolist(),
            "W2": W2.tolist(), "b2": b2.tolist(),
            "W3": W3.tolist(), "b3": b3.tolist()},
    "gallery": gallery,
}
p = os.path.join(WEB, A.out_name)
json.dump(out, open(p, "w", encoding="utf-8"))
print(f"  wrote {A.out_name}  ({os.path.getsize(p)/1e6:.2f} MB)")

# ---------------------------------------------------------------- style clusters, all val
K = A.k
rngc = np.random.default_rng(1)
tr = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_split.npz"))["train"]
C = STYLE[tr][rngc.choice(len(tr), K, replace=False)].copy()
lab_tr = np.zeros(len(tr), dtype=int)
for _ in range(40):
    d = ((STYLE[tr][:, None, :] - C[None, :, :]) ** 2).sum(2)
    new = d.argmin(1)
    if (new == lab_tr).all():
        break
    lab_tr = new
    for j in range(K):
        if (lab_tr == j).any():
            C[j] = STYLE[tr][lab_tr == j].mean(0)
lab_va = ((STYLE[va][:, None, :] - C[None, :, :]) ** 2).sum(2).argmin(1)
sizes = np.bincount(lab_va, minlength=K)
print(f"\nstyle clustering (k={K}) on the validation split: sizes {sizes.tolist()}")

# per-cluster statistics so the grouping is described by numbers, not just pictures
g = full[va].astype(np.float64)
top = g[:, :3, :].mean(axis=(1, 2)); bot = g[:, 5:, :].mean(axis=(1, 2))
prof = []
print(f"\n{'cl':>3} {'n':>6} {'share':>6} {'bright':>7} {'top-bot':>8} "
      f"{'contrast':>9} {'edge':>6} {'flat%':>6}")
print("-" * 62)
for j in range(K):
    m = lab_va == j
    if m.sum() == 0:
        continue
    pr = {"cluster": j, "n": int(m.sum()), "share": float(m.mean()),
          "brightness": float(g[m].mean()), "top_minus_bot": float((top - bot)[m].mean()),
          "contrast": float(g[m].std(axis=(1, 2)).mean()),
          "edge": float((np.abs(np.diff(g[m], axis=2)).mean(axis=(1, 2)) +
                         np.abs(np.diff(g[m], axis=1)).mean(axis=(1, 2))).mean()),
          "flat_frac": float((g[m].std(axis=(1, 2)) < 0.6).mean())}
    prof.append(pr)
    print(f"{j:3d} {pr['n']:6d} {pr['share']*100:5.1f}% {pr['brightness']:7.2f} "
          f"{pr['top_minus_bot']:+8.2f} {pr['contrast']:9.2f} {pr['edge']:6.2f} "
          f"{pr['flat_frac']*100:5.1f}")
print()
print(f"whole validation set: brightness {g.mean():.2f}  top-bot {(top-bot).mean():+.2f} "
      f"contrast {g.std(axis=(1,2)).mean():.2f}  flat {(g.std(axis=(1,2))<0.6).mean()*100:.1f}%")


def sheet(rows, cols, label=True, tile=48):
    imgs = full[rows]
    pad, lab = 6, (14 if label else 0)
    n = len(rows)
    r_ = (n + cols - 1) // cols
    W = cols * (tile + pad) + pad
    H = r_ * (tile + pad + lab) + pad
    cv = Image.new("RGB", (W, H), (24, 26, 30))
    dr = ImageDraw.Draw(cv)
    for i in range(n):
        rr, cc = divmod(i, cols)
        x = pad + cc * (tile + pad)
        y = pad + rr * (tile + pad + lab)
        t = np.kron(LEVELS[imgs[i]], np.ones((tile // 8, tile // 8), dtype=np.uint8))
        cv.paste(Image.fromarray(np.clip(t, 0, 255).astype(np.uint8), mode="L").convert("RGB"), (x, y))
        if label:
            dr.text((x + 2, y + tile + 1), str(rows[i]), fill=(180, 190, 200))
    return cv


# one sheet per cluster: 24 random members, each labelled with its validation index
for j in range(K):
    idx = va[lab_va == j]
    if len(idx) == 0:
        continue
    pick = np.random.default_rng(100 + j).choice(len(idx), min(A.per_cluster, len(idx)),
                                                 replace=False)
    sh = sheet(idx[pick], 8, label=True)
    sh.save(os.path.join(OUT, f"cluster_{j}_samples.png"))
    print(f"  wrote cluster_{j}_samples.png ({len(pick)} faces)")

# also one big sheet of everything ordered by cluster, so the grouping is visible at once
order = np.concatenate([va[lab_va == j][:24] for j in range(K) if (lab_va == j).any()])
sheet(order, 24, label=False, tile=32).save(os.path.join(OUT, "clusters_all_ordered.png"))
print("  wrote clusters_all_ordered.png (24 per cluster, rows = clusters)")

json.dump({"k": K, "sizes": sizes.tolist(), "profiles": prof,
           "note": "clusters computed in the PCA-22 style space via k-means; "
                   "the same centres were used to label the validation split"},
          open(os.path.join(OUT, "cluster_visual_meta.json"), "w"), indent=2)
print(f"\nwrote {os.path.join(WEB, A.out_name)} and cluster sheets in {OUT}")
print("EXPORT_BEST_DONE")
