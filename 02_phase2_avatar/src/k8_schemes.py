#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
Three K=8 colour models that differ only in where the palette comes from.

THE QUESTION
The user wants to know whether a palette taken from real Minecraft colours is worse than one
fitted freely, and whether it matters that the palette is fixed BEFORE training versus snapped
onto the finished model afterwards. Those are two different questions and they get two different
arms:

    k8free   palette fitted by k-means on the art, trained normally      (the existing model)
    k8snap   SAME weights, palette replaced by nearest MC colours        (no retraining at all)
    k8mc     palette constrained to MC colours BEFORE training, retrained

k8free vs k8snap isolates "does the colour substitution cost anything".
k8snap vs k8mc isolates "is it worth retraining for the new palette".

THE MC-CONSTRAINED PALETTE
Chosen by k-medoids: assign every art pixel to its nearest chosen MC colour, then replace each
chosen colour with the MC colour that best represents its cluster, and repeat. This gives the
best 8 MC colours for the data rather than 8 arbitrary ones.

Restricted to the HIGH column of the map palette, since that is the only single column that
contains pure white -- NORMAL tops out at #DCDCDC.
"""
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
OUT = os.path.join(ROOT, "results", "colour", "k8schemes")
WEB = os.path.join(ROOT, "webapp")
os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(8)

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("layout", os.path.join(HERE, "layout.py"))
layout = _ilu.module_from_spec(_spec); _spec.loader.exec_module(layout)

mcj = json.load(open(os.path.join(ROOT, "data", "mc_map_colors.json"), encoding="utf-8"))
MOD = mcj["_modifiers"]
BASE = np.array([b["rgb"] for b in mcj["bases"]], dtype=np.float64)
NAMES = [b["name"] for b in mcj["bases"]]
HIGH = np.clip(BASE * MOD["HIGH"], 0, 255)          # 61 colours, includes SNOW = pure white
HIGH15 = HIGH / 17.0

z = np.load(os.path.join(ROOT, "results", "colour", "cache.npz"))
rgb96 = z["rgb96"].astype(np.float64)
cpc = z["cpc"]
N = len(rgb96)
sp = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_split.npz"))
tr, va = sp["train"], sp["val"]
POS = rgb96.reshape(N, 32, 3)

K = 8
print("=" * 100)
print(f"BUILDING THE MC-CONSTRAINED {K}-COLOUR PALETTE (k-medoids over 61 HIGH colours)")
print("=" * 100)
sample = POS[np.random.default_rng(0).choice(N, 30000, replace=False)].reshape(-1, 3)
rng = np.random.default_rng(1)
chosen = rng.choice(len(HIGH15), K, replace=False)
for it in range(60):
    d = ((sample[:, None, :] - HIGH15[chosen][None, :, :]) ** 2).sum(-1)
    lab = d.argmin(1)
    new = chosen.copy()
    for j in range(K):
        m = lab == j
        if not m.any():
            continue
        sub = sample[m]
        # the medoid: the MC colour closest to this cluster's mean, restricted to MC colours
        mu = sub.mean(0)
        new[j] = int(np.argmin(((HIGH15 - mu) ** 2).sum(-1)))
    if np.array_equal(np.sort(new), np.sort(chosen)) and it > 2:
        chosen = np.unique(new)
        if len(chosen) < K:
            break
        break
    chosen = new
chosen = np.unique(chosen)
print(f"  selected {len(chosen)} MC colours (a medoid step can collapse two picks):")
for i in chosen:
    print(f"    {NAMES[i]:20s} ({HIGH15[i][0]:5.2f},{HIGH15[i][1]:5.2f},{HIGH15[i][2]:5.2f})")
K_eff = len(chosen)
PAL_MC = HIGH15[chosen]
err_mc = float(np.abs(PAL_MC[np.abs(sample[:, None, :] - PAL_MC[None]).sum(-1).argmin(1)] - sample).mean())

# the free palette, for reference
freej = json.load(open(os.path.join(WEB, "colour_pal8_model.json"), encoding="utf-8"))
PAL_FREE = np.array(freej["palette"], dtype=np.float64)
# the same free palette snapped onto the nearest HIGH colours
d = np.array([[np.linalg.norm(PAL_FREE[i] * 17.0 - HIGH[j]) for j in range(len(HIGH))]
              for i in range(len(PAL_FREE))])
snap_best = d.argmin(1)
PAL_SNAP = HIGH15[snap_best]
collide = len(PAL_FREE) - len({tuple(np.round(PAL_SNAP[i], 4)) for i in range(len(PAL_FREE))})

print()
print("=" * 100)
print("THREE PALETTES FOR K=8")
print("=" * 100)
print(f"  {'':14s} {'颜色数':>6s} {'贴到真实颜色的误差':>18s} {'撞色':>5s}")
print("  " + "-" * 50)
for name, P, err, col in (("k8free 自由拟合", PAL_FREE, None, 0),
                          ("k8snap 事后吸附", PAL_SNAP, None, collide),
                          ("k8mc 事前约束", PAL_MC, err_mc, 0)):
    print(f"  {name:14s} {len(P):6d} {(f'{err:.3f}' if err is not None else '—'):>18s} "
          f"{col:5d}")
print()
print("  k8snap 与 k8free 用的是同一批权重，只换调色板；k8mc 是用新调色板重新训练的。")
print(f"  k8snap 有 {collide} 个颜色撞到同一个 MC 色，所以实际只有 {len(PAL_FREE)-collide} 种。")

# ---------------------------------------------------------------- train with a given palette
def train(palette, steps=12000, hid=128, seed=0):
    Kp = len(palette)
    dd = ((POS[:, :, None, :] - palette[None, None, :, :]) ** 2).sum(-1)
    Y = torch.tensor(dd.argmin(-1).reshape(N, 32), dtype=torch.long)
    CT = torch.tensor(cpc[:, :8])
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(8, hid), nn.ReLU(), nn.Linear(hid, hid), nn.ReLU(),
                        nn.Linear(hid, 32 * Kp))
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    tr_t = torch.tensor(tr, dtype=torch.long)
    rng2 = np.random.default_rng(seed)
    for _ in range(steps):
        bi = tr_t[torch.from_numpy(rng2.integers(0, len(tr_t), 256))]
        lg = net(CT[bi]).view(-1, 32, Kp)
        loss = F.cross_entropy(lg.reshape(-1, Kp), Y[bi].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    return net


print()
print("=" * 100)
print("TRAINING THE THIRD ARM (k8mc)")
print("=" * 100)
net_mc = train(PAL_MC)
print(f"  trained 12000 steps with the {K_eff}-colour MC palette")


def export(net, palette, name, note):
    sd = net.state_dict()
    # the Sequential is Linear, ReLU, Linear, ReLU, Linear -- indices 0, 2, 4, not 0, 2, 4, 6
    lin_idx = sorted({int(k.split(".")[0]) for k in sd if k.endswith(".weight")})
    body = [{"W": sd[f"{i}.weight"].tolist(), "b": sd[f"{i}.bias"].tolist()}
            for i in lin_idx[:-1]]
    head = {"W": sd[f"{lin_idx[-1]}.weight"].tolist(),
            "b": sd[f"{lin_idx[-1]}.bias"].tolist()}
    Kp = len(palette)
    mults = 8 * 128 + 128 * 128 + 128 * 32 * Kp
    out = {"type": "palette", "body": body, "head": head,
           "palette": [[float(v) for v in c] for c in palette],
           "meta": {"K": Kp, "hid": 128, "body_layers": len(body), "mults": int(mults),
                    "type": "palette_index", "bits_per_pixel": int(np.ceil(np.log2(Kp))),
                    "note": note}}
    json.dump(out, open(os.path.join(WEB, name), "w", encoding="utf-8"), ensure_ascii=False)
    print(f"  wrote {name}  ({os.path.getsize(os.path.join(WEB, name))/1e3:.0f} KB, "
          f"K={Kp}, {mults:,} mults)")


# k8snap reuses the EXISTING weights, only the palette changes
free_net_layers = freej["body"]
snap_out = {"type": "palette", "body": freej["body"], "head": freej["head"],
            "palette": [[float(v) for v in c] for c in PAL_SNAP],
            "meta": dict(freej["meta"],
                         note=f"k8snap: 原 pal8 权重不变，调色板替换为最近的 {K_eff} 个 MC HIGH 色",
                         scheme="mc_high_snapped")}
json.dump(snap_out, open(os.path.join(WEB, "colour_k8snap_model.json"), "w",
                         encoding="utf-8"), ensure_ascii=False)
print(f"  wrote colour_k8snap_model.json  (weights unchanged, palette replaced)")

export(net_mc, PAL_MC, "colour_k8mc_model.json",
       "k8mc: 调色板事前限定在 MC HIGH 列，重新训练")

json.dump({"mc_palette": [[float(v) for v in c] for c in PAL_MC],
           "mc_names": [NAMES[i] for i in chosen],
           "snap_palette": [[float(v) for v in c] for c in PAL_SNAP],
           "snap_names": [NAMES[i] for i in snap_best],
           "collide": int(collide), "err_mc": err_mc},
          open(os.path.join(OUT, "palettes.json"), "w", encoding="utf-8"),
          indent=2, ensure_ascii=False)
print(f"\nwrote {os.path.join(OUT, 'palettes.json')}")
print("K8SCHEMES_DONE")
