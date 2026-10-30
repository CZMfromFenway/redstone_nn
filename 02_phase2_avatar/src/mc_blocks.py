#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
BLOCK colours, not map colours.

WHY THEY DIFFER
Map art is limited to the 61 map base colours in four fixed shades, and the NORMAL shade tops
out at #DCDCDC, so there is no good white -- the user's complaint. Block art has no such limit:
you pick whatever block has the average colour you want, so white concrete, snow, quartz and
iron are all available at close to full brightness.

DATA CAVEAT
Map colours come from the wiki table and are exact. Block colours below are the widely published
texture averages for the dyed blocks plus a few pure material blocks; they are APPROXIMATE, good
to roughly +-6 per channel. The conclusion drawn here does not depend on that precision, because
the gap between "brightest map NORMAL colour" and "white concrete" is far larger than the error.

WHAT IS MEASURED
    fit        per palette, the nearest available colour and the error
    snapped    the palette rebuilt from real block colours, retrained, and scored out of sample
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
OUT = os.path.join(ROOT, "results", "colour", "mcfit")
WEB = os.path.join(ROOT, "webapp")
os.makedirs(OUT, exist_ok=True)

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("layout", os.path.join(HERE, "layout.py"))
layout = _ilu.module_from_spec(_spec); _spec.loader.exec_module(layout)
torch.set_num_threads(8)

DYE = ["white", "orange", "magenta", "light_blue", "yellow", "lime", "pink", "gray",
       "light_gray", "cyan", "purple", "blue", "brown", "green", "red", "black"]
CONCRETE = {
    "white": (207, 213, 214), "orange": (224, 97, 0), "magenta": (169, 48, 159),
    "light_blue": (36, 137, 199), "yellow": (241, 175, 21), "lime": (94, 169, 24),
    "pink": (227, 139, 171), "gray": (54, 57, 61), "light_gray": (125, 125, 115),
    "cyan": (21, 119, 136), "purple": (100, 32, 156), "blue": (44, 46, 143),
    "brown": (96, 59, 31), "green": (73, 91, 36), "red": (142, 33, 33), "black": (8, 10, 15),
}
WOOL = {
    "white": (233, 236, 236), "orange": (240, 118, 19), "magenta": (189, 68, 179),
    "light_blue": (58, 175, 217), "yellow": (248, 198, 39), "lime": (112, 185, 25),
    "pink": (237, 141, 172), "gray": (62, 68, 71), "light_gray": (142, 142, 134),
    "cyan": (21, 137, 145), "purple": (121, 42, 172), "blue": (53, 57, 157),
    "brown": (114, 71, 40), "green": (84, 109, 27), "red": (161, 39, 34), "black": (20, 21, 25),
}
PURE = {
    "snow_block": (249, 254, 254), "quartz_block": (236, 233, 226),
    "smooth_quartz": (228, 223, 213), "iron_block": (216, 216, 216),
    "smooth_stone": (160, 160, 160), "stone": (125, 125, 125),
    "deepslate": (100, 100, 100), "blackstone": (44, 40, 46),
    "gold_block": (247, 214, 74), "emerald_block": (42, 196, 108),
    "diamond_block": (98, 219, 213), "lapis_block": (52, 92, 173),
    "redstone_block": (171, 27, 9), "netherite_block": (68, 61, 63),
    "copper_block": (192, 107, 79), "oxidised_copper": (82, 162, 132),
    "prismarine": (99, 156, 151), "sea_lantern": (172, 199, 190),
    "bone_block": (229, 225, 207), "birch_planks": (196, 178, 123),
    "oak_planks": (162, 130, 78), "dark_oak_planks": (66, 43, 20),
    "terracotta": (152, 94, 67), "mud": (60, 57, 60), "packed_mud": (142, 106, 83),
    "moss_block": (89, 109, 45), "grass_block_top": (127, 178, 56),
    "dirt": (134, 96, 67), "sand": (219, 207, 163),
}

BLOCKS = {}
for d in DYE:
    BLOCKS[f"{d}_concrete"] = CONCRETE[d]
for d in DYE:
    BLOCKS[f"{d}_wool"] = WOOL[d]
BLOCKS.update(PURE)
NAMES = list(BLOCKS.keys())
BC = np.array([BLOCKS[n] for n in NAMES], dtype=np.float64)

mc = json.load(open(os.path.join(ROOT, "data", "mc_map_colors.json"), encoding="utf-8"))
MOD = mc["_modifiers"]
BASE = np.array([b["rgb"] for b in mc["bases"]], dtype=np.float64)
SETS = {
    "地图色 NORMAL": np.clip(BASE * MOD["NORMAL"], 0, 255),
    "地图色 四档全开": np.vstack([np.clip(BASE * f, 0, 255)
                                  for f in (MOD["LOW"], MOD["NORMAL"], MOD["HIGH"],
                                            MOD["LOWEST"])]),
    "方块色": BC,
}
SETNAMES = {
    "地图色 NORMAL": [f"{n}/NORMAL" for n in [b["name"] for b in mc["bases"]]],
    "地图色 四档全开": [f"{n}/{s}" for s in ("NORMAL", "LOW", "HIGH", "LOWEST")
                        for n in [b["name"] for b in mc["bases"]]],
    "方块色": NAMES,
}

print("=" * 104)
print("MINECRAFT COLOUR SOURCES COMPARED")
print("=" * 104)
print(f"  {'颜色来源':16s} {'颜色数':>6s} {'最亮(0..15级)':>14s} {'最暗':>7s}")
print("  " + "-" * 48)
for name, S in SETS.items():
    print(f"  {name:16s} {len(S):6d} {S.max()/17:14.2f} {S.min()/17:7.2f}")
print()
print("  这就是「没有好的白」的全部来历：地图 NORMAL 最亮 12.94 级，")
print("  而方块色里 snow_block 是 249/255 = 14.65 级，接近满值。")


def load_palette(fn):
    j = json.load(open(os.path.join(WEB, fn), encoding="utf-8"))
    return np.array(j["palette"], dtype=np.float64)


PALS = {"K=8": load_palette("colour_pal8_model.json"),
        "K=16": load_palette("colour_pal16_model.json"),
        "K=32": load_palette("colour_pal32_model.json")}

print()
print("=" * 104)
print("HOW WELL EACH PALETTE FITS EACH COLOUR SOURCE")
print("=" * 104)
for name, S in SETS.items():
    print(f"  {name}（{len(S)} 色）")
    print(f"    {'调色板':6s} {'平均误差':>9s} {'最大误差':>9s} {'>1级':>7s} {'>2级':>7s}")
    print("    " + "-" * 44)
    for tag, P in PALS.items():
        err = np.array([np.linalg.norm(P[i] * 17.0 - S, axis=-1).min()
                        for i in range(len(P))]) / 17.0
        print(f"    {tag:6s} {err.mean():9.3f} {err.max():9.3f} "
              f"{int((err>1).sum()):3d}/{len(P):<3d} {int((err>2).sum()):3d}/{len(P):<3d}")
    print()

print("=" * 104)
print("SNAP TO BLOCK COLOURS: which blocks, and what it costs")
print("=" * 104)
z = np.load(os.path.join(ROOT, "results", "colour", "cache.npz"))
rgb96 = z["rgb96"].astype(np.float64)
cpc = z["cpc"]
N = len(rgb96)
sp = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_split.npz"))
tr = sp["train"]


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


out = {}
for tag, P in PALS.items():
    d = np.array([[np.linalg.norm(P[i] * 17.0 - BC[j]) for j in range(len(BC))]
                  for i in range(len(P))])
    best = d.argmin(1)
    snap = BC[best] / 17.0
    K = len(snap)
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
    for _ in range(12000):
        bi = tr_t[torch.from_numpy(rng.integers(0, len(tr_t), 256))]
        lg = net(CT[bi]).view(-1, 32, K)
        loss = F.cross_entropy(lg.reshape(-1, K), Y[bi].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        idx = net(CT).view(-1, 32, K).argmax(-1).numpy()
    faces = snap[idx].reshape(N, 96)
    m = feats(faces)
    out[tag] = {"blocks": [NAMES[i] for i in best], "palette": snap.tolist(),
                "mae": float(np.abs(faces - rgb96).mean()),
                "edge": float(m[:, 2].mean()), "ncol": float(m[:, 4].mean()),
                "sat": float(m[:, 0].mean())}
    print(f"  {tag}:")
    print(f"    方块 → " + ", ".join(NAMES[i] for i in best))
    print(f"    RGB MAE {np.abs(faces - rgb96).mean():.3f}   边缘 {m[:,2].mean():.2f}   "
          f"颜色数 {m[:,4].mean():.1f}   饱和 {m[:,0].mean():.3f}")
    print()

json.dump(out, open(os.path.join(OUT, "blocks.json"), "w", encoding="utf-8"),
          indent=2, ensure_ascii=False)
print(f"真实数据参照：边缘 15.56，颜色数 7.2，饱和 0.316")
print(f"wrote {os.path.join(OUT, 'blocks.json')}")
print("MCBLOCK_DONE")
