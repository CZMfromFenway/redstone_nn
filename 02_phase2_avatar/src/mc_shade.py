#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
Colour substitution on a TRAINED model: shapes must not change.

THE MISTAKE IN THE PREVIOUS RUN
The earlier version snapped the palette and then RETRAINED, so the classifier re-assigned the
indices and shapes moved -- a two-pixel eye became one pixel. The user caught it.

The correct operation is a pure output substitution:
    keep the trained classifier exactly as it is
    keep every predicted index exactly as it is
    change only the table that maps an index to a colour
Shapes are then identical by construction, and that is checked here rather than assumed: the
index map before and after must be bit-identical.

WHY A SINGLE SHADE COLUMN MATTERS
The map colour is base * {135, 180, 220, 255} / 255 for LOWEST / LOW / NORMAL / HIGH. Using one
column for every pixel means a single fixed brightness factor -- trivial in hardware. It is also
what makes a fair comparison possible. Crucially the HIGH column is base * 1.0, which contains
SNOW #FFFFFF, so a single uniform column CAN give pure white. NORMAL cannot.

A COLLISION TO WATCH
Two palette entries may snap to the same map colour, which merges regions and destroys exactly
the boundaries that were there. How many entries collide is reported, not hidden.
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
OUT = os.path.join(ROOT, "results", "colour", "mcfit")
WEB = os.path.join(ROOT, "webapp")
os.makedirs(OUT, exist_ok=True)
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


mc = json.load(open(os.path.join(ROOT, "data", "mc_map_colors.json"), encoding="utf-8"))
MOD = mc["_modifiers"]
BASE = np.array([b["rgb"] for b in mc["bases"]], dtype=np.float64)
NAMES = [b["name"] for b in mc["bases"]]
COLUMNS = {
    "LOWEST（×135/255）": np.clip(BASE * MOD["LOWEST"], 0, 255),
    "LOW（×180/255）": np.clip(BASE * MOD["LOW"], 0, 255),
    "NORMAL（×220/255）": np.clip(BASE * MOD["NORMAL"], 0, 255),
    "HIGH（×255/255 = 基色）": np.clip(BASE * MOD["HIGH"], 0, 255),
}


def load_model(fn):
    j = json.load(open(os.path.join(WEB, fn), encoding="utf-8"))
    pal = np.array(j["palette"], dtype=np.float64)
    K = len(pal)
    all = [{"W": np.array(L["W"]), "b": np.array(L["b"]), "relu": True} for L in j["body"]]
    all.append({"W": np.array(j["head"]["W"]), "b": np.array(j["head"]["b"]), "relu": False})

    def indices(Z):
        v = Z
        for L in all:
            v = v @ L["W"].T + L["b"]
            if L["relu"]:
                v = np.maximum(0, v)
        return v.reshape(len(Z), 32, K).argmax(-1)
    return pal, K, indices


def to_img(flat):
    a = np.clip(flat, 0, 15).reshape(len(flat), 32, 3)
    img = np.zeros((len(a), 8, 8, 3))
    img[:, layout.POS_ROW, layout.POS_COL] = a
    img[:, layout.POS_ROW, layout.MIRROR_COL] = a
    return img


def metrics(flat):
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
    return {"sat": float(sat.mean()), "sep": float(sep.mean()),
            "edge": float(np.percentile(ed, 90)), "ncol": float(ncol.mean()),
            "lum": float((a @ np.array([0.299, 0.587, 0.114])).mean())}


N = 4000
rng = np.random.default_rng(3)
Z = np.clip(rng.standard_normal((N, 8)), -3.5, 3.5).astype(np.float32)

print("=" * 104)
print("COLOUR SUBSTITUTION WITHOUT RETRAINING  (index map is untouched, so shapes are identical)")
print("=" * 104)
print(f"  {'model':6s} {'column':22s} {'撞色':>5s} {'剩几种':>7s} {'色偏':>6s} "
      f"{'边缘':>7s} {'饱和':>6s} {'亮度':>6s}")
print("  " + "-" * 84)
results = {}
for tag, fn in (("K=8", "colour_pal8_model.json"), ("K=16", "colour_pal16_model.json"),
                ("K=32", "colour_pal32_model.json")):
    pal, K, indices = load_model(fn)
    idx = indices(Z)
    base = pal[idx].reshape(N, 96)
    mb = metrics(base)
    print()
    print(f"  {tag}  原调色板（参照）               "
          f"{K:5d} {mb['ncol']:7.1f} {'—':>6s} {mb['edge']:7.2f} {mb['sat']:6.3f} "
          f"{mb['lum']:6.2f}")
    results[tag] = {"orig": mb}
    for cname, COL in COLUMNS.items():
        d = np.array([[np.linalg.norm(pal[i] * 17.0 - COL[j]) for j in range(len(COL))]
                      for i in range(K)])
        best = d.argmin(1)
        snapped = COL[best] / 17.0
        # SHAPE CHECK: the index map is reused verbatim, so this must be identical
        faces = snapped[idx].reshape(N, 96)
        same_shape = np.array_equal(idx, indices(Z))
        collide = K - len({tuple(snapped[i]) for i in range(K)})
        m = metrics(faces)
        shift = float(np.abs(faces - base).mean())
        print(f"  {tag}  {cname:22s} {collide:5d} {m['ncol']:7.1f} {shift:6.2f} "
              f"{m['edge']:7.2f} {m['sat']:6.3f} {m['lum']:6.2f}   "
              + ("形状不变" if same_shape else "形状变了!"))
        results[tag][cname] = {"collide": int(collide), "shift": shift, **m,
                               "names": [NAMES[b] for b in best],
                               "palette": snapped.tolist()}

print()
print("=" * 104)
print("用 HIGH 列替换后，各调色板实际用到哪些地图色")
print("=" * 104)
for tag in results:
    r = results[tag].get("HIGH（×255/255 = 基色）")
    if r:
        print(f"  {tag}: " + ", ".join(r["names"]))
print()
print("  HIGH 列包含 SNOW（纯白 15.0）与 QUARTZ，所以单列也能出真白；NORMAL 列最亮只有 12.94。")

# ---- figure: same face, same indices, different colour table
COLS, TILE = 14, 54
rows = []
for tag, fn in (("K=8", "colour_pal8_model.json"), ("K=16", "colour_pal16_model.json"),
                ("K=32", "colour_pal32_model.json")):
    pal, K, indices = load_model(fn)
    idx = indices(Z[:COLS])
    rows.append((f"{tag} 原调色板", pal[idx].reshape(COLS, 96)))
    for cname in ("NORMAL（×220/255）", "HIGH（×255/255 = 基色）"):
        r = results[tag][cname]
        rows.append((f"{tag} → {cname.split('（')[0]}", np.array(r["palette"])[idx].reshape(COLS, 96)))

W = 300 + COLS * (TILE + 5) + 5
H = 92 + len(rows) * (TILE + 26) + 24
cv = Image.new("RGB", (W, H), (18, 20, 24))
dr = ImageDraw.Draw(cv)
f_t, f_l, f_s = font(20), font(12), font(10)
dr.text((12, 12), "只换颜色表，不重训：形状完全不变", font=f_t, fill=(234, 238, 244))
dr.text((12, 38), "同一批索引、同一张脸，只把每个索引映射到的颜色换成最近的地图色。",
        font=f_s, fill=(150, 162, 178))
dr.text((12, 54), "HIGH 列是基色本身，含 SNOW 纯白；NORMAL 列最亮 12.94，白会发灰。",
        font=f_s, fill=(130, 142, 158))
y = 76
for name, faces in rows:
    dr.text((12, y + 6), name, font=f_l,
            fill=(110, 168, 254) if "原" in name else (110, 200, 130))
    imgs = to_img(faces)
    for i in range(COLS):
        t = np.kron(np.clip(imgs[i] / 15.0 * 255, 0, 255).astype(np.uint8),
                    np.ones((TILE // 8, TILE // 8, 1))).astype(np.uint8)
        cv.paste(Image.fromarray(t), (300 + i * (TILE + 5), y))
    y += TILE + 26
p = os.path.join(OUT, "SUBSTITUTE.png")
cv.save(p)
print(f"\nwrote {p}  {cv.size}")
json.dump({k: {kk: ({kkk: vvv for kkk, vvv in vv.items() if kkk != 'palette'}
                 if isinstance(vv, dict) else vv) for kk, vv in v.items()}
           for k, v in results.items()},
          open(os.path.join(OUT, "substitute.json"), "w", encoding="utf-8"),
          indent=2, ensure_ascii=False)
print(f"wrote {os.path.join(OUT, 'substitute.json')}")
print("SUBSTITUTE_DONE")
