#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
Export the C1 and C2 attempts with the snapped palette, and measure the banding artefact C1
introduced.

THE ARTEFACT
C1 was rewarded for raising the spread of the eight row means, and it found the cheapest way to do
that: a dark horizontal band at the eye rows. Every face gets the same band, so the pictures look
like everyone is wearing a visor. That is measured here as the drop in luminance at rows 3-4
relative to the rows above and below -- a normal face has some, a C1 face has much more.

The palette is snapped onto the same eight Minecraft colours as the shipped k8snap model, so the
three arms in the next blind test differ by weights only.
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
OUT = os.path.join(ROOT, "results", "colour", "yield")
WEB = os.path.join(ROOT, "webapp")
torch.set_num_threads(8)

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("layout", os.path.join(HERE, "layout.py"))
layout = _ilu.module_from_spec(_spec); _spec.loader.exec_module(layout)
LUM = np.array([0.299, 0.587, 0.114])

snap = json.load(open(os.path.join(WEB, "colour_k8snap_model.json"), encoding="utf-8"))
PAL_SNAP = np.array(snap["palette"], dtype=np.float64)
K = len(PAL_SNAP)


def load_ck(fn):
    ck = torch.load(os.path.join(OUT, fn), weights_only=False)
    st = ck["G"]
    idx = sorted({int(k.split(".")[0]) for k in st if k.endswith(".weight")})
    seq = []
    for j in idx:
        seq.append(nn.Linear(int(st[f"{j}.weight"].shape[1]), int(st[f"{j}.weight"].shape[0])))
        if j != idx[-1]:
            seq.append(nn.ReLU())
    m = nn.Sequential(*seq); m.load_state_dict(st); m.eval()
    return m


def export(m, name, note):
    body, head = [], None
    sd = m.state_dict()
    idx = sorted({int(k.split(".")[0]) for k in sd if k.endswith(".weight")})
    body = [{"W": sd[f"{i}.weight"].tolist(), "b": sd[f"{i}.bias"].tolist()}
            for i in idx[:-1]]
    head = {"W": sd[f"{idx[-1]}.weight"].tolist(), "b": sd[f"{idx[-1]}.bias"].tolist()}
    mults = 8 * 128 + 128 * 128 + 128 * 32 * K
    out = {"type": "palette", "body": body, "head": head,
           "palette": [[float(v) for v in c] for c in PAL_SNAP],
           "meta": {"K": K, "hid": 128, "mults": int(mults), "type": "palette_index",
                    "bits_per_pixel": 3, "note": note}}
    json.dump(out, open(os.path.join(WEB, name), "w", encoding="utf-8"), ensure_ascii=False)
    print(f"  wrote {name}  ({mults:,} mults)")


def measure(m, tag):
    Z = np.clip(np.random.default_rng(31).standard_normal((4000, 8)), -3.5, 3.5).astype(np.float32)
    with torch.no_grad():
        i = m(torch.tensor(Z)).view(-1, 32, K).argmax(-1).numpy()
    a = PAL_SNAP[i].reshape(len(Z), 32, 3)
    lu = a @ LUM
    rows = np.stack([lu[:, [c * 8 + r for c in range(4)]].mean(1) for r in range(8)], 1)
    # banding: how much darker the eye rows are than the rows above and below
    band = ((rows[:, :2].mean(1) + rows[:, 4:].mean(1)) / 2) - rows[:, 2:4].mean(1)
    print(f"  {tag:16s} 行方差 {rows.std(1).mean():.2f}  "
          f"眼带深度 {band.mean():.2f}  行均值 "
          + " ".join(f"{v:.1f}" for v in rows.mean(0)))
    return float(band.mean()), float(rows.std(1).mean())


print("=" * 100)
print("BANDING ARTEFACT")
print("=" * 100)
res = {}
for tag, fn, name, note in (
        ("C1 λ=1", "ck_c1_1.0_1.pt", "colour_c1_model.json",
         "C1 结构损失 λ=1 + 吸附调色板"),
        ("C2 对抗", "ck_c2_1.0_3.pt", "colour_c2_model.json",
         "C2 对抗损失 λ=0.3 + 吸附调色板")):
    p = os.path.join(OUT, fn)
    if not os.path.exists(p):
        print(f"  skip {tag}: missing {fn}")
        continue
    m = load_ck(p)
    band, spread = measure(m, tag)
    res[tag] = {"band": band, "row_spread": spread}
    export(m, name, note)

# reference: the shipped model
layers = [{"W": np.array(L["W"]), "b": np.array(L["b"]), "relu": True} for L in snap["body"]]
layers.append({"W": np.array(snap["head"]["W"]), "b": np.array(snap["head"]["b"]), "relu": False})
Z = np.clip(np.random.default_rng(31).standard_normal((4000, 8)), -3.5, 3.5).astype(np.float32)
v = Z
for L in layers:
    v = v @ L["W"].T + L["b"]
    if L["relu"]:
        v = np.maximum(0, v)
a = PAL_SNAP[v.reshape(len(Z), 32, K).argmax(-1)].reshape(len(Z), 32, 3)
lu = a @ LUM
rows = np.stack([lu[:, [c * 8 + r for c in range(4)]].mean(1) for r in range(8)], 1)
band = ((rows[:, :2].mean(1) + rows[:, 4:].mean(1)) / 2) - rows[:, 2:4].mean(1)
print(f"  {'k8snap 现役':16s} 行方差 {rows.std(1).mean():.2f}  "
      f"眼带深度 {band.mean():.2f}  行均值 "
      + " ".join(f"{v:.1f}" for v in rows.mean(0)))
res["k8snap"] = {"band": float(band.mean()), "row_spread": float(rows.std(1).mean())}

print()
print("  眼带深度＝（上两行＋下四行的平均亮度）－（中间两行的平均亮度）。")
print("  正常脸有一定深度是因为眼睛本来就暗；C1 的深度远超正常，那就是横带。")
json.dump(res, open(os.path.join(OUT, "banding.json"), "w", encoding="utf-8"),
          indent=2, ensure_ascii=False)

man = [
    {"key": "k8snap", "file": "colour_k8snap_model.json",
     "note": "K=8 现役：自由拟合调色板 + 训练后吸附"},
    {"key": "c2adv", "file": "colour_c2_model.json",
     "note": "K=8 对抗损失 λ=0.3 + 吸附调色板"},
    {"key": "c1struct", "file": "colour_c1_model.json",
     "note": "K=8 结构损失 λ=1 + 吸附调色板"},
]
p = os.path.join(WEB, "colour_c1c2_manifest.json")
json.dump(man, open(p, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print(f"\nwrote {p}")
for a in man:
    print(f"  {a['key']:9s} {a['file']}")
print("C1C2_EXPORT_DONE")
