#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Export the stretch-trained k=4 model and rebuild the blind-test manifest.

The decision this round answers: PCA-4 wants more contrast. There are two places to put it:

    dim4x        post-process: stretch the decoded levels to [0,15] at display time.
                 Works (round 2: 62.1%), but in redstone it needs a min/max circuit.
    dim4_stretch baked in: the model was trained with every target already stretched to
                 [0,15], so its output is full-range by construction -- no min/max needed.

If the baked version scores like the post-processed one, the redstone build can drop the
min/max step entirely. That is the whole point of this round.
"""
import importlib.util as _ilu
import io
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
WEB = os.path.join(ROOT, "webapp")
SRC = os.path.join(ROOT, "results", "dimsweep")
NLEV = 16
LEVELS = np.linspace(0, 255, 16)

_spec = _ilu.spec_from_file_location("layout", os.path.join(HERE, "layout.py"))
layout = _ilu.module_from_spec(_spec); _spec.loader.exec_module(layout)


def export(tag, out_name, post):
    path = os.path.join(SRC, tag, "final.pt")
    if not os.path.exists(path):
        print(f"  {tag}: checkpoint missing")
        return None
    ck = torch.load(path, weights_only=False)
    st = ck.get("G") or ck.get("S")
    st = {(k.replace("net.", "", 1) if k.startswith("net.") else k): v for k, v in st.items()}
    idx = sorted({int(k.split(".")[0]) for k in st if k.endswith(".weight")})
    layers = []
    for j in idx:
        W = st[f"{j}.weight"].numpy()
        b = st[f"{j}.bias"].numpy()
        assert W.shape[0] == b.shape[0]
        layers.append({"W": [[float(v) for v in row] for row in W],
                       "b": [float(v) for v in b]})
    in_dim = len(layers[0]["W"][0])
    out_dim = len(layers[-1]["W"])

    # verify the JS forward from the emitted weights
    seq = []
    for jn, j in enumerate(idx):
        W = st[f"{j}.weight"]
        seq.append(nn.Linear(int(W.shape[1]), int(W.shape[0])))
        if jn != len(idx) - 1:
            seq.append(nn.ReLU())
    m = nn.Sequential(*seq); m.load_state_dict(st); m.eval()
    rng = np.random.default_rng(0)
    Z = rng.standard_normal((64, in_dim)).astype(np.float32)
    with torch.no_grad():
        ref = m(torch.tensor(Z)).numpy()
    h = Z
    for li, L in enumerate(layers):
        h = h @ np.array(L["W"], dtype=np.float32).T + np.array(L["b"], dtype=np.float32)
        if li != len(layers) - 1:
            h = np.maximum(h, 0)
    err = float(np.abs(h - ref).max())

    # quality on a fixed hold-out, plus the contrast behaviour that this arm is about
    full = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_full_idx.npy")).astype(np.int64)
    va = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_split.npz"))["val"]
    Xl = full.reshape(len(full), -1).astype(np.float64)
    mu = Xl.mean(0, keepdims=True)
    sub = np.random.default_rng(0).choice(len(Xl), min(60000, len(Xl)), replace=False)
    _, _, Vt = np.linalg.svd(Xl[sub] - mu, full_matrices=False)
    comps = Vt[:22]
    sdv = (((Xl - mu) @ comps.T).std(0, keepdims=True) + 1e-6)
    ZZ = ((Xl - mu) @ comps.T) / sdv[0]
    bv = va[:1000]
    with torch.no_grad():
        out = np.clip(net_eval(m, ZZ[bv][:, :in_dim].astype(np.float32)), 0, 15)
    truth = full[bv][:, layout.POS_ROW, layout.MIRROR_COL].astype(float)
    dec = np.rint(out).astype(int)
    agree = []
    for g in layout.image_from_left_half(dec):
        pairs = same = 0
        for r_ in range(8):
            for c in range(8):
                for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    rr, cc = r_ + dr, c + dc
                    if 0 <= rr < 8 and 0 <= cc < 8:
                        pairs += 1
                        if g[r_, cc] == g[rr, cc]:
                            same += 1
        agree.append(same / pairs)
    # random-draw contrast profile, which is what the sampler produces
    Zr = np.clip(np.random.default_rng(7).standard_normal((4000, in_dim)), -3.5, 3.5).astype(np.float32)
    with torch.no_grad():
        orand = np.clip(net_eval(m, Zr), 0, 15)
    con = orand.max(1) - orand.min(1)
    bri = orand.mean(1)

    doc = {"gen": {"layers": layers},
           "meta": {"in_dim": in_dim, "out_dim": out_dim, "k": in_dim,
                    "multiplies": int(sum(len(L["W"]) * len(L["W"][0]) for L in layers)),
                    "steps": ck.get("args", {}).get("steps"),
                    "mode": "gray32", "output_mode": "continuous32",
                    "post": post,
                    "acc": round(float((dec == truth.astype(int)).mean()), 4),
                    "agree": round(float(np.mean(agree)), 4),
                    "contrast": round(float(con.mean()), 2),
                    "washed": round(float(np.mean(con <= 5)), 4),
                    "bright": round(float(bri.mean()), 3),
                    "note": "trained with --stretch-targets: every training face was "
                            "rescaled to [0,15], so full-range output is baked in"}}
    json.dump(doc, io.open(os.path.join(WEB, out_name), "w", encoding="utf-8"), separators=(",", ":"))
    print(f"  {tag:12s} -> {out_name}  in={in_dim} out={out_dim} "
          f"contrast={doc['meta']['contrast']} washed={doc['meta']['washed']*100:.2f}% "
          f"JS-vs-torch {err:.1e}")
    return doc["meta"]


def net_eval(m, Z):
    with torch.no_grad():
        return m(torch.tensor(Z)).numpy()


meta = export("dim4_stretch", "pca22_dim4st_model.json", "none")
if meta is None:
    raise SystemExit("cannot export dim4_stretch")

# ---------------------------------------------------------------- manifest
arms = []
for key, f, note in (
    ("dim4x", "pca22_dim4x_model.json", "PCA-4 + 显示拉伸（上轮 62.1%）"),
    ("dim4st", "pca22_dim4st_model.json", "PCA-4 训练时烘焙拉伸（免 min/max）"),
    ("dim22", "pca22_dim22_model.json", "PCA-22 原始（参照锚点）"),
):
    d = json.load(open(os.path.join(WEB, f), encoding="utf-8"))
    m = d["meta"]
    arms.append({"key": key, "file": f, "k": m["in_dim"], "in_dim": m["in_dim"],
                 "out_dim": m["out_dim"], "multiplies": m["multiplies"],
                 "acc": m.get("acc"), "agree": m.get("agree"),
                 "post": m.get("post", "none"),
                 "contrast": m.get("contrast"), "washed": m.get("washed"),
                 "note": note})
json.dump(arms, io.open(os.path.join(WEB, "dimsweep_manifest.json"), "w", encoding="utf-8"),
          indent=2, ensure_ascii=False)
print()
print("manifest:")
for a in arms:
    print(f"  {a['key']:7s} post={a['post']:6s} contrast={a['contrast']} "
          f"washed={None if a['washed'] is None else round(a['washed']*100,2)}  {a['note']}")
print("EXPORT_STRETCH_DONE")
