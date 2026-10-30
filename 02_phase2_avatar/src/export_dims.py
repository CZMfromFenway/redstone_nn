#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Export the dimension-ladder checkpoints to the JSON the labelling page consumes.

The page's contract (read from label.html) is:
    { "gen": { "layers": [ {"W": [[...in_dim], ...out_dim], "b": [...]}, ... ] },
      "meta": {...} }
where W is OUTPUT-major (PyTorch's native layout, no transpose) and a ReLU sits between
consecutive layers with none on the last. A previous bug transposed this and turned a
256x32 layer into a 32x256 one, so the layout is asserted here rather than assumed.

Also writes a manifest that label.html can use to register the models and to know each
one's input dimension.
"""
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
SRC = os.path.join(ROOT, "results", "dimsweep")
WEB = os.path.join(ROOT, "webapp")
NLEV = 16
LEVELS = np.linspace(0, 255, 16)


def load_state(path):
    ck = torch.load(path, weights_only=False)
    st = ck.get("G") or ck.get("S")
    return {(k.replace("net.", "", 1) if k.startswith("net.") else k): v
            for k, v in st.items()}, ck.get("args", {})


def to_net(st):
    idx = sorted({int(k.split(".")[0]) for k in st if k.endswith(".weight")})
    seq = []
    for j in idx:
        W = st[f"{j}.weight"]
        seq.append(nn.Linear(int(W.shape[1]), int(W.shape[0])))
        if j != idx[-1]:
            seq.append(nn.ReLU())
    m = nn.Sequential(*seq)
    m.load_state_dict(st)
    m.eval()
    return m, idx


def export_dim(k):
    path = os.path.join(SRC, f"dim{k}", "final.pt")
    if not os.path.exists(path):
        print(f"  dim{k}: checkpoint missing, skipped")
        return None
    st, args = load_state(path)
    net, idx = to_net(st)
    layers = []
    for j in idx:
        W = st[f"{j}.weight"].numpy()          # (out, in) -- output major, keep as is
        b = st[f"{j}.bias"].numpy()
        assert W.shape[0] == b.shape[0], "bias length mismatch"
        layers.append({"W": [[float(v) for v in row] for row in W],
                       "b": [float(v) for v in b]})
    in_dim = len(layers[0]["W"][0])
    out_dim = len(layers[-1]["W"])
    assert in_dim == k, f"first layer expects {in_dim}, expected {k}"
    assert out_dim == 32, f"last layer emits {out_dim}, expected 32"

    # verify the JS forward from the emitted weights matches torch
    rng = np.random.default_rng(0)
    Z = rng.standard_normal((64, in_dim)).astype(np.float32)
    with torch.no_grad():
        ref = net(torch.tensor(Z)).numpy()
    h = Z
    for li, L in enumerate(layers):
        Wm = np.array(L["W"], dtype=np.float32)
        bm = np.array(L["b"], dtype=np.float32)
        h = h @ Wm.T + bm
        if li != len(layers) - 1:
            h = np.maximum(h, 0)
    err = float(np.abs(h - ref).max())

    # quality on a fixed hold-out, for the manifest.
    # NOTE: the condition and the target must come from the SAME array (L0_full_idx).
    # Building the PCA from the symmetry-filtered RGB instead produced conditions paired
    # with a different face's target; see docs/ALIGNMENT_BUG.md.
    import importlib.util as _ilu
    spec = _ilu.spec_from_file_location("layout", os.path.join(HERE, "layout.py"))
    layout = _ilu.module_from_spec(spec); spec.loader.exec_module(layout)
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
        out = net(torch.tensor(ZZ[bv][:, :k].astype(np.float32))).clamp(0, 15).round().numpy()
    truth = full[bv][:, layout.POS_ROW, layout.MIRROR_COL]
    acc = float((out.astype(int) == truth).mean())
    imgs = layout.image_from_left_half(out.astype(int))
    agree = []
    for g in imgs:
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

    doc = {"gen": {"layers": layers},
           "meta": {"in_dim": in_dim, "out_dim": out_dim, "k": k,
                    "multiplies": int(sum(len(L["W"]) * len(L["W"][0]) for L in layers)),
                    "steps": args.get("steps"), "lam": args.get("lam"),
                    "acc": round(acc, 4), "agree": round(float(np.mean(agree)), 4),
                    # label.html keys off meta.output_mode, NOT meta.mode. Omitting it
                    # makes OUT_MODE fall back to logits_gray, and decode() then reads
                    # raw[p*16+c] up to index 511 on a 32-long array, so every position
                    # past the first two becomes level 0 -- an all-black face with two
                    # bright corners. This is the exact failure the page showed.
                    "mode": "gray32", "output_mode": "continuous32",
                    "note": "dimension ladder, same recipe as s32_tv5"}}
    name = f"pca22_dim{k}_model.json"
    json.dump(doc, open(os.path.join(WEB, name), "w"), separators=(",", ":"))
    print(f"  dim{k:2d}: {name}  in={in_dim} out={out_dim} "
          f"mult={doc['meta']['multiplies']:,} acc={acc:.4f} "
          f"agree={np.mean(agree)*100:.1f}%  JS-vs-torch max err {err:.2e}")
    return doc["meta"] | {"file": name, "key": f"dim{k}"}


manifest = []
for k in (4, 8, 22):
    r = export_dim(k)
    if r:
        manifest.append(r)

json.dump(manifest, io.open(os.path.join(WEB, "dimsweep_manifest.json"), "w", encoding="utf-8"),
          indent=2, ensure_ascii=False)
print(f"\nwrote {os.path.join(WEB, 'dimsweep_manifest.json')}  ({len(manifest)} models)")
print("EXPORT_DIMS_DONE")
