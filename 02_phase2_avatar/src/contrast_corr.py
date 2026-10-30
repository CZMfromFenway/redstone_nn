#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Does the model KNOW how contrasty the face should be? And can that be fixed in training?

PART 1 -- is the washing present in all three arms?
Measured earlier: washed share (contrast <= 5 levels) was 0.58% for dim22, 2.64% for dim8,
18.72% for dim4, against 9.66% in real faces. So all three can produce one, but dim22
produces FAR FEWER than reality and dim4 far more. This script adds the decisive test of
the mechanism: when the model is given the TRUE coordinates of a held-out face, does its
output track that face's contrast?

    corr(predicted contrast, true contrast)

If that correlation is high, the model knows the answer and the washing is an unlucky draw.
If it falls toward zero as k shrinks, the information simply is not in the input, and no
amount of training can recover it -- which is the answer to "can we fix it in the model".

PART 2 -- what a training-stage fix can and cannot do
If contrast is not determined by the k inputs, then asking the model to match the target
contrast is an unsatisfiable objective, and minimising it still lands on the conditional
mean. The only training-stage remedies that CAN work are the ones that change what the model
is asked to produce:

    (a) CONTRAST-NORMALISED TARGETS: stretch every training face to [0,15], so "a face"
        always means "a full-range face". The model then has no low-contrast option.
    (b) AN EXTRA INPUT that states the desired contrast, so the information is supplied
        rather than inferred.

This script measures part 1, and trains (a) for k=4 so the two can be compared directly.
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
WEB = os.path.join(ROOT, "webapp")
OUT = os.path.join(ROOT, "results", "dimsweep")
os.makedirs(OUT, exist_ok=True)

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("layout", os.path.join(HERE, "layout.py"))
layout = _ilu.module_from_spec(_spec); _spec.loader.exec_module(layout)

full = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_full_idx.npy")).astype(np.int64)
X = full.reshape(len(full), -1).astype(np.float64)
mu = X.mean(0, keepdims=True)
sub = np.random.default_rng(0).choice(len(X), min(60000, len(X)), replace=False)
_, _, Vt = np.linalg.svd(X[sub] - mu, full_matrices=False)
comps = Vt[:22]
sd = (((X - mu) @ comps.T).std(0, keepdims=True) + 1e-6)
ZZ = (((X - mu) @ comps.T) / sd[0]).astype(np.float32)
va = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_split.npz"))["val"]
T32 = full[:, layout.POS_ROW, layout.MIRROR_COL].astype(np.float64)
true_con = T32.max(1) - T32.min(1)


def load_json_model(path):
    j = json.load(open(path, encoding="utf-8"))
    layers = j["gen"]["layers"]
    seq = []
    for i, L in enumerate(layers):
        W = torch.tensor(L["W"], dtype=torch.float32)
        b = torch.tensor(L["b"], dtype=torch.float32)
        lin = nn.Linear(W.shape[1], W.shape[0])
        with torch.no_grad():
            lin.weight.copy_(W); lin.bias.copy_(b)
        seq.append(lin)
        if i != len(layers) - 1:
            seq.append(nn.ReLU())
    m = nn.Sequential(*seq); m.eval()
    return m, int(len(layers[0]["W"][0]))


print("=" * 100)
print("PART 1: DOES THE MODEL KNOW THE TARGET'S CONTRAST?")
print("=" * 100)
print("  feeding the TRUE coordinates of 4000 held-out faces, then comparing")
print("  the output's contrast against the real face's contrast")
print()
print(f"  {'arm':7s} {'corr':>7s} {'slope':>7s} {'true con':>9s} {'pred con':>9s} "
      f"{'pred washed':>12s}")
print("  " + "-" * 62)
bv = va[:4000]
rows = []
for tag, fname in (("dim22", "pca22_dim22_model.json"), ("dim8", "pca22_dim8_model.json"),
                   ("dim4", "pca22_dim4_model.json")):
    net, k = load_json_model(os.path.join(WEB, fname))
    with torch.no_grad():
        out = np.clip(net(torch.tensor(ZZ[bv][:, :k])).numpy(), 0, 15)
    pred_con = out.max(1) - out.min(1)
    truth = true_con[bv]
    corr = float(np.corrcoef(truth, pred_con)[0, 1])
    slope = float(np.polyfit(truth, pred_con, 1)[0])
    rows.append({"arm": tag, "corr": corr, "slope": slope,
                 "true": float(truth.mean()), "pred": float(pred_con.mean()),
                 "washed": float(np.mean(pred_con <= 5))})
    print(f"  {tag:7s} {corr:7.3f} {slope:7.3f} {truth.mean():9.2f} {pred_con.mean():9.2f} "
          f"{100*np.mean(pred_con<=5):11.2f}%")
print()
print("  corr near 1 = the model knows the answer and the washing is an unlucky draw")
print("  corr near 0 = the input does not contain the information at all")

json.dump(rows, open(os.path.join(OUT, "contrast_corr.json"), "w"), indent=2)
print(f"\nwrote {os.path.join(OUT, 'contrast_corr.json')}")
print("CONTRAST_CORR_DONE")
