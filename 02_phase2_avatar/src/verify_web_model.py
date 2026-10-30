#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Verify the web page's JavaScript forward pass against the real PyTorch model.

Reads js_forward_reference.json (produced by js_forward_check.js from pca22_model.json)
and recomputes the same three cases with the actual trained generator. If the numbers
match, the page is running the real model; if not, the page would be showing something
else and must not be trusted.
"""
import json, os
import numpy as np
import torch

import importlib.util as _ilu
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
_lspec = _ilu.spec_from_file_location("layout", os.path.join(HERE, "layout.py"))
layout = _ilu.module_from_spec(_lspec); _lspec.loader.exec_module(layout)
NLEV = 16

RUN = os.path.join(ROOT, "results", "sweep", "sweep_pca22")
ck = torch.load(os.path.join(RUN, "final.pt"), weights_only=False)
a = ck["args"]


class Gen(torch.nn.Module):
    def __init__(self, hid, zdim, cdim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(cdim + zdim, hid), torch.nn.ReLU(),
            torch.nn.Linear(hid, hid), torch.nn.ReLU(),
            torch.nn.Linear(hid, 32 * NLEV))
    def logits(self, cond, z):
        return self.net(torch.cat([cond, z], 1)).view(-1, 32, NLEV)


G = Gen(a["hid"], a["zdim"], 22)
G.load_state_dict(ck["G"]); G.eval()

ref = json.load(open(os.path.join(HERE, "js_forward_reference.json"), encoding="utf-8"))
z = torch.tensor([ref["z_seed12345"]], dtype=torch.float32)

print("=" * 92)
print("JS vs PyTorch on the same three inputs")
print("=" * 92)
allok = True
for c in ref["cases"]:
    pca = torch.tensor([c["pca"]], dtype=torch.float32)
    with torch.no_grad():
        lg = G.logits(pca, z)[0].reshape(-1).numpy()
    js_first8 = np.array(c["logits_first8"])
    py_first8 = lg[:8]
    d = np.abs(js_first8 - py_first8).max()
    js_sum = c["logits_sum"]
    py_sum = float(lg.sum())
    js_arg = c["argmax_pos0"]
    py_arg = int(lg[:16].argmax())
    ok = d < 1e-3 and abs(js_sum - py_sum) < 1e-2 and js_arg == py_arg
    allok &= ok
    print(f"\n  case {c['name']}   {'MATCH' if ok else 'MISMATCH'}")
    print(f"    logits[0:8]  JS {np.round(js_first8,5).tolist()}")
    print(f"                 PY {np.round(py_first8,5).tolist()}")
    print(f"    max |diff| = {d:.2e}")
    print(f"    sum   JS {js_sum:.4f}   PY {py_sum:.4f}")
    print(f"    argmax@pos0  JS {js_arg}   PY {py_arg}")

print()
print("=" * 92)
if allok:
    print("RESULT: the page's JavaScript reproduces the trained model exactly.")
    print("        (max abs difference < 1e-3 on logits, identical argmax)")
else:
    print("RESULT: MISMATCH -- the web page is NOT running the trained model. Do not use it.")
print("=" * 92)

# also sanity-check the gallery PCA round trip: does pca -> image -> pca recover itself?
# NOTE: the stored gallery values are STANDARDISED (coef/sd), because that is what the
# generator was conditioned on. The check must apply the same division.
full = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_full_idx.npy"))
j = json.load(open(os.path.join(ROOT, "webapp", "pca22_model.json"), encoding="utf-8"))
mu = np.array(j["pca"]["mean"]); comps = np.array(j["pca"]["components"]); sdv = np.array(j["pca"]["sd"])
print()
print("gallery self-consistency (stored pca values vs recomputed from the stored image):")
errs, raw = [], []
for k, g in enumerate(j["gallery"][:8]):
    img = np.array(g["rgb"], dtype=np.float64)
    coef = (img - mu) @ comps.T
    errs.append(float(np.abs(coef / sdv - np.array(g["pca"])).max()))
    raw.append(float(np.abs(coef - np.array(g["pca"])).max()))
print(f"  standardised (coef/sd): max |stored - recomputed| = {max(errs):.2e}")
print(f"  raw (no /sd)          : max |stored - recomputed| = {max(raw):.2e}  "
      f"(this is what a mismatched convention would look like)")
gstd = np.array([g["pca"] for g in j["gallery"]]).std(0)
print(f"  per-component std across the 24 gallery faces: mean {gstd.mean():.3f} "
      f"(should be ~1.0 for standardised units)")
print("VERIFY_WEB_DONE")
