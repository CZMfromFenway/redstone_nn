#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Export the chosen colour model to webapp JSON, matching the grayscale export format.

The exported object has the same shape as pca22_dim4x_model.json so the labelling page and any
downstream tool can load it without a special case:

    {"gen": {"layers": [{"W":..., "b":...}, ...]}, "meta": {...}}

The colour stage 2 has TWO inputs (the 32 gray levels and k colour components) and the gray
stage has one. Rather than invent a new schema, the export records both stages in one file:
    gen      the colour network (input: [gray32, colour k])
    stage1   the frozen grayscale generator (input: the 4 PCs)
That way a consumer that only understands "gen" still works, and a consumer that wants the
full colour pipeline reads stage1 first.

The meta records the exact multiplication count so the redstone estimate is not hand-computed.
"""
import argparse
import json
import os
import sys
import torch
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "results", "colour")
WEB = os.path.join(ROOT, "webapp")

A = argparse.ArgumentParser()
A.add_argument("--tag", default="E3")
A.add_argument("--name", default="")
A = A.parse_args()

res = json.load(open(os.path.join(OUT, "stage2_results.json"), encoding="utf-8"))
if A.tag not in res:
    raise SystemExit(f"no result for {A.tag}")
m = torch.load(os.path.join(OUT, f"stage2_{A.tag}.pt"), weights_only=False)

hid = m["hid"]
d_in = m["d_in"]
layers = [
    torch.nn.Linear(d_in, hid),
    torch.nn.Linear(hid, hid),
    torch.nn.Linear(hid, 96),
]
sd = m["state"]
seq = []
i = 0
spec = [(d_in, hid), (hid, hid), (hid, 96)]
for j, (ni, no) in enumerate(spec):
    seq.append({"W": sd[f"{i}.weight"].tolist(), "b": sd[f"{i}.bias"].tolist()})
    i += 2

g = json.load(open(os.path.join(WEB, "pca22_dim4x_model.json"), encoding="utf-8"))

mults = d_in * hid + hid * hid + hid * 96
grey_mults = g["meta"]["multiplies"]
meta = {
    "in_dim": int(d_in),
    "out_dim": 96,
    "k_colour": int(m["k"]),
    "scheme": m["scheme"],
    "hid": int(hid),
    "multiplies": int(mults),
    "multiplies_total": int(mults + grey_mults),
    "rgb_mae": res[A.tag]["rgb_mae"],
    "hue_err": res[A.tag]["hue_err"],
    "sat_err": res[A.tag]["sat_err"],
    "gray_consistency": res[A.tag]["gray_consistency"],
    "quant_floor": 0.2049,
    "output_mode": "rgb96_4bit",
    "note": res[A.tag]["note"],
    "layout": "colour values are 32 positions x 3 channels, canonical order p=col*8+row, "
              "mirrored within each row; colour is 98.7% mirror-symmetric",
    "usage": "stage1(4 PCs) -> gray32 (clipped, per-face min-max stretched, rounded); "
             "then gen([gray32/15, colour_pca_k]) -> 96 values in 0..15",
}
out = {"gen": {"layers": seq}, "stage1": g["gen"], "meta": meta}
name = A.name or f"pca22_colour_{A.tag.lower()}_model.json"
p = os.path.join(WEB, name)
json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False)

print(f"wrote {p}  ({os.path.getsize(p)/1e6:.1f} MB)")
print()
print(f"  stage 2 : {d_in} -> {hid} -> {hid} -> 96    {mults:,} multiplies")
print(f"  stage 1 : 4 -> 256 -> 256 -> 32    {grey_mults:,} multiplies (already deployed)")
print(f"  total   : {mults+grey_mults:,} multiplies")
print()
for k in ("rgb_mae", "hue_err", "sat_err", "gray_consistency"):
    print(f"  {k:18s} {meta[k]:.4f}")
print(f"  quant floor        {meta['quant_floor']:.4f}")
print("EXPORT_COLOUR_DONE")
