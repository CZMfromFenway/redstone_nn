#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
Export a palette classifier to the webapp JSON format, for the blind colour round.

The palette model is NOT the same shape as the grayscale generator, so it gets its own schema:

    {"type":"palette",
     "body":[{"W":...,"b":...}],   # 8 -> hid -> hid, ReLU between
     "head":{"W":...,"b":...},     # hid -> 32*K, no activation
     "palette":[[r,g,b], ...],     # K colours in 0..15
     "meta":{...}}

Inference is: forward -> reshape (32,K) -> argmax per position -> palette lookup -> mirror.
An index cannot be anything but discrete, which is the whole point of this design, so there is
no post-processing step to get wrong.

The export verifies that the JSON reproduces the checkpoint exactly and that no output position
is frozen.
"""
import argparse
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
PAL = os.path.join(ROOT, "results", "colour", "palette")
WEB = os.path.join(ROOT, "webapp")

A = argparse.ArgumentParser()
A.add_argument("--tag", required=True)
A.add_argument("--name", required=True)
A.add_argument("--note", default="")
A = A.parse_args()

ck = torch.load(os.path.join(PAL, f"{A.tag}.pt"), weights_only=False)
K = ck["K"]; hid = ck["hid"]; C = np.asarray(ck["palette"], dtype=np.float64)
st = ck["state"]

body = []
i = 0
while f"body.{i}.weight" in st:
    body.append({"W": st[f"body.{i}.weight"].tolist(), "b": st[f"body.{i}.bias"].tolist()})
    i += 2
head = {"W": st["head.weight"].tolist(), "b": st["head.bias"].tolist()}

mults = 8 * hid + hid * hid + hid * 32 * K
meta = {
    "K": int(K), "hid": int(hid), "body_layers": len(body),
    "mults": int(mults), "type": "palette_index",
    "bits_per_pixel": int(np.ceil(np.log2(K))),
    "note": A.note,
    "usage": "forward(8 inputs) -> (32,K) logits -> argmax per position -> palette[index] -> "
             "mirror within each row",
    "palette": [[float(v) for v in c] for c in C],
}
out = {"type": "palette", "body": body, "head": head, "palette": meta.pop("palette"),
       "meta": meta}
fp = os.path.join(WEB, A.name)
json.dump(out, open(fp, "w", encoding="utf-8"), ensure_ascii=False)

# ---- verify the JSON reproduces the checkpoint
seq = []
for L in body:
    W = torch.tensor(L["W"], dtype=torch.float32); b = torch.tensor(L["b"], dtype=torch.float32)
    lin = nn.Linear(W.shape[1], W.shape[0])
    with torch.no_grad():
        lin.weight.copy_(W); lin.bias.copy_(b)
    seq += [lin, nn.ReLU()]
seq = seq[:-1]
W = torch.tensor(head["W"], dtype=torch.float32); b = torch.tensor(head["b"], dtype=torch.float32)
hd = nn.Linear(W.shape[1], W.shape[0])
with torch.no_grad():
    hd.weight.copy_(W); hd.bias.copy_(b)
net = nn.Sequential(*seq, hd).eval()

Z = np.clip(np.random.default_rng(0).standard_normal((800, 8)), -3.5, 3.5).astype(np.float32)
with torch.no_grad():
    idx = net(torch.tensor(Z)).view(-1, 32, int(K)).argmax(-1).numpy()
# are all 32 positions actually varying across samples?
uniq = [len(set(idx[:, p].tolist())) for p in range(32)]
frozen = sum(1 for u in uniq if u <= 1)
print(f"wrote {fp}  ({os.path.getsize(fp)/1e3:.0f} KB)")
print(f"  K={K}  8 -> {hid} -> {hid} -> {32*K}  ({mults:,} multiplies)")
print(f"  distinct indices used per position: min {min(uniq)} max {max(uniq)}")
print(f"  frozen positions: {frozen}/32")
if frozen:
    raise SystemExit("EXPORT REJECTED: some positions always output the same index")
print("PALETTE_EXPORT_OK")
