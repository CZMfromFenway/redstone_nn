#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
C1 and C2: two attempts to raise colour yield by changing the model rather than stacking filters.

WHY THE LOSS SHAPE MATTERS HERE
The colour model is a palette classifier: for each of the 32 positions it outputs K logits, and
the face is the argmax colour. Cross-entropy against the nearest palette colour only asks "is
each position the right colour on average" -- it never asks "does the result read as a face".
That is exactly the failure the labels showed: flat fields with no eyes.

C1 -- STRUCTURE REGULARISER
Both quantities can be computed differentiably from the expected colour, so they can be added
straight to the loss:
    row spread    the standard deviation of the eight row-mean luminances. Flat faces score ~0.
    eye contrast  mean skin luminance minus the luminance at the inner eye band.
A soft minimum is used for the eye so gradients flow.

C2 -- ADVERSARIAL
A small discriminator sees the expected colour image and is trained to tell real avatars from
generated ones. The generator is pushed to fool it. This is the request, but note a previous
adversarial attempt in this project failed badly (val MSE 8.58 at adv=3 against 1.35 for pure
MSE), so it is run with a small weight and reported honestly even if it loses.

EVALUATION
No new human labels are available, so the models are compared on the structure statistics the
labels showed to matter, plus the predicted accept rate from the feature probe fitted on all
573 colour labels. The proxy is imperfect; the direction is what matters.
"""
import argparse
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
OUT = os.path.join(ROOT, "results", "colour", "yield")
WEB = os.path.join(ROOT, "webapp")
os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(8)

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("layout", os.path.join(HERE, "layout.py"))
layout = _ilu.module_from_spec(_spec); _spec.loader.exec_module(layout)
LUM = np.array([0.299, 0.587, 0.114], dtype=np.float32)
LUMT = torch.tensor(LUM)

ROW = torch.tensor([p % 8 for p in range(32)])
COL = torch.tensor([p // 8 for p in range(32)])
EYE = torch.tensor([c * 8 + r for r in (3, 4) for c in (1, 2)])
SKIN = torch.tensor([c * 8 + r for r in (5, 6) for c in range(4)])

A = argparse.ArgumentParser()
A.add_argument("--var", default="c1")
A.add_argument("--lam", type=float, default=1.0)
A.add_argument("--lam-eye", type=float, default=1.0)
A.add_argument("--lam-adv", type=float, default=0.3)
A.add_argument("--steps", type=int, default=12000)
A.add_argument("--seed", type=int, default=0)
A.add_argument("--target-row", type=float, default=3.0)
A.add_argument("--target-eye", type=float, default=5.5)
A = A.parse_args()

z = np.load(os.path.join(ROOT, "results", "colour", "cache.npz"))
rgb96 = z["rgb96"].astype(np.float64)
cpc = z["cpc"]
N = len(rgb96)
sp = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_split.npz"))
tr, va = sp["train"], sp["val"]
POS = rgb96.reshape(N, 32, 3)
CT = torch.tensor(cpc[:, :8])
POS_T = torch.tensor(POS.astype(np.float32))

palj = json.load(open(os.path.join(WEB, "colour_pal8_model.json"), encoding="utf-8"))
PAL = np.array(palj["palette"], dtype=np.float64)
K = len(PAL)
PAL_FLAT = np.array([c for c in PAL], dtype=np.float32)
PAL_T = torch.tensor(PAL_FLAT)

# index targets, in RGB space, against this palette
dd = ((POS[:, :, None, :] - PAL[None, None, :, :]) ** 2).sum(-1)
YIDX = torch.tensor(dd.argmin(-1).reshape(N, 32), dtype=torch.long)


def structure_terms(logits, pal=PAL_T):
    """Differentiable row spread and eye contrast from the expected colour."""
    p = F.softmax(logits, -1)                       # (B,32,K)
    exp_c = p @ pal                                  # (B,32,3)
    lum = exp_c @ LUMT                               # (B,32)
    rows = torch.stack([lum[:, ROW == r].mean(1) for r in range(8)], 1)   # (B,8)
    row_spread = rows.std(1)                          # (B,)
    skin = lum[:, SKIN].mean(1)
    # soft minimum over the eye band so the gradient is not sparse
    eye_logits = -lum[:, EYE] * 2.0
    eye_min = (lum[:, EYE] * F.softmax(eye_logits, -1)).sum(1)
    eye_contrast = skin - eye_min
    return row_spread, eye_contrast, exp_c


def build():
    hid = 128
    torch.manual_seed(A.seed)
    return nn.Sequential(nn.Linear(8, hid), nn.ReLU(), nn.Linear(hid, hid), nn.ReLU(),
                         nn.Linear(hid, 32 * K))


class Disc(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(96, 128), nn.LeakyReLU(0.2),
                                 nn.Linear(128, 64), nn.LeakyReLU(0.2), nn.Linear(64, 1))
    def forward(self, x):
        return self.net(x)


net = build()
opt = torch.optim.Adam(net.parameters(), lr=3e-4)
disc = Disc() if A.var == "c2" else None
dopt = torch.optim.Adam(disc.parameters(), lr=3e-4) if disc else None

tr_t = torch.tensor(tr, dtype=torch.long)
rng = np.random.default_rng(A.seed)
print(f"training {A.var}  lam={A.lam} lam_eye={A.lam_eye} lam_adv={A.lam_adv} "
      f"steps={A.steps}")
for step in range(A.steps):
    bi = tr_t[torch.from_numpy(rng.integers(0, len(tr_t), 256))]
    lg = net(CT[bi]).view(-1, 32, K)
    loss = F.cross_entropy(lg.reshape(-1, K), YIDX[bi].reshape(-1))
    row_spread, eye_contrast, exp_c = structure_terms(lg)
    if A.var == "c1":
        loss = loss + A.lam * F.relu(A.target_row - row_spread).mean() \
                    + A.lam_eye * F.relu(A.target_eye - eye_contrast).mean()
    elif A.var == "c2":
        # discriminator sees the expected colour image, real avatars from the dataset
        real = POS_T[bi].reshape(-1, 96)
        fake = exp_c.reshape(-1, 96)
        d_real = disc(real); d_fake = disc(fake.detach())
        d_loss = F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real)) \
               + F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake))
        dopt.zero_grad(); d_loss.backward(); dopt.step()
        adv = F.binary_cross_entropy_with_logits(disc(fake), torch.ones(len(fake), 1))
        loss = loss + A.lam_adv * adv
    opt.zero_grad(); loss.backward(); opt.step()
    if (step + 1) % 4000 == 0:
        with torch.no_grad():
            print(f"  step {step+1:5d}  loss {loss.item():.4f}  "
                  f"row_spread {row_spread.mean():.2f}  eye_contrast {eye_contrast.mean():.2f}")

# ---------------------------------------------------------------- evaluate on held-out seeds
net.eval()
M = 6000
Zr = np.clip(np.random.default_rng(999).standard_normal((M, 8)), -3.5, 3.5).astype(np.float32)
with torch.no_grad():
    idx = net(torch.tensor(Zr)).view(-1, 32, K).argmax(-1).numpy()
faces = PAL[idx].reshape(M, 96)
AF = np.clip(faces, 0, 15).reshape(M, 32, 3)
LUMP = AF @ LUM
rows = np.stack([LUMP[:, [c * 8 + r for c in range(4)]].mean(1) for r in range(8)], 1)
row_spread = rows.std(1)
eye_min = LUMP[:, [c * 8 + r for r in (3, 4) for c in (1, 2)]].min(1)
skin_med = np.median(LUMP[:, [c * 8 + r for r in (5, 6) for c in range(4)]], 1)
eye_contrast = skin_med - eye_min

img = np.zeros((M, 8, 8, 3))
img[:, layout.POS_ROW, layout.POS_COL] = AF
img[:, layout.POS_ROW, layout.MIRROR_COL] = AF
dh = np.linalg.norm(img[:, :, 1:] - img[:, :, :-1], axis=-1)
dv = np.linalg.norm(img[:, 1:] - img[:, :-1], axis=-1)
edge_p90 = np.percentile(np.concatenate([dh.reshape(M, -1), dv.reshape(M, -1)], 1), 90, axis=1)
ncol = np.array([len({tuple(np.rint(c).astype(int)) for c in AF[i]}) for i in range(M)])

print()
print("=" * 100)
print(f"RESULT  {A.var}")
print("=" * 100)
print(f"  行亮度方差 {row_spread.mean():.2f}   眼睛对比 {eye_contrast.mean():.2f}   "
      f"边缘p90 {edge_p90.mean():.2f}   颜色数 {ncol.mean():.2f}")

# predicted accept rate from a probe fitted on the labelled features
lab = [json.loads(l) for l in open(os.path.join(ROOT, "results", "preference", "labels.jsonl"),
                                   encoding="utf-8") if l.strip()]
col = [r for r in lab if str(r.get("round") or "") in ("8", "11")
       and len(r.get("predicted_levels") or []) == 96 and r.get("scheme") == "style"]
FL = np.array([r["predicted_levels"] for r in col], dtype=np.float64)
GY = np.array([1.0 if r["grade"] >= 2 else 0.0 for r in col])
AL = np.clip(FL, 0, 15).reshape(len(FL), 32, 3)
LUL = AL @ LUM
rowsL = np.stack([LUL[:, [c * 8 + r for c in range(4)]].mean(1) for r in range(8)], 1)
fL = np.c_[rowsL.std(1),
           np.median(LUL[:, [c * 8 + r for r in (5, 6) for c in range(4)]], 1)
           - LUL[:, [c * 8 + r for r in (3, 4) for c in (1, 2)]].min(1),
           np.abs(np.diff(LUL, axis=1)).mean(1),
           [len({tuple(np.rint(c).astype(int)) for c in AL[i]}) for i in range(len(AL))]]
fM = np.c_[row_spread, eye_contrast, np.abs(np.diff(LUMP, axis=1)).mean(1), ncol]
mu = fL.mean(0); sd = fL.std(0) + 1e-9
Am_ = np.c_[(fL - mu) / sd, np.ones(len(fL))]
w = np.linalg.solve(Am_.T @ Am_ + 3 * np.eye(Am_.shape[1]), Am_.T @ GY)
s = np.c_[(fM - mu) / sd, np.ones(len(fM))] @ w
thr = np.quantile(np.c_[(fL - mu) / sd, np.ones(len(fL))] @ w, 1 - GY.mean())
pred = float((s > thr).mean())
print(f"  预测可接受率（结构探针，仅作相对比较）{100*pred:.1f}%")

res = {"var": A.var, "lam": A.lam, "lam_eye": A.lam_eye, "lam_adv": A.lam_adv,
       "row_spread": float(row_spread.mean()), "eye_contrast": float(eye_contrast.mean()),
       "edge_p90": float(edge_p90.mean()), "ncol": float(ncol.mean()),
       "predicted_accept": pred}
json.dump(res, open(os.path.join(OUT, f"train_{A.var}_{A.lam}_{A.seed}.json"), "w",
                    encoding="utf-8"), indent=2, ensure_ascii=False)

# keep the weights so the best variant can be exported later
torch.save({"G": net.state_dict(), "palette": PAL.tolist(), "args": vars(A)},
           os.path.join(OUT, f"ck_{A.var}_{A.lam}_{A.seed}.pt"))
print(f"  saved {os.path.join(OUT, f'ck_{A.var}_{A.lam}_{A.seed}.pt')}")
print("TRAIN_DONE")
