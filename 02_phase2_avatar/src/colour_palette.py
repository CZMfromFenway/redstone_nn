#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
Classify each position into a palette index instead of regressing a colour.

WHY THE ARCHITECTURE IS THE PROBLEM
The models emit 96 continuous values. The real art is not continuous: a real face uses about
7 distinct colours and the boundaries between them are hard. Under squared error the best
continuous answer is the conditional mean, which is why every variant so far lands at 17-23
colours and an edge sharpness near 9.5 while the data sits at 7.2 and 15.56. More steps moved
it 9.31 -> 10.05; raising the adversarial weight to 3 blew reconstruction up to MSE 8.7. Neither
is the lever.

So the output changes: predict, for each of the 32 positions, WHICH palette colour it is.
An index is discrete by construction, so boundaries are hard and the number of distinct colours
cannot exceed the palette size, no matter what the weights do.

CONSEQUENCE FOR HARDWARE
An index is also cheaper than three channels: 4 bits per pixel (K=16) instead of 12, plus a
lookup table. The thing that helps the look is the same thing that helps the build.

MEASURED
Reconstruction error WILL look worse, because a palette cannot represent in-between colours.
That is expected and is not the metric that matters here; the user's own criteria are the
number of distinct colours, the sharpness of boundaries, the saturation balance and the
separation between the main colours.
"""
import argparse
import json
import os
import sys
import time
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
OUT = os.path.join(ROOT, "results", "colour")
PAL = os.path.join(OUT, "palette")
os.makedirs(PAL, exist_ok=True)

A = argparse.ArgumentParser()
A.add_argument("--steps", type=int, default=12000)
A.add_argument("--batch", type=int, default=256)
A.add_argument("--hid", type=int, default=128)
A.add_argument("--k", type=int, default=16, help="palette size")
A.add_argument("--threads", type=int, default=8)
A.add_argument("--tag", default="P16")
A.add_argument("--saturate", type=float, default=1.0,
               help="scale each palette colour away from grey; 1.0 = plain k-means")
A = A.parse_args()
torch.set_num_threads(A.threads)

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("layout", os.path.join(HERE, "layout.py"))
layout = _ilu.module_from_spec(_spec); _spec.loader.exec_module(layout)

z = np.load(os.path.join(OUT, "cache.npz"))
rgb96 = z["rgb96"].astype(np.float64)
cpc = z["cpc"]
N = len(rgb96)
sp = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_split.npz"))
tr, va = sp["train"], sp["val"]
K = A.k
print("=" * 100)
print(f"PALETTE CLASSIFIER  tag={A.tag}  K={K}  steps={A.steps}")
print("=" * 100)

# ---------------------------------------------------------------- fit the palette on the data
pos = rgb96.reshape(N, 32, 3)
sample = pos[np.random.default_rng(0).choice(N, 40000, replace=False)].reshape(-1, 3)
rng = np.random.default_rng(1)
C = sample[rng.choice(len(sample), K, replace=False)].copy()
for it in range(40):
    d = ((sample[:, None, :] - C[None, :, :]) ** 2).sum(-1)
    lab = d.argmin(1)
    newC = np.array([sample[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(K)])
    if np.abs(newC - C).max() < 1e-4:
        C = newC
        break
    C = newC
if A.saturate != 1.0:
    # Reach the real data's saturation by moving each colour along its own chroma direction.
    # Luminance and hue are held, only the distance from the grey axis changes, so no new hue is
    # invented -- k-means minimises RGB error and therefore lands on duller colours than the art.
    D = C.mean(1, keepdims=True)
    C = np.clip(D + (C - D) * A.saturate, 0, 15)
    print(f"  palette pushed away from grey by x{A.saturate}: saturations now "
          + ", ".join(f"{float(((c.max()-c.min())/max(c.max(),1e-6))):.2f}" for c in C))
print(f"  palette fitted on 40k pixels, {K} entries; reconstruction error of the real data itself:")
d_all = ((pos[:, :, None, :] - C[None, None, :, :]) ** 2).sum(-1)
idx_real = d_all.argmin(-1)
recon = C[idx_real]
pal_err = float(np.abs(recon - pos).mean())
print(f"    {pal_err:.3f} level units  (a {K}-colour palette cannot do better than this)")

Y = torch.tensor(idx_real.reshape(N, 32), dtype=torch.long)
CT = torch.tensor(cpc[:, :8])


class M(nn.Module):
    def __init__(self, k=8, hid=128, K=16):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(k, hid), nn.ReLU(),
                                  nn.Linear(hid, hid), nn.ReLU())
        self.head = nn.Linear(hid, 32 * K)
        self.K = K

    def forward(self, c):
        return self.head(self.body(c)).view(-1, 32, self.K)


torch.manual_seed(0)
net = M(8, A.hid, K)
opt = torch.optim.Adam(net.parameters(), lr=3e-4)
tr_t = torch.tensor(tr, dtype=torch.long)
va_t = torch.tensor(va, dtype=torch.long)
rng2 = np.random.default_rng(0)
t0 = time.time()
for step in range(A.steps):
    bi = tr_t[torch.from_numpy(rng2.integers(0, len(tr_t), A.batch))]
    logits = net(CT[bi])
    loss = F.cross_entropy(logits.reshape(-1, K), Y[bi].reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()
    if (step + 1) % 3000 == 0:
        with torch.no_grad():
            vl = F.cross_entropy(net(CT[va_t]).reshape(-1, K), Y[va_t].reshape(-1)).item()
            pr = net(CT[va_t]).argmax(-1).numpy()
            acc = float((pr == Y[va_t].numpy()).mean())
        print(f"    {A.tag} step {step+1:6d}  train {loss.item():.4f}  val {vl:.4f}  "
              f"index acc {acc*100:.1f}%  {time.time()-t0:.0f}s")

net.eval()
with torch.no_grad():
    pi = net(CT[va_t]).argmax(-1).numpy()
pred = C[pi].reshape(len(pi), 96)


def metrics(flat):
    a = np.clip(flat, 0, 15).reshape(len(flat), 32, 3)
    img = np.zeros((len(a), 8, 8, 3))
    img[:, layout.POS_ROW, layout.POS_COL] = a
    img[:, layout.POS_ROW, layout.MIRROR_COL] = a
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
    return {"rgb_mae": float(np.abs(a - rgb96[va_t].reshape(-1, 32, 3)).mean()),
            "sat": float(sat.mean()), "sep": float(sep.mean()),
            "edge_p90": float(np.percentile(ed, 90)), "edge_mean": float(ed.mean()),
            "n_colours": float(ncol.mean())}


m = metrics(pred)
print()
print("=" * 100)
print(f"RESULT  {A.tag}")
print("=" * 100)
print(f"  真实数据参照   饱和度 0.315  分离度 6.67  边缘p90 15.56  颜色数 7.2")
print(f"  本模型         饱和度 {m['sat']:.3f}  分离度 {m['sep']:.2f}   "
      f"边缘p90 {m['edge_p90']:.2f}  颜色数 {m['n_colours']:.1f}")
print(f"  RGB MAE {m['rgb_mae']:.3f}（调色板本身的下限是 {pal_err:.3f}）")
m.update({"tag": A.tag, "K": K, "pal_err": pal_err, "palette": C.tolist()})
json.dump(m, open(os.path.join(PAL, f"{A.tag}.json"), "w", encoding="utf-8"),
          indent=2, ensure_ascii=False)
torch.save({"state": net.state_dict(), "K": K, "hid": A.hid, "palette": C}, 
           os.path.join(PAL, f"{A.tag}.pt"))
print("PALETTE_DONE")
