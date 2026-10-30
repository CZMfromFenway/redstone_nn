#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Visualise the learned kernels AND what they actually respond to.

For every kernel we produce
  1. the kernel itself as a colour image (blue = negative, red = positive, white = 0),
     upscaled so a 3x3 filter is legible, with the numeric values printed below it,
  2. the PATCHES of real faces that activate it hardest, so you can see what structure
     it is detecting rather than guessing from the weights.

The second half is the important one: a filter "means" whatever it responds to.
"""
import json, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "results", "kernel_style", "kernel_style")
full = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_full_idx.npy"))
NLEV = 16


class KernelAE(nn.Module):
    def __init__(self, embed):
        super().__init__()
        self.k3 = nn.Conv2d(1, embed, 3, padding=1, bias=False)
        self.k2 = nn.Conv2d(1, embed, 2, padding=0, bias=False)
        self.enc = nn.Sequential(nn.Flatten(), nn.Linear(2 * embed * 64, 32), nn.ReLU())
        self.dec = nn.Sequential(nn.Linear(32, 128), nn.ReLU(), nn.Linear(128, 64))

    def forward(self, x):
        r3 = torch.relu(self.k3(x))
        r2 = torch.relu(self.k2(x))
        return None, None, r3, r2


ck = torch.load(os.path.join(D, "final.pt"), weights_only=False) if os.path.exists(os.path.join(D, "final.pt")) else None
info = json.load(open(os.path.join(D, "kernel_info.json"), encoding="utf-8"))
E = info["embed"]
print(f"embed={E}, code_k={info['code_k']}, params={info['params']}, steps={info['steps']}")
print(f"recon MSE (held-out) = {info['recon_mse']}")

k3 = np.array(info["kernels_3x3"]).reshape(E, 3, 3)
k2 = np.array(info["kernels_2x2"]).reshape(E, 2, 2)


# ---------------------------------------------------------------- 1. kernels as images
def kernel_img(w, scale=48):
    """blue = negative, red = positive, white = 0; scaled by the max |w| over all kernels."""
    vmax = float(np.abs(w).max()) + 1e-9
    h, wd = w.shape
    rgb = np.ones((h, wd, 3), dtype=np.float32)
    pos = np.clip(w / vmax, 0, 1)[..., None]
    neg = np.clip(-w / vmax, 0, 1)[..., None]
    rgb = rgb * (1 - pos) * (1 - neg) + pos * np.array([0.85, 0.15, 0.10]) \
        + neg * np.array([0.10, 0.35, 0.85])
    rgb = np.clip(rgb, 0, 1)
    im = Image.fromarray((rgb * 255).astype(np.uint8))
    return im.resize((wd * scale, h * scale), Image.NEAREST)


VM = max(float(np.abs(k3).max()), float(np.abs(k2).max()))
CELL3, CELL2 = 3 * 48, 2 * 48
PAD = 14
W = max(E * (CELL3 + PAD), E * (CELL2 + PAD)) + PAD
lab_h = 18
H = lab_h + CELL3 + 40 + lab_h + CELL2 + 30
canvas = Image.new("RGB", (W, H), (255, 255, 255))
dr = ImageDraw.Draw(canvas)
y = 4
dr.text((6, y), f"learned 3x3 kernels (blue = negative, red = positive; common scale |w|max={VM:.3f})",
        fill=(0, 0, 0))
y += lab_h
for i in range(E):
    im = kernel_img(k3[i])
    canvas.paste(im, (PAD + i * (CELL3 + PAD), y))
for i in range(E):
    x = PAD + i * (CELL3 + PAD)
    for r in range(3):
        for c in range(3):
            dr.text((x + c * 48 + 8, y + CELL3 + 2 + r * 11), f"{k3[i][r,c]:+.2f}"[:6], fill=(0, 0, 0))
y += CELL3 + 40
dr.text((6, y), "learned 2x2 kernels", fill=(0, 0, 0))
y += lab_h
for i in range(E):
    im = kernel_img(k2[i])
    canvas.paste(im, (PAD + i * (CELL2 + PAD), y))
canvas.save(os.path.join(D, "KERNELS_visual.png"))
print("wrote KERNELS_visual.png")

# ---------------------------------------------------------------- 2. what each kernel responds to
# NOTE: kernel_style.py never saved the model, only the kernel weights in kernel_info.json.
# That is enough: the 3x3 / 2x2 responses depend only on those weights, not on the
# encoder/decoder. So rebuild the two conv layers straight from the JSON.
m = KernelAE(E)
m.k3.weight.data = torch.tensor(k3, dtype=torch.float32).reshape(E, 1, 3, 3)
m.k2.weight.data = torch.tensor(k2, dtype=torch.float32).reshape(E, 1, 2, 2)
print("rebuilt the feature extractor from kernel_info.json (final.pt was never written)")

m = m.float().eval()
N = 20000
idx = np.random.default_rng(0).choice(len(full), N, replace=False)
X = torch.from_numpy(full[idx].astype(np.float32)).unsqueeze(1) / 15.0 * 2 - 1
with torch.no_grad():
    r3 = torch.relu(m.k3(X)).numpy()          # (N,E,8,8)
    r2 = torch.relu(m.k2(X)).numpy()          # (N,E,7,7)

def patch_grid(patches, scale=10, cols=None):
    n = len(patches)
    cols = cols or n
    h, w = patches[0].shape
    rows = (n + cols - 1) // cols
    cv = np.full((rows * h, cols * w), 255, dtype=np.uint8)
    for i, p in enumerate(patches):
        r, c = divmod(i, cols)
        cv[r*h:(r+1)*h, c*w:(c+1)*w] = p
    return Image.fromarray(cv, mode="L").resize((cols*w*scale, rows*h*scale), Image.NEAREST)


# one figure per kernel: kernel image + its top-activating receptive fields
band_h = 3 * 48 + 24
figs = []
print()
print("=== what each kernel responds to (top-8 activating 3x3 patches, shown as the actual image) ===")
for i in range(E):
    a = r3[:, i]                              # (N,8,8)
    flat = a.reshape(len(a), -1)
    top = np.argsort(-flat.max(1))[:8]
    patches = []
    for t in top:
        r, c = divmod(int(flat[t].argmax()), 8)
        r0, c0 = max(0, r-1), max(0, c-1)
        p = np.full((3, 3), 255, dtype=np.uint8)
        sub = full[idx[t], r0:r0+3, c0:c0+3]
        p[:sub.shape[0], :sub.shape[1]] = (sub * 17).astype(np.uint8)
        patches.append(p)
    pg = patch_grid(patches, scale=16, cols=8)
    ki = kernel_img(k3[i], scale=40)
    row = Image.new("RGB", (max(ki.width, pg.width), ki.height + pg.height + 18), (255, 255, 255))
    d2 = ImageDraw.Draw(row)
    d2.text((2, 0), f"k3[{i}]  peak activation {flat.max():.3f}  mean {flat.mean():.3f}", fill=(0, 0, 0))
    row.paste(ki, (0, 16))
    row.paste(pg, (ki.width + 12, 16))
    row.save(os.path.join(D, f"kernel_{i:02d}_what_it_sees.png"))
    figs.append(row)
    print(f"  k3[{i}]: peak {flat.max():.3f}  mean {flat.mean():.3f}  "
          f"sparsity(frac==0) {float((a==0).mean()):.3f}")

# combined sheet
Wc = max(f.width for f in figs)
Hc = sum(f.height + 6 for f in figs)
sheet = Image.new("RGB", (Wc, Hc), (255, 255, 255))
yy = 0
for f in figs:
    sheet.paste(f, (0, yy)); yy += f.height + 6
sheet.save(os.path.join(D, "KERNELS_what_they_see.png"))
print(f"\nwrote kernel_XX_what_it_sees.png (one per kernel) and KERNELS_what_they_see.png")

json.dump({"k3": k3.tolist(), "k2": k2.tolist(),
           "activation_stats": [{"kernel": i,
                                 "peak": float(r3[:, i].max()),
                                 "mean": float(r3[:, i].mean()),
                                 "zero_frac": float((r3[:, i] == 0).mean())}
                                for i in range(E)]},
          open(os.path.join(D, "kernel_visual_stats.json"), "w"), indent=2)
print("VISUAL_DONE")
