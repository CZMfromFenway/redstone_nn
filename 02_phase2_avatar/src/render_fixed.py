#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Render the L0 generator's TRUE output directly from the checkpoint, with correct
spatial layout, and measure it. This settles the "black lower half / asymmetric"
questions by measurement rather than by looking at a picture.

Spatial layout used here (and verified below):
    generator emits 32 values = rows 0..3, each row 8 columns (0..7)
    position p of the 32 maps to (row = p // 8, col = p % 8)
    the full 8x8 image mirrors WITHIN each row: img[r, 7-c] = img[r, c]
    -> nothing ever writes rows 4..7, so a full 8x8 layout CANNOT come from this
       generator. Any picture showing 8 rows of content was produced by a different
       layout convention (that is exactly the bug being hunted).
"""
import os, json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
G = os.path.join(ROOT, "results", "gan_v2")
LEVELS = np.linspace(0, 255, 16)

# --- import the generator definition without running argparse -----------------
import importlib.util
spec = importlib.util.spec_from_file_location("gl", os.path.join(ROOT, "src", "gan_L0.py"))
gl = importlib.util.module_from_spec(spec)
import sys
sys.argv = ["gan_L0.py", "--tag", "import_only"]     # argparse reads argv, give it something valid
spec.loader.exec_module(gl)

TAG = "L0_zeroint_test"          # L0_full_s0 was killed mid-run and has no checkpoint
ck = torch.load(os.path.join(G, TAG, "final.pt"), weights_only=False)
a = ck["args"]
Gm = gl.Gen(a["zdim"], a["g_hidden"], zero_init=a["g_zero_init"])
Gm.load_state_dict(ck["G"]); Gm.eval()
print(f"generator from {TAG}: zdim={a['zdim']} hid={a['g_hidden']}  steps={a['steps']}  "
      f"zero_init={a['g_zero_init']}")

with torch.no_grad():
    z = torch.randn(32, a["zdim"])
    lg = Gm.logits(z)
    print(f"logits shape {tuple(lg.shape)}  -> 32 positions x 16 levels")
    probs = F.softmax(lg, -1)
    idx = F.gumbel_softmax(lg, tau=1.0, hard=True).argmax(-1).numpy()   # (32,32)

print(f"generated index array {idx.shape}")
print(f"  per-sample distinct levels: mean {np.mean([len(np.unique(r)) for r in idx]):.2f}")
# is the output the same for every z? (a degenerate generator ignores z)
print(f"  std across samples, averaged over positions: {idx.astype(float).std(0).mean():.3f}")
print(f"  entropy of the per-position marginal (nats): "
      f"{float(-(np.bincount(idx.reshape(-1), minlength=16)/idx.size * np.log(np.clip(np.bincount(idx.reshape(-1), minlength=16)/idx.size,1e-12,1))).sum()):.3f}")


def to_full_correct(idx32):
    """(n,32) -> (n,8,8). Positions p=row*8+col for rows 0..3, cols 0..7.
    Mirror WITHIN the row. Rows 4..7 are never written."""
    n = len(idx32)
    img = np.zeros((n, 8, 8), dtype=np.uint8)
    for p in range(32):
        r, c = divmod(p, 8)
        img[:, r, c] = idx32[:, p]
        img[:, r, 7 - c] = idx32[:, p]
    return img


full = to_full_correct(idx)
print()
print("=== layout facts ===")
nonzero_rows = sorted({r for r in range(8) if (full[:, r, :] != 0).any()})
print(f"  rows containing any non-zero value: {nonzero_rows}")
print(f"  rows that are ALL zero: {[r for r in range(8) if (full[:, r, :] == 0).all()]}")
print(f"  rows 4..7 all-zero count: {int((full[:, 4:, :] == 0).all(axis=(1,2)).sum())}/{len(full)}")
sym = float((full[:, :, :4] == full[:, :, 4:][:, :, ::-1]).mean())
print(f"  mirror symmetry (left vs reversed right): {sym:.4f}  <- must be 1.0000")

# also check: is the top-vs-bottom brightness asymmetric?
print()
print("=== per-row ink (fraction of pixels at level 0 == darkest) ===")
for r in range(8):
    z0 = float((full[:, r, :] == 0).mean())
    print(f"    row {r}: mean level {full[:, r, :].mean():5.2f}   frac at level 0: {z0*100:5.1f}%")

# --- render with the CORRECT layout, 2 rows of 16 ---
def grid(imgs, cols=16, sc=8):
    rows_ = (len(imgs) + cols - 1) // cols
    cv = np.full((rows_ * 8, cols * 8), 40, dtype=np.uint8)
    for i, im in enumerate(imgs):
        r_, c_ = divmod(i, cols)
        cv[r_*8:(r_+1)*8, c_*8:(c_+1)*8] = np.clip(np.asarray(im), 0, 255).astype(np.uint8)
    big = np.kron(cv, np.ones((sc, sc), dtype=np.uint8))
    for r_ in range(rows_ + 1):
        big[r_*8*sc:r_*8*sc+2, :] = 200
    for c_ in range(cols + 1):
        big[:, c_*8*sc:c_*8*sc+2] = 200
    return Image.fromarray(big, mode="L").convert("RGB")

outp = os.path.join(G, TAG, "RENDER_FIXED_generated.png")
grid([LEVELS[full[i]] for i in range(32)], 16).save(outp)
print(f"\nwrote {outp}")

# real reference rendered the SAME (correct) way: rows 0..7 from the data
seed_full = np.load(os.path.join(DATA, "L0_full_idx.npy"))
sp = np.load(os.path.join(DATA, "L0_split.npz"))
va = sp["val"][:32]
outp2 = os.path.join(G, TAG, "RENDER_FIXED_real.png")
grid([LEVELS[seed_full[i]] for i in va], 16).save(outp2)
print(f"wrote {outp2}")

json.dump({"nonzero_rows": nonzero_rows, "symmetry": sym,
           "per_row_level0_frac": [float((full[:, r, :] == 0).mean()) for r in range(8)],
           "std_across_samples": float(idx.astype(float).std(0).mean())},
          open(os.path.join(G, TAG, "render_fix.json"), "w"), indent=2)
print("RENDER_FIX_DONE")

