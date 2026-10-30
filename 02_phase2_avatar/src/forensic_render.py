#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Forensic check of the rendered PNGs: is the apparent asymmetry (and the "black lower
half") a real property of the generated data, or a bug in MY rendering/scrambling code?

We measure directly on the saved sample PNGs and on freshly loaded data, not by eye.

Checks
  1. In the L0 unconditional-failure samples: per-row ink statistics. If rows 4..7 are
     uniformly the background grid colour (200) the "black lower half" is a rendering
     artefact; if they are near 0 they are genuinely black DATA.
  2. Column mapping: does the rendered image match the index array read row-major?
  3. Mirror symmetry measured on the index arrays themselves.
"""
import os
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G = os.path.join(ROOT, "results", "gan_v2")

def load(p):
    if not os.path.exists(p):
        print(f"  MISSING {p}")
        return None
    return np.asarray(Image.open(p).convert("L")).astype(np.int32)

print("=" * 78)
print("1. Failed unconditional samples: what is actually in the lower half?")
print("=" * 78)
for tag, step in (("L0_full_s0", "008000"), ("L0_zeroint_test", "003000"), ("diag_freezeD", "004000")):
    p = os.path.join(G, tag, f"samples_step{step}.png")
    a = load(p)
    if a is None:
        # try any samples_step file
        d = os.path.join(G, tag)
        if os.path.isdir(d):
            fs = sorted(f for f in os.listdir(d) if f.startswith("samples_step"))
            if fs:
                p = os.path.join(d, fs[-1]); a = load(p)
    if a is None:
        continue
    # grid is cols=8 tiles of 8px, scale 8 -> 64 px per tile; 4 rows x 8 cols of tiles
    # tile grid: the samples are laid out 8 per row, so the image is 8*64 wide
    H, W = a.shape
    print(f"\n{tag} ({os.path.basename(p)}) shape {a.shape}")
    # find the tile grid by scanning: tiles are 64x64 with 2px separators every 64
    # instead of guessing, report raw row profiles of the whole canvas
    rowmean = a.mean(1)
    colmean = a.mean(0)
    print(f"  canvas min {a.min()} max {a.max()}  (background/grid line = 200)")
    # count pixels that are exactly the grid colour vs near-black
    print(f"  pixels ==200 (grid): {100*(a==200).mean():5.1f}%   pixels <20 (near black): {100*(a<20).mean():5.1f}%")
    # split the canvas into its 4 tile-rows and compare their ink
    th = H // 4
    for r in range(4):
        blk = a[r*th:(r+1)*th]
        print(f"    tile-row {r}: mean {blk.mean():6.1f}   frac<20 {100*(blk<20).mean():5.1f}%   frac==200 {100*(blk==200).mean():5.1f}%")

print()
print("=" * 78)
print("2. cond_gan rendering: is the right half horizontally flipped?")
print("=" * 78)
p = os.path.join(G, "cond_gan_ce", "cond_REAL.png")
a = load(p)
print(f"cond_REAL.png shape {a.shape}")
# the grid uses cols=16, sc=8 -> 16 tiles of 64px
tile = 64
ntiles_x = a.shape[1] // tile
print(f"tiles across: {ntiles_x}")
# compare each tile against its own horizontal mirror: real faces are ~86% mirror-equal,
# so a FLIPPED right half would show up as a large left-right asymmetry
tl = []
for i in range(min(32, ntiles_x * 4)):
    r, c = divmod(i, ntiles_x)
    t = a[r*tile:(r+1)*tile, c*tile:(c+1)*tile]
    if t.shape != (tile, tile):
        continue
    # downsample to 8x8 by taking every 8th pixel (tiles are 8x upscaled)
    t8 = t[::8, ::8][:8, :8]
    left = t8[:, :4]; right = t8[:, 4:]
    asc = float((left == right[:, ::-1]).mean())          # right reversed -> should match if mirrored
    asc_plain = float((left == right).mean())             # no reversal
    tl.append((asc, asc_plain))
if tl:
    asc = np.mean([x[0] for x in tl]); ascp = np.mean([x[1] for x in tl])
    print(f"  tile left-half vs REVERSED right-half agreement: {asc:.3f}")
    print(f"  tile left-half vs PLAIN    right-half agreement: {ascp:.3f}")
    print("  -> if REVERSED is much higher, the rendering FLIPPED the right half")

print()
print("=" * 78)
print("3. Is the asymmetry a real property of cond_samples_T1 (right half is GENERATED)?")
print("=" * 78)
print("  cond_samples_T1 has a REAL left half (cols 0..3) and a GENERATED right half (cols 4..7).")
print("  So left != right is EXPECTED and correct: the left half is copied from the data and")
print("  the right half is the model's own completion. Real faces are only ~80-86% mirror-equal,")
print("  so a generated completion will not match the observed half exactly.")
print("  The L0 unconditional GAN, by contrast, MIRRORS its output, so it must be exactly symmetric.")

# verify that claim on the L0 gumbel output by rebuilding it
print()
print("=" * 78)
print("4. Rebuild the L0 output from the saved generator and measure real symmetry")
print("=" * 78)
import torch, torch.nn.functional as F
import importlib.util
spec = importlib.util.spec_from_file_location("gl", os.path.join(ROOT, "src", "gan_L0.py"))
gl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gl)          # defines Gen, to_full, etc. (argparse only runs in __main__)

ck = torch.load(os.path.join(G, "L0_full_s0", "final.pt"), weights_only=False)
args = ck["args"]
Gm = gl.Gen(args["zdim"], args["g_hidden"], zero_init=args["g_zero_init"])
Gm.load_state_dict(ck["G"]); Gm.eval()
with torch.no_grad():
    z = torch.randn(256, args["zdim"])
    idx = F.gumbel_softmax(Gm.logits(z), tau=1.0, hard=True).argmax(-1)
a32 = idx.numpy()
# native layout: rows 0..3 (8 each) x cols 0..7; mirror partner of col c is 7-c
rows = np.repeat(np.arange(4), 4); lc = np.array([c for _ in range(4) for c in range(4)]); rc = 7 - lc
li = rows * 8 + lc; ri = rows * 8 + rc
print(f"  native-layout symmetry  (a[:, row*8+c] == a[:, row*8+(7-c)]): {(a32[:,li]==a32[:,ri]).mean():.4f}")
full = gl.to_full(idx).numpy()
sym_full = float((full[:, :, :4] == full[:, :, 4:][:, :, ::-1]).mean())
print(f"  rendered 8x8 symmetry (left half vs reversed right half): {sym_full:.4f}")
written = np.unique(np.flatnonzero((full != 0).any(axis=(0, 2))))
print(f"  rows that contain any non-zero value: {written.tolist()}")
zero_rows = [r for r in range(8) if (full[:, r, :] == 0).all()]
print(f"  rows that are entirely zero (would render as pure black): {zero_rows}")