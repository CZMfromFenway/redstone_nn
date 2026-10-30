#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Build the L0 dataset: nearly-symmetric, 16-level grayscale, left-half targets.

Task definition (agreed with the user):
    learn "nearly symmetric 16-level grayscale 8x8 avatars"
    generate "perfectly symmetric 16-level grayscale 8x8 avatars"

The symmetry filter is TASK DEFINITION, not a defect: the target output is exactly
symmetric, so the training set is the nearly-symmetric subpopulation.

Everything is RELATIVE to this script. D: is never touched; no existing file is modified.
The output arrays are cached in data/ and refuse to overwrite.

Outputs (in <root>/data/):
    L0_left_idx.npy    (N,32) uint8    left-half gray levels 0..15  <- generator target
    L0_full_idx.npy    (N,8,8) uint8   full 16-level gray image (real, for the D)
    L0_split.npz       indices of the train / val split
    L0_manifest.json   counts, thresholds, hashes
"""
import argparse, hashlib, json, os
import numpy as np

AP = argparse.ArgumentParser()
AP.add_argument("--min-sym", type=float, default=0.95,
                help="keep avatars whose fraction of equal mirror pairs (in 16-level gray) is >= this")
AP.add_argument("--val-frac", type=float, default=0.03)
AP.add_argument("--max-n", type=int, default=0, help="0 = all")
AP.add_argument("--outdir", default="data_v2_canonical",
                help="directory under 02_phase2_avatar/ for the outputs")
A = AP.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "data")                     # clean RGB source (unaffected by the bug)
DATA = os.path.join(ROOT, A.outdir)
os.makedirs(DATA, exist_ok=True)

# ONE canonical layout. The earlier build put the left half in row-major order
# (p = row*4 + col), which did not match the generator's spatial convention and left
# half the image unwritten downstream. Use the shared module instead.
import importlib.util as _ilu
_lspec = _ilu.spec_from_file_location("layout", os.path.join(HERE, "layout.py"))
layout = _ilu.module_from_spec(_lspec); _lspec.loader.exec_module(layout)

rgb = np.load(os.path.join(SRC, "faces8_rgb_v2.npy"))          # (N,8,8,3) uint8
N = len(rgb)
print(f"clean source: {rgb.shape}  N={N:,}")

LUM = np.array([0.299, 0.587, 0.114])
gray = (rgb.astype(np.float64) * LUM).sum(-1)                   # (N,8,8)
LEVELS = np.linspace(0, 255, 16)
gi = np.abs(gray[:, :, :, None] - LEVELS[None, None, None, :]).argmin(-1).astype(np.uint8)
print(f"16-level gray: quant MSE {((LEVELS[gi]-gray)**2).mean():.2f}, "
      f"RMS {np.sqrt(((LEVELS[gi]-gray)**2).mean()):.2f}/255")

# symmetry filter: within each row, col c (0..3) vs col 7-c
lc = np.array([c for _ in range(8) for c in range(4)])
lr = np.array([r for r in range(8) for _ in range(4)])
rc = 7 - lc
sym_rate = (gi[:, lr, lc] == gi[:, lr, rc]).mean(1)            # (N,)
keep = sym_rate >= A.min_sym
print(f"symmetry filter >= {A.min_sym}: kept {int(keep.sum()):,} / {N:,} "
      f"({100*keep.mean():.2f}%)")
if A.max_n:
    k = np.flatnonzero(keep)[:A.max_n]
    keep = np.zeros(N, dtype=bool); keep[k] = True
    print(f"  capped to {int(keep.sum()):,}")

gi = gi[keep]
n = len(gi)
# canonical left half, p = col*8 + row  -> all 8 rows, first 4 columns
left = layout.left_half_from_image(gi).astype(np.uint8)        # (n,32)
full = gi                                                       # (n,8,8) uint8

# sanity: the canonical left half must rebuild the (symmetrized) image exactly
rebuilt = layout.image_from_left_half(left)
forced = gi.copy()
forced[:, :, 4:] = gi[:, :, :4][:, :, ::-1]
assert (rebuilt == forced).all(), "canonical layout round-trip mismatch"
flip = float((forced != gi).mean())
print(f"forcing perfect mirror flips {100*flip:.3f}% of pixels")
print(f"rows covered by the left half: "
      f"{sorted({r for r in range(8) if (rebuilt[:, r, :] != 0).any()})}  (must be 0..7)")

rng = np.random.default_rng(0)
perm = rng.permutation(n)
nval = int(n * A.val_frac)
val_idx, tr_idx = perm[:nval], perm[nval:]
print(f"split: train {len(tr_idx):,}  val {len(val_idx):,}")

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

outs = {}
for name, arr in (("L0_left_idx.npy", left), ("L0_full_idx.npy", full)):
    p = os.path.join(DATA, name)
    if os.path.exists(p):
        raise SystemExit(f"refusing to overwrite {p}")
    np.save(p, arr)
    outs[name] = {"shape": list(arr.shape), "dtype": str(arr.dtype),
                  "sha256": sha(p), "bytes": os.path.getsize(p)}

p = os.path.join(DATA, "L0_split.npz")
if os.path.exists(p):
    raise SystemExit(f"refusing to overwrite {p}")
np.savez(p, train=tr_idx.astype(np.int64), val=val_idx.astype(np.int64))
outs["L0_split.npz"] = {"train": int(len(tr_idx)), "val": int(len(val_idx)),
                        "sha256": sha(p), "bytes": os.path.getsize(p)}

man = {
    "created_by": "02_phase2_avatar/src/build_L0_data.py",
    "command": " ".join(__import__("sys").argv),
    "source": "data/faces8_rgb_v2.npy (clean, fully opaque, deduplicated)",
    "task": "learn nearly-symmetric 16-level gray 8x8; generate perfectly symmetric ones",
    "layout": {
        "canonical": "left half, p = col*8 + row  (col 0..3, row 0..7)",
        "mirror": "within each row: image[row, 7-col] = image[row, col]",
        "module": "src/layout.py (self-tested)",
        "bug_fixed": "an earlier build used row-major order p = row*4 + col and downstream "
                     "code generated the TOP half, leaving rows 4..7 black",
    },
    "min_symmetry": A.min_sym,
    "lum_weights": LUM.tolist(),
    "levels": LEVELS.tolist(),
    "counts": {"source": int(N), "kept": int(n), "train": int(len(tr_idx)), "val": int(len(val_idx))},
    "mirror_forcing_flip_rate": flip,
    "generator_target": "L0_left_idx.npy  (N,32) canonical left half, 16 classes",
    "discriminator_input": "L0_full_idx.npy (N,8,8) real; fake is the mirrored left half",
    "outputs": outs,
    "outdir": A.outdir,
}
mp = os.path.join(DATA, "L0_manifest.json")
if os.path.exists(mp):
    raise SystemExit(f"refusing to overwrite {mp}")
json.dump(man, open(mp, "w", encoding="utf-8"), indent=2)

print("\n=== wrote ===")
for k, v in outs.items():
    print(f"  {k:22s} {v.get('shape', '')}  {v['bytes']/1e6:.1f} MB  {v['sha256'][:16]}")
print("  L0_manifest.json")
print("BUILD_L0_DONE")
