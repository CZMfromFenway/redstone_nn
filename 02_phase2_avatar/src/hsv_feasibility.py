#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Before building an HSV model, measure whether the split can actually pay off.

The user's plan is (1) predict luminance, hue and saturation as three channels, and
(2) feed the luminance model's 32 outputs into the hue/saturation models, with the goal
of a COLOUR generator whose output dimension is as small as possible.

Whether step 2 saves anything depends on one number: how much of (hue, saturation) is
still uncertain ONCE LUMINANCE IS KNOWN. If luminance nearly determines colour, the
hue/saturation channels cost almost nothing to omit (reconstruct them from luminance with
a fixed table) and the output dimension really does collapse to 32. If luminance leaves
most of the hue/saturation entropy, two more channels are genuinely required and the
total goes UP, not down.

This script answers that with the actual 16-colour palette and the actual pixel
frequencies, before any network is trained.

It also reports the practical hazards that make HSV regression awkward:
  * hue is CIRCULAR, so MSE on degrees is wrong (350 and 10 are 20 apart, not 340)
  * hue is UNDEFINED for neutral colours, which this palette is full of
"""
import json, os
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "results", "hsv")
os.makedirs(OUT, exist_ok=True)

PAL = np.load(os.path.join(ROOT, "data", "palette_avatar16_v2.npy")).astype(np.float64)
lab = np.load(os.path.join(ROOT, "data", "faces8_v2.npy")).astype(np.int64).reshape(-1, 8, 8)
print(f"palette {PAL.shape}, {len(lab):,} label images")

# ---- palette in HSV ----------------------------------------------------------------
def rgb_to_hsv(c):
    r, g, b = c[..., 0] / 255.0, c[..., 1] / 255.0, c[..., 2] / 255.0
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    d = mx - mn
    h = np.zeros_like(mx)
    m = d > 1e-9
    # hue in degrees, 0..360
    idx = m & (mx == r)
    h[idx] = (60 * ((g[idx] - b[idx]) / d[idx])) % 360
    idx = m & (mx == g) & ~(mx == r)
    h[idx] = 60 * ((b[idx] - r[idx]) / d[idx]) + 120
    idx = m & (mx == b) & ~(mx == r) & ~(mx == g)
    h[idx] = 60 * ((r[idx] - g[idx]) / d[idx]) + 240
    s = np.where(mx > 1e-9, d / np.maximum(mx, 1e-9), 0.0)
    return h, s, mx


H, S, V = rgb_to_hsv(PAL)
print()
print("=" * 100)
print("1. THE PALETTE IN HSV")
print("=" * 100)
print(f"{'idx':>3} {'RGB':>15} {'H(deg)':>8} {'S':>7} {'V':>7}   note")
print("-" * 100)
for i in range(16):
    note = ""
    if S[i] < 0.06:
        note = "nearly neutral -> hue is meaningless/undefined"
    print(f"{i:3d} {str(PAL[i].astype(int).tolist()):>15} {H[i]:8.1f} {S[i]:7.3f} "
          f"{V[i]:7.3f}   {note}")

neutral = S < 0.06
print()
print(f"  {int(neutral.sum())}/16 palette colours are nearly neutral (S<0.06): "
      f"{np.flatnonzero(neutral).tolist()}")
print(f"  for those, hue is numerically arbitrary: this is why HSV regression on hue "
      f"needs masking or a circular loss")

# ---- pixel frequencies over the whole dataset -------------------------------------
freq = np.bincount(lab.reshape(-1), minlength=16).astype(np.float64)
p = freq / freq.sum()
print()
print("=" * 100)
print("2. HOW MUCH OF (HUE, SATURATION) IS STILL UNCERTAIN GIVEN LUMINANCE?")
print("=" * 100)
print(f"  pixel-frequency of each palette colour (%): "
      f"{np.round(p*100, 2).tolist()}")

# luminance bucket: use V (value) since that is what a luminance model predicts
# discretise V into 4 coarse bands so the conditional is estimable
vband = np.digitize(V, np.quantile(V, [0.25, 0.5, 0.75]))
# also a finer 8-band version
vband8 = np.digitize(V, np.quantile(V, np.linspace(0, 1, 9)[1:-1]))


def ent(pr):
    pr = pr[pr > 0]
    return float(-(pr * np.log2(pr)).sum())


def cond_entropy(band, target_idx):
    """H(target | band) in bits, weighted by band mass. target_idx groups palette
    colours into the classes we would have to predict."""
    tot = 0.0
    for b in np.unique(band):
        m = band == b
        w = p[m].sum()
        if w <= 0:
            continue
        q = p[m] / w
        # aggregate q over target classes
        agg = np.zeros(max(target_idx) + 1)
        for i in np.flatnonzero(m):
            agg[target_idx[i]] += q[np.flatnonzero(m) == i][0] if False else 0
        # simpler: sum q by class
        agg = np.zeros(max(target_idx) + 1)
        for k, i in enumerate(np.flatnonzero(m)):
            agg[target_idx[i]] += q[k]
        tot += w * ent(agg)
    return float(tot)


# treat each palette colour as its own "colour class"
col = np.arange(16)
H_col = ent(p)
for name, band in (("V in 4 bands", vband), ("V in 8 bands", vband8)):
    hc = cond_entropy(band, col)
    print(f"  H(colour) = {H_col:.3f} bits")
    print(f"  H(colour | {name}) = {hc:.3f} bits  ->  "
          f"{100*(1-hc/H_col):.1f}% of the colour entropy is removed by knowing luminance")
print()
print("  Read this as: if only a small share is removed, luminance does NOT determine")
print("  colour and the hue/saturation channels carry real, separate information.")

# ---- how many palette colours share a luminance band? -----------------------------
print()
print("=" * 100)
print("3. THE AMBIGUITY THAT DECIDES IT")
print("=" * 100)
for b in np.unique(vband8):
    m = np.flatnonzero(vband8 == b)
    if len(m) < 1:
        continue
    print(f"  luminance band {b}: {len(m)} palette colour(s) "
          f"V={np.round(V[m],3).tolist()}")
    print(f"                    hue={np.round(H[m],0).tolist()}  "
          f"S={np.round(S[m],2).tolist()}  pixel share {p[m].sum()*100:.1f}%")
print()
print("  A band with several colours of DIFFERENT hue is direct evidence that luminance")
print("  alone cannot recover colour, so those extra channels are not free.")

# ---- what a tiny model would have to predict -------------------------------------
print()
print("=" * 100)
print("4. OUTPUT-DIMENSION ARITHMETIC")
print("=" * 100)
bits_palette = 32 * np.log2(16)
print(f"  a 32-value, 16-level output carries 32 x 4 = {bits_palette:.0f} bits")
print(f"  splitting into L/H/S at the same 16 levels would carry 32 x 4 x 3 = "
      f"{32*4*3} bits -- MORE, not less")
print(f"  it only wins if hue and saturation need FEWER levels, or if they can be")
print(f"  reconstructed from luminance instead of transmitted")
# how many levels do hue/sat actually need?
print()
print(f"  distinct hue values in the palette (non-neutral only): "
      f"{len(np.unique(np.round(H[~neutral])))}")
print(f"  distinct saturation values: {len(np.unique(np.round(S, 2)))}")
print(f"  distinct luminance values: {len(np.unique(np.round(V, 3)))}")

json.dump({
    "palette_rgb": PAL.astype(int).tolist(),
    "palette_hsv": [[float(H[i]), float(S[i]), float(V[i])] for i in range(16)],
    "pixel_freq": p.tolist(),
    "H_colour_bits": H_col,
    "H_colour_given_V4": cond_entropy(vband, col),
    "H_colour_given_V8": cond_entropy(vband8, col),
    "neutral_colours": np.flatnonzero(neutral).tolist(),
    "bits_palette_output": float(bits_palette),
}, open(os.path.join(OUT, "palette_hsv.json"), "w"), indent=2)
print(f"\nwrote {os.path.join(OUT, 'palette_hsv.json')}")
print("HSV_FEASIBILITY_DONE")
