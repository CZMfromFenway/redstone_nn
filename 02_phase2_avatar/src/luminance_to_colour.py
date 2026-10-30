#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Can a LUMINANCE model drive a COLOUR model? The answer hinges on which "luminance".

The previous script showed the 16 palette colours have 16 DISTINCT V values (max RGB
channel), so V alone identifies the colour in principle. But the grayscale images used
throughout this project are not V: they are perceptual LUMINANCE (0.299R + 0.587G +
0.114B), quantised to 16 uniformly spaced levels. Those are different quantities and the
difference decides the whole design.

This measures, for the actual palette:
  * the perceptual luminance of each palette colour, and its 16-level quantisation
  * how many palette colours COLLIDE at the same luminance level (a collision means
    luminance cannot recover colour: two colours, one input)
  * H(colour | luminance level) in bits, the residual the hue/saturation channels
    would have to carry
  * the same for V, to see whether V would have been the better choice
"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "results", "hsv")
os.makedirs(OUT, exist_ok=True)

PAL = np.load(os.path.join(ROOT, "data", "palette_avatar16_v2.npy")).astype(np.float64)
lab = np.load(os.path.join(ROOT, "data", "faces8_v2.npy")).astype(np.int64).reshape(-1, 8, 8)
freq = np.bincount(lab.reshape(-1), minlength=16).astype(np.float64)
p = freq / freq.sum()

LUM = np.array([0.299, 0.587, 0.114])
lum = PAL @ LUM                      # perceptual luminance, 0..255
V = PAL.max(1)                       # max channel, 0..255
LEVELS = np.linspace(0, 255, 16)
lum16 = np.abs(lum[:, None] - LEVELS[None, :]).argmin(1)
v16 = np.abs(V[:, None] - LEVELS[None, :]).argmin(1)

print("=" * 104)
print("THE PALETTE: perceptual luminance vs max-channel value")
print("=" * 104)
print(f"{'idx':>3} {'RGB':>15} {'lum':>7} {'to16':>5} {'V':>7} {'Vto16':>6} {'pix%':>6}")
print("-" * 104)
for i in range(16):
    print(f"{i:3d} {str(PAL[i].astype(int).tolist()):>15} {lum[i]:7.2f} {lum16[i]:5d} "
          f"{V[i]:7.2f} {v16[i]:6d} {p[i]*100:6.2f}")


def ent(pr):
    pr = pr[pr > 0]
    return float(-(pr * np.log2(pr)).sum())


def cond_h(class_of, name):
    """H(colour | class), weighted by class mass."""
    tot = 0.0
    for c in np.unique(class_of):
        m = class_of == c
        w = p[m].sum()
        if w <= 0:
            continue
        q = p[m] / w
        tot += w * ent(q)
    return float(tot)


H_col = ent(p)
print()
print("=" * 104)
print("HOW MUCH COLOUR INFORMATION SURVIVES")
print("=" * 104)
print(f"  H(colour) = {H_col:.3f} bits")
res = {}
for name, c in (("perceptual luminance, 16 uniform levels", lum16),
                ("max-channel V, 16 uniform levels", v16)):
    h = cond_h(c, name)
    res[name] = h
    print(f"  H(colour | {name}) = {h:.3f} bits  "
          f"-> {100*h/H_col:.1f}% of the colour entropy REMAINS")
print()
print(f"  In practice: {len(np.unique(lum16))}/16 luminance levels are used, and")
coll = {}
for lv in np.unique(lum16):
    m = np.flatnonzero(lum16 == lv)
    if len(m) > 1:
        coll[int(lv)] = m.tolist()
print(f"  {len(coll)} luminance level(s) are shared by MORE THAN ONE palette colour:")
for lv, idx in coll.items():
    print(f"    level {lv:2d}: colours {idx} -> RGB "
          f"{[PAL[i].astype(int).tolist() for i in idx]}  "
          f"(pixel share {p[idx].sum()*100:.1f}%)")
share_ambiguous = sum(p[idx].sum() for idx in coll.values())
print(f"  {100*share_ambiguous:.1f}% of all pixels sit on such an ambiguous level")

print()
print("=" * 104)
print("THE OTHER DIRECTION: WHAT PRECISION WOULD V NEED?")
print("=" * 104)
srt = np.sort(V)
gaps = np.diff(srt)
i = int(np.argmin(gaps))
print(f"  the two closest V values are {srt[i]:.3f} and {srt[i+1]:.3f} "
      f"(gap {gaps[i]:.3f} out of 255), i.e. {100*gaps[i]/255:.2f}% of the range")
print(f"  to tell EVERY colour apart from V alone, a continuous predictor would need")
print(f"  better than {gaps[i]:.2f}/255 precision -- but the measured 32-channel model")
print(f"  has MAE 0.315 levels = {0.315/15*255:.1f}/255. So it cannot resolve that pair.")
print()
print("  => in principle V determines colour; in practice the precision needed exceeds")
print("     what the model delivers, so the 16-level ROUNDING is what recovers colour.")

json.dump({
    "palette_rgb": PAL.astype(int).tolist(),
    "palette_lum": lum.tolist(), "palette_lum16": lum16.tolist(),
    "palette_V": V.tolist(), "palette_V16": v16.tolist(),
    "pixel_freq": p.tolist(),
    "H_colour": H_col,
    "H_colour_given_lum16": res["perceptual luminance, 16 uniform levels"],
    "H_colour_given_V16": res["max-channel V, 16 uniform levels"],
    "ambiguous_levels": {str(k): v for k, v in coll.items()},
    "ambiguous_pixel_share": float(share_ambiguous),
    "closest_V_gap": float(gaps[i]),
}, open(os.path.join(OUT, "luminance_to_colour.json"), "w"), indent=2)
print(f"\nwrote {os.path.join(OUT, 'luminance_to_colour.json')}")
print("LUM2COLOR_DONE")
