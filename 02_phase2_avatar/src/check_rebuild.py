#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
"""
Check whether the rebuilt v2 dataset actually removed the contamination the audit found.
Compares the OLD faces8.npy against the NEW faces8_v2.npy on the same metrics.
"""
import json, os
import numpy as np

ARCH = r"E:\redstone_nn\02_phase2_avatar\archive_phase1_attempts\data"
NEW = r"E:\redstone_nn\02_phase2_avatar\data"

old_lab = np.load(os.path.join(ARCH, "faces8.npy"))
old_rgb = np.load(os.path.join(ARCH, "faces8_rgb.npy"))
new_lab = np.load(os.path.join(NEW, "faces8_v2.npy"))
new_rgb = np.load(os.path.join(NEW, "faces8_rgb_v2.npy"))
cov = np.load(os.path.join(NEW, "alpha_cov_v2.npy"))
man = json.load(open(os.path.join(NEW, "manifest_v2.json"), encoding="utf-8"))

print("old:", old_lab.shape, old_rgb.shape)
print("new:", new_lab.shape, new_rgb.shape, "coverage", cov.shape)

PAIRS = np.array([(r * 8 + c, r * 8 + (7 - c)) for r in range(8) for c in range(4)])


def stats(tag, lab, rgb, cov=None):
    n = len(lab)
    uc = float(np.mean([len(np.unique(s)) for s in lab[:20000]]))
    sym = float((lab[:, PAIRS[:, 0]] == lab[:, PAIRS[:, 1]]).mean())
    flat = rgb.reshape(-1, 3)
    black = float((flat == 0).all(1).mean())
    allblack_rows = float((rgb.reshape(n, -1) == 0).all(1).mean())
    # duplicate label maps
    uniq_maps = len({s.tobytes() for s in lab})
    print(f"{tag:8s} n={n:>7,}  colors/av={uc:5.2f}  sym={sym:.4f}  "
          f"pure-black px={black*100:5.2f}%  all-black rows={allblack_rows*100:5.2f}%  "
          f"distinct maps={uniq_maps:,}")
    return dict(n=int(n), colors=round(uc, 4), sym=round(sym, 4),
                black_px=round(black, 5), all_black_rows=round(allblack_rows, 5),
                distinct_maps=int(uniq_maps))


print()
o = stats("OLD", old_lab, old_rgb.reshape(len(old_lab), 8, 8, 3))
nw = stats("NEW", new_lab, new_rgb)

print()
print("palette index 8 (near-black (4,4,4) in the old palette) usage:")
print(f"  OLD index-8 share : {(old_lab == 8).mean()*100:.2f}%   (audit reported 12.6%)")
# new palette: find the darkest entry
pal = np.load(os.path.join(NEW, "palette_avatar16_v2.npy"))
lum = pal.astype(int).sum(1)
dark = int(lum.argmin())
print(f"  NEW palette darkest entry = index {dark} rgb={pal[dark].tolist()} share {(new_lab == dark).mean()*100:.2f}%")

print()
print("contamination the rebuild was meant to remove:")
print(f"  new dataset alpha coverage: min {cov.min():.3f}  mean {cov.mean():.4f}  "
      f"rows with any transparency {int((cov < 1.0).sum()):,} ({(cov < 1.0).mean()*100:.2f}%)")
print(f"  manifest says alpha_dropped={man['counts']['alpha_dropped']:,}  crop_dup={man['counts']['crop_dup']:,}  kept={man['counts']['kept']:,}")
print(f"  quantization: per-channel RMS {man['quantization']['per_channel_rms_255']}/255 "
      f"(definition: {man['quantization']['definition']})")
print()
print(f"SUMMARY old_colors={o['colors']} -> new_colors={nw['colors']}")
print(f"SUMMARY old_blackpx={o['black_px']*100:.2f}% -> new_blackpx={nw['black_px']*100:.2f}%")
print("CHECK_DONE")
