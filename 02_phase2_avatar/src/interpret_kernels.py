#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
What DID the learned kernels find? Answer it by measurement, not by eyeballing.

Two questions:
  Q1 Is each learned kernel really an edge/gradient detector, or did they collapse to
     near-duplicates? -> pairwise cosine similarity between kernels, and the shape of
     each kernel's weight matrix.
  Q2 Does each style CLUSTER correspond to a nameable property of the face?
     -> for every cluster, report interpretable statistics computed from the data:
        mean brightness, vertical gradient (dark top vs light bottom), contrast,
        edge density, eye-row darkness (row 3), and the fraction that are near-flat.

Q2 is what makes the style labels honest: a cluster is only "dark hair on top, light
skin below" if its measured vertical gradient actually is strongly negative.
"""
import json, os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "results", "kernel_style", "kernel_style")
full = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_full_idx.npy")).astype(np.float64)
lab = np.load(os.path.join(D, "labels.npy"))
info = json.load(open(os.path.join(D, "kernel_info.json"), encoding="utf-8"))
K = info["K"]

print("=" * 100)
print("Q1  are the 9 learned 3x3 kernels distinct, and are they edge detectors?")
print("=" * 100)
k3 = np.array(info["kernels_3x3"])                     # (9,9)
kn = k3 / (np.linalg.norm(k3, axis=1, keepdims=True) + 1e-9)
C = kn @ kn.T
print("  pairwise |cosine| between 3x3 kernels (1.00 = identical direction):")
print("        " + "".join(f"{j:7d}" for j in range(len(k3))))
for i in range(len(k3)):
    print(f"   k3[{i}] " + "".join(f"{abs(C[i,j]):7.2f}" for j in range(len(k3))))
off = np.abs(C[~np.eye(len(k3), dtype=bool)])
print(f"  mean off-diagonal |cosine| = {off.mean():.3f}, max = {off.max():.3f}")
print("  -> below ~0.7 means the kernels found genuinely different filters")
print()
# a kernel is an edge detector if its weights sum to ~0 (no DC response)
print("  DC response (sum of weights) and anisotropy (|row means| spread):")
for i in range(len(k3)):
    w = k3[i].reshape(3, 3)
    dc = w.sum()
    rowm = w.mean(1); colm = w.mean(0)
    print(f"   k3[{i}]  sum={dc:+6.3f}  row-means=[{rowm[0]:+6.3f}{rowm[1]:+6.3f}{rowm[2]:+6.3f}]"
          f"  col-means=[{colm[0]:+6.3f}{colm[1]:+6.3f}{colm[2]:+6.3f}]")
print("  (a near-zero sum with large row/col spread = a directional edge / gradient filter)")

print()
print("=" * 100)
print("Q2  what property does each style CLUSTER actually capture?")
print("=" * 100)
# interpretable features per face
g = full                                                  # (N,8,8) levels 0..15
bright = g.mean(axis=(1, 2))                              # overall brightness
top = g[:, :3, :].mean(axis=(1, 2))                       # rows 0..2
bot = g[:, 5:, :].mean(axis=(1, 2))                       # rows 5..7
vert = top - bot                                          # >0 = dark on top, light below? (levels: high=bright)
contrast = g.std(axis=(1, 2))
# edge density: count adjacent differences above a threshold
dx = np.abs(np.diff(g, axis=2)).mean(axis=(1, 2))
dy = np.abs(np.diff(g, axis=1)).mean(axis=(1, 2))
edge = dx + dy
flat = (contrast < 0.6)                                   # near-uniform face
eye = g[:, 3, :].mean(1)                                  # eye row brightness

print(f"{'cl':>3} {'n':>7} {'share':>6} {'bright':>7} {'top-row':>8} {'bot-row':>8} "
      f"{'top-bot':>8} {'contrast':>9} {'edge':>6} {'eye-row':>8} {'flat%':>6}  label")
print("-" * 120)
rows = []
for j in range(K):
    m = lab == j
    n = int(m.sum())
    if n == 0:
        continue
    r = {"cluster": j, "n": n, "share": n / len(lab),
         "bright": float(bright[m].mean()), "top": float(top[m].mean()),
         "bot": float(bot[m].mean()), "top_minus_bot": float(vert[m].mean()),
         "contrast": float(contrast[m].mean()), "edge": float(edge[m].mean()),
         "eye_row": float(eye[m].mean()), "flat_frac": float(flat[m].mean())}
    # name the style from measurements only
    tags = []
    if r["flat_frac"] > 0.35:
        tags.append("mostly flat/uniform")
    if r["bright"] < 6.2:
        tags.append("dark overall")
    elif r["bright"] > 10.0:
        tags.append("light overall")
    if r["top_minus_bot"] < -1.5:
        tags.append("light on TOP / dark below")
    elif r["top_minus_bot"] > 1.5:
        tags.append("dark hair on top / light below")
    if r["contrast"] > 4.0:
        tags.append("high contrast (strong features)")
    elif r["contrast"] < 2.5:
        tags.append("low contrast")
    if r["eye_row"] < r["bright"] - 1.5:
        tags.append("dark eye row")
    r["label"] = ", ".join(tags) if tags else "(no single dominant trait)"
    rows.append(r)
    print(f"{j:3d} {n:7d} {r['share']*100:5.1f}% {r['bright']:7.2f} {r['top']:8.2f} {r['bot']:8.2f} "
          f"{r['top_minus_bot']:+8.2f} {r['contrast']:9.2f} {r['edge']:6.2f} {r['eye_row']:8.2f} "
          f"{r['flat_frac']*100:5.1f}  {r['label']}")

print()
print("=" * 100)
print("dataset-wide reference values (so the cluster numbers mean something)")
print("=" * 100)
print(f"  brightness {bright.mean():.2f}   top {top.mean():.2f}   bottom {bot.mean():.2f}   "
      f"top-bottom {vert.mean():+.2f}")
print(f"  contrast {contrast.mean():.2f}   edge {edge.mean():.2f}   eye-row {eye.mean():.2f}   "
      f"flat fraction {flat.mean()*100:.1f}%")
json.dump({"cluster_profiles": rows, "k3_cosine_abs_mean": float(off.mean()),
           "k3_cosine_abs_max": float(off.max()),
           "dataset": {"brightness": float(bright.mean()), "top_minus_bot": float(vert.mean()),
                       "contrast": float(contrast.mean()), "edge": float(edge.mean()),
                       "flat_frac": float(flat.mean())}},
          open(os.path.join(D, "cluster_interpretation.json"), "w"), indent=2)
print()
print(f"wrote {os.path.join(D, 'cluster_interpretation.json')}")
print("INTERPRET_DONE")
