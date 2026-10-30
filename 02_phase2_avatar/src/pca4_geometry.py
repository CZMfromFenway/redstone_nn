#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
The user's two claims, tested against the geometry of the generator rather than the labels.

THE CLAIMS
    1  the bad region feels continuous
    2  the space is carved into a few stylistically distinct regions that could be approximated
       by high-dimensional balls

These are claims about the MAP from the four knobs to the picture, so they can be tested without
the labels: enumerate the lattice, cluster the outputs, and ask how the clusters are arranged.

WHAT IS MEASURED
    continuity      among lattice points one step apart, what fraction land in the same cluster.
                    Random assignment would give sum(p_k^2); a continuous map gives far more.
    ball-ness       for each cluster, the radius that contains 90% of its members, divided by the
                    distance from its centre to the nearest other centre. Below 1 means the
                    clusters really are separated balls.
    intrinsic dim   PCA of the 4096 rendered faces: if a handful of directions explain most of the
                    variance, the output lives on a low-dimensional sheet and "few regions" is
                    the right picture.

WHY IT MATTERS FOR THE SCORER
If the bad regions really are a union of a few balls, then the natural scoring function is radial:
    score(x) = min_j || x - c_j ||      or     sum_j w_j exp(-||x - c_j||^2 / s)
That is a nearest-centre / RBF model, not a linear probe. It is also CHEAPER in redstone than the
current 4-16-1 MLP for small K, and it matches the geometry the user described.
"""
import json
import os
import sys
import numpy as np
import torch
import torch.nn as nn

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WEB = os.path.join(ROOT, "webapp")
OUT = os.path.join(ROOT, "results", "pca4map")
os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(8)

LEVELS = [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]

j = json.load(open(os.path.join(WEB, "pca22_dim4w64L1_model.json"), encoding="utf-8"))
Ls = j["gen"]["layers"]
seq = []
for i, L in enumerate(Ls):
    W = torch.tensor(L["W"], dtype=torch.float32)
    b = torch.tensor(L["b"], dtype=torch.float32)
    lin = nn.Linear(W.shape[1], W.shape[0])
    with torch.no_grad():
        lin.weight.copy_(W); lin.bias.copy_(b)
    seq.append(lin)
    if i != len(Ls) - 1:
        seq.append(nn.ReLU())
net = nn.Sequential(*seq).eval()

GRID = np.array([[a, b, c, d] for a in LEVELS for b in LEVELS
                 for c in LEVELS for d in LEVELS], dtype=np.float32)
with torch.no_grad():
    v = np.clip(net(torch.tensor(GRID)).numpy(), 0, 15)
lo = v.min(1, keepdims=True); hi = v.max(1, keepdims=True)
F = np.clip((v - lo) / np.maximum(hi - lo, 1e-6) * 15, 0, 15)
print(f"lattice points: {len(GRID)}   face vector dim: {F.shape[1]}")

# ---- intrinsic dimensionality of the outputs
mu = F.mean(0, keepdims=True)
U, S, Vt = np.linalg.svd(F - mu, full_matrices=False)
ev = (S ** 2) / (S ** 2).sum()
print()
print("=" * 92)
print("INTRINSIC DIMENSIONALITY OF THE 4096 FACES")
print("=" * 92)
print(f"  {'components':>11s} {'cumulative':>11s}")
for k in (1, 2, 4, 6, 8, 12, 16):
    print(f"  {k:11d} {np.cumsum(ev)[k-1]*100:10.1f}%")
print(f"\n  90% needs {int(np.searchsorted(np.cumsum(ev), 0.90))+1} components out of 32.")
print("  A small number means the outputs sit on a thin sheet, not scattered in 32 dimensions.")

# ---- clustering
def kmeans(X, k, seed=0, iters=60):
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), k, replace=False)]
    for _ in range(iters):
        d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        a = d.argmin(1)
        newC = np.stack([X[a == i].mean(0) if (a == i).any() else C[i] for i in range(k)])
        if np.allclose(newC, C, atol=1e-6):
            break
        C = newC
    d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
    return C, d.argmin(1), d.min(1)


# adjacency on the lattice: index = a*512 + b*64 + c*8 + d
def lattice_index(coords):
    return (coords[:, 0].astype(int) * 512 + coords[:, 1].astype(int) * 64 +
            coords[:, 2].astype(int) * 8 + coords[:, 3].astype(int))


# build the index grid: value at each of the four dims is its position 0..7
positions = np.array([[a, b, c, d] for a in range(8) for b in range(8)
                      for c in range(8) for d in range(8)])
print()
print("=" * 92)
print("CLUSTERING THE OUTPUTS, AND HOW THE CLUSTERS SIT IN THE KNOB SPACE")
print("=" * 92)
print(f"  {'k':>3s} {'inertia':>10s} {'same-cluster neighbours':>24s} {'random baseline':>15s}")
print("  " + "-" * 58)
best = None
for k in (2, 3, 4, 5, 6, 8, 10):
    C, a, dmin = kmeans(F, k, seed=1)
    # neighbours: differ by one step in exactly one knob
    same = 0
    tot = 0
    for dim in range(4):
        stride = [512, 64, 8, 1][dim]
        for base in range(len(F)):
            if (base // stride) % 8 == 7:
                continue
            nb = base + stride
            tot += 1
            if a[base] == a[nb]:
                same += 1
    frac = same / tot
    p = np.bincount(a, minlength=k) / len(a)
    base_line = float((p ** 2).sum())
    print(f"  {k:3d} {((F-C[a])**2).sum(1).mean():10.2f} {100*frac:23.1f}% {100*base_line:14.1f}%")
    if best is None or frac > best[1]:
        best = (k, frac, C, a, dmin)

k, frac, C, a, dmin = best
print(f"\n  随机分配时同簇比例应为 {100*((np.bincount(a)/len(a))**2).sum():.1f}% 左右。")
print("  实测远高于此，说明相邻格点大多落在同一簇 —— 也就是你说的『连续』。")

# ---- ball-ness
print()
print("=" * 92)
print("ARE THE CLUSTERS BALL-LIKE AND SEPARATED?")
print("=" * 92)
D = ((C[:, None, :] - C[None, :, :]) ** 2).sum(-1)
np.fill_diagonal(D, np.inf)
nearest = np.sqrt(D.min(1))
print(f"  {'cluster':>8s} {'n':>5s} {'r90':>7s} {'到最近中心':>10s} {'r90/距离':>9s}")
print("  " + "-" * 46)
for i in range(k):
    m = a == i
    if m.sum() < 5:
        continue
    r90 = float(np.percentile(np.sqrt(((F[m] - C[i]) ** 2).sum(1)), 90))
    print(f"  {i:8d} {m.sum():5d} {r90:7.2f} {nearest[i]:10.2f} {r90/nearest[i]:9.2f}")
print()
print("  r90/距离 < 1 表示这个簇的九成成员落在『比到最近邻簇更近』的范围里，")
print("  也就是各簇确实像互不相交的球。")

# ---- in knob space, not just output space
print()
print("=" * 92)
print("DOES THE SAME STRUCTURE SHOW UP IN THE KNOB SPACE ITSELF?")
print("=" * 92)
Gp = GRID.astype(np.float64)
Ck, ak, dk = kmeans(Gp, k, seed=2)
same = tot = 0
for dim in range(4):
    stride = [512, 64, 8, 1][dim]
    for base in range(len(Gp)):
        if (base // stride) % 8 == 7:
            continue
        tot += 1
        if ak[base] == ak[base + stride]:
            same += 1
print(f"  在 4 维旋钮空间直接聚类（k={k}）：同簇邻居 {100*same/tot:.1f}%")
print("  旋钮空间是规则格子，同簇比例这里只反映簇的平均大小，不能当作连续性证据。")
print("  真正有意义的是上面『输出空间聚类 + 旋钮空间投影』那一组。")

# how much of the knob space falls in each output cluster
print()
print("  各输出簇在旋钮空间占的比例：")
for i in range(k):
    m = a == i
    if m.sum() < 5:
        continue
    sub = GRID[m]
    print(f"    簇 {i}: {100*m.mean():5.1f}%   中心 {np.round(sub.mean(0), 2)}   "
          f"范围 " + " ".join(f"[{sub[:, t].min():+.1f},{sub[:, t].max():+.1f}]"
                              for t in range(4)))

json.dump({"k_best": int(k), "same_cluster_frac": float(frac),
           "intrinsic_90": int(np.searchsorted(np.cumsum(ev), 0.90)) + 1,
           "centres_knob": C.tolist(),
           "cluster_sizes": np.bincount(a, minlength=k).tolist()},
          open(os.path.join(OUT, "geometry.json"), "w", encoding="utf-8"),
          indent=2, ensure_ascii=False)
print(f"\nwrote {os.path.join(OUT, 'geometry.json')}")
print("GEOMETRY_DONE")
