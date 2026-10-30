#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
Export the six surviving styles as Gaussians the page can sample from, and measure how much of
the grayscale variation the four knobs actually keep.

THE USER'S CONCLUSION, PUT ON A NUMBER
"这个 4 维度的空间还是太小了，很多精彩的黑白皮肤方案都无法生成。"

That is testable. The grayscale PCA basis is 64-dimensional; the generator receives only the first
four components. Whatever the remaining dimensions carry cannot be produced at all, no matter what
the knobs are set to. The explained-variance curve says how much is out of reach.

ON 潦草大叔
Dropped at the user's request: it is a real style but occupies too small a region to be worth a
preset (16.7% bad at one sigma). 铁傀儡 and 黑山羊 were rejected for the same reason, which is a
property of the four-dimensional space, not of those styles.
"""
import json
import os
import sys
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WEB = os.path.join(ROOT, "webapp")
OUT = os.path.join(ROOT, "results", "styles")

# ---------------------------------------------------------------- how much of grayscale is kept
full = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_full_idx.npy")).astype(np.float64)
N = len(full)
X = full.reshape(N, -1)
mu = X.mean(0, keepdims=True)
sub = np.random.default_rng(0).choice(N, 60000, replace=False)
U, S, Vt = np.linalg.svd(X[sub] - mu, full_matrices=False)
ev = (S ** 2) / (S ** 2).sum()
cum = np.cumsum(ev)
print("=" * 96)
print("HOW MUCH OF THE GRAYSCALE VARIATION THE FOUR KNOBS KEEP")
print("=" * 96)
print(f"  grayscale image is {X.shape[1]} numbers (8x8 minus the mirror half, times 1 channel)")
print(f"  {'components':>11s} {'cumulative':>11s}")
for k in (1, 2, 3, 4, 6, 8, 12, 16, 22, 32):
    if k <= len(cum):
        tail = "   <- the generator sees this much" if k == 4 else ""
        print(f"  {k:11d} {cum[k-1]*100:10.1f}%{tail}")
print()
print(f"  4 个分量保留了 {cum[3]*100:.1f}% 的方差，也就是丢掉了 {100-cum[3]*100:.1f}%。")
print(f"  要达到 90% 需要 {int(np.searchsorted(cum, 0.90))+1} 个分量，"
      f"95% 需要 {int(np.searchsorted(cum, 0.95))+1} 个。")
print()
print("  这就是『很多精彩的皮肤方案生成不出来』的量化解释：那部分变化根本不在 4 个输入里。")

# ---------------------------------------------------------------- export the six styles
ST = json.load(open(os.path.join(OUT, "styles.json"), encoding="utf-8"))
KEEP = ["少女", "少年", "骷髅", "毒液", "黑色悦灵", "黑暗法师"]
DROP = ["潦草大叔"]
print()
print("=" * 96)
print("STYLE EXPORT")
print("=" * 96)
print(f"  保留 {len(KEEP)} 个：{'、'.join(KEEP)}")
print(f"  删除：{'、'.join(DROP)}（用户要求：占地太小）")

out = {"_note": "Sampling presets. Draw z ~ N(0, I), then x = mu + chol @ z, then clip to [-3.5, 3.5].",
       "_chol_note": "cholesky of the covariance, lower triangular, row-major",
       "styles": []}
for name in KEEP:
    s = ST[name]
    mu_v = np.array(s["mu"], dtype=np.float64)
    C = np.array(s["cov"], dtype=np.float64)
    # symmetrise and add a small ridge so the factorisation is always defined
    C = (C + C.T) / 2 + np.eye(4) * 1e-6
    try:
        L = np.linalg.cholesky(C)
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh(C)
        w = np.maximum(w, 1e-6)
        L = V @ np.diag(np.sqrt(w))
    out["styles"].append({"name": name, "mu": mu_v.tolist(),
                          "chol": L.tolist(),
                          "sd": [float(np.sqrt(C[i, i])) for i in range(4)],
                          "bad1": s["bad1"], "bad2": s["bad2"]})
    print(f"  {name:10s} mu {np.round(mu_v,2)!s:>24s}  sd {np.round(np.sqrt(np.diag(C)),2)!s}")

fp = os.path.join(WEB, "styles_gauss.json")
json.dump(out, open(fp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"\nwrote {fp}  ({os.path.getsize(fp)/1e3:.1f} KB)")

# ---------------------------------------------------------------- sanity: samples stay in range
print()
print("=" * 96)
print("SANITY CHECK: what fraction of style samples land inside the cube")
print("=" * 96)
rng = np.random.default_rng(0)
for s in out["styles"]:
    mu_v = np.array(s["mu"]); L = np.array(s["chol"])
    Z = rng.standard_normal((4000, 4))
    Xs = Z @ L.T + mu_v
    inside = np.all(np.abs(Xs) <= 3.5, axis=1)
    print(f"  {s['name']:10s} 在立方体内 {100*inside.mean():5.1f}%   "
          f"（越界的会被夹到边界或重抽）")
print("\nSTYLE_EXPORT_DONE")
