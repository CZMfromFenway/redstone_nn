#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
按用户的定义重算 A_j 与 c_j，并与旧公式对照。

用户定义
    A_j = F_j 在 η_j ≈ 0 处的值                （整数 RELU 阈值／零点）
    c_j = 使 A_j + c_j·2^b >= F_j 理论最大值 的最小整数（输出量化步长）
    Y_j = RELU(F_j) := clip(floor((F_j - A_j)/c_j), 0, 2^b-1)
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
sys.path.insert(0, HERE)

P2 = os.path.join(os.path.dirname(ROOT), "02_phase2_avatar")
mj = json.load(open(os.path.join(P2, "webapp", "pca22_dim4w64L1_model.json"),
                    encoding="utf-8"))
W = np.array(mj["gen"]["layers"][0]["W"], dtype=np.float64)    # u_{j,i}
TH = np.array(mj["gen"]["layers"][0]["b"], dtype=np.float64)   # theta_j

B = 8
NB = 1 << B
XLO, XHI = -3.5, 3.5
DELTA = XHI - XLO
LIMIT = 882

# ---- 第 2 步
alpha = LIMIT / (np.abs(W).max(1) * DELTA / 4)
U = np.rint(alpha[:, None, None] * W[:, :, None] * (DELTA / NB)
            * (1 << np.arange(B))[None, None, :]).astype(np.int64)
Hmin = np.where(U < 0, U, 0).sum(axis=(1, 2))
Hmax = np.where(U > 0, U, 0).sum(axis=(1, 2))
Fmax = Hmax - Hmin                                  # = sum |U|

# ---- 用户的 A_j：F_j 在 eta_j = 0 处的值
#   H_j ~ alpha_j * (DELTA/2^b) * sum_i u_ji * X_i
#   eta_j = theta_j + x_min*sum u + (DELTA/2^b)*sum u*(X_i + 1/2)
#   => H_j(eta=0) = -alpha_j*(theta_j + x_min*sum u + (DELTA/2^{b+1})*sum u)
H_at_eta0 = -alpha * (TH + XLO * W.sum(1) + (DELTA / (2 ** (B + 1))) * W.sum(1))
A = np.rint(H_at_eta0 - Hmin).astype(np.int64)

# ---- 用户的 c_j：最小整数使 A_j + c*2^b >= Fmax
c = np.maximum(np.ceil((Fmax - A) / NB), 1).astype(np.int64)
c_alt = np.maximum(np.ceil((Fmax - A) / (NB - 1)), 1).astype(np.int64)   # 用 2^b-1

print("=" * 100)
print("A_j 与 c_j")
print("=" * 100)
print(f"  alpha_j   范围 [{alpha.min():.1f}, {alpha.max():.1f}]")
print(f"  sum|U| = Fmax 范围 [{Fmax.min()}, {Fmax.max()}]")
print(f"  H_j(eta=0) 范围 [{H_at_eta0.min():.1f}, {H_at_eta0.max():.1f}]")
print(f"  A_j       范围 [{A.min()}, {A.max()}]")
print(f"  c_j       范围 [{c.min()}, {c.max()}]，中位 {int(np.median(c))}")
print(f"  c_j(用 2^b-1) 范围 [{c_alt.min()}, {c_alt.max()}]，中位 {int(np.median(c_alt))}")
print(f"  两者不同的通道数 {int((c != c_alt).sum())} / {len(c)}")

# ---- 验证 A_j：找 eta 最接近 0 的样本，看那时的 F
rng = np.random.default_rng(3)
NS = 200000
X = np.clip(np.floor((np.clip(rng.standard_normal((NS, 4)), XLO, XHI) - XLO) / DELTA * NB),
            0, NB - 1).astype(np.int64)
bits = ((X[:, :, None] >> np.arange(B)[None, None, :]) & 1).astype(np.int64)
H = np.einsum("nik,jik->nj", bits, U)
F = H - Hmin[None, :]
xh = XLO + (X + 0.5) * (DELTA / NB)
eta = xh @ W.T + TH

err = []
for j in range(64):
    k = int(np.argmin(np.abs(eta[:, j])))
    err.append(int(F[k, j]) - int(A[j]))
err = np.array(err)
print()
print("=" * 100)
print("验证 A_j：取 eta 最接近 0 的样本，看 F 与 A_j 的差")
print("=" * 100)
print(f"  差的分布：中位 {int(np.median(err))}，|差| <= 2 的通道 "
      f"{int((np.abs(err) <= 2).sum())}/64，最大 |差| {int(np.abs(err).max())}")
print("  （样本随机，eta 不一定精确为 0，所以差几个单位是正常的）")

# ---- 验证 Y 的上端
Y = np.clip(np.floor_divide(F - A[None, :], c[None, :]), 0, NB - 1)
print()
print("=" * 100)
print("Y 的实际范围与饱和")
print("=" * 100)
print(f"  Y 全体范围 [{Y.min()}, {Y.max()}]")
print(f"  达到 255 的比例 {100*(Y==255).mean():.3f}%   达到 0 的比例 {100*(Y==0).mean():.3f}%")
top = np.clip(np.floor_divide(Fmax - A, c), 0, NB - 1)
print(f"  每个通道在 F=Fmax 时的 Y：范围 [{top.min()}, {top.max()}]，"
      f"达到 255 的通道 {int((top==255).sum())}/64")
print(f"  每个通道在 F=0 时的 Y（理论上界，应为 0）：最大 "
      f"{int(np.clip(np.floor_divide(0-A,c),0,NB-1).max())}")
print()
print("  用 c_alt（除以 2^b-1）时：")
top2 = np.clip(np.floor_divide(Fmax - A, c_alt), 0, NB - 1)
print(f"  在 F=Fmax 时的 Y：范围 [{top2.min()}, {top2.max()}]，"
      f"达到 255 的通道 {int((top2==255).sum())}/64")

print()
print("=" * 100)
print("与前一次审计的 c_j 对照")
print("=" * 100)
zs = TH + 3.5 * np.abs(W).sum(1)
c_old = np.maximum(np.ceil(alpha * zs / NB), 1).astype(np.int64)
print(f"  旧公式 ceil(alpha*zeta_sup/2^b)：范围 [{c_old.min()}, {c_old.max()}]，中位 {int(np.median(c_old))}")
print(f"  你的公式 ceil((Fmax-A)/2^b)      ：范围 [{c.min()}, {c.max()}]，中位 {int(np.median(c))}")
print(f"  逐通道相同的个数 {int((c == c_old).sum())}/64，"
      f"新减旧 的范围 [{int((c-c_old).min())}, {int((c-c_old).max())}]")
print("  差别来自旧公式用 zeta 的解析上确界（连续），新公式用 Fmax（整数可达值）。")
print("STEP3_DONE")
