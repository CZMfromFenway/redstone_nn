#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
按用户的重新定义：Z_p = G_p + B_p，B_p 叫通道 p 平移量。

B_p 的定义（用户）：使得「若 Z_{p1} > Z_{p2} 则必有 z_{p1} > z_{p2}」的整数。

构造
    G_p ≈ β z_p - β(φ_p + ½ Σ_j v_pj δ_j) + 取整误差
    令 B_p = round(β(φ_p + ½ Σ_j v_pj δ_j)) + B_0
    则 Z_p = G_p + B_p ≈ β z_p + B_0
    同一个 B_0 对 32 个通道相同，所以差值全是 β 倍的真差值。

    B_0 取最小整数使所有可达的 Z_p >= 0。
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

P2 = os.path.join(os.path.dirname(ROOT), "02_phase2_avatar")
mj = json.load(open(os.path.join(P2, "webapp", "pca22_dim4w64L1_model.json"),
                    encoding="utf-8"))
U1W = np.array(mj["gen"]["layers"][0]["W"], dtype=np.float64)
TH1 = np.array(mj["gen"]["layers"][0]["b"], dtype=np.float64)
V2W = np.array(mj["gen"]["layers"][1]["W"], dtype=np.float64)
PHI = np.array(mj["gen"]["layers"][1]["b"], dtype=np.float64)

B = 8
NB = 1 << B
XLO, XHI = -3.5, 3.5
DELTA = XHI - XLO
LIMIT = 882

# 第 2~3 步
alpha = LIMIT / (np.abs(U1W).max(1) * DELTA / 4)
U1 = np.rint(alpha[:, None, None] * U1W[:, :, None] * (DELTA / NB)
             * (1 << np.arange(B))[None, None, :]).astype(np.int64)
Hmin = np.where(U1 < 0, U1, 0).sum(axis=(1, 2))
Hmax = np.where(U1 > 0, U1, 0).sum(axis=(1, 2))
A1 = np.rint(-alpha * (TH1 + XLO * U1W.sum(1)
                       + DELTA / (2 ** (B + 1)) * U1W.sum(1)) - Hmin).astype(np.int64)
c1 = np.maximum(np.ceil((Hmax - Hmin - A1) / NB), 1).astype(np.int64)
Ymax = np.clip(np.floor_divide(Hmax - Hmin - A1, c1), 0, NB - 1)

# 第 4 步
delta = c1 / alpha
g = np.abs(V2W) * delta[None, :]
beta = LIMIT / (g.max() * (1 << (B - 2)))
V2 = np.rint(beta * V2W[:, :, None] * delta[None, :, None]
             * (1 << np.arange(B))[None, None, :]).astype(np.int64)

# ---- 用户的 B_p
half_sum = 0.5 * (V2W * delta[None, :]).sum(1)          # ½ Σ_j v_pj δ_j
Bp_core = np.rint(beta * (PHI + half_sum)).astype(np.int64)

# B_0：使所有可达 Z_p >= 0 的最小公共常数
ymax = (Ymax + 0.5) * delta                              # 各输入的最大可达 y
zmin = PHI + (V2W.clip(max=0) * ymax[None, :]).sum(1)    # z_p 的最小可达值
B0 = int(np.ceil(max(-beta * zmin)))
Bp = Bp_core + B0

print("=" * 100)
print("通道 p 平移量 B_p")
print("=" * 100)
print(f"  β = {beta:.4f}      15β = {15*beta:.2f}")
print(f"  β(φ_p + ½Σ_j v_pj δ_j) 范围 [{Bp_core.min()}, {Bp_core.max()}]，"
      f"极差 {int(Bp_core.max()-Bp_core.min())}")
print(f"  B_0 = ceil(max_p(-β z_p^min)) = {B0}")
print(f"  B_p = 上面两项之和，范围 [{Bp.min()}, {Bp.max()}]")
print()
print("  B_p 逐通道：")
for r in range(4):
    print("    " + "  ".join(f"p{p:2d}:{int(Bp[p]):6d}" for p in range(r * 8, r * 8 + 8)))
print()
print(f"  显示截断区间 = [B_0, B_0 + 15β] = [{B0}, {B0 + int(round(15*beta))}]"
      f"   ← 全局两个常数")

# ---- 实测
rng = np.random.default_rng(41)
NF = 20000
xt = np.clip(rng.standard_normal((NF, 4)), XLO, XHI)
X1 = np.clip(np.floor((xt - XLO) / DELTA * NB), 0, NB - 1).astype(np.int64)
bt = ((X1[:, :, None] >> np.arange(B)[None, None, :]) & 1).astype(np.int64)
H = np.einsum("nik,jik->nj", bt, U1)
F = H - Hmin[None, :]
Y = np.clip(np.floor_divide(F - A1[None, :], c1[None, :]), 0, NB - 1)
btY = ((Y[:, :, None] >> np.arange(B)[None, None, :]) & 1).astype(np.int64)
G = np.einsum("njk,pjk->np", btY, V2)
Z = G + Bp[None, :]

yh = (Y + 0.5) * delta[None, :]
z2 = yh @ V2W.T + PHI

print()
print("=" * 100)
print("验证 Z_p - B_0 ≈ β z_p")
print("=" * 100)
e = (Z - B0) - beta * z2
print(f"  误差 均值 {e.mean():+.2f}  平均|误差| {np.abs(e).mean():.2f}  最大 {np.abs(e).max():.0f}")
print(f"  占 15β 的 {100*np.abs(e).max()/(15*beta):.3f}%")
print(f"  Z_p 全体范围 [{Z.min()}, {Z.max()}]，最低值 vs B_0 = {B0}："
      f"{'全部 >= B_0' if Z.min() >= B0 else '有低于 B_0 的'}")

print()
print("=" * 100)
print("序关系检验：Z 的序与 z 的序是否一致")
print("=" * 100)
# 每一对通道、每张脸，检查大小关系是否一致
pairs = [(i, j) for i in range(32) for j in range(i + 1, 32)]
flip = 0
tot = 0
big_flip = 0
for i, j in pairs:
    dz = z2[:, i] - z2[:, j]
    dZ = Z[:, i] - Z[:, j]
    s = (dz > 0) != (dZ > 0)
    flip += int(s.sum())
    tot += len(dz)
    # 张量级：差值绝对值超过 1/β 还翻的，才是真问题
    s2 = s & (np.abs(dz) > 1.0 / beta)
    big_flip += int(s2.sum())
print(f"  通道对数 {len(pairs)}，样本 {NF}，比较次数 {tot:,}")
print(f"  序翻转 {flip:,} 次 = {100*flip/tot:.4f}%")
print(f"  其中 |z_i - z_j| > 1/β = {1/beta:.4f} 仍翻转的 {big_flip:,} 次 = "
      f"{100*big_flip/tot:.6f}%")
print(f"  （|z 差| < 1/β 的翻转不可避免：整数分辨率只有 1/β = {1/beta:.4f}）")

print()
print("=" * 100)
print("最终 16 级的效果")
print("=" * 100)


def lvi(V):
    vm, vn = V.max(1, keepdims=True), V.min(1, keepdims=True)
    sp = np.maximum(vm - vn, 1)
    return np.clip(np.floor_divide((V - vn) * 15 + (sp >> 1), sp), 0, 15)


def lvf(V):
    vm, vn = V.max(1, keepdims=True), V.min(1, keepdims=True)
    sp = np.maximum(vm - vn, 1e-9)
    return np.clip(np.rint((V - vn) / sp * 15), 0, 15).astype(int)


ref = lvf(np.clip(z2, 0, 15))
Zc = np.clip(Z, B0, B0 + int(round(15 * beta)))
d = lvi(Zc) - ref
print(f"  截断到 [B_0, B_0+15β] 后：不同像素 {100*(d!=0).mean():.2f}%  "
      f"平均|差| {np.abs(d).mean():.3f}  ≥2 级的脸 {100*(np.abs(d).max(1)>=2).mean():.2f}%")
print(f"  逐通道的 B_p 已把 φ_p 造成的地板差消掉，所以截断用两个全局常数即可。")
print("STEP5B_DONE")
