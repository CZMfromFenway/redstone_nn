#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
最小验证 v3：用最小二乘定标，绕开手推归一化
=============================================
阶梯构造（可建造）:
  rho_{j,i} = 868*|W1_ji|/max_i|W1_ji|
  W>0: lam_up(x) = round(x*rho)                (反式, x=0 处 0)
  W<0: lam_dn(x) = round(15*rho - x*rho)       (顺式, x=15 处 0), 非负
折叠: W>0 取 lam_up ; W<0 取 (868*rho - lam_dn)     <- 两者都正比于 |W|*x/7.5
  fold_sum_j = sum_i 折叠值
再由最小二乘求 u = a_j * fold_sum_j + c_j （应为严格仿射，残差 = 量化噪声）
最后: tanh -> 8bit 拆分 -> 整数读出
"""
import numpy as np, json, os, sys

base = r'D:\czm我的世界\文件\Comparator Network'
rec = os.path.join(base, 'DT_Redstone_Research_Archive_20261001', 'DT_baseline_recovery')
m = json.load(open(os.path.join(rec, 'teachers', 'N4_MLP_T1.json'), encoding='utf-8'))
lab = np.load(os.path.join(rec, 'teachers', 'N4_MLP_T1.npy')).astype(np.int8)
W1 = np.array(m['W1']); b1 = np.array(m['b1'])
W2 = np.array(m['W2']); b2 = float(m['b2'])
X = np.indices((16,) * 4, dtype=np.int8).reshape(4, -1).T
Xi = X.astype(np.int64); N = len(X); h = 8
CAP = 868.0; x = np.arange(16)
Xn = (X.astype(np.float64) - 7.5) / 7.5
Utrue = Xn @ W1.T + b1

rho = np.zeros((h, 4))
for j in range(h):
    rho[j] = CAP * np.abs(W1[j]) / np.abs(W1[j]).max()
lam_up = np.round(x[None, None, :] * rho[:, :, None]).astype(np.int64)
lam_dn = np.round(15 * rho[:, :, None] - x[None, None, :] * rho[:, :, None]).astype(np.int64)

print("=" * 90)
print("可建造性自检 v3")
print("=" * 90)
print(f"  lam_up [{lam_up.min()}, {lam_up.max()}] 负值 {int((lam_up<0).sum())}")
print(f"  lam_dn [{lam_dn.min()}, {lam_dn.max()}] 负值 {int((lam_dn<0).sum())}")
print(f"  满档 15*rho_max = {15*rho.max():.0f} <= 13455 ? {15*rho.max()<=13455}")

fold = np.zeros((h, 4, 16), dtype=np.int64)
for j in range(h):
    for i in range(4):
        fold[j, i] = lam_up[j, i] if W1[j, i] > 0 else (int(round(CAP * rho[j, i])) - lam_dn[j, i])

# 逐坐标校验: fold 的斜率应与 |W1_ji|/7.5 成正比, 比例系数对同一神经元相同
slopes = np.zeros((h, 4))
for j in range(h):
    for i in range(4):
        slopes[j, i] = (fold[j, i, 1] - fold[j, i, 0]) if abs(W1[j, i]) > 1e-12 else 0.0
print("\n  逐坐标斜率比 (fold斜率 / |W1_ji|, 同一神经元内应恒定):")
allok = True
for j in range(h):
    r = [slopes[j, i] / abs(W1[j, i]) for i in range(4) if abs(W1[j, i]) > 1e-12]
    spread = (max(r) - min(r)) / max(r) if r else 0
    print(f"    j={j}: {[round(v,4) for v in r]}  相对离散 {spread*100:.2f}%")
    if spread > 0.02:
        allok = False
print(f"  => {'一致' if allok else '不一致'}")

# 用最小二乘定标
fold_sum = np.zeros((N, h))
for j in range(h):
    for i in range(4):
        if abs(W1[j, i]) <= 1e-12:
            continue
        fold_sum[:, j] += fold[j, i, Xi[:, i]]

U = np.zeros((N, h))
resid = []
for j in range(h):
    A = np.hstack([fold_sum[:, j:j+1], np.ones((N, 1))])
    c, *_ = np.linalg.lstsq(A, Utrue[:, j], rcond=None)
    U[:, j] = A @ c
    resid.append(np.max(np.abs(A @ c - Utrue[:, j])))
print(f"\n  最小二乘定标后 u 的残差 max = {max(resid):.3e}")
if max(resid) > 1e-2:
    print("  !! 残差过大, 折叠公式仍有误"); sys.exit(1)
print("  自检通过 ✓")

Hc = np.tanh(U)
print(f"\n  第二层浮点(不可建造, 参照) 错误 = {int(((Hc@W2+b2>=0).astype(np.int8)!=lab).sum())}")

v8 = np.clip(np.round((Hc + 1) * 255 / 2), 0, 255).astype(np.int64)
Z = np.concatenate([(v8 >> 4), (v8 & 0xF)], axis=1).astype(np.float64)
wmax = np.abs(W2).max(); Sopt = 897 * 255 / (32 * wmax)
print(f"\n[由 max|W2|={wmax:.4f} 反解] max|A_hi|=897 => S = {Sopt:.0f}")
print(f"  {'S':>7} | {'A_hi_max':>8} | {'A_lo_max':>8} | {'错误':>6} | {'错误率':>9} | 容器")
print("  " + "-" * 64)
res = {}
for S_ in (587, 2000, 4000, 8000, int(Sopt), 128000):
    A_hi = np.round(S_ * 32 * W2 / 255); A_lo = np.round(S_ * 2 * W2 / 255)
    w = np.concatenate([A_hi, A_lo]); th = -S_ * (b2 - W2.sum())
    e = int(((Z @ w - th >= 0).astype(np.int8) != lab).sum())
    wm = int(np.abs(w).max())
    print(f"  {S_:7d} | {np.abs(A_hi).max():8.0f} | {np.abs(A_lo).max():8.0f} | "
          f"{e:6d} | {e/N*100:8.4f}% | {'OK' if wm<=897 else '越界%d'%wm}")
    res[str(S_)] = e
json.dump({"N": N, "u_resid": float(max(resid)), "results": res},
          open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "results", "verify_buildable.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
