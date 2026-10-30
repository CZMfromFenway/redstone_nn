#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
最终验证：可建造阶梯（正→反式上升 / 负→顺式下降）-> 8bit 拆分 -> 整数读出
==========================================================================
形式已经过仿射自检（残差 ~3e-4，与量化噪声同量级）:
  rho_{j,i} = 868*|W1_ji| / max_i|W1_ji|
  W>0: lam_up(x)=round(x*rho)           直加   (反式, x=0 -> 0)
  W<0: lam_dn(x)=round(15*rho - x*rho)  直加   (顺式, x=15 -> 0), 非负
  S_j = sum_i lam(x_i)
  u_j = a_j*S_j + c_j         (a_j 由最小二乘定, 理论 = max_i|W1_ji|/(868*7.5))
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

print("=" * 92)
print("最终验证：正->反式上升 / 负->顺式下降，直接相加")
print("=" * 92)
print(f"  lam_up [{lam_up.min()}, {lam_up.max()}]  负值 {int((lam_up<0).sum())}")
print(f"  lam_dn [{lam_dn.min()}, {lam_dn.max()}]  负值 {int((lam_dn<0).sum())}")
print(f"  满档 15*rho_max = {15*rho.max():.0f} <= 13455 ? {15*rho.max()<=13455}  (容器 897 每档)")
print(f"  单档最大 rho = {rho.max():.1f} <= 897 ? {rho.max()<=897}")

S = np.zeros((N, h))
for j in range(h):
    for i in range(4):
        if abs(W1[j, i]) <= 1e-12:
            continue
        S[:, j] += (lam_up if W1[j, i] > 0 else lam_dn)[j, i, Xi[:, i]]

U = np.zeros((N, h)); res = []
mx = np.array([np.abs(W1[j]).max() for j in range(h)])
for j in range(h):
    A = np.hstack([S[:, j:j + 1], np.ones((N, 1))])
    c, *_ = np.linalg.lstsq(A, Utrue[:, j], rcond=None)
    U[:, j] = A @ c
    res.append(np.max(np.abs(A @ c - Utrue[:, j])))
    theo = mx[j] / (CAP * 7.5)
    print(f"  j={j}: 斜率={c[0]:.9f}  理论={theo:.9f}  比值={c[0]/theo:.6f}  残差={res[-1]:.2e}")
print(f"\n  自检: 残差 max = {max(res):.2e}  {'通过 ✓' if max(res)<1e-2 else '失败'}")
if max(res) >= 1e-2:
    sys.exit(1)

print(f"  第一层 LUT(第二层浮点, 参照, 不可建造) 错误 = "
      f"{int(((np.tanh(U)@W2+b2>=0).astype(np.int8)!=lab).sum())}")

Hc = np.tanh(U)
v8 = np.clip(np.round((Hc + 1) * 255 / 2), 0, 255).astype(np.int64)
Z = np.concatenate([(v8 >> 4), (v8 & 0xF)], axis=1).astype(np.float64)
wmax = np.abs(W2).max(); Sopt = 897 * 255 / (32 * wmax)
print(f"\n[按 czm 规则: 由 max|W2|={wmax:.4f} 反解 S]  max|A_hi| = 897  =>  S = {Sopt:.0f}")
print(f"  {'S':>7} | {'A_hi_max':>8} | {'A_lo_max':>8} | {'错误':>6} | {'错误率':>9} | 容器")
print("  " + "-" * 64)
resd = {}
for S_ in (587, 1024, 2000, 4000, 8000, int(Sopt), 128000):
    A_hi = np.round(S_ * 32 * W2 / 255); A_lo = np.round(S_ * 2 * W2 / 255)
    w = np.concatenate([A_hi, A_lo]); th = -S_ * (b2 - W2.sum())
    e = int(((Z @ w - th >= 0).astype(np.int8) != lab).sum())
    wm = int(np.abs(w).max())
    print(f"  {S_:7d} | {np.abs(A_hi).max():8.0f} | {np.abs(A_lo).max():8.0f} | "
          f"{e:6d} | {e/N*100:8.4f}% | {'OK' if wm<=897 else '越界%d'%wm}")
    resd[str(S_)] = e
json.dump({"N": N, "resid": float(max(res)), "results": resd},
          open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "results", "final_tanh_buildable.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print("\n已写出: results/final_tanh_buildable.json")
