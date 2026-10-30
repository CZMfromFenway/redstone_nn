#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
鸢尾花编译 v3（修正阈值符号 + 逐项自检）
==========================================
代数关系（关键）:
  logit_k = sum_j W2_kj * r_j + b2_k
          = sum_j A_kj * v_j + S * b2_k          （A_kj 已含 R_j/lv 因子）
  即  logit = Z @ A.T + S*b2 ,  判据 logit >= 0
"""
import numpy as np, json, os

d = json.load(open(r'D:\czm我的世界\文件\Comparator Network\StrongAnalog_LUT_Network\results\iris_teacher.json', encoding='utf-8'))
W1f = np.array(d['W1']); b1f = np.array(d['b1'])
W2f = np.array(d['W2']); b2f = np.array(d['b2'])
CAP = 868.0; x = np.arange(16)

Xall = np.indices((16,) * 4, dtype=np.int64).reshape(4, -1).T
Uf = Xall.astype(np.float64) @ W1f.T + b1f[None, :]
alive = [j for j in range(W1f.shape[0]) if np.maximum(Uf[:, j], 0).max() > 1e-9]
W1 = W1f[alive]; b1 = b1f[alive]; W2 = W2f[alive]
h = len(alive); N = len(Xall)
print("=" * 96)
print(f"鸢尾花编译 v3: 4 -> {h} -> 3")
print("=" * 96)

# ---- 第一层 ----
rho = np.zeros((h, 4))
for j in range(h):
    mx = np.abs(W1[j]).max()
    rho[j] = CAP * np.abs(W1[j]) / mx
up = np.round(x[None, None, :] * rho[:, :, None]).astype(np.int64)
dn = np.round(15 * rho[:, :, None] - x[None, None, :] * rho[:, :, None]).astype(np.int64)
off = np.where(W1 > 0, 0, np.round(15 * rho)).astype(np.int64)
Xi = Xall.astype(np.int64)
Q = np.zeros((N, h), dtype=np.int64)
for j in range(h):
    for i in range(4):
        if abs(W1[j, i]) <= 1e-12:
            continue
        Q[:, j] += (up if W1[j, i] > 0 else dn)[j, i, Xi[:, i]] - off[j, i]
a = np.array([np.abs(W1[j]).max() for j in range(h)]) / CAP
U = Q * a[None, :] + b1[None, :]
Utrue = Xall.astype(np.float64) @ W1.T + b1[None, :]
print(f"[第一层] max|U_lut-U_true| = {np.max(np.abs(U-Utrue)):.4f}")
print(f"[可建造] lam_up[{up.min()},{up.max()}] lam_dn[{dn.min()},{dn.max()}] "
      f"负值 {int((up<0).sum())+int((dn<0).sum())}; rho_max {rho.max():.1f}<=897")

# ---- 中间层 + 输出层 ----
def build(bits, S):
    lv = (1 << bits) - 1
    rr = np.maximum(U, 0)
    R = rr.max(axis=0)
    v = np.clip(np.round(lv * rr / R[None, :]), 0, lv).astype(np.int64)
    if bits == 4:
        Z = v.astype(np.float64); Rj = R
    else:
        Z = np.concatenate([(v >> 4), (v & 0xF)], axis=1).astype(np.float64)
        Rj = np.concatenate([R, R])
    Wt = ((R / lv)[None, :] * W2.T)                 # (3,h)
    if bits == 4:
        A = np.round(S * Wt)
    else:
        A = np.concatenate([np.round(S * 16 * Wt), np.round(S * Wt)], axis=1)
    th = np.round(S * b2f)                          # logit = Z@A + th
    return Z, A, th, R, lv

Yt = (np.maximum(U, 0) @ W2 + b2f >= 0).astype(int)
print(f"\nteacher 二值目标 正类数 {[int(Yt[:,k].sum()) for k in range(3)]} / {N}")

print("\n" + "=" * 96)
print("编译结果（全 65536 状态）")
print("=" * 96)
res = {}
for bits in (4, 8):
    print(f"\n--- 中间层 {bits} bit {'(1 条线)' if bits==4 else '(2 条线, 拆分)'} ---")
    best = None
    for S in (25, 50, 100, 200, 400, 800, 1600, 3200):
        Z, A, th, R, lv = build(bits, S)
        pred = (Z @ A.T + th[None, :] >= 0).astype(int)
        err = int(np.sum(pred != Yt)); wm = int(np.abs(A).max())
        ok = "OK" if wm <= 897 else "越界"
        print(f"  S={S:5d} 权重max={wm:6d} {ok:>6}  错误 {err:7d}/{N}  ({err/N*100:7.4f}%)")
        if wm <= 897 and (best is None or err < best[1]):
            best = (S, err, wm, A, th, R)
    if best:
        S, err, wm, A, th, R = best
        Xq = np.array(d['Xq'], dtype=np.int64); y3 = np.array(d['y3'])
        idx150 = Xq[:, 0] * 4096 + Xq[:, 1] * 256 + Xq[:, 2] * 16 + Xq[:, 3]
        Z, A, th, R, lv = build(bits, S)
        logit = Z[idx150] @ A.T + th[None, :]
        acc3 = (np.argmax(logit, 1) == y3).mean()
        print(f"  => 最优 S={S}: 全空间错误 {err} ({err/N*100:.4f}%), 权重max {wm}")
        print(f"     150 点 3 类准确率 = {acc3:.4f}")
        print(f"     R_j = {np.round(R,6).tolist()}   R_j/lv = {np.round(R/lv,6).tolist()}")
        res[f"mid{bits}bit"] = {"S": int(S), "err": int(err), "wmax": int(wm),
                                "acc3_150": float(acc3), "R": R.tolist(),
                                "A": A.tolist(), "th": th.tolist()}

json.dump({"alive": [int(v) for v in alive], "results": res,
           "W1": W1.tolist(), "b1": b1.tolist(), "W2": W2.tolist(), "b2": b2f.tolist()},
          open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "results", "iris_compiled_v3.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print("\n已写出: results/iris_compiled_v3.json")
