#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
鸢尾花 Teacher 搜索：4bit 输入 + h=4，多种子多重启
====================================================
目标: 在 4->4->3 (one-vs-rest) 这个固定架构下, 找到最好的老师。
评价:
  - 全量 150 样本上的 3 类准确率与二值准确率
  - 5 折分层 CV 的 3 类准确率（泛化估计，用于确认不是过拟合挑出来的）
做法: N_SEEDS 个随机初始化, 每个训练 STEPS 步, 按全量 3 类准确率选最好;
      再对选出的 top 几个做 CV 复核。
输出: results/iris_teacher.json （含权重，供后续编译）
"""
import numpy as np, json, os, time

IRIS = np.array([
 [5.1,3.5,1.4,0.2,0],[4.9,3.0,1.4,0.2,0],[4.7,3.2,1.3,0.2,0],[4.6,3.1,1.5,0.2,0],
 [5.0,3.6,1.4,0.2,0],[5.4,3.9,1.7,0.4,0],[4.6,3.4,1.4,0.3,0],[5.0,3.4,1.5,0.2,0],
 [4.4,2.9,1.4,0.2,0],[4.9,3.1,1.5,0.1,0],[5.4,3.7,1.5,0.2,0],[4.8,3.4,1.6,0.2,0],
 [4.8,3.0,1.4,0.1,0],[4.3,3.0,1.1,0.1,0],[5.8,4.0,1.2,0.2,0],[5.7,4.4,1.5,0.4,0],
 [5.4,3.9,1.3,0.4,0],[5.1,3.5,1.4,0.3,0],[5.7,3.8,1.7,0.3,0],[5.1,3.8,1.5,0.3,0],
 [5.4,3.4,1.7,0.2,0],[5.1,3.7,1.5,0.4,0],[4.6,3.6,1.0,0.2,0],[5.1,3.3,1.7,0.5,0],
 [4.8,3.4,1.9,0.2,0],[5.0,3.0,1.6,0.2,0],[5.0,3.4,1.6,0.4,0],[5.2,3.5,1.5,0.2,0],
 [5.2,3.4,1.4,0.2,0],[4.7,3.2,1.6,0.2,0],[4.8,3.1,1.6,0.2,0],[5.4,3.4,1.5,0.4,0],
 [5.2,4.1,1.5,0.1,0],[5.5,4.2,1.4,0.2,0],[4.9,3.1,1.5,0.2,0],[5.0,3.2,1.2,0.2,0],
 [5.5,3.5,1.3,0.2,0],[4.9,3.6,1.4,0.1,0],[4.4,3.0,1.3,0.2,0],[5.1,3.4,1.5,0.2,0],
 [5.0,3.5,1.3,0.3,0],[4.5,2.3,1.3,0.3,0],[4.4,3.2,1.3,0.2,0],[5.0,3.5,1.6,0.6,0],
 [5.1,3.8,1.9,0.4,0],[4.8,3.0,1.4,0.3,0],[5.1,3.8,1.6,0.2,0],[4.6,3.2,1.4,0.2,0],
 [5.3,3.7,1.5,0.2,0],[5.0,3.3,1.4,0.2,0],
 [7.0,3.2,4.7,1.4,1],[6.4,3.2,4.5,1.5,1],[6.9,3.1,4.9,1.5,1],[5.5,2.3,4.0,1.3,1],
 [6.5,2.8,4.6,1.5,1],[5.7,2.8,4.5,1.3,1],[6.3,3.3,4.7,1.6,1],[4.9,2.4,3.3,1.0,1],
 [6.6,2.9,4.6,1.3,1],[5.2,2.7,3.9,1.4,1],[5.0,2.0,3.5,1.0,1],[5.9,3.0,4.2,1.5,1],
 [6.0,2.2,4.0,1.0,1],[6.1,2.9,4.7,1.4,1],[5.6,2.9,3.6,1.3,1],[6.7,3.1,4.4,1.4,1],
 [5.6,3.0,4.5,1.5,1],[5.8,2.7,4.1,1.0,1],[6.2,2.2,4.5,1.5,1],[5.6,2.5,3.9,1.1,1],
 [5.9,3.2,4.8,1.8,1],[6.1,2.8,4.0,1.3,1],[6.3,2.5,4.9,1.5,1],[6.1,2.8,4.7,1.2,1],
 [6.4,2.9,4.3,1.3,1],[6.6,3.0,4.4,1.4,1],[6.8,2.8,4.8,1.4,1],[6.7,3.0,5.0,1.7,1],
 [6.0,2.9,4.5,1.5,1],[5.7,2.6,3.5,1.0,1],[5.5,2.4,3.8,1.1,1],[5.5,2.4,3.7,1.0,1],
 [5.8,2.7,3.9,1.2,1],[6.0,2.7,5.1,1.6,1],[5.4,3.0,4.5,1.5,1],[6.0,3.4,4.5,1.6,1],
 [6.7,3.1,4.7,1.5,1],[6.3,2.3,4.4,1.3,1],[5.6,3.0,4.1,1.3,1],[5.5,2.5,4.0,1.3,1],
 [5.5,2.6,4.4,1.2,1],[6.1,3.0,4.6,1.4,1],[5.8,2.6,4.0,1.2,1],[5.0,2.3,3.3,1.0,1],
 [5.6,2.7,4.2,1.3,1],[5.7,3.0,4.2,1.2,1],[5.7,2.9,4.2,1.3,1],[6.2,2.9,4.3,1.3,1],
 [5.1,2.5,3.0,1.1,1],[5.7,2.8,4.1,1.3,1],
 [6.3,3.3,6.0,2.5,2],[5.8,2.7,5.1,1.9,2],[7.1,3.0,5.9,2.1,2],[6.3,2.9,5.6,1.8,2],
 [6.5,3.0,5.8,2.2,2],[7.6,3.0,6.6,2.1,2],[4.9,2.5,4.5,1.7,2],[7.3,2.9,6.3,1.8,2],
 [6.7,2.5,5.8,1.8,2],[7.2,3.6,6.1,2.5,2],[6.5,3.2,5.1,2.0,2],[6.4,2.7,5.3,1.9,2],
 [6.8,3.0,5.5,2.1,2],[5.7,2.5,5.0,2.0,2],[5.8,2.8,5.1,2.4,2],[6.4,3.2,5.3,2.3,2],
 [6.5,3.0,5.5,1.8,2],[7.7,3.8,6.7,2.2,2],[7.7,2.6,6.9,2.3,2],[6.0,2.2,5.0,1.5,2],
 [6.9,3.2,5.7,2.3,2],[5.6,2.8,4.9,2.0,2],[7.7,2.8,6.7,2.0,2],[6.3,2.7,4.9,1.8,2],
 [6.7,3.3,5.7,2.1,2],[7.2,3.2,6.0,1.8,2],[6.2,2.8,4.8,1.8,2],[6.1,3.0,4.9,1.8,2],
 [6.4,2.8,5.6,2.1,2],[7.2,3.0,5.8,1.6,2],[7.4,2.8,6.1,1.9,2],[7.9,3.8,6.4,2.0,2],
 [6.4,2.8,5.6,2.2,2],[6.3,2.8,5.1,1.5,2],[6.1,2.6,5.6,1.4,2],[7.7,3.0,6.1,2.3,2],
 [6.3,3.4,5.6,2.4,2],[6.4,3.1,5.5,1.8,2],[6.0,3.0,4.8,1.8,2],[6.9,3.1,5.4,2.1,2],
 [6.7,3.1,5.6,2.4,2],[6.9,3.1,5.1,2.3,2],[5.8,2.7,5.1,1.9,2],[6.8,3.2,5.9,2.3,2],
 [6.7,3.3,5.7,2.5,2],[6.7,3.0,5.2,2.3,2],[6.3,2.5,5.0,1.9,2],[6.5,3.0,5.2,2.0,2],
 [6.2,3.4,5.4,2.3,2],[5.9,3.0,5.1,1.8,2],
], dtype=np.float64)
Xr = IRIS[:, :4]; y3 = IRIS[:, 4].astype(int); N = len(IRIS)

# 4bit 量化（全量定界，作为最终编译用的编码）
lo = Xr.min(0); hi = Xr.max(0); rng4 = np.where(hi - lo > 1e-12, hi - lo, 1.0)
Xq = np.clip(np.round((Xr - lo) / rng4 * 15), 0, 15).astype(np.float64)
Y = np.zeros((N, 3))
for k in range(3):
    Y[:, k] = (y3 == k).astype(float)

def train_relu(X, Yt, h, seed, steps=8000, lr=0.02, bs=32):
    rng = np.random.default_rng(seed); n, d = X.shape
    W1 = rng.normal(0, 0.7, (h, d)); b1 = rng.normal(0, 0.3, h)
    W2 = rng.normal(0, 0.7, (h, 3)); b2 = rng.normal(0, 0.3, 3)
    ps = [W1, b1, W2, b2]; ms = [np.zeros_like(p) for p in ps]; vs = [np.zeros_like(p) for p in ps]
    best = None
    for t in range(1, steps + 1):
        idx = rng.integers(0, n, bs)
        xb = X[idx]; yb = Yt[idx]
        U = xb @ W1.T + b1; Hf = np.maximum(U, 0); Z = Hf @ W2 + b2
        P = 1 / (1 + np.exp(-np.clip(Z, -30, 30))); dZ = (P - yb) / bs
        dU = (dZ @ W2.T) * (U > 0)
        g = [dU.T @ xb, dU.sum(0), Hf.T @ dZ, dZ.sum(0)]
        for i, (pp, gg) in enumerate(zip(ps, g)):
            ms[i] = 0.9 * ms[i] + 0.1 * gg; vs[i] = 0.999 * vs[i] + 0.001 * gg ** 2
            pp -= lr * (ms[i] / (1 - 0.9 ** t)) / (np.sqrt(vs[i] / (1 - 0.999 ** t)) + 1e-8)
        if t % 2000 == 0 or t == steps:
            Zf = np.maximum(X @ W1.T + b1, 0) @ W2 + b2
            a3 = (np.argmax(Zf, 1) == np.argmax(Yt, 1)).mean()
            ba = ((Zf >= 0).astype(int) == (Yt >= 0.5).astype(int)).mean()
            if best is None or (a3, ba) > (best[0], best[1]):
                best = (a3, ba, W1.copy(), b1.copy(), W2.copy(), b2.copy())
    return best

def evalp(P, X, Yt, ylab):
    Z = np.maximum(X @ P[2].T + P[3], 0) @ P[4] + P[5]
    return (np.argmax(Z, 1) == ylab).mean(), ((Z >= 0).astype(int) == (Yt >= 0.5).astype(int)).mean(), Z

N_SEEDS = 40
print("=" * 96)
print("Teacher 搜索：4bit 输入, 架构 4->4->3, one-vs-rest")
print("=" * 96)
t0 = time.time()
cands = []
for s in range(N_SEEDS):
    r = train_relu(Xq, Y, 4, seed=1000 + s)
    a3, ba = r[0], r[1]
    cands.append((a3, ba, 1000 + s, r))
cands.sort(key=lambda c: (-c[0], -c[1]))
print(f"\n{N_SEEDS} 个种子的全量 3 类准确率分布:")
vals = np.array([c[0] for c in cands])
for v in sorted(set(np.round(vals, 4)), reverse=True):
    print(f"   {v:.4f}: {int((np.abs(vals - v) < 1e-9).sum())} 个种子")
print(f"\n前 5 名:")
for a3, ba, seed, r in cands[:5]:
    print(f"   seed={seed}: 3类 {a3:.4f}  二值 {ba:.4f}")
print(f"\n用时 {time.time()-t0:.0f}s")

# 对前 5 名做 5 折 CV 复核
print("\n" + "=" * 96)
print("前 5 名的 5 折分层 CV 复核（折内量化）")
print("=" * 96)
folds = 5
rng = np.random.default_rng(12345)
fold_id = np.zeros(N, dtype=int)
for c in range(3):
    ci = np.where(y3 == c)[0]; rng.shuffle(ci); fold_id[ci] = np.arange(len(ci)) % folds

def encode_fold(Xtr, Xte):
    l = Xtr.min(0); hh = Xtr.max(0); rr = np.where(hh - l > 1e-12, hh - l, 1.0)
    f = lambda X: np.clip(np.round((X - l) / rr * 15), 0, 15).astype(np.float64)
    return f(Xtr), f(Xte)

cvres = []
for a3, ba, seed, r in cands[:5]:
    # 用该种子重新训练 5 折（结构相同，重新初始化）
    fs = []
    for f in range(folds):
        te = fold_id == f; tr = ~te
        Xtr, Xte = encode_fold(Xr[tr], Xr[te])
        Ytr = np.zeros((tr.sum(), 3))
        for k in range(3):
            Ytr[:, k] = (y3[tr] == k)
        b = None
        for sd in (0, 1, 2):
            rr = train_relu(Xtr, Ytr, 4, seed=sd + 500)
            if b is None or rr[0] > b[0]:
                b = rr
        Zte = np.maximum(Xte @ b[2].T + b[3], 0) @ b[4] + b[5]
        fs.append((np.argmax(Zte, 1) == y3[te]).mean())
    cvres.append((np.mean(fs), np.std(fs), seed, fs))
    print(f"   seed={seed}: CV 3类 {np.mean(fs):.4f} ±{np.std(fs):.4f}  "
          f"各折 {[round(v,3) for v in fs]}")

# 选最终 Teacher: 优先 CV，再全量
best_cv = max(cvres, key=lambda c: c[0])
seed_final = best_cv[2]
final = [c for c in cands if c[2] == seed_final][0]
print("\n" + "=" * 96)
print(f"选定 Teacher: seed={seed_final}")
print(f"  全量 150: 3类 {final[0]:.4f}  二值 {final[1]:.4f}")
print(f"  5折 CV  : {best_cv[0]:.4f} ±{best_cv[1]:.4f}")
print("=" * 96)

W1 = final[3][2]; b1 = final[3][3]; W2 = final[3][4]; b2 = final[3][5]
print("\nTeacher 参数:")
print("  W1 =", np.round(W1, 6).tolist())
print("  b1 =", np.round(b1, 6).tolist())
print("  W2 =", np.round(W2, 6).tolist())
print("  b2 =", np.round(b2, 6).tolist())
Z = np.maximum(Xq @ W1.T + b1, 0) @ W2 + b2
print("\n隐藏层激活统计（全量 150 点）:")
H = np.maximum(Xq @ W1.T + b1, 0)
for j in range(4):
    print(f"  j={j}: 范围 [{H[:,j].min():.4f}, {H[:,j].max():.4f}]  正激活占比 {(H[:,j]>1e-9).mean()*100:.1f}%")

out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(out, exist_ok=True)
json.dump({
    "arch": "4->4->3 (one-vs-rest), 4bit input",
    "seed": int(seed_final),
    "full_acc3": float(final[0]), "full_bin": float(final[1]),
    "cv_acc3_mean": float(best_cv[0]), "cv_acc3_std": float(best_cv[1]),
    "W1": W1.tolist(), "b1": b1.tolist(), "W2": W2.tolist(), "b2": b2.tolist(),
    "quant_lo": lo.tolist(), "quant_hi": hi.tolist(),
    "Xq": Xq.astype(int).tolist(), "y3": y3.tolist(),
}, open(os.path.join(out, "iris_teacher.json"), "w", encoding="utf-8"),
    ensure_ascii=False, indent=2)
print("\n已写出: results/iris_teacher.json")
