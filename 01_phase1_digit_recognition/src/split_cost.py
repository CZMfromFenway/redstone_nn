#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
拆分层成本分析：8->16 要读的 Lambda 到底有多少个不同取值
==========================================================
问题: tanh 从 0x00 跳到 0x01 (或 0xFE -> 0xFF) 需要 Lambda 变化多少？
这决定了 8->16 拆分层的物理成本。
"""
import numpy as np, json, os

base = r'D:\czm我的世界\文件\Comparator Network'
rec = os.path.join(base, 'DT_Redstone_Research_Archive_20261001', 'DT_baseline_recovery')
m = json.load(open(os.path.join(rec, 'teachers', 'N4_MLP_T1.json'), encoding='utf-8'))
W1 = np.array(m['W1']); b1 = np.array(m['b1'])
X = np.indices((16,) * 4, dtype=np.int8).reshape(4, -1).T
Xi = X.astype(np.int64); N = len(X); h = 8; CAP = 868.0; x = np.arange(16)
Xn = (X.astype(float) - 7.5) / 7.5

rho = np.zeros((h, 4))
for j in range(h):
    rho[j] = CAP * np.abs(W1[j]) / np.abs(W1[j]).max()
up = np.round(x[None, None, :] * rho[:, :, None]).astype(np.int64)
dn = np.round(15 * rho[:, :, None] - x[None, None, :] * rho[:, :, None]).astype(np.int64)

S = np.zeros((N, h), dtype=np.int64)
for j in range(h):
    for i in range(4):
        if abs(W1[j, i]) <= 1e-12:
            continue
        S[:, j] += (up if W1[j, i] > 0 else dn)[j, i, Xi[:, i]]

mx = np.array([np.abs(W1[j]).max() for j in range(h)])
U = S * (mx / (CAP * 7.5))[None, :] - np.abs(W1).sum(axis=1)[None, :] + b1[None, :]
Hc = np.tanh(U)
v8 = np.clip(np.round((Hc + 1) * 255 / 2), 0, 255).astype(np.int64)

print("=" * 96)
print("拆分层要读的 Lambda: 不同取值数 vs 输出码值数")
print("=" * 96)
print(f"{'j':>4} | {'Lambda取值数':>12} | {'可达码值数':>10} | {'Lambda范围':>16} | {'压缩比':>8}")
print("-" * 96)
for j in range(8):
    nd = len(np.unique(S[:, j])); nc = len(np.unique(v8[:, j]))
    print(f"{j:4d} | {nd:12d} | {nc:10d} | {S[:,j].min():7d}~{S[:,j].max():7d} | "
          f"{nd/max(nc,1):7.1f}x")
print("-" * 96)

print("\n以 j=0 为例，相邻码值之间的 Lambda 间隔：")
Lam = S[:, 0]; codes = v8[:, 0]
ucs = np.unique(codes)
means = np.array([Lam[codes == c].mean() for c in ucs])
gaps = np.diff(means)
print(f"  前 5 格: {np.round(gaps[:5], 1).tolist()}")
mid = len(gaps) // 2
print(f"  中间 5 格: {np.round(gaps[mid-2:mid+3], 1).tolist()}")
print(f"  后 5 格: {np.round(gaps[-5:], 1).tolist()}")

print("\n" + "=" * 96)
print("关键结论")
print("=" * 96)
allg = []
for j in range(8):
    c = v8[:, j]; u = np.unique(c)
    mm = np.array([S[c == k, j].mean() for k in u])
    allg.append(np.diff(mm))
ag = np.concatenate(allg)
print(f"  相邻码值的 Lambda 间隔: 最小 {ag.min():.0f}  中位 {np.median(ag):.0f}  "
      f"平均 {ag.mean():.0f}  最大 {ag.max():.0f}")
print(f"  最大/最小 = {ag.max()/ag.min():.0f} 倍")
print()
print("  若拆分层用【等间隔取样】的查表:")
print(f"    输入步长必须 <= {ag.min():.0f} 才能分辨最窄的那一格")
print(f"    覆盖 0~{S.max()} 需要 {S.max()/ag.min():.0f} 个输入格")
print(f"    而输出只有 256 个码值 -> 表格规模被撑大 {S.max()/ag.min()/256:.1f} 倍")
print()
print("  若拆分层用【非等间隔断点】(按 tanh 逆函数布置):")
print("    只需 255 个断点 + 256 个输出值")
print("    但每个断点都要比较器判位 -> 255 个比较器的成本")
print()
print("  若用【分段线性近似】(中心区线性 + 两端钳位):")
print(f"    tanh 中心区近似线性, 两端饱和 -> 少数几段即可")
print("    这是最省的做法, 但精度会掉")
