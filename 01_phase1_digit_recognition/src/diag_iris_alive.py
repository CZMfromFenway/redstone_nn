#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""查清: 各神经元 ReLU 后的实际范围（哪些是死神经元）"""
import numpy as np, json

d = json.load(open(r'D:\czm我的世界\文件\Comparator Network\StrongAnalog_LUT_Network\results\iris_teacher.json', encoding='utf-8'))
W1 = np.array(d['W1']); b1 = np.array(d['b1'])
W2 = np.array(d['W2']); b2 = np.array(d['b2'])
Xq = np.array(d['Xq'], dtype=np.int64); y3 = np.array(d['y3'])
h = W1.shape[0]

Xall = np.indices((16,) * 4, dtype=np.int64).reshape(4, -1).T
Utrue_all = Xall.astype(np.float64) @ W1.T + b1[None, :]
Uq = Xq.astype(np.float64) @ W1.T + b1[None, :]

print("=" * 92)
print("各神经元 ReLU 后的实际范围")
print("=" * 92)
print(f"{'j':>3} | {'u范围(150点)':>22} | {'relu范围(150点)':>18} | {'relu范围(全空间)':>18} | 状态")
print("-" * 92)
for j in range(h):
    ru = np.maximum(Uq[:, j], 0); ra = np.maximum(Utrue_all[:, j], 0)
    st = "死" if ra.max() < 1e-9 else "活"
    print(f"{j:3d} | [{Uq[:,j].min():8.3f},{Uq[:,j].max():8.3f}] | "
          f"[{ru.min():7.3f},{ru.max():7.3f}] | [{ra.min():7.3f},{ra.max():7.3f}] | {st}")
print("-" * 92)
alive = [j for j in range(h) if np.maximum(Utrue_all[:, j], 0).max() > 1e-9]
print(f"全空间上存活的神经元: {alive}  （共 {len(alive)}/{h}）")
print()
print("=> 这说明 teacher 的真实结构只有", len(alive), "个有效隐藏单元")
print()
# 用只有存活神经元的等价网络重算
W2e = W2[alive]; b2e = b2
H = np.maximum(Uq[:, alive], 0)
Z = H @ W2e + b2e
print(f"等价网络 4->{len(alive)}->3 在 150 点上的 3 类准确率 = {(np.argmax(Z,1)==y3).mean():.4f}")
