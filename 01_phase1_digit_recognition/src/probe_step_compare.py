#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""官方前向 vs 我的 logit_real 逐步对照"""
import numpy as np, json, os
d = r"D:\czm我的世界\文件\Comparator Network\DT_Redstone_Research_Archive_20261001\DT_baseline_recovery\teachers"
X = np.indices((16,) * 4, dtype=np.int8).reshape(4, -1).T
lab = np.load(os.path.join(d, "N4_MLP_T1.npy")).astype(np.int64)
m = json.load(open(os.path.join(d, "N4_MLP_T1.json"), encoding="utf-8"))
W1 = np.array(m["W1"]); b1 = np.array(m["b1"]); W2 = np.array(m["W2"]); b2 = float(m["b2"])
Xn = (X.astype(np.float64) - 7.5) / 7.5

H = np.tanh(Xn @ W1.T + b1)              # tanh 输出
logit = H @ W2 + b2                       # 官方 logit
yp = (logit >= 0)
print("官方 logit: 正类 =", int(yp.sum()), " 不一致 =", int((yp != (lab > 0)).sum()))
print("logit 范围 =", logit.min(), logit.max())

# 我的口径: 用 bit
bits_real = (H >= 0).astype(np.int64)
logit_b = bits_real @ W2 + b2
yp_b = (logit_b >= 0)
print("\nbits_real @ W2 + b2: 正类 =", int(yp_b.sum()), " 与官方一致 =",
      int((yp_b != yp).sum()))
print("H>=0 的比例 =", float((H >= 0).mean()))
print("bits_real 与 H 的符号是否一致: ", np.array_equal(bits_real, (H >= 0).astype(np.int64)))

# H 的统计
print("\nH 统计: min=%.6f max=%.6f mean=%.6f  大于0的比例=%.4f"
      % (H.min(), H.max(), H.mean(), (H > 0).mean()))
print("H 的每一列 大于0 的比例:", [round(float((H[:, j] > 0).mean()), 4) for j in range(H.shape[1])])

# tanh 的近线性区: H 与预激活几乎同号, 但 tanh(H) 在 H 小的时候会削弱
pre = Xn @ W1.T + b1
print("\n预激活统计: min=%.6f max=%.6f" % (pre.min(), pre.max()))
print("预激活与 tanh 符号一致的比例 =", float((np.sign(pre) == np.sign(H)).mean()))
