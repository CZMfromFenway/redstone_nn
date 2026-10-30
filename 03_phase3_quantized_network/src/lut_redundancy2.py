#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
LUT 列的容器冗余，按正确的截断方式分符号计算。

两种截断（doc 05 §3.2 / §4.2）
    红石粉：Δλ(x) = 15 − b_{x+1}         窗口 [−882, +15]
    比较器：Δλ(x) = (29−x) − b_{x+1}     窗口 [−868, +29−x]

所以
    负权重（下降阶梯，Δλ<0） → 只能用红石粉，b = 15 − Δλ
    正权重（上升阶梯，Δλ>0） → 只能用比较器，b = (29−x) − Δλ
                              （红石粉正窗口只有 +15，装不下大步长）
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
P1 = os.path.join(os.path.dirname(ROOT), "01_phase1_digit_recognition")
RESULTS = os.path.join(ROOT, "results")

d = json.load(open(os.path.join(P1, "results", "iris_teacher.json"), encoding="utf-8"))
W1 = np.array(d["W1"], dtype=float)
CAP = 868.0
M, N = W1.shape

print("=" * 92)
print("一、按符号选正确的截断，算 15 个容器的取值")
print("=" * 92)
print(f"  {'神':>2s} {'入':>2s} {'W':>9s} {'ρ':>8s} {'截断':>6s} "
      f"{'容器值（前 8 个）':>30s} {'不同值':>6s} {'越界':>5s}")
print("  " + "-" * 82)
stat = {}
for j in range(M):
    for i in range(N):
        rho = CAP * abs(W1[j, i]) / np.abs(W1).max(axis=1)[j]
        lam = np.rint(np.arange(16) * rho)
        dlam = np.diff(lam)
        if W1[j, i] < 0:
            b = 15 - dlam
            mech = "粉"
        else:
            b = (29 - np.arange(15)) - dlam
            mech = "比"
        bad = int(((b < 0) | (b > 897)).sum())
        nd = len(set(b.tolist()))
        stat[(j, i)] = (mech, nd, bad)
        if i < 2:
            txt = "/".join(str(int(v)) for v in b[:8])
            print(f"  {j:2d} {i:2d} {W1[j,i]:9.4f} {rho:8.2f} {mech:>6s} "
                  f"{txt:>30s} {nd:6d} {bad:5d}")

cc = sum(1 for v in stat.values() if v[0] == "粉")
cb = sum(1 for v in stat.values() if v[0] == "比")
print(f"\n  红石粉列 {cc} 个，比较器列 {cb} 个")

print()
print("=" * 92)
print("二、两件事要分开")
print("=" * 92)
print("  (A) 容器值是否相同")
for mech, nm in (("粉", "红石粉（负权重）"), ("比", "比较器（正权重）")):
    nds = [v[1] for v in stat.values() if v[0] == mech]
    if nds:
        print(f"      {nm}: 平均每列 {np.mean(nds):.2f} 个不同值 / 15 个容器")
print("      红石粉：Δλ 只取 ⌊ρ⌋/⌈ρ⌉ 两值 ⇒ 容器值几乎全同")
print("      比较器：Δλ 相同，但 (29−x) 每步减 1 ⇒ 容器值呈等差递减，15 个都不同")
print()
print("  (B) 信息内容（这一条才是本质）")
print("      LUT 列的 15 个容器值由 1 个整数（斜率 ρ）完全决定")
print("         ⇒ 每列信息量 = 1 个整数，占用 15 个容器")
print("      位权重的 b 个容器 = b 个互相独立的整数")
print("         ⇒ 每个容器都是独立自由度")
print()
print(f"  所以容器利用率的对比是：  LUT 1/15 = {100/15:.1f}%   vs   位权重 b/b = 100%")

print()
print("=" * 92)
print("三、但 LUT 也有它的道理")
print("=" * 92)
print("  15 个自由容器是「任意单调函数」的代价。若真需要 15 段折线，")
print("  那 15 个容器都在干活；我们只是恰好只需要线性，所以只用一个。")
print()
print("  ⇒ 结论不是「LUT 设计得差」，而是「LUT 的通用性对线性权重是浪费」。")

json.dump({"dust_columns": cc, "comparator_columns": cb,
           "lut_containers_per_column": 15,
           "free_parameters_per_column": 1},
          open(os.path.join(RESULTS, "lut_redundancy.json"), "w", encoding="utf-8"),
          indent=1)
print("\n  已写入 results/lut_redundancy.json")
print("REDUND2_DONE")
