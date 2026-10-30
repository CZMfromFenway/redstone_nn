#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
按真正的投掷器约束重算。

纠正三处
    1. 容器不只能覆盖 W^(6)。最高位 W^(7)=2·w_ref 可以用比较器环突破 897，
       只是体积大，所以要落在最外侧。布置时可安排最高位与次高位贴边。
       → 「w_ref ≤ 448」是我错了。
    2. 投掷器里任何物品不得超过其正常堆叠数。所以
           不可堆叠：每槽恰好 1 个（A = a 固定，没有自由度）
           十六堆叠：每槽 1..16 个（B ∈ [b, 16b]）
           可堆叠  ：每槽 1..64 个（C ∈ [c, 64c]）
       → 我上一版「不可堆叠×N」的调法在投掷器里是非法的。
    3. 投掷器接空漏斗，喷出的物品会回去，不需要 setblock 复位。

另外：a,b,c 里有两个 0 的形状退化成单个木桶/漏斗能解决的问题，不讨论。
所以只看 a,b,c >= 1 的 28 种。
"""
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
N = 9
CAP = 64 * N          # 576


def f(U):
    return 0 if U <= 0 else (14 * U) // CAP


def good(U):
    if U <= 0:
        return False
    return (f(U) - f(max(0, U - 64)) == 2 and
            f(U) - f(max(0, U - 4)) == 1 and
            f(U) - f(max(0, U - 1)) == 0)


def sig(U):
    return 0 if U == 0 else 1 + (14 * U) // CAP


print("=" * 96)
print("投掷器：a,b,c 全部 >= 1 的 28 种形状")
print("=" * 96)
print("  U = 64·a + 4·B + C，  B ∈ [b, 16b]（十六堆叠每槽最多 16 个）")
print("                        C ∈ [c, 64c]（可堆叠每槽最多 64 个）")
print("  条件：7U mod 288 ∈ [7, 27]")
print()

rows = []
for a in range(1, 8):
    for b in range(1, 9 - a):
        c = 9 - a - b
        if c < 1:
            continue
        base = 64 * a
        found = None
        for B in range(b, 16 * b + 1):
            U0 = base + 4 * B
            for C in range(c, 64 * c + 1):
                U = U0 + C
                if good(U):
                    found = (B, C, U)
                    break
            if found:
                break
        rows.append((a, b, c, base + 4 * b + c, found))

print(f"  {'(a,b,c)':>10s} {'B范围':>9s} {'C范围':>10s} {'最小 U':>7s} {'可行?':>6s} "
      f"{'一个可行配置':>22s} {'U':>5s} {'信号':>5s}")
print("  " + "-" * 88)
ok_cnt = 0
for a, b, c, Umin, found in rows:
    if found:
        ok_cnt += 1
        B, C, U = found
        cfg = f"十六×{B} 可堆叠×{C}"
        print(f"  {str((a,b,c)):>10s} {f'{b}..{16*b}':>9s} {f'{c}..{64*c}':>10s} "
              f"{Umin:7d} {'✓':>6s} {cfg:>22s} {U:5d} {sig(U):5d}")
    else:
        print(f"  {str((a,b,c)):>10s} {f'{b}..{16*b}':>9s} {f'{c}..{64*c}':>10s} "
              f"{Umin:7d} {'✗':>6s} {'无解':>22s} {'':>5s} {'':>5s}")

print()
print(f"  可行 {ok_cnt}/{len(rows)} 种")
if ok_cnt < len(rows):
    print(f"  不可行：{[ (a,b,c) for a,b,c,_,fnd in rows if not fnd ]}")

# 逐类分析为什么不可行
print()
print("=" * 96)
print("不可行的原因")
print("=" * 96)
for a, b, c, Umin, found in rows:
    if found:
        continue
    # 扫一遍看 7U mod 288 的落点
    res = set()
    for B in range(b, 16 * b + 1):
        for C in range(c, 64 * c + 1):
            res.add((7 * (64 * a + 4 * B + C)) % 288)
    hit = sorted(r for r in res if 7 <= r <= 27)
    print(f"  ({a},{b},{c})  U ∈ [{Umin}, {64*a+64*b+64*c}]  "
          f"7U mod 288 取 {len(res)} 种值，落在 [7,27] 的有 {len(hit)}")

# ---------------- 木桶装位权重：w_ref 的新约束
print()
print("=" * 96)
print("w_ref 的定义与约束（修正）")
print("=" * 96)
print(r"""
  定义：编译器唯一的全局尺度参数。
        sigma = w_ref / (max|w| · delta · 2^(b-2))
      于是
        |W^(b-2)| = w_ref          （次高位 = w_ref）
        |W^(b-1)| = 2·w_ref        （最高位 = 2 w_ref）

  约束：
    次高位  必须落在一个普通容器里  ->  w_ref <= 897
    最高位  2·w_ref 超过 897 时用比较器环，可突破，但体积大
            -> 只能放在最外侧；布线时把最高位与次高位安排到边上

  所以 w_ref 的可行上限是 897，不是我上一轮说的 448。
""")
for w in (882, 897):
    print(f"    w_ref={w}:  次高位 {w} "
          f"{'可入桶' if w <= 897 else '不可'};  最高位 {2*w} "
          f"{'需比较器环' if 2*w > 897 else '可入桶'}")

print()
print("  注：882 与 897 之差 = 15，来自 LUT 的阶跃公式 Δλ = 15 − b。")
print("      位权重若直接当容器内容，上限 897；")
print("      若当作 LUT 的阶跃量，上限 882。这条取决于符号怎么处理（未设计）。")

json.dump({"dropper_shapes_ok": [[a, b, c] for a, b, c, _, f2 in rows if f2],
           "dropper_shapes_fail": [[a, b, c] for a, b, c, _, f2 in rows if not f2],
           "w_ref_note": "w_ref <= 897; top bit 2*w_ref may use a comparator ring"},
          open(os.path.join(RESULTS, "dropper_shapes.json"), "w",
               encoding="utf-8"), indent=1)
print()
print("  已写入 results/dropper_shapes.json")
print("REDO_DONE")
