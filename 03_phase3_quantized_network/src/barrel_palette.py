#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
修正版调色板生成器：0..897 全覆盖。

上一版两个 bug
    1. signal=1 时取 fill_num = 0，生成空桶 → 回算 0。要 max(1, lo)。
    2. 865..896 段：我固定取「最小可行 fill」，余数 R>0 就要多占一格，
       于是需要 28 格，超过桶的 27 格。正确做法是在可行区间里搜一个
       (A, R) 组合，使 ceil(A/64) + [R>0] <= 27。

可达性
    令 fill_num = 64A + R，0 <= R <= 63。
    signal = 1 + (14*fill_num)//1728
    A 把斧占 ceil(A/64) 格；R > 0 时红石占 1 格。
"""
import json
import math
import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
FD = os.path.join(os.path.dirname(os.path.dirname(HERE)),
                  "01_phase1_digit_recognition", "datapack", "tiny_cnn",
                  "data", "vessel", "function")
RESULTS = os.path.join(HERE, "results")

ITEM_RE = re.compile(r"Slot:(\d+)b,count:(\d+)b?,?[^}]*id:\"minecraft:(\w+)\"")
STAT_RE = re.compile(r"signal: (\d+), item: (\d+)")
SLOTS = 27
DENOM = 64 * 27          # 1728


def sig_of(fill_num):
    return 0 if fill_num == 0 else 1 + (14 * fill_num) // DENOM


def build(s, prefer_min_items=True):
    """返回 (items_str, fill_num) 或 None。"""
    if s == 0:
        return "Items:[]", 0
    lo = max(1, math.ceil((s - 1) * DENOM / 14))
    hi = math.ceil(s * DENOM / 14) - 1
    cands = []
    for fill in range(lo, hi + 1):
        A, R = divmod(fill, 64)
        if A == 0 and R < 1:
            continue
        slots = math.ceil(A / 64) + (1 if R > 0 else 0)
        if slots <= SLOTS:
            cands.append((fill, A, R, slots))
    if not cands:
        return None
    fill, A, R, slots = min(cands, key=lambda t: t[0]) if prefer_min_items \
        else max(cands, key=lambda t: t[0])
    parts = []
    slot = 0
    left = A
    while left > 0:
        c = min(64, left)
        parts.append(f'{{Slot:{slot}b,count:{c}b,id:"minecraft:wooden_axe"}}')
        left -= c
        slot += 1
    if R > 0:
        parts.append(f'{{Slot:{slot}b,count:{R}b,id:"minecraft:redstone"}}')
    return "Items:[" + ",".join(parts) + "]", fill


print("=" * 84)
print("一、现有 512 个 barrel 的公式复核")
print("=" * 84)
bad = 0
for fn in sorted(os.listdir(FD)):
    if not fn.startswith("barrel_"):
        continue
    txt = open(os.path.join(FD, fn), encoding="utf-8").read()
    m = STAT_RE.search(txt)
    if not m:
        continue
    sig, item = int(m.group(1)), int(m.group(2))
    fn_num = 0
    for _slot, cnt, name in ITEM_RE.findall(txt):
        cnt = int(cnt)
        fn_num += cnt * 64 if name == "wooden_axe" else cnt
    if sig_of(fn_num) != sig or fn_num != item:
        bad += 1
print(f"  signal = 1 + 14·fill//1728，fill = axe×64 + redstone")
print(f"  512 个文件，不符 {bad} 个 → {'公式正确' if bad == 0 else '有误'}")

print()
print("=" * 84)
print("二、0..897 全覆盖生成")
print("=" * 84)
fail = []
maxslots = 0
info = {}
for s in range(898):
    r = build(s)
    if r is None:
        fail.append(s)
    else:
        items, fill = r
        info[s] = fill
        maxslots = max(maxslots, items.count("Slot"))
print(f"  可生成 {898-len(fail)} / 898，失败 {fail if fail else '无'}")
print(f"  最多用格 {maxslots} / 27")
print()
print("  抽查：")
for s in (0, 1, 2, 15, 16, 100, 255, 511, 700, 864, 865, 880, 882, 896, 897):
    r = build(s)
    if r is None:
        print(f"    {s:4d} 失败")
        continue
    items, fill = r
    print(f"    {s:4d}  fill={fill:7d}  slots={items.count('Slot'):2d}  "
          f"回算={sig_of(fill):4d}  {'OK' if sig_of(fill)==s else '错'}")

print()
print("=" * 84)
print("三、上限的来源")
print("=" * 84)
print(f"  27 格 × 每格 64 把斧 = 1728 个单位")
print(f"  signal = 1 + 14×1728/27 = 1 + 896 = 897")
print(f"  → 容器上限 897 = (27 格 × 64) 而非 (27 × 64 个物品)")
print()
print("  注意：882 与 897 都在可达集内：")
for s in (882, 897):
    print(f"    signal {s}: fill={info[s]}, slots={build(s)[0].count('Slot')}")

json.dump({"formula": "signal = 1 + 14*fill//1728; fill = 64*A + R",
           "slots": SLOTS, "max_signal": 897,
           "all_reachable": len(fail) == 0,
           "fill_of": {str(k): v for k, v in info.items()}},
          open(os.path.join(RESULTS, "barrel_palette.json"), "w",
               encoding="utf-8"), indent=1)
print()
print("  已写入 results/barrel_palette.json")
print("PALETTE_DONE")
