#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
反推容器信号强度与物品数量的关系，并核对现有 512 个 barrel 函数。

发现
    桶 27 格。把「最大堆叠 1」的物品（木斧）超叠到 count=64 塞进一格，
    该格贡献 64 个「填充单位」；红石最大堆叠 64，一格贡献 count/64 个单位。

    fill = Σ(斧的 count) + Σ(红石的 count)/64          （单位：格当量）
    signal = 0                         若 fill = 0
    signal = floor(1 + 14·fill/27)     若 fill > 0

    上限：27 格 × 64 = 1728 填充单位  →  1 + 14×1728/27 = 897
    这就是容器上限 897 的来源。
"""
import json
import os
import re
import sys
import math

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
FD = os.path.join(os.path.dirname(os.path.dirname(HERE)), "01_phase1_digit_recognition", "datapack",
                  "tiny_cnn", "data", "vessel", "function")
RESULTS = os.path.join(HERE, "results")

ITEM_RE = re.compile(r"Slot:(\d+)b,count:(\d+)b?,?[^}]*id:\"minecraft:(\w+)\"")
STAT_RE = re.compile(r"signal: (\d+), item: (\d+)")


def parse(path):
    txt = open(path, encoding="utf-8").read()
    m = STAT_RE.search(txt)
    if not m:
        return None
    sig, item = int(m.group(1)), int(m.group(2))
    fill_num = 0      # 以 1/64 为单位累加，避免浮点
    for slot, cnt, name in ITEM_RE.findall(txt):
        cnt = int(cnt)
        if name == "wooden_axe":
            fill_num += cnt * 64
        else:
            fill_num += cnt
    return sig, item, fill_num


def signal_from_fill(fill_num):
    if fill_num == 0:
        return 0
    # fill = fill_num/64 ; signal = floor(1 + 14*fill/27)
    return 1 + (14 * fill_num) // (64 * 27)


print("=" * 88)
print("核对现有 barrel 函数")
print("=" * 88)
files = sorted(os.listdir(FD))
bad = 0
n = 0
for fn in files:
    if not fn.startswith("barrel_"):
        continue
    got = parse(os.path.join(FD, fn))
    if got is None:
        continue
    sig, item, fill_num = got
    calc = signal_from_fill(fill_num)
    n += 1
    if calc != sig:
        bad += 1
        if bad <= 8:
            print(f"  不符 {fn}: 文件说 {sig}，公式给 {calc}（fill_num={fill_num}）")
print(f"  共 {n} 个文件，不符 {bad} 个")
print(f"  → 公式 signal = floor(1 + 14·fill/27)  {'成立' if bad == 0 else '不成立'}")

print()
print("=" * 88)
print("若有红石项被计入 item 总数：核对 item 与 fill_num 的关系")
print("=" * 88)
for fn in ("barrel_1.mcfunction", "barrel_100.mcfunction", "barrel_511.mcfunction"):
    p = os.path.join(FD, fn)
    if os.path.exists(p):
        sig, item, fill_num = parse(p)
        print(f"  {fn:24s} signal={sig:4d}  item={item:7d}  fill_num={fill_num:7d}"
              f"  (item/fill_num = {item/fill_num:.6f})")
print("  → item 就是 fill_num，即「把斧按 64 折算成红石当量」后的总数")

print()
print("=" * 88)
print("上限推导")
print("=" * 88)
for slots in (16, 17, 27):
    fill_num = slots * 64 * 64
    print(f"  {slots:2d} 格斧（每格 64 把）→ fill={fill_num/64:7.1f}  "
          f"signal={signal_from_fill(fill_num)}")


def gen_barrel(sig):
    """生成 signal 对应的 Items 串。"""
    if sig == 0:
        return "Items:[],CustomName:'\"signal: 0, item: 0\"'"
    # 需要 fill_num = 使 floor(1+14*fill_num/(64*27)) = sig
    # 即 14*fill_num/(1728) ∈ [sig-1, sig)
    # fill_num ∈ [ ceil((sig-1)*1728/14), ceil(sig*1728/14) - 1 ]
    lo = math.ceil((sig - 1) * 1728 / 14)
    hi = math.ceil(sig * 1728 / 14) - 1
    target = lo                      # 取最小可行值，最省物品
    axe_units = target // 64         # 斧个数（每把算 64）
    red = target % 64
    items = []
    slot = 0
    left = axe_units
    while left > 0:
        c = min(64, left)
        items.append(f'{{Slot:{slot}b,count:{c}b,id:"minecraft:wooden_axe"}}')
        left -= c
        slot += 1
    if red > 0:
        items.append(f'{{Slot:26b,count:{red}b,id:"minecraft:redstone"}}')
    if len(items) > 27:
        return None                  # 超过 27 格，做不到
    return ("Items:[" + ",".join(items) + "],"
            f"CustomName:'\"signal: {sig}, item: {target}\"'")


print()
print("=" * 88)
print("把调色板从 0..511 扩到 0..897")
print("=" * 88)
ok = 0
fail = []
for sig in range(0, 898):
    g = gen_barrel(sig)
    if g is None:
        fail.append(sig)
    else:
        ok += 1
print(f"  可生成 {ok} 个，失败 {len(fail)} 个"
      + (f"（{fail[:5]}...{fail[-5:]}）" if fail else ""))

# 抽查生成的正确性
print()
print("  抽查：")
for sig in (0, 1, 2, 15, 100, 255, 400, 511, 700, 897):
    g = gen_barrel(sig)
    if g is None:
        print(f"    {sig:4d}  失败")
        continue
    m = STAT_RE.search(g)
    item = int(m.group(2))
    calc = signal_from_fill(item)
    print(f"    {sig:4d}  item={item:7d}  回算 signal={calc:4d}  "
          f"{'OK' if calc == sig else '错'}")

os.makedirs(RESULTS, exist_ok=True)
json.dump({"formula": "signal = floor(1 + 14*fill/27), fill = axe_count + redstone_count/64",
           "max": 897,
           "sample": {str(s): gen_barrel(s) for s in (0, 1, 100, 511, 700, 897)}},
          open(os.path.join(RESULTS, "barrel_formula.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print()
print("  已写入 results/barrel_formula.json")
print("BARREL_DONE")
