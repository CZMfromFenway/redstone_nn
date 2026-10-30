# -*- coding: utf-8 -*-
"""从 planes64.json 反推 32 根火把塔的占地（不依赖槽表硬编码）。

关键：同一个 (i,k) 的塔，是**贴着平面自己的格子**长的：
    输入块 = 该通道里 ik=[i,k] 的那个彩色块      -> (slotCol, blockRow)
    读入比较器 = rear 正好指到这个输入块的比较器  -> (slotCol, readRow)
    塔列 otherCol = 同一条"列对"里的另一列（rx 异或 1）

于是 inRev（输入顺序反转，改的是 XI(i)）、bitRev（Z 镜像 + N/S 互换）、
parSwap（列奇偶互换）都自动被吸收 —— 因为位置是从建好的平面里读出来的，
朝向也是从坐标关系算出来的，不是查表。

用法：python src/tower_spec.py
输出：results/towers32.json
"""
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DV = {"N": (0, -1), "S": (0, 1), "E": (1, 0), "W": (-1, 0)}
ORIGIN_X = 3


def partner_col(x):
    """同一条列对里的另一列（rx 异或 1）=> x 与 x^1+3 同理，这里直接在绝对 x 上算。"""
    rx = x - ORIGIN_X
    return (rx ^ 1) + ORIGIN_X


def true_sign(Wm, i_par, nch, b):
    """该输入该通道的权重真实符号 = 绝对值最大的那个位的符号。

    不能直接用 W[ch][i][k] 的符号：小 k 的权重可能被量化成 0（层1 有 26 处），
    而 0 的符号任取都不影响算术（S = 15+|0| = 15，火把亮灭都读 15）。
    用真实符号可让同一输入 8 个位的翻相位置完全一致。

    层2 还有整列 8 个位全为 0 的情形（该权重太小）：整列读 15 恒定、极性完全自由，
    取邻居值即可（翻相最少）。
    """
    raw = []
    for ch in range(nch):
        row = Wm[ch][i_par]
        best = max(range(b), key=lambda k: abs(row[k]))
        v = row[best]
        raw.append(None if v == 0 else (1 if v > 0 else 0))
    out = list(raw)
    last = None
    for ch in range(nch):
        if out[ch] is None:
            out[ch] = last
        else:
            last = out[ch]
    nxt = None
    for ch in range(nch - 1, -1, -1):
        if out[ch] is None:
            out[ch] = nxt
        else:
            nxt = out[ch]
    if any(v is None for v in out):
        raise RuntimeError(f"i={i_par} 整列全自由，无处取符号")
    nfree = sum(1 for v in raw if v is None)
    if nfree:
        print(f"    i={i_par}: {nfree} 个通道整列全零 -> 极性自由，取邻居值")
    return out


def derive(planes, i_par, k_par, signs, XBIT=0):
    """返回该塔的几何 + 每通道状态。planes 是 64 片（只需第 0 片确认几何）。"""
    p0 = planes[0]["cells"]
    by_ik = {}
    for c in p0:
        if c["ik"] and c["t"] == "blk" and (c["c"] or "").startswith("col"):
            by_ik[(c["ik"][0], c["ik"][1])] = (c["x"], c["z"])
    bx, bz = by_ik[(i_par, k_par)]

    # 找 rear 指到输入块的比较器
    read = None
    for c in p0:
        if c["t"] != "cmp":
            continue
        dx, dz = DV[c["out"]]
        if (c["x"] - dx, c["z"] - dz) == (bx, bz):
            read = (c["x"], c["z"])
            break
    if read is None:
        raise RuntimeError(f"(i={i_par},k={k_par}) 找不到读入比较器")
    if read[0] != bx:
        raise RuntimeError(f"(i={i_par},k={k_par}) 读入比较器不在同一列：块 x={bx} cmp x={read[0]}")

    slotCol, blockRow, readRow = bx, bz, read[1]
    otherCol = partner_col(slotCol)
    if abs(readRow - blockRow) != 1:
        raise RuntimeError(f"(i={i_par},k={k_par}) 读入行与槽行不相邻：{readRow} vs {blockRow}")

    # 朝向由坐标关系决定（火把 facing 指向"离开支撑块"的方向）
    tap_f = "W" if otherCol > slotCol else "E"          # 支撑 = A(ch) 在塔列
    car_f = "N" if readRow > blockRow else "S"          # 壁面 carrier 支撑 = B(ch) 在读入行
    rel_f = "S" if readRow > blockRow else "N"          # relay 支撑 = A(ch) 在槽行

    nch = len(signs)
    s = signs
    flip = [(s[ch + 1] != s[ch]) for ch in range(nch - 1)] + [False]
    tap = [1 if XBIT != s[ch] else 0 for ch in range(nch)]
    car = [(tap[ch] if flip[ch] else 1 - tap[ch]) for ch in range(nch)]

    return {
        "i": i_par, "k": k_par,
        "slotCol": slotCol, "otherCol": otherCol,
        "blockRow": blockRow, "readRow": readRow,
        "tap_face": tap_f, "carrier_face": car_f, "relay_face": rel_f,
        "bottom_inv": (s[0] == 0),
        "sign": s, "flip": flip, "tap": tap, "carrier": car,
        # 5 个方块/通道的平面格：(x, z) 绝对
        "cells": {
            "tap": [slotCol, blockRow],        # 偶层火把
            "A": [otherCol, blockRow],         # 偶层块 / 奇层 carrier 火把
            "B": [otherCol, readRow],          # 奇层块 / 偶层 relay 火把
        },
    }


def main():
    planes_path = (sys.argv[1] if len(sys.argv) > 1
                   else os.path.join(ROOT, "results", "planes64.json"))
    layer_key = (sys.argv[2] if len(sys.argv) > 2 else "L1")
    out_path = (sys.argv[3] if len(sys.argv) > 3
                else os.path.join(ROOT, "results", "towers32.json"))
    d = json.load(open(planes_path, encoding="utf-8"))
    planes = d["planes"]
    ref = json.load(open(os.path.join(ROOT, "results", "py_compile_reference.json"),
                         encoding="utf-8"))
    Wm = ref["layers"][layer_key]["8"]["W"]
    b = d["b"]
    n = (d["W"] - 6) // 2          # 宽度 = 2n + 6
    nch = len(planes)
    print(f"{planes_path}  层={layer_key}  n={n}  b={b}  通道={nch}  塔数={n*b}")

    # 占位（用于零冲突校验）
    occ = []
    for p in planes:
        odd, even = set(), set()
        for c in p["cells"]:
            odd.add((c["x"], c["z"]))
            if c["t"] == "cmp":
                even.add((c["x"], c["z"]))
        occ.append((odd, even))

    out = []
    bad = []
    verbose = (n * b <= 32)
    if verbose:
        print("i k | 槽列 塔列 | 槽行 读入行 | tap car relay | 塔底 | 翻相")
    for i in range(n):
        signs = true_sign(Wm, i, nch, b)          # 同一输入 8 个位共用
        for k in range(b):
            T = derive(planes, i, k, signs)
            out.append(T)
            nf = sum(T["flip"])
            if verbose:
                print(f"{i} {k} | {T['slotCol']:4d} {T['otherCol']:4d} | "
                      f"{T['blockRow']:4d} {T['readRow']:5d} | "
                      f"{T['tap_face']}   {T['carrier_face']}   {T['relay_face']}   | "
                      f"{'反相' if T['bottom_inv'] else '双火把'} | {nf:2d}")
            # 零冲突：塔新增的位必须在 64 片里都是空气
            #   tap 只在偶层（奇层那格是平面自己的输入块，不动）
            #   A/B 两格奇偶层都用
            for ch, (odd, even) in enumerate(occ):
                for name, (x, z) in T["cells"].items():
                    if name == "tap":
                        if (x, z) in even:
                            bad.append((ch, i, k, name, "偶层已有方块", x, z))
                    else:
                        if (x, z) in odd:
                            bad.append((ch, i, k, name, "奇层已有方块", x, z))
                        if (x, z) in even:
                            bad.append((ch, i, k, name, "偶层已有方块", x, z))

    # 同一通道内 32 个塔互不重叠
    overl = []
    for ch in range(nch):
        seen = {}
        for T in out:
            for name, p in T["cells"].items():
                key = (p[0], p[1])
                if key in seen:
                    overl.append((ch, seen[key], (T["i"], T["k"], name), key))
                seen[key] = (T["i"], T["k"], name)

    print()
    if bad:
        print(f"✗ 与平面冲突 {len(bad)} 处，前 8：")
        for r in bad[:8]:
            print("   ch=%d (i%d,k%d) 格=%s %s @(%d,%d)" % r)
    else:
        print(f"✓ 零冲突：{len(out)} 塔 × {nch} 通道 × 5 块 = {len(out)*nch*5} 个位置全部是空气")
    if overl:
        print(f"✗ 塔与塔重叠 {len(overl)} 处，前 8：{overl[:8]}")
    else:
        print("✓ 塔与塔不重叠")

    nflip = sum(sum(T["flip"]) for T in out)
    ninv = sum(1 for T in out if T["bottom_inv"])
    print(f"✓ 直立火把合计 {nflip}，需塔底反相级的塔 {ninv}/{len(out)}")

    json.dump({"n": n, "b": b, "nch": nch, "towers": out},
              open(out_path, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"已写 {out_path}")
    return 0 if not bad and not overl else 1


if __name__ == "__main__":
    sys.exit(main())
