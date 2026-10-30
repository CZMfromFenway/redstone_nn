#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""predict.py —— 这个 schem 的"预期输出"计算器（纯 Python 3，无第三方依赖）。

它做两件事：
  1) 给定 4 个输入整数，列出 32 个需要充能的方块（哪些要充、哪些不充）；
  2) 给出 64 个输出比较器的期望 OutputSignal。

用法：
    python predict.py 0 0 0 0             # 全 0（不接线就能验）
    python predict.py 186 37 204 91       # 任意一组
    python predict.py 186 37 204 91 --json
    python predict.py --coords            # 只打印坐标表

公式：
    N[ch] = sum_i sum_k ( W[ch][i][k] * bit_k(X[i]) - min(0, W[ch][i][k]) )
    输出比较器的 OutputSignal 就等于 N[ch]。
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def load():
    with open(os.path.join(HERE, "weights_L1_b8.json"), encoding="utf-8") as f:
        W = json.load(f)["W"]
    with open(os.path.join(HERE, "io_map.json"), encoding="utf-8") as f:
        m = json.load(f)
    return W, m


def expected(W, X):
    out = []
    for ch in range(len(W)):
        n = 0
        for i in range(4):
            for k in range(8):
                w = W[ch][i][k]
                n += w * ((X[i] >> k) & 1) - min(0, w)
        out.append(n)
    return out


def main():
    W, m = load()
    if "--coords" in sys.argv:
        print("输入块（bit=1 时给该方块充能）:")
        for e in m["inputs"]:
            print("  i=%d k=%d  (%d, %d, %d)" % (e["i"], e["k"], e["x"], e["y"], e["z"]))
        print("输出比较器:")
        for e in m["outputs"]:
            print("  ch=%d  (%d, %d, %d)" % (e["ch"], e["x"], e["y"], e["z"]))
        return

    xs = [int(a) for a in sys.argv[1:5]]
    if len(xs) != 4:
        print(__doc__)
        return 1
    if any(v < 0 or v > 255 for v in xs):
        print("输入必须落在 0..255")
        return 1

    exp = expected(W, xs)
    if "--json" in sys.argv:
        print(json.dumps({"X": xs, "expected": exp}, ensure_ascii=False))
        return

    print("输入 X = %s" % (xs,))
    print("  二进制（位7..0）: " + "  ".join("X%d=%s" % (i, format(xs[i], "08b")) for i in range(4)))
    print()
    print("需要充能的方块（%d 个中，充能的那些）:" % len(m["inputs"]))
    hot = [e for e in m["inputs"] if (xs[e["i"]] >> e["k"]) & 1]
    for e in hot:
        print("  (%d, %d, %d)   <- 输入%d 的第 %d 位" % (e["x"], e["y"], e["z"], e["i"], e["k"]))
    print("  （其余 %d 个保持不充能）" % (len(m["inputs"]) - len(hot)))
    print()
    print("输出比较器的期望 OutputSignal：")
    for ch in range(len(exp)):
        e = m["outputs"][ch]
        print("  ch%-2d (%2d, %3d, %2d)  ->  %d" % (ch, e["x"], e["y"], e["z"], exp[ch]))
    print()
    print("合计：%d 个通道，期望输出范围 [%d, %d]" % (len(exp), min(exp), max(exp)))


if __name__ == "__main__":
    sys.exit(main())
