#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
强模 LUT 求解器 —— 使用示例
============================
运行:  python example_run.py
       python example_run.py 0 8 15 8 8 7 7 15 7 0 15 15 0 0 8 15
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from strong_lut_solver import solve

EX1 = [0, 8, 15, 8, 8, 7, 7, 15, 7, 0, 15, 15, 0, 0, 8, 15]
EX2 = [0, 12, 3, 14, 8, 8, 5, 8, 4, 12, 12, 12, 12, 6, 2, 0]
# 三个应当不可行的目标（相邻差超出窗口）
NEG = [
    ([0] + [100 * (i % 2) for i in range(1, 16)], "交替大跳变"),
    ([0] + [500] * 14 + [0], "单次巨大正跳"),
]

def show(f, tag):
    print("\n" + "#" * 84)
    print(f"# {tag}")
    print("#" * 84)
    solve(list(f))

if __name__ == "__main__":
    if len(sys.argv) > 1:
        show([int(a) for a in sys.argv[1:]], "命令行输入")
    else:
        show(EX1, "示例一（顺式 + 比较器截断可行）")
        show(EX2, "示例二（顺式 + 红石粉截断可行）")
        for f, tag in NEG:
            show(f, f"反例：{tag}")
