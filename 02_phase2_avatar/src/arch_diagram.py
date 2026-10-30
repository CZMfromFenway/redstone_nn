#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Draw the exact architecture of the model the user is labelling, with the dimensions and
the multiply count of every stage, so the redstone cost is visible at a glance.

This only reads the checkpoint; it trains nothing.
"""
import json, os
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "results", "redstone")
os.makedirs(OUT, exist_ok=True)

ck = torch.load(os.path.join(ROOT, "results", "coherence", "final_lam5_20k", "final.pt"),
                weights_only=False)
a = ck["args"]
Shapes = {k: tuple(v.shape) for k, v in ck["G"].items()}


def font(sz):
    for c in ("msyh.ttc", "simhei.ttf", "simsun.ttc", "Deng.ttf"):
        p = os.path.join("C:\\Windows\\Fonts", c)
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, sz)
            except Exception:
                pass
    return ImageFont.load_default()


F_T = font(21)
F_H = font(17)
F = font(14)
F_S = font(12)

W, H = 1500, 1150
cv = Image.new("RGB", (W, H), (18, 20, 24))
dr = ImageDraw.Draw(cv)

dr.text((28, 18), "现役模型结构  final_lam5_20k", font=F_T, fill=(232, 236, 242))
dr.text((28, 48), f"训练 {a['steps']} 步 · TV 一致性惩罚 λ={a['lam']} · 解码方式 {a['decode']}",
        font=F, fill=(150, 160, 175))

# ---- boxes -------------------------------------------------------------------------
BOX_W, BOX_H = 300, 96
rows = [
    ("① 输入：22 个数字", (46, 92, 160),
     ["每个 = 某个 PCA 主成分系数 ÷ 该成分标准差",
      "单位是「标准差倍数」，典型范围 −3.5 … +3.5",
      "没有随机噪声（64 维 z 已去掉）"],
     "22 个模拟量"),
    ("② 第 1 层：Linear + ReLU", (52, 120, 96),
     ["h1[j] = relu( b1[j] + Σ(i<22) W1[j][i]·x[i] )",
      f"W1 {Shapes['net.0.weight']}   b1 {Shapes['net.0.bias']}",
      "乘法 22×256 = 5,632    ReLU 256 次"],
     "256 个值"),
    ("③ 第 2 层：Linear + ReLU", (52, 120, 96),
     ["h2[j] = relu( b2[j] + Σ(i<256) W2[j][i]·h1[i] )",
      f"W2 {Shapes['net.2.weight']}   b2 {Shapes['net.2.bias']}",
      "乘法 256×256 = 65,536    ReLU 256 次"],
     "256 个值"),
    ("④ 第 3 层：Linear（无激活）", (150, 110, 52),
     ["logits[j] = b3[j] + Σ(i<256) W3[j][i]·h2[i]",
      f"W3 {Shapes['net.4.weight']}   b3 {Shapes['net.4.bias']}",
      "乘法 256×512 = 131,072"],
     "512 个 logits"),
    ("⑤ 解码：每 16 个取最大值下标", (150, 60, 120),
     ["512 = 32 个位置 × 16 个灰度级",
      "每个位置在 16 个数里取最大者的下标 → 0…15",
      "比较 32×15 = 480 次（不需要随机数）"],
     "32 个灰度级"),
    ("⑥ 布局 + 渲染", (90, 90, 110),
     ["位置 p = col×8 + row，只覆盖左半 4×8",
      "右半镜像：face[row][7−col] = face[row][col]",
      "灰度级 L → 像素值 255·L/15 = 0,17,34,…,255"],
     "8×8 图像"),
]

y = 92
txt_x = 28
box_x = W - BOX_W - 40
for title, col, lines, out in rows:
    # text column on the left
    dr.text((txt_x, y + 6), title, font=F_H, fill=(210, 218, 230))
    yy = y + 34
    for ln in lines:
        dr.text((txt_x + 12, yy), ln, font=F_S, fill=(150, 162, 178))
        yy += 19
    # coloured box on the right showing the tensor shape
    dr.rounded_rectangle([box_x, y, box_x + BOX_W, y + BOX_H], radius=9,
                         outline=col, width=2, fill=(col[0] // 5, col[1] // 5, col[2] // 5))
    dr.text((box_x + 14, y + 12), out, font=F_H, fill=col)
    # arrow to the next
    if title != rows[-1][0]:
        ax = box_x + BOX_W // 2
        dr.line([ax, y + BOX_H, ax, y + BOX_H + 20], fill=(90, 100, 115), width=2)
        dr.polygon([(ax - 5, y + BOX_H + 16), (ax + 5, y + BOX_H + 16), (ax, y + BOX_H + 23)],
                   fill=(90, 100, 115))
    y += BOX_H + 24

# ---- the totals --------------------------------------------------------------------
tot_y = y + 8
dr.rounded_rectangle([24, tot_y, W - 40, tot_y + 250], radius=10, outline=(200, 90, 80),
                     width=2, fill=(40, 24, 24))
dr.text((40, tot_y + 12), "总计（每生成一张头像）", font=F_H, fill=(240, 170, 160))
totals = [
    "参数 203,264 个   |   乘法-累加 202,240 次   |   加法 1,024 次   |   ReLU 512 次   "
    "|   取最大 480 次",
    "",
    "红石没有「乘法」这个原语，一次乘法就是一个电路：",
    "  · 若每个乘法是 4 位乘法器（约 60 个门）：202,240 × 60 ≈ 1,210 万门",
    "  · 若每个乘法是 8 位乘法器（约 300 个门）：202,240 × 300 ≈ 6,070 万门",
    "",
    "作为对照，红石社区里能跑起来的 CPU 大致在 10⁴ ~ 10⁶ 个元件。",
    "所以：现役模型 比可实际建造的规模 大 2~3 个数量级 —— 纯红石不可实现。",
    "这不是接线取巧能解决的，乘法次数是结构性的。",
]
yy = tot_y + 42
for ln in totals:
    dr.text((44, yy), ln, font=F_S if ln.startswith("  ") else F, fill=(215, 200, 198))
    yy += 20

cv.save(os.path.join(OUT, "ARCHITECTURE.png"))
print(f"wrote {os.path.join(OUT, 'ARCHITECTURE.png')}  {cv.size}")

# also dump the numbers as JSON so the report can quote them
macs = 22 * 256 + 256 * 256 + 256 * 512
json.dump({
    "model": "final_lam5_20k",
    "steps": int(a["steps"]), "lam": float(a["lam"]), "decode": a["decode"],
    "layers": [
        {"name": "Linear 22->256", "W": Shapes["net.0.weight"],
         "b": Shapes["net.0.bias"], "multiplies": 22 * 256, "relu": 256},
        {"name": "Linear 256->256", "W": Shapes["net.2.weight"],
         "b": Shapes["net.2.bias"], "multiplies": 256 * 256, "relu": 256},
        {"name": "Linear 256->512", "W": Shapes["net.4.weight"],
         "b": Shapes["net.4.bias"], "multiplies": 256 * 512, "relu": 0},
    ],
    "total_params": int(sum(int(np.prod(s)) for s in Shapes.values())),
    "total_multiplies": int(macs),
    "bias_additions": 256 + 256 + 512,
    "relu_comparisons": 512,
    "argmax_comparisons": 32 * 15,
    "input_dim": 22, "output_dim": 512, "positions": 32, "levels": 16,
}, open(os.path.join(OUT, "architecture.json"), "w"), indent=2)
print(f"wrote {os.path.join(OUT, 'architecture.json')}")
print("ARCH_DIAGRAM_DONE")
