#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从皮肤提取 8x8 头像数据
=========================
Minecraft 皮肤是 64x64 的模型展开图。其中:
  头部前面(脸)     : (8,8)-(16,16)   -> 正好 8x8   == 我们要的头像
  头部整体覆盖层   : 第二层帽子 (40,8)-(48,16)

本脚本:
  1. 校验下载的皮肤是否合法 PNG
  2. 提取"脸部 8x8"作为头像数据
  3. 用 MC 16 色混凝土调色板量化（按 czm 给的色环顺序）
  4. 统计色数分布
"""
import os, json, numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
SKINS = os.path.join(DATA, "skins")

names = ['white', 'light_gray', 'gray', 'black', 'brown', 'red', 'orange', 'yellow',
         'lime', 'green', 'cyan', 'light_blue', 'blue', 'purple', 'magenta', 'pink']
pal16 = json.load(open(os.path.join(DATA, 'mc16_palette.json')))
PAL = np.array([pal16[n][0] for n in names], dtype=np.float64)

files = sorted(f for f in os.listdir(SKINS) if f.endswith('.png'))
print(f"皮肤文件: {len(files)}")

faces, bad = [], []
sizes = {}
for f in files:
    try:
        im = Image.open(os.path.join(SKINS, f)).convert('RGBA')
        sizes[im.size] = sizes.get(im.size, 0) + 1
        if im.size[0] < 16 or im.size[1] < 16:
            bad.append(f); continue
        # 脸: (8,8)-(16,16)
        face = im.crop((8, 8, 16, 16))
        a = np.array(face)[:, :, :3].astype(np.float64)
        al = np.array(face)[:, :, 3]
        faces.append(a)
    except Exception:
        bad.append(f)

print(f"尺寸分布: {sizes}")
print(f"有效: {len(faces)}  失败: {len(bad)}")
F = np.stack(faces)
print(f"脸部数据: {F.shape}")

# 用 MC16 调色板量化
flat = F.reshape(-1, 3)
d = ((flat[:, None, :] - PAL[None, :, :]) ** 2).sum(2)
lab = d.argmin(1).reshape(len(F), 8, 8)
qerr = float(((PAL.reshape(-1, 3)[lab.reshape(-1)].reshape(-1, 8, 8, 3) - F) ** 2).mean())
print(f"\n用 MC16 调色板量化的 MSE = {qerr:.5f}  (RMS={np.sqrt(qerr):.2f}/255)")

# 色数分布
uc = np.array([len(np.unique(l)) for l in lab])
print(f"\n每张脸用到的颜色数分布:")
for k in range(1, 13):
    c = int((uc == k).sum())
    if c:
        print(f"   {k:2d} 色: {c:5d}  ({c/len(uc)*100:5.1f}%)")
print(f"   >12 色: {int((uc>12).sum())}")
print(f"   平均 {uc.mean():.1f} 色")

np.save(os.path.join(DATA, 'faces8.npy'), lab.astype(np.uint8))
np.save(os.path.join(DATA, 'faces8_rgb.npy'), F.astype(np.uint8))
print(f"\n已保存: data/faces8.npy (标签)  data/faces8_rgb.npy (原色)")
