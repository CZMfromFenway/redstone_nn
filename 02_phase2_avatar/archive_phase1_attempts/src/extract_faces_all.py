#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从全部已下载皮肤提取 8x8 头像（扩充数据集）
==============================================
- 脸部: (8,8)-(16,16)
- 用定制 16 色调色板量化  -> 供 GAN 训练
- 同时保留 RGB 原图
- 统计: 色数分布 / 与旧数据集的重叠
"""
import os, json, numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
SKINS = os.path.join(DATA, "skins")

pal = np.array(json.load(open(os.path.join(DATA, 'palette_avatar16.json')))['palette_ordered'],
               dtype=np.float64)
files = sorted(f for f in os.listdir(SKINS) if f.endswith('.png'))
print(f"皮肤文件: {len(files)}")

rgb, bad = [], 0
seen = set()
for f in files:
    try:
        im = Image.open(os.path.join(SKINS, f)).convert('RGBA')
        face = im.crop((8, 8, 16, 16))
        a = np.array(face)[:, :, :3]
        key = a.tobytes()
        if key in seen:
            continue
        seen.add(key)
        rgb.append(a)
    except Exception:
        bad += 1

F = np.stack(rgb).astype(np.float64)
print(f"有效（去重后）: {len(F)}   失败: {bad}")

flat = F.reshape(-1, 3)
idx = ((flat[:, None] - pal[None]) ** 2).sum(2).argmin(1)
lab = idx.reshape(len(F), 64).astype(np.uint8)
qerr = float(((pal[idx].reshape(F.shape) - F) ** 2).mean())
print(f"定制调色板量化 MSE = {qerr:.1f}  (RMS {np.sqrt(qerr/3):.1f}/255)")

uc = np.array([len(np.unique(l)) for l in lab])
print(f"每张脸颜色数: 平均 {uc.mean():.1f}  中位 {np.median(uc):.0f}  最大 {uc.max()}")

np.save(os.path.join(DATA, 'faces8_rgb.npy'), F.astype(np.uint8))
np.save(os.path.join(DATA, 'faces8.npy'), lab)
print(f"已保存: faces8_rgb.npy {F.shape}   faces8.npy {lab.shape}")
