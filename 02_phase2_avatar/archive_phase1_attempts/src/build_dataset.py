#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
用本地 Minecraft 材质构建 8x8 像素画数据集
============================================
数据源: .minecraft 里的资源包材质（原版方块/物品纹理，本身就是像素画）
提取: 每个纹理切成 8x8 非重叠块 -> 得到大量 8x8 彩色图
用途: 作为"玩家头像生成"的代理数据集（像素画分布，但非头像）

局限（必须记住）: 方块/物品纹理 != 玩家头像，数字只作数量级参考。
"""
import os, json, collections
import numpy as np
from PIL import Image

ROOT = r'D:\czm我的世界\.minecraft'
OUT = r'D:\czm我的世界\文件\Comparator Network\StrongAnalog_LUT_Network\avatar_gan\data'

# 优先这些资源包（完整、原版风格）
PREFER = [
    r'versions\1.20.1-Fabric 0.15.11\resourcepacks\oCd_Pack_1.20',
    r'versions\1.20-Fabric 0.15.11\resourcepacks\oCd_Pack_1.20',
    r'versions\1.20Redstone\resourcepacks\oCd_Pack_1.20',
    r'versions\1.21-Fabric 0.15.11\resourcepacks\CZM''s oCD patch',
    r'versions\1.21.8-Fabric 0.16.14\resourcepacks\CZM''s oCD patch',
    r'resourcepacks\oCd HD 1.8(128x128) modified by Sidney',
]


def find_texture_dirs():
    dirs = []
    for pr in PREFER:
        p = os.path.join(ROOT, pr)
        if not os.path.isdir(p):
            continue
        for dp, dn, fn in os.walk(p):
            if any(f.lower().endswith('.png') for f in fn):
                dirs.append(dp)
    return dirs


def patchify(im, size=8):
    """切成 size x size 非重叠块"""
    W, H = im.size
    if W < size or H < size:
        return []
    out = []
    for y in range(0, H - size + 1, size):
        for x in range(0, W - size + 1, size):
            out.append(im.crop((x, y, x + size, y + size)))
    return out


def main():
    dirs = find_texture_dirs()
    print(f"找到材质目录 {len(dirs)} 个")
    for d in dirs[:10]:
        print("   ", d)

    sizes = collections.Counter()
    patches = []
    nfiles = 0
    for d in dirs:
        for f in sorted(os.listdir(d)):
            if not f.lower().endswith('.png'):
                continue
            nfiles += 1
            try:
                with Image.open(os.path.join(d, f)) as im:
                    im = im.convert('RGBA')
                    sizes[im.size] += 1
                    for p in patchify(im, 8):
                        a = np.array(p, dtype=np.uint8)
                        # 跳过全透明的
                        if (a[:, :, 3] > 8).mean() < 0.5:
                            continue
                        patches.append(a[:, :, :3])
            except Exception:
                pass
    print(f"\n扫描 PNG: {nfiles}")
    print("尺寸分布 top10:")
    for s, c in sizes.most_common(10):
        print(f"   {s}: {c}")

    X = np.stack(patches).astype(np.uint8)
    print(f"\n8x8 图块总数: {len(X)}   shape={X.shape}")
    np.save(os.path.join(OUT, 'mc8.npy'), X)
    print(f"已保存: {OUT}\\mc8.npy")

    # 颜色统计
    px = X.reshape(-1, 3)
    print(f"唯一颜色数(全数据): {len(np.unique(px, axis=0))}")
    print(f"每图平均唯一颜色数: {np.mean([len(np.unique(x.reshape(-1,3),axis=0)) for x in X[:2000]]):.1f}")


if __name__ == '__main__':
    main()
