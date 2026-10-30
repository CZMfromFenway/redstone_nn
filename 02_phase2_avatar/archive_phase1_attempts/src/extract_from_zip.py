#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
高速提取：直接从完整 zip 批量提取 8x8 头像
=============================================
输入: avatar_gan/data/minecraft_skins_64x64.zip （完整的 2.53 GiB，110 万张）
输出: faces8_rgb.npy / faces8.npy （可指定数量）

用法:
  python extract_from_zip.py --n 50000
  python extract_from_zip.py --n 200000 --stride 5

要点:
  - zipfile 直接读，无需解压（zip 中央目录在尾部，Python 自动定位）
  - 脸部 (8,8)-(16,16) 正好 8x8
  - 用定制 16 色调色板量化
  - 逐字节去重
  - 全程 numpy + zipfile，约 500~1500 张/秒
"""
import os, sys, json, zipfile, time, argparse
import numpy as np
from PIL import Image
import io

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
ZIP = os.path.join(DATA, "minecraft_skins_64x64.zip")
PALJ = os.path.join(DATA, "palette_avatar16.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=50000, help='目标头像数')
    ap.add_argument('--stride', type=int, default=0,
                    help='若 >0，按此步长抽样条目（加快大 zip 的遍历）')
    ap.add_argument('--out', default='faces8', help='输出文件名前缀')
    args = ap.parse_args()

    if not os.path.exists(ZIP):
        print(f"[错误] 找不到 zip: {ZIP}")
        print(f"       请把 minecraft_skins_64x64.zip 放到 {DATA}")
        sys.exit(1)

    pal = np.array(json.load(open(PALJ))['palette_ordered'], dtype=np.float64)
    C = len(pal)
    print("=" * 92)
    print(f"从 zip 提取 8x8 头像   目标 {args.n} 张   调色板 {C} 色")
    print("=" * 92)

    t0 = time.time()
    zf = zipfile.ZipFile(ZIP, 'r')
    names = [n for n in zf.namelist() if n.lower().endswith('.png')]
    print(f"zip 内 PNG: {len(names)}   [{time.time()-t0:.1f}s]")

    if args.stride > 0:
        names = names[::args.stride]
        print(f"按步长 {args.stride} 抽样 -> {len(names)} 条")

    rgb, seen, bad = [], set(), 0
    t0 = time.time()
    n_try = 0
    for nm in names:
        if len(rgb) >= args.n:
            break
        n_try += 1
        try:
            with zf.open(nm) as f:
                im = Image.open(io.BytesIO(f.read())).convert('RGBA')
            if im.size[0] < 16 or im.size[1] < 16:
                bad += 1; continue
            a = np.array(im.crop((8, 8, 16, 16)))[:, :, :3]
            key = a.tobytes()
            if key in seen:
                continue
            seen.add(key)
            rgb.append(a)
        except Exception:
            bad += 1
        if n_try % 5000 == 0:
            el = time.time() - t0
            print(f"   已读 {n_try}  收 {len(rgb)}  坏 {bad}  {el:.0f}s  "
                  f"({n_try/max(el,1e-9):.0f} 张/秒)", flush=True)

    zf.close()
    F = np.stack(rgb).astype(np.float64)
    el = time.time() - t0
    print(f"\n有效（去重后）: {len(F)}   坏: {bad}   用时 {el:.0f}s "
          f"({n_try/max(el,1e-9):.0f} 张/秒)")

    # 量化
    t1 = time.time()
    flat = F.reshape(-1, 3)
    idx = ((flat[:, None] - pal[None]) ** 2).sum(2).argmin(1)
    lab = idx.reshape(len(F), 64).astype(np.uint8)
    qerr = float(((pal[idx].reshape(F.shape) - F) ** 2).mean())
    uc = np.array([len(np.unique(l)) for l in lab])
    print(f"调色板量化 MSE = {qerr:.1f} (RMS {np.sqrt(qerr/3):.1f}/255)   "
          f"平均颜色数 {uc.mean():.2f}    [{time.time()-t1:.0f}s]")

    np.save(os.path.join(DATA, f'{args.out}_rgb.npy'), F.astype(np.uint8))
    np.save(os.path.join(DATA, f'{args.out}.npy'), lab)
    print(f"已保存: {args.out}_rgb.npy {F.shape}   {args.out}.npy {lab.shape}")


if __name__ == '__main__':
    main()
