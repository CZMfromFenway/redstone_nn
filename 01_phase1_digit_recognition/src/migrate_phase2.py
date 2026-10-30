#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
"""
一期成果迁移 + 重构为多期工程结构
====================================
源: D:\\czm我的世界\\文件\\Comparator Network        （保持不动，第二轮再删）
目标: E:\\redstone_nn                                （纯英文路径）

新结构（为二期、三期预留）:
  E:\\redstone_nn\\
  ├─ 00_shared\\                 跨期通用资料
  │   ├─ archive_dt_20260927\\    DT 研究归档 1
  │   ├─ archive_dt_20261001\\    DT 研究归档 2
  │   └─ reference_cohomologo\\   辰占鳌头的资料
  ├─ 01_phase1_digit_recognition\\   一期：强模比较器网络数字识别（已完成）
  │   ├─ docs\\  src\\  results\\
  │   ├─ build\\                  draw.html 等入口
  │   ├─ models\\                 权重
  │   ├─ datapack\\               tiny_cnn 数据包
  │   ├─ data_mnist\\             MNIST 原始数据
  │   └─ archive\\                一期内部的历史归档
  ├─ 02_phase2_avatar\\           二期：神经网络生成头像（待启动）
  │   ├─ data\\  docs\\  src\\  results\\
  │   └─ archive_phase1_attempts\\  一期头像九方案的失败记录
  ├─ 99_models\\                  跨期模型仓库
  ├─ 99_tools\\                   跨期工具
  └─ README.md
"""
import os, shutil, sys, time

SRC = r'D:\czm我的世界\文件\Comparator Network'
DST = r'E:\redstone_nn'

# 顶层目录/文件的映射规则
# (源顶层名, 目标相对路径, 是否目录)
MAPPING = [
    # 一期主体：解包后重排
    ('StrongAnalog_LUT_Network', '01_phase1_digit_recognition', 'unpack'),
    ('tiny_cnn',                 r'01_phase1_digit_recognition\data_mnist', 'dir'),
    ('datapack',                 r'01_phase1_digit_recognition\datapack', 'dir'),
    ('DT_Redstone_Research_Archive_20260927', r'00_shared\archive_dt_20260927', 'dir'),
    ('DT_Redstone_Research_Archive_20261001', r'00_shared\archive_dt_20261001', 'dir'),
    ('redstone_cnn_by_cohomologo', r'00_shared\reference_cohomologo', 'dir'),
    ('.obsidian',                r'01_phase1_digit_recognition\obsidian', 'dir'),
]

# 一期内部：哪些子目录平移到一期根，哪些留在 archive
P1_KEEP = {          # 源子目录名 -> 一期根下的目标名
    'docs': 'docs',
    'src': 'src',
    'results': 'results',
    'avatar_gan': r'02_phase2_avatar\archive_phase1_attempts',
    'models': 'models',
}


def copytree(src, dst, log):
    if not os.path.exists(src):
        log.append(f'  [跳过] 源不存在: {src}')
        return 0
    os.makedirs(dst, exist_ok=True)
    n = 0
    for root, dirs, files in os.walk(src):
        rel = os.path.relpath(root, src)
        tgt = dst if rel == '.' else os.path.join(dst, rel)
        os.makedirs(tgt, exist_ok=True)
        for f in files:
            s = os.path.join(root, f)
            d = os.path.join(tgt, f)
            try:
                shutil.copy2(s, d)
                n += 1
            except Exception as e:
                log.append(f'  [失败] {s}: {e}')
    return n


def main():
    if os.path.exists(DST):
        print(f'目标已存在: {DST}')
        print('如需重来，请先删除目标目录。')
        return 1
    log = []
    t0 = time.time()
    total = 0
    print('=' * 88)
    print(f'迁移: {SRC}')
    print(f'  ->  {DST}')
    print('=' * 88)

    os.makedirs(DST, exist_ok=True)

    # ---- 一期主体（解包重排）----
    p1_src = os.path.join(SRC, 'StrongAnalog_LUT_Network')
    p1_dst = os.path.join(DST, '01_phase1_digit_recognition')
    print(f'\n[一期主体] {p1_src}')
    if os.path.isdir(p1_src):
        moved = set()
        for name in os.listdir(p1_src):
            s = os.path.join(p1_src, name)
            if name in ('docs', 'src', 'results'):
                tgt = os.path.join(p1_dst, name)
            elif name == 'avatar_gan':
                tgt = os.path.join(DST, '02_phase2_avatar', 'archive_phase1_attempts')
            else:
                tgt = os.path.join(p1_dst, 'archive', name)
            if os.path.isdir(s):
                n = copytree(s, tgt, log)
            else:
                os.makedirs(os.path.dirname(tgt), exist_ok=True)
                shutil.copy2(s, tgt); n = 1
            moved.add(name)
            total += n
            print(f'  {name:<40} -> {os.path.relpath(tgt, DST):<50} {n:>5} 文件')

    # ---- 其他顶层项 ----
    print(f'\n[其他顶层项]')
    for name, rel, kind in MAPPING:
        if kind == 'unpack':
            continue
        s = os.path.join(SRC, name)
        if not os.path.exists(s):
            print(f'  {name:<40} [跳过] 不存在')
            continue
        tgt = os.path.join(DST, rel)
        n = copytree(s, tgt, log) if os.path.isdir(s) else 0
        if not os.path.isdir(s):
            os.makedirs(os.path.dirname(tgt), exist_ok=True)
            shutil.copy2(s, tgt); n = 1
        total += n
        print(f'  {name:<40} -> {rel:<50} {n:>5} 文件')

    # ---- 跨期仓库 ----
    print(f'\n[跨期仓库]')
    os.makedirs(os.path.join(DST, '99_models'), exist_ok=True)
    os.makedirs(os.path.join(DST, '99_tools'), exist_ok=True)
    # 把一期的权重复制到跨期模型库
    p1_res = os.path.join(DST, '01_phase1_digit_recognition', 'results')
    nm = 0
    if os.path.isdir(p1_res):
        for f in os.listdir(p1_res):
            if f.endswith('.npz') or f == 'Wint_225x10.csv':
                shutil.copy2(os.path.join(p1_res, f),
                             os.path.join(DST, '99_models', f'phase1_{f}'))
                nm += 1
    total += nm
    print(f'  一期权重 -> 99_models/  {nm} 个文件')
    # draw.html -> 99_tools
    dh = os.path.join(DST, '01_phase1_digit_recognition', 'draw.html')
    if os.path.exists(dh):
        shutil.copy2(dh, os.path.join(DST, '99_tools', 'draw_digit_phase1.html'))
        total += 1
        print(f'  draw.html -> 99_tools/draw_digit_phase1.html')

    dt = time.time() - t0
    print(f'\n{"="*88}')
    print(f'完成: {total} 个文件, 用时 {dt:.1f} 秒')
    if log:
        print(f'\n异常 {len(log)} 条:')
        for l in log[:20]:
            print(l)
    else:
        print('无异常')

    # 大小统计
    tot = 0; cnt = 0
    for root, dirs, files in os.walk(DST):
        for f in files:
            try:
                tot += os.path.getsize(os.path.join(root, f)); cnt += 1
            except Exception:
                pass
    print(f'目标大小: {tot/1024/1024:.1f} MB, {cnt} 个文件')
    return 0


if __name__ == '__main__':
    sys.exit(main())
