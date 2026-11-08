#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
训练集清洗：找出"纯色头像 / 无效头像"，给出保留掩码，并量化它们的影响。

判定（在 32 个自由位置 × 3 通道、0..15 级上算）：
    uniform      整张脸只有一个颜色（distinct <= 1）
    near_flat    只有 2 个颜色，或亮度极差 < 1 级
    black        整张脸几乎为黑（平均亮度 < 0.5 且最大色度 < 0.5）——二期查到过
                 52,690 张"头部全透明被当全黑"，这里是同一批的残留
    dup          与前面某张完全相同（重复样本）
    low_chroma   最大色度 < 0.5 且亮度极差 < 2（灰卡，没有可学的颜色）

产物：results/clean/keep_index.npz（keep32 / 各类掩码）、clean_report.json、clean_examples.png
用法：python clean_dataset.py
"""
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
P4 = os.path.dirname(HERE)
REPO = os.path.dirname(P4)
P2 = os.path.join(REPO, "02_phase2_avatar")
OUT = os.path.join(P4, "results", "clean")
os.makedirs(OUT, exist_ok=True)


def font(sz):
    for fn in ("msyh.ttc", "msyhbd.ttc", "simhei.ttf"):
        p = os.path.join("C:\\Windows\\Fonts", fn)
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, sz)
            except Exception:
                pass
    return ImageFont.load_default()


def main():
    z = np.load(os.path.join(P2, "results", "colour", "cache.npz"))
    rgb96 = z["rgb96"].astype(np.float64)                 # (N,96) 0..15
    N = len(rgb96)
    POS = rgb96.reshape(N, 32, 3)
    print(f"样本 {N:,} 张，32 位置 × 3 通道")

    mx = POS.max(-1)
    mn = POS.min(-1)
    lum = POS @ np.array([0.299, 0.587, 0.114])
    chroma = mx - mn
    spread = lum.max(1) - lum.min(1)

    # distinct colours per face（把 0..15 级取整后数一数）
    q = np.rint(POS).astype(np.int16)
    key = (q[:, :, 0] * 256 + q[:, :, 1] * 16 + q[:, :, 2]).astype(np.int32)
    distinct = np.array([len(np.unique(k)) for k in key])

    # 完全重复（用 96 维取整后的哈希）
    packed = key.astype(np.int64)
    h = np.zeros(N, dtype=np.int64)
    for i in range(32):
        h = (h * 1000003 + packed[:, i]) % (2 ** 61 - 1)
    _, first = np.unique(h, return_index=True)
    dup = np.ones(N, dtype=bool)
    dup[first] = False

    uniform = distinct <= 1
    near_flat = (distinct <= 2) | (spread < 1.0)
    black = (lum.mean(1) < 0.5) & (chroma.max(1) < 0.5)
    low_chroma = (chroma.max(1) < 0.5) & (spread < 2.0)

    bad = uniform | black | low_chroma | dup
    keep = ~bad
    rep = {
        "n": int(N),
        "uniform": int(uniform.sum()),
        "near_flat": int(near_flat.sum()),
        "black": int(black.sum()),
        "low_chroma": int(low_chroma.sum()),
        "duplicate": int(dup.sum()),
        "bad_union": int(bad.sum()),
        "bad_frac": round(float(bad.mean()), 4),
        "keep": int(keep.sum()),
        "distinct_hist": {str(int(k)): int((distinct == k).sum()) for k in range(0, 9)},
        "distinct_gt8": int((distinct > 8).sum()),
    }
    print(json.dumps(rep, ensure_ascii=False, indent=1))

    # 训练集里被剔掉多少
    sp = np.load(os.path.join(P2, "data_v2_canonical", "L0_split.npz"))
    tr, va = sp["train"], sp["val"]
    rep["train_removed"] = int((~keep[tr]).sum())
    rep["train_removed_frac"] = round(float((~keep[tr]).mean()), 4)
    rep["val_removed"] = int((~keep[va]).sum())
    print(f"训练集将剔除 {rep['train_removed']:,} / {len(tr):,} "
          f"({rep['train_removed_frac']*100:.2f}%)；验证集 {rep['val_removed']} 张")

    np.savez_compressed(os.path.join(OUT, "keep_index.npz"),
                        keep=keep, uniform=uniform, black=black,
                        low_chroma=low_chroma, duplicate=dup, distinct=distinct)
    json.dump(rep, open(os.path.join(OUT, "clean_report.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)

    # 例子图：上排＝被剔除的各类，下排＝保留的随机样本
    def sheet(rows, path, title):
        TILE, PAD, LW = 52, 4, 300
        W = LW + 16 * (TILE + PAD) + 20
        H = 90 + len(rows) * (TILE + 26)
        cv = Image.new("RGB", (W, H), (16, 18, 22))
        dr = ImageDraw.Draw(cv)
        dr.text((14, 12), title, font=font(19), fill=(234, 238, 244))
        dr.text((14, 40), "每格是一张头像（行内镜像后显示）；被剔掉的是纯色、全黑、近乎无色的灰卡与重复样本。",
                font=font(11), fill=(148, 160, 176))
        y = 74
        for label, idxs in rows:
            dr.text((14, y + 8), label, font=font(12), fill=(112, 170, 254))
            dr.text((14, y + 24), f"{len(idxs):,} 张", font=font(10), fill=(150, 162, 178))
            for i, j in enumerate(idxs[:16]):
                im = POS[j].reshape(32, 3)
                rows_ = np.array([p % 8 for p in range(32)])
                cols_ = np.array([p // 8 for p in range(32)])
                img = np.zeros((8, 8, 3))
                img[rows_, cols_] = im
                img[rows_, 7 - cols_] = im
                v = np.clip(np.rint(img * 17), 0, 255).astype(np.uint8)
                t = np.repeat(np.repeat(v, TILE // 8, 0), TILE // 8, 1)
                cv.paste(Image.fromarray(t), (LW + i * (TILE + PAD), y))
            y += TILE + 26
        cv.save(path)
        print(f"wrote {os.path.relpath(path, REPO)}")

    rng = np.random.default_rng(0)
    def pick(mask, n=16):
        idx = np.where(mask)[0]
        return rng.choice(idx, min(n, len(idx)), replace=False)
    sheet([("纯色 uniform", pick(uniform)),
           ("全黑 black", pick(black)),
           ("近灰 low_chroma", pick(low_chroma)),
           ("保留（随机）", pick(keep))],
          os.path.join(OUT, "clean_examples.png"),
          "训练集清洗：被剔除的四类样本")
    print("CLEAN_DONE")


if __name__ == "__main__":
    main()
