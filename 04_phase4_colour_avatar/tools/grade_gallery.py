#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
把某一轮盲测的等级标准「显影」：每个等级（0..4）各抽若干张，按记录里的旋钮坐标
**重新生成同一张脸**，并给出该等级的观感统计。用来回答"作者的 0/1/2/3/4 到底长什么样"。

用法：python grade_gallery.py --round 4 [--per 12]
产物：results/labels/round<r>_grades.png、round<r>_grades.json
"""
import argparse
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
LDIR = os.path.join(P4, "results", "labels")
sys.path.insert(0, HERE)
from make_manifest import colours_of  # noqa: E402

ROW = np.array([p % 8 for p in range(32)])
COL = np.array([p // 8 for p in range(32)])
LUM = np.array([0.299, 0.587, 0.114])


def font(sz):
    for fn in ("msyh.ttc", "msyhbd.ttc", "simhei.ttf"):
        p = os.path.join("C:\\Windows\\Fonts", fn)
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, sz)
            except Exception:
                pass
    return ImageFont.load_default()


def feats(cols):
    a = np.asarray(cols, dtype=np.float64).reshape(len(cols), 32, 3)
    mx, mn = a.max(-1), a.min(-1)
    sat = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0).mean()
    ncol = np.mean([len({tuple(np.rint(c).astype(int)) for c in a[i]}) for i in range(len(a))])
    img = np.zeros((len(a), 8, 8, 3))
    img[:, ROW, COL] = a
    img[:, ROW, 7 - COL] = a
    dh = np.linalg.norm(img[:, :, 1:] - img[:, :, :-1], axis=-1)
    dv = np.linalg.norm(img[:, 1:] - img[:, :-1], axis=-1)
    return {"sat": round(float(sat), 3), "ncol": round(float(ncol), 2),
            "edge": round(float(np.percentile(np.concatenate([dh.ravel(), dv.ravel()]), 90)), 2),
            "lum": round(float((a @ LUM).mean()), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", default="4")
    ap.add_argument("--per", type=int, default=12)
    A = ap.parse_args()
    man = json.load(open(os.path.join(HERE, "label_manifest.json"), encoding="utf-8"))
    specs = {a["name"]: a for a in man["arms"]}
    rows = [json.loads(l) for l in
            open(os.path.join(LDIR, f"colour_round{A.round}.jsonl"), encoding="utf-8")
            if l.strip()]
    print(f"第 {A.round} 轮：{len(rows)} 条标签，臂 {sorted({r['model'] for r in rows})}")

    by = {g: [] for g in range(5)}
    missing = 0
    rng = np.random.default_rng(7)
    for r in rows:
        spec = specs.get(r["model"])
        if spec is None:
            missing += 1
            continue
        z = r.get("z") or r["pca"][:spec["k"]]
        z = [float(v) for v in z[:spec["k"]]]
        cols = colours_of(spec, dict(spec, z=z))
        by[r["grade"]].append({"model": r["model"], "z": z, "cols": cols})
    if missing:
        print(f"  （{missing} 条标签的臂不在当前清单里，跳过）")

    stat = {}
    for g in range(5):
        if not by[g]:
            continue
        cols = np.array([d["cols"] for d in by[g]])
        s = feats(cols)
        stat[g] = {"n": len(by[g]), **s,
                   "arms": {m: sum(1 for d in by[g] if d["model"] == m)
                            for m in sorted({d['model'] for d in by[g]})}}
        print(f"  grade {g}: n={stat[g]['n']:3d}  饱和 {s['sat']:.2f}  色数 {s['ncol']:.2f}  "
              f"边缘 {s['edge']:5.1f}  明度 {s['lum']:.1f}   臂 {stat[g]['arms']}")
    json.dump({"round": int(A.round), "by_grade": stat},
              open(os.path.join(LDIR, f"round{A.round}_grades.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)

    # ---- 出图：每级一行
    per = A.per
    TILE, PAD, LW = 52, 4, 300
    W = LW + per * (TILE + PAD) + 20
    grades = [g for g in range(5) if by[g]]
    H = 96 + len(grades) * (TILE + 30)
    cv = Image.new("RGB", (W, H), (16, 18, 22))
    dr = ImageDraw.Draw(cv)
    dr.text((14, 12), f"第 {A.round} 轮：作者的 0/1/2/3/4 标准（每级随机抽 {per} 张，按旋钮重绘）",
            font=font(19), fill=(234, 238, 244))
    dr.text((14, 40), "这些都是当时真实展示过的脸（坐标取自标签），不是事后挑的。"
                      "右端是该级的观感统计。", font=font(11), fill=(148, 160, 176))
    dr.text((14, 58), "主判据 ≥3；本轮 0~4 都有明确的对应标准。", font=font(11), fill=(130, 142, 158))
    y = 84
    for g in grades:
        idx = rng.choice(len(by[g]), min(per, len(by[g])), replace=False)
        dr.text((14, y + 8), f"grade {g}", font=font(13), fill=(112, 170, 254))
        st = stat[g]
        dr.text((14, y + 24), f"n={st['n']} · 饱和 {st['sat']:.2f} · 色数 {st['ncol']:.1f}",
                font=font(10), fill=(150, 162, 178))
        for i, j in enumerate(idx):
            cols = np.array(by[g][j]["cols"]).reshape(32, 3)
            img = np.zeros((8, 8, 3))
            img[ROW, COL] = cols
            img[ROW, 7 - COL] = cols
            v = np.clip(np.rint(img * 17), 0, 255).astype(np.uint8)
            t = np.repeat(np.repeat(v, TILE // 8, 0), TILE // 8, 1)
            cv.paste(Image.fromarray(t), (LW + i * (TILE + PAD), y))
        y += TILE + 30
    out = os.path.join(LDIR, f"round{A.round}_grades.png")
    cv.save(out)
    print(f"wrote {os.path.relpath(out, REPO)}")


if __name__ == "__main__":
    main()
