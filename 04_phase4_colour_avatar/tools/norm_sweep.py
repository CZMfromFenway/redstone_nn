#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
灰度归一化的强度扫描：从"完全不归一化"到"逐张 min-max 拉满"，以及两种"对齐训练集统计量"的做法。

对照的政策（都作用在灰度模型输出的 32 个连续值上）：
    none        只 clip 0..15，完全不做归一化
    s25/s50/s75 在 raw 与"拉满"之间插值（α = 0.25 / 0.5 / 0.75）
    stretch     当前部署用的做法：逐张 min-max 拉到 [0,15]
    match_std   逐张仿射，使该脸的 **均值/标准差** 等于训练集的平均（7.33 / 3.56）
    match_range 逐张仿射，使该脸的 **[p2,p98] 跨度** 等于训练集的平均（10.47）

产物：
    results/twostage/norm_gray.png    每行一个政策，灰度半成品对比
    results/twostage/norm_colour.png  同一批政策经过颜色头之后的彩色结果
    results/twostage/norm_report.json 每个政策的统计量（对比度、与训练集分布的差距…）
用法：python norm_sweep.py [--n 12] [--samples 288]
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
ROW = np.array([q % 8 for q in range(32)])
COL = np.array([q // 8 for q in range(32)])

POLICIES = ["none", "s25", "s50", "s75", "stretch", "match_std", "match_range"]
LABELS = {
    "none": "不归一化（只 clip）",
    "s25": "弱归一化 α=0.25",
    "s50": "弱归一化 α=0.50",
    "s75": "弱归一化 α=0.75",
    "stretch": "逐张 min-max 拉满（当前）",
    "match_std": "对齐训练集 均值/标准差",
    "match_range": "对齐训练集 [p2,p98] 跨度",
}


def font(sz):
    for fn in ("msyh.ttc", "msyhbd.ttc", "simhei.ttf"):
        p = os.path.join("C:\\Windows\\Fonts", fn)
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, sz)
            except Exception:
                pass
    return ImageFont.load_default()


def fwd(layers, x):
    h = np.asarray(x, float)
    for i, (W, b) in enumerate(layers):
        h = h @ W.T + b
        if i < len(layers) - 1:
            h = np.maximum(h, 0)
    return h


def apply_policy(raw, policy, st):
    """raw: (N,32) 连续值，0..15 量纲。返回同样形状的"送去上色"的灰度。"""
    raw = np.clip(raw, 0, 15)
    lo = raw.min(1, keepdims=True)
    hi = raw.max(1, keepdims=True)
    span = np.maximum(hi - lo, 1e-6)
    full = (raw - lo) / span * 15.0
    if policy == "none":
        return raw
    if policy[0] == "s" and policy[1:].isdigit():
        a = int(policy[1:]) / 100.0
        return raw + a * (full - raw)
    if policy == "stretch":
        return full
    if policy == "match_std":
        mu = raw.mean(1, keepdims=True)
        sd = np.maximum(raw.std(1, keepdims=True), 1e-6)
        v = (raw - mu) / sd * st["face_std"]["mean"] + st["face_mean"]["mean"]
        return np.clip(v, 0, 15)
    if policy == "match_range":
        p2 = np.percentile(raw, 2, axis=1, keepdims=True)
        p98 = np.percentile(raw, 98, axis=1, keepdims=True)
        rng = np.maximum(p98 - p2, 1e-6)
        v = (raw - p2) / rng * st["p2_p98"]["mean"] + 1.0
        return np.clip(v, 0, 15)
    raise ValueError(policy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12, help="图上每行几张")
    ap.add_argument("--samples", type=int, default=288)
    A = ap.parse_args()

    g = json.load(open(os.path.join(P4, "results", "twostage", "gray_pca4.json"),
                       encoding="utf-8"))
    GL = [(np.array(L["W"]), np.array(L["b"])) for L in g["layers"]]
    p = json.load(open(os.path.join(P4, "results", "twostage", "pal16.json"),
                       encoding="utf-8"))
    LAY = ([(np.array(L["W"]), np.array(L["b"])) for L in p["body"]]
           + [(np.array(p["head"]["W"]), np.array(p["head"]["b"]))])
    PAL = np.array(p["palette"], float)
    sc = json.load(open(os.path.join(P4, "results", "labels", "scorers_round5.json"),
                        encoding="utf-8"))["families"]["two_stage"]
    w = np.array(sc["weights"], float)
    st = json.load(open(os.path.join(P4, "results", "twostage", "gray_stats.json"),
                        encoding="utf-8"))

    def score(z):
        z = list(z)
        f = ([1.0] + z + [v * v for v in z]
             + [z[i] * z[j] for i in range(4) for j in range(i + 1, 4)])
        return float(np.dot(w, f))

    rng = np.random.default_rng(9)
    Z = []
    for _ in range(A.samples):
        cs = [np.clip(rng.standard_normal(4), -3.5, 3.5) for _ in range(4)]
        Z.append(cs[int(np.argmax([score(c) for c in cs]))])
    Z = np.array(Z)
    raw = fwd(GL, Z)
    print(f"样本 {len(Z)} 张（都经过 4 选 1）；每行展示前 {A.n} 张")

    stats = {}
    gray_imgs, col_imgs = {}, {}
    for pol in POLICIES:
        gray = apply_policy(raw, pol, st)
        lv = np.clip(np.rint(gray), 0, 15)
        lg = fwd(LAY, gray / 15.0).reshape(len(Z), 32, len(PAL))
        idx = lg.argmax(-1)
        rgb = PAL[idx]
        # 统计
        gstd = gray.std(1)
        grange = gray.max(1) - gray.min(1)
        mx, mn = rgb.max(-1), rgb.min(-1)
        sat = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0).mean()
        ncol = np.mean([len({tuple(np.rint(c).astype(int)) for c in rgb[i]})
                        for i in range(len(Z))])
        # 与训练集灰度的分布差距（按 16 档直方图做总变差距离）
        hist = np.bincount(lv.ravel().astype(int), minlength=16) / lv.size
        ref = np.zeros(16)
        ref[1:16] = 1.0 / 15            # 训练集近似：均匀覆盖 0..15
        tv = 0.5 * np.abs(hist - ref).sum()
        stats[pol] = {"gray_std_mean": round(float(gstd.mean()), 2),
                      "gray_range_mean": round(float(grange.mean()), 2),
                      "gray_mean": round(float(gray.mean()), 2),
                      "hist_tv_vs_ref": round(float(tv), 3),
                      "sat": round(float(sat), 3),
                      "ncol": round(float(ncol), 2),
                      "levels_used": int(len(np.unique(lv)))}
        gray_imgs[pol] = lv
        col_imgs[pol] = rgb
        print(f"  {pol:11s} {LABELS[pol]:22s} 灰度 std {gstd.mean():5.2f} "
              f"极差 {grange.mean():5.2f} 均值 {gray.mean():5.2f} 档位 {len(np.unique(lv)):2d} "
              f"饱和 {sat:.3f} 色数 {ncol:.2f}")

    def render(kind, imgs, path, title, sub):
        T, PADW, LBL = 50, 6, 300
        W = LBL + A.n * (T + PADW) + 16
        H = 100 + len(POLICIES) * (T + 40)
        cv = Image.new("RGB", (W, H), (16, 18, 22))
        dr = ImageDraw.Draw(cv)
        dr.text((14, 12), title, font=font(19), fill=(234, 238, 244))
        dr.text((14, 40), sub, font=font(11), fill=(148, 160, 176))
        dr.text((14, 58), "每行同一批旋转钮（都经过 4 选 1），只改灰度归一化。", font=font(11),
                fill=(130, 142, 158))
        y = 84
        for pol in POLICIES:
            arr = imgs[pol]
            s = stats[pol]
            dr.text((14, y + 8), LABELS[pol], font=font(11), fill=(112, 170, 254))
            dr.text((14, y + 24), f"std {s['gray_std_mean']:.2f} · 极差 {s['gray_range_mean']:.1f}",
                    font=font(9), fill=(150, 162, 178))
            for i in range(min(A.n, len(arr))):
                if kind == "gray":
                    v = np.stack([arr[i] / 15.0] * 3, -1)
                else:
                    v = arr[i] / 15.0
                img = np.zeros((8, 8, 3))
                img[ROW, COL] = v
                img[ROW, 7 - COL] = v
                t = np.repeat(np.repeat(
                    np.clip(np.rint(img * 255), 0, 255).astype(np.uint8),
                    T // 8, 0), T // 8, 1)
                cv.paste(Image.fromarray(t), (LBL + i * (T + PADW), y))
            y += T + 40
        cv.save(path)
        print(f"wrote {os.path.relpath(path, REPO)}")

    render("gray", gray_imgs, os.path.join(P4, "results", "twostage", "norm_gray.png"),
           "灰度归一化强度扫描（颜色头看到的输入）",
           "从上到下：不归一化 → 弱 → 拉满 → 对齐训练集统计量。")
    render("colour", col_imgs, os.path.join(P4, "results", "twostage", "norm_colour.png"),
           "同一批政策经过颜色头之后的彩色结果",
           "颜色头是在「拉满」的灰度上训的，所以其它政策对它来说是分布外输入。")
    json.dump({"dataset": st, "policies": stats},
              open(os.path.join(P4, "results", "twostage", "norm_report.json"), "w",
                   encoding="utf-8"), indent=2, ensure_ascii=False)
    print("wrote results/twostage/norm_report.json")


if __name__ == "__main__":
    main()
