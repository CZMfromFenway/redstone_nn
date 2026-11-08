#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
彩色头像的 4 旋钮全空间地图（对应二期 `pca4_map.py` 的灰度版）。

网格：每个旋钮取 −3.5, −2.5, …, +3.5 共 8 个值（**跳过 0**：0 是分布均值，输出最平），
      共 8⁴ = 4096 张。拆成 8 张图：

        行        PC0（8）
        横轴      PC1（8 组）× PC2（每组 8 张）
        每张图   一个 PC3 值

另外出：
    SCORE_MAP.png   同一布局的「打分器分数」热图（红→绿），一眼看出好区在哪
    score.json      4096 个点的分数与观感统计（色数/饱和/边缘）、预测可接受覆盖率

产物：04_phase4_colour_avatar/results/pca4map/
用法：python pca4_map_colour.py
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
OUT = os.path.join(P4, "results", "pca4map")
os.makedirs(OUT, exist_ok=True)

LEVELS = [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]
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


def fwd(layers, x, relu_last=False):
    h = np.asarray(x, dtype=np.float64)
    for i, (W, b) in enumerate(layers):
        h = h @ W.T + b
        if i < len(layers) - 1 or relu_last:
            h = np.maximum(h, 0)
    return h


def load():
    g = json.load(open(os.path.join(P4, "results", "twostage", "gray_pca4.json"),
                       encoding="utf-8"))
    gl = [(np.array(L["W"]), np.array(L["b"])) for L in g["layers"]]
    p = json.load(open(os.path.join(P4, "results", "twostage", "pal16.json"),
                       encoding="utf-8"))
    pbody = [(np.array(L["W"]), np.array(L["b"])) for L in p["body"]]
    phead = [(np.array(p["head"]["W"]), np.array(p["head"]["b"]))]
    pal = np.array(p["palette"], dtype=np.float64)
    sc = json.load(open(os.path.join(P4, "results", "labels", "scorers_round5.json"),
                        encoding="utf-8"))["families"]["two_stage"]
    return gl, pbody + phead, pal, sc


def score_all(Z, sc):
    w = np.array(sc["weights"], dtype=np.float64)
    k = 4
    f = [np.ones((len(Z), 1)), Z]
    f.append(Z ** 2)
    iu = np.triu_indices(k, 1)
    f.append(Z[:, iu[0]] * Z[:, iu[1]])
    F = np.hstack(f)
    return F @ w


def main():
    gl, players, pal, sc = load()
    K = len(pal)

    # ---- 4096 点
    pts = np.array([[a, b, c, d] for a in LEVELS for b in LEVELS
                    for c in LEVELS for d in LEVELS], dtype=np.float64)
    print(f"格点 {len(pts)}；灰度 4→64→32，颜色头 32→128→128→{32*K}")

    raw = fwd(gl, pts)
    lo = raw.min(1, keepdims=True)
    hi = raw.max(1, keepdims=True)
    lv = np.clip(np.rint((raw - lo) / np.maximum(hi - lo, 1e-6) * 15), 0, 15)
    x = lv / 15.0
    lg = fwd(players, x, relu_last=False).reshape(len(pts), 32, K)
    idx = lg.argmax(-1)
    rgb = pal[idx]                                   # (4096,32,3) 0..15
    img = np.zeros((len(pts), 8, 8, 3))
    img[:, ROW, COL] = rgb
    img[:, ROW, 7 - COL] = rgb

    scores = score_all(pts, sc)
    # 观感统计
    mx = rgb.max(-1); mn = rgb.min(-1)
    sat = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0).mean(1)
    ncol = np.array([len({tuple(np.rint(c).astype(int)) for c in rgb[i]}) for i in range(len(pts))])
    dh = np.linalg.norm(img[:, :, 1:] - img[:, :, :-1], axis=-1)
    dv = np.linalg.norm(img[:, 1:] - img[:, :-1], axis=-1)
    edge = np.percentile(np.concatenate([dh.reshape(len(pts), -1),
                                         dv.reshape(len(pts), -1)], 1), 90, axis=1)
    print(f"  预测可接受（score>0）占比 {float((scores>0).mean())*100:.1f}%；"
          f"分数范围 [{scores.min():+.2f}, {scores.max():+.2f}]")
    print(f"  色数 均值 {ncol.mean():.2f}（范围 {ncol.min()}–{ncol.max()}）；"
          f"饱和 均值 {sat.mean():.3f}；边缘 p90 均值 {edge.mean():.1f}")

    # ---- 8 张地图
    TILE, PAD, BLOCKGAP = 40, 2, 10
    BW = 8 * (TILE + PAD)
    W = 96 + 8 * BW + 7 * BLOCKGAP + 16
    H = 104 + 8 * (TILE + PAD) + 20
    f_t, f_l, f_s = font(20), font(13), font(10)
    for pi, p3 in enumerate(LEVELS):
        cv = Image.new("RGB", (W, H), (18, 20, 24))
        dr = ImageDraw.Draw(cv)
        dr.text((12, 12), f"彩色 4-PCA 地图　PC3 = {p3:+.1f}", font=f_t, fill=(234, 238, 244))
        dr.text((12, 40), "行 = PC0（8 值），横轴 = PC1（8 组）× PC2（每组 8 张）；"
                          "每张都是同一批等级 −3.5…+3.5，共 4096 张。", font=f_s,
                fill=(150, 162, 178))
        dr.text((12, 56), "模型：PCA4 灰度 4→64→32（2,304）＋ 颜色头 32→128→128→32×16（86,016）；"
                          "调色板 16 槽吸附到 61 个 HIGH 色。", font=f_s, fill=(130, 142, 158))
        for c1 in range(8):
            x0 = 96 + c1 * (BW + BLOCKGAP)
            dr.text((x0 + 2, 76), f"PC1={LEVELS[c1]:+.1f}", font=f_s, fill=(110, 168, 254))
        for c0 in range(8):
            y0 = 90 + c0 * (TILE + PAD)
            dr.text((12, y0 + 10), f"PC0={LEVELS[c0]:+.1f}", font=f_l, fill=(200, 210, 226))
            for c1 in range(8):
                for c2 in range(8):
                    k = ((c0 * 8 + c1) * 8 + c2) * 8 + pi
                    x = 96 + c1 * (BW + BLOCKGAP) + c2 * (TILE + PAD)
                    t = np.kron(np.clip(img[k] / 15.0 * 255, 0, 255).astype(np.uint8),
                                np.ones((TILE // 8, TILE // 8, 1))).astype(np.uint8)
                    cv.paste(Image.fromarray(t), (x, y0))
        tag = f"{p3:+.1f}".replace("+", "p").replace("-", "m").replace(".", "_")
        p = os.path.join(OUT, f"COLOUR_MAP_p3_{tag}.png")
        cv.save(p)
        print(f"  wrote {os.path.basename(p)}  {cv.size}")

    # ---- 分数热图（同一布局）
    CELL, PADR, BGAPC = 26, 2, 8
    BWC = 8 * (CELL + PADR)
    W2 = 96 + 8 * BWC + 7 * BGAPC + 16
    H2 = 104 + 8 * (CELL + PADR) + 20
    cv = Image.new("RGB", (W2, H2), (18, 20, 24))
    dr = ImageDraw.Draw(cv)
    dr.text((12, 12), "打分器在 4 旋钮全空间上的分布（红=差，绿=好）", font=f_t, fill=(234, 238, 244))
    dr.text((12, 40), "行 = PC0，横轴 = PC1 × PC2；8 个分块自上而下 = PC3（另出 8 张单图时同序）。",
            font=f_s, fill=(150, 162, 178))
    dr.text((12, 56), f"4096 点中 score>0 占 {float((scores>0).mean())*100:.1f}%；"
                      f"最坏 {scores.min():+.2f}、最好 {scores.max():+.2f}。", font=f_s,
            fill=(130, 142, 158))
    smin, smax = float(scores.min()), float(scores.max())
    for c1 in range(8):
        x0 = 96 + c1 * (BWC + BGAPC)
        dr.text((x0 + 2, 76), f"PC1={LEVELS[c1]:+.1f}", font=f_s, fill=(110, 168, 254))
    for c0 in range(8):
        y0 = 90 + c0 * (CELL + PADR)
        dr.text((12, y0 + 6), f"PC0={LEVELS[c0]:+.1f}", font=f_l, fill=(200, 210, 226))
        for c1 in range(8):
            for c2 in range(8):
                k = ((c0 * 8 + c1) * 8 + c2) * 8 + 0
                s = float(scores[k])
                t = (s - smin) / max(smax - smin, 1e-6)
                col = (int(220 * (1 - t) + 20), int(40 + 190 * t), int(60 * (1 - abs(t - 0.5) * 2)))
                x = 96 + c1 * (BWC + BGAPC) + c2 * (CELL + PADR)
                dr.rectangle([x, y0, x + CELL - 1, y0 + CELL - 1], fill=col)
    p = os.path.join(OUT, "SCORE_MAP.png")
    cv.save(p)
    print(f"  wrote {os.path.basename(p)}  {cv.size}")

    json.dump({"levels": LEVELS,
               "n": int(len(pts)),
               "accept_pred_frac": round(float((scores > 0).mean()), 4),
               "score_min": round(smin, 4), "score_max": round(smax, 4),
               "score_mean": round(float(scores.mean()), 4),
               "ncol_mean": round(float(ncol.mean()), 3),
               "sat_mean": round(float(sat.mean()), 4),
               "edge_mean": round(float(edge.mean()), 2),
               "points": [{"pc": pts[i].tolist(), "score": round(float(scores[i]), 4),
                           "ncol": int(ncol[i]), "edge": round(float(edge[i]), 2)}
                          for i in range(len(pts))]},
              open(os.path.join(OUT, "score.json"), "w", encoding="utf-8"),
              ensure_ascii=False)
    print("  wrote score.json")
    print("PCA4MAP_COLOUR_DONE")


if __name__ == "__main__":
    main()
