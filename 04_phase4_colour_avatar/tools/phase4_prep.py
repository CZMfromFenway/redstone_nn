#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
四期第一批实验的「地基数字」：颜色 PCA 维数、16 色调色板与吸附、红石成本。

这个脚本**不训练任何模型**，只回答三个在开训之前必须先定下来的问题：

  1. PCA 维数 8/16/24/32 各能解释多少颜色方差？各维之后协方差还是不是单位阵？
     （二期只验过 8 维，且 8 维丢 25% 方差正是四期想加维的理由。）
  2. 16 个「最佳颜色」从数据里怎么选、吸附到地图色之后还剩几种、误差多大？
     （二期 K=8 吸附后 8 色无撞色；K=16 有 5 条离最近地图色不到 1 级，四期要重测。）
  3. 各候选结构的乘法数，以及一个二期没算过的量：**输入随机源的条数**。

数据一律只读：`02_phase2_avatar/results/colour/cache.npz`（rgb96 / rgb192 / cpc / cmean）
          `02_phase2_avatar/data/mc_map_colors.json`（62 基色 × 4 修饰）
产物：`04_phase4_colour_avatar/results/prep/palette16.json`
      `04_phase4_colour_avatar/results/prep/palette16.png`
      `04_phase4_colour_avatar/results/prep/prep.json`

用法：python phase4_prep.py
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
OUT = os.path.join(P4, "results", "prep")
os.makedirs(OUT, exist_ok=True)

LUM = np.array([0.299, 0.587, 0.114])


def font(sz, bold=False):
    for fn in ("msyhbd.ttc" if bold else "msyh.ttc", "simhei.ttf"):
        p = os.path.join("C:\\Windows\\Fonts", fn)
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, sz)
            except Exception:
                pass
    return ImageFont.load_default()


def load_data():
    z = np.load(os.path.join(P2, "results", "colour", "cache.npz"))
    return z["rgb96"].astype(np.float64), z["rgb192"].astype(np.float64)


def load_mapcolours():
    j = json.load(open(os.path.join(P2, "data", "mc_map_colors.json"), encoding="utf-8"))
    mods = j["_modifiers"]
    bases = np.array([b["rgb"] for b in j["bases"]], dtype=np.float64)   # 0..255
    names = [b["name"] for b in j["bases"]]
    cols, tag = [], []
    for m in ("HIGH", "NORMAL", "LOW", "LOWEST"):
        c = np.clip(bases * mods[m], 0, 255)
        cols.append(c / 17.0)                      # 换成 0..15 级
        tag += [(n, m) for n in names]
    return np.concatenate(cols), tag, [b["rgb"] for b in j["bases"]], names, mods


# ------------------------------------------------------------------ 1. PCA


def pca_report(rgb192, ks=(8, 16, 24, 32)):
    x = rgb192
    mu = x.mean(0, keepdims=True)
    sub = np.random.default_rng(0).choice(len(x), 40000, replace=False)
    _, S, Vt = np.linalg.svd(x[sub] - mu, full_matrices=False)
    ev = S ** 2
    ev = ev / ev.sum()
    cum = np.cumsum(ev)
    out = {"cum_var": {str(k): round(float(cum[k - 1]), 4) for k in ks},
           "recon_rms_lsb": {}}
    # 每个 k 的「重建 RMS（以 0..15 级为单位）」：丢掉的分量还有多少能量
    for k in ks:
        lost = float(ev[k:].sum())
        # 每像素每通道的残差标准差（级）：总方差 × 丢掉的比例，开根
        total_var = float(x.var(0).mean())
        out["recon_rms_lsb"][str(k)] = round(float(np.sqrt(total_var * lost)), 3)
    # 各 k 之后的前 k 个系数协方差是否仍是单位阵
    comps = Vt[:32]
    coef = (x[sub] - mu) @ comps.T
    sd = coef.std(0, keepdims=True) + 1e-9
    zs = coef / sd
    iso = {}
    for k in ks:
        c = np.cov(zs[:, :k].T)
        w = np.linalg.eigvalsh(c)
        off = np.abs(np.corrcoef(zs[:, :k].T) - np.eye(k))
        iso[str(k)] = {"eig_min": round(float(w.min()), 4),
                       "eig_max": round(float(w.max()), 4),
                       "cond": round(float(w.max() / w.min()), 3),
                       "mean_abs_offdiag_corr": round(float(off.sum() / (k * k - k)), 5)}
    out["isotropy"] = iso
    return out, comps


# ------------------------------------------------------------------ 2. 16 色调色板


def kmeans(X, k, iters=60, seed=0):
    rng = np.random.default_rng(seed)
    # k-means++
    C = [X[rng.integers(len(X))]]
    d2 = ((X - C[0]) ** 2).sum(1)
    for _ in range(k - 1):
        p = d2 / d2.sum()
        i = rng.choice(len(X), p=p)
        C.append(X[i])
        d2 = np.minimum(d2, ((X - X[i]) ** 2).sum(1))
    C = np.array(C)
    for _ in range(iters):
        d = ((X[:, None, :] - C[None]) ** 2).sum(-1)
        lab = d.argmin(1)
        new = C.copy()
        for j in range(k):
            m = lab == j
            if m.any():
                new[j] = X[m].mean(0)
        if np.allclose(new, C, atol=1e-6):
            break
        C = new
    return C, lab


def saturate(pal, mult=1.2):
    g = pal.mean(1, keepdims=True)
    return np.clip(g + mult * (pal - g), 0, 15)


def snap(pal, candidates):
    d = ((pal[:, None, :] - candidates[None]) ** 2).sum(-1)
    idx = d.argmin(1)
    return candidates[idx], idx, np.sqrt(d[np.arange(len(pal)), idx])


def quant_err(pix, pal):
    d = np.sqrt(((pix[:, None, :] - pal[None]) ** 2).sum(-1)).min(1)
    return float(d.mean()), float(np.percentile(d, 95))


def hexof(c):
    v = np.clip(np.rint(np.asarray(c) * 17), 0, 255).astype(int)
    return "#%02X%02X%02X" % tuple(v)


def main():
    rgb96, rgb192 = load_data()
    N = len(rgb96)
    print(f"data: {N:,} 张彩色样本，rgb96 per-pixel 0..15")
    cands_all, tags_all, base_rgb, names, mods = load_mapcolours()
    HIGH = np.clip(np.array(base_rgb, float) * mods["HIGH"], 0, 255) / 17.0
    print(f"map colours: {len(cands_all)} 个（61 基色 × 4 修饰，含透明 NONE 已由上游剔除）")

    pca, comps = pca_report(rgb192)
    print("\n== PCA ==")
    for k, v in pca["cum_var"].items():
        iso = pca["isotropy"][k]
        print(f"  k={k:>2}: 累计方差 {v*100:5.1f}%  重建残差 {pca['recon_rms_lsb'][k]:5.3f} 级"
              f"  协方差条件数 {iso['cond']:.3f}  |offdiag| {iso['mean_abs_offdiag_corr']:.5f}")

    # 像素池：按频率抽样
    pix96 = rgb96.reshape(-1, 32, 3)                      # N,32,3
    rng = np.random.default_rng(3)
    sel = rng.choice(N, 40000, replace=False)
    P = pix96[sel].reshape(-1, 3)
    print(f"\n== 调色板 ==")
    print(f"  像素池 {len(P):,} 个（{len(sel):,} 张脸 × 32 位置）")

    result = {}
    for K in (8, 16):
        free, lab = kmeans(P, K, seed=1)
        # 按簇大小排序，便于阅读
        order = np.argsort(-np.bincount(lab, minlength=K))
        free = free[order]
        sat = saturate(free)
        for name, pal in ((f"K{K}_free", free), (f"K{K}_sat", sat)):
            snapH, idxH, errH = snap(pal, HIGH)
            snapA, idxA, errA = snap(pal, cands_all)
            e_pal, e_p95 = quant_err(P, pal)
            result[name] = {
                "palette": np.round(pal, 4).tolist(),
                "snap_high": {"colours": np.round(snapH, 4).tolist(),
                              "names": [tags_all[0][0][0] and names[i] for i in idxH],
                              "err_mean": round(float(errH.mean()), 3),
                              "err_max": round(float(errH.max()), 3),
                              "distinct": int(len({tuple(np.round(c, 3)) for c in snapH}))},
                "snap_all": {"colours": np.round(snapA, 4).tolist(),
                             "names": [(tags_all[i][0], tags_all[i][1]) for i in idxA],
                             "err_mean": round(float(errA.mean()), 3),
                             "err_max": round(float(errA.max()), 3),
                             "distinct": int(len({tuple(np.round(c, 3)) for c in snapA}))},
                "quant_err_mean": round(e_pal, 3),
                "quant_err_p95": round(e_p95, 3),
                "hex_free": [hexof(c) for c in pal],
                "hex_snap_high": [hexof(c) for c in snapH],
                "hex_snap_all": [hexof(c) for c in snapA],
            }
            print(f"  {name:8s} 吸附(HIGH 61): 误差 {errH.mean():.2f}±{errH.max():.2f} 级、"
                  f"撞色后 {result[name]['snap_high']['distinct']}/{K} 种 | "
                  f"吸附(全部 244): {errA.mean():.2f}±{errA.max():.2f} 级、"
                  f"{result[name]['snap_all']['distinct']}/{K} 种 | "
                  f"量化误差 {e_pal:.2f} 级 (p95 {e_p95:.2f})")

    # 地图介质本身的下限：像素到最近地图色的距离
    def nearest_stats(pix, cands):
        d = np.sqrt(((pix[:, None, :] - cands[None]) ** 2).sum(-1))
        dm = d.min(1)
        idx = d.argmin(1)
        used = len(np.unique(idx))
        return {"mean": round(float(dm.mean()), 3),
                "p95": round(float(np.percentile(dm, 95)), 3),
                "max": round(float(dm.max()), 3),
                "frac_within_1lv": round(float((dm <= 1).mean()), 3),
                "frac_within_2lv": round(float((dm <= 2).mean()), 3),
                "distinct_map_colours_used": int(used)}

    S = 200000
    sub = P[np.random.default_rng(5).choice(len(P), S, replace=False)]
    medium = {"to_61_HIGH": nearest_stats(sub, HIGH),
              "to_244_all": nearest_stats(sub, cands_all)}
    print("\n== 地图介质下限（真实像素 → 最近地图色）==")
    for k2, v in medium.items():
        print(f"  {k2:10s} 平均 {v['mean']:.2f} 级、p95 {v['p95']:.2f}、最大 {v['max']:.2f}、"
              f"≤1 级 {v['frac_within_1lv']*100:.0f}%、≤2 级 {v['frac_within_2lv']*100:.0f}%、"
              f"实际用到 {v['distinct_map_colours_used']} 种")

    # ---------------------------------------------------------------- 3. 成本
    print("\n== 成本（乘法 = k·W + W·W + W·512；K=16）==")
    rows = []
    for k in (8, 16, 24, 32):
        for W in (128, 256):
            m = k * W + W * W + W * 512
            rows.append({"k": k, "W": W, "multiplies": m, "params": m + (W + W + 512),
                         "input_random_sources": k, "deployable": W <= 128})
            print(f"  k={k:>2} W={W:>3}: {m:>9,} 乘法   随机源 {k:>2} 条   "
                  f"{'可红石' if W <= 128 else '仅作上限参照'}")

    json.dump({"pca": pca, "palettes": result, "medium_floor": medium, "cost": rows},
              open(os.path.join(OUT, "prep.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    json.dump(result["K16_sat"], open(os.path.join(OUT, "palette16.json"), "w",
                                      encoding="utf-8"), indent=2, ensure_ascii=False)

    # ---------------------------------------------------------------- 图
    K = 16
    pal = np.array(result["K16_sat"]["palette"])
    snapH = np.array(result["K16_sat"]["snap_high"]["colours"])
    snapA = np.array(result["K16_sat"]["snap_all"]["colours"])
    namesH = result["K16_sat"]["snap_high"]["names"]
    namesA = result["K16_sat"]["snap_all"]["names"]
    SW, GAP, H = 74, 10, 74
    W = 40 + K * (SW + GAP) + 20
    cv = Image.new("RGB", (W, 430), (16, 18, 22))
    dr = ImageDraw.Draw(cv)
    dr.text((16, 12), "四期 16 色方案：数据里选出的 16 色 → 吸附到地图色", font=font(20),
            fill=(234, 238, 244))
    dr.text((16, 40), "上：自由 k-means 16 色（饱和 ×1.2）；中：吸附到 61 个 HIGH 色；"
                      "下：吸附到全部 244 个地图色（61 基色 × 4 修饰）",
            font=font(12), fill=(148, 160, 176))
    y = 66
    for row, tag, nm in ((pal, "free", None), (snapH, "HIGH", namesH), (snapA, "ALL", namesA)):
        dr.text((16, y + 26), tag, font=font(12), fill=(112, 170, 254))
        for i, c in enumerate(row):
            x = 40 + i * (SW + GAP)
            v = np.clip(np.rint(c * 17), 0, 255).astype(int)
            dr.rectangle([x, y, x + SW - 1, y + H - 1], fill=tuple(v))
            if nm:
                if isinstance(nm[i], tuple):
                    base, mod = nm[i]
                    tagm = {"HIGH": "HI", "NORMAL": "NO", "LOW": "LO", "LOWEST": "LW"}[mod]
                    t = base.replace("TERRACOTTA_", "T_").replace("COLOR_", "C_")[:12] + "·" + tagm
                else:
                    t = str(nm[i])
                dr.rectangle([x, y + H - 18, x + SW - 1, y + H - 1], fill=(20, 22, 26))
                dr.text((x + 2, y + H - 17), t, font=font(8), fill=(214, 222, 234))
        y += H + 26
    dr.text((16, y + 6), "吸附误差（16 色平均 / 最大，单位=0..15 级）："
                         f"HIGH {result['K16_sat']['snap_high']['err_mean']:.2f} / "
                         f"{result['K16_sat']['snap_high']['err_max']:.2f}   ·   "
                         f"ALL {result['K16_sat']['snap_all']['err_mean']:.2f} / "
                         f"{result['K16_sat']['snap_all']['err_max']:.2f}",
            font=font(11), fill=(200, 210, 226))
    cv.save(os.path.join(OUT, "palette16.png"))
    print(f"\nwrote {os.path.relpath(os.path.join(OUT, 'palette16.png'), REPO)}")
    print(f"wrote {os.path.relpath(os.path.join(OUT, 'prep.json'), REPO)}")
    print("PHASE4_PREP_DONE")


if __name__ == "__main__":
    main()
