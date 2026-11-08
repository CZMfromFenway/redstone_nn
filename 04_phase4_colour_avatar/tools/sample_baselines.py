#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
出「对照组」的样本，写成与 jim_diffusion_run.js / train_mlp_diffusion.py 相同的 JSON，
以便用同一个 jim_diffusion_gallery.py / diffusion_compare.py 出图（同一套渲染、同一套代理指标）。

  --which real   真实数据（cache.npz 的 rgb96），4 bit 原样 + 吸附到 16 槽
  --which cur    现役两阶段（PCA4 灰度 -> 16 色头）＋ **逐张换调色板** ＋ 4 选 1 筛选 ＋ WOOL->QUARTZ
  --which cur0   同上但不筛选
  --which curfix 同上但**不换调色板**（固定 base palette）——用来量化"换调色板"这一档的作用

「逐张换调色板」的实现逐字照 `tools/label.html` 的 `samplePalette()`：
  1. 若模型有 `configs`（精选 64 组合）：先随机抽一个组合，把它的槽位写死（fixed）；
  2. 其余槽位看 `palette_sets[i]`：
     - 只有 1 个候选 → 直接用它（注意：**不是**用 base）；
     - 多个候选 → 以 `slot_p[i]` 的概率从候选里按 `slot_w[i]` 权重抽一个，否则保留 base。
  这一档在 06 号文档里实测值 +20 点（60% -> 80%），所以不打开它比"现役"是低估的。

用法：
  python sample_baselines.py --which real --n 200
  python sample_baselines.py --which cur  --n 200 --select 4
  python sample_baselines.py --which curfix --n 200 --select 4
"""
import argparse
import json
import os
import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
P4 = os.path.dirname(HERE)
REPO = os.path.dirname(P4)
OUTDIR = os.path.join(P4, "results", "diffusion")
CACHE = os.path.join(REPO, "02_phase2_avatar", "results", "colour", "cache.npz")

ROW = np.array([p % 8 for p in range(32)])
COL = np.array([p // 8 for p in range(32)])


def mirror_to64(col32):
    img = np.zeros((64, 3), dtype=np.float64)
    img[ROW * 8 + COL] = col32
    img[ROW * 8 + (7 - COL)] = col32
    return img


def hexes_of(col32):
    img = mirror_to64(np.asarray(col32, dtype=np.float64).reshape(32, 3))
    return ["#%02x%02x%02x" % tuple(int(round(v * 17)) for v in np.clip(img[k], 0, 15))
            for k in range(64)]


def quad_score(w, z):
    f = [1.0]
    f += list(z)
    f += [v * v for v in z]
    for i in range(len(z)):
        for j in range(i + 1, len(z)):
            f.append(z[i] * z[j])
    return float(np.dot(np.array(w), np.array(f)))


def snap_to_palette(col96, palette):
    lev = np.asarray(col96, dtype=np.float64).reshape(32, 3)
    d = ((lev[:, None, :] - palette[None, :, :]) ** 2).sum(-1)
    return palette[d.argmin(1)]


# ---------------------------------------------------------------- 模型前向
def fwd(layers, x, relu_last=False):
    """layers = [(W[out][in], b)], 最后一层不带 ReLU（与页面一致）。"""
    h = np.asarray(x, dtype=np.float64)
    for i, (W, b) in enumerate(layers):
        h = h @ np.asarray(W).T + np.asarray(b)
        if i < len(layers) - 1 or relu_last:
            h = np.maximum(h, 0.0)
    return h


def load_gray(path):
    g = json.load(open(path, encoding="utf-8"))
    return [(np.array(L["W"]), np.array(L["b"])) for L in g["layers"]]


def sample_palette(model, base, rng, enable=True):
    """逐字照 label.html 的 samplePalette()。"""
    pal = np.array(base, dtype=np.float64).copy()
    fixed = set()
    if not enable:
        return pal
    cfgs = model.get("configs")
    if cfgs:
        c = cfgs[int(rng.integers(0, len(cfgs)))]
        for k, v in c["cfg"].items():
            pal[int(k)] = np.array(v, dtype=np.float64)
            fixed.add(int(k))
    sets = model.get("palette_sets")
    if sets:
        sp = model.get("slot_p") or [1.0] * len(pal)
        sw = model.get("slot_w") or [None] * len(pal)
        for i in range(len(pal)):
            if i in fixed:
                continue
            s = sets[i] if i < len(sets) else None
            if not s:
                continue
            if len(s) <= 1:
                pal[i] = np.array(s[0], dtype=np.float64)   # 页面口径：单候选就用它
                continue
            p = sp[i] if i < len(sp) and sp[i] is not None else 1.0
            if rng.random() >= p:
                continue                                     # 保留 base
            w = sw[i] if i < len(sw) and sw[i] else [1.0] * len(s)
            w = np.array(w, dtype=np.float64)
            tot = w.sum()
            if tot <= 0:
                continue
            r = rng.random() * tot
            for k in range(len(s)):
                r -= w[k]
                if r <= 0:
                    pal[i] = np.array(s[k], dtype=np.float64)
                    break
            else:
                pal[i] = np.array(s[-1], dtype=np.float64)
    return pal


def palette_colours(model, x, post, pal):
    lay = [(np.array(L["W"]), np.array(L["b"])) for L in model["body"]]
    h = fwd(lay, x, relu_last=True)
    lg = np.asarray(model["head"]["W"]) @ h + np.asarray(model["head"]["b"])
    K = len(pal)
    idx = lg.reshape(32, K).argmax(-1)
    if post == "wool2white":
        # 逐字照 label.html：**只有画面里完全没有 QUARTZ(15)** 时才把 WOOL(5) 提成白
        if not (idx == 15).any() and (idx == 5).any():
            idx[idx == 5] = 15
    elif post == "wool2brightest":
        L = pal @ np.array([0.299, 0.587, 0.114])
        bright = int(np.argmax(L))
        if (idx == 5).any() and L[idx].max() <= L[5] + 1e-6:
            idx[idx == 5] = bright
    return pal[idx], idx


def gray32_of(gray_layers, z, norm=1.0):
    g = fwd(gray_layers, z, relu_last=False)
    lo, hi = float(g.min()), float(g.max())
    span = max(hi - lo, 1e-6)
    raw = np.clip(g, 0, 15)
    full = np.clip((g - lo) / span * 15, 0, 15)
    return np.round(raw + norm * (full - raw)) / 15.0


def verify_against_page():
    """与 `label.html` 的 JS 逐值对账（防"度量本身是错的"）。

    参考值由 `tools/_cur_crosscheck.js` 生成：它把页面里**原样的** coloursOf() 跑一遍，
    对同一批 z、同一套（强制）调色板输出 96 个颜色值。这里用 Python 再算一遍，必须一致。

    历史教训：2026-11-08 本文件把 wool2white 写成 `not (idx==15).all()`（应为 `.any()`），
    于是"画面里有白"时也会把 WOOL 提成白 → 第十二轮的 cur_pal 样本整体偏白，
    好脸率被压到 38%（页面上的同一配置是 72.5%）。所以对账这一步必须有。
    """
    js_path = os.path.join(OUTDIR, "..", "labels", "_crosscheck_js.json")
    if not os.path.exists(js_path):
        print("没有 %s —— 先跑 `node tools/_cur_crosscheck.js`" % os.path.relpath(js_path, P4))
        return False
    ref = json.load(open(js_path, encoding="utf-8-sig"))
    man = json.load(open(os.path.join(HERE, "label_manifest.json"), encoding="utf-8"))
    arm = [x for x in man["arms"] if x["name"] == "cur64_rule"][0]
    model = json.load(open(os.path.join(HERE, arm["file"]), encoding="utf-8"))
    gray = load_gray(os.path.join(HERE, arm["pipe"]["gray"]))
    base = np.array(model["palette"], dtype=np.float64)
    worst, n = 0.0, 0
    for case in ref["cases"]:
        if case["pal"] != "base":
            continue
        z = np.array(case["z"], dtype=np.float64)
        g32 = gray32_of(gray, z, arm.get("norm", 1.0))
        col, _ = palette_colours(model, g32, arm.get("post"), base)
        got = np.round(np.asarray(col).reshape(-1), 4)
        want = np.array(case["cols"], dtype=np.float64)
        worst = max(worst, float(np.abs(got - want).max()))
        n += 1
    ok = worst < 1e-6
    print("与页面 JS 对账：%d 组 · 96 个颜色值 · 最大差 %.8f  %s"
          % (n, worst, "✓ 通过" if ok else "✗ 不一致"))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="real", choices=["real", "cur", "cur0", "curfix"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--select", type=int, default=4)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--pal-seed", type=int, default=999, help="逐张换调色板的抽签流")
    ap.add_argument("--out", default="")
    ap.add_argument("--verify", action="store_true", help="只与页面 JS 对账")
    a = ap.parse_args()

    if a.verify:
        sys.exit(0 if verify_against_page() else 1)

    rng = np.random.default_rng(a.seed)
    rng_pal = np.random.default_rng(a.pal_seed)

    rows = []
    if a.which == "real":
        z = np.load(CACHE)
        pal_all = json.load(open(os.path.join(P4, "results", "fit", "deploy_palette.json"),
                                 encoding="utf-8"))
        palette = np.array(pal_all["K16_w128_k16"]["palette"], dtype=np.float64)
        rgb = z["rgb96"][-4000:].astype(np.float64)
        idx = rng.permutation(len(rgb))[:a.n]
        for k in idx:
            lev = rgb[k].reshape(32, 3)
            snap = snap_to_palette(lev, palette)
            rows.append({"seed": int(k), "ms": 0.0,
                         "raw": hexes_of(lev), "snapped": hexes_of(snap),
                         "sym": 1.0, "ncol": int(len(set(hexes_of(snap))))})
        meta = {"which": "real", "note": "真实数据（cache rgb96 尾部），吸附到 16 槽后算指标"}
    else:
        man = json.load(open(os.path.join(HERE, "label_manifest.json"), encoding="utf-8"))
        arm = [x for x in man["arms"] if x["name"] == "cur64_rule"][0]
        model = json.load(open(os.path.join(HERE, arm["file"]), encoding="utf-8"))
        base = model["palette"]
        gray = load_gray(os.path.join(HERE, arm["pipe"]["gray"]))
        w = arm["select"]["weights"]
        post = arm.get("post")
        use_pal = a.which != "curfix"
        for i in range(a.n):
            if a.which == "cur0":
                z = np.clip(rng.standard_normal(4), -3.5, 3.5)
            else:
                best, bz = -1e18, None
                for _ in range(a.select):
                    zz = np.clip(rng.standard_normal(4), -3.5, 3.5)
                    s = quad_score(w, zz)
                    if s > best:
                        best, bz = s, zz
                z = bz
            g32 = gray32_of(gray, z, arm.get("norm", 1.0))
            pal = sample_palette(model, base, rng_pal, enable=use_pal)
            col, idx = palette_colours(model, g32, post, pal)
            rows.append({"seed": int(i), "ms": 0.0,
                         "raw": hexes_of(col), "snapped": hexes_of(col),
                         "sym": 1.0, "ncol": int(len(set(hexes_of(col)))),
                         "z": [round(float(v), 3) for v in z],
                         "pal": [list(map(float, c)) for c in pal]})
        meta = {"which": a.which,
                "select": 0 if a.which == "cur0" else a.select,
                "palette_swap": bool(use_pal),
                "model": arm["file"], "post": post,
                "note": "现役两阶段：PCA4 灰度 -> 16 色头（%s%s）"
                        % ("逐张换调色板" if use_pal else "固定调色板",
                           "，不筛选" if a.which == "cur0" else "，4 选 1")}

    out = a.out or os.path.join(OUTDIR, "%s_n%d.json" % (a.which, a.n))
    if not os.path.isabs(out):
        out = os.path.join(P4, out)          # --out 一律相对「四期目录」
    summary = {"source": "sample_baselines.py", "tag": a.which,
               "steps": 0 if a.which == "real" else 1,
               "cfg_w": 0, "qb": None, "alpha": 0, "k": 16, "sym_tau": None,
               "palette_size": 16, "palette_source": "results/twostage/pal16_curated64.json",
               "pal_arm": "K16_w128_k16", "ms_mean": 0.0,
               "mac_per_pass": {"cur": 88380, "cur0": 88320, "curfix": 88380, "real": 0}[a.which],
               **meta, "samples": rows}
    os.makedirs(OUTDIR, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False)
    print("写了 %s（%d 张，%s）" % (os.path.relpath(out, P4), len(rows),
                                 "换调色板" if summary.get("palette_swap") else "固定调色板"))


if __name__ == "__main__":
    main()

