#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
把「作者调色板的自由组合」做成一个可以逐个打分的网页。

只枚举**视觉上有区别**的槽位（用量 ≥1%），其余槽位固定为原色。

样本脸的挑选（按作者要求）：
    1. 只取历次盲测里**被评为 4 分**的脸；
    2. 要求**可替换面积大**（被可见槽位覆盖的像素 ≥15%，且黑白占比 ≤75%），
       否则换配色也看不出变化；
    3. 按覆盖度从高到低取 8 张（去重）。

产物：04_phase4_colour_avatar/webapp/config_ranker.html
用法：python build_config_ranker.py [--faces 8]
"""
import argparse
import itertools
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
WEB = os.path.join(P4, "webapp")
LAB = os.path.join(P4, "results", "labels")


def fwd(layers, x):
    h = np.asarray(x, float)
    for i, (W, b) in enumerate(layers):
        h = h @ W.T + b
        if i < len(layers) - 1:
            h = np.maximum(h, 0)
    return h


def hsv(c):
    import colorsys
    r, g, b = np.asarray(c, float) / 15.0
    mx, mn = max(r, g, b), min(r, g, b)
    L = (0.299 * r + 0.587 * g + 0.114 * b) * 15
    s = 0.0 if mx <= 1e-9 else (mx - mn) / mx
    h = 0.0 if mx - mn <= 1e-9 else colorsys.rgb_to_hsv(r, g, b)[0] * 360
    return L, s, h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", default=os.path.join(P4, "results", "twostage",
                                                    "pal16_user.json"))
    ap.add_argument("--min-usage", type=float, default=0.01)
    ap.add_argument("--faces", type=int, default=8)
    ap.add_argument("--min-cov", type=float, default=0.15)
    ap.add_argument("--max-bw", type=float, default=0.75)
    A = ap.parse_args()

    m = json.load(open(A.scheme, encoding="utf-8"))
    usage = json.load(open(os.path.join(P4, "results", "pca4map", "slot_usage.json"),
                           encoding="utf-8"))["deploy"]["s"]
    base = [[round(float(v), 3) for v in c] for c in m["palette"]]
    sets = m.get("palette_sets") or [[c] for c in base]

    vis = [i for i in range(16)
           if len(sets[i]) > 1 and float(usage.get(str(i), 0)) >= A.min_usage]
    opts = {}
    for i in vis:
        cand = [base[i]] + [c for c in sets[i]
                            if np.linalg.norm(np.asarray(c) - np.asarray(base[i])) > 0.6]
        seen, uniq = set(), []
        for c in cand:
            key = tuple(round(float(v), 2) for v in c)
            if key in seen:
                continue
            seen.add(key)
            L, s, h = hsv(c)
            uniq.append({"rgb": [round(float(v), 3) for v in c], "L": round(L, 2),
                         "sat": round(s, 3), "hue": round(h, 1)})
        opts[i] = uniq
    total = 1
    for i in vis:
        total *= len(opts[i])
    print(f"方案 {os.path.basename(A.scheme)}：可见槽位 {vis}，"
          f"候选数 {[len(opts[i]) for i in vis]}，组合 {total} 种")

    # ---- 模型（必须先于选脸）
    g = json.load(open(os.path.join(P4, "results", "twostage", "gray_pca4.json"),
                       encoding="utf-8"))
    GL = [(np.array(L["W"]), np.array(L["b"])) for L in g["layers"]]
    LAY = ([(np.array(L["W"]), np.array(L["b"])) for L in m["body"]]
           + [(np.array(m["head"]["W"]), np.array(m["head"]["b"]))])

    # ---- 候选脸：历次 4 分
    rated = []
    for rnd in ("5", "6", "7", "8"):
        f = os.path.join(LAB, f"colour_round{rnd}.jsonl")
        if not os.path.exists(f):
            continue
        for line in open(f, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("grade") != 4:
                continue
            z = r.get("z") or r.get("pca")
            if not z or len(z) < 4:
                continue
            rated.append([float(v) for v in z[:4]])
    print(f"  候选：历次评为 4 分的 {len(rated)} 条")
    rated = np.array(rated)
    raw = fwd(GL, rated)
    lo = raw.min(1, keepdims=True); hi = raw.max(1, keepdims=True)
    lv = np.clip(np.rint((raw - lo) / np.maximum(hi - lo, 1e-6) * 15), 0, 15)
    idx_all = fwd(LAY, lv / 15.0).reshape(len(rated), 32, 16).argmax(-1)
    cov = np.array([np.isin(row, vis).mean() for row in idx_all])
    bw = np.array([np.isin(row, [2, 15]).mean() for row in idx_all])
    keep = (cov >= A.min_cov) & (bw <= A.max_bw)
    print(f"  可替换面积 ≥{A.min_cov*100:.0f}% 且 黑白 ≤{A.max_bw*100:.0f}%："
          f"{int(keep.sum())} 张")

    picked, seen = [], set()
    for i in np.argsort(-cov):
        if not keep[i]:
            continue
        key = tuple(np.round(rated[i], 2))
        if key in seen:
            continue
        seen.add(key)
        picked.append(int(i))
        if len(picked) == A.faces:
            break
    if len(picked) < A.faces:
        print(f"  警告：只找到 {len(picked)} 张，放宽条件补足到 {A.faces}")
        for i in np.argsort(-cov):
            if i in picked:
                continue
            picked.append(int(i))
            if len(picked) == A.faces:
                break
    Z = rated[picked]
    face_cov = [round(float(cov[i]), 3) for i in picked]
    print(f"  选中 {len(picked)} 张；可替换面积 "
          + ", ".join(f"{c*100:.0f}%" for c in face_cov))
    print(f"  黑白占比 " + ", ".join(f"{bw[i]*100:.0f}%" for i in picked))

    raw = fwd(GL, Z)
    lo = raw.min(1, keepdims=True); hi = raw.max(1, keepdims=True)
    lv = np.clip(np.rint((raw - lo) / np.maximum(hi - lo, 1e-6) * 15), 0, 15)
    idx = fwd(LAY, lv / 15.0).reshape(len(Z), 32, 16).argmax(-1)
    faces = [idx[k].tolist() for k in range(len(Z))]

    combos = []
    for pick in itertools.product(*[range(len(opts[i])) for i in vis]):
        cfg = {str(vis[k]): opts[vis[k]][pick[k]]["rgb"] for k in range(len(vis))}
        combos.append({"pick": list(pick), "cfg": cfg})
    print(f"  枚举 {len(combos)} 种组合，每卡 {len(faces)} 张脸")

    data = {"vis": vis, "opts": {str(i): opts[i] for i in vis}, "base": base,
            "faces": faces, "face_cov": face_cov, "combos": combos,
            "usage": {str(i): float(usage.get(str(i), 0)) for i in vis}}
    tpl = open(os.path.join(HERE, "config_ranker_template.html"), encoding="utf-8").read()
    out = tpl.replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False,
                                                     separators=(",", ":")))
    p = os.path.join(WEB, "config_ranker.html")
    open(p, "w", encoding="utf-8").write(out)
    print(f"wrote {os.path.relpath(p, REPO)}  ({os.path.getsize(p)/1024:.1f} KB)")


if __name__ == "__main__":
    main()
