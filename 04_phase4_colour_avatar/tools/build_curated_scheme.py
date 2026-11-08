#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
把作者手工打分的配色组合固化成「精选先验」调色板方案。

做法：
    · 可见槽位（#4 GRAY / #7 T_LIGHT_GREEN / #14 LIGHT_GRAY）**只从选中的组合里抽**，
      从而保留"哪几个颜色搭在一起"的相关性（红配绿这类组合根本不会出现）；
    · 其余槽位（作者方案里那 6 个低用量槽）仍然按原方案的多候选随机；
    · 颜色先验校正（revive）的 bias 一并带上。

组合的选择：先取全部 4 分；不足 64 时，用 3 分里"最大色相差最小"（最协调）的补足。

产物：results/twostage/pal16_curated{N}.json
用法：python build_curated_scheme.py [--want 64]
"""
import argparse
import colorsys
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


def hsv(c):
    r, g, b = [v / 15.0 for v in c]
    mx, mn = max(r, g, b), min(r, g, b)
    s = 0 if mx <= 1e-9 else (mx - mn) / mx
    h = 0 if mx - mn <= 1e-9 else colorsys.rgb_to_hsv(r, g, b)[0] * 360
    return h, s


def max_hue_gap(cfg, vis):
    hs = [hsv(cfg[str(i)])[0] for i in vis]
    gaps = []
    for a, b in itertools.combinations(hs, 2):
        d = abs(a - b) % 360
        gaps.append(min(d, 360 - d))
    return max(gaps) if gaps else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--want", type=int, default=64)
    ap.add_argument("--tag", default=None)
    A = ap.parse_args()

    d = json.load(open(os.path.join(P4, "results", "palette_shift",
                                    "config_scores.json"), encoding="utf-8-sig"))
    rows = [r for r in d["rows"] if r.get("score") is not None]
    vis = d["vis"]
    fours = [r for r in rows if r["score"] >= 4]
    threes = [r for r in rows if r["score"] == 3]
    threes.sort(key=lambda r: max_hue_gap(r["cfg"], vis))     # 最协调的优先
    pick = fours[:A.want]
    if len(pick) < A.want:
        pick += threes[:A.want - len(pick)]
    pick.sort(key=lambda r: -r["score"])
    print(f"选中 {len(pick)} 个组合（4 分 {sum(1 for r in pick if r['score']>=4)} 个，"
          f"3 分 {sum(1 for r in pick if r['score']==3)} 个）")

    usr = json.load(open(os.path.join(P4, "results", "twostage", "pal16_user.json"),
                         encoding="utf-8"))
    rev = json.load(open(os.path.join(P4, "results", "twostage", "pal16_revive.json"),
                         encoding="utf-8"))
    m = json.loads(json.dumps(usr))
    m["head"]["b"] = rev["head"]["b"]                       # 先验校正
    # 可见槽位交给 configs，这里把它们的 palette_sets 收成单色（原色）
    for i in vis:
        m["palette_sets"][i] = [m["palette"][i]]
        if "slot_p" in m:
            m["slot_p"][i] = 0.0
    configs = []
    for r in pick:
        configs.append({"score": r["score"],
                        "cfg": {str(i): [round(float(v), 3) for v in r["cfg"][str(i)]]
                                for i in vis}})
    m["configs"] = configs
    m["meta"] = dict(m.get("meta", {}), variant="curated",
                     n_configs=len(configs),
                     note="精选先验：可见槽位只在打分 ≥4（不足则补最协调的 3 分）的"
                          "组合里抽；其余槽位沿用原方案随机；带颜色先验校正")
    tag = A.tag or str(len(configs))
    out = os.path.join(P4, "results", "twostage", f"pal16_curated{tag}.json")
    json.dump(m, open(out, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"wrote {os.path.relpath(out, REPO)}")

    # 统计：被固定的三个槽位各用到了多少种颜色
    for i in vis:
        cols = {tuple(np.round(c["cfg"][str(i)], 1)) for c in configs}
        print(f"  槽 {i}：{len(cols)} 种颜色参与")
    gaps = [max_hue_gap(r["cfg"], vis) for r in pick]
    print(f"  选中组合的最大色相差：均值 {np.mean(gaps):.0f}°，最大 {max(gaps):.0f}°"
          f"（原来 144 个里 120°+ 占 {sum(1 for r in rows if max_hue_gap(r['cfg'],vis)>=120)/len(rows)*100:.0f}%）")


if __name__ == "__main__":
    main()
