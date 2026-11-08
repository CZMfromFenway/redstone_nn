#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
生成盲测清单与前向自检参考。

  --round 1  PCA 维数扫描（k=8/16/24/32 + 二期 k8snap）
  --round 2  选择 vs 随机（打分器留 2 / 留 4）
  --round 3  两阶段 vs 单阶段（作者 2026-11-02 指定）：
                two_pal16  二期 PCA4 灰度 → 16 色索引（目标＝原彩图吸附后的索引）
                two_rgb96  二期 PCA4 灰度 → 连续 RGB
                k8         单阶段 16 色
                k8snap     二期 8 色基线
自检参考存的是**最终 96 个颜色值**（含灰度那一段），所以页面同时验证两段管线。

用法：python make_manifest.py --round 3
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
P2 = os.path.join(REPO, "02_phase2_avatar")
LDIR = os.path.join(P4, "results", "labels")

ARMS1 = [
    {"name": "k8", "file": "../results/train/K16_w128_k8/model_snap.json", "k": 8},
    {"name": "k16", "file": "../results/train/K16_w128_k16/model_snap.json", "k": 16},
    {"name": "k24", "file": "../results/train/K16_w128_k24/model_snap.json", "k": 24},
    {"name": "k32", "file": "../results/train/K16_w128_k32/model_snap.json", "k": 32},
    {"name": "k8snap", "file": "../../02_phase2_avatar/webapp/colour_k8snap_model.json", "k": 8},
]


def fwd(layers, x, relu_last=False):
    h = np.asarray(x, dtype=np.float64)
    for i, (W, b, relu) in enumerate(layers):
        h = h @ W.T + b
        if relu and (i < len(layers) - 1 or relu_last):
            h = np.maximum(h, 0)
    return h


def load_palette_model(path):
    j = json.load(open(path, encoding="utf-8"))
    lay = [(np.array(L["W"]), np.array(L["b"]), True) for L in j["body"]]
    lay.append((np.array(j["head"]["W"]), np.array(j["head"]["b"]), False))
    return j, lay


def colours_of(arm, spec):
    z = np.array(spec["z"], dtype=np.float64)
    x = z
    if "pipe" in spec:
        g = json.load(open(os.path.join(HERE, spec["pipe"]["gray"]), encoding="utf-8"))
        gl = [(np.array(L["W"]), np.array(L["b"]), i < len(g["layers"]) - 1)
              for i, L in enumerate(g["layers"])]
        v = fwd(gl, x, relu_last=False)
        lo, hi = float(v.min()), float(v.max())
        span = max(hi - lo, 1e-6)
        alpha = float(spec.get("norm", 1.0))
        raw = np.clip(v, 0, 15)
        full = np.clip((v - lo) / span * 15, 0, 15)
        x = np.round(raw + alpha * (full - raw)) / 15.0
    j, lay = load_palette_model(os.path.join(HERE, spec["file"]))
    raw = fwd(lay, x, relu_last=False)
    if (j.get("meta") or {}).get("output_mode") == "rgb96":
        # 导出的 rgb96 模型输出是 0..1（训练目标 rgb96/15），显示前要乘 15
        return np.clip(np.asarray(raw).reshape(-1) * 15.0, 0, 15).round(4).tolist()
    K = len(j["palette"])
    pal = np.array(j["palette"], dtype=np.float64)
    idx = raw.reshape(32, K).argmax(-1)
    if spec.get("post") == "wool2white":
        if (idx != 15).all() and (idx == 5).any():
            idx[idx == 5] = 15
    if spec.get("post") == "wool2brightest":
        L = pal @ np.array([0.299, 0.587, 0.114])
        bright = int(np.argmax(L))
        if (idx == 5).any() and L[idx].max() <= L[5] + 1e-6:
            idx[idx == 5] = bright
    return pal[idx].round(4).reshape(-1).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", default="1")
    A = ap.parse_args()
    if A.round == "1":
        arms, note = ARMS1, "四期第一轮：16 色 PCA 维数扫描 + 二期基线"
    elif A.round == "2":
        sc = json.load(open(os.path.join(LDIR, "scorer_good.json"), encoding="utf-8"))
        arms = [
            {"name": "k8", "file": "../results/train/K16_w128_k8/model_snap.json", "k": 8},
            {"name": "k8snap", "file": "../../02_phase2_avatar/webapp/colour_k8snap_model.json",
             "k": 8},
            {"name": "k8_sel2", "file": "../results/train/K16_w128_k8/model_snap.json", "k": 8,
             "select": {"weights": sc["weights"], "candidates": 2}},
            {"name": "k8_sel4", "file": "../results/train/K16_w128_k8/model_snap.json", "k": 8,
             "select": {"weights": sc["weights"], "candidates": 4}},
        ]
        note = "四期第二轮：选择 vs 随机；主判据 grade>=3"
    elif A.round == "11":
        fam = json.load(open(os.path.join(LDIR, "scorers_round5.json"),
                             encoding="utf-8"))["families"]
        two = fam["two_stage"]
        sel = {"weights": two["weights"], "kind": two["kind"], "candidates": 4}
        pipe = {"gray": "../results/twostage/gray_pca4.json"}
        arms = [
            {"name": "cur64", "file": "../results/twostage/pal16_curated64.json",
             "k": 4, "pipe": dict(pipe), "select": dict(sel), "norm": 1.0},
            {"name": "cur64_rule", "file": "../results/twostage/pal16_curated64.json",
             "k": 4, "pipe": dict(pipe), "select": dict(sel), "norm": 1.0,
             "post": "wool2white"},
        ]
        note = ("四期第十一轮（2 臂）：精选 64 组合之上，加「画面里没有 QUARTZ 时把 WOOL 换成 "
                "QUARTZ」这条规则（不管 SAND）；主判据 grade>=3")
    elif A.round == "10":
        fam = json.load(open(os.path.join(LDIR, "scorers_round5.json"),
                             encoding="utf-8"))["families"]
        two = fam["two_stage"]
        sel = {"weights": two["weights"], "kind": two["kind"], "candidates": 4}
        pipe = {"gray": "../results/twostage/gray_pca4.json"}
        arms = [
            {"name": "cur_base", "file": "../results/twostage/pal16_revive_user.json",
             "k": 4, "pipe": dict(pipe), "select": dict(sel), "norm": 1.0},
            {"name": "cur_64", "file": "../results/twostage/pal16_curated64.json",
             "k": 4, "pipe": dict(pipe), "select": dict(sel), "norm": 1.0},
        ]
        note = ("四期第十轮（2 臂）：精选配色先验（作者打分的 64 个组合，可见槽位只在其中抽，"
                "其余槽位仍随机）vs 原方案；主判据 grade>=3")
    elif A.round == "9":
        fam = json.load(open(os.path.join(LDIR, "scorers_round5.json"),
                             encoding="utf-8"))["families"]
        two = fam["two_stage"]
        sel = {"weights": two["weights"], "kind": two["kind"], "candidates": 4}
        pipe = {"gray": "../results/twostage/gray_pca4.json"}
        arms = [
            {"name": "n_stretch", "file": "../results/twostage/pal16_revive_user.json",
             "k": 4, "pipe": dict(pipe), "select": dict(sel), "norm": 1.0},
            {"name": "n_s50", "file": "../results/twostage/pal16_revive_user.json",
             "k": 4, "pipe": dict(pipe), "select": dict(sel), "norm": 0.5},
            {"name": "n_keepwhite", "file": "../results/twostage/pal16_revive_kw_user.json",
             "k": 4, "pipe": dict(pipe), "select": dict(sel), "norm": 1.0,
             "post": "wool2white"},
        ]
        note = ("四期第九轮（3 臂）：灰度归一化强度（拉满 vs α=0.5）与「保 WOOL/QUARTZ + "
                "无白则 WOOL→白」；主判据 grade>=3")
    elif A.round == "8":
        fam = json.load(open(os.path.join(LDIR, "scorers_round5.json"),
                             encoding="utf-8"))["families"]
        two = fam["two_stage"]
        sel = {"weights": two["weights"], "kind": two["kind"], "candidates": 4}
        pipe = {"gray": "../results/twostage/gray_pca4.json"}
        arms = [
            {"name": "pal_base", "file": "../results/twostage/pal16.json", "k": 4,
             "pipe": dict(pipe), "select": dict(sel)},
            {"name": "pal_revive", "file": "../results/twostage/pal16_revive.json", "k": 4,
             "pipe": dict(pipe), "select": dict(sel)},
            {"name": "pal_rev_user", "file": "../results/twostage/pal16_revive_user.json",
             "k": 4, "pipe": dict(pipe), "select": dict(sel)},
        ]
        note = ("四期第八轮（3 臂）：颜色先验校正（只改最终层 bias，把数据里真实的类别占比找回来）"
                "是否改善观感 / 多样性；第三臂叠加作者的调色板方案。主判据 grade>=3")
    elif A.round == "7":
        fam = json.load(open(os.path.join(LDIR, "scorers_round5.json"),
                             encoding="utf-8"))["families"]
        two = fam["two_stage"]
        sel = {"weights": two["weights"], "kind": two["kind"], "candidates": 4}
        pipe = {"gray": "../results/twostage/gray_pca4.json"}
        arms = [
            {"name": "pal_base", "file": "../results/twostage/pal16.json", "k": 4,
             "pipe": dict(pipe), "select": dict(sel)},
            {"name": "pal_s30", "file": "../results/twostage/pal16_s30.json", "k": 4,
             "pipe": dict(pipe), "select": dict(sel)},
            {"name": "pal_user", "file": "../results/twostage/pal16_user.json", "k": 4,
             "pipe": dict(pipe), "select": dict(sel)},
        ]
        note = ("四期第七轮（3 臂）：作者自配色 vs 原色板 vs 第六轮最好的 s30。"
                "pal_user 是多候选方案（9 槽 31 候选、逐张抽样，可见配置 144 种）；"
                "主判据 grade>=3")
    elif A.round == "6":
        fam = json.load(open(os.path.join(LDIR, "scorers_round5.json"),
                             encoding="utf-8"))["families"]
        two = fam["two_stage"]
        sel = {"weights": two["weights"], "kind": two["kind"], "candidates": 4}
        pipe = {"gray": "../results/twostage/gray_pca4.json"}
        arms = [
            {"name": "pal_base", "file": "../results/twostage/pal16.json", "k": 4,
             "pipe": dict(pipe), "select": dict(sel)},
            {"name": "pal_s30", "file": "../results/twostage/pal16_s30.json", "k": 4,
             "pipe": dict(pipe), "select": dict(sel)},
            {"name": "pal_s60", "file": "../results/twostage/pal16_s60.json", "k": 4,
             "pipe": dict(pipe), "select": dict(sel)},
            {"name": "pal_s100", "file": "../results/twostage/pal16_s100.json", "k": 4,
             "pipe": dict(pipe), "select": dict(sel)},
        ]
        note = ("四期第六轮：保亮度换色相（同模型、同筛选 4 选 1，只换调色板）；"
                "主判据 grade>=3")
    elif A.round == "5":
        fam = json.load(open(os.path.join(LDIR, "scorers_round5.json"),
                             encoding="utf-8"))["families"]
        two = fam["two_stage"]; snp = fam["k8snap"]
        arms = [
            {"name": "two_pal16", "file": "../results/twostage/pal16.json", "k": 4,
             "pipe": {"gray": "../results/twostage/gray_pca4.json"}},
            {"name": "two_sel4", "file": "../results/twostage/pal16.json", "k": 4,
             "pipe": {"gray": "../results/twostage/gray_pca4.json"},
             "select": {"weights": two["weights"], "kind": two["kind"], "candidates": 4}},
            {"name": "k8snap", "file": "../../02_phase2_avatar/webapp/colour_k8snap_model.json",
             "k": 8},
            {"name": "snap_sel4", "file": "../../02_phase2_avatar/webapp/colour_k8snap_model.json",
             "k": 8, "select": {"weights": snp["weights"], "kind": snp["kind"],
                                "candidates": 4}},
        ]
        note = (f"四期第五轮：真正的筛选实验（打分器合并第三、四轮 n=240；"
                f"两阶段 {two['kind']} CV AUC {two['cv_auc']}，"
                f"k8snap {snp['kind']} CV AUC {snp['cv_auc']}）；留 4 选 1；主判据 grade>=3")
    elif A.round == "4":
        sc = json.load(open(os.path.join(LDIR, "scorers_round3.json"), encoding="utf-8"))["scorers"]
        arms = [
            {"name": "two_pal16", "file": "../results/twostage/pal16.json", "k": 4,
             "pipe": {"gray": "../results/twostage/gray_pca4.json"}},
            {"name": "two_sel", "file": "../results/twostage/pal16.json", "k": 4,
             "pipe": {"gray": "../results/twostage/gray_pca4.json"},
             "select": {"weights": sc["two_pal16"]["weights"], "kind": "quad",
                        "candidates": 3}},
            {"name": "k8snap", "file": "../../02_phase2_avatar/webapp/colour_k8snap_model.json",
             "k": 8},
            {"name": "snap_sel", "file": "../../02_phase2_avatar/webapp/colour_k8snap_model.json",
             "k": 8, "select": {"weights": sc["k8snap"]["weights"], "kind": "linear",
                                "candidates": 3}},
        ]
        note = ("四期第四轮：筛选 vs 随机（打分器按臂拟合于第三轮；two_pal16 用二次 15 项，"
                "k8snap 用线性 9 项）；主判据 grade>=3")
    else:
        arms = [
            {"name": "two_pal16", "file": "../results/twostage/pal16.json", "k": 4,
             "pipe": {"gray": "../results/twostage/gray_pca4.json"}},
            {"name": "p22_pal16", "file": "../results/twostage/pal16_p22.json", "k": 22,
             "pipe": {"gray": "../results/twostage/gray_pca22.json"}},
            {"name": "k8", "file": "../results/train/K16_w128_k8/model_snap.json", "k": 8},
            {"name": "k8snap", "file": "../../02_phase2_avatar/webapp/colour_k8snap_model.json",
             "k": 8},
        ]
        note = ("四期第三轮：两阶段（PCA4 灰度→16 色 / PCA22 灰度→16 色）vs 单阶段；"
                "主判据 grade>=3；按作者指示已排除连续 RGB 臂")
    json.dump({"round": int(A.round), "note": note, "arms": arms},
              open(os.path.join(HERE, "label_manifest.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)

    sel = {}
    for a in arms:
        k = a["k"]
        z = [round(float(v), 4) for v in
             np.clip(np.random.default_rng(2026).standard_normal(k), -3.5, 3.5)]
        spec = dict(a, z=z)
        j = json.load(open(os.path.join(HERE, a["file"]), encoding="utf-8"))
        if j.get("palette_sets"):
            print(f"  {a['name']:10s} k={k:2d} 多候选调色板 → 跳过自检参考（页面逐张抽样）")
            continue
        cols = colours_of(a, spec)
        sel[a["name"]] = {"k": k, "z": z, "colours": cols}
        print(f"  {a['name']:10s} k={k:2d} colours[:6]={np.round(cols[:6],3).tolist()}")
    json.dump(sel, open(os.path.join(HERE, "selftest.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print(f"wrote label_manifest.json（round {A.round}，{len(arms)} 臂）+ selftest.json")


if __name__ == "__main__":
    main()
