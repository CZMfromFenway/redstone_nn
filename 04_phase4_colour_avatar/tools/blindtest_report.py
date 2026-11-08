#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
盲测全记录：把每一轮每个臂生成的随机头像按**等级描框**重绘出来，并出结构图。

等级配色（作者指定方向：最好→最差）：
    grade 4 = 青   grade 3 = 绿   grade 2 = 黄   grade 1 = 橙   grade 0 = 红

产物：
    results/labels/round{1..5}_graded.png   每轮一图，行=臂，格=脸，框=等级
    results/twostage/pipeline.png           灰度生成器 + 颜色头 的结构与数据流
    results/twostage/selection_ba.png       筛选前 / 筛选后（PCA4 灰度 + 颜色头）
    results/labels/blindtest_summary.json   每轮每臂的结构、成本、良率（给文档用）
用法：python blindtest_report.py [--per 20]
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
P2 = os.path.join(REPO, "02_phase2_avatar")
LDIR = os.path.join(P4, "results", "labels")
OUT2 = os.path.join(P4, "results", "twostage")
sys.path.insert(0, HERE)
from make_manifest import colours_of  # noqa: E402

GRADE_COL = {4: (0, 214, 214), 3: (64, 196, 96), 2: (232, 200, 64),
             1: (240, 148, 48), 0: (228, 64, 64)}
GRADE_NAME = {4: "4 很好", 3: "3 好（会展示）", 2: "2 能看但不值", 1: "1 差", 0: "0 废"}

K8 = "../results/train/K16_w128_k8/model_snap.json"
K16 = "../results/train/K16_w128_k16/model_snap.json"
K24 = "../results/train/K16_w128_k24/model_snap.json"
K32 = "../results/train/K16_w128_k32/model_snap.json"
SNAP = "../../02_phase2_avatar/webapp/colour_k8snap_model.json"
TWO = "../results/twostage/pal16.json"
TWO22 = "../results/twostage/pal16_p22.json"
GP4 = {"gray": "../results/twostage/gray_pca4.json"}
GP22 = {"gray": "../results/twostage/gray_pca22.json"}

ROUNDS = {
    "1": [("k8", K8, 8, None), ("k16", K16, 16, None), ("k24", K24, 24, None),
          ("k32", K32, 32, None), ("k8snap", SNAP, 8, None)],
    "2": [("k8", K8, 8, None), ("k8snap", SNAP, 8, None),
          ("k8_sel2", K8, 8, None), ("k8_sel4", K8, 8, None)],
    "3": [("two_pal16", TWO, 4, GP4), ("p22_pal16", TWO22, 22, GP22),
          ("k8", K8, 8, None), ("k8snap", SNAP, 8, None)],
    "4": [("two_pal16", TWO, 4, GP4), ("two_sel", TWO, 4, GP4),
          ("k8snap", SNAP, 8, None), ("snap_sel", SNAP, 8, None)],
    "5": [("two_pal16", TWO, 4, GP4), ("two_sel4", TWO, 4, GP4),
          ("k8snap", SNAP, 8, None), ("snap_sel4", SNAP, 8, None)],
}

ARM_META = {
    "k8": ("单阶段 16 色（k=8 颜色 PCA）", "8→128→128→32×16", 82944),
    "k16": ("单阶段 16 色（k=16）", "16→128→128→32×16", 83968),
    "k24": ("单阶段 16 色（k=24）", "24→128→128→32×16", 84992),
    "k32": ("单阶段 16 色（k=32）", "32→128→128→32×16", 86016),
    "k8snap": ("二期 8 色（MC 吸附）", "8→128→128→32×8", 50176),
    "k8_sel2": ("单阶段 16 色 + 打分器留 2（页面 bug，实为随机）", "8→128→128→32×16", 82944),
    "k8_sel4": ("单阶段 16 色 + 打分器留 4（页面 bug，实为随机）", "8→128→128→32×16", 82944),
    "two_pal16": ("两阶段：PCA4 灰度 → 16 色", "4→64→32 | 32→128→128→32×16", 88320),
    "p22_pal16": ("两阶段：PCA22 灰度 → 16 色", "22→128→128→32 | 32→128→128→32×16", 109312),
    "two_sel": ("两阶段 + 打分器留 3（页面 bug，实为随机）", "4→64→32 | 32→128→128→32×16", 88320),
    "snap_sel": ("二期 8 色 + 打分器留 3（页面 bug，实为随机）", "8→128→128→32×8", 50176),
    "two_sel4": ("两阶段 + 二次打分器留 4 选 1", "4→64→32 | 32→128→128→32×16", 88380),
    "snap_sel4": ("二期 8 色 + 线性打分器留 4 选 1", "8→128→128→32×8", 50212),
}


def font(sz, bold=False):
    for fn in ("msyhbd.ttc" if bold else "msyh.ttc", "simhei.ttf"):
        p = os.path.join("C:\\Windows\\Fonts", fn)
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, sz)
            except Exception:
                pass
    return ImageFont.load_default()


def load_round(r):
    p = os.path.join(LDIR, f"colour_round{r}.jsonl")
    if not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def faces_for_arm(rows, spec, k, pipe):
    """把某臂的标签按等级排好，重绘成 8×8 图。"""
    out = []
    for r in rows:
        if r["model"] != spec:
            continue
        z = (r.get("z") or r["pca"])[:k]
        arm = {"name": spec, "file": None, "k": k}
        if pipe:
            arm["pipe"] = pipe
        # colours_of 需要 file/pipe；用外部传入的完整 spec 更稳妥
        out.append((r["grade"], [float(v) for v in z]))
    return out


def render(cols):
    a = np.asarray(cols, dtype=np.float64).reshape(32, 3)
    img = np.zeros((8, 8, 3))
    rows = np.array([p % 8 for p in range(32)])
    cols_ = np.array([p // 8 for p in range(32)])
    img[rows, cols_] = a
    img[rows, 7 - cols_] = a
    return np.clip(np.rint(img * 17), 0, 255).astype(np.uint8)


def draw_round(r, per, summary):
    rows_lab = load_round(r)
    if not rows_lab:
        return
    specs = ROUNDS[r]
    TILE, BORD, PAD, LW = 52, 4, 8, 330
    nrow = len(specs)
    W = LW + per * (TILE + PAD) + 16
    H = 118 + nrow * (TILE + 46) + 30
    cv = Image.new("RGB", (W, H), (16, 18, 22))
    dr = ImageDraw.Draw(cv)
    dr.text((14, 12), f"第 {r} 轮盲测：各臂生成的随机头像（按等级描框）", font=font(20),
            fill=(234, 238, 244))
    dr.text((14, 40), f"{len(rows_lab)} 条标签；每臂按等级从高到低取前 {per} 张。"
                      "框色：青=4 · 绿=3 · 黄=2 · 橙=1 · 红=0。", font=font(11),
            fill=(148, 160, 176))
    # 图例
    x = 14
    for g in (4, 3, 2, 1, 0):
        dr.rectangle([x, 62, x + 16, 76], outline=GRADE_COL[g], width=3)
        dr.text((x + 22, 60), GRADE_NAME[g], font=font(10), fill=(200, 210, 226))
        x += 130

    y = 96
    for name, path, k, pipe in specs:
        lab = [x for x in rows_lab if x["model"] == name]
        # 重绘
        items = []
        arm = {"name": name, "file": path, "k": k}
        if pipe:
            arm["pipe"] = pipe
        for x in lab:
            z = (x.get("z") or x["pca"])[:k]
            cols = colours_of(arm, dict(arm, z=[float(v) for v in z]))
            items.append((x["grade"], cols))
        items.sort(key=lambda t: -t[0])
        grades = np.array([g for g, _ in items])
        good = float((grades >= 3).mean()) * 100 if len(grades) else 0
        acc = float((grades >= 2).mean()) * 100 if len(grades) else 0
        title, shape, mults = ARM_META.get(name, (name, "?", 0))
        dr.text((14, y + 4), name, font=font(13, True), fill=(112, 170, 254))
        dr.text((14, y + 22), title, font=font(10), fill=(190, 200, 214))
        dr.text((14, y + 36), f"{shape} · {mults:,} 乘法 · n={len(grades)}", font=font(9),
                fill=(140, 152, 168))
        dr.text((250, y + 22), f"好 {good:.0f}%", font=font(10), fill=(120, 200, 140))
        dr.text((250, y + 36), f"可接受 {acc:.0f}%", font=font(9), fill=(150, 162, 178))
        for i, (g, cols) in enumerate(items[:per]):
            x0 = LW + i * (TILE + PAD)
            t = render(cols)
            t = np.repeat(np.repeat(t, TILE // 8, 0), TILE // 8, 1)
            cv.paste(Image.fromarray(t), (x0 + BORD, y + BORD))
            dr.rectangle([x0, y, x0 + TILE + 2 * BORD - 1, y + TILE + 2 * BORD - 1],
                         outline=GRADE_COL[g], width=BORD)
        summary.setdefault(f"round{r}", {})[name] = {
            "n": int(len(grades)), "good_ge3": round(good / 100, 3),
            "accept_ge2": round(acc / 100, 3),
            "hist": {str(i): int((grades == i).sum()) for i in range(5)},
            "mean": round(float(grades.mean()), 3) if len(grades) else None,
            "shape": shape, "mults": mults, "desc": title}
        y += TILE + 46

    p = os.path.join(LDIR, f"round{r}_graded.png")
    cv.save(p)
    print(f"wrote {os.path.relpath(p, REPO)}  ({nrow} 臂 × {per} 张)")


def draw_pipeline():
    """灰度生成器 + 颜色头 的结构与数据流。"""
    W, H = 1180, 520
    cv = Image.new("RGB", (W, H), (16, 18, 22))
    dr = ImageDraw.Draw(cv)
    dr.text((16, 14), "两阶段的结构：灰度生成器 + 颜色头", font=font(20), fill=(234, 238, 244))
    dr.text((16, 42), "上排是二期终版 PCA4 灰度模型；下排是四期新训的 16 色颜色头。"
                      "颜色头的输入是**生成灰度**（训练时也用生成灰度，避免训练/部署错配）。",
            font=font(11), fill=(148, 160, 176))

    def box(x, y, w, h, title, sub, col=(52, 70, 100)):
        dr.rounded_rectangle([x, y, x + w, y + h], 8, fill=(24, 28, 38), outline=col, width=2)
        dr.text((x + 10, y + 8), title, font=font(12, True), fill=(226, 232, 242))
        for i, line in enumerate(sub):
            dr.text((x + 10, y + 28 + 15 * i), line, font=font(9), fill=(150, 162, 178))

    def arrow(x1, y1, x2, y2, label=""):
        dr.line([x1, y1, x2, y2], fill=(90, 110, 140), width=2)
        dr.polygon([(x2, y2), (x2 - 7, y2 - 4), (x2 - 7, y2 + 4)], fill=(90, 110, 140))
        if label:
            dr.text(((x1 + x2) // 2 - 20, (y1 + y2) // 2 - 14), label, font=font(9),
                    fill=(190, 200, 214))

    # 上排
    box(20, 80, 120, 74, "4 个 PCA 旋钮", ["N(0,1) 截断 ±3.5", "4 条随机源"], (70, 120, 180))
    box(180, 80, 200, 74, "灰度网络 4→64→32",
        ["Linear 4×64 = 256", "Linear 64×32 = 2,048", "共 2,304 乘法 · ReLU"], (70, 120, 180))
    box(420, 80, 180, 74, "后处理", ["clip 0..15 → 逐张 min-max", "→ 四舍五入到 16 级"], (70, 120, 180))
    box(640, 80, 200, 74, "32 个灰度级 /15",
        ["= 左半脸 32 位置", "行内镜像后成 8×8"], (70, 120, 180))
    arrow(140, 117, 180, 117)
    arrow(380, 117, 420, 117)
    arrow(600, 117, 640, 117)

    # 下排
    box(180, 250, 220, 74, "颜色头 32→128→128→32×16",
        ["Linear 32×128 = 4,096", "Linear 128×128 = 16,384", "Linear 128×512 = 65,536",
         "共 86,016 乘法 · ReLU"], (120, 90, 170))
    box(440, 250, 190, 74, "逐位置 argmax",
        ["32 个位置各取 16 类中最大", "→ 32 个调色板索引"], (120, 90, 170))
    box(670, 250, 210, 74, "调色板查表",
        ["16 个槽（吸附到 61 HIGH 后", "只有 15 个不同颜色）"], (120, 90, 170))
    box(920, 250, 200, 74, "8×8 彩色头像",
        ["每像素 4 bit 索引", "全脸 32×4 = 128 bit"], (120, 90, 170))
    arrow(860, 117, 180, 250, "32 个灰度级")
    arrow(400, 287, 440, 287)
    arrow(630, 287, 670, 287)
    arrow(880, 287, 920, 287)

    # 选择分支
    box(20, 380, 300, 84, "打分器（只在筛选版里）",
        ["4 个旋钮 → 15 项二次式", "score = w0 + Σw_i z_i + Σw_ij z_i z_j",
         "λ=20，拟合自 240 条标签，外样本 AUC 0.859"], (60, 130, 110))
    box(360, 380, 300, 84, "留 4 选 1",
        ["抽 4 组旋钮 → 各打分", "留分数最高的一组再生成", "代价是 4 倍采样时间，不是 4 倍电路"],
        (60, 130, 110))
    arrow(140, 154, 140, 380, "4 个旋钮")   # 旋钮 → 打分器（生成之前）
    arrow(320, 422, 360, 422)
    dr.text((700, 380), "总成本（筛选版）：2,304 + 86,016 + 4×15 = 88,380 乘法 · 4 条随机源",
            font=font(11), fill=(200, 210, 226))
    dr.text((700, 402), "好脸率 76% · 可接受率 99%（第五轮，同轮对照 37% / 90%）",
            font=font(11), fill=(120, 200, 140))
    p = os.path.join(OUT2, "pipeline.png")
    cv.save(p)
    print(f"wrote {os.path.relpath(p, REPO)}")


def draw_selection_ba(per=16):
    """筛选前 / 筛选后：PCA4 灰度 + 颜色头。"""
    rows_lab = load_round("5")
    arm_spec = {"two_pal16": ("抽 1 组（随机）", TWO, 4, GP4),
                "two_sel4": ("抽 4 组留最高", TWO, 4, GP4)}
    TILE, BORD, PAD, LW = 56, 4, 8, 300
    W = LW + per * (TILE + PAD) + 16
    H = 150 + 2 * (TILE + 52) + 20
    cv = Image.new("RGB", (W, H), (16, 18, 22))
    dr = ImageDraw.Draw(cv)
    dr.text((14, 12), "PCA4 灰度 + 颜色头：筛选前 vs 筛选后（第五轮，同一模型）",
            font=font(20), fill=(234, 238, 244))
    dr.text((14, 40), f"每行按等级从高到低取前 {per} 张；框色：青=4 · 绿=3 · 黄=2 · 橙=1 · 红=0。",
            font=font(11), fill=(148, 160, 176))
    x = 14
    for g in (4, 3, 2, 1, 0):
        dr.rectangle([x, 62, x + 16, 76], outline=GRADE_COL[g], width=3)
        dr.text((x + 22, 60), GRADE_NAME[g], font=font(10), fill=(200, 210, 226))
        x += 130

    y = 100
    for name, (label, path, k, pipe) in arm_spec.items():
        arm = {"name": name, "file": path, "k": k, "pipe": pipe}
        items = []
        for r in rows_lab:
            if r["model"] != name:
                continue
            z = (r.get("z") or r["pca"])[:k]
            items.append((r["grade"], colours_of(arm, dict(arm, z=[float(v) for v in z]))))
        items.sort(key=lambda t: -t[0])
        grades = np.array([g for g, _ in items])
        dr.text((14, y + 6), label, font=font(14, True), fill=(112, 170, 254))
        dr.text((14, y + 26), f"好 ≥3：{float((grades>=3).mean())*100:.0f}%   "
                              f"可接受 ≥2：{float((grades>=2).mean())*100:.0f}%   n={len(grades)}",
                font=font(10), fill=(150, 162, 178))
        dr.text((14, y + 42), "框色 = 人工等级", font=font(9), fill=(130, 142, 158))
        for i, (g, cols) in enumerate(items[:per]):
            x0 = LW + i * (TILE + PAD)
            t = np.repeat(np.repeat(render(cols), TILE // 8, 0), TILE // 8, 1)
            cv.paste(Image.fromarray(t), (x0 + BORD, y + BORD))
            dr.rectangle([x0, y, x0 + TILE + 2 * BORD - 1, y + TILE + 2 * BORD - 1],
                         outline=GRADE_COL[g], width=BORD)
        y += TILE + 52
    p = os.path.join(OUT2, "selection_ba.png")
    cv.save(p)
    print(f"wrote {os.path.relpath(p, REPO)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", type=int, default=20)
    A = ap.parse_args()
    summary = {}
    for r in ROUNDS:
        draw_round(r, A.per, summary)
    draw_pipeline()
    draw_selection_ba(16)
    json.dump(summary, open(os.path.join(LDIR, "blindtest_summary.json"), "w",
                            encoding="utf-8"), indent=2, ensure_ascii=False)
    print("wrote results/labels/blindtest_summary.json")
    print("BLINDTEST_REPORT_DONE")


if __name__ == "__main__":
    main()
