#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
用训练集里的真实头像测「拟合能力」：把每个臂喂进这张脸自己的 PCA 坐标，看它画回多像。

回答的问题：这些模型是不是只会画"平均脸"？给它真实坐标它能不能复现？
（这是二期没做过的一项：二期只比过随机采样的观感，没比过"给定真值坐标的重建"。）

另外做两件事：
  1. 把每个臂的调色板**吸附到 61 个 HIGH 地图色**（按作者 2026-11-02 的决定：撞色就撞色），
     写 `model_snap.json` 与 `deploy_palette.json`；
  2. 给出"调色板上限"（真实脸直接量化到该调色板）作为一切重建误差的地板。

产物：results/fit/fit.json、results/fit/fit.png
用法：python eval_fit.py
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
OUT = os.path.join(P4, "results", "fit")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.join(P4, "src"))

ARMS = ["K16_w128_k8", "K16_w128_k16", "K16_w128_k24", "K16_w128_k32"]
N_FACES = 16
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


def load_high():
    j = json.load(open(os.path.join(P2, "data", "mc_map_colors.json"), encoding="utf-8"))
    base = np.array([b["rgb"] for b in j["bases"]], dtype=np.float64)
    names = [b["name"] for b in j["bases"]]
    high = np.clip(base * j["_modifiers"]["HIGH"], 0, 255) / 17.0
    return high, names


def snap_to(pal, cands):
    d = ((pal[:, None, :] - cands[None]) ** 2).sum(-1)
    idx = d.argmin(1)
    return cands[idx], idx, np.sqrt(d[np.arange(len(pal)), idx])


def mirror(v):
    v = np.asarray(v)
    rows = np.array([p % 8 for p in range(32)])
    cols = np.array([p // 8 for p in range(32)])
    if v.ndim == 2:
        img = np.zeros((len(v), 8, 8))
        img[:, rows, cols] = v
        img[:, rows, 7 - cols] = v
        return img
    img = np.zeros((len(v), 8, 8, v.shape[-1]))
    img[:, rows, cols] = v
    img[:, rows, 7 - cols] = v
    return img


class Net:
    def __init__(self, path):
        j = json.load(open(path, encoding="utf-8"))
        self.j = j
        self.pal = np.array(j["palette"], dtype=np.float64)
        self.layers = [(np.array(L["W"]), np.array(L["b"]), True) for L in j["body"]]
        self.layers.append((np.array(j["head"]["W"]), np.array(j["head"]["b"]), False))
        self.K = len(self.pal)
        self.meta = j["meta"]
        self.k = self.meta["k"]

    def logits(self, x):
        h = x
        for W, b, relu in self.layers:
            h = h @ W.T + b
            if relu:
                h = np.maximum(h, 0)
        return h.reshape(len(x), 32, self.K)

    def face(self, x, pal=None):
        pal = self.pal if pal is None else pal
        idx = self.logits(x).argmax(-1)
        return mirror(pal[idx]), idx


def main():
    import train_palette as tp          # 复用同一套数据与基，保证坐标口径一致
    rgb96, cpc, tr, va = tp.load_cache(32, "ext32")
    POS = rgb96.reshape(len(rgb96), 32, 3)
    HIGH, HIGH_NAMES = load_high()

    # 训练集取样：均匀间隔，避免"挑好看的"
    step = len(tr) // (N_FACES + 1)
    rows_tr = tr[::step][:N_FACES]
    rows_va = va[::(len(va) // (N_FACES + 1))][:N_FACES]
    real_tr = POS[rows_tr]
    real_va = POS[rows_va]

    print(f"palette candidates: {len(HIGH)} HIGH colours")
    results, snaps = {}, {}
    grids = {"train": [("真实（训练集）", mirror(real_tr))],
             "val": [("真实（验证集）", mirror(real_va))]}
    for tag in ARMS:
        model_path = os.path.join(P4, "results", "train", tag, "model.json")
        net = Net(model_path)
        k = net.k
        pal_snap, snap_idx, snap_err = snap_to(net.pal, HIGH)
        distinct = len({tuple(np.round(c, 3)) for c in pal_snap})
        # 部署件：同权重 + 吸附后的调色板（二期 k8snap 的做法）
        dep = dict(net.j)
        dep["palette"] = np.round(pal_snap, 5).tolist()
        dep["meta"] = dict(net.meta, palette_scheme="mc_high_snapped",
                           palette_names=[HIGH_NAMES[i] for i in snap_idx],
                           snap_err_mean=round(float(snap_err.mean()), 3),
                           distinct_colours=distinct)
        json.dump(dep, open(os.path.join(P4, "results", "train", tag, "model_snap.json"),
                            "w", encoding="utf-8"), ensure_ascii=False)
        snaps[tag] = {"palette": np.round(pal_snap, 5).tolist(),
                      "names": [HIGH_NAMES[i] for i in snap_idx],
                      "snap_err_mean": round(float(snap_err.mean()), 3),
                      "snap_err_max": round(float(snap_err.max()), 3),
                      "distinct_colours": distinct, "K": net.K, "k": k}

        for split, coords, real in (("train", cpc[rows_tr][:, :k], real_tr),
                                    ("val", cpc[rows_va][:, :k], real_va)):
            img, idx = net.face(coords, pal_snap)
            floor_img = mirror(pal_snap[
                ((real[:, :, None, :] - pal_snap[None, None, :, :]) ** 2).sum(-1).argmin(-1)])
            # 指标用 2,000 行算（16 张脸只有 512 个像素，抽样噪声约 ±4 点，会误导）
            mrows = tr[:2000] if split == "train" else va[:2000]
            mco = cpc[mrows][:, :k]
            mreal = POS[mrows]
            mimg, midx = net.face(mco, pal_snap)
            d = ((mreal[:, :, None, :] - pal_snap[None, None, :, :]) ** 2).sum(-1)
            exact = float((midx == d.argmin(-1)).mean())
            idx_free = net.logits(mco).argmax(-1)
            d2 = ((mreal[:, :, None, :] - net.pal[None, None, :, :]) ** 2).sum(-1)
            exact_free = float((idx_free == d2.argmin(-1)).mean())
            mae = float(np.abs(mimg - mirror(mreal)).mean())
            floor_mae = float(np.abs(mirror(pal_snap[d.argmin(-1)]) - mirror(mreal)).mean())
            key = f"{tag}/{split}"
            results[key] = {"exact_snap": round(exact, 4), "exact_free_palette": round(exact_free, 4),
                            "rgb_mae": round(mae, 3), "floor_mae": round(floor_mae, 3),
                            "gap_over_floor": round(mae - floor_mae, 3)}
            if split in grids:
                grids[split].append((f"{tag}", img))
                if tag == ARMS[0]:
                    grids[split].insert(1, ("调色板上限（真值→最近色）", floor_img))
            print(f"  {key:26s} 命中(部署色/训练色) {exact*100:5.1f}% / {exact_free*100:5.1f}%"
                  f"  RGB MAE {mae:5.2f} 级（地板 {floor_mae:.2f}）  [n=2000]")

    # 同一个 z 在不同臂上一致（截图用的对照图已有）；这里出训练集/验证集重建图
    COLS, TILE, PAD, LW = N_FACES, 52, 4, 300
    W = LW + COLS * (TILE + PAD) + 20
    for split in ("train", "val"):
        rows = grids[split]
        H = 92 + len(rows) * (TILE + 30) + 20
        cv = Image.new("RGB", (W, H), (16, 18, 22))
        dr = ImageDraw.Draw(cv)
        dr.text((14, 12), f"给定真实坐标的重建（{split}）—— 每列是同一张真实头像", font=font(19),
                fill=(234, 238, 244))
        dr.text((14, 40), "最上一行是真实头像，第二行是【调色板上限】：把真值逐像素吸到 61 个 HIGH 地图色，"
                          "任何模型都不可能比它更像。", font=font(11), fill=(148, 160, 176))
        dr.text((14, 58), "其余行是把该脸自己的 k 个 PCA 坐标喂给模型、argmax 解码的结果。"
                          "行尾是位置命中率与 RGB 平均误差。", font=font(11), fill=(148, 160, 176))
        y = 84
        for label, imgs in rows:
            m = results.get(f"{label}/{split}") if not label.startswith("调色板") else None
            note = ""
            if m:
                note = f"命中(训练色) {m['exact_free_palette']*100:.0f}% · MAE {m['rgb_mae']:.2f}"
            elif label.startswith("调色板"):
                note = "= 地板"
            dr.text((14, y + 6), label, font=font(12), fill=(112, 170, 254))
            dr.text((14, y + 22), note, font=font(10), fill=(150, 162, 178))
            for i in range(COLS):
                v = np.clip(np.rint(imgs[i] * 17), 0, 255).astype(np.uint8)
                t = np.repeat(np.repeat(v, TILE // 8, 0), TILE // 8, 1)
                cv.paste(Image.fromarray(t), (LW + i * (TILE + PAD), y))
            y += TILE + 30
        p = os.path.join(OUT, f"fit_{split}.png")
        cv.save(p)
        print(f"wrote {os.path.relpath(p, REPO)}")

    json.dump({"arms": ARMS, "snaps": snaps, "results": results,
               "n_faces": N_FACES,
               "note": "exact_pos 与 rgb_mae 都在吸附到 61 个 HIGH 色的调色板下计算；"
                       "floor_mae 是同一张真值在该调色板下的量化误差，是重建误差的地板。"},
              open(os.path.join(OUT, "fit.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    json.dump({t: s for t, s in snaps.items()},
              open(os.path.join(OUT, "deploy_palette.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print("FIT_DONE")


if __name__ == "__main__":
    main()
