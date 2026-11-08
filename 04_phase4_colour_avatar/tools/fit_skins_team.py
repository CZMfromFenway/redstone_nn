#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
第二步：给"这四个人的皮肤"专门配一套 16 色调色板，再看拟合能好多少。
（红石成本完全不变：还是 16 槽查表，只是表的内容换成团队专用。）

做法：
    1. 取四张脸（对称化后）的 32×4 = 128 个颜色；
    2. 全部吸附到 244 个地图色；
    3. 在这 128 个吸附色上跑 k-means（k=14），再把质心吸附一次，补上 BLACK / QUARTZ；
    4. 用这套「团队调色板」重新拟合同四个皮肤，与原色板的拟合对比。

产物：results/skins_team_fit.png、results/skins_team_fit.json
用法：python fit_skins_team.py
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
DAT = os.path.join(P4, "dat")
OUT = os.path.join(P4, "results")
ROW = np.arange(32) % 8
COL = np.arange(32) // 8
LUM = np.array([0.299, 0.587, 0.114])

sys.path.insert(0, HERE)
from fit_skins import font, load_model, forward, metrics, search  # noqa: E402


def build_team_palette(t32s, K=16, seed=0):
    mc = json.load(open(os.path.join(P2, "data", "mc_map_colors.json"), encoding="utf-8"))
    base = np.array([b["rgb"] for b in mc["bases"]], float)
    cand = []
    for m in ("HIGH", "NORMAL", "LOW", "LOWEST"):
        cc = np.clip(base * mc["_modifiers"][m], 0, 255) / 17.0
        cand.extend([c for c in cc])
    cand = np.array(cand)
    # 128 个位置颜色 → 吸附到最近的地图色
    pts = np.vstack([t for t in t32s])
    d = ((pts[:, None, :] - cand[None, :, :]) ** 2).sum(-1)
    snapped = cand[d.argmin(1)]
    # k-means（在吸附色集合上）
    rng = np.random.default_rng(seed)
    C = snapped[rng.choice(len(snapped), K, replace=False)]
    for _ in range(60):
        d2 = ((snapped[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        lab = d2.argmin(1)
        for k in range(K):
            if (lab == k).any():
                C[k] = snapped[lab == k].mean(0)
    # 质心再吸附一次
    d3 = ((C[:, None, :] - cand[None, :, :]) ** 2).sum(-1)
    C = cand[d3.argmin(1)]
    # 去重后补到 K 个
    uniq = []
    for c in C:
        if not any(np.allclose(c, u, atol=0.6) for u in uniq):
            uniq.append(c)
    for c in (np.array([1.47, 1.47, 1.47]), np.array([15.0, 14.82, 14.41])):
        if len(uniq) < K and not any(np.allclose(c, u, atol=0.6) for u in uniq):
            uniq.append(c)
    while len(uniq) < K:
        uniq.append(uniq[len(uniq) % max(1, len(uniq))].copy())
    return np.array(uniq[:K]), cand


def main():
    GL, LAY, PAL0 = load_model()
    skins = {}
    for f in sorted(os.listdir(DAT)):
        if not f.lower().endswith(".png"):
            continue
        a = np.array(Image.open(os.path.join(DAT, f)).convert("RGB")).astype(float)
        face = a[8:16, 8:16] / 17.0
        name = os.path.splitext(f)[0]
        half = "right" if name.upper().startswith("TNT46") else "left"
        symL = face.copy(); symL[:, 4:] = face[:, :4][:, ::-1]
        symR = face.copy(); symR[:, :4] = face[:, 4:][:, ::-1]
        sym = symR if half == "right" else symL
        skins[name] = np.stack([sym[ROW, COL]], 0)[0]

    TEAM, cand = build_team_palette(list(skins.values()))
    print("团队调色板（16 槽）：")
    for i, c in enumerate(TEAM):
        print(f"  {i:2d} {np.round(c, 2).tolist()}")

    res = {}
    for name, t32 in skins.items():
        r0 = search(t32, 1.0, 1.0, GL, LAY, PAL0, n_rand=12000, topk=20)
        r1 = search(t32, 1.0, 1.0, GL, LAY, TEAM, n_rand=12000, topk=20)
        d0 = ((t32[:, None, :] - PAL0[None, :, :]) ** 2).sum(-1)
        d1 = ((t32[:, None, :] - TEAM[None, :, :]) ** 2).sum(-1)
        f0 = PAL0[d0.argmin(1)]; f1 = TEAM[d1.argmin(1)]
        res[name] = {"orig": r0, "team": r1,
                     "floor_orig": {"dL": round(float(np.abs(f0 @ LUM - t32 @ LUM).mean()), 3),
                                    "dC": round(float(np.abs(f0 - t32).mean()), 3)},
                     "floor_team": {"dL": round(float(np.abs(f1 @ LUM - t32 @ LUM).mean()), 3),
                                    "dC": round(float(np.abs(f1 - t32).mean()), 3)}}
        print(f"\n{name}")
        print(f"  原色板   z={r0['z']}  dL {r0['dL']:.2f}  dC {r0['dC']:.2f}   下限 dC {res[name]['floor_orig']['dC']:.2f}")
        print(f"  团队色板 z={r1['z']}  dL {r1['dL']:.2f}  dC {r1['dC']:.2f}   下限 dC {res[name]['floor_team']['dC']:.2f}")
    json.dump({"palette_team": TEAM.tolist(), "skins": res},
              open(os.path.join(OUT, "skins_team_fit.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)

    # ---- 出图
    cols = [("目标", None), ("原色板", "orig"), ("团队色板", "team")]
    T, PADG, LW = 92, 10, 150
    W = LW + len(cols) * (T + PADG) + 16
    H = 104 + len(skins) * (T + 58)
    cv = Image.new("RGB", (W, H), (16, 18, 22))
    dr = ImageDraw.Draw(cv)
    dr.text((14, 12), "团队专用调色板：同样 16 槽，拟合能好多少", font=font(19),
            fill=(234, 238, 244))
    dr.text((14, 40), "左：目标（对称化）；中：用现有 16 色拟合；右：用为这四个人定制的 16 色拟合。"
                      "调色板由四张脸的 128 个颜色做 k-means 得到，全部取自地图色。",
            font=font(11), fill=(148, 160, 176))
    for j, (label, _) in enumerate(cols):
        dr.text((LW + j * (T + PADG), 70), label, font=font(12), fill=(112, 170, 254))
    y = 92
    for name, t32 in skins.items():
        dr.text((14, y + 12), name, font=font(13), fill=(226, 232, 242))
        r = res[name]
        dr.text((14, y + 32), f"dC {r['orig']['dC']:.2f} → {r['team']['dC']:.2f}",
                font=font(10), fill=(120, 200, 140))
        dr.text((14, y + 46), f"dL {r['orig']['dL']:.2f} → {r['team']['dL']:.2f}",
                font=font(9), fill=(150, 162, 178))
        for j, (label, tag) in enumerate(cols):
            x = LW + j * (T + PADG)
            if tag is None:
                img = t32
            else:
                pal = PAL0 if tag == "orig" else TEAM
                img = forward(np.array([r[tag]["z"]]), GL, LAY, pal)[0]
            full = np.zeros((8, 8, 3)); full[ROW, COL] = img; full[ROW, 7 - COL] = img
            t = np.repeat(np.repeat(
                np.clip(np.rint(full / 15.0 * 255), 0, 255).astype(np.uint8),
                T // 8, 0), T // 8, 1)
            cv.paste(Image.fromarray(t), (x, y))
            if tag is not None:
                dr.text((x, y + T + 1), ", ".join(f"{v:+.2f}" for v in r[tag]["z"]),
                        font=font(8), fill=(150, 162, 178))
        y += T + 58
    p = os.path.join(OUT, "skins_team_fit.png")
    cv.save(p)
    print(f"\nwrote {os.path.relpath(p, REPO)}")


if __name__ == "__main__":
    main()
