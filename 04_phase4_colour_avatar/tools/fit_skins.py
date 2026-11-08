#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
把开发团队的四张皮肤"拟合"成 4 旋钮参数。

思路（按作者的猜测逐条验证）：
    1. 皮肤的是 64×64 贴图，脸部在 [8:16, 8:16]，取出来就是 8×8；
    2. 我们的生成器**逐行镜像**，所以先把目标左右对称化（左半镜像），得到"可达目标"；
    3. 参数搜索在 32 个位置上比较，两种距离：
           dL = 平均亮度差（级，0..15）
           dC = 平均 RGB 差（级）
       各用三种权重搜索，看"亮度优先"是不是真的更好；
    4. 搜索：随机 8000 点 + 取前 24 名做模式搜索细化（输出是量化过的，梯度无意义）。

产物：
    results/skins_fit.png     每个皮肤：目标 / 亮度优先 / 均衡 / 色彩优先
    results/skins_fit.json    每个皮肤的最优参数与各项距离
用法：python fit_skins.py
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
DAT = os.path.join(P4, "dat")
OUT = os.path.join(P4, "results")
ROW = np.arange(32) % 8
COL = np.arange(32) // 8

HINT = {"HJKV_MC": [-3.05, 2.65, 1.50, -0.50]}
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


def load_model():
    g = json.load(open(os.path.join(P4, "results", "twostage", "gray_pca4.json"),
                       encoding="utf-8"))
    GL = [(np.array(L["W"]), np.array(L["b"])) for L in g["layers"]]
    p = json.load(open(os.path.join(P4, "results", "twostage", "pal16.json"),
                       encoding="utf-8"))
    LAY = ([(np.array(L["W"]), np.array(L["b"])) for L in p["body"]]
           + [(np.array(p["head"]["W"]), np.array(p["head"]["b"]))])
    return GL, LAY, np.array(p["palette"], float)


def fwd(layers, x):
    h = np.asarray(x, float)
    for i, (W, b) in enumerate(layers):
        h = h @ W.T + b
        if i < len(layers) - 1:
            h = np.maximum(h, 0)
    return h


def forward(Z, GL, LAY, PAL):
    """Z:(n,4) → 32 位置的 RGB（0..15 单位）"""
    raw = fwd(GL, Z)
    lo = raw.min(1, keepdims=True); hi = raw.max(1, keepdims=True)
    lv = np.clip(np.rint((raw - lo) / np.maximum(hi - lo, 1e-6) * 15),
                 0, 15) / 15.0
    lg = fwd(LAY, lv).reshape(len(Z), 32, len(PAL))
    idx = lg.argmax(-1)
    return PAL[idx]


def metrics(rgb, tgt):
    dL = np.abs(rgb @ LUM - tgt @ LUM).mean(1)
    dC = np.abs(rgb - tgt).mean((1, 2))
    return dL, dC


def search(tgt, wL, wC, GL, LAY, PAL, n_rand=20000, topk=32, seed=0):
    rng = np.random.default_rng(seed)
    Z = rng.uniform(-3.5, 3.5, size=(n_rand, 4))
    # 再加一圈边界点，避免漏掉极值区
    grid = np.array(np.meshgrid(*[[-3.5, 0.0, 3.5]] * 4)).reshape(4, -1).T
    Z = np.vstack([Z, grid])
    rgb = forward(Z, GL, LAY, PAL)
    dL, dC = metrics(rgb, tgt)
    loss = wL * dL + wC * dC
    order = np.argsort(loss)[:topk]

    rand_best = float(loss[order[0]])
    best = (loss[order[0]], Z[order[0]].copy())
    # 模式搜索：坐标方向、逐步缩小步长
    steps = [0.8, 0.4, 0.2, 0.1, 0.05, 0.02]
    for k in order:
        z = Z[k].copy()
        cur = loss[k]
        for st in steps:
            improved = True
            while improved:
                improved = False
                cand = []
                for j in range(4):
                    for s in (+st, -st):
                        zz = z.copy(); zz[j] = np.clip(zz[j] + s, -3.5, 3.5)
                        cand.append(zz)
                C = np.array(cand)
                r = forward(C, GL, LAY, PAL)
                dl, dc = metrics(r, tgt)
                L = wL * dl + wC * dc
                i = int(np.argmin(L))
                if L[i] < cur - 1e-9:
                    cur = float(L[i]); z = C[i]
                    improved = True
        if cur < best[0]:
            best = (cur, z)
    z = best[1]
    r = forward(z[None, :], GL, LAY, PAL)[0]
    dl, dc = metrics(r[None, :], tgt)
    return {"z": [round(float(v), 3) for v in z], "loss": round(float(best[0]), 4),
            "dL": round(float(dl[0]), 3), "dC": round(float(dc[0]), 3),
            "rand_best": round(rand_best, 4),
            "gain_from_polish": round(rand_best - float(best[0]), 4)}


def main():
    GL, LAY, PAL = load_model()
    skins = {}
    for f in sorted(os.listdir(DAT)):
        if not f.lower().endswith(".png"):
            continue
        a = np.array(Image.open(os.path.join(DAT, f)).convert("RGB")).astype(float)
        face = a[8:16, 8:16] / 17.0                      # 8×8，0..15 单位
        name = os.path.splitext(f)[0]
        half = "right" if name.upper().startswith("TNT46") else "left"
        symL = face.copy(); symL[:, 4:] = face[:, :4][:, ::-1]     # 左半镜像
        symR = face.copy(); symR[:, :4] = face[:, 4:][:, ::-1]     # 右半镜像
        sym = symR if half == "right" else symL
        t32 = np.stack([sym[ROW, COL]], 0)
        skins[name] = {"face": face, "t32": t32[0], "half": half}
        print(f"{f}: 用{('右' if half=='right' else '左')}半镜像；与原始平均差 "
              f"{np.abs(sym - face).mean():.2f} 级")

    schemes = [("L_only", 1.0, 0.0), ("balanced", 1.0, 1.0), ("C_heavy", 0.3, 1.0)]
    results = {}
    for name, d in skins.items():
        results[name] = {}
        print(f"\n=== {name} ===")
        for tag, wL, wC in schemes:
            r = search(d["t32"], wL, wC, GL, LAY, PAL)
            results[name][tag] = r
            print(f"  {tag:9s} z={r['z']}  dL={r['dL']:.2f}  dC={r['dC']:.2f}")
        if name in HINT:
            z = np.array(HINT[name], float)
            rgb = forward(z[None, :], GL, LAY, PAL)[0]
            dl, dc = metrics(rgb[None, :], d["t32"])
            results[name]["hint"] = {"z": HINT[name], "dL": round(float(dl[0]), 3),
                                     "dC": round(float(dc[0]), 3)}
            print(f"  作者给的  z={HINT[name]}  dL={dl[0]:.2f}  dC={dc[0]:.2f}")

    # ---- 调色板下限：逐位置量化目标到最近的调色板颜色
    for name, d in skins.items():
        t = d["t32"]
        dmat = ((t[:, None, :] - PAL[None, :, :]) ** 2).sum(-1)
        q = PAL[dmat.argmin(1)]
        dl = float(np.abs(q @ LUM - t @ LUM).mean())
        dc = float(np.abs(q - t).mean())
        results[name]["palette_floor"] = {"dL": round(dl, 3), "dC": round(dc, 3)}
        print(f"{name} 调色板下限：dL {dl:.2f}  dC {dc:.2f}")

    json.dump(results, open(os.path.join(OUT, "skins_fit.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)

    # ---- 出图
    cols = [("目标（对称化）", None), ("调色板下限", "floor"), ("亮度优先", "L_only"),
            ("均衡", "balanced"), ("色彩优先", "C_heavy")]
    T, PADG, LW = 84, 8, 150
    W = LW + len(cols) * (T + PADG) + 16
    H = 92 + len(skins) * (T + 62)
    cv = Image.new("RGB", (W, H), (16, 18, 22))
    dr = ImageDraw.Draw(cv)
    dr.text((14, 12), "四张皮肤 → 最接近的 4 旋钮参数", font=font(19), fill=(234, 238, 244))
    dr.text((14, 40), "皮肤取 64×64 贴图的脸部 [8:16,8:16]；因为生成器逐行镜像，"
                      "目标先左半镜像（第一列）。dL/RGB 距离都在 32 个位置上算。",
            font=font(11), fill=(148, 160, 176))
    for j, (label, _) in enumerate(cols):
        dr.text((LW + j * (T + PADG), 68), label, font=font(12), fill=(112, 170, 254))
    y = 88
    for name, d in skins.items():
        dr.text((14, y + 10), name, font=font(13), fill=(226, 232, 242))
        dr.text((14, y + 28), ("右半镜像" if d.get("half") == "right" else "左半镜像"),
                font=font(9), fill=(150, 162, 178))
        for j, (label, tag) in enumerate(cols):
            x = LW + j * (T + PADG)
            if tag is None:
                img = d["t32"].reshape(32, 3)
            elif tag == "floor":
                t = d["t32"]
                dmat = ((t[:, None, :] - PAL[None, :, :]) ** 2).sum(-1)
                img = PAL[dmat.argmin(1)]
            else:
                r = results[name][tag]
                img = forward(np.array([r["z"]]), GL, LAY, PAL)[0]
            full = np.zeros((8, 8, 3))
            full[ROW, COL] = img
            full[ROW, 7 - COL] = img
            t = np.repeat(np.repeat(
                np.clip(np.rint(full / 15.0 * 255), 0, 255).astype(np.uint8),
                T // 8, 0), T // 8, 1)
            cv.paste(Image.fromarray(t), (x, y))
            if tag == "floor":
                f = results[name]["palette_floor"]
                dr.text((x, y + T + 1), f"dL {f['dL']:.2f}", font=font(9), fill=(120, 200, 140))
                dr.text((x, y + T + 13), f"dC {f['dC']:.2f}", font=font(8), fill=(120, 200, 140))
            elif tag is not None:
                r = results[name][tag]
                dr.text((x, y + T + 1), f"dL {r['dL']:.2f}", font=font(9), fill=(190, 200, 214))
                dr.text((x, y + T + 13),
                        ", ".join(f"{v:+.2f}" for v in r["z"]), font=font(8),
                        fill=(150, 162, 178))
        if "hint" in results[name]:
            h = results[name]["hint"]
            dr.text((14, y + T + 20), f"作者参考：dL {h['dL']:.2f} / dC {h['dC']:.2f}",
                    font=font(9), fill=(200, 170, 120))
        y += T + 62
    p = os.path.join(OUT, "skins_fit.png")
    cv.save(p)
    print(f"\nwrote {os.path.relpath(p, REPO)}")
    print(f"wrote {os.path.relpath(os.path.join(OUT,'skins_fit.json'), REPO)}")


if __name__ == "__main__":
    main()
