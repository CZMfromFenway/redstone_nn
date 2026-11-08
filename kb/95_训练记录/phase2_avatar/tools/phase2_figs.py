#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
二期里程碑模型：效果图 + 数字复验。只用 numpy + Pillow，不依赖 torch。

用法：
    python phase2_figs.py figs     # 生成 kb/figures/phase2/*.png
    python phase2_figs.py verify   # 复算数字，写 verify_phase2.json
    python phase2_figs.py both

模型一律从 02_phase2_avatar/webapp/*.json 读，因为那是部署页真正加载的权重；
不读 .pt，避免 checkpoint 与导出不一致。

布局常量与 02_phase2_avatar/src/layout.py 逐字一致：
    32 个输出按列优先覆盖左半脸：p = col*8 + row -> [row = p%8, col = p//8]
    重建时行内镜像：face[row, col] = face[row, 7-col]
layout.py 本身 import torch，所以这里重写一份，并在 verify 里断言两边一致。
"""
import json
import os
import sys
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
P2 = os.path.dirname(HERE)                       # kb/95_训练记录/phase2_avatar
KB95 = os.path.dirname(P2)                       # kb/95_训练记录
REPO = os.path.dirname(os.path.dirname(KB95))    # 仓库根
# 权重与数据源目录（不是本页的资产，本页只读它）
P2 = os.path.join(REPO, "02_phase2_avatar")
WEB = os.path.join(P2, "webapp")
FIG = os.path.join(REPO, "kb", "figures", "phase2")
os.makedirs(FIG, exist_ok=True)

POS_ROW = np.array([p % 8 for p in range(32)])
POS_COL = np.array([p // 8 for p in range(32)])
MIRROR_COL = 7 - POS_COL

# ------------------------------------------------------------------ 模型与解码


def load_gen(fn):
    """-> (layers, meta, in_dim)  layers = [(W(no,ni), b(no), relu_flag)]"""
    j = json.load(open(os.path.join(WEB, fn), encoding="utf-8"))
    if "gen" in j:
        g = j["gen"]
        layers = []
        if "layers" in g:                    # 新导出格式（任意层数）
            Ls = g["layers"]
            for i, L in enumerate(Ls):
                layers.append((np.array(L["W"], dtype=np.float64),
                               np.array(L["b"], dtype=np.float64),
                               i < len(Ls) - 1))
        else:                                # 早期导出格式：固定 W1..W3
            n = len([k for k in g if k.startswith("W")])
            for i in range(1, n + 1):
                layers.append((np.array(g[f"W{i}"], dtype=np.float64),
                               np.array(g[f"b{i}"], dtype=np.float64),
                               i < n))
        return layers, j.get("meta", {}), layers[0][0].shape[1]
    # palette / rgb 模型：body + head
    layers = []
    for L in j["body"]:
        layers.append((np.array(L["W"], dtype=np.float64),
                       np.array(L["b"], dtype=np.float64), True))
    layers.append((np.array(j["head"]["W"], dtype=np.float64),
                   np.array(j["head"]["b"], dtype=np.float64), False))
    return layers, j.get("meta", {}), layers[0][0].shape[1]


def forward(layers, X, z=None):
    """z 非空时拼到第一层输入后面（sweep_pca22 是 22 PCA + 64 噪声）。"""
    h = np.asarray(X, dtype=np.float64)
    for li, (W, b, relu) in enumerate(layers):
        if li == 0 and z is not None:
            h = np.concatenate([h, z], axis=1)
        s = h @ W.T + b
        h = np.maximum(s, 0.0) if relu else s
    return h


_GRAY = {}


def gray_basis():
    """按 dim_sweep.py 的配方，从 data_v2_canonical 重建灰度 PCA 坐标。

    返回 (ZZ, RTRUE, va, Xc, sub)：
        ZZ    (N,22) 已标准化的 PCA 坐标（z 分数），前 k 列就是 k 个旋钮
        RTRUE (N,32) 目标灰度级（左半脸规范序，0..15）
        va    验证集行号
        Xc    去均值后的 64 维图像矩阵（V3 用）
        sub   算 SVD 用的 60,000 行子样
    """
    if _GRAY:
        return _GRAY["ZZ"], _GRAY["RTRUE"], _GRAY["va"], _GRAY["Xc"], _GRAY["sub"]
    full = np.load(os.path.join(P2, "data_v2_canonical", "L0_full_idx.npy")).astype(np.float64)
    X = full.reshape(len(full), -1)
    mu = X.mean(0, keepdims=True)
    sub = np.random.default_rng(0).choice(len(X), 60000, replace=False)
    _, _, Vt = np.linalg.svd(X[sub] - mu, full_matrices=False)
    comps = Vt[:22]
    sd = ((X - mu) @ comps.T).std(0, keepdims=True) + 1e-6
    Xc = X - mu
    ZZ = ((Xc @ comps.T) / sd[0]).astype(np.float32)
    RTRUE = full[:, POS_ROW, MIRROR_COL].reshape(-1, 32)
    va = np.load(os.path.join(P2, "data_v2_canonical", "L0_split.npz"))["val"]
    _GRAY.update(ZZ=ZZ, RTRUE=RTRUE, va=va, Xc=Xc, sub=sub, full=full)
    return ZZ, RTRUE, va, Xc, sub


def mirror32(v):
    """(B,32) 或 (B,32,3) -> (B,8,8) / (B,8,8,3)"""
    v = np.asarray(v)
    if v.ndim == 2:
        img = np.zeros((len(v), 8, 8))
        img[:, POS_ROW, POS_COL] = v
        img[:, POS_ROW, MIRROR_COL] = v
        return img
    img = np.zeros((len(v), 8, 8, v.shape[-1]))
    img[:, POS_ROW, POS_COL] = v
    img[:, POS_ROW, MIRROR_COL] = v
    return img


def stretch15(v):
    lo = v.min(1, keepdims=True)
    hi = v.max(1, keepdims=True)
    return np.clip((v - lo) / np.maximum(hi - lo, 1e-6) * 15.0, 0, 15)


def quant16(v):
    return np.clip(np.rint(np.clip(v, 0, 15)), 0, 15)


def softmax_decode(raw, ncls, mode="mode", temp=1.0, rng=None):
    """512 logits -> (B,32) 类别索引。mode=取众数, sample=按分布抽。"""
    B = raw.shape[0]
    lg = raw.reshape(B, 32, ncls) / temp
    lg = lg - lg.max(-1, keepdims=True)
    p = np.exp(lg)
    p /= p.sum(-1, keepdims=True)
    if mode == "mode":
        return p.argmax(-1).astype(np.float64)
    cdf = np.cumsum(p, -1)
    u = rng.random((B, 32, 1))
    return (cdf < u).sum(-1).clip(0, ncls - 1).astype(np.float64)


def argmax_palette(raw, K):
    return raw.reshape(len(raw), 32, K).argmax(-1)


def lum(rgb):
    return rgb @ np.array([0.299, 0.587, 0.114])


# ------------------------------------------------------------------ 画图


def font(sz):
    for fn in ("msyh.ttc", "msyhbd.ttc", "simhei.ttf"):
        p = os.path.join("C:\\Windows\\Fonts", fn)
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, sz)
            except Exception:
                pass
    return ImageFont.load_default()


def sheet(path, title, subs, rows, cols=16, tile=56, pad=4, labelw=300):
    """rows = [(标签, 注释, imgs[(16,n,n)或(16,n,n,3)]), ...]
    每张脸是 0..15 的量纲，n 为 8；灰度与彩色都支持。"""
    left = labelw
    blockw = cols * (tile + pad)
    titleh = 34 + 22 * len(subs)
    rowh = tile + 30
    W = left + blockw + 20
    H = titleh + 16 + len(rows) * (rowh + 14) + 30
    cv = Image.new("RGB", (W, H), (16, 18, 22))
    dr = ImageDraw.Draw(cv)
    dr.text((14, 12), title, font=font(21), fill=(234, 238, 244))
    for i, s in enumerate(subs):
        dr.text((14, 40 + 22 * i), s, font=font(12), fill=(148, 160, 176))
    y = titleh + 12
    for label, note, imgs in rows:
        dr.text((14, y + 4), label, font=font(13), fill=(112, 170, 254))
        yy = y + 24
        for part in note.split("\n"):
            dr.text((14, yy), part, font=font(10), fill=(140, 152, 168))
            yy += 14
        up = tile // imgs.shape[1]
        for k in range(min(cols, len(imgs))):
            im = imgs[k]
            if im.ndim == 2:
                g = np.clip(im / 15.0 * 255, 0, 255).astype(np.uint8)
                t = np.repeat(np.repeat(g, up, 0), up, 1)
                t = np.stack([t] * 3, -1)
            else:
                c = np.clip(im / 15.0 * 255, 0, 255).astype(np.uint8)
                t = np.repeat(np.repeat(c, up, 0), up, 1)
            cv.paste(Image.fromarray(t), (left + k * (tile + pad), y))
        y += rowh + 14
    cv.save(path)
    print(f"  wrote {os.path.relpath(path, REPO)}  {cv.size}")


def draw_note(m):
    return (f"饱和 {m['sat']:.2f} · 色数 {m['ncol']:.1f} · 边缘 {m['edge']:.1f} "
            f"· 明度 {m['lum']:.2f}")


def metrics(imgs):
    """imgs: (B,8,8,3) 或 (B,8,8) -> 三个不依赖人眼的统计量"""
    a = np.asarray(imgs, dtype=np.float64)
    if a.ndim == 3:
        a = np.stack([a] * 3, -1)
    flat = a.reshape(len(a), 64, 3)
    mx, mn = flat.max(-1), flat.min(-1)
    sat = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0).mean()
    ncol = np.mean([len({tuple(np.rint(c).astype(int)) for c in flat[i]})
                    for i in range(len(flat))])
    dh = np.linalg.norm(a[:, :, 1:] - a[:, :, :-1], axis=-1)
    dv = np.linalg.norm(a[:, 1:] - a[:, :-1], axis=-1)
    ed = np.concatenate([dh.reshape(len(a), -1), dv.reshape(len(a), -1)], 1)
    return {"sat": float(sat), "ncol": float(ncol),
            "edge": float(np.percentile(ed, 90)), "lum": float(lum(flat).mean())}


# ------------------------------------------------------------------ 图 1-2：条件 GAN


def fig01():
    rng = np.random.default_rng(2024)
    layers, meta, _ = load_gen("pca22_model.json")
    ZZ, RTRUE, va, _, _ = gray_basis()
    # 真实坐标：验证集里真实脸的 22 个 PCA（z 分数）。模型就是在这种坐标上训练的。
    Xr = ZZ[va[:16], :22].astype(np.float64)
    # 随机坐标：独立标准正态。分布上应当接近，但离开数据流形后模型更吃力。
    Xn = np.clip(rng.standard_normal((16, 22)), -3.5, 3.5)
    zr = np.clip(rng.standard_normal((16, 64)), -3, 3)
    r0 = forward(layers, Xr, z=np.zeros((16, 64)))
    r1 = forward(layers, Xr, z=zr)
    r2 = forward(layers, Xn, z=zr)
    rows = [
        ("真实坐标 · z ≡ 0", "验证集真实脸的 22 个 PCA，噪声置零\n"
         "取众数解码 · 1,500 步 · 输入 86 维 · hid 256",
         mirror32(softmax_decode(r0, 16, "mode"))),
        ("同一批坐标 · z ~ N(0,1)", "只换噪声：PCA 决定 89% 的像素，\n噪声只改 8%，所以两行几乎一样",
         mirror32(softmax_decode(r1, 16, "mode"))),
        ("随机坐标 N(0,1) · z ~ N(0,1)", "离开数据流形后同一模型明显更麻\n"
         "这正是第 3 阶段要治的『麻麻赖赖』", mirror32(softmax_decode(r2, 16, "mode"))),
        ("真实数据参照", "对应的 16 张真实头像", mirror32(RTRUE[va[:16]])),
    ]
    sheet(os.path.join(FIG, "G01_pca条件GAN_噪声与坐标开关.png"),
          "里程碑 G1：PCA 条件 GAN（sweep_pca22，1,500 步，202,240 乘法）",
          ["条件由 22 个 PCA 给出，噪声只负责补细节。这是二期第一个『能给条件就画脸』的模型。",
           "本图把『坐标来源』与『噪声开关』分成两个开关：前两行只差噪声，第三行换成随机坐标。"],
          rows)


def _fmt_meta(m):
    bits = []
    if "steps_trained" in m:
        bits.append(f"{m['steps_trained']:,} 步")
    if "in_dim" in m:
        bits.append(f"输入 {m['in_dim']} 维")
    if "hid" in m:
        bits.append(f"hid {m['hid']}")
    return " · ".join(bits)


def fig02():
    rng = np.random.default_rng(31)
    layers, meta, _ = load_gen("pca22_lam5_model.json")
    X = np.clip(rng.standard_normal((16, 22)), -3.5, 3.5)
    raw = forward(layers, X)
    rows = [
        ("取众数 argmax", "16 类里的最大值，天然给出整数级\n输出 100% 落在 16 级之一",
         mirror32(softmax_decode(raw, 16, "mode"))),
        ("采样 T=1.0", "按 softmax 抽，同一批输入\n多样性更高，但会出现孤立像素",
         mirror32(softmax_decode(raw, 16, "sample", 1.0, np.random.default_rng(1)))),
        ("采样 T=0.3", "温度降低后接近 argmax\n但仍不是同一个解码器",
         mirror32(softmax_decode(raw, 16, "sample", 0.3, np.random.default_rng(2)))),
    ]
    sheet(os.path.join(FIG, "G02_512logit教师_解码开关.png"),
          "里程碑 G2：512 logits 教师（pca22_lam5，20,000 步，202,240 乘法）",
          ["这是整条灰度线的『上限参照』：22 PCA → 256 → 256 → 512，每像素 16 类分布。",
           "它之后的每一次压缩都以它为质量基准。开关＝解码方式。"],
          rows)


# ------------------------------------------------------------------ 图 3：32 连续输出


def fig03():
    # 用真实坐标：教师与学生的输入必须相同，两者才可比；而且只有真实坐标才落在训练流形上
    ZZ, RTRUE, va, _, _ = gray_basis()
    X = ZZ[va[:16], :22].astype(np.float64)
    lt, mt, _ = load_gen("pca22_lam5_model.json")
    ls, ms, _ = load_gen("pca22_gray32_model.json")
    teach = softmax_decode(forward(lt, X), 16, "mode")
    v = np.clip(forward(ls, X), 0, 15)
    vs = stretch15(v)
    lut = _lut11()
    rows = [
        ("教师 512 logits（argmax）", "每像素一个 16 类分布，argmax 一定落在整数级\n"
         f"202,240 乘法 · {mt['steps_trained']:,} 步", mirror32(teach)),
        ("学生 32 连续 · 无拉伸", "同一批坐标，输出 32 个连续值\n"
         "只有 26.1% 落在整数 ±0.1 内 → 渲染成渐变", mirror32(v)),
        ("学生 32 连续 · min-max 拉伸", "逐张拉到 0..15 再量化；拉伸是独立的后处理开关\n"
         "round 2 盲测：20.9%（不拉）→ 62.1%（拉）", mirror32(quant16(vs))),
        ("学生 32 连续 · 拉伸 + 11 档", "再套一层 16→11 的地图灰 LUT\n"
         "round 10：66.0% vs 64.8%，p=0.90", mirror32(lut[quant16(vs).astype(int)])),
    ]
    sheet(os.path.join(FIG, "G03_输出512到32_教师与学生.png"),
          "里程碑 G3：输出 512 → 32 连续值（s32_tv5 / pca22_gray32，12,000 步，79,360 乘法）",
          ["成本从 202,240 降到 79,360 乘法，代价是约 28% 的双峰位置只能取均值——这是数学上限，不是调参问题。",
           "四行同一批真实坐标，逐列可比。教师与学生的差别不在表达力，在决策规则。"],
          rows)


def _lut11():
    return np.array(json.load(open(os.path.join(WEB, "merge_mc11.json"),
                                  encoding="utf-8"))["_lut"], dtype=np.float64)


# ------------------------------------------------------------------ 图 4：输入维数阶梯


def fig04():
    rng = np.random.default_rng(404)
    seed = np.clip(rng.standard_normal((16, 22)), -3.5, 3.5)
    # 标在行尾的 acc / 同色率要用与文档同口径的样本：验证集前 1000 张真实坐标
    ZZ, RTRUE, va, _, _ = gray_basis()
    bv = va[:1000]
    rows = []
    for fn, k, label in (("pca22_dim4_model.json", 4, "PCA-4"),
                         ("pca22_dim8_model.json", 8, "PCA-8"),
                         ("pca22_dim22_model.json", 22, "PCA-22")):
        layers, meta, _ = load_gen(fn)
        v = np.clip(forward(layers, seed[:, :k]), 0, 15)
        img = mirror32(quant16(stretch15(v)))
        # 与 dim_sweep.py 同口径：不拉伸、直接四舍五入
        ref = quant16(np.clip(forward(layers, ZZ[bv][:, :k]), 0, 15))
        acc = float((ref == RTRUE[bv]).mean())
        rows.append((f"{label} 输入", f"{meta['multiplies']:,} 乘法 · acc {acc:.4f}\n"
                     f"同色率 {agree_corrected(mirror32(ref))*100:.1f}%"
                     f"（修正口径；旧口径 {meta['agree']*100:.1f}%）\n"
                     f"图上 16 张 = 同一批种子的前 {k} 个坐标", img))
    sheet(os.path.join(FIG, "G04_输入维数阶梯_同种子.png"),
          "里程碑 G4：输入 22 → 4（同一配方，只改输入维数，12,000 步）",
          ["判据不是 acc，是眼睛：低维丢掉的高频主成分正是杂色的来源，所以 PCA-4 反而更干净。",
           "三行是同一批种子的前 k 个坐标，逐列可比。行尾的 acc / 同色率在验证集前 1,000 张真实坐标上重算。"],
          rows)


# ------------------------------------------------------------------ 图 5：结构家族


def fig05():
    rng = np.random.default_rng(515)
    seed = np.clip(rng.standard_normal((16, 4)), -3.5, 3.5)
    arms = [("pca22_dim4w32L2_model.json", "4-32-32-32（两层）"),
            ("pca22_dim4w64L1_model.json", "4-64-32（单层，定案）"),
            ("pca22_dim4w128L1_model.json", "4-128-32（单层）"),
            ("pca22_dim4w256_model.json", "4-256-256-32（两层）")]
    rows = []
    for fn, label in arms:
        layers, meta, _ = load_gen(fn)
        v = np.clip(forward(layers, seed), 0, 15)
        rows.append((label, f"{meta['multiplies']:,} 乘法", mirror32(quant16(stretch15(v)))))
    sheet(os.path.join(FIG, "G05_结构家族_同种子.png"),
          "里程碑 G5：结构定宽定层（同一批 4 维种子，12,000 步/臂）",
          ["同等乘法预算下单层优于两层：4-64-32（2,304）好于 4-32-32-32（2,176），",
           "因为两层的第二个隐层只有 32 宽。三轮盲测测不出它与 74,752 乘法的差异。"],
          rows)


# ------------------------------------------------------------------ 图 6：最终模型 + 五个开关


def fig06():
    fn = "pca22_dim4w64L1_model.json"
    layers, meta, _ = load_gen(fn)
    scorer = _load_scorer()
    lut = _lut11()
    rng = np.random.default_rng(600)

    # ① 未拉伸     ② 拉伸+16级
    seed = np.clip(rng.standard_normal((16, 4)), -3.5, 3.5)
    v = np.clip(forward(layers, seed), 0, 15)
    vs = stretch15(v)
    # ③ 拉伸 + 11 档
    m11 = lut[quant16(vs).astype(int)]
    # ④ 矩形夹取：PC2 >= -0.5 且 PC1 <= -1.5
    X4 = np.clip(rng.standard_normal((4000, 4)), -3.5, 3.5)
    keep = (X4[:, 2] >= -0.5) & (X4[:, 1] <= -1.5)
    Xr = X4[keep][:16]
    vr = stretch15(np.clip(forward(layers, Xr), 0, 15))
    # ⑤ 二次打分器取头部：抽 1/0.15 倍候选，分低的留下
    Xc = np.clip(rng.standard_normal((800, 4)), -3.5, 3.5)
    s = _score(scorer, Xc)
    Xs = Xc[np.argsort(s)[:16]]
    vsc = stretch15(np.clip(forward(layers, Xs), 0, 15))
    # ⑥ 少女风格高斯
    styles = json.load(open(os.path.join(WEB, "styles_gauss.json"), encoding="utf-8"))["styles"]
    st = styles[0]
    mu = np.array(st["mu"])
    chol = np.array(st["chol"])
    z = rng.standard_normal((16, 4))
    Xg = np.clip(mu + z @ chol.T, -3.5, 3.5)
    vg = stretch15(np.clip(forward(layers, Xg), 0, 15))

    rows = [
        ("① 未拉伸 · 16 级", "输出直接量化：动态范围挤在少数几级\n实测 20.9%（round 2，dim4）",
         mirror32(quant16(v))),
        ("② min-max 拉伸 · 16 级", "逐张拉到 0..15 再量化\n实测 62.1%（round 2，dim4x）",
         mirror32(quant16(vs))),
        ("③ 拉伸 · 11 档地图灰", "每像素 4 bit 索引，11 个 HIGH 列近灰\n实测 66.0%（round 10，w64_mc）",
         mirror32(m11)),
        ("④ 拉伸 · 矩形夹取", "PC2 ≥ −0.5 且 PC1 ≤ −1.5\n覆盖 23.4% 空间，坏率 10.9%（良率 89.1%）",
         mirror32(quant16(vr))),
        ("⑤ 拉伸 · 二次打分器取头部", "scorer_quad 14 乘法，分数越低越好\n取随机候选里最低的 15%",
         mirror32(quant16(vsc))),
        ("⑥ 拉伸 · 少女风格高斯", "mu + chol @ z，clip 到 ±3.5\n1σ 坏率 0.0%",
         mirror32(quant16(vg))),
    ]
    sheet(os.path.join(FIG, "G06_最终4-64-32_六个开关.png"),
          "里程碑 G6（灰度终态）：4-64-32，2,304 乘法",
          ["同一个模型，六个开关：拉伸 / 量化档位 / 采样分布 / 打分器。每行 16 张，独立随机。",
           "这张图就是二期灰度交付物的全貌——网络只有 2,304 乘法，其余都是零成本的后处理与采样。"],
          rows)


def _load_scorer():
    return json.load(open(os.path.join(WEB, "scorer_quad.json"), encoding="utf-8"))


def _score(sc, X):
    c = sc["coef"]
    lin = np.array(c["linear"])
    sq = np.array(c["square"])
    cross = c["cross"]
    s = c["c"] + X @ lin + (X ** 2) @ sq
    for key, w in cross.items():
        i, j = int(key[0]), int(key[1])
        s = s + w * X[:, i] * X[:, j]
    return s


# ------------------------------------------------------------------ 图 7：彩色 RGB96


def fig07():
    layers, meta, _ = load_gen("colour_e6_model.json")
    rng = np.random.default_rng(901)
    X = np.clip(rng.standard_normal((16, 8)), -3.5, 3.5)
    # 导出的 e6 输出是 0..1（训练目标为 rgb96/15），meta["output_post"] 要求在页面里乘 15
    out = np.clip(forward(layers, X) * 15.0, 0, 15).reshape(16, 32, 3)
    rows = [
        ("连续 RGB（原样）", "96 个输出当 32×3 个通道，0..15 量纲\n"
         "上下端各有约 5.9%/5.5% 被截断", mirror32(out)),
        ("4 bit 量化（取整）", "每通道取整，这是真正可上游戏的形式\n"
         "量化下限 0.205 级/通道（4 bit 固有损失）", mirror32(np.rint(out))),
    ]
    # 真实数据参照
    z = np.load(os.path.join(P2, "results", "colour", "cache.npz"))
    real = z["rgb96"][rng.choice(len(z["rgb96"]), 16, replace=False)].astype(np.float64)
    rows.append(("真实数据参照", "同一批 321,219 张彩色样本里随机 16 张",
                 mirror32(real.reshape(16, 32, 3))))
    sheet(os.path.join(FIG, "C01_彩色RGB96_量化开关.png"),
          "里程碑 C1：单阶段 RGB96（colour_e6，8→128→128→96，29,696 乘法）",
          ["颜色 PCA 前 8 维的协方差几乎就是单位阵（特征值 0.98~1.01），",
           "所以 8 个独立标准正态数就是正确的输入分布——彩色线不需要灰度那条线的四个旋钮。"],
          rows)


# ------------------------------------------------------------------ 图 8：调色板 K


def fig08():
    rng = np.random.default_rng(808)
    seed = np.clip(rng.standard_normal((16, 8)), -3.5, 3.5)
    rows = []
    for fn, label, bits in (("colour_pal8_model.json", "K=8", 3),
                            ("colour_pal16_model.json", "K=16", 4),
                            ("colour_pal32_model.json", "K=32", 5)):
        layers, meta, _ = load_gen(fn)
        j = json.load(open(os.path.join(WEB, fn), encoding="utf-8"))
        pal = np.array(j["palette"])
        idx = argmax_palette(forward(layers, seed), len(pal))
        rows.append((label, f"{meta['mults']:,} 乘法 · {bits} bit/像素\n"
                     f"round 8 实测：{_round8(label)}", mirror32(pal[idx])))
    sheet(os.path.join(FIG, "C02_调色板尺寸K_同种子.png"),
          "里程碑 C2：调色板分类（K=8/16/32，每臂 12,000 步）",
          ["argmax 让边界天然变硬，解决了连续 RGB 的『糊』。三臂两两 Fisher 都不显著，",
           "所以选 K=8：3 bit/像素、50,176 乘法，成本只有 K=32 的三分之一。"],
          rows)


def _round8(label):
    return {"K=8": "50.0%（44/88）", "K=16": "51.8%（43/83）", "K=32": "60.7%（54/89）"}[label]


# ------------------------------------------------------------------ 图 9：三臂配色


def fig09():
    rng = np.random.default_rng(909)
    seed = np.clip(rng.standard_normal((16, 8)), -3.5, 3.5)
    rows = []
    for fn, label, note in (
            ("colour_pal8_model.json", "k8free 自由拟合调色板",
             "round 11：50.5%（52/103）"),
            ("colour_k8snap_model.json", "k8snap 事后吸附（同一批权重）",
             "round 11：57.3%（59/103）\nsnap vs mc p=0.0008"),
            ("colour_k8mc_model.json", "k8mc 事前约束重训",
             "round 11：33.6%（36/107）\nfree vs mc p=0.017")):
        layers, meta, _ = load_gen(fn)
        j = json.load(open(os.path.join(WEB, fn), encoding="utf-8"))
        pal = np.array(j["palette"])
        idx = argmax_palette(forward(layers, seed), len(pal))
        rows.append((label, note, mirror32(pal[idx])))
    sheet(os.path.join(FIG, "C03_三种配色方案_同种子.png"),
          "里程碑 C3：训练后吸附远好于事前的等价约束（K=8，同一批输入）",
          ["k8free 与 k8snap 是同一批权重，只换调色板；k8mc 是事前把调色板限定在 MC 色后重训。",
           "事前约束失败的机制：k-medoids 优化全像素重建误差，而像素主要是皮肤和头发，",
           "于是选出三个褐色、没有纯白、没有点缀色。"],
          rows)


# ------------------------------------------------------------------ 图 10：C1/C2 两臂


def fig10():
    rng = np.random.default_rng(1010)
    seed = np.clip(rng.standard_normal((16, 8)), -3.5, 3.5)
    rows = []
    for fn, label, note in (
            ("colour_k8snap_model.json", "k8snap（对照）", "round 12：61.1%（66/108）"),
            ("colour_c1_model.json", "C1 结构损失 λ=1",
             "round 12：41.0%（43/105）\n行方差 + 眼睛对比，p=0.004"),
            ("colour_c2_model.json", "C2 对抗 λ=0.3",
             "round 12：43.2%（51/118）\n边缘 +19%、色数 +31%，p=0.008")):
        layers, meta, _ = load_gen(fn)
        j = json.load(open(os.path.join(WEB, fn), encoding="utf-8"))
        pal = np.array(j["palette"])
        idx = argmax_palette(forward(layers, seed), len(pal))
        imgs = mirror32(pal[idx])
        rows.append((label, note + "\n" + draw_note(metrics(imgs)), imgs))
    sheet(os.path.join(FIG, "C04_结构与对抗两臂_同种子.png"),
          "里程碑 C4：两次『指标漂亮但人看着更差』（round 12，同一批输入）",
          ["C1 的模型自评良率 95.1%，实测 41.0%：所有脸戴上同一条深色眼带。",
           "C2 把边缘拉高 19%、颜色数拉高 31%，实测 43.2%。判据只能是人，"],
          rows)


# ------------------------------------------------------------------ 复验


def verify():
    out = {}
    # ---- V0 前向实现与部署页的 JS 参考逐位对齐
    ref = json.load(open(os.path.join(P2, "src", "js_forward_reference.json"), encoding="utf-8"))
    layers, meta, _ = load_gen("pca22_model.json")
    zv = np.array([ref["z_seed12345"]], dtype=np.float64)
    v0 = []
    for c in ref["cases"]:
        raw = forward(layers, np.array([c["pca"]], dtype=np.float64), z=zv)[0]
        v0.append({"case": c["name"],
                   "max_abs_err": float(np.abs(raw[:16] - np.array(c["logits_at_pos0"])).max()),
                   "argmax_matches": int(raw[:16].argmax()) == int(c["argmax_pos0"])})
    out["V0_forward_matches_js"] = v0
    # ---- V1 乘法数复算
    files = ["pca22_model.json", "pca22_lam5_model.json", "pca22_gray32_model.json",
             "pca22_w128_128_model.json", "pca22_w32_model.json", "pca22_distill32_model.json",
             "pca22_dim4_model.json", "pca22_dim8_model.json", "pca22_dim22_model.json",
             "pca22_dim4w64L1_model.json", "pca22_dim4w32L2_model.json",
             "pca22_dim4w128L1_model.json", "pca22_dim4w256L1_model.json",
             "pca22_dim4w256_model.json", "colour_e6_model.json", "colour_pal8_model.json",
             "colour_pal16_model.json", "colour_pal32_model.json",
             "colour_k8snap_model.json", "colour_k8mc_model.json",
             "colour_c1_model.json", "colour_c2_model.json"]
    v1 = {}
    for fn in files:
        layers, meta, _ = load_gen(fn)
        n = sum(W.shape[0] * W.shape[1] for W, b, r in layers) - 0
        declared = meta.get("multiplies", meta.get("mults"))
        v1[fn] = {"recount": int(n), "declared": declared,
                  "ok": declared is None or int(n) == int(declared)}
    out["V1_multiplies"] = v1

    # ---- V2 复现 dim4/dim8/dim22 的验证集准确率
    ZZ, RTRUE, va, Xc, sub = gray_basis()
    v2 = {}
    # dim_sweep.py 的口径：验证集前 1000 张、不拉伸、直接 round
    bv = va[:1000]
    for fn, k in (("pca22_dim4_model.json", 4), ("pca22_dim8_model.json", 8),
                  ("pca22_dim22_model.json", 22)):
        layers, meta, _ = load_gen(fn)
        v = np.clip(forward(layers, ZZ[bv][:, :k]), 0, 15)
        pred = quant16(v)
        acc = float((pred == RTRUE[bv]).mean())
        # 同色率：dim_sweep.py 的 speckle() 原样重放 + 修正版
        img = mirror32(pred)
        agree_buggy = speckle_legacy(img)
        h = int((img[:, :, :-1] == img[:, :, 1:]).sum())
        vv = int((img[:, :-1, :] == img[:, 1:, :]).sum())
        agree_true = (h + vv) / (len(img) * 8 * 7 * 2)
        v2[fn] = {"recorded_acc": meta.get("acc"), "recomputed_acc": round(acc, 4),
                  "recorded_agree": meta.get("agree"),
                  "recomputed_agree_legacy": round(agree_buggy, 4),
                  "recomputed_agree_corrected": round(agree_true, 4)}
    out["V2_accuracy_reproduction"] = v2

    # ---- V3 灰度 PCA 累计解释方差
    var = var_by_k(Xc[sub], [4, 8, 12, 16, 21, 22, 32])
    out["V3_gray_pca_var"] = var

    # ---- V4 彩色 PCA 解释方差 + 缓存协方差
    z = np.load(os.path.join(P2, "results", "colour", "cache.npz"))
    rgb192 = z["rgb192"].astype(np.float64)
    cm = rgb192.mean(0, keepdims=True)
    csub = np.random.default_rng(0).choice(len(rgb192), 30000, replace=False)
    cpc_all = z["cpc"].astype(np.float64)[:, :8]
    cov = np.corrcoef(cpc_all.T)
    ev = np.linalg.eigvalsh(np.cov(cpc_all.T))
    off = np.abs(cov - np.eye(8))
    out["V4_colour"] = {
        "colour_pca_var": var_by_k(rgb192[csub] - cm, [4, 8, 12, 16, 21, 24]),
        "cpc_cov_eig_minmax": [round(float(ev.min()), 4), round(float(ev.max()), 4)],
        "cpc_cond": round(float(ev.max() / ev.min()), 3),
        "cpc_mean_abs_offdiag_corr": round(float(off.sum() / (8 * 7)), 4),
    }

    # ---- V5 镜像对称率
    g = _GRAY["full"].astype(int)
    gray_sym = float((g[:, :, :4] == g[:, :, 4:][:, :, ::-1]).mean())
    rgb = z["rgb192"].reshape(-1, 8, 8, 3).astype(int)
    col_sym = float((rgb[:, :, :4] == rgb[:, :, 4:][:, :, ::-1]).mean())
    out["V5_mirror"] = {"gray_pair_equal": round(gray_sym, 4),
                        "colour_pair_equal": round(col_sym, 4)}

    # ---- V6 二次打分器：在 4096 点标签上复算 AUC
    marks = json.load(open(os.path.join(P2, "results", "pca4map", "pca4_plane_marks.json"),
                           encoding="utf-8"))
    sc = _load_scorer()
    pts, ys = _marks_xy(marks)
    s = _score(sc, pts)
    thr = sc["thresholds"]["0.5"]
    out["V6_scorer_quad"] = {
        "n": int(len(pts)),
        "auc_insample_recomputed": round(auc(ys, s), 4),
        "auc_insample_recorded": sc["meta"].get("auc_insample"),
        "acc_at_recorded_threshold": round(float(((s >= thr) == (ys == 1)).mean()), 4),
        "auc_cv_recorded": sc["meta"].get("auc_cv"),
    }

    # ---- V7 调色板吸附：k8snap 是否 = free 调色板最近的 HIGH 色
    mcj = json.load(open(os.path.join(P2, "data", "mc_map_colors.json"), encoding="utf-8"))
    HIGH = np.clip(np.array([b["rgb"] for b in mcj["bases"]], float) * mcj["_modifiers"]["HIGH"],
                   0, 255) / 17.0
    NAMES = [b["name"] for b in mcj["bases"]]
    pal_free = np.array(json.load(open(os.path.join(WEB, "colour_pal8_model.json"),
                                       encoding="utf-8"))["palette"])
    pal_snap = np.array(json.load(open(os.path.join(WEB, "colour_k8snap_model.json"),
                                       encoding="utf-8"))["palette"])
    pal_mc = np.array(json.load(open(os.path.join(WEB, "colour_k8mc_model.json"),
                                     encoding="utf-8"))["palette"])
    d = ((pal_free[:, None, :] - HIGH[None]) ** 2).sum(-1)
    nearest = HIGH[d.argmin(1)]
    names = [NAMES[i] for i in d.argmin(1)]
    out["V7_palette_snap"] = {
        "snap_matches_nearest_HIGH": bool(np.allclose(nearest, pal_snap, atol=1e-4)),
        "snap_palette_names": names,
        "snap_distinct_colours": len({tuple(np.round(p, 3)) for p in pal_snap}),
        "mc_palette_all_in_HIGH": bool(all(
            np.min(((HIGH - p) ** 2).sum(-1)) < 1e-8 for p in pal_mc)),
        "mc_palette_names": [NAMES[int(np.argmin(((HIGH - p) ** 2).sum(-1)))] for p in pal_mc],
        "max_abs_snap_vs_mc": round(float(np.abs(pal_snap - pal_mc).max()), 3),
    }

    # ---- V8 11 档归并：色值与可分档位
    lut = _lut11()
    gaps = np.diff(np.unique(lut))
    out["V8_merge11"] = {
        "n_levels": int(len(np.unique(lut))),
        "min_gap": round(float(gaps.min()), 3),
        "gaps_lt_half_level": int((gaps < 0.5).sum()),
        "distinct_if_gap_ge_0.5": int(1 + (gaps >= 0.5).sum()),
    }

    # ---- V9 落在整数级附近的比例（真实坐标，验证集前 1000 张；与文档 26.1% 同口径）
    v9 = {}
    Xreal22 = ZZ[bv][:, :22].astype(np.float64)
    Xreal4 = ZZ[bv][:, :4].astype(np.float64)
    for fn, k, do_stretch in (("pca22_lam5_model.json", 22, None),
                              ("pca22_gray32_model.json", 22, False),
                              ("pca22_dim22_model.json", 22, False),
                              ("pca22_dim4_model.json", 4, False),
                              ("pca22_dim4w64L1_model.json", 4, True),
                              ("pca22_distill32_model.json", 22, False)):
        layers, meta, _ = load_gen(fn)
        raw = forward(layers, Xreal4 if k == 4 else Xreal22)
        if do_stretch is None:                    # 教师：argmax 后必然落在整数级
            v = softmax_decode(raw, 16, "mode")
            v9[fn] = {"near_integer_frac": round(float((np.abs(v - np.rint(v)) <= 0.1).mean()), 4),
                      "mean_abs_frac_from_int": 0.0, "out_of_range_frac": 0.0}
            continue
        v = np.clip(raw, 0, 15)
        if do_stretch:
            v = stretch15(v)
        v9[fn] = {"near_integer_frac": round(float((np.abs(v - np.rint(v)) <= 0.1).mean()), 4),
                  "mean_abs_frac_from_int": round(float(np.abs(v - np.rint(v)).mean()), 4),
                  "near_integer_frac_unclipped": round(
                      float((np.abs(raw - np.rint(raw)) <= 0.1).mean()), 4),
                  "out_of_range_frac": round(float(((raw < 0) | (raw > 15)).mean()), 4)}
    out["V9_banding"] = v9

    # ---- V10 盲测轮次台账复算 + Fisher
    rounds = _round_ledger()
    out["V10_rounds"] = rounds

    # ---- V11 冻结位置守卫
    v11 = {}
    for fn, k in (("pca22_dim4w64L1_model.json", 4), ("pca22_dim4w32L2_model.json", 4),
                  ("pca22_dim4w256_model.json", 4), ("pca22_dim22_model.json", 22),
                  ("pca22_dim4_model.json", 4), ("colour_k8snap_model.json", 8),
                  ("colour_pal32_model.json", 8), ("colour_e6_model.json", 8)):
        layers, meta, _ = load_gen(fn)
        r2 = np.random.default_rng(99)
        Z = np.clip(r2.standard_normal((800, k)), -3.5, 3.5)
        v = forward(layers, Z)
        if fn.startswith("pca22"):
            v = np.clip(v, 0, 15)
        sd = v.std(0)
        v11[fn] = {"constant_outputs": int((sd < 1e-6).sum()),
                   "low_variance_outputs": int((sd < 0.05).sum()), "n_out": int(v.shape[1])}
    out["V11_frozen"] = v11

    p = os.path.join(HERE, "verify_phase2.json")
    json.dump(out, open(p, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nwrote {os.path.relpath(p, REPO)}")


def agree_corrected(imgs):
    """8x8（或 B,8,8）上真正的四邻域同色率。speckle_legacy 给出的值 = 0.5 + 0.5*本函数。"""
    a = np.asarray(imgs)
    if a.ndim == 2:
        a = a[None]
    h = int((a[:, :, :-1] == a[:, :, 1:]).sum())
    v = int((a[:, :-1, :] == a[:, 1:, :]).sum())
    return (h + v) / (len(a) * 8 * 7 * 2)


def speckle_legacy(imgs):
    """dim_sweep.py 里 speckle() 的逐字重放。

    它有 bug：水平方向（dr=0, dc=±1）比较的是 g[r_, cc] == g[rr, cc]，而 rr = r_，
    于是两端是同一个元素，恒为真。结果是这个指标的下限被抬到 0.5，
    实际值 = 0.5 + 0.5 × 真实同色率。保留它是为了复现文档里的『同色率』数值。
    """
    out = []
    for g in imgs:
        pairs = same = 0
        for r_ in range(8):
            for c in range(8):
                for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    rr, cc = r_ + dr, c + dc
                    if 0 <= rr < 8 and 0 <= cc < 8:
                        pairs += 1
                        if g[r_, cc] == g[rr, cc]:
                            same += 1
        out.append(same / pairs)
    return float(np.mean(out))


def var_by_k(Xc, ks):
    """Xc: 已去均值的样本矩阵 -> 每个 k 的累计解释方差"""
    _, S, _ = np.linalg.svd(Xc, full_matrices=False)
    ev = S ** 2
    ev = ev / ev.sum()
    return {str(k): round(float(ev[:k].sum()), 4) for k in ks}


def _marks_xy(marks):
    """pca4_plane_marks.json 只存了 1,372 个被标『中（差）』的点，不是全部 4,096 格。

    所以标签要这样还原：把 8^4 个半整数格点全列出来，落在这 1,372 个坐标上的记 bad=1，
    其余记 bad=0（bad_rate = 1372/4096 = 0.33496，与 scorer_quad 的 meta 一致）。
    """
    LEV = [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]
    grid = np.array([[a, b, c, d] for a in LEV for b in LEV for c in LEV for d in LEV])
    bad = {tuple(np.round(m["pca"], 3)) for m in marks["marks"]}
    ys = np.array([1 if tuple(np.round(g, 3)) in bad else 0 for g in grid])
    return grid, ys


def auc(y, s):
    """y=1 表示 bad；分数越高越 bad。"""
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    # 处理并列：取平均秩
    su = np.sort(s)
    i = 0
    while i < len(su):
        j = i
        while j + 1 < len(su) and su[j + 1] == su[i]:
            j += 1
        if j > i:
            m = (i + j) / 2 + 1
            ranks[s == su[i]] = m
        i = j + 1
    return float((ranks[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def fisher_p(a, b, c, d):
    """2x2 精确检验（双侧），a/b 为臂1的可接受/不可接受，c/d 为臂2。"""
    n = a + b + c + d
    r1, c1 = a + b, a + c

    def p_tab(x):
        return (math.comb(r1, x) * math.comb(n - r1, c1 - x) / math.comb(n, c1))

    p0 = p_tab(a)
    tot = 0.0
    lo = max(0, c1 - (n - r1))
    hi = min(r1, c1)
    for x in range(lo, hi + 1):
        px = p_tab(x)
        if px <= p0 + 1e-12:
            tot += px
    return float(min(1.0, tot))


def _round_ledger():
    p = os.path.join(P2, "results", "preference", "labels.jsonl")
    rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    ledger = {}
    for r in rows:
        key = f"round={r.get('round')}/arm={r.get('model')}"
        e = ledger.setdefault(key, {"n": 0, "accept_ge2": 0})
        e["n"] += 1
        e["accept_ge2"] += int(r["grade"] >= 2)
    for k, e in ledger.items():
        e["yield"] = round(e["accept_ge2"] / e["n"], 4)
    # 与已记录的 round 11 / 12 统计对账
    rec = {}
    for f in ("results/colour/k8schemes/round11.json",
              "results/colour/yield/round12.json"):
        rec[os.path.basename(f)] = json.load(open(os.path.join(P2, f), encoding="utf-8"))
    cmp11 = []
    for arm, e in rec["round11.json"]["arms"].items():
        led = ledger[f"round=11/arm={arm}"]
        cmp11.append({"arm": arm, "recomputed": led["yield"], "recorded": round(e["acc"] / e["n"], 4)})
    cmp12 = []
    for arm, e in rec["round12.json"]["arms"].items():
        led = ledger[f"round=12/arm={arm}"]
        cmp12.append({"arm": arm, "recomputed": led["yield"], "recorded": round(e["acc"] / e["n"], 4)})
    # Fisher：round 11 三对比、round 12 两对比
    def cell(rnd, arm):
        e = ledger[f"round={rnd}/arm={arm}"]
        return e["accept_ge2"], e["n"] - e["accept_ge2"]

    out = {"ledger": ledger, "round11_check": cmp11, "round12_check": cmp12}
    for a, b in (("k8free", "k8snap"), ("k8free", "k8mc"), ("k8snap", "k8mc")):
        out[f"fisher_11_{a}_vs_{b}"] = round(fisher_p(*cell(11, a), *cell(11, b)), 5)
    for a, b in (("k8snap", "c1struct"), ("k8snap", "c2adv")):
        out[f"fisher_12_{a}_vs_{b}"] = round(fisher_p(*cell(12, a), *cell(12, b)), 5)
    return out


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "both"
    if what in ("figs", "both"):
        for f in (fig01, fig02, fig03, fig04, fig05, fig06, fig07, fig08, fig09, fig10):
            print(f"[{f.__name__}]")
            f()
    if what in ("verify", "both"):
        verify()
