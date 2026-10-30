#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
n=16 的单色头像，用真实红石灯材质渲染。

显示规模：256×256 方块 = 128×128 像素 ⇒ n = 128/8 = 16
每像素 = 2×2 方块（一个红石灯表示不了一个像素）

关键一量：红石灯「熄灭」不是黑色，会压缩可表现灰度
"""
import json
import os
import sys
import numpy as np
from PIL import Image, ImageDraw

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
TEX = os.path.join(RESULTS, "mc_tex")
NLEV = 15


def lum(rgb):
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


OFF = np.array(Image.open(os.path.join(TEX, "redstone_lamp.png")).convert("RGB"))
ON = np.array(Image.open(os.path.join(TEX, "redstone_lamp_on.png")).convert("RGB"))
off_m = OFF.reshape(-1, 3).mean(0)
on_m = ON.reshape(-1, 3).mean(0)
Lo, Ln = float(lum(off_m)), float(lum(on_m))

print("=" * 84)
print("一、红石灯两种状态的真实亮度")
print("=" * 84)
print(f"  熄灭 平均색 RGB = {np.round(off_m,1)}   亮度 {Lo:.1f}")
print(f"  点亮 平均色 RGB = {np.round(on_m,1)}   亮度 {Ln:.1f}")
print(f"  对比度 = {Ln/Lo:.2f} : 1")
print()
print("  含义：单色显示的可表现灰度只覆盖 [熄灭, 点亮]，不是 [黑, 白]。")
print(f"  16 级相邻间距 = {(Ln-Lo)/NLEV:.2f} 亮度单位")
print(f"  若以 0..255 为满量程，这套灯只能用到 {Lo:.0f}..{Ln:.0f}")

# ---------------- 生成 16 级头像
sys.path.insert(0, HERE)
from compiler import Layer, compile_network, integer_forward  # noqa: E402

P2 = os.path.join(os.path.dirname(ROOT), "02_phase2_avatar")
sys.path.insert(0, os.path.join(P2, "src"))
from layout import image_from_left_half  # noqa: E402

mj = json.load(open(os.path.join(P2, "webapp", "pca22_dim4w64L1_model.json"),
                    encoding="utf-8"))
styles = json.load(open(os.path.join(P2, "webapp", "styles_gauss.json"),
                        encoding="utf-8"))["styles"]
comp = compile_network([Layer(np.array(mj["gen"]["layers"][0]["W"]),
                              np.array(mj["gen"]["layers"][0]["b"]),
                              "relu", comparable=False, out_bits=8),
                        Layer(np.array(mj["gen"]["layers"][1]["W"]),
                              np.array(mj["gen"]["layers"][1]["b"]),
                              "clip", comparable=True, lam=np.full(32, 15.0))],
                       -3.5, 3.5, 8, verbose=False)


def int_levels(X):
    Z = integer_forward(comp, np.asarray(X, dtype=np.int64))
    m, n = Z.max(1, keepdims=True), Z.min(1, keepdims=True)
    dZ = np.maximum(m - n, 1)
    return np.clip(np.floor_divide((Z - n) * NLEV + (dZ >> 1), dZ), 0, NLEV)


def bayer(n):
    if n == 1:
        return np.zeros((1, 1), dtype=int)
    h = bayer(n // 2)
    return np.block([[4 * h + 0, 4 * h + 2], [4 * h + 3, 4 * h + 1]])


N = 16
RK = bayer(N)

rng = np.random.default_rng(11)
FACES = []
for e in styles[:4]:
    mu = np.array(e["mu"], dtype=float)
    Lc = np.array(e["chol"], dtype=float)
    z = rng.standard_normal((1, 4))
    x = np.clip(z @ Lc.T + mu, -3.5, 3.5)
    Xq = np.clip(np.floor((x + 3.5) / 7.0 * 256), 0, 255).astype(np.int64)
    Lv = int_levels(Xq)[0]
    FACES.append(image_from_left_half(Lv).reshape(8, 8))
FACES = np.array(FACES, dtype=float)


def mono16(img8):
    out = np.zeros((8 * N, 8 * N), dtype=np.uint8)
    for i in range(8):
        for j in range(8):
            k = int(round(img8[i, j] * N * N / NLEV))
            k = max(0, min(N * N, k))
            out[i * N:(i + 1) * N, j * N:(j + 1) * N] = (RK < k).astype(np.uint8)
    return out


print()
print("=" * 84)
print("二、n=16 的抖动精度")
print("=" * 84)
print(f"  每格 {N}×{N} = {N*N} 个灯，可表现 {N*N+1} 级灰度")
err = max(abs(15 * round(L * N * N / NLEV) / (N * N) - L) for L in range(16))
print(f"  16 级映射的最大误差 = {err:.4f} 级  （n=4 时是 0.438）")
print(f"  整块显示 = {8*N}×{8*N} = {8*N*8*N} 个灯")

# ---------------- 图一：用真实材质渲染（每灯 8×8 纹素）
BLOCKPX = 8
ZOOM = 3
per = 8 * N * BLOCKPX * ZOOM          # 每张头像的像素边长
Wtot = 4 * (per + 14) + 14
cv = np.full((per + 56, Wtot, 3), 30, dtype=np.uint8)
for fi in range(4):
    mono = mono16(FACES[fi])
    big = np.zeros((8 * N * BLOCKPX, 8 * N * BLOCKPX, 3), dtype=np.uint8)
    for bi in range(8 * N):
        for bj in range(8 * N):
            t = ON if mono[bi, bj] else OFF
            tl = np.array(Image.fromarray(t).resize((BLOCKPX, BLOCKPX), Image.NEAREST))
            big[bi*BLOCKPX:(bi+1)*BLOCKPX, bj*BLOCKPX:(bj+1)*BLOCKPX] = tl
    img = Image.fromarray(big).resize((per, per), Image.LANCZOS)
    cv[42:42 + per, 14 + fi * (per + 14):14 + fi * (per + 14) + per] = np.array(img)
im = Image.fromarray(cv)
dr = ImageDraw.Draw(im)
for fi, e in enumerate(styles[:4]):
    dr.text((14 + fi * (per + 14) + 4, 22), e["name"], fill=(240, 240, 250))
dr.text((14, 6), f"n=16  128x128 像素 / 256x256 方块  真实红石灯材质", fill=(240, 240, 250))
im.save(os.path.join(RESULTS, "n16_mc.png"))
print(f"\n  wrote results/n16_mc.png  ({os.path.getsize(os.path.join(RESULTS,'n16_mc.png'))/1e6:.2f} MB)")

# ---------------- 图二：局部放大（一格 = 16×16 灯 = 32×32 方块）
BL = 24
crop = np.zeros((16 * BL, 16 * BL, 3), dtype=np.uint8)
mono = mono16(FACES[0])
for bi in range(16):
    for bj in range(16):
        t = ON if mono[bi, bj] else OFF
        tl = np.array(Image.fromarray(t).resize((BL, BL), Image.NEAREST))
        crop[bi*BL:(bi+1)*BL, bj*BL:(bj+1)*BL] = tl
Image.fromarray(crop).save(os.path.join(RESULTS, "n16_zoom.png"))
print(f"  wrote results/n16_zoom.png（一个头像格 = 16×16 个灯，1:1 纹素）")

# ---------------- 图三：n=4/8/16 同一显示尺寸对照
im3 = Image.new("RGB", (3 * (per // 2 + 12) + 12, per // 2 + 46), (30, 30, 36))
dr3 = ImageDraw.Draw(im3)
for ci, nn in enumerate((4, 8, 16)):
    rk = bayer(nn)
    mono = np.zeros((8 * nn, 8 * nn), dtype=np.uint8)
    for i in range(8):
        for j in range(8):
            k = int(round(FACES[0][i, j] * nn * nn / NLEV))
            mono[i*nn:(i+1)*nn, j*nn:(j+1)*nn] = (rk < max(0, min(nn*nn, k))).astype(np.uint8)
    # 用平均亮度上色
    col = np.where(mono[..., None] == 1, on_m, off_m).astype(np.uint8)
    img = Image.fromarray(col).resize((per // 2, per // 2), Image.LANCZOS)
    im3.paste(img, (12 + ci * (per // 2 + 12), 32))
    dr3.text((12 + ci * (per // 2 + 12) + 4, 16), f"n={nn}", fill=(240, 240, 250))
dr3.text((12, 2), "同一张头像、同一显示尺寸（128 像素宽）下的 n=4/8/16", fill=(240, 240, 250))
im3.save(os.path.join(RESULTS, "n_compare_levels.png"))
print(f"  wrote results/n_compare_levels.png")

# ---------------- 图四：远视模拟（高斯模糊）
def far_view(mono, sig_px):
    col = np.where(mono[..., None] == 1, on_m, off_m).astype(np.float64)
    k = int(sig_px * 4) | 1
    yy, xx = np.mgrid[-k//2:k//2+1, -k//2:k//2+1]
    g = np.exp(-(yy**2 + xx**2) / (2 * sig_px**2))
    g /= g.sum()
    out = np.zeros_like(col)
    for c in range(3):
        out[..., c] = np.array(Image.fromarray(col[..., c].astype(np.uint8))
                               .filter(ImageFilter.GaussianBlur(sig_px)))
    return out.astype(np.uint8)


from PIL import ImageFilter  # noqa: E402
mono16_0 = mono16(FACES[0])
im4 = Image.new("RGB", (3 * (per // 2 + 12) + 12, per // 2 + 46), (30, 30, 36))
dr4 = ImageDraw.Draw(im4)
for ci, sg in enumerate((0.5, 2.0, 6.0)):
    v = far_view(mono16_0, sg)
    img = Image.fromarray(v).resize((per // 2, per // 2), Image.LANCZOS)
    im4.paste(img, (12 + ci * (per // 2 + 12), 32))
    dr4.text((12 + ci * (per // 2 + 12) + 4, 16), f"模糊 σ={sg}灯", fill=(240, 240, 250))
dr4.text((12, 2), "远视模拟：n=16 的头像随距离增大", fill=(240, 240, 250))
im4.save(os.path.join(RESULTS, "n16_farview.png"))
print(f"  wrote results/n16_farview.png")

json.dump({"n": 16, "grid": 128, "blocks": 256,
           "lamp_off_lum": Lo, "lamp_on_lum": Ln,
           "contrast": Ln / Lo,
           "level_error": err},
          open(os.path.join(RESULTS, "n16.json"), "w", encoding="utf-8"), indent=1)
print("  已写入 results/n16.json")
print("N16_DONE")
