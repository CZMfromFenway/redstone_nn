#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
「G 能多大程度骗过 D」+「先训再 GAN 微调」的实测。

做四件事：
  1. 判别器 D（MLP 96→128→128→1）在「真实 rgb96」与三类生成结果之间能分多开：
        a) 16 色调色板模型（k8，离散、每像素 16 选 1）
        b) 二期 k8snap（8 色、MC 吸附）
        c) 只给灰度上色得到的连续 RGB（两阶段的产物）
  2. 用人类可解释的 6 个统计量（色数、饱和、边缘、亮度均值/标准差、色相熵）拟合逻辑回归，
     看「不用神经网络」能分多开——这是分支两分布有多远的下界。
  3. 热身式 GAN 微调：以 k8 为 G 的初值，CE + 0.1×对抗（D 看 softmax 期望色），2,000 步，
     记录 D 准确率与 CE 的轨迹，以及微调前后的观感统计。
  4. 两阶段管线的可行性：z → 灰度 → 只给灰度上色，看统计量是否接近真实。

产物：results/gan/gan_probe.json、gan_fig.png
用法：python gan_probe.py
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
P4 = os.path.dirname(HERE)
REPO = os.path.dirname(P4)
P2 = os.path.join(REPO, "02_phase2_avatar")
OUT = os.path.join(P4, "results", "gan")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.join(P4, "src"))

ROW = np.array([p % 8 for p in range(32)])
COL = np.array([p // 8 for p in range(32)])
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


def mirror(img32):
    img = np.zeros((len(img32), 8, 8, 3))
    img[:, ROW, COL] = img32
    img[:, ROW, 7 - COL] = img32
    return img


def stats(p):
    flat = np.asarray(p).reshape(len(p), 32, 3)
    mx, mn = flat.max(-1), flat.min(-1)
    sat = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    ncol = np.mean([len({tuple(np.rint(c).astype(int)) for c in flat[i]}) for i in range(len(flat))])
    a = mirror(flat)
    dh = np.linalg.norm(a[:, :, 1:] - a[:, :, :-1], axis=-1)
    dv = np.linalg.norm(a[:, 1:] - a[:, :-1], axis=-1)
    edge = float(np.percentile(np.concatenate([dh.ravel(), dv.ravel()]), 90))
    lum = flat @ LUM
    # 色相熵（10 个桶）
    c = mx - mn
    r, g, b = flat[..., 0], flat[..., 1], flat[..., 2]
    h = np.zeros_like(mx)
    m1 = (c > 1e-6) & (mx == r); h[m1] = ((g[m1] - b[m1]) / c[m1]) % 6
    m2 = (c > 1e-6) & (mx == g); h[m2] = (b[m2] - r[m2]) / c[m2] + 2
    m3 = (c > 1e-6) & (mx == b); h[m3] = (r[m3] - g[m3]) / c[m3] + 4
    hh = (h[c > 1] / 6.0 * 10).astype(int)
    p10 = np.bincount(hh, minlength=10) / max(1, len(hh))
    ent = float(-(p10[p10 > 0] * np.log(p10[p10 > 0])).sum())
    return {"sat": round(float(sat.mean()), 3), "ncol": round(float(ncol), 2),
            "edge": round(edge, 2), "lum_mean": round(float(lum.mean()), 2),
            "lum_std": round(float(lum.std()), 2), "hue_ent": round(ent, 3)}


def feat6(p):
    s = stats(p)
    return [s["ncol"], s["sat"], s["edge"], s["lum_mean"], s["lum_std"], s["hue_ent"]]


def load_palette_model(path):
    j = json.load(open(path, encoding="utf-8"))
    lay = [(torch.tensor(L["W"]), torch.tensor(L["b"]), True) for L in j["body"]]
    lay.append((torch.tensor(j["head"]["W"]), torch.tensor(j["head"]["b"]), False))
    return j, lay, torch.tensor(np.array(j["palette"], dtype=np.float32))


def fwd(lay, x):
    h = x
    for W, b, relu in lay:
        h = h @ W.T + b
        if relu:
            h = torch.relu(h)
    return h


def run_d(real, fake, seed=0, steps=3000, hid=128):
    """训练一个小 D，返回 (测试准确率, AUC)。real/fake: (N,96) 0..15"""
    torch.manual_seed(seed)
    D = nn.Sequential(nn.Linear(96, hid), nn.ReLU(), nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, 1))
    opt = torch.optim.Adam(D.parameters(), lr=3e-4)
    ntr = min(len(real), len(fake)) * 4 // 5
    R = torch.tensor(real[:ntr] / 15.0, dtype=torch.float32)
    Fk = torch.tensor(fake[:ntr] / 15.0, dtype=torch.float32)
    Rt = torch.tensor(real[ntr:] / 15.0, dtype=torch.float32)
    Ft = torch.tensor(fake[ntr:] / 15.0, dtype=torch.float32)
    for _ in range(steps):
        idx = torch.randint(0, len(R), (256,))
        loss = F.binary_cross_entropy_with_logits(D(R[idx]), torch.ones(256, 1)) + \
               F.binary_cross_entropy_with_logits(D(Fk[idx]), torch.zeros(256, 1))
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        sr = D(Rt).numpy().ravel()
        sf = D(Ft).numpy().ravel()
        acc = float(((sr > 0).mean() + (sf <= 0).mean()) / 2)
        s = np.concatenate([sr, sf])
        y = np.concatenate([np.ones(len(sr)), np.zeros(len(sf))])
        o = np.argsort(s); rk = np.empty(len(s)); rk[o] = np.arange(1, len(s) + 1)
        auc = float((rk[y == 1].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (len(y) - y.sum())))
    return round(acc, 4), round(auc, 4)


def run_logit6(real, fake):
    """只用 6 个人类可解释统计量做逻辑回归（闭式岭），返回 AUC。"""
    Xr = np.array([feat6(r[None]) for r in real[:600]])
    Xf = np.array([feat6(r[None]) for r in fake[:600]])
    Xr = np.hstack([Xr, np.ones((len(Xr), 1))])
    Xf = np.hstack([Xf, np.ones((len(Xf), 1))])
    X = np.vstack([Xr, Xf])
    y = np.concatenate([np.ones(len(Xr)), np.zeros(len(Xf))])
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xn = (X - mu) / sd
    w = np.linalg.solve(Xn.T @ Xn + 1.0 * np.eye(Xn.shape[1]), Xn.T @ (y * 2 - 1))
    s = Xn @ w
    o = np.argsort(s); rk = np.empty(len(s)); rk[o] = np.arange(1, len(s) + 1)
    return round(float((rk[y == 1].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (len(y) - y.sum()))), 4)


def main():
    import train_palette as tp
    rgb96, cpc, tr, va = tp.load_cache(8, "ext32")
    POS = rgb96.reshape(len(rgb96), 32, 3)
    rng = np.random.default_rng(11)
    real = POS[rng.choice(va, 3000, replace=False)].reshape(-1, 96)     # (3000,96) 0..15

    # ---- 生成三类假样本
    j, lay, pal = load_palette_model(os.path.join(P4, "results", "train",
                                                  "K16_w128_k8", "model.json"))
    Z = np.clip(np.random.default_rng(5).standard_normal((3000, 8)), -3.5, 3.5)
    with torch.no_grad():
        lg = fwd(lay, torch.tensor(Z, dtype=torch.float32))
        idx = lg.reshape(-1, 32, 16).argmax(-1)
        fake_pal16 = pal[idx].numpy().reshape(-1, 96)
        soft16 = (torch.softmax(lg.reshape(-1, 32, 16), -1) @ pal).numpy().reshape(-1, 96)
    j2, lay2, pal2 = load_palette_model(os.path.join(P2, "webapp", "colour_k8snap_model.json"))
    with torch.no_grad():
        lg2 = fwd(lay2, torch.tensor(Z, dtype=torch.float32))
        idx2 = lg2.reshape(-1, 32, 8).argmax(-1)
        fake_k8snap = pal2[idx2].numpy().reshape(-1, 96)

    # 灰度上色（两阶段的产物）：用 k8 生成的灰度 + 灰度→颜色模型
    full = np.load(os.path.join(P2, "data_v2_canonical", "L0_full_idx.npy")).astype(np.float64)
    g2c_path = os.path.join(P4, "src", "train_gray2colour.py")
    print("D 实验：real vs 三种生成结果（D: 96-128-128-1，3,000 步）")
    res = {}
    for name, fake in (("16色调色板 k8（argmax）", fake_pal16),
                       ("16色 k8（softmax 期望色）", soft16),
                       ("二期 k8snap（8 色）", fake_k8snap)):
        acc, auc = run_d(real, fake)
        auc6 = run_logit6(real, fake)
        res[name] = {"D_acc": acc, "D_auc": auc, "logit6_auc": auc6, **stats(fake)}
        print(f"  {name:24s} D 准确率 {acc:.3f}  AUC {auc:.3f} | 6 统计量 AUC {auc6:.3f}"
              f" | 色数 {res[name]['ncol']:.1f} 饱和 {res[name]['sat']:.2f} 边缘 {res[name]['edge']:.1f}")
    res["真实数据"] = stats(real)
    res["counts"] = {"n": int(len(real))}
    json.dump(res, open(os.path.join(OUT, "gan_probe.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)

    # ---- 热身式 GAN 微调（2,000 步）
    print("\n热身 GAN 微调：k8 为初值，CE + 0.1×对抗（D 看 softmax 期望色），2,000 步")
    G_lay = [(W.clone().requires_grad_(True), b.clone().requires_grad_(True), r) for W, b, r in lay]
    D = nn.Sequential(nn.Linear(96, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1))
    optG = torch.optim.Adam([p for W, b, _ in G_lay for p in (W, b)], lr=1e-4)
    optD = torch.optim.Adam(D.parameters(), lr=3e-4)
    POSi = ((POS[:, :, None, :] - pal.numpy()[None, None, :, :]) ** 2).sum(-1).argmin(-1)
    Yi = torch.tensor(POSi, dtype=torch.long)
    Rreal = torch.tensor(real / 15.0, dtype=torch.float32)
    rng2 = np.random.default_rng(7)
    hist = []
    t0 = time.time()
    for step in range(1, 2001):
        bi = torch.tensor(rng2.integers(0, len(tr), 256))
        # 关键：CE 的输入必须是这一行自己的旋钮坐标（否则等于拿 A 脸的条件去配 B 脸的目标）
        x = torch.tensor(cpc[bi][:, :8], dtype=torch.float32)
        lg = fwd(G_lay, x).reshape(-1, 32, 16)
        ce = F.cross_entropy(lg.reshape(-1, 16), Yi[bi].reshape(-1))
        soft = (torch.softmax(lg, -1) @ pal).reshape(-1, 96)
        # D 步
        loss_d = F.binary_cross_entropy_with_logits(D(Rreal[torch.randint(0, len(Rreal), (256,))]),
                                                    torch.ones(256, 1)) + \
                 F.binary_cross_entropy_with_logits(D(soft.detach()), torch.zeros(256, 1))
        optD.zero_grad(); loss_d.backward(); optD.step()
        adv = F.binary_cross_entropy_with_logits(D(soft), torch.ones(256, 1))
        loss_g = ce + 0.1 * adv
        optG.zero_grad(); loss_g.backward(); optG.step()
        if step % 500 == 0:
            with torch.no_grad():
                with torch.no_grad():
                    sr = D(Rreal[:500]).numpy().ravel(); sf = D(soft).numpy().ravel()
                dacc = float(((sr > 0).mean() + (sf <= 0).mean()) / 2)
                sample = np.clip(pal[lg.argmax(-1)].numpy().reshape(-1, 32, 3), 0, 15)
            st = stats(sample)
            hist.append({"step": step, "ce": round(float(ce.item()), 3),
                         "adv": round(float(adv.item()), 3), "D_acc": round(dacc, 3), **st})
            print(f"  step {step:5d}  CE {float(ce.item()):.3f}  adv {float(adv.item()):.3f}  "
                  f"D 准确率 {dacc:.3f}  色数 {st['ncol']:.1f} 饱和 {st['sat']:.2f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
    finals = hist[-1] if hist else None
    json.dump({"finetune": hist, "final": finals},
              open(os.path.join(OUT, "gan_finetune.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)

    # ---- 两阶段可行性：z → 灰度 → 上色
    print("\n两阶段：z → k8 灰度 → 只给灰度上色")
    with torch.no_grad():
        gidx = lg.argmax(-1).numpy() if False else None
    # 重新前向拿灰度（上面的 lg 已被微调后的 G 覆盖，这里用原始 k8）
    with torch.no_grad():
        lg0 = fwd(lay, torch.tensor(Z[:2000], dtype=torch.float32)).reshape(-1, 32, 16)
        gidx = lg0.argmax(-1).numpy()
    gray = gidx / 15.0
    g2c = json.load(open(os.path.join(P4, "results", "colourise", "model_g2c.json"),
                         encoding="utf-8")) if os.path.exists(
        os.path.join(P4, "results", "colourise", "model_g2c.json")) else None
    if g2c is None:
        print("  （没有 model_g2c.json，跳过）")
    else:
        h = gray.astype(np.float32)
        for i, L in enumerate(g2c["body"]):
            h = np.maximum(0, h @ np.array(L["W"]).T + np.array(L["b"]))
        h = h @ np.array(g2c["head"]["W"]).T + np.array(g2c["head"]["b"])
        two = np.clip(h * 15.0, 0, 15).reshape(-1, 32, 3)
        st = stats(two)
        print(f"  两阶段产物：色数 {st['ncol']:.1f} 饱和 {st['sat']:.2f} 边缘 {st['edge']:.1f} "
              f"亮度 {st['lum_mean']:.1f}（真实：色数 {res['真实数据']['ncol']:.1f} "
              f"饱和 {res['真实数据']['sat']:.2f} 边缘 {res['真实数据']['edge']:.1f} "
              f"亮度 {res['真实数据']['lum_mean']:.1f}）")
        json.dump({"two_stage": st, "real": res["真实数据"]},
                  open(os.path.join(OUT, "two_stage.json"), "w", encoding="utf-8"),
                  indent=2, ensure_ascii=False)
        # 图
        TILE, PAD, LW, COLS = 52, 4, 300, 14
        rows = [("真实", POS[rng.choice(va, COLS, replace=False)]),
                ("单阶段 16 色（k8）", np.clip(pal[idx[:COLS]].numpy(), 0, 15)),
                ("两阶段：灰度→上色", two[:COLS]),
                ("微调后（2,000 步）", sample[:COLS])]
        W = LW + COLS * (TILE + PAD) + 20
        H = 92 + len(rows) * (TILE + 30)
        cv = Image.new("RGB", (W, H), (16, 18, 22))
        dr = ImageDraw.Draw(cv)
        dr.text((14, 12), "两阶段可行性 与 GAN 微调后的样子", font=font(19), fill=(234, 238, 244))
        dr.text((14, 40), "第二行单阶段；第三行把 z 生成的灰度交给「只给灰度上色」的模型；"
                          "第四行是 2,000 步对抗微调后的采样。", font=font(11), fill=(148, 160, 176))
        dr.text((14, 58), f"两阶段统计：饱和 {st['sat']:.2f}（真实 {res['真实数据']['sat']:.2f}）"
                          f" · 色数 {st['ncol']:.1f}（真实 {res['真实数据']['ncol']:.1f}）"
                          f" · 边缘 {st['edge']:.1f}（真实 {res['真实数据']['edge']:.1f}）",
                font=font(11), fill=(130, 142, 158))
        y = 84
        for label, arr in rows:
            dr.text((14, y + 8), label, font=font(12), fill=(112, 170, 254))
            img = mirror(np.asarray(arr).reshape(-1, 32, 3) / 15.0 * 15.0)
            for i in range(min(COLS, len(img))):
                v = np.clip(np.rint(img[i] * 17), 0, 255).astype(np.uint8)
                t = np.repeat(np.repeat(v, TILE // 8, 0), TILE // 8, 1)
                cv.paste(Image.fromarray(t), (LW + i * (TILE + PAD), y))
            y += TILE + 30
        p = os.path.join(OUT, "gan_fig.png")
        cv.save(p)
        print(f"wrote {os.path.relpath(p, REPO)}")
    print("GAN_PROBE_DONE")


if __name__ == "__main__":
    main()
