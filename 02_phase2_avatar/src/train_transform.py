#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
A 4 -> 4 transform of the input knobs, trained to raise the acceptable rate.

WHAT IS BEING TRAINED
    f : R^4 -> R^4        z' = f(z),   z ~ N(0, I)
maximising the expected reward of f(z). Without a constraint the optimum is a constant
collapse (a single point with the best score), so a diversity term keeps each output axis at
unit spread:

    L = -mean(reward(f(z)))  +  lam_div * mean_i (std_i(f(z)) - 1)^2

Three transforms are compared:
    identity   no transform (the current sampler)
    linear     z' = A z + b                       (20 parameters)
    mlp        4 -> 16 -> 4                       (148 parameters)
No quadratic transform is included: with 153 labels it would have more parameters than
constraints and the comparison would be meaningless.

TWO MODES
    global   reward = P(acceptable)               -- the user does not care about style
    anime    reward = the 类二次元 probe score    -- the user asked for that style

HOW THE RESULT IS MEASURED WITHOUT NEW LABELS
The honest protocol is nested: for each of five folds, fit the reward on the other four folds,
train the transform against THAT reward, then score the transformed samples with a reward
fitted on the held-out fold alone. The held-out reward is independent of both the transform
and the training reward, so the comparison "score after vs score before" is not circular.
It is still a MODEL JUDGEMENT, not a measured human rate -- verifying it needs a labelling
round, and that caveat is stated in the output.
"""
import json
import math
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WEB = os.path.join(ROOT, "webapp")
LABELS = os.path.join(ROOT, "results", "preference", "labels.jsonl")
OUT = os.path.join(ROOT, "results", "transform")
os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(4)

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("layout", os.path.join(HERE, "layout.py"))
layout = _ilu.module_from_spec(_spec); _spec.loader.exec_module(layout)


def font(sz):
    for fn in ("msyh.ttc", "simhei.ttf", "simsun.ttc"):
        p = os.path.join("C:\\Windows\\Fonts", fn)
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, sz)
            except Exception:
                pass
    return ImageFont.load_default()


def loadj(f):
    j = json.load(open(os.path.join(WEB, f), encoding="utf-8"))
    layers = j["gen"]["layers"]
    seq = []
    for i, L in enumerate(layers):
        W = torch.tensor(L["W"], dtype=torch.float32)
        b = torch.tensor(L["b"], dtype=torch.float32)
        lin = nn.Linear(W.shape[1], W.shape[0])
        with torch.no_grad():
            lin.weight.copy_(W); lin.bias.copy_(b)
        seq.append(lin)
        if i != len(layers) - 1:
            seq.append(nn.ReLU())
    m = nn.Sequential(*seq); m.eval()
    return m, int(len(layers[0]["W"][0]))


# ---------------------------------------------------------------- data
recs = [json.loads(l) for l in open(LABELS, encoding="utf-8") if l.strip()]
new = [r for r in recs if str(r.get("round") or "") == "5" and r.get("model") == "dim4x"
       and r.get("grade") in (0, 1, 2, 3, 4)]
X = np.array([r["pca"][:4] for r in new], dtype=np.float32)
g = np.array([r["grade"] for r in new])
acc = (g >= 2).astype(np.float32)
anime = (g == 4).astype(np.float32)
N = len(X)
print(f"labels: {N}   acceptable {100*acc.mean():.1f}%   anime {100*anime.mean():.1f}%")


def quad_np(P):
    d = P.shape[1]
    F = [P]
    for i in range(d):
        F.append((P[:, i] ** 2)[:, None])
        for j in range(i + 1, d):
            F.append((P[:, i] * P[:, j])[:, None])
    return np.hstack(F)


class QuadReward:
    """s(z) = c + a.z + 1/2 z^T B z, fitted by ridge on standardised quadratic features."""

    def __init__(self, Z, y, lam=30.0):
        self.mu = Z.mean(0); self.sd = Z.std(0) + 1e-9
        Zs = (Z - self.mu) / self.sd
        F = quad_np(Zs)
        self.fmu = F.mean(0); self.fsd = F.std(0) + 1e-12
        A = np.c_[(F - self.fmu) / self.fsd, np.ones(len(F))]
        w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y)
        self.w = w[:-1] / self.fsd
        self.b = float(w[-1] - (self.w * self.fmu).sum())

    def torch_score(self, Z):
        Zs = (Z - torch.tensor(self.mu)) / torch.tensor(self.sd)
        F = [Zs]
        for i in range(4):
            F.append((Zs[:, i] ** 2)[:, None])
            for j in range(i + 1, 4):
                F.append((Zs[:, i] * Zs[:, j])[:, None])
        F = torch.cat(F, 1)
        return F @ torch.tensor(self.w, dtype=torch.float32) + self.b


def auc(s, t):
    hi = s[t == 1][:, None]; lo = s[t == 0][None, :]
    return float((hi > lo).mean() + 0.5 * (hi == lo).mean())


CLIP = 3.5      # the sampler draws N(0,1) and clips here; stay inside that box


class Identity(nn.Module):
    def forward(self, z):
        return z


class Lin(nn.Module):
    def __init__(self):
        super().__init__()
        self.A = nn.Linear(4, 4)
        nn.init.eye_(self.A.weight); nn.init.zeros_(self.A.bias)

    def forward(self, z):
        return torch.clamp(self.A(z), -CLIP, CLIP)


class MLP(nn.Module):
    def __init__(self, h=16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, h), nn.ReLU(), nn.Linear(h, 4))
        with torch.no_grad():
            self.net[2].weight.mul_(0.01); self.net[2].bias.zero_()

    def forward(self, z):
        return torch.clamp(self.net(z), -CLIP, CLIP)


def train_transform(reward, kind, lam_div=10.0, steps=600, nz=3000, seed=0):
    torch.manual_seed(seed)
    m = {"identity": Identity, "linear": Lin, "mlp": MLP}[kind]()
    if kind == "identity":
        return m
    Z = torch.randn(nz, 4) * 1.0
    opt = torch.optim.Adam(m.parameters(), lr=0.01)
    eye = torch.eye(4)
    for _ in range(steps):
        opt.zero_grad()
        out = m(Z)
        r = reward.torch_score(out).mean()
        # full covariance must stay near the identity: this forbids the 1-D ridge trick
        C = torch.cov(out.T)
        loss = -r + lam_div * ((C - eye) ** 2).sum()
        loss.backward()
        opt.step()
    return m


# ---------------------------------------------------------------- nested protocol
FOLDS = 5
idx = np.random.default_rng(0).permutation(N)
VARIANTS = ["identity", "linear", "mlp"]
MODES = ["global", "anime"]
print()
print("=" * 100)
print("NESTED EVALUATION: reward trained on 4 folds, transform trained on it,")
print("samples scored by a reward fitted on the 5th fold ONLY (independent).")
print("=" * 100)
results = {mo: {v: [] for v in VARIANTS} for mo in MODES}
spread = {mo: {v: [] for v in VARIANTS} for mo in MODES}
acc_pred = {mo: {v: [] for v in VARIANTS} for mo in MODES}
min_eig = {mo: {v: [] for v in VARIANTS} for mo in MODES}
for k in range(FOLDS):
    te = idx[k::FOLDS]
    tr = np.setdiff1d(idx, te)
    if len(np.unique((g[tr] >= 2))) < 2 or len(np.unique((g[te] >= 2))) < 2:
        continue
    rew_train = QuadReward(X[tr], (acc[tr] if True else None))
    rew_eval = QuadReward(X[te], acc[te])
    for mo in MODES:
        target_tr = acc[tr] if mo == "global" else anime[tr]
        rtr = QuadReward(X[tr], target_tr)
        for v in VARIANTS:
            m = train_transform(rtr, v, seed=k)
            torch.manual_seed(1000 + k)
            Z = torch.randn(4000, 4)
            with torch.no_grad():
                out = m(Z)
            sc = rew_eval.torch_score(out).numpy()
            results[mo][v].append(float(sc.mean()))
            spread[mo][v].append(float(np.mean(out.std(0).numpy())))
            cov = np.cov(out.numpy().T)
            min_eig[mo][v].append(float(np.linalg.eigvalsh(cov).min()))
            # fraction the held-out reward calls acceptable (score > 0.5 on a 0/1 ridge)
            acc_pred[mo][v].append(float((sc > 0.5).mean()))

print()
for mo in MODES:
    print(f"  MODE = {mo}   " + ("(user does not care about style)" if mo == "global"
                                 else "(user asked for 类二次元)"))
    print(f"    {'transform':10s} {'held-out reward':>16s} {'判为可接受':>11s} "
          f"{'spread':>8s}")
    print("    " + "-" * 50)
    for v in VARIANTS:
        if not results[mo][v]:
            continue
        s = np.mean(results[mo][v]); sp = np.mean(spread[mo][v])
        ap = np.mean(acc_pred[mo][v])
        me = np.mean(min_eig[mo][v]) if min_eig[mo][v] else float("nan")
        print(f"    {v:10s} {s:16.4f} {100*ap:10.1f}% {sp:8.3f}  min-eig {me:6.3f}")
    print()

# ---------------------------------------------------------------- pictures
net, kk = loadj("pca22_dim4x_model.json")


def render(P):
    with torch.no_grad():
        v = np.clip(net(torch.tensor(P.astype(np.float32))).numpy(), 0, 15)
    lo = v.min(1, keepdims=True); hi = v.max(1, keepdims=True)
    v = np.clip((v - lo) / np.maximum(hi - lo, 1e-6) * 15, 0, 15)
    return layout.image_from_left_half(np.rint(v).astype(int))


final = {}
for mo in MODES:
    target = acc if mo == "global" else anime
    rew = QuadReward(X, target)
    for v in VARIANTS:
        m = train_transform(rew, v, seed=0)
        final[(mo, v)] = m

COLS, TILE, PAD = 20, 40, 4
rows = [(f"{mo} / {v}", final[(mo, v)]) for mo in MODES for v in ("identity", "linear", "mlp")]
Wf = 210 + COLS * (TILE + PAD) + PAD
Hf = 84 + len(rows) * (TILE + 26) + 24
cv = Image.new("RGB", (Wf, Hf), (18, 20, 24))
dr = ImageDraw.Draw(cv)
f_t, f_l, f_s = font(19), font(12), font(9)
dr.text((10, 10), "4→4 输入变换：把四个旋钮整体重映射，而不是筛选", font=f_t,
        fill=(234, 238, 244))
dr.text((10, 34), "训练目标：最大化奖励，同时约束各轴标准差保持 1（防塌缩）",
        font=f_s, fill=(150, 162, 178))
y = 62
for name, m in rows:
    torch.manual_seed(7)
    Z = torch.randn(COLS, 4)
    with torch.no_grad():
        out = m(Z).numpy()
    imgs = render(out)
    dr.text((10, y + 4), name, font=f_l, fill=(110, 168, 254))
    dr.text((10, y + 18), f"sd {np.mean(out.std(0)):.2f}", font=f_s, fill=(150, 162, 178))
    for i, gg in enumerate(imgs):
        t = np.kron(np.clip(gg / 15.0 * 255.0, 0, 255), np.ones((TILE // 8, TILE // 8)))
        cv.paste(Image.fromarray(t.astype(np.uint8), mode="L").convert("RGB"),
                 (210 + i * (TILE + PAD), y))
    y += TILE + 26
dr.text((10, Hf - 18), "注意：这些是模型判定，不是实测良率。要验证必须再打一轮标签。",
        font=f_s, fill=(224, 192, 90))
p = os.path.join(OUT, "TRANSFORM.png")
cv.save(p)
print(f"wrote {p}")

json.dump({mo: {v: results[mo][v] for v in VARIANTS} for mo in MODES},
          open(os.path.join(OUT, "transform.json"), "w"), indent=2)
print(f"wrote {os.path.join(OUT, 'transform.json')}")
print("TRANSFORM_DONE")
