#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
The transform raises the predicted score while making every face the same. Measure both.

THE FAILURE, STATED
The 4->4 transform lifted the held-out reward from 0.32 to 0.95 and the predicted acceptable
share from 16% to 78%. The rendered faces, however, were nearly identical. The constraint was
on the INPUT distribution -- per-axis sd first, then the full covariance -- and both were
satisfied (min eigenvalue 0.935 against 0.959 for the identity). So the input did not collapse.

What collapsed is the OUTPUT. The generator's sensitivity varies across the input box: in some
regions a shift of one unit barely changes the face. A transform that maximises a reward
computed FROM THE INPUT ALONE has no way to notice, so it parks the sampler wherever the score
is high, including regions where the model is flat.

That is a structural problem with optimising an input-only reward, not a tuning issue. This
script measures it: input covariance (preserved) against output pixel diversity (collapsed).
"""
import json
import os
import sys
import numpy as np
import torch
import torch.nn as nn

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
CLIP = 3.5

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("layout", os.path.join(HERE, "layout.py"))
layout = _ilu.module_from_spec(_spec); _spec.loader.exec_module(layout)


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


recs = [json.loads(l) for l in open(LABELS, encoding="utf-8") if l.strip()]
new = [r for r in recs if str(r.get("round") or "") == "5" and r.get("model") == "dim4x"
       and r.get("grade") in (0, 1, 2, 3, 4)]
X = np.array([r["pca"][:4] for r in new], dtype=np.float32)
acc = np.array([1.0 if r["grade"] >= 2 else 0.0 for r in new], dtype=np.float32)


def quad_np(P):
    d = P.shape[1]
    F = [P]
    for i in range(d):
        F.append((P[:, i] ** 2)[:, None])
        for j in range(i + 1, d):
            F.append((P[:, i] * P[:, j])[:, None])
    return np.hstack(F)


class QR:
    def __init__(self, Z, y, lam=30.0):
        self.mu = Z.mean(0); self.sd = Z.std(0) + 1e-9
        F = quad_np((Z - self.mu) / self.sd)
        self.fmu = F.mean(0); self.fsd = F.std(0) + 1e-12
        A = np.c_[(F - self.fmu) / self.fsd, np.ones(len(F))]
        w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y)
        self.w = w[:-1] / self.fsd
        self.b = float(w[-1] - (self.w * self.fmu).sum())

    def ts(self, Z):
        Zs = (Z - torch.tensor(self.mu)) / torch.tensor(self.sd)
        F = [Zs]
        for i in range(4):
            F.append((Zs[:, i] ** 2)[:, None])
            for j in range(i + 1, 4):
                F.append((Zs[:, i] * Zs[:, j])[:, None])
        return torch.cat(F, 1) @ torch.tensor(self.w, dtype=torch.float32) + self.b


class Lin(nn.Module):
    def __init__(self):
        super().__init__()
        self.A = nn.Linear(4, 4)
        nn.init.eye_(self.A.weight); nn.init.zeros_(self.A.bias)

    def forward(self, z):
        return torch.clamp(self.A(z), -CLIP, CLIP)


rew = QR(X, acc)
torch.manual_seed(0)
m = Lin()
Ztr = torch.randn(3000, 4)
opt = torch.optim.Adam(m.parameters(), lr=0.01)
eye = torch.eye(4)
for _ in range(600):
    opt.zero_grad()
    out = m(Ztr)
    loss = -rew.ts(out).mean() + 10.0 * ((torch.cov(out.T) - eye) ** 2).sum()
    loss.backward(); opt.step()

net, kk = loadj("pca22_dim4x_model.json")


def render(P):
    with torch.no_grad():
        v = np.clip(net(torch.tensor(P.astype(np.float32))).numpy(), 0, 15)
    lo = v.min(1, keepdims=True); hi = v.max(1, keepdims=True)
    return np.rint(np.clip((v - lo) / np.maximum(hi - lo, 1e-6) * 15, 0, 15)).astype(int)


def outdiv(P, n=2000):
    a = render(P[:n]).astype(float)
    pr = np.random.default_rng(2).integers(0, n, size=(8000, 2))
    sel = pr[:, 0] != pr[:, 1]
    return (float(np.abs(a[pr[sel, 0]] - a[pr[sel, 1]]).mean()),
            float(a.std(0).mean()))


torch.manual_seed(7)
Z = torch.randn(4000, 4)
with torch.no_grad():
    Zi = Z.numpy()
    Zt = m(Z).numpy()

print("=" * 100)
print("INPUT DIVERSITY vs OUTPUT DIVERSITY")
print("=" * 100)
print(f"  {'dist':26s} {'input cov min-eig':>18s} {'output pixel diff':>18s} "
      f"{'per-position sd':>16s}")
print("  " + "-" * 82)
for name, P in (("identity N(0,1)", Zi), ("linear transform", Zt)):
    c = np.cov(P.T)
    pd, pp = outdiv(P.astype(np.float32))
    print(f"  {name:26s} {np.linalg.eigvalsh(c).min():18.3f} {pd:18.2f} {pp:16.3f}")

print()
print("  Reading: the input covariance is essentially unchanged (its smallest eigenvalue stays")
print("  near 1) while the OUTPUT diversity drops. The transform is not squeezing the input;")
print("  it is parking the sampler where the generator has gone flat.")
print()
print("  This is a structural consequence of optimising a reward that is a function of the")
print("  FOUR INPUTS ONLY. It cannot see the face. Any such optimiser will find regions where")
print("  the reward is high and the model is degenerate, and there is no way to tell from the")
print("  reward alone.")

json.dump({"identity": {"in_min_eig": float(np.linalg.eigvalsh(np.cov(Zi.T)).min()),
                        "out_diff": outdiv(Zi.astype(np.float32))[0]},
           "transform": {"in_min_eig": float(np.linalg.eigvalsh(np.cov(Zt.T)).min()),
                         "out_diff": outdiv(Zt.astype(np.float32))[0]}},
          open(os.path.join(OUT, "collapse.json"), "w"), indent=2)
print(f"\nwrote {os.path.join(OUT, 'collapse.json')}")
print("COLLAPSE_DONE")
