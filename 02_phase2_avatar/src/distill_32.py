#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
PHASE 2 of 512 -> 32: distillation.

Why distill at all? The 512-logit teacher emits, for each of the 32 positions, a full
distribution over 16 levels. Measured earlier: 96.2% of those distributions are unimodal
and the mean number of peaks is 1.05, so almost all of that 16-dim vector is described by
where its mass sits. The EXPECTED LEVEL, E[level] = sum_c c * P(c), is therefore a compact
continuous summary of the teacher's belief at that position.

Training a 32-output student directly on the hard level throws that away: the hard target
is a step function, so the student gets a piecewise-constant signal even where the teacher
was uncertain. Distilling the teacher's expected level instead gives a smoother,
better-conditioned target that also encodes where the teacher was unsure.

Three students are compared on held-out faces:
  hard32   : MSE against the true level (the baseline already measured: acc 0.7594,
             agreement 51.7%)
  distill  : MSE against the teacher's expected level
  distill+tv: the same, plus a total-variation penalty on the continuous output field,
             which is the correct place to put a coherence term for a continuous output
             (the earlier 32-output runs applied TV to a soft distribution built from the
             value, which is a weaker regulariser)

Also reported: how often rounding the teacher's own expected level reproduces its argmax,
which bounds what the student can inherit.
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import importlib.util as _ilu
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
_lspec = _ilu.spec_from_file_location("layout", os.path.join(HERE, "layout.py"))
layout = _ilu.module_from_spec(_lspec); _lspec.loader.exec_module(layout)
NLEV = 16
LEVELS = np.linspace(0, 255, 16)
OUT = os.path.join(ROOT, "results", "distill")
os.makedirs(OUT, exist_ok=True)

AP = argparse.ArgumentParser()
AP.add_argument("--mode", default="distill", choices=["hard32", "distill", "distill_tv"])
AP.add_argument("--steps", type=int, default=12000)
AP.add_argument("--lam", type=float, default=1.0, help="TV weight (distill_tv only)")
AP.add_argument("--threads", type=int, default=10)
AP.add_argument("--tag", required=True)
A = AP.parse_args()
torch.set_num_threads(A.threads)
OUT = os.path.join(OUT, A.tag)
os.makedirs(OUT, exist_ok=True)
print(f"tag={A.tag} mode={A.mode} steps={A.steps} lam={A.lam}")

# ---------------------------------------------------------------- data and teacher
class Teacher(nn.Module):
    def __init__(self, hid=256, cdim=22):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(cdim, hid), nn.ReLU(),
                                 nn.Linear(hid, hid), nn.ReLU(),
                                 nn.Linear(hid, 32 * NLEV))
    def logits(self, c):
        return self.net(c).view(-1, 32, NLEV)


ck = torch.load(os.path.join(ROOT, "results", "coherence", "final_lam5_20k", "final.pt"),
                weights_only=False)
T = Teacher(ck["args"]["hid"], 22); T.load_state_dict(ck["G"]); T.eval()

rgb = np.load(os.path.join(ROOT, "data", "faces8_rgb_v2.npy"))
LUM = np.array([0.299, 0.587, 0.114])
gray = (rgb.astype(np.float64) * LUM).sum(-1)
gyi = np.abs(gray[:, :, :, None] - LEVELS[None, None, None, :]).argmin(-1)
kn = (gyi[:, layout.POS_ROW, layout.POS_COL] ==
      gyi[:, layout.POS_ROW, layout.MIRROR_COL]).mean(1) >= 0.95
Gg = gray[kn]
full = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_full_idx.npy"))
X = Gg.reshape(len(Gg), -1)
mu = X.mean(0, keepdims=True)
sub = np.random.default_rng(0).choice(len(X), min(60000, len(X)), replace=False)
U_, S_, Vt = np.linalg.svd(X[sub] - mu, full_matrices=False)
comps = Vt[:22]
sd = (((X - mu) @ comps.T).std(0, keepdims=True) + 1e-6)
COND = torch.tensor(((X - mu) @ comps.T) / sd[0], dtype=torch.float32)
RTRUE = torch.from_numpy(full.astype(np.int64))[:, layout.POS_ROW, layout.MIRROR_COL].reshape(-1, 32)
sp = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_split.npz"))
tr, va = sp["train"], sp["val"]


def teacher_expected(idx):
    """E[level] per position, the distillation target."""
    with torch.no_grad():
        lg = T.logits(COND[idx])
        p = F.softmax(lg, -1)
        lv = torch.arange(NLEV, dtype=torch.float32)
        return (p * lv).sum(-1)                       # (n,32) continuous


# how well does rounding the teacher's E[level] reproduce its argmax?
with torch.no_grad():
    bv = va[:2000]
    lg = T.logits(COND[bv])
    argm = lg.argmax(-1)
    p = F.softmax(lg, -1)
    lv = torch.arange(NLEV, dtype=torch.float32)
    expv = (p * lv).sum(-1)
    same = float((expv.round().long() == argm).float().mean())
    acc_t = float((argm == RTRUE[bv]).float().mean())
    acc_e = float((expv.round().long() == RTRUE[bv]).float().mean())
print(f"\nteacher on 2000 held-out faces:")
print(f"  argmax accuracy                 {acc_t:.4f}")
print(f"  round(expected level) accuracy  {acc_e:.4f}   "
      f"(agrees with argmax on {same*100:.1f}% of positions)")
print(f"  -> the expected level is a faithful compact summary, so it is a good target")


class Student(nn.Module):
    def __init__(self, hid=256, cdim=22):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(cdim, hid), nn.ReLU(),
                                 nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, 32))
    def forward(self, c):
        return self.net(c).clamp(0, 15)


def tv_cont(v):
    p = v.view(-1, 4, 8)
    return ((p[:, 1:] - p[:, :-1]).abs().mean()
            + (p[:, :, 1:] - p[:, :, :-1]).abs().mean())


def speckle(imgs):
    agree = []
    for g in imgs:
        pairs = same_n = 0
        for r_ in range(8):
            for c in range(8):
                for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    rr, cc = r_ + dr, c + dc
                    if 0 <= rr < 8 and 0 <= cc < 8:
                        pairs += 1
                        if g[r_, cc] == g[rr, cc]:
                            same_n += 1
        agree.append(same_n / pairs)
    return float(np.mean(agree))


# precompute the distillation target for the training rows (cheap enough to do once)
print("\nprecomputing the teacher's expected levels for the training split ...")
t0 = time.time()
EXP_TR = np.empty((len(tr), 32), dtype=np.float32)
for s in range(0, len(tr), 8192):
    e = min(s + 8192, len(tr))
    EXP_TR[s:e] = teacher_expected(tr[s:e]).numpy()
print(f"  done in {time.time()-t0:.1f}s")

S = Student(ck["args"]["hid"], 22)
opt = torch.optim.Adam(S.parameters(), lr=3e-4, betas=(0.5, 0.999))
rng = np.random.default_rng(0)
t0 = time.time()
hist = []
for step in range(1, A.steps + 1):
    bi = rng.integers(0, len(tr), 256)
    pred = S(COND[tr[bi]])
    if A.mode == "hard32":
        loss = F.mse_loss(pred, RTRUE[tr[bi]].float())
    else:
        loss = F.mse_loss(pred, torch.from_numpy(EXP_TR[bi]))
        if A.mode == "distill_tv":
            loss = loss + A.lam * tv_cont(pred)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % max(A.steps // 6, 1) == 0 or step == A.steps:
        with torch.no_grad():
            pv = S(COND[va[:1000]])
            dec = pv.round().long()
            Rt = RTRUE[va[:1000]]
            acc = float((dec == Rt).float().mean())
            mae = float((dec.float() - Rt.float()).abs().mean())
            # how closely does the student imitate the teacher's own output?
            with torch.no_grad():
                lgv = T.logits(COND[va[:1000]])
                targm = lgv.argmax(-1)
                p2 = F.softmax(lgv, -1)
                texp = (p2 * torch.arange(NLEV, dtype=torch.float32)).sum(-1)
            agree_t = float((dec == targm).float().mean())
        ag = speckle(layout.image_from_left_half(dec.numpy()))
        rec = {"step": step, "acc": round(acc, 4), "mae": round(mae, 3),
               "agree": round(ag, 4), "agrees_with_teacher": round(agree_t, 4),
               "sec": round(time.time() - t0, 1), "loss": round(loss.item(), 4)}
        hist.append(rec)
        print(f"  step {step:5d}  loss {loss.item():.4f}  acc {acc:.4f}  MAE {mae:.3f}  "
              f"agree {ag*100:5.1f}%  matches teacher {agree_t*100:5.1f}%  "
              f"({rec['sec']:.0f}s)", flush=True)

torch.save({"S": S.state_dict(), "args": vars(A)}, os.path.join(OUT, "final.pt"))
json.dump({"args": vars(A), "history": hist,
           "teacher": {"acc_argmax": acc_t, "acc_expected_round": acc_e,
                       "same": same}},
          open(os.path.join(OUT, "summary.json"), "w"), indent=2)
print(f"\n  teacher reference: acc {acc_t:.4f}, agreement 60.7%")
print(f"wrote {OUT}")
print("DISTILL_DONE")
