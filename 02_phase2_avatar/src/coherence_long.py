#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Long run with the winning coherence setting, and a decode-matched comparison.

The lam sweep showed:
    lam=0   : mode agreement 49.2%  acc 0.4320
    lam=2   : mode agreement 56.6%  acc 0.4363   <- close to real 59.7%, accuracy not hurt
    lam=8   : mode agreement 73.6%  acc 0.3796   <- over-smoothed, accuracy falling
    lam=32  : agreement 100%        acc 0.1053   <- collapsed to one blob
so lam=2 is the operating point. This script trains LONG at lam=2 and also compares
training with sample-decoding vs mode-decoding, since the user judges the argmax and the
training signal should therefore match the argmax.
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import importlib.util as _ilu
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
_lspec = _ilu.spec_from_file_location("layout", os.path.join(HERE, "layout.py"))
layout = _ilu.module_from_spec(_lspec); _lspec.loader.exec_module(layout)
NLEV = 16
LEVELS = np.linspace(0, 255, 16)

AP = argparse.ArgumentParser()
AP.add_argument("--steps", type=int, default=8000)
AP.add_argument("--batch", type=int, default=256)
AP.add_argument("--lam", type=float, default=2.0)
AP.add_argument("--lam-final", type=float, default=0.0,
                help="if >0, linearly ramp the coherence weight from --lam to this value. "
                     "Measured reason: at a constant lam=2 the neighbour-agreement PEAKED "
                     "early (55.7% at step 1000) and then decayed to 54.3% by step 8000, "
                     "because the cross-entropy term grew as the model got confident and "
                     "the fixed weight became relatively weaker.")
AP.add_argument("--adv-weight", type=float, default=1.0,
                help="weight on the adversarial term; the D has sat at ~0.92 accuracy all "
                     "night and never been caught, so this may be pure noise")
AP.add_argument("--decode", default="mode", choices=["sample", "mode"])
AP.add_argument("--hid", type=int, default=256)
AP.add_argument("--threads", type=int, default=8)
AP.add_argument("--eval-every", type=int, default=1000)
AP.add_argument("--tag", required=True)
A = AP.parse_args()
torch.set_num_threads(A.threads)
OUT = os.path.join(ROOT, "results", "coherence", A.tag)
os.makedirs(OUT, exist_ok=True)

rgb = np.load(os.path.join(ROOT, "data", "faces8_rgb_v2.npy"))
LUM = np.array([0.299, 0.587, 0.114])
gray = (rgb.astype(np.float64) * LUM).sum(-1)
gyi = np.abs(gray[:, :, :, None] - LEVELS[None, None, None, :]).argmin(-1)
mask2 = (gyi[:, layout.POS_ROW, layout.POS_COL] ==
         gyi[:, layout.POS_ROW, layout.MIRROR_COL]).mean(1) >= 0.95
Gg = gray[mask2]
full = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_full_idx.npy"))
X = Gg.reshape(len(Gg), -1)
mu = X.mean(0, keepdims=True)
sub = np.random.default_rng(0).choice(len(X), min(60000, len(X)), replace=False)
U, S_, Vt = np.linalg.svd(X[sub] - mu, full_matrices=False)
comps = Vt[:22]
sd = (((X - mu) @ comps.T).std(0, keepdims=True) + 1e-6)
sp = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_split.npz"))
tr, va = sp["train"], sp["val"]
COND = torch.tensor(((X - mu[0]) @ comps.T) / sd[0], dtype=torch.float32)
FULL_T = torch.from_numpy(full.astype(np.int64))
RTRUE = FULL_T[:, layout.POS_ROW, layout.MIRROR_COL].reshape(-1, 32)
print(f"lam={A.lam}  decode={A.decode}  steps={A.steps}  tag={A.tag}")


class Gen(nn.Module):
    def __init__(self, hid=256, cdim=22):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(cdim, hid), nn.ReLU(),
                                 nn.Linear(hid, hid), nn.ReLU(),
                                 nn.Linear(hid, 32 * NLEV))
    def logits(self, cond):
        return self.net(cond).view(-1, 32, NLEV)


class Dis(nn.Module):
    def __init__(self, c=64, emb=8, cdim=22):
        super().__init__()
        self.emb = nn.Embedding(NLEV, emb)
        self.conv = nn.Sequential(
            nn.Conv2d(emb, c, 3, padding=1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(c, c, 3, padding=1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(c, c, 3, padding=1), nn.LeakyReLU(0.2, True))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(c * 64 + cdim, 64),
                                  nn.LeakyReLU(0.2, True), nn.Linear(64, 1))
    def forward(self, idx_img, cond):
        oh = F.one_hot(idx_img, NLEV).permute(0, 3, 1, 2).float()
        e = torch.einsum("bchw,ce->behw", oh, self.emb.weight)
        return self.head(torch.cat([self.conv(e).flatten(1), cond], 1))


def pen_tv(lg):
    p = F.softmax(lg, -1).reshape(-1, 4, 8, NLEV)
    return (p[:, 1:] - p[:, :-1]).abs().mean() + (p[:, :, 1:] - p[:, :, :-1]).abs().mean()


def speckle_np(imgs):
    n = imgs.shape[0]
    agree, iso, blobs = [], [], []
    for k in range(n):
        g = imgs[k]
        pairs = same = 0; i = 0
        for r in range(8):
            for c in range(8):
                ns = 0
                for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < 8 and 0 <= cc < 8:
                        pairs += 1
                        if g[r, c] == g[rr, cc]:
                            same += 1; ns += 1
                if ns == 0: i += 1
        agree.append(same / pairs); iso.append(i)
        seen = np.zeros((8, 8), bool); cnt = 0
        for r in range(8):
            for c in range(8):
                if seen[r, c]: continue
                lv = g[r, c]; st = [(r, c)]; seen[r, c] = True
                while st:
                    rr, cc = st.pop()
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        y, x = rr + dr, cc + dc
                        if 0 <= y < 8 and 0 <= x < 8 and not seen[y, x] and g[y, x] == lv:
                            seen[y, x] = True; st.append((y, x))
                cnt += 1
        blobs.append(cnt)
    return float(np.mean(agree)), float(np.mean(iso)), float(np.mean(blobs))


REF = speckle_np(full[va[:400]])
G = Gen(A.hid); D = Dis()
optG = torch.optim.Adam(G.parameters(), lr=3e-4, betas=(0.5, 0.999))
optD = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
rng = np.random.default_rng(0)
logf = open(os.path.join(OUT, "metrics.jsonl"), "w", encoding="utf-8")
hist = []
t0 = time.time()

for step in range(1, A.steps + 1):
    bi = tr[rng.integers(0, len(tr), A.batch)]
    cb = COND[bi]
    with torch.no_grad():
        lg0 = G.logits(cb)
        f = (F.gumbel_softmax(lg0, 1.0, hard=True).argmax(-1) if A.decode == "sample"
             else lg0.argmax(-1))
    rl = D(FULL_T[bi], cb)
    fl = D(layout.image_from_left_half(f).long(), cb)
    loss_d = F.binary_cross_entropy_with_logits(rl, torch.full_like(rl, 0.9)) + \
             F.binary_cross_entropy_with_logits(fl, torch.zeros_like(fl))
    optD.zero_grad(); loss_d.backward(); optD.step()

    lg = G.logits(cb)
    f = (F.gumbel_softmax(lg, 1.0, hard=True).argmax(-1) if A.decode == "sample"
         else lg.argmax(-1))
    fl = D(layout.image_from_left_half(f).long(), cb)
    loss_g = A.adv_weight * F.binary_cross_entropy_with_logits(fl, torch.ones_like(fl))
    tgt = RTRUE[bi]
    ce = F.cross_entropy(lg.reshape(-1, NLEV), tgt.reshape(-1))
    # The coherence weight may be RAMPED. Measured reason: at a constant lam=2 the
    # neighbour-agreement peaked early (55.7% at step 1000) and decayed to 54.3% by step
    # 8000, because the cross-entropy term grew as the model became confident and the
    # fixed weight therefore became relatively weaker over time.
    frac = step / max(A.steps, 1)
    lam_now = A.lam + (A.lam_final - A.lam) * frac if A.lam_final > 0 else A.lam
    loss_g = loss_g + ce + lam_now * pen_tv(lg)
    optG.zero_grad(); loss_g.backward(); optG.step()

    if step % A.eval_every == 0 or step == A.steps:
        G.eval()
        with torch.no_grad():
            bv = va[:400]
            cbv = COND[bv]
            lgv = G.logits(cbv)
            mod = lgv.argmax(-1).numpy()
            smp = F.gumbel_softmax(lgv, 1.0, hard=True).argmax(-1).numpy()
            Rt = RTRUE[bv]
            acc_m = float((torch.from_numpy(mod) == Rt).float().mean())
            acc_s = float((torch.from_numpy(smp) == Rt).float().mean())
            pr = F.softmax(lgv, -1)
            ent = float((-(pr * pr.clamp_min(1e-12).log()).sum(-1)).mean())
            mx = float(pr.max(-1).values.mean())
        a_m, i_m, b_m = speckle_np(layout.image_from_left_half(mod))
        a_s, i_s, b_s = speckle_np(layout.image_from_left_half(smp))
        rec = {"step": step, "lam_now": round(lam_now, 4), "ce": round(float(ce.detach()), 4),
               "acc_mode": round(acc_m, 4), "acc_sample": round(acc_s, 4),
               "entropy": round(ent, 4), "maxprob": round(mx, 4),
               "agree_mode": round(a_m, 4), "iso_mode": round(i_m, 3), "blobs_mode": round(b_m, 2),
               "agree_sample": round(a_s, 4), "iso_sample": round(i_s, 3),
               "blobs_sample": round(b_s, 2), "sec": round(time.time() - t0, 1)}
        hist.append(rec)
        logf.write(json.dumps(rec) + "\n"); logf.flush()
        print(f"  step {step:5d}  CE {rec['ce']:.4f}  acc {acc_m:.4f}  ent {ent:.3f} "
              f"maxp {mx:.3f}  | mode agree {a_m*100:5.1f}% iso {i_m:5.2f} | "
              f"sample agree {a_s*100:5.1f}%  ({rec['sec']:.0f}s)", flush=True)
        G.train()

torch.save({"G": G.state_dict(), "D": D.state_dict(), "args": vars(A),
            "levels": LEVELS.tolist()}, os.path.join(OUT, "final.pt"))

# final pictures
def grid(imgs, cols=16, sc=8):
    from PIL import Image
    rows_ = (len(imgs) + cols - 1) // cols
    cv = np.full((rows_ * 8, cols * 8), 40, dtype=np.uint8)
    for i, im in enumerate(imgs):
        r_, c_ = divmod(i, cols)
        cv[r_*8:(r_+1)*8, c_*8:(c_+1)*8] = np.clip(np.asarray(im), 0, 255).astype(np.uint8)
    big = np.kron(cv, np.ones((sc, sc), dtype=np.uint8))
    return Image.fromarray(big, mode="L").convert("RGB")

G.eval()
with torch.no_grad():
    bv = va[:32]
    lgv = G.logits(COND[bv])
    mod = lgv.argmax(-1).numpy()
    smp = F.gumbel_softmax(lgv, 1.0, hard=True).argmax(-1).numpy()
top = grid([LEVELS[full[i]] for i in bv], 16)
mid = grid([LEVELS[layout.image_from_left_half(mod)[i]] for i in range(32)], 16)
bot = grid([LEVELS[layout.image_from_left_half(smp)[i]] for i in range(32)], 16)
from PIL import Image
combo = Image.new("RGB", (top.width, top.height * 3 + 24), (30, 30, 30))
combo.paste(top, (0, 0)); combo.paste(mid, (0, top.height + 12)); combo.paste(bot, (0, 2 * top.height + 24))
combo.save(os.path.join(OUT, "COMPARE_real_mode_sample.png"))

json.dump({"args": vars(A), "reference": {"agree": REF[0], "iso": REF[1], "blobs": REF[2]},
           "history": hist, "seconds": round(time.time() - t0, 1)},
          open(os.path.join(OUT, "summary.json"), "w"), indent=2)
print(f"\n  REAL reference: agree {REF[0]*100:.1f}%  iso {REF[1]:.2f}  blobs {REF[2]:.1f}")
print(f"wrote {OUT}")
print("COHERENCE_LONG_DONE")
