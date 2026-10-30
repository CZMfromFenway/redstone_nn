#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Conditional GAN: complete the right half of a face given the left half.

WHAT "CONDITIONAL" MEANS HERE (this is the whole point of the file)
------------------------------------------------------------------
An *unconditional* generator is asked "produce a face" and receives only noise. It has
to decide everything at once, and we measured that it fails completely: the
discriminator separates real from fake within a few hundred steps and the generator
receives no usable gradient (output stayed at the uniform distribution, entropy
2.768 nats vs ln16 = 2.772).

A *conditional* generator is asked a narrower question:
    "given THESE 32 left-half pixel values, what are the 32 right-half values?"
The left half is supplied as input. So the condition is:

    the observed left half of the face  (32 of the 64 gray levels)

Two consequences, both intentional:
  1. The generator never has to invent the left half, so its job has a well-defined
     answer and it cannot collapse to "invent something face-like from nothing".
  2. The discriminator's job becomes non-trivial: it sees the ASSEMBLED 8x8 image plus
     a mask marking which half was given, and must judge whether the completed right
     half is CONSISTENT with the observed left half -- not merely whether it looks like
     a face. That is the regime where adversarial training is supposed to add value.

Training pair: (real left half) -> (real right half). At generation we feed a real left
half and take the generated right half. Everything is rendered FORCED SYMMETRIC, so a
face is built by mirroring one 4-column half across the middle.

Layout: src/layout.py is the single authority. It fixes the two bugs that made earlier
pictures wrong: (a) this file used to glue the right half back in REVERSED column order,
horizontally flipping it; (b) the unconditional script generated the TOP half, leaving
rows 4..7 black.

Relative paths only. Existing files are never overwritten.
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# ------------------------------------------------------------------ canonical layout
import importlib.util as _ilu
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
_lspec = _ilu.spec_from_file_location("layout", os.path.join(HERE, "layout.py"))
layout = _ilu.module_from_spec(_lspec); _lspec.loader.exec_module(layout)

NLEV = 16
LEVELS = np.linspace(0, 255, 16)

AP = argparse.ArgumentParser()
AP.add_argument("--tag", default="cond_gan")
AP.add_argument("--steps", type=int, default=8000)
AP.add_argument("--batch", type=int, default=256)
AP.add_argument("--lr-g", type=float, default=3e-4)
AP.add_argument("--lr-d", type=float, default=2e-4)
AP.add_argument("--hid", type=int, default=256)
AP.add_argument("--zdim", type=int, default=64)
AP.add_argument("--d-c", type=int, default=64)
AP.add_argument("--d-embed", type=int, default=8)
AP.add_argument("--r1-every", type=int, default=8)
AP.add_argument("--r1-gamma", type=float, default=2.0)
AP.add_argument("--ce-weight", type=float, default=1.0,
                help=">0 adds a teacher-forced CE term to G (adversarial + likelihood). "
                     "Measured: pure adversarial collapses to uniform here, CE is what makes it learn.")
AP.add_argument("--eval-every", type=int, default=1000)
AP.add_argument("--seed", type=int, default=0)
AP.add_argument("--threads", type=int, default=8)
A = AP.parse_args()

torch.manual_seed(A.seed); np.random.seed(A.seed)
torch.set_num_threads(A.threads)

DATA = os.path.join(ROOT, "data_v2_canonical")
OUT = os.path.join(ROOT, "results", "gan_v2", A.tag)
os.makedirs(OUT, exist_ok=True)

# ---- geometry, taken from the canonical module (NOT re-derived here) ----------------
# The generator emits the left half of the 4x8 tile in canonical order p = col*8 + row.
# "right half" = the OTHER four columns of the same rows, in canonical order too.
POS_ROW = layout.POS_ROW                 # p -> row
POS_COL = layout.POS_COL                 # p -> column (0..3, the left half)
MIRROR_COL = layout.MIRROR_COL           # p -> the mirrored column (7..4)

full = np.load(os.path.join(DATA, "L0_full_idx.npy"))       # (N,8,8)
sp = np.load(os.path.join(DATA, "L0_split.npz"))
tr, va = sp["train"], sp["val"]
print(f"data {full.shape}  train {len(tr):,}  val {len(va):,}")


def halves(idx):
    """(B,) indices -> (left (B,32), right (B,32)) both in canonical order p = col*8+row.

    left  column set = {0,1,2,3}   right column set = {7,6,5,4}
    Both use the SAME canonical ordering, which is what the old version got wrong: it
    returned the right half in reversed column order, so gluing the two back together
    flipped the right half horizontally.
    """
    x = torch.from_numpy(full[idx].astype(np.int64))          # (B,8,8)
    L = x[:, POS_ROW, POS_COL]                                # (B,32)
    R = x[:, POS_ROW, MIRROR_COL]                             # (B,32)
    return L, R


def assemble(L, R):
    """(B,32) left + (B,32) right, canonical -> (B,8,8). Writes all 8 rows."""
    B = len(L)
    img = torch.zeros(B, 8, 8, dtype=torch.long)
    rows = torch.from_numpy(POS_ROW)
    img[:, rows, torch.from_numpy(POS_COL)] = L
    img[:, rows, torch.from_numpy(MIRROR_COL)] = R
    return img


def symmetrize(idx_half):
    """(B,32) one 4-column half in canonical order -> (B,8,8) mirror-symmetric image."""
    return layout.image_from_left_half(idx_half)


def chw_from_idx(img):
    """(B,8,8) long -> (B,16,8,8) float one-hot, channel-first and DIFFERENTIABLE in img."""
    return F.one_hot(img, NLEV).permute(0, 3, 1, 2).float()


class CondG(nn.Module):
    """G(observed left half, z) -> right-half logits.

    input  : left half as a 16-way one-hot (32*16 = 512) concatenated with z
    output : 32 * 16 = 512 logits, one 16-way distribution per right-half position
    """
    def __init__(self, hid, zdim):
        super().__init__()
        self.zdim = zdim
        self.net = nn.Sequential(
            nn.Linear(32 * NLEV + zdim, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
            nn.Linear(hid, 32 * NLEV),
        )

    def logits(self, L, z):
        Lo = F.one_hot(L, NLEV).float().reshape(len(L), -1)
        return self.net(torch.cat([Lo, z], 1)).view(-1, 32, NLEV)


class Disc(nn.Module):
    """Judges an ASSEMBLED image, and is additionally told which half was observed."""
    def __init__(self, c=64, emb=8):
        super().__init__()
        self.emb = nn.Embedding(NLEV, emb)
        self.conv = nn.Sequential(
            nn.Conv2d(emb + 1, c, 3, padding=1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(c, c, 3, padding=1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(c, c, 3, padding=1), nn.LeakyReLU(0.2, True),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(c * 64, 64),
                                  nn.LeakyReLU(0.2, True), nn.Linear(64, 1))

    def from_chw(self, chw, mask):
        """chw (B,16,8,8) differentiable; mask (B,1,8,8) = 1 where OBSERVED.

        The embedding lookup is a matmul, not an index lookup: argmax/index would make
        the R1 gradient penalty unable to reach the input.
        """
        e = torch.einsum("bchw,ce->behw", chw, self.emb.weight)
        return torch.cat([e, mask], 1)

    def score(self, chw, mask):
        return self.head(self.conv(self.from_chw(chw, mask)))


# observed-half mask: 1 on the left half (columns 0..3), 0 on the right half
MASK = torch.zeros(1, 1, 8, 8)
MASK[0, 0, :, :4] = 1.0

G = CondG(A.hid, A.zdim)
D = Disc(A.d_c, A.d_embed)
print(f"G params {sum(p.numel() for p in G.parameters()):,}   "
      f"D params {sum(p.numel() for p in D.parameters()):,}   ce_weight={A.ce_weight}")

optG = torch.optim.Adam(G.parameters(), lr=A.lr_g, betas=(0.5, 0.999))
optD = torch.optim.Adam(D.parameters(), lr=A.lr_d, betas=(0.5, 0.999))

rng = np.random.default_rng(0)
t0 = time.time()
hist = []

for step in range(1, A.steps + 1):
    bi = tr[rng.integers(0, len(tr), A.batch)]
    Lr, Rr = halves(bi)

    # ---- discriminator: real (Lr, Rr) vs fake (Lr, G(Lr))
    with torch.no_grad():
        Rf = F.gumbel_softmax(G.logits(Lr, torch.randn(A.batch, A.zdim)), 1.0, hard=True).argmax(-1)
    real_img = assemble(Lr, Rr)
    fake_img = assemble(Lr, Rf)
    mask_b = MASK.expand(A.batch, -1, -1, -1)
    real_chw = chw_from_idx(real_img)
    fake_chw = chw_from_idx(fake_img)
    rl = D.score(real_chw, mask_b)
    fl = D.score(fake_chw, mask_b)
    loss_d = F.binary_cross_entropy_with_logits(rl, torch.full_like(rl, 0.9)) \
           + F.binary_cross_entropy_with_logits(fl, torch.zeros_like(fl))
    if A.r1_every and step % A.r1_every == 0:
        real_chw = real_chw.detach().requires_grad_(True)
        rl2 = D.score(real_chw, mask_b)
        g = torch.autograd.grad(rl2.sum(), real_chw, create_graph=True)[0]
        loss_d = loss_d + (A.r1_gamma / 2) * g.flatten(1).pow(2).sum(1).mean()
    optD.zero_grad(); loss_d.backward(); optD.step()

    # ---- generator
    lg = G.logits(Lr, torch.randn(A.batch, A.zdim))
    Rf = F.gumbel_softmax(lg, 1.0, hard=True).argmax(-1)
    fl = D.score(chw_from_idx(assemble(Lr, Rf)), MASK.expand(A.batch, -1, -1, -1))
    loss_g = F.binary_cross_entropy_with_logits(fl, torch.ones_like(fl))
    ce = torch.zeros(())
    if A.ce_weight > 0:
        ce = F.cross_entropy(lg.reshape(-1, NLEV), Rr.reshape(-1))
        loss_g = loss_g + A.ce_weight * ce
    optG.zero_grad(); loss_g.backward(); optG.step()

    if step % 1000 == 0 or step == 1:
        print(f"  step {step:6d}  D {float(loss_d.detach()):7.4f}  G {float(loss_g.detach()):7.4f}  "
              f"CE {float(ce):7.4f}  ({time.time()-t0:5.0f}s)", flush=True)

    if step % A.eval_every == 0 or step == A.steps:
        with torch.no_grad():
            bv = va[:2048]
            Lv, Rv = halves(bv)
            lgv = G.logits(Lv, torch.zeros(len(bv), A.zdim))
            cev = float(F.cross_entropy(lgv.reshape(-1, NLEV), Rv.reshape(-1)))
            accv = float((lgv.argmax(-1) == Rv).float().mean())
            mb = MASK.expand(len(bv), -1, -1, -1)
            rl2 = D.score(chw_from_idx(assemble(Lv, Rv)), mb).squeeze(1)
            Rg = F.gumbel_softmax(lgv, 1.0, hard=True).argmax(-1)
            fl2 = D.score(chw_from_idx(assemble(Lv, Rg)), mb).squeeze(1)
            dacc = float((torch.cat([rl2, fl2]) > 0).float()
                         .eq(torch.cat([torch.ones_like(rl2), torch.zeros_like(fl2)])).float().mean())
        print(f"    EVAL {step}: heldout CE {cev:.4f}  acc {accv:.4f}  Dacc {dacc:.3f}", flush=True)
        hist.append({"step": step, "ce": cev, "acc": accv, "d_acc": dacc})

# ------------------------------------------------------------------ rendering
def grid(imgs, cols=16, sc=8):
    rows_ = (len(imgs) + cols - 1) // cols
    cv = np.full((rows_ * 8, cols * 8), 40, dtype=np.uint8)
    for i, im in enumerate(imgs):
        r_, c_ = divmod(i, cols)
        cv[r_*8:(r_+1)*8, c_*8:(c_+1)*8] = np.clip(np.asarray(im), 0, 255).astype(np.uint8)
    big = np.kron(cv, np.ones((sc, sc), dtype=np.uint8))
    for r_ in range(rows_ + 1):
        big[r_*8*sc:r_*8*sc+2, :] = 200
    for c_ in range(cols + 1):
        big[:, c_*8*sc:c_*8*sc+2] = 200
    return Image.fromarray(big, mode="L").convert("RGB")


N = 32
bv = va[:N]
Lv, Rv = halves(bv)
with torch.no_grad():
    Rg = F.gumbel_softmax(G.logits(Lv, torch.randn(N, A.zdim)), 1.0, hard=True).argmax(-1)

# A: the true faces (as stored), rendered symmetric via their own left half
real_sym = symmetrize(Lv)
# B: real left half + GENERATED right half, then forced symmetric
gen_mixed = symmetrize(Rg)
# C: the model's right-half prediction mirrored on both sides (pure model output)
gen_pure = symmetrize(Rg)

L_r = np.array([LEVELS[real_sym[i].numpy()] for i in range(N)], dtype=np.uint8)
L_m = np.array([LEVELS[gen_mixed[i].numpy()] for i in range(N)], dtype=np.uint8)
L_p = np.array([LEVELS[gen_pure[i].numpy()] for i in range(N)], dtype=np.uint8)

# the requested two-band picture: real on top, forced-symmetric generated below
top = grid([L_r[i] for i in range(N)], 16)
bot = grid([L_m[i] for i in range(N)], 16)
W = max(top.width, bot.width)
combo = Image.new("RGB", (W, top.height + bot.height + 12), (30, 30, 30))
combo.paste(top, (0, 0))
combo.paste(bot, (0, top.height + 12))
combo.save(os.path.join(OUT, "COMPARE_real_top_generated_bottom.png"))
top.save(os.path.join(OUT, "band_REAL_forced_symmetric.png"))
bot.save(os.path.join(OUT, "band_GENERATED_forced_symmetric.png"))
grid([L_p[i] for i in range(N)], 16).save(os.path.join(OUT, "band_GENERATED_pure_half.png"))

json.dump({"args": vars(A), "history": hist, "seconds": round(time.time() - t0, 1),
           "layout": "canonical p = col*8 + row (src/layout.py)",
           "condition": "observed left half (32 of 64 gray levels)",
           "rendering": {"top_band": "real faces, forced symmetric",
                         "bottom_band": "generated right half + real left half, forced symmetric",
                         "third": "generated right half mirrored on both sides"}},
          open(os.path.join(OUT, "summary.json"), "w"), indent=2)
print(f"wrote {OUT}")
print(f"  COMPARE_real_top_generated_bottom.png   <- real above, forced-symmetric generated below")
print("COND_DONE")
