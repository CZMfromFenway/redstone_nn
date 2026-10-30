#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Condition-strength sweep: how weak can the condition get before generation collapses?

Motivation (from the user's critique): conditioning on the left half scored 0.98 pixel
accuracy, but on the >=95%-symmetric subset that condition nearly RESTATES the answer
(mirror-forcing error is only 0.34%), so the score mostly measures "did it learn to
mirror". That is not a real generation test. So we sweep condition strength:

    rung            generator input                       carries the answer?
    ---------------------------------------------------------------------------
    uncond          z only                                no
    pca_k           z + PCA_k(face)  for k in {4,8,16,22} partly, and k sets how much
    half            z + the actual 32 left-half values     yes (the old, weak test)

and we judge every rung with ONE honest measuring stick, applied identically:
    * distinct levels per sample
    * nearest-neighbour distance between generated samples (mode collapse)
    * exact duplicate count
    * **distance to the nearest REAL face** -- separates "generating" from "copying"
    * held-out cross-entropy and pixel accuracy against the real data

The nearest-real-face distance is the number that matters for the user's concern: if a
method scores well only because it reproduces training faces, that distance collapses.

Relative paths only. Existing files are never overwritten.
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

import importlib.util as _ilu
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
_lspec = _ilu.spec_from_file_location("layout", os.path.join(HERE, "layout.py"))
layout = _ilu.module_from_spec(_lspec); _lspec.loader.exec_module(layout)

NLEV = 16
LEVELS = np.linspace(0, 255, 16)

AP = argparse.ArgumentParser()
AP.add_argument("--tag", required=True)
AP.add_argument("--mode", required=True,
                choices=["uncond", "pca", "half", "code"])
AP.add_argument("--pca-k", type=int, default=0, help="number of PCA components for --mode pca")
AP.add_argument("--code-k", type=int, default=0, help="style-code dimension for --mode code")
AP.add_argument("--steps", type=int, default=3000)
AP.add_argument("--batch", type=int, default=256)
AP.add_argument("--hid", type=int, default=256)
AP.add_argument("--zdim", type=int, default=64)
AP.add_argument("--lr-g", type=float, default=3e-4)
AP.add_argument("--lr-d", type=float, default=2e-4)
AP.add_argument("--d-c", type=int, default=64)
AP.add_argument("--ce-weight", type=float, default=1.0)
AP.add_argument("--r1-every", type=int, default=8)
AP.add_argument("--r1-gamma", type=float, default=2.0)
AP.add_argument("--eval-every", type=int, default=1000)
AP.add_argument("--seed", type=int, default=0)
AP.add_argument("--threads", type=int, default=8)
AP.add_argument("--nmatch", type=int, default=2000, help="real pool size for novelty distance")
AP.add_argument("--no-noise", action="store_true",
                help="generator sees ONLY the condition (no latent z), so it must commit "
                     "to one answer per condition. The user's control experiment.")
AP.add_argument("--deadline", default="", help="HH:MM; stop early if past it")
A = AP.parse_args()

torch.manual_seed(A.seed); np.random.seed(A.seed)
torch.set_num_threads(A.threads)

DATA = os.path.join(ROOT, "data_v2_canonical")
OUT = os.path.join(ROOT, "results", "sweep", A.tag)
os.makedirs(OUT, exist_ok=True)


def past_deadline():
    if not A.deadline:
        return False
    try:
        h, m = A.deadline.split(":")
        now = time.localtime()
        return (now.tm_hour * 60 + now.tm_min) >= (int(h) * 60 + int(m))
    except Exception:
        return False


# ------------------------------------------------------------------ data
full = np.load(os.path.join(DATA, "L0_full_idx.npy"))            # (N,8,8) uint8
sp = np.load(os.path.join(DATA, "L0_split.npz"))
tr, va = sp["train"], sp["val"]
print(f"data {full.shape}  train {len(tr):,}  val {len(va):,}")

# Canonical left half (all 8 rows, columns 0..3) -> one value per position.
# NOTE: layout.left_half_from_image() does NOT extract a half -- it REBUILDS the full
# 8x8 image from a left half. Calling it here silently produced 64 values instead of 32,
# which then blew up the discriminator input (256x576 vs 96x256).
left = full[:, layout.POS_ROW, layout.POS_COL].astype(np.int64)    # (N,32)
assert left.shape[1] == 32, f"expected 32 left-half positions, got {left.shape}"
left_canon = left

# ---- condition builders ------------------------------------------------------------
COND_DIM = 0
PCA_MEAN = None
PCA_COMPS = None
CODEBOOK = None
CODE_PRIOR = None


def build_condition():
    """Returns (cond_matrix (N, COND_DIM) float32, description)."""
    global COND_DIM, PCA_MEAN, PCA_COMPS, CODEBOOK, CODE_PRIOR
    if A.mode == "uncond":
        COND_DIM = 0
        return np.zeros((len(left), 0), dtype=np.float32), "none (z only)"

    if A.mode == "half":
        # The condition is 32 positions x 16 one-hot levels = 512 values. COND_DIM must
        # record the ACTUAL width that gets returned, otherwise the discriminator tail is
        # built for the wrong input size (this caused "256x576 vs 96x256").
        COND_DIM = 32 * NLEV
        oh = np.zeros((len(left), 32, NLEV), dtype=np.float32)
        oh[np.arange(len(left))[:, None], np.arange(32)[None, :], left] = 1.0
        return oh.reshape(len(left), -1), "the 32 real left-half levels (one-hot, 512)"

    if A.mode == "pca":
        k = A.pca_k
        # PCA on the CONTINUOUS grayscale image (well-posed), then use the k leading
        # coefficients of each face as its style vector.
        rgb = np.load(os.path.join(ROOT, "data", "faces8_rgb_v2.npy"))
        LUM = np.array([0.299, 0.587, 0.114])
        gray = (rgb.astype(np.float64) * LUM).sum(-1)             # (N,8,8)
        # keep only the rows that survived the symmetry filter, in the same order
        # rebuild the mask used by build_L0_data.py
        gyi = np.abs(gray[:, :, :, None] - LEVELS[None, None, None, :]).argmin(-1)
        lc = layout.POS_COL; lr = layout.POS_ROW; rc = layout.MIRROR_COL
        sym = (gyi[:, lr, lc] == gyi[:, lr, rc]).mean(1)
        m = sym >= 0.95
        # NOTE: do not name this G -- it would shadow the Gen class below.
        Gg = gray[m]                                              # (n,8,8) aligned with `full`
        assert len(Gg) == len(full), f"alignment broke: {len(Gg)} vs {len(full)}"
        X = Gg.reshape(len(Gg), -1)
        mu = X.mean(0, keepdims=True)
        Xc = X - mu
        # economy SVD on a subsample to get the components, then project everything
        sub = np.random.default_rng(0).choice(len(Xc), min(60000, len(Xc)), replace=False)
        U, S, Vt = np.linalg.svd(Xc[sub], full_matrices=False)
        comps = Vt[:k]
        coef = Xc @ comps.T                                       # (N,k)
        sd = coef.std(0, keepdims=True) + 1e-6
        coef = coef / sd
        # per-face raw condition = leading k coefficients (unnormalised magnitudes kept)
        COND_DIM = k
        PCA_MEAN = mu.astype(np.float32)
        PCA_COMPS = comps.astype(np.float32)
        ev = (S ** 2) / (S ** 2).sum()
        print(f"  PCA on the >=95%-symmetric grayscale set: k={k} explains "
              f"{100*ev[:k].sum():.2f}% of variance")
        return coef.astype(np.float32), f"top {k} PCA coefficients of the face"

    if A.mode == "code":
        k = A.code_k
        # A DISCRETE style code learned without labels: k-means on the PCA-16 embedding
        # of each face. This is the "find styles automatically" idea in its simplest
        # form; the codebook + prior are both cheap and portable (a lookup table).
        rgb = np.load(os.path.join(ROOT, "data", "faces8_rgb_v2.npy"))
        LUM = np.array([0.299, 0.587, 0.114])
        gray = (rgb.astype(np.float64) * LUM).sum(-1)
        gyi = np.abs(gray[:, :, :, None] - LEVELS[None, None, None, :]).argmin(-1)
        lc = layout.POS_COL; lr = layout.POS_ROW; rc = layout.MIRROR_COL
        m = (gyi[:, lr, lc] == gyi[:, lr, rc]).mean(1) >= 0.95
        Gg = gray[m]
        assert len(Gg) == len(full)
        X = Gg.reshape(len(Gg), -1)
        mu = X.mean(0, keepdims=True)
        sub = np.random.default_rng(0).choice(len(X), min(80000, len(X)), replace=False)
        U, S, Vt = np.linalg.svd(X[sub] - mu, full_matrices=False)
        E = (X - mu) @ Vt[:16].T                                  # (N,16) embedding
        # k-means, k = 2**code_k
        K = 2 ** k
        rng = np.random.default_rng(1)
        C = E[rng.choice(len(E), K, replace=False)].copy()
        for _ in range(25):
            d = ((E[:, None, :] - C[None, :, :]) ** 2).sum(2)
            lab = d.argmin(1)
            for j in range(K):
                if (lab == j).any():
                    C[j] = E[lab == j].mean(0)
        prior = np.bincount(lab, minlength=K) / len(lab)
        CODEBOOK = C.astype(np.float32)
        CODE_PRIOR = prior.astype(np.float32)
        eff = float(np.exp(-(prior * np.log(np.clip(prior, 1e-12, 1))).sum()))
        print(f"  style code: {k} bits -> {K} clusters; effective used clusters {eff:.1f}; "
              f"cluster sizes min {prior.min()*len(lab):.0f}")
        # condition = the continuous cluster centre (a learned style vector)
        cond = CODEBOOK[lab]
        COND_DIM = cond.shape[1]
        np.save(os.path.join(OUT, "codebook.npy"), CODEBOOK)
        json.dump({"code_k": k, "K": K, "prior": prior.tolist(),
                   "effective_clusters": eff}, open(os.path.join(OUT, "code_info.json"), "w"), indent=2)
        return cond.astype(np.float32), f"learned {k}-bit style code ({K} clusters, continuous centre)"


cond_all, cond_desc = build_condition()
print(f"mode={A.mode}  condition: {cond_desc}   cond_dim={COND_DIM}")

# ------------------------------------------------------------------ model
class Gen(nn.Module):
    """Condition -> 32x16 logits.

    use_noise=True   : input is [condition ; z], the usual conditional GAN
    use_noise=False  : input is the condition ONLY. The generator then has no source of
                       randomness, so it must commit to a single answer per condition --
                       which in a likelihood-trained model means it learns the conditional
                       MODE rather than a distribution. The user proposed this as a control:
                       measured, z carried only 0.2% of the logit variance while the
                       condition carried 99.8%, so z may be pure dither rather than signal.
    """
    def __init__(self, hid, zdim, cdim, use_noise=True):
        super().__init__()
        self.zdim = zdim
        self.cdim = cdim
        self.use_noise = use_noise
        in_dim = cdim + (zdim if use_noise else 0)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
            nn.Linear(hid, 32 * NLEV),
        )

    def logits(self, cond, z):
        if self.cdim == 0:
            x = z
        elif self.use_noise:
            x = torch.cat([cond, z], 1)
        else:
            x = cond
        return self.net(x).view(-1, 32, NLEV)

    def forward(self, cond, z):
        """Alias so the module can be CALLED as G_(cond, z). Without this, calling the
        module raises 'missing the required forward function'."""
        return self.logits(cond, z)


class Dis(nn.Module):
    """Judges the assembled symmetric image, and is TOLD the condition."""
    def __init__(self, c, emb, cdim):
        super().__init__()
        self.emb = nn.Embedding(NLEV, emb)
        self.cdim = cdim
        self.conv = nn.Sequential(
            nn.Conv2d(emb, c, 3, padding=1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(c, c, 3, padding=1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(c, c, 3, padding=1), nn.LeakyReLU(0.2, True),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(c * 64 + cdim, 64),
                                  nn.LeakyReLU(0.2, True), nn.Linear(64, 1))

    def from_idx(self, idx_img, cond):
        oh = F.one_hot(idx_img, NLEV).permute(0, 3, 1, 2).float()
        e = torch.einsum("bchw,ce->behw", oh, self.emb.weight)
        f = self.conv(e).flatten(1)                    # (B, c*64)
        # when there is no condition, skip the concatenation entirely; branching INSIDE
        # the head call was a bug (the no-condition branch shape came out (B,1)).
        if self.cdim:
            f = torch.cat([f, cond], 1)
        return self.head(f)

    def forward(self, idx_img, cond):
        """Alias so the module can be CALLED as D_(img, cond)."""
        return self.from_idx(idx_img, cond)


G_ = Gen(A.hid, A.zdim, COND_DIM, use_noise=not A.no_noise)
D_ = Dis(A.d_c, 8, COND_DIM)
nG = sum(p.numel() for p in G_.parameters()); nD = sum(p.numel() for p in D_.parameters())
print(f"G {nG:,} params   D {nD:,} params")

optG = torch.optim.Adam(G_.parameters(), lr=A.lr_g, betas=(0.5, 0.999))
optD = torch.optim.Adam(D_.parameters(), lr=A.lr_d, betas=(0.5, 0.999))

cond_t_all = torch.from_numpy(cond_all)
full_t_all = torch.from_numpy(full.astype(np.int64))


def to_img(left_idx):
    return layout.image_from_left_half(left_idx)


def cond_of(rows):
    return cond_t_all[rows]


rng = np.random.default_rng(0)
t0 = time.time()
hist = []

# real pool for the novelty distance
pool_idx = va[np.random.default_rng(5).choice(len(va), min(A.nmatch, len(va)), replace=False)]
POOL = full[pool_idx]                                            # (M,8,8) uint8


def metrics(gen_idx):
    """gen_idx: (n,32) -> dict. Uses the same definitions for every rung."""
    img = to_img(torch.from_numpy(gen_idx)).numpy()              # (n,8,8)
    a = gen_idx
    n = len(a)
    lv = np.array([len(np.unique(a[i])) for i in range(n)])
    Dm = (a[:, None, :] != a[None, :, :]).sum(2).astype(np.int32)
    np.fill_diagonal(Dm, 10 ** 6)
    nn = Dm.min(1)
    dups = n - len({a[i].tobytes() for i in range(n)})
    # novelty: Hamming distance from each generated image to the nearest real face
    flat_g = img.reshape(n, -1).astype(np.int16)
    flat_p = POOL.reshape(len(POOL), -1).astype(np.int16)
    nr = []
    for s in range(0, n, 64):
        d = (flat_g[s:s + 64, None, :] != flat_p[None, :, :]).sum(2)
        nr.extend(d.min(1).tolist())
    sym = float((img[:, :, :4] == img[:, :, 4:][:, :, ::-1]).mean())
    return {"levels": float(lv.mean()), "nn_med": int(np.median(nn)), "nn_min": int(nn.min()),
            "dups": int(dups), "near_real_mean": float(np.mean(nr)),
            "near_real_min": int(np.min(nr)), "symmetry": sym}


print(f"\nreference: real faces have near_real_min = 0 by definition; "
      f"a COPYING generator would show near_real < ~5")

for step in range(1, A.steps + 1):
    if step % 200 == 0 and past_deadline():
        print(f"  deadline reached at step {step}; stopping early", flush=True)
        break

    bi = tr[rng.integers(0, len(tr), A.batch)]
    cb = cond_of(bi)
    Lb = left_canon[bi]
    with torch.no_grad():
        zf = torch.randn(A.batch, A.zdim)
        Rf = F.gumbel_softmax(G_(cb, zf), 1.0, hard=True).argmax(-1)
    real_img = full_t_all[bi]
    fake_img = to_img(Rf)
    rl = D_(real_img, cb)
    fl = D_(fake_img, cb)
    loss_d = F.binary_cross_entropy_with_logits(rl, torch.full_like(rl, 0.9)) \
           + F.binary_cross_entropy_with_logits(fl, torch.zeros_like(fl))
    if A.r1_every and step % A.r1_every == 0:
        # R1 penalty: differentiate D's score w.r.t. the one-hot of the REAL image.
        # Needs the one-hot to carry a gradient, so rebuild it here with requires_grad.
        real_oh = F.one_hot(real_img, NLEV).permute(0, 3, 1, 2).float().requires_grad_(True)
        e = torch.einsum("bchw,ce->behw", real_oh, D_.emb.weight)
        f = D_.conv(e).flatten(1)
        sc = D_.head(torch.cat([f, cb], 1) if COND_DIM else f)
        g = torch.autograd.grad(sc.sum(), real_oh, create_graph=True)[0]
        loss_d = loss_d + (A.r1_gamma / 2) * g.flatten(1).pow(2).sum(1).mean()
    optD.zero_grad(); loss_d.backward(); optD.step()

    zf = torch.randn(A.batch, A.zdim)
    lg = G_(cb, zf)
    Rf = F.gumbel_softmax(lg, 1.0, hard=True).argmax(-1)
    fl = D_(to_img(Rf), cb)
    loss_g = F.binary_cross_entropy_with_logits(fl, torch.ones_like(fl))
    ce = torch.zeros(())
    if A.ce_weight > 0:
        Rtrue = full_t_all[bi][:, layout.POS_ROW, layout.MIRROR_COL].reshape(-1, 32)
        ce = F.cross_entropy(lg.reshape(-1, NLEV), Rtrue.reshape(-1))
        loss_g = loss_g + A.ce_weight * ce
    optG.zero_grad(); loss_g.backward(); optG.step()

    if step % A.eval_every == 0 or step == A.steps:
        with torch.no_grad():
            bv = va[:1024]
            cv = cond_of(bv)
            Rtv = full_t_all[bv][:, layout.POS_ROW, layout.MIRROR_COL].reshape(-1, 32)
            lgv = G_(cv, torch.zeros(len(bv), A.zdim))
            cev = float(F.cross_entropy(lgv.reshape(-1, NLEV), Rtv.reshape(-1)))
            accv = float((lgv.argmax(-1) == Rtv).float().mean())
            Rg = F.gumbel_softmax(lgv, 1.0, hard=True).argmax(-1)
            rm = D_(full_t_all[bv], cv).squeeze(1)
            fm = D_(to_img(Rg), cv).squeeze(1)
            dacc = float((torch.cat([rm, fm]) > 0).float()
                         .eq(torch.cat([torch.ones_like(rm), torch.zeros_like(fm)])).float().mean())
            gen = F.gumbel_softmax(G_(cv, torch.randn(len(bv), A.zdim)), 1.0, hard=True).argmax(-1).numpy()
            mm = metrics(gen)
        rec = {"step": step, "ce": round(cev, 4), "acc": round(accv, 4), "d_acc": round(dacc, 4),
               "loss_d": round(float(loss_d.detach()), 4), "loss_g": round(float(loss_g.detach()), 4),
               **mm, "sec": round(time.time() - t0, 1)}
        hist.append(rec)
        print(f"  step {step:5d}  CE {cev:.4f} acc {accv:.4f}  Dacc {dacc:.3f}  "
              f"levels {mm['levels']:.2f}  nn_med {mm['nn_med']}  dups {mm['dups']}  "
              f"near_real {mm['near_real_mean']:.2f}/{mm['near_real_min']}", flush=True)

# ------------------------------------------------------------------ final output
with torch.no_grad():
    bv = va[:64]
    cv = cond_of(bv)
    gen = F.gumbel_softmax(G_(cv, torch.randn(64, A.zdim)), 1.0, hard=True).argmax(-1)
    gen_img = to_img(gen).numpy()
    real_img = full[bv]


def grid(imgs, cols=16, sc=8):
    rows_ = (len(imgs) + cols - 1) // cols
    cv2 = np.full((rows_ * 8, cols * 8), 40, dtype=np.uint8)
    for i, im in enumerate(imgs):
        r_, c_ = divmod(i, cols)
        cv2[r_*8:(r_+1)*8, c_*8:(c_+1)*8] = np.clip(np.asarray(im), 0, 255).astype(np.uint8)
    big = np.kron(cv2, np.ones((sc, sc), dtype=np.uint8))
    return Image.fromarray(big, mode="L").convert("RGB")


top = grid([LEVELS[real_img[i]] for i in range(32)], 16)
bot = grid([LEVELS[gen_img[i]] for i in range(32)], 16)
combo = Image.new("RGB", (max(top.width, bot.width), top.height + bot.height + 12), (30, 30, 30))
combo.paste(top, (0, 0)); combo.paste(bot, (0, top.height + 12))
combo.save(os.path.join(OUT, "COMPARE_real_top_generated_bottom.png"))
torch.save({"G": G_.state_dict(), "D": D_.state_dict(), "args": vars(A),
            "cond_desc": cond_desc, "cond_dim": COND_DIM},
           os.path.join(OUT, "final.pt"))
json.dump({"args": vars(A), "cond_desc": cond_desc, "cond_dim": COND_DIM,
           "nG": nG, "nD": nD, "history": hist, "seconds": round(time.time() - t0, 1)},
          open(os.path.join(OUT, "summary.json"), "w"), indent=2)
print(f"wrote {OUT}")
print("SWEEP_DONE")


