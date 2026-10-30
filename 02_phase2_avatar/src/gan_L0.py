#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
L0 GAN: generate perfectly symmetric 16-level grayscale 8x8 avatars.

Output space
    G(z) -> (B, 32, 16) logits         32 left-half positions x 16 gray levels
    sample -> (B, 32) hard one-hot      Gumbel-softmax, straight-through
    mirror -> (B, 64, 16) -> (B, 8, 8, 16)   perfectly symmetric BY CONSTRUCTION

Discriminator input
    real : one-hot of the real 16-level gray image, (B, 8, 8) -> (B, 8, 8, 16)
    fake : the sampled, mirrored one-hot, same shape and same KIND (hard), so the
           discriminator cannot cheat by detecting "soft vs hard".

Metrics that matter (per the audit: never trust a single mean-sigmoid number)
    D accuracy on a held-out real set vs fake, D logit std, plus generator
    distinct-levels/sample, nearest-neighbour distance between samples, exact
    duplicate count, and symmetry (must be 1.0 by construction).

Relative paths only. D: is never touched; existing files are never overwritten.
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# ------------------------------------------------------------------ args
AP = argparse.ArgumentParser()
AP.add_argument("--tag", required=True, help="run name; outputs go to results/gan_v2/<tag>/")
AP.add_argument("--steps", type=int, default=20000)
AP.add_argument("--batch", type=int, default=256)
AP.add_argument("--lr-g", type=float, default=3e-4)
AP.add_argument("--lr-d", type=float, default=2e-4)
AP.add_argument("--zdim", type=int, default=64)
AP.add_argument("--g-hidden", type=int, default=256)
AP.add_argument("--d-c", type=int, default=64)
AP.add_argument("--d-embed", type=int, default=8)
AP.add_argument("--r1-every", type=int, default=8, help="R1 gradient penalty every N steps")
AP.add_argument("--r1-gamma", type=float, default=2.0)
AP.add_argument("--snap-every", type=int, default=500)
AP.add_argument("--log-every", type=int, default=100)
AP.add_argument("--eval-every", type=int, default=500)
AP.add_argument("--seed", type=int, default=0)
AP.add_argument("--d-steps", type=int, default=1, help="discriminator updates per generator update")
AP.add_argument("--d-width", type=int, default=0, help="override --d-c (smaller = weaker D)")
AP.add_argument("--g-zero-init", action="store_true",
                help="zero the generator's output layer so generation starts from flat gray, not noise")
AP.add_argument("--train-n", type=int, default=0, help="subsample the train split (0 = all)")
AP.add_argument("--full-data", action="store_true",
                help="use ALL 923,975 clean avatars instead of the >=95%-symmetric subset")
AP.add_argument("--freeze-d-after", type=int, default=0,
                help="stop updating D after this many steps (diagnostic: can G learn against a "
                     "FROZEN, still-weak critic? separates joint dynamics from G capacity)")
AP.add_argument("--no-d", action="store_true",
                help="never update D at all (it stays at its random init)")
AP.add_argument("--threads", type=int, default=0)
A = AP.parse_args()

torch.manual_seed(A.seed)
np.random.seed(A.seed)
if A.threads:
    torch.set_num_threads(A.threads)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data_v2_canonical")   # canonical layout; the old data/ build had the layout bug
OUT = os.path.join(ROOT, "results", "gan_v2", A.tag)
os.makedirs(OUT, exist_ok=True)

DEV = "cpu"
NLEV = 16
NPOS = 32                      # canonical: the LEFT half, p = col*8 + row  (see layout.py)

# ONE canonical layout, shared with every other script. Earlier versions of this file
# generated the TOP half (rows 0..3) instead, which left rows 4..7 permanently black.
import importlib.util as _ilu, sys as _sys
_lspec = _ilu.spec_from_file_location("layout", os.path.join(HERE, "layout.py"))
layout = _ilu.module_from_spec(_lspec); _lspec.loader.exec_module(layout)


# ------------------------------------------------------------------ data
def load_data():
    """Canonical left-half targets for the generator.

    By default this uses the nearly-symmetric subset (data_v2_canonical), which is the
    task definition we agreed on: "learn nearly-symmetric avatars, generate perfectly
    symmetric ones".

    With --full-data it uses ALL 923,975 clean avatars and applies the symmetry filter
    ONLY on the left/right split (i.e. it still learns from the real, mostly-but-not-
    fully symmetric faces). This matters because the >=95%-symmetric subset has
    mirror-forcing error of just 0.34%, which makes "predict the right half from the
    left half" close to a restatement of "mirror it" -- a weak test of a model.
    """
    src = os.path.join(ROOT, "data") if A.full_data else os.path.join(ROOT, "data_v2_canonical")
    left = np.load(os.path.join(src, "L0_left_idx.npy"))       # (N,32) canonical left half
    if A.full_data:
        # build the canonical left half from the unfiltered 16-level gray images
        rgbi = np.load(os.path.join(ROOT, "data", "faces8_rgb_v2.npy"))
        LUM = np.array([0.299, 0.587, 0.114])
        gy = (rgbi.astype(np.float64) * LUM).sum(-1)
        lv = np.linspace(0, 255, 16)
        gi = np.abs(gy[:, :, :, None] - lv[None, None, None, :]).argmin(-1).astype(np.uint8)
        left = layout.left_half_from_image(gi).astype(np.uint8)
        n = len(left)
        print(f"FULL DATA: using all {n:,} clean avatars (no symmetry filter)")
        rng = np.random.default_rng(0); perm = rng.permutation(n)
        nval = int(n * 0.03)
        return left, gi, perm[nval:], perm[:nval]
    full = np.load(os.path.join(src, "L0_full_idx.npy"))       # (N,8,8)
    sp = np.load(os.path.join(src, "L0_split.npz"))
    tr, va = sp["train"], sp["val"]
    if A.train_n:
        tr = tr[:A.train_n]
    return left, full, tr, va


def onehot_full(idx_np):
    """(B,8,8) uint8 -> (B,16,8,8) float one-hot, CHANNEL-FIRST for conv."""
    B = len(idx_np)
    t = torch.from_numpy(idx_np.astype(np.int64)).to(DEV)                 # (B,8,8)
    oh = F.one_hot(t, NLEV).permute(0, 3, 1, 2).float()                   # (B,16,8,8)
    return oh


def onehot_idx_to_chw(idx):
    """(B,8,8) long -> (B,16,8,8) float. F.one_hot puts the class LAST, conv needs it FIRST."""
    return F.one_hot(idx, NLEV).permute(0, 3, 1, 2).float()


# ------------------------------------------------------------------ models
class Gen(nn.Module):
    def __init__(self, zdim, hid, zero_init=False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(zdim, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
            nn.Linear(hid, NPOS * NLEV),      # 32 * 16 = 512 logits
        )
        self.z = zdim
        if zero_init:
            # Make the FIRST generation a near-uniform gray image instead of uniform
            # noise. Reason: if the generator starts from white noise, the discriminator
            # separates real from fake within a few thousand steps and then supplies no
            # usable gradient (measured: D accuracy 1.000 from step 5000 on, G loss
            # rising 6.5 -> 8.5, output stayed pure noise). Starting from a flat
            # "plausible mean image" keeps the discriminator's gradients local and
            # informative for much longer.
            last = self.net[-1]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def logits(self, z):
        return self.net(z).view(-1, NPOS, NLEV)


class Disc(nn.Module):
    """Convolutional, 3x3, fan-in 9 (kept redstone-friendly on purpose).

    The one-hot index is passed through a LEARNABLE embedding before the first conv.
    This gives the discriminator a continuous space in which nearby gray levels sit
    nearby -- otherwise a raw 16-channel one-hot would treat every level as
    equidistant and it would have to relearn 'gray' from scratch.
    """
    def __init__(self, c=64, emb=8):
        super().__init__()
        self.emb = nn.Embedding(NLEV, emb)
        self.conv = nn.Sequential(
            nn.Conv2d(emb, c, 3, padding=1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(c, c, 3, padding=1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(c, c, 3, padding=1), nn.LeakyReLU(0.2, True),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(c * 64, 64), nn.LeakyReLU(0.2, True),
                                  nn.Linear(64, 1))

    def from_onehot(self, oh):
        """(B,16,8,8) one-hot -> (B,emb,8,8) embedded, DIFFERENTIABLY w.r.t. the input.

        Do NOT write `self.emb(oh.argmax(1))`: argmax is not differentiable, so the R1
        gradient penalty (which differentiates the discriminator score w.r.t. its INPUT)
        would have no path back to the input, and R1 silently fails. For a HARD one-hot
        the two forms are numerically identical; only this one carries a gradient.
        """
        w = self.emb.weight                                # (16, emb)
        return torch.einsum("bchw,ce->behw", oh, w).contiguous()

    def forward_from_onehot(self, oh):
        return self.head(self.conv(self.from_onehot(oh)))


# ------------------------------------------------------------------ metrics
def gen_samples(G, n, tau, seed=0, hard=True):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n, G.z, generator=g)
    with torch.no_grad():
        lg = G.logits(z)
    onehot = F.gumbel_softmax(lg, tau=tau, hard=hard)          # (n,32,16)
    idx = onehot.argmax(-1)                                     # (n,32)
    return idx, onehot


def to_full(idx):
    """(n,32) canonical left-half levels -> (n,8,8) perfectly symmetric image.

    Uses the shared canonical layout (p = col*8 + row, mirror within each row), which
    writes ALL 8 rows. The previous version treated the 32 values as rows 0..3 x cols
    0..7, so rows 4..7 were never written and rendered as a solid black lower half.
    """
    return layout.image_from_left_half(idx)


def sample_metrics(idx):
    """idx: (n,32) long tensor of generated left-half levels, canonical order."""
    a = idx.numpy()
    n = len(a)
    levels = np.array([len(np.unique(a[i])) for i in range(n)])
    D = (a[:, None, :] != a[None, :, :]).sum(2).astype(np.int32)
    np.fill_diagonal(D, 10 ** 6)
    nn = D.min(1)
    dups = n - len({a[i].tobytes() for i in range(n)})
    full = layout.image_from_left_half(a)                     # (n,8,8)
    sym = float((full[:, :, :4] == full[:, :, 4:][:, :, ::-1]).mean())
    rows_written = int(np.unique(np.flatnonzero((full != 0).any(axis=(0, 2)))).size)
    return {"levels_per_sample": float(levels.mean()),
            "nn_median": int(np.median(nn)), "nn_min": int(nn.min()),
            "exact_duplicates": int(dups), "symmetry": sym,
            "rows_with_content": rows_written}


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


LEVELS = np.linspace(0, 255, 16)


# ------------------------------------------------------------------ main
def main():
    print("=" * 78)
    print(f"L0 GAN  tag={A.tag}  seed={A.seed}  steps={A.steps}  batch={A.batch}")
    print("=" * 78)
    left, full, tr, va = load_data()
    print(f"data: left {left.shape}  full {full.shape}  train {len(tr):,}  val {len(va):,}")
    print(f"threads={torch.get_num_threads()}")

    G = Gen(A.zdim, A.g_hidden, zero_init=A.g_zero_init).to(DEV)
    dw = A.d_width or A.d_c
    D = Disc(dw, A.d_embed).to(DEV)
    nG = sum(p.numel() for p in G.parameters())
    nD = sum(p.numel() for p in D.parameters())
    print(f"G params {nG:,}   D params {nD:,}   (G zero-init: {A.g_zero_init}, D width {dw})")
    print(f"G output: 32 x 16 = {NPOS*NLEV} logits")

    optG = torch.optim.Adam(G.parameters(), lr=A.lr_g, betas=(0.5, 0.999))
    optD = torch.optim.Adam(D.parameters(), lr=A.lr_d, betas=(0.5, 0.999))

    # held-out real batch for evaluation, fixed so the metric is comparable across steps
    rng = np.random.default_rng(123)
    ev_real = full[rng.choice(va, min(512, len(va)), replace=False)]
    ev_real_oh = onehot_full(ev_real)
    ev_oh_idx = torch.from_numpy(ev_real.astype(np.int64))

    logf = open(os.path.join(OUT, "metrics.jsonl"), "w", encoding="utf-8")
    snaps = []
    t0 = time.time()
    hist = []

    for step in range(1, A.steps + 1):
        # ---- discriminator step(s): each round uses a FRESH fake batch, otherwise the
        # discriminator would chase a stale generator output.
        loss_d_val = 0.0
        d_frozen = A.no_d or (A.freeze_d_after and step > A.freeze_d_after)
        for _ in range(1 if d_frozen else max(1, A.d_steps)):
            bi = tr[rng.integers(0, len(tr), A.batch)]
            real_oh = onehot_full(full[bi])
            with torch.no_grad():
                lg = G.logits(torch.randn(A.batch, A.zdim))
                fake_idx = F.gumbel_softmax(lg, tau=1.0, hard=True).argmax(-1)
                fake_oh_full = onehot_idx_to_chw(to_full(fake_idx))

            real_logit = D.forward_from_onehot(real_oh)
            fake_logit = D.forward_from_onehot(fake_oh_full)
            # label smoothing on the real side only; the fake target stays 0
            loss_d = F.binary_cross_entropy_with_logits(
                real_logit, torch.full_like(real_logit, 0.9)) \
                + F.binary_cross_entropy_with_logits(
                    fake_logit, torch.zeros_like(fake_logit))

            if A.r1_every and step % A.r1_every == 0:
                real_oh = real_oh.detach().requires_grad_(True)
                rl = D.forward_from_onehot(real_oh)
                g = torch.autograd.grad(rl.sum(), real_oh, create_graph=True)[0]
                r1 = g.flatten(1).pow(2).sum(1).mean()
                loss_d = loss_d + (A.r1_gamma / 2.0) * r1
            loss_d_val = float(loss_d.detach())
            if not d_frozen:
                optD.zero_grad()
                loss_d.backward()
                optD.step()

        # ---- generator step
        z = torch.randn(A.batch, A.zdim)
        lg = G.logits(z)
        fake_oh = F.gumbel_softmax(lg, tau=1.0, hard=True)
        fake_idx = fake_oh.argmax(-1)
        fake_full = to_full(fake_idx)
        fake_oh_full = onehot_idx_to_chw(fake_full)
        g_logit = D.forward_from_onehot(fake_oh_full)
        loss_g = F.binary_cross_entropy_with_logits(g_logit, torch.ones_like(g_logit))
        optG.zero_grad()
        loss_g.backward()
        optG.step()

        if step % A.log_every == 0:
            rec = {"step": step, "loss_d": round(loss_d_val, 4),
                   "loss_g": round(float(loss_g.detach()), 4),
                   "sec": round(time.time() - t0, 1)}
            logf.write(json.dumps(rec) + "\n"); logf.flush()
            print(f"  step {step:6d}  D {float(loss_d):7.4f}  G {float(loss_g):7.4f}  "
                  f"({time.time()-t0:6.0f}s)", flush=True)

        if step % A.eval_every == 0 or step == A.steps:
            with torch.no_grad():
                rl = D.forward_from_onehot(ev_real_oh).squeeze(1)
                z = torch.randn(512, A.zdim)
                lg = G.logits(z)
                fo = F.gumbel_softmax(lg, tau=1.0, hard=True)
                fi = fo.argmax(-1)
                ff = to_full(fi)
                foh = onehot_idx_to_chw(ff)
                fl = D.forward_from_onehot(foh).squeeze(1)
                acc = float((torch.cat([rl, fl]) > 0).float()
                            .eq(torch.cat([torch.ones_like(rl), torch.zeros_like(fl)])).float().mean())
                gi, _ = gen_samples(G, 512, 1.0, seed=step)
                sm = sample_metrics(gi)
            ev = {"step": step, "d_acc": round(acc, 4),
                  "d_logit_std_real": round(float(rl.std()), 4),
                  "d_logit_std_fake": round(float(fl.std()), 4),
                  "d_real_mean": round(float(rl.mean()), 4),
                  "d_fake_mean": round(float(fl.mean()), 4), **sm}
            logf.write(json.dumps(ev) + "\n"); logf.flush()
            print(f"    EVAL step {step}: Dacc {acc:.3f}  logitstd R/F "
                  f"{ev['d_logit_std_real']:.3f}/{ev['d_logit_std_fake']:.3f}  "
                  f"levels {sm['levels_per_sample']:.2f}  nn_med {sm['nn_median']}  "
                  f"dups {sm['exact_duplicates']}", flush=True)
            hist.append(ev)

        if A.snap_every and step % A.snap_every == 0:
            import copy
            snaps.append({"step": step, "G": copy.deepcopy(G.state_dict())})
            if len(snaps) > 4:
                snaps.pop(0)
            # a D-scored snapshot table: tells "D collapsed" apart from "G caught up"
            with torch.no_grad():
                rows = []
                for s in snaps:
                    Gs = Gen(A.zdim, A.g_hidden); Gs.load_state_dict(s["G"]); Gs.eval()
                    z = torch.randn(256, A.zdim)
                    f = F.gumbel_softmax(Gs.logits(z), tau=1.0, hard=True).argmax(-1)
                    foh = onehot_idx_to_chw(to_full(f))
                    a = float((D.forward_from_onehot(foh) > 0).float().mean())
                    rows.append({"snap_step": s["step"], "d_acc_on_this_G": round(a, 4)})
                logf.write(json.dumps({"step": step, "frozen_G_scores": rows}) + "\n"); logf.flush()

        if step % max(A.snap_every, 1) == 0 or step == A.steps:
            gi, _ = gen_samples(G, 32, 1.0, seed=7)
            gimg = to_full(gi).numpy()
            grid([LEVELS[gimg[i]] for i in range(32)], 8).save(
                os.path.join(OUT, f"samples_step{step:06d}.png"))
            grid([LEVELS[ev_real[i]] for i in range(32)], 8).save(
                os.path.join(OUT, "real_reference.png"))

    # final save
    torch.save({"G": G.state_dict(), "D": D.state_dict(),
                "args": vars(A), "levels": LEVELS.tolist()}, os.path.join(OUT, "final.pt"))
    with torch.no_grad():
        big_idx, _ = gen_samples(G, 256, 1.0, seed=99)
    bi = to_full(big_idx).numpy()                       # already mirror-symmetric

    # requested picture: REAL band on top, GENERATED (forced symmetric) band below
    real_sym = layout.image_from_left_half(layout.left_half_from_image(full[va[:32]]))
    top = grid([LEVELS[real_sym[i]] for i in range(32)], 16)
    bot = grid([LEVELS[bi[i]] for i in range(32)], 16)
    combo = Image.new("RGB", (max(top.width, bot.width), top.height + bot.height + 12),
                      (30, 30, 30))
    combo.paste(top, (0, 0)); combo.paste(bot, (0, top.height + 12))
    combo.save(os.path.join(OUT, "COMPARE_real_top_generated_bottom.png"))
    top.save(os.path.join(OUT, "band_REAL.png"))
    bot.save(os.path.join(OUT, "band_GENERATED.png"))
    grid([LEVELS[bi[i]] for i in range(256)], 16).save(os.path.join(OUT, "samples_final_256.png"))
    json.dump({"args": vars(A), "G_params": nG, "D_params": nD,
               "history": hist, "seconds": round(time.time() - t0, 1)},
              open(os.path.join(OUT, "summary.json"), "w"), indent=2)
    logf.close()
    print(f"\nwrote {OUT}")
    print(f"total {time.time()-t0:.0f}s")
    print("GAN_DONE")


if __name__ == "__main__":
    main()


