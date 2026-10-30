#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
"""
Phase-2 restart: independent baseline probe (no neural nets, numpy only)
========================================================================
Purpose: replace "trust the report" with measured numbers, and test one
hypothesis the old work never isolated:

  H1: the sampled mixture generator and the *deterministic mode* of the same
      neighborhood distribution differ only in blurriness, not in structure?

Also measures, on the same protocol, the four metrics the old docs quote
(colors / diversity / symmetry / distance-to-nearest-real) for:
  A. real holdout faces                      (reference)
  B. mixture neighbor, symmetric, T=30       (old "best success")
  C. deterministic neighbor mode (per position argmax over the k-NN colors)
  D. same as C but forcing global symmetry decision consistent per pair

Writes only: results/restart_probe.json and results/restart_probe_*.png
"""
import os, json
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# The phase-1 attempt assets still live under archive_phase1_attempts/; prefer a local data/ if present.
ARCH = os.path.join(ROOT, "archive_phase1_attempts")
DATA = os.path.join(ROOT, "data")
if not os.path.exists(os.path.join(DATA, "faces8.npy")):
    DATA = os.path.join(ARCH, "data")
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)
C = 16

PAIRS = np.array([(r * 8 + c, r * 8 + (7 - c)) for r in range(8) for c in range(4)])
LIDX, RIDX = PAIRS[:, 0], PAIRS[:, 1]
POOL = 20000
NIMG = 64
K_NB = 30


def grid(imgs, cols=16, sc=8, pal=None):
    rows = (len(imgs) + cols - 1) // cols
    cv = np.full((rows * 8, cols * 8, 3), 40, dtype=np.uint8)
    for i, l in enumerate(imgs):
        r, c = divmod(i, cols)
        cv[r * 8:(r + 1) * 8, c * 8:(c + 1) * 8] = pal[l].reshape(8, 8, 3)
    big = np.kron(cv, np.ones((sc, sc, 1), dtype=np.uint8))
    for r in range(rows + 1):
        big[r * 8 * sc:r * 8 * sc + 2, :] = 200
    for c in range(cols + 1):
        big[:, c * 8 * sc:c * 8 * sc + 2] = 200
    return big


def metrics(S, P):
    """S:(N,64) generated, P:(M,64) reference pool for novelty distance."""
    uniq = float(np.mean([len(np.unique(s)) for s in S]))
    A = S[:, None, :] != S[None, :, :]
    div = float(A.sum() / (len(S) * (len(S) - 1)))
    sym = float((S[:, LIDX] == S[:, RIDX]).mean())
    # distance to nearest real face
    dd = []
    for i in range(0, len(S), 16):
        chunk = S[i:i + 16]
        d = (chunk[:, None, :] != P[None, :, :]).sum(2).min(1)
        dd.extend(d.tolist())
    # nearest-neighbour distance between generated samples (mode-collapse detector):
    # a median of 0 means at least half the samples are exact duplicates of another sample.
    D = (S[:, None, :] != S[None, :, :]).sum(2).astype(np.int32)
    np.fill_diagonal(D, 10 ** 6)
    nn = D.min(1)
    return dict(colors=round(uniq, 3), diversity=round(div, 2), sym=round(sym, 3),
                near_real=round(float(np.mean(dd)), 1),
                nn_gen_median=int(np.median(nn)),
                nn_gen_min=int(nn.min()),
                uniq_exact=int(len({s.tobytes() for s in S})))


def knn(pool, a, k):
    d = np.empty(len(pool), dtype=np.int32)
    for s in range(0, len(pool), 5000):
        d[s:s + 5000] = (pool[s:s + 5000] != pool[a]).sum(1)
    nb = np.argpartition(d, k)[:k]
    return nb, d[nb]


def main():
    rng0 = np.random.default_rng(0)
    lab = np.load(os.path.join(DATA, 'faces8.npy')).astype(np.int64)
    pal = np.array(json.load(open(os.path.join(DATA, 'palette_avatar16.json')))['palette_ordered'], dtype=np.uint8)
    perm = rng0.permutation(len(lab))
    hold = lab[perm[:NIMG]]                      # real reference samples
    pool = lab[perm[NIMG:NIMG + POOL]]
    print(f"faces8 {lab.shape}  pool {pool.shape}  holdout {hold.shape}")

    real_sym = float((pool[:, LIDX] == pool[:, RIDX]).mean())
    print(f"measure: real symmetry rate = {real_sym:.4f}")

    r = np.random.default_rng(1)
    anchors = r.choice(POOL, NIMG, replace=False)

    # ---- B: sampled mixture, symmetric, T=30 (old best) ----
    B = np.zeros((NIMG, 64), dtype=np.int64)
    # ---- C: deterministic mode of the same weighted k-NN distribution ----
    Cm = np.zeros((NIMG, 64), dtype=np.int64)
    # ---- D: deterministic, pair-shared context (left-half context) ----
    Dm = np.zeros((NIMG, 64), dtype=np.int64)
    # ---- E: sampled, T=1 (low temperature => closer to mode) ----
    E = np.zeros((NIMG, 64), dtype=np.int64)

    for i, a in enumerate(anchors):
        nb, dist = knn(pool, a, K_NB)
        vals = pool[nb]                                     # (k,64)
        wT30 = np.exp(-dist / 30.0); wT30 /= wT30.sum()
        wT1 = np.exp(-dist / 1.0); wT1 /= wT1.sum()
        # sampled, pair-symmetric
        cum = np.cumsum(wT30)[None, :]
        pick = np.clip((r.random(32)[:, None] > cum).sum(1), 0, K_NB - 1)
        colL = vals[pick, LIDX]
        B[i, LIDX] = colL; B[i, RIDX] = colL
        cum1 = np.cumsum(wT1)[None, :]
        pick1 = np.clip((r.random(32)[:, None] > cum1).sum(1), 0, K_NB - 1)
        colL1 = vals[pick1, LIDX]
        E[i, LIDX] = colL1; E[i, RIDX] = colL1
        # deterministic per-position weighted mode, independently per position
        for pos in range(64):
            cnt = np.bincount(vals[:, pos], weights=wT30, minlength=C)
            Cm[i, pos] = int(cnt.argmax())
        # deterministic on left half then mirror (pair decision from pair context)
        for pi, (pl, pr) in enumerate(PAIRS):
            cnt = np.bincount(vals[:, pl], weights=wT30, minlength=C)
            v = int(cnt.argmax())
            Dm[i, pl] = v; Dm[i, pr] = v
        if (i + 1) % 16 == 0:
            print(f"  generated {i+1}/{NIMG}")

    rows = {}
    for name, S in (('real_holdout', hold), ('mixture_sym_T30', B), ('mixture_sym_T1', E),
                    ('mode_permposition', Cm), ('mode_pairmirror', Dm)):
        m = metrics(S, pool)
        rows[name] = m
        print(f"  {name:20s} {m}")
        Image.fromarray(grid(S, 16, 8, pal)).save(os.path.join(RES, f'restart_probe_{name}.png'))

    out = dict(protocol=dict(pool=POOL, nimg=NIMG, k=K_NB, real_symmetry=round(real_sym, 4)),
               results=rows)
    json.dump(out, open(os.path.join(RES, 'restart_probe.json'), 'w'), indent=2)
    print("saved restart_probe.json + restart_probe_*.png")


if __name__ == '__main__':
    main()
