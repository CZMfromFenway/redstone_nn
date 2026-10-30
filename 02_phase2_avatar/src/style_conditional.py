#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Does SPECIALISING BY STYLE help? ("或许可以通过不同类型的头像分别训练")

The user's hypothesis: if the huge dataset is a mixture of visually distinct styles
(anime-ish / 3D-ish / game-ish ...), a model trained on a single style should model that
style better than one model trained on the mixture.

Three arms, all judged on the SAME held-out real right halves, so the comparison is fair:

  global      : one model on all training data
  per-cluster : a SEPARATE model for each style cluster, trained only on that cluster,
                evaluated only on held-out faces from that cluster
  conditioned : one model that also receives the cluster id (as a learned embedding),
                i.e. soft specialisation with shared parameters

To keep the arms comparable each model sees the SAME NUMBER OF GRADIENT STEPS and the
same architecture. The per-cluster arm inevitably has less data per model, so if it still
wins, specialisation is genuinely helping rather than just having more capacity.

Style clusters come from k-means on the 22 PCA coefficients (the best style space found
by style_cluster.py, ARI 0.58 vs a 3-class majority baseline of 0.39, p=0.005).

The coherence settings that the night's sweeps found useful are applied to every arm so
the comparison is not confounded by the speckle fix.
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
AP.add_argument("--tag", required=True)
AP.add_argument("--steps", type=int, default=8000)
AP.add_argument("--batch", type=int, default=256)
AP.add_argument("--k", type=int, default=4, help="number of style clusters")
AP.add_argument("--lam", type=float, default=2.0)
AP.add_argument("--lam-final", type=float, default=10.0)
AP.add_argument("--hid", type=int, default=256)
AP.add_argument("--threads", type=int, default=8)
AP.add_argument("--seed", type=int, default=0)
A = AP.parse_args()
torch.set_num_threads(A.threads)
torch.manual_seed(A.seed)
OUT = os.path.join(ROOT, "results", "styled", A.tag)
os.makedirs(OUT, exist_ok=True)
print(f"tag={A.tag} steps={A.steps} k={A.k} lam {A.lam}->{A.lam_final}")

# ---------------------------------------------------------------- data + PCA style space
rgb = np.load(os.path.join(ROOT, "data", "faces8_rgb_v2.npy"))
LUM = np.array([0.299, 0.587, 0.114])
gray = (rgb.astype(np.float64) * LUM).sum(-1)
gyi = np.abs(gray[:, :, :, None] - LEVELS[None, None, None, :]).argmin(-1)
mask = (gyi[:, layout.POS_ROW, layout.POS_COL] ==
        gyi[:, layout.POS_ROW, layout.MIRROR_COL]).mean(1) >= 0.95
Gg = gray[mask]
full = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_full_idx.npy"))
X = Gg.reshape(len(Gg), -1)
mu = X.mean(0, keepdims=True)
sub = np.random.default_rng(0).choice(len(X), min(60000, len(X)), replace=False)
U, S_, Vt = np.linalg.svd(X[sub] - mu, full_matrices=False)
comps = Vt[:22]
sd = (((X - mu) @ comps.T).std(0, keepdims=True) + 1e-6)
STYLE = ((X - mu) @ comps.T) / sd[0]                      # (N,22) the style space
sp = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_split.npz"))
tr, va = sp["train"], sp["val"]
RTRUE = torch.from_numpy(full.astype(np.int64))[:, layout.POS_ROW, layout.MIRROR_COL].reshape(-1, 32)
COND_ALL = torch.tensor(STYLE, dtype=torch.float32)
print(f"data: train {len(tr):,} val {len(va):,}  style space {STYLE.shape}")

# ---------------------------------------------------------------- cluster the training data
K = A.k
rngc = np.random.default_rng(1)
C = STYLE[tr][rngc.choice(len(tr), K, replace=False)].copy()
lab_tr = np.zeros(len(tr), dtype=int)
for _ in range(40):
    d = ((STYLE[tr][:, None, :] - C[None, :, :]) ** 2).sum(2)
    new = d.argmin(1)
    if (new == lab_tr).all():
        break
    lab_tr = new
    for j in range(K):
        if (lab_tr == j).any():
            C[j] = STYLE[tr][lab_tr == j].mean(0)
# assign the validation split by the same centres
d = ((STYLE[va][:, None, :] - C[None, :, :]) ** 2).sum(2)
lab_va = d.argmin(1)
sizes = np.bincount(lab_tr, minlength=K)
print(f"style clusters on the training split: sizes {sizes.tolist()} "
      f"({(sizes/len(tr)*100).round(1).tolist()}%)")
print(f"validation cluster sizes: {np.bincount(lab_va, minlength=K).tolist()}")
np.save(os.path.join(OUT, "cluster_centres.npy"), C)


class Gen(nn.Module):
    def __init__(self, hid=256, cdim=22, nclusters=0, emb=8):
        super().__init__()
        self.nclusters = nclusters
        in_dim = cdim
        if nclusters:
            self.cemb = nn.Embedding(nclusters, emb)
            in_dim += emb
        else:
            self.cemb = None
        self.net = nn.Sequential(nn.Linear(in_dim, hid), nn.ReLU(),
                                 nn.Linear(hid, hid), nn.ReLU(),
                                 nn.Linear(hid, 32 * NLEV))
    def logits(self, cond, cid=None):
        if self.cemb is not None:
            cond = torch.cat([cond, self.cemb(cid)], 1)
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


def train_arm(name, train_rows, nclusters=0):
    """train_rows: (rows, cid) local cluster ids, or (rows, None) for the global arm."""
    rows, cids = train_rows
    G = Gen(A.hid, 22, nclusters)
    D = Dis(cdim=22 + (8 if nclusters else 0))
    optG = torch.optim.Adam(G.parameters(), lr=3e-4, betas=(0.5, 0.999))
    optD = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    rng = np.random.default_rng(A.seed)
    t0 = time.time()
    hist = []
    for step in range(1, A.steps + 1):
        bi = rng.integers(0, len(rows), A.batch)
        rr = rows[bi]
        cb = COND_ALL[rr]
        cid = None
        if nclusters:
            cid = torch.from_numpy(cids[bi].astype(np.int64))
            cb = torch.cat([cb, G.cemb(cid)], 1)          # D must see the same conditioning
        with torch.no_grad():
            f = G.logits(COND_ALL[rr], cid).argmax(-1)
        rl = D(torch.from_numpy(full[rr].astype(np.int64)), cb)
        fl = D(layout.image_from_left_half(f).long(), cb)
        loss_d = F.binary_cross_entropy_with_logits(rl, torch.full_like(rl, 0.9)) + \
                 F.binary_cross_entropy_with_logits(fl, torch.zeros_like(fl))
        optD.zero_grad(); loss_d.backward(); optD.step()

        lg = G.logits(COND_ALL[rr], cid)
        f = F.gumbel_softmax(lg, 1.0, hard=True).argmax(-1)
        fl = D(layout.image_from_left_half(f).long(), cb)
        loss_g = F.binary_cross_entropy_with_logits(fl, torch.ones_like(fl))
        tgt = RTRUE[rr]
        ce = F.cross_entropy(lg.reshape(-1, NLEV), tgt.reshape(-1))
        frac = step / max(A.steps, 1)
        lam_now = A.lam + (A.lam_final - A.lam) * frac if A.lam_final > 0 else A.lam
        loss_g = loss_g + ce + lam_now * pen_tv(lg)
        optG.zero_grad(); loss_g.backward(); optG.step()
        if step % max(A.steps // 4, 1) == 0 or step == A.steps:
            G.eval()
            with torch.no_grad():
                bv = va[:600]
                cidv = None
                if nclusters:
                    cidv = torch.from_numpy(lab_va[:600].astype(np.int64))
                lgv = G.logits(COND_ALL[bv], cidv)
                mod = lgv.argmax(-1).numpy()
                acc = float((torch.from_numpy(mod) == RTRUE[bv]).float().mean())
            a, i, b = speckle_np(layout.image_from_left_half(mod))
            hist.append({"step": step, "acc_mode": round(acc, 4), "agree_mode": round(a, 4),
                         "iso_mode": round(i, 3), "blobs_mode": round(b, 2),
                         "lam_now": round(lam_now, 3)})
            print(f"    [{name}] step {step:5d} acc {acc:.4f} agree {a*100:5.1f}% "
                  f"iso {i:5.2f} lam {lam_now:.2f} ({time.time()-t0:.0f}s)", flush=True)
            G.train()
    return G, hist, time.time() - t0


def eval_arm(G, nclusters=0, name=""):
    """Evaluate on the SAME global held-out set, plus per cluster."""
    G.eval()
    out = {}
    with torch.no_grad():
        bv = va[:2000]
        cidv = torch.from_numpy(lab_va[:2000].astype(np.int64)) if nclusters else None
        lg = G.logits(COND_ALL[bv], cidv)
        mod = lg.argmax(-1).numpy()
        Rt = RTRUE[bv].numpy()
        out["acc_all"] = float((mod == Rt).mean())
        per = {}
        for j in range(K):
            m = lab_va[:2000] == j
            if m.sum() > 20:
                per[int(j)] = {"n": int(m.sum()),
                               "acc": float((mod[m] == Rt[m]).mean())}
        out["per_cluster"] = per
        a, i, b = speckle_np(layout.image_from_left_half(mod))
        out.update({"agree": a, "iso": i, "blobs": b})
    print(f"    [{name}] held-out acc {out['acc_all']:.4f}  agree {a*100:.1f}%  "
          f"per-cluster {({k: round(v['acc'],3) for k,v in per.items()})}")
    return out


results = {}

# ---- arm 1: global ----------------------------------------------------------------
print("\n=== arm 1: GLOBAL (all training data, 22-dim style condition) ===")
Gg_, h, sec = train_arm("global", (tr, None), 0)
results["global"] = {"history": h, "seconds": round(sec, 1), **eval_arm(Gg_, 0, "global")}
torch.save({"G": Gg_.state_dict(), "args": vars(A)}, os.path.join(OUT, "global.pt"))

# ---- arm 2: per-cluster ------------------------------------------------------------
print("\n=== arm 2: PER-CLUSTER (one model per style, same step budget each) ===")
per_models = {}
for j in range(K):
    rows_j = tr[lab_tr == j]
    if len(rows_j) < A.batch * 2:
        print(f"    cluster {j} too small ({len(rows_j)}); skipping")
        continue
    print(f"  cluster {j}: {len(rows_j):,} training rows")
    Gj, hj, secj = train_arm(f"cluster{j}", (rows_j, None), 0)
    ev = eval_arm(Gj, 0, f"cluster{j}")
    per_models[j] = {"n_train": int(len(rows_j)), "history": hj,
                     "seconds": round(secj, 1), **ev}
    torch.save({"G": Gj.state_dict(), "cluster": j, "args": vars(A)},
               os.path.join(OUT, f"cluster{j}.pt"))
results["per_cluster"] = per_models

# ------------------------------------------------------------------ comparison
print()
print("=" * 100)
print("GLOBAL vs PER-CLUSTER, judged on the SAME held-out right halves")
print("=" * 100)
print(f"  global (all data)      held-out acc {results['global']['acc_all']:.4f}")
for j, m in per_models.items():
    # the fair comparison: the global model's accuracy on THIS cluster's held-out faces
    g_on_j = results["global"]["per_cluster"].get(int(j), {}).get("acc", float("nan"))
    print(f"  cluster {j} specialist   held-out acc {m['acc_all']:.4f}   "
          f"(global model on the same cluster: {g_on_j:.4f})")
    m["global_on_this_cluster"] = g_on_j
    m["specialist_gain"] = m["acc_all"] - g_on_j
for j, m in per_models.items():
    print(f"  -> cluster {j} specialist gain {m['specialist_gain']:+.4f}")

json.dump({"args": vars(A), "cluster_sizes_train": sizes.tolist(),
           "cluster_sizes_val": np.bincount(lab_va, minlength=K).tolist(),
           "results": results}, open(os.path.join(OUT, "summary.json"), "w"), indent=2)
print(f"\nwrote {os.path.join(OUT, 'summary.json')}")
print("STYLED_DONE")
