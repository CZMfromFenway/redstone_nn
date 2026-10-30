#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Settle whether the 32-output student is genuinely worse than the 512 teacher.

Neighbour agreement has been the contested number: teacher 77.6%, student 73.2%, real
77.6%. But agreement conflates two different things -- how much texture there is, and
whether that texture is organised into patches or scattered as isolated pixels. A face can
have the right amount of texture and still look wrong if the changes sit on lone pixels.

So this reports, on the SAME 1000 faces for every variant:
    agreement        share of adjacent pairs with identical level
    isolated         pixels with no same-level neighbour (the "speckle" the user hates)
    blobs            connected same-level regions per face (fewer = larger patches)
    detail           mean neighbouring difference (how much texture)
and compares each against REAL, which is the only reference that matters.
"""
import json, os, sys
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


class Teacher(nn.Module):
    def __init__(self, hid=256, cdim=22):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(cdim, hid), nn.ReLU(),
                                 nn.Linear(hid, hid), nn.ReLU(),
                                 nn.Linear(hid, 32 * NLEV))
    def logits(self, c):
        return self.net(c).view(-1, 32, NLEV)


class Student(nn.Module):
    def __init__(self, hid=256, cdim=22):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(cdim, hid), nn.ReLU(),
                                 nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, 32))
    def forward(self, c):
        return self.net(c)


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
va = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_split.npz"))["val"]

NS = 1000
bv = va[:NS]


def metrics(imgs):
    ag, iso, blobs, det = [], [], [], []
    for g in imgs:
        pairs = same = 0
        i_count = 0
        for r_ in range(8):
            for c in range(8):
                ns = 0
                for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    rr, cc = r_ + dr, c + dc
                    if 0 <= rr < 8 and 0 <= cc < 8:
                        pairs += 1
                        if g[r_, cc] == g[rr, cc]:
                            same += 1
                            ns += 1
                if ns == 0:
                    i_count += 1
        ag.append(same / pairs)
        iso.append(i_count)
        # connected components of equal level
        seen = np.zeros((8, 8), bool)
        cnt = 0
        for r_ in range(8):
            for c in range(8):
                if seen[r_, c]:
                    continue
                lv = g[r_, c]
                st = [(r_, c)]
                seen[r_, c] = True
                while st:
                    rr, cc = st.pop()
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        y, x = rr + dr, cc + dc
                        if 0 <= y < 8 and 0 <= x < 8 and not seen[y, x] and g[y, x] == lv:
                            seen[y, x] = True
                            st.append((y, x))
                cnt += 1
        blobs.append(cnt)
        det.append((np.abs(np.diff(g.astype(float), axis=1)).mean()
                    + np.abs(np.diff(g.astype(float), axis=0)).mean()) / 2)
    return {"agree": float(np.mean(ag)), "iso": float(np.mean(iso)),
            "blobs": float(np.mean(blobs)), "detail": float(np.mean(det))}


print("=" * 100)
print(f"FULL METRIC SET on the same {NS} faces (real is the only reference that matters)")
print("=" * 100)
print(f"  {'variant':32s} {'acc':>7s} {'agree':>8s} {'iso':>6s} {'blobs':>7s} "
      f"{'detail':>7s} {'vs real':>9s}")
print("  " + "-" * 84)

rows = []
real_img = full[bv].reshape(-1, 8, 8)
m_real = metrics(real_img)
print(f"  {'REAL':32s} {'--':>7s} {m_real['agree']*100:7.1f}% {m_real['iso']:6.2f} "
      f"{m_real['blobs']:7.1f} {m_real['detail']:7.2f} {'':>9s}")
rows.append({"name": "REAL", **m_real})

with torch.no_grad():
    lg = T.logits(COND[bv])
    t_arg = lg.argmax(-1).numpy()
m_t = metrics(layout.image_from_left_half(t_arg))
acc_t = float((t_arg == RTRUE[bv].numpy()).mean())
print(f"  {'teacher 512 (argmax)':32s} {acc_t:7.4f} {m_t['agree']*100:7.1f}% "
      f"{m_t['iso']:6.2f} {m_t['blobs']:7.1f} {m_t['detail']:7.2f} "
      f"{(m_t['agree']-m_real['agree'])*100:+8.1f}")
rows.append({"name": "teacher 512", "acc": acc_t, **m_t})


def add_student(path, name):
    if not os.path.exists(path):
        print(f"  {name:32s} (missing)")
        return
    c = torch.load(path, weights_only=False)
    st = c.get("S") or c.get("G")
    hid = 256
    for k, v in st.items():
        if k.endswith("0.weight"):
            hid = int(v.shape[0])
            break
    # Build a BARE Sequential. The checkpoints come from three different training scripts
    # whose modules either wrap the layers in `self.net` or not; a bare Sequential plus
    # key-prefix normalisation loads both.
    state = {}
    for k, v in st.items():
        state[k.replace("net.", "", 1) if k.startswith("net.") else k] = v
    # after the "net." prefix is gone the keys look like "0.weight", "2.bias", so the
    # layer index is the part BEFORE the first dot
    lens = sorted({int(k.split(".")[0]) for k in state if k.endswith(".weight")})
    shapes = {j: state[f"{j}.weight"].shape for j in lens}
    seq = []
    prev = None
    for j in lens:
        out_dim, in_dim = int(shapes[j][0]), int(shapes[j][1])
        seq.append(nn.Linear(in_dim, out_dim))
        # a ReLU follows every layer except the last
        if j != lens[-1]:
            seq.append(nn.ReLU())
    S = nn.Sequential(*seq)
    S.load_state_dict(state)
    S.eval()
    with torch.no_grad():
        sv = S(COND[bv]).clamp(0, 15).numpy()
    dec = sv.round().astype(int)
    m = metrics(layout.image_from_left_half(dec))
    acc = float((dec == RTRUE[bv].numpy()).mean())
    print(f"  {name:32s} {acc:7.4f} {m['agree']*100:7.1f}% {m['iso']:6.2f} "
          f"{m['blobs']:7.1f} {m['detail']:7.2f} {(m['agree']-m_real['agree'])*100:+8.1f}")
    rows.append({"name": name, "acc": acc, **m})


add_student(os.path.join(ROOT, "results", "distill", "distill_plain", "final.pt"),
            "student distill (no TV)")
add_student(os.path.join(ROOT, "results", "distill", "tv_0p2", "final.pt"),
            "student distill TV 0.2")
add_student(os.path.join(ROOT, "results", "shrink", "s32_tv5", "final.pt"),
            "student hard+TV5 (earlier)")
add_student(os.path.join(ROOT, "results", "shrink", "s32_tv30", "final.pt"),
            "student hard+TV30 (earlier)")

print()
print("=" * 100)
print("VERDICT")
print("=" * 100)
print(f"  real: agreement {m_real['agree']*100:.1f}%, isolated {m_real['iso']:.2f}, "
      f"blobs {m_real['blobs']:.1f}, detail {m_real['detail']:.2f}")
print("  The question is not 'agreement below real' but 'is the gap visible':")
print("  isolated and blobs are the perceptually relevant ones -- isolated is the speckle")
print("  the user complained about, blobs says whether patches are large enough.")
json.dump({"real": m_real, "rows": rows}, open(os.path.join(OUT, "full_metrics.json"), "w"),
          indent=2)
print(f"\nwrote {os.path.join(OUT, 'full_metrics.json')}")
print("FULL_METRICS_DONE")
