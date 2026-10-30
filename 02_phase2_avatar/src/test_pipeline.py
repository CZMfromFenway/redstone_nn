#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
End-to-end test of the labelling pipeline with SYNTHETIC labels.

Purpose: prove that label -> fit_reward.py -> preference_reward.json -> page
actually works BEFORE the user spends 100 clicks on it. Uses the measurable proxy to
invent grades, so the grades are meaningful (not random), and deletes everything it
created afterwards so the user's real label file starts empty.

It also checks the property that matters: after fitting, does the reward model rank
'good' above 'bad' on held-out synthetic points?
"""
import json, os, shutil, subprocess, sys
# The Windows console defaults to GBK, which cannot encode some glyphs that the child
# process may emit. Force UTF-8 on our own stdout so the test cannot die on printing.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import numpy as np
import torch
import torch.nn as nn

import importlib.util as _ilu
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
_lspec = _ilu.spec_from_file_location("layout", os.path.join(HERE, "layout.py"))
layout = _ilu.module_from_spec(_lspec); _lspec.loader.exec_module(layout)
NLEV = 16
PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
LABELS = os.path.join(ROOT, "results", "preference", "labels.jsonl")
REWARD = os.path.join(ROOT, "webapp", "preference_reward.json")

if os.path.exists(LABELS) and os.path.getsize(LABELS) > 0:
    print("REFUSING: labels.jsonl already has real labels; not touching it.")
    sys.exit(0)


class Gen(nn.Module):
    def __init__(self, hid=256, cdim=22):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(cdim, hid), nn.ReLU(),
                                 nn.Linear(hid, hid), nn.ReLU(),
                                 nn.Linear(hid, 32 * NLEV))
    def logits(self, c):
        return self.net(c).view(-1, 32, NLEV)


ck = torch.load(os.path.join(ROOT, "results", "coherence", "final_lam5_20k", "final.pt"),
                weights_only=False)
G = Gen(ck["args"]["hid"], 22); G.load_state_dict(ck["G"]); G.eval()
va = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_split.npz"))["val"]
full = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_full_idx.npy"))
POOL = full[va[:3000]].reshape(3000, -1).astype(np.int16)


def proxy(Z):
    Zt = torch.tensor(np.asarray(Z, dtype=np.float32))
    with torch.no_grad():
        dec = G.logits(Zt).argmax(-1).numpy()
    img = layout.image_from_left_half(dec).reshape(len(Z), -1).astype(np.int16)
    d = np.empty(len(img))
    for s in range(0, len(img), 128):
        d[s:s + 128] = (img[s:s + 128][:, None, :] != POOL[None, :, :]).sum(2).min(1)
    return -d


rng = np.random.default_rng(7)
Z = np.clip(rng.standard_normal((60, 22)), -3.5, 3.5)
sc = proxy(Z)
# map the proxy onto 0..3 grades so the distribution looks like a real session
q = np.argsort(np.argsort(sc)) / (len(sc) - 1)
grade = np.digitize(q, [0.35, 0.65, 0.85])
os.makedirs(os.path.dirname(LABELS), exist_ok=True)
with open(LABELS, "w", encoding="utf-8") as f:
    for i in range(len(Z)):
        f.write(json.dumps({"ts": "synthetic", "model": "final_lam5_20k",
                            "grade": int(grade[i]),
                            "pca": [round(float(v), 4) for v in Z[i]],
                            "note": "SYNTHETIC TEST", "predicted_levels": None}) + "\n")
counts = np.bincount(grade, minlength=4)
print(f"wrote {len(Z)} synthetic labels, grade counts 差/中/良/优 = "
      f"{counts[0]}/{counts[1]}/{counts[2]}/{counts[3]}")

# ---- run the fitter -----------------------------------------------------------------
print("\n--- running fit_reward.py --write ---")
r = subprocess.run([PY, os.path.join(HERE, "fit_reward.py"), "--write",
                    "--min-labels", "14"],
                   capture_output=True, text=True, encoding="utf-8", errors="replace")
print(r.stdout[-4000:] if r.stdout else "(no stdout)")
if r.stderr:
    print("STDERR:", r.stderr[-2000:])
ok = r.returncode == 0 and os.path.exists(REWARD)

# ---- validate the export and its ranking quality ------------------------------------
if ok:
    R = json.load(open(REWARD, encoding="utf-8"))
    print("\n--- export validation ---")
    print(f"  n_labels {R['n_labels']}  counts {R['counts']}")
    print(f"  W1 {len(R['W1'])}x{len(R['W1'][0])}  W2 {len(R['W2'])}x{len(R['W2'][0])}  "
          f"W3 {len(R['W3'])}x{len(R['W3'][0])}")
    assert len(R["W1"]) == 22, "W1 must start at 22 inputs"
    assert len(R["zmean"]) == 22 and len(R["zsd"]) == 22

    def rscore(Zq):
        zs = (np.asarray(Zq) - np.array(R["zmean"])) / np.array(R["zsd"])
        a = np.maximum(zs @ np.array(R["W1"]) + np.array(R["b1"]), 0)
        b = np.maximum(a @ np.array(R["W2"]) + np.array(R["b2"]), 0)
        return (b @ np.array(R["W3"]) + np.array(R["b3"])).ravel()

    # held-out check: does the model rank new good points above new bad ones?
    Zt = np.clip(rng.standard_normal((400, 22)), -3.5, 3.5)
    st = proxy(Zt)
    pred = rscore(Zt)
    top = np.argsort(-st)[:80]
    bot = np.argsort(st)[:80]
    mtop, mbot = pred[top].mean(), pred[bot].mean()
    sep = float(mtop - mbot)
    print(f"  HELD-OUT separation: predicted score of the truly-best 80 = {mtop:.3f}, "
          f"of the truly-worst 80 = {mbot:.3f}  (gap {sep:+.3f})")
    # rank correlation as a scalar summary
    rk = np.argsort(np.argsort(pred)); rt = np.argsort(np.argsort(st))
    corr = float(np.corrcoef(rk, rt)[0, 1])
    print(f"  Spearman rank correlation on held-out points: {corr:+.3f}")
    print(f"  PASS pipeline works: label -> reward model -> JSON -> page"
          if sep > 0 else "  FAIL reward model does not separate held-out good from bad")
    ok = ok and sep > 0

# ---- clean up so the user starts from zero -------------------------------------------
for p in (LABELS, REWARD):
    if os.path.exists(p):
        os.remove(p)
print(f"\ncleaned up: labels.jsonl exists={os.path.exists(LABELS)} "
      f"preference_reward.json exists={os.path.exists(REWARD)}")
print("PIPELINE_TEST_DONE" if ok else "PIPELINE_TEST_FAILED")
