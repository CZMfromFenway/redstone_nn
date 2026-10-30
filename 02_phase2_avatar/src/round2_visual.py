#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Show the user their OWN round-2 data: the faces drawn from pure noise versus the faces
drawn from the purified sampler, each labelled with the grade they gave it.

This is not a simulation. It re-renders the exact 22-dim vectors that were recorded with
each label, so the two rows are literally the faces that were judged, in the order they
were judged. The grade is written under each face so the pattern is visible directly.

If purification works, the second row should show visibly more 良 and fewer 差. If it had
destroyed diversity, the second row would look like one face repeated.
"""
import json, os, sys
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont

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
OUT = os.path.join(ROOT, "results", "preference")
os.makedirs(OUT, exist_ok=True)


def font(sz):
    for c in ("msyh.ttc", "simhei.ttf", "simsun.ttc", "Deng.ttf"):
        p = os.path.join("C:\\Windows\\Fonts", c)
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, sz)
            except Exception:
                pass
    return ImageFont.load_default()


def load_model(path):
    c = torch.load(path, weights_only=False)
    st = c.get("G") or c.get("S")
    norm = {k.replace("net.", "", 1) if k.startswith("net.") else k: v
            for k, v in st.items()}
    idx = sorted({int(k.split(".")[0]) for k in norm if k.endswith(".weight")})
    seq = []
    for j in idx:
        W = norm[f"{j}.weight"]
        seq.append(nn.Linear(int(W.shape[1]), int(W.shape[0])))
        if j != idx[-1]:
            seq.append(nn.ReLU())
    m = nn.Sequential(*seq)
    m.load_state_dict(norm)
    m.eval()
    return m


M = load_model(os.path.join(ROOT, "results", "shrink5", "ladder_128_128.pt"))

recs = []
for line in open(os.path.join(ROOT, "results", "preference", "labels.jsonl"), encoding="utf-8"):
    line = line.strip()
    if not line:
        continue
    r = json.loads(line)
    if r.get("model") == "w128" and isinstance(r.get("pca"), list):
        r["_s"] = (r.get("note") or "").strip()
        recs.append(r)

E = [r for r in recs if r["_s"] == "explore"]
G = [r for r in recs if r["_s"] == "guided"]
print(f"explore {len(E)}  guided {len(G)}")

GRADE_CN = {0: "差", 1: "中", 2: "良", 3: "优"}
GRADE_COL = {0: (224, 122, 95), 1: (224, 192, 90), 2: (168, 212, 106), 3: (95, 214, 138)}


def render(rs):
    Z = np.array([r["pca"] for r in rs], dtype=np.float32)
    with torch.no_grad():
        out = M(torch.tensor(Z)).clamp(0, 15)
    dec = out.round().numpy().astype(int)
    return layout.image_from_left_half(dec)


def sheet(rows, cols, tile, title):
    pad, lab = 8, 26
    n = max(len(r[1]) for r in rows)
    W = cols * (tile + pad) + pad
    H = 34 + len(rows) * (tile + pad + lab + 16) + pad
    cv = Image.new("RGB", (W, H), (22, 24, 28))
    dr = ImageDraw.Draw(cv)
    dr.text((pad, 8), title, font=font(17), fill=(230, 234, 242))
    y = 34
    for name, imgs, grades in rows:
        dr.text((pad, y), f"{name}   n={len(imgs)}", font=font(14), fill=(180, 192, 208))
        y += 20
        for c in range(min(cols, len(imgs))):
            g = np.clip(imgs[c] / 15.0 * 255.0, 0, 255)
            t = np.kron(g, np.ones((tile // 8, tile // 8)))
            cv.paste(Image.fromarray(t.astype(np.uint8), mode="L").convert("RGB"),
                     (pad + c * (tile + pad), y))
            # the grade the user gave it
            dr.rectangle([pad + c * (tile + pad), y + tile,
                          pad + c * (tile + pad) + tile, y + tile + 14],
                         fill=(30, 33, 39))
            dr.text((pad + c * (tile + pad) + 2, y + tile),
                    GRADE_CN[grades[c]], font=font(12), fill=GRADE_COL[grades[c]])
        y += tile + 16 + lab
    return cv


# take the first N of each so the reading order matches the session
N = 14
rows = [("explore（纯随机对照）", render(E[:N]), [r["grade"] for r in E[:N]]),
        ("guided（纯化采样）", render(G[:N]), [r["grade"] for r in G[:N]])]
cv2 = sheet(rows, N, 56, "你自己第二轮的数据：随机 vs 纯化（下排小字是你给的评分）")
p = os.path.join(OUT, "ROUND2_YOUR_DATA.png")
cv2.save(p)
print(f"wrote {p}  {cv2.size}")

# summary of the two subsets, printed next to the figure
print()
print(f"  explore 优+良 {sum(1 for r in E if r['grade']>=2)}/{len(E)} = "
      f"{100*sum(1 for r in E if r['grade']>=2)/len(E):.1f}%")
print(f"  guided  优+良 {sum(1 for r in G if r['grade']>=2)}/{len(G)} = "
      f"{100*sum(1 for r in G if r['grade']>=2)/len(G):.1f}%")
print("ROUND2_VISUAL_DONE")
