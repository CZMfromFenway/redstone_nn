#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Aggregate the sweep into one honest comparison table.

Reads every results/sweep/<tag>/summary.json and reports the same metrics for all rungs,
plus the two numbers that decide whether a rung is really generating:
  * near_real_min : Hamming distance from a generated face to the nearest REAL face
                    (0 means it reproduced a training face -> memorisation, not creation)
  * near_real_mean: average such distance; larger = more novel
"""
import glob, json, os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SW = os.path.join(ROOT, "results", "sweep")

ORDER = ["sweep_uncond", "sweep_pca4", "sweep_pca8", "sweep_pca16", "sweep_pca22",
         "sweep_half", "sweep_code3", "sweep_code5"]

rows = []
for tag in ORDER:
    p = os.path.join(SW, tag, "summary.json")
    if not os.path.exists(p):
        continue
    d = json.load(open(p, encoding="utf-8"))
    h = d.get("history", [])
    if not h:
        continue
    last = h[-1]
    rows.append({
        "tag": tag, "cond": d.get("cond_desc", "?"), "cdim": d.get("cond_dim"),
        "nG": d.get("nG"), "steps": d["args"]["steps"],
        "CE": last["ce"], "acc": last["acc"], "Dacc": last["d_acc"],
        "levels": last["levels"], "nn_med": last["nn_med"], "dups": last["dups"],
        "near_real_mean": last["near_real_mean"], "near_real_min": last["near_real_min"],
        "first_CE": h[0]["ce"], "first_acc": h[0]["acc"], "first_Dacc": h[0]["d_acc"],
    })

print("=" * 118)
print("CONDITION-STRENGTH SWEEP  (canonical layout; every rung judged with the same stick)")
print("=" * 118)
print(f"{'run':16s} {'cond_dim':>8s} {'CE':>7s} {'acc':>6s} {'Dacc':>6s} {'levels':>6s} "
      f"{'nn_med':>6s} {'dups':>5s} {'near_real_mean':>15s} {'near_real_min':>13s}")
print("-" * 118)
for r in rows:
    print(f"{r['tag']:16s} {str(r['cdim']):>8s} {r['CE']:7.4f} {r['acc']:6.4f} {r['Dacc']:6.3f} "
          f"{r['levels']:6.2f} {r['nn_med']:6d} {r['dups']:5d} {r['near_real_mean']:15.2f} "
          f"{r['near_real_min']:13d}")
print("-" * 118)

print()
print("REFERENCE POINTS")
print(f"  uniform distribution over 16 levels : CE = ln(16) = {np.log(16):.4f}, acc = 0.0625")
print(f"  generated-face nearest-real distance : 0 = a training face was reproduced")
print(f"  real faces vs the real pool          : near_real_min = 0 by definition")

print()
print("=" * 118)
print("READING")
print("=" * 118)
if rows:
    best_acc = max(rows, key=lambda r: r["acc"])
    most_novel = max(rows, key=lambda r: r["near_real_mean"])
    print(f"  best pixel accuracy : {best_acc['tag']} ({best_acc['acc']:.4f})")
    print(f"  most novel          : {most_novel['tag']} (near_real_mean "
          f"{most_novel['near_real_mean']:.2f})")
    print()
    print("  A rung is only interesting if it is BOTH accurate and novel.")
    print("  High accuracy with near_real_min = 0 means the model reproduced training")
    print("  faces, which is memorisation rather than generation.")
print()
json.dump(rows, open(os.path.join(ROOT, "results", "sweep", "COMPARISON.json"), "w"), indent=2)
print("AGGREGATE_DONE")
