#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Build plan.json from the ALREADY-MEASURED timings.json, against the live clock.

This exists because measure_then_plan.py crashed on a format bug AFTER the pilots had
finished and timings.json had been written, so the expensive measurement does not need
repeating -- only the planning step. Costs come from timings.json (real measurements),
never from assumptions.
"""
import argparse, json, os, time

AP = argparse.ArgumentParser()
AP.add_argument("--deadline", default="21:00")
AP.add_argument("--reserve", type=int, default=150)
AP.add_argument("--safety", type=float, default=1.0)
A = AP.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BUD = os.path.join(ROOT, "results", "_budget")

ORDER = ["sweep_uncond", "sweep_pca4", "sweep_pca8", "sweep_pca16", "sweep_pca22",
         "sweep_half", "sweep_code3", "sweep_code5", "kernel_style"]


def now_sec():
    t = time.localtime()
    return t.tm_hour * 3600 + t.tm_min * 60 + t.tm_sec


def dl_sec():
    h, m = (int(x) for x in A.deadline.split(":"))
    d = h * 3600 + m * 60
    if d <= now_sec():
        d += 86400
    return d


timings = json.load(open(os.path.join(BUD, "timings.json"), encoding="utf-8"))
left = dl_sec() - now_sec()
budget = left - A.reserve

print(f"clock now : {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"deadline  : {A.deadline}")
print(f"time left : {left/60:.1f} min   usable after {A.reserve}s reserve: {budget/60:.1f} min")
print()
print(f"{'label':16s} {'measured s/step':>16s} {'steps':>6s} {'cost':>10s}  verdict")
print("-" * 68)

plan, skipped = [], []
for label in ORDER:
    t = timings.get(label, {})
    if "measured_cost_seconds" not in t:
        skipped.append({"label": label, "why": "no measurement"}); continue
    steps = t["wanted_steps"]
    cost = t["measured_cost_seconds"] * A.safety
    fits = cost <= budget
    print(f"{label:16s} {t['seconds_per_step']:16.4f} {steps:6d} {cost/60:9.1f}m  "
          f"{'RUN' if fits else 'SKIP (no room)'}")
    if fits:
        plan.append({"label": label, "cost_seconds": round(cost, 1), "steps": steps})
        budget -= cost
    else:
        skipped.append({"label": label, "why": f"needs {cost/60:.1f} min, "
                                               f"only {budget/60:.1f} min left"})

fin = int(now_sec() + sum(p["cost_seconds"] for p in plan))
print("-" * 68)
print(f"planned {len(plan)} experiments, total {sum(p['cost_seconds'] for p in plan)/60:.1f} min")
print(f"predicted finish ~ {fin//3600:02d}:{(fin%3600)//60:02d}   (deadline {A.deadline})")
for s in skipped:
    print(f"  SKIP {s['label']}: {s['why']}")

json.dump({"clock_at_plan": time.strftime("%Y-%m-%d %H:%M:%S"), "deadline": A.deadline,
           "seconds_left": int(left), "reserve": A.reserve, "safety": A.safety,
           "source": "timings.json (real pilot measurements)",
           "plan": plan, "skipped": skipped,
           "predicted_finish_seconds_after_midnight": fin},
          open(os.path.join(BUD, "plan.json"), "w", encoding="utf-8"), indent=2)
print(f"\nwrote {os.path.join(BUD, 'plan.json')}")
