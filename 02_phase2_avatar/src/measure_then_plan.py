#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Measure the ACTUAL cost of every experiment, then plan against the real clock.

Nothing here is guessed. For each experiment configuration we time a real pilot run,
derive seconds-per-step from that measurement, extrapolate to the full step count, and
record the measurement. The planner then compares those MEASURED costs against the
remaining wall-clock time read from the system clock.

Outputs (in results/_budget/):
    timings.json   the measured seconds-per-step for every configuration
    plan.json      what will actually run, with measured costs and a finish-time
                   prediction, plus what is skipped for lack of time
"""
import argparse, json, os, subprocess, sys, time, statistics

AP = argparse.ArgumentParser()
AP.add_argument("--deadline", required=True, help="HH:MM local wall-clock deadline")
AP.add_argument("--pilot-steps", type=int, default=120)
AP.add_argument("--threads", type=int, default=8)
AP.add_argument("--safety", type=float, default=1.15, help="multiply measured cost by this")
AP.add_argument("--reserve", type=int, default=180, help="seconds to keep free at the end")
A = AP.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
BUD = os.path.join(ROOT, "results", "_budget")
os.makedirs(BUD, exist_ok=True)


# ---------------------------------------------------------------- real clock helpers
def clock_str():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def now_seconds():
    """Seconds since local midnight, read from the system clock."""
    t = time.localtime()
    return t.tm_hour * 3600 + t.tm_min * 60 + t.tm_sec


def deadline_seconds():
    h, m = A.deadline.split(":")
    d = int(h) * 3600 + int(m) * 60
    now = now_seconds()
    if d <= now:                     # deadline is tomorrow if already past
        d += 24 * 3600
    return d


def seconds_left():
    return deadline_seconds() - now_seconds()


print(f"clock now      : {clock_str()}")
print(f"deadline       : {A.deadline}  ->  {seconds_left()/60:.1f} min from now")

SWEEP = os.path.join(HERE, "sweep_condition.py")
KERN = os.path.join(HERE, "kernel_style.py")

# full step counts we WANT to run
FULL_STEPS_SWEEP = 1500
FULL_STEPS_KERN = 4000

# (label, script, args-without-steps, wanted_steps)
CONFIGS = [
    ("sweep_uncond", SWEEP, ["--mode", "uncond"], FULL_STEPS_SWEEP),
    ("sweep_pca4",   SWEEP, ["--mode", "pca", "--pca-k", "4"], FULL_STEPS_SWEEP),
    ("sweep_pca8",   SWEEP, ["--mode", "pca", "--pca-k", "8"], FULL_STEPS_SWEEP),
    ("sweep_pca16",  SWEEP, ["--mode", "pca", "--pca-k", "16"], FULL_STEPS_SWEEP),
    ("sweep_pca22",  SWEEP, ["--mode", "pca", "--pca-k", "22"], FULL_STEPS_SWEEP),
    ("sweep_half",   SWEEP, ["--mode", "half"], FULL_STEPS_SWEEP),
    ("sweep_code3",  SWEEP, ["--mode", "code", "--code-k", "3"], FULL_STEPS_SWEEP),
    ("sweep_code5",  SWEEP, ["--mode", "code", "--code-k", "5"], FULL_STEPS_SWEEP),
    ("kernel_style", KERN, ["--embed", "9", "--code-k", "4"], FULL_STEPS_KERN),
]

# ---------------------------------------------------------------- pilot: measure
print(f"\n=== pilot runs: {A.pilot_steps} steps each, to MEASURE the rate ===")
timings = {}
for label, script, extra, want in CONFIGS:
    steps = A.pilot_steps
    tag = f"pilot_{label}"
    cmd = [PY, script] + extra + ["--steps", str(steps), "--tag", tag,
                                  "--threads", str(A.threads)]
    # a pilot must never run past the deadline
    t0 = time.time()
    try:
        rc = subprocess.call(cmd)
    except Exception as e:
        print(f"  {label:14s} PILOT FAILED {type(e).__name__}: {e}")
        timings[label] = {"error": f"{type(e).__name__}: {e}"}
        continue
    dt = time.time() - t0
    per_step = dt / steps
    cost = per_step * want * A.safety
    timings[label] = {"pilot_steps": steps, "pilot_seconds": round(dt, 2),
                      "seconds_per_step": round(per_step, 5),
                      "wanted_steps": want, "measured_cost_seconds": round(cost, 1),
                      "rc_ok": rc == 0, "clock": clock_str()}
    print(f"  {label:14s} pilot {dt:6.1f}s / {steps} steps = {per_step:.4f} s/step"
          f"  -> full {want} steps ≈ {cost/60:.1f} min  (rc_ok={rc==0})", flush=True)
    if seconds_left() < A.reserve:
        print("  !! out of time during piloting; stopping measurements")
        break

json.dump(timings, open(os.path.join(BUD, "timings.json"), "w"), indent=2)
print(f"\nmeasured timings -> {os.path.join(BUD, 'timings.json')}")

# ---------------------------------------------------------------- plan against the clock
print(f"\n=== plan (clock {clock_str()}, {seconds_left()/60:.1f} min left) ===")
order = ["sweep_uncond", "sweep_pca4", "sweep_pca8", "sweep_pca16", "sweep_pca22",
         "sweep_half", "sweep_code3", "sweep_code5", "kernel_style"]
plan, skipped = [], []
budget = seconds_left() - A.reserve
for label in order:
    t = timings.get(label, {})
    if "measured_cost_seconds" not in t:
        skipped.append({"label": label, "why": "pilot failed"})
        continue
    cost = t["measured_cost_seconds"]
    if cost <= budget:
        plan.append({"label": label, "cost_seconds": cost,
                     "steps": t["wanted_steps"]})
        budget -= cost
    else:
        skipped.append({"label": label, "why": f"needs {cost/60:.1f} min, only "
                                               f"{budget/60:.1f} min left"})
for p in plan:
    print(f"  RUN     {p['label']:14s} {p['cost_seconds']/60:5.1f} min  ({p['steps']} steps)")
for s in skipped:
    print(f"  SKIP    {s['label']:14s} {s['why']}")
predicted_finish = clock_str()
fin = int(now_seconds() + sum(p["cost_seconds"] for p in plan))
fin_hh, fin_mm = fin // 3600, (fin % 3600) // 60
print(f"  predicted finish at ~{fin_hh:02d}:{fin_mm:02d}  (deadline {A.deadline}, "
      f"reserve {A.reserve}s)")
json.dump({"clock_at_plan": clock_str(), "deadline": A.deadline,
           "seconds_left": seconds_left(), "reserve": A.reserve,
           "plan": plan, "skipped": skipped,
           "predicted_finish_seconds_after_midnight": fin},
          open(os.path.join(BUD, "plan.json"), "w"), indent=2)
print(f"\nplan -> {os.path.join(BUD, 'plan.json')}")
print("BUDGET_DONE")
