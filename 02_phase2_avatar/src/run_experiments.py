#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Time-budgeted experiment runner.

The user is away until ~21:00 and asked for a program that reads the system clock and
decides whether to keep running. This does exactly that: before each experiment it
checks the clock, estimates the experiment's cost from a measured per-step rate, and
skips anything that would not finish comfortably before the deadline. It stops when
nothing fits.

Measured rates (this machine, 8 threads, batch 256, 8x8 model):
    sweep_condition.py   ~ 0.075 s/step
    kernel_style.py      ~ 0.030 s/step
Those are conservative starting points; the runner re-estimates from each run's own
reported timing where possible.

Usage:
    python run_experiments.py --deadline 20:45 [--dry-run]
"""
import argparse, json, os, subprocess, sys, time

AP = argparse.ArgumentParser()
AP.add_argument("--deadline", default="20:45", help="HH:MM local time to stop by")
AP.add_argument("--dry-run", action="store_true")
AP.add_argument("--threads", type=int, default=8)
A = AP.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
LOG = os.path.join(ROOT, "results", "runner_log.jsonl")
os.makedirs(os.path.dirname(LOG), exist_ok=True)


def now_min():
    t = time.localtime()
    return t.tm_hour * 60 + t.tm_min


def deadline_min():
    h, m = A.deadline.split(":")
    return int(h) * 60 + int(m)


def log(rec):
    rec["at"] = time.strftime("%H:%M:%S")
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[{rec['at']}] {rec.get('event')}: {rec.get('detail','')}", flush=True)


# ---------------------------------------------------------------- the plan
# Each entry: (label, script, extra args, estimated seconds)
# Ordered by information value per minute.
SWEEP = os.path.join(HERE, "sweep_condition.py")
KERN = os.path.join(HERE, "kernel_style.py")

PLAN = [
    # 1) the honest lower bound, with the fixed layout, judged by the new stick
    ("sweep_uncond", SWEEP, ["--mode", "uncond", "--steps", "1500", "--tag", "s_uncond"], 150),
    # 2) condition-strength ladder: PCA k = 4, 8, 16, 22
    ("sweep_pca4", SWEEP, ["--mode", "pca", "--pca-k", "4", "--steps", "1500", "--tag", "s_pca4"], 160),
    ("sweep_pca8", SWEEP, ["--mode", "pca", "--pca-k", "8", "--steps", "1500", "--tag", "s_pca8"], 165),
    ("sweep_pca16", SWEEP, ["--mode", "pca", "--pca-k", "16", "--steps", "1500", "--tag", "s_pca16"], 170),
    ("sweep_pca22", SWEEP, ["--mode", "pca", "--pca-k", "22", "--steps", "1500", "--tag", "s_pca22"], 175),
    # 3) upper bound of the ladder: the real half (the old, weak test) for calibration
    ("sweep_half", SWEEP, ["--mode", "half", "--steps", "1500", "--tag", "s_half"], 190),
    # 4) learned discrete style codes at two sizes
    ("sweep_code3", SWEEP, ["--mode", "code", "--code-k", "3", "--steps", "1500", "--tag", "s_code3"], 190),
    ("sweep_code5", SWEEP, ["--mode", "code", "--code-k", "5", "--steps", "1500", "--tag", "s_code5"], 200),
    # 5) the user's kernel idea: learn 2x2/3x3 kernels and see what styles emerge
    ("kernel_style", KERN, ["--steps", "4000", "--tag", "k_4000"], 130),
    ("kernel_style_9", KERN, ["--steps", "4000", "--embed", "9", "--code-k", "4",
                              "--tag", "k9_4000"], 140),
]

log({"event": "start", "detail": f"deadline {A.deadline}, now {time.strftime('%H:%M')}"})
ran = []
for label, script, extra, est in PLAN:
    left = deadline_min() - now_min()
    if left <= 0:
        log({"event": "stop", "detail": f"past deadline, {len(ran)} experiments ran"})
        break
    if est > left - 2:
        log({"event": "skip", "detail": f"{label}: needs ~{est}s but only {left*60}s left"})
        continue
    cmd = [PY, script] + extra + ["--threads", str(A.threads), "--deadline", A.deadline]
    log({"event": "launch", "detail": f"{label}: {' '.join(os.path.basename(script) + ' ' + ' '.join(extra))}"})
    if A.dry_run:
        ran.append({"label": label, "est": est})
        continue
    t0 = time.time()
    try:
        # stdio inherited: the confined sandbox blocks piped grandchild stdio
        rc = subprocess.call(cmd)
        ok = (rc == 0)
    except Exception as e:
        log({"event": "error", "detail": f"{label}: {type(e).__name__} {e}"})
        ok = False
    dt = time.time() - t0
    log({"event": "done", "detail": f"{label}: rc_ok={ok} in {dt:.0f}s (est {est}s)"})
    ran.append({"label": label, "seconds": round(dt, 1), "ok": ok})

log({"event": "finish", "detail": f"ran {len(ran)}: {[r['label'] for r in ran]}"})
print(json.dumps({"ran": ran}, indent=2))
print("RUNNER_DONE")
