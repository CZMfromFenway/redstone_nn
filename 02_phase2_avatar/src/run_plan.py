#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Execute the MEASURED plan, checking the wall clock between every experiment.

Reads results/_budget/plan.json (produced by measure_then_plan.py from real pilot
timings, never from guesses). Before launching each experiment it re-reads the system
clock and skips anything that no longer fits before the deadline, so the run always
stops in time no matter how the machine's speed varies.

Every decision is appended to results/runner_log.jsonl together with the clock reading
at that moment, so the whole session is auditable.
"""
import argparse, json, os, subprocess, time

AP = argparse.ArgumentParser()
AP.add_argument("--deadline", default="21:00")
AP.add_argument("--reserve", type=int, default=150, help="seconds kept free at the end")
AP.add_argument("--overhead", type=int, default=15, help="measured extra per launch (s)")
AP.add_argument("--threads", type=int, default=8)
AP.add_argument("--dry-run", action="store_true")
AP.add_argument("--only", default="", help="comma-separated labels to run")
A = AP.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
BUD = os.path.join(ROOT, "results", "_budget")
LOG = os.path.join(ROOT, "results", "runner_log.jsonl")
os.makedirs(os.path.dirname(LOG), exist_ok=True)

SWEEP = os.path.join(HERE, "sweep_condition.py")
KERN = os.path.join(HERE, "kernel_style.py")

# label -> (script, args). Step counts come from plan.json.
WHAT = {
    "sweep_uncond": (SWEEP, ["--mode", "uncond"]),
    "sweep_pca4":   (SWEEP, ["--mode", "pca", "--pca-k", "4"]),
    "sweep_pca8":   (SWEEP, ["--mode", "pca", "--pca-k", "8"]),
    "sweep_pca16":  (SWEEP, ["--mode", "pca", "--pca-k", "16"]),
    "sweep_pca22":  (SWEEP, ["--mode", "pca", "--pca-k", "22"]),
    "sweep_half":   (SWEEP, ["--mode", "half"]),
    "sweep_code3":  (SWEEP, ["--mode", "code", "--code-k", "3"]),
    "sweep_code5":  (SWEEP, ["--mode", "code", "--code-k", "5"]),
    "kernel_style": (KERN, ["--embed", "9", "--code-k", "4"]),
}


def now_sec():
    t = time.localtime()
    return t.tm_hour * 3600 + t.tm_min * 60 + t.tm_sec


def clock():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def dl_sec():
    h, m = (int(x) for x in A.deadline.split(":"))
    d = h * 3600 + m * 60
    if d <= now_sec():
        d += 86400
    return d


def left_sec():
    return dl_sec() - now_sec()


def log(rec):
    rec["clock"] = clock()
    rec["seconds_left"] = int(left_sec())
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[{rec['clock']}] {rec.get('event')}: {rec.get('detail','')} "
          f"(left {left_sec()/60:.1f} min)", flush=True)


plan = json.load(open(os.path.join(BUD, "plan.json"), encoding="utf-8"))
order = [p for p in plan["plan"]]
if A.only:
    keep = {s.strip() for s in A.only.split(",")}
    order = [p for p in order if p["label"] in keep]

log({"event": "start", "detail": f"executing {len(order)} measured steps; "
                                 f"deadline {A.deadline}, reserve {A.reserve}s"})
print(f"\nregistry: clock {clock()}, deadline {A.deadline}, "
      f"left {left_sec()/60:.1f} min\n")

done = []
for p in order:
    label = p["label"]
    if label not in WHAT:
        log({"event": "skip", "detail": f"{label}: not in registry"}); continue
    script, extra = WHAT[label]
    steps = p["steps"]
    measured = p["cost_seconds"]
    need = measured + A.overhead
    left = left_sec()
    if need > left - A.reserve:
        log({"event": "skip", "detail": f"{label}: needs {need/60:.1f} min (measured), "
                                        f"only {(left-A.reserve)/60:.1f} min usable"})
        continue
    cmd = [PY, script] + extra + ["--steps", str(steps), "--tag", label,
                                  "--threads", str(A.threads), "--deadline", A.deadline]
    log({"event": "launch", "detail": f"{label}: {steps} steps, measured {measured/60:.1f} min"})
    if A.dry_run:
        done.append({"label": label, "dry": True}); continue
    t0 = time.time()
    try:
        rc = subprocess.call(cmd)          # stdio inherited (sandbox blocks piped stdio)
        ok = rc == 0
    except Exception as e:
        log({"event": "error", "detail": f"{label}: {type(e).__name__} {e}"}); ok = False
    dt = time.time() - t0
    log({"event": "done", "detail": f"{label}: ok={ok} actual {dt/60:.1f} min "
                                    f"(predicted {measured/60:.1f} min)"})
    done.append({"label": label, "seconds": round(dt, 1), "ok": ok})

log({"event": "finish", "detail": f"ran {len(done)}: {[d['label'] for d in done]}"})
print(json.dumps({"ran": done}, indent=2))
print("RUNNER_DONE")
