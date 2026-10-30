#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Autonomous overnight runner.

The user is asleep and cannot approve anything, so this script's rules are:
  * the deadline and every "is there time left?" decision come from the SYSTEM CLOCK
  * a task's cost is MEASURED by piloting it, never assumed
  * measured costs are cached in results/overnight/timings.json and reused
  * the disk is checked before every launch; on low disk it stops rather than risks
    filling the volume
  * tasks run one at a time (the box has ~24 threads; one 8-10 thread job is the sweet
    spot, and overlapping jobs made earlier timings unreliable)
  * every launch, finish and skip is appended to a log with the clock reading, so the
    whole night is auditable in the morning
  * nothing is overwritten: each task writes to its own tag

A task list is a JSON array of objects:
    {"tag": "name", "script": "src\\foo.py", "args": ["--x", "1"],
     "pilot_args": ["--steps", "60"], "steps_arg": "--steps", "steps": 9000,
     "pilot_steps": 60}

Usage:
    python overnight_runner.py --queue queue.json --deadline 07:00 [--dry-run]
"""
import argparse, json, os, subprocess, sys, time

import runtime_util as ru

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
TIMINGS = os.path.join(ROOT, "results", "overnight", "timings.json")

AP = argparse.ArgumentParser()
AP.add_argument("--queue", required=True)
AP.add_argument("--deadline", default="07:00")
AP.add_argument("--threads", type=int, default=10)
AP.add_argument("--reserve-min", type=int, default=25, help="minutes to keep free at the end")
AP.add_argument("--safety", type=float, default=1.25, help="multiply the measured cost")
AP.add_argument("--min-free-gb", type=float, default=3.0)
AP.add_argument("--max-pilot-seconds", type=int, default=420)
AP.add_argument("--dry-run", action="store_true")
AP.add_argument("--only", default="", help="comma-separated tags to run")
A = AP.parse_args()

queue = json.load(open(os.path.join(HERE, A.queue) if not os.path.isabs(A.queue) else A.queue,
                       encoding="utf-8"))
if A.only:
    keep = {s.strip() for s in A.only.split(",")}
    queue = [t for t in queue if t["tag"] in keep]

timings = ru.load_timings(TIMINGS)
ru.log("start", f"{len(queue)} tasks queued, deadline {A.deadline}, "
                f"reserve {A.reserve_min} min, threads {A.threads}")
ru.log("clock", f"now {ru.clock()}, {ru.seconds_until(A.deadline)/3600:.2f} h until "
                f"{A.deadline}, disk free {ru.disk_free_gb():.1f} GB")

done, skipped = [], []


def measure(task):
    """Pilot the task to MEASURE seconds-per-step. Returns None on failure.

    TWO-POINT SLOPE. A single pilot of N steps measures (fixed_startup + N*per_step)/N,
    which badly OVERSTATES the per-step cost for long runs: a 40-step pilot on a script
    that spends ~12 s loading data and PCA reported 0.3375 s/step when the true marginal
    cost was 0.078 s/step, i.e. a 4.3x overestimate that made the planner predict 70 min
    for a 13 min job. Running two pilot lengths and taking the slope removes the fixed
    term entirely.
    """
    script = os.path.join(HERE, task["script"])
    args = list(task.get("pilot_args", []))
    sarg = task.get("steps_arg", "--steps")
    n1 = task.get("pilot_steps", 40)
    n2 = n1 * 3
    times = []
    for n in (n1, n2):
        cmd = [PY, script] + task["args"] + args + [sarg, str(n),
                                                   "--threads", str(A.threads),
                                                   "--tag", f"pilot_{task['tag']}_n{n}"]
        t0 = time.time()
        try:
            rc = subprocess.call(cmd, timeout=A.max_pilot_seconds)
        except subprocess.TimeoutExpired:
            ru.log("pilot", f"{task['tag']} n={n}: TIMEOUT after {A.max_pilot_seconds}s")
            return None
        dt = time.time() - t0
        if rc != 0:
            ru.log("pilot", f"{task['tag']} n={n}: failed rc={rc} after {dt:.0f}s")
            return None
        times.append((n, dt))
        ru.log("pilot", f"{task['tag']} n={n}: {dt:.1f}s")
    (na, ta), (nb, tb) = times
    slope = (tb - ta) / max(nb - na, 1)
    fixed = ta - slope * na
    single = ta / na
    if slope <= 0:
        # the two lengths disagreed (noise or caching): fall back to the single pilot
        ru.log("pilot", f"{task['tag']}: non-positive slope ({slope:.4f}); using single-pilot "
                        f"estimate {single:.4f} s/step")
        return single
    ru.log("pilot", f"{task['tag']}: slope {slope:.4f} s/step, fixed startup {fixed:.1f}s "
                    f"(single-pilot estimate would have been {single:.4f}, "
                    f"{single/slope:.1f}x too high)")
    return slope


for task in queue:
    tag = task["tag"]
    if not ru.health_ok(A.min_free_gb):
        ru.log("stop", "disk too low; stopping before this task")
        break
    left = ru.seconds_until(A.deadline)
    if left <= A.reserve_min * 60:
        ru.log("stop", f"only {left/60:.1f} min left (reserve {A.reserve_min} min); stopping")
        break

    # ---- cost: reuse a measurement if we have one, otherwise pilot now ---------------
    key = f"{task['script']}|{','.join(task['args'])}"
    per = timings.get(key, {}).get("seconds_per_step")
    if per is None:
        if A.dry_run:
            per = 0.0
        else:
            per = measure(task)
            if per is None:
                skipped.append({"tag": tag, "why": "pilot failed"})
                continue
            timings[key] = {"seconds_per_step": per, "measured_at": ru.clock(),
                            "pilot_steps": task.get("pilot_steps", 60), "tag": tag}
            ru.save_timings(TIMINGS, timings)
    if per == 0.0:
        cost = 0.0
    else:
        # account for the fixed startup cost the pilot also paid
        cost = per * task["steps"] * A.safety
    need = cost + 30
    if need > left - A.reserve_min * 60 and not A.dry_run:
        ru.log("skip", f"{tag}: measured need {need/60:.1f} min but only "
                       f"{(left - A.reserve_min*60)/60:.1f} min usable")
        skipped.append({"tag": tag, "why": f"needs {need/60:.1f} min"})
        continue

    cmd = [PY, os.path.join(HERE, task["script"])] + task["args"] + \
          [task.get("steps_arg", "--steps"), str(task["steps"]),
           "--threads", str(A.threads), "--tag", tag]
    ru.log("launch", f"{tag}: {task['steps']} steps, measured {per:.4f} s/step "
                     f"-> {cost/60:.1f} min")
    if A.dry_run:
        done.append({"tag": tag, "dry": True})
        continue
    t0 = time.time()
    try:
        rc = subprocess.call(cmd)
        ok = (rc == 0)
    except Exception as e:
        ru.log("error", f"{tag}: {type(e).__name__} {e}")
        ok = False
    dt = time.time() - t0
    predicted = cost
    ru.log("done", f"{tag}: ok={ok} actual {dt/60:.1f} min (predicted {predicted/60:.1f} min, "
                   f"ratio {dt/max(predicted,1e-9):.2f})")
    done.append({"tag": tag, "seconds": round(dt, 1), "ok": ok,
                 "predicted_seconds": round(predicted, 1)})

ru.log("finish", f"ran {len(done)}, skipped {len(skipped)}")
print(json.dumps({"ran": done, "skipped": skipped}, indent=2, ensure_ascii=False))
print("OVERNIGHT_RUNNER_DONE")
