#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Chain the overnight queues: run one, then the next, until the deadline.

The single-queue runner stops when its list is exhausted. This wrapper keeps feeding it
new lists so the machine stays busy all night without me having to be awake, while still
obeying the same clock/health rules. It re-reads the clock between chains, so a chain
that overruns simply causes the next one to be skipped rather than to run late.
"""
import argparse, json, os, subprocess, sys, time

import runtime_util as ru

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")

AP = argparse.ArgumentParser()
AP.add_argument("--queues", default="queue_night2.json,queue_night3.json,queue_night1.json")
AP.add_argument("--deadline", default="07:00")
AP.add_argument("--threads", type=int, default=10)
AP.add_argument("--min-free-gb", type=float, default=3.0)
AP.add_argument("--reserve-min", type=int, default=25)
A = AP.parse_args()

queues = [q.strip() for q in A.queues.split(",") if q.strip()]
ru.log("chain", f"{len(queues)} queue files: {queues}; deadline {A.deadline}")
ru.log("clock", f"now {ru.clock()}, {ru.seconds_until(A.deadline)/3600:.2f} h left, "
                f"disk {ru.disk_free_gb():.1f} GB")

for q in queues:
    left = ru.seconds_until(A.deadline)
    if left <= A.reserve_min * 60:
        ru.log("chain", f"only {left/60:.1f} min left; not starting {q}")
        break
    if not ru.health_ok(A.min_free_gb):
        ru.log("chain", "disk low; stopping chain")
        break
    path = q if os.path.isabs(q) else os.path.join(HERE, q)
    if not os.path.exists(path):
        ru.log("chain", f"{q} missing; skipping")
        continue
    ru.log("chain", f"starting {q} with {left/60:.1f} min left")
    rc = subprocess.call([PY, os.path.join(HERE, "overnight_runner.py"),
                          "--queue", path, "--deadline", A.deadline,
                          "--threads", str(A.threads),
                          "--reserve-min", str(A.reserve_min),
                          "--safety", "1.0"],
                         )
    ru.log("chain", f"{q} finished rc={rc} at {ru.clock()}")

# generate the morning report as the very last step, whatever happened
ru.log("chain", "generating the morning report")
try:
    subprocess.call([PY, os.path.join(HERE, "morning_report.py")])
    ru.log("chain", "morning report written")
except Exception as e:
    ru.log("chain", f"morning report failed: {type(e).__name__} {e}")
ru.log("chain", f"done at {ru.clock()}")
print("CHAIN_DONE")
