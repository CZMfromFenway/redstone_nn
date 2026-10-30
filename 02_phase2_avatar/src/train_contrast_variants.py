#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Train the two training-stage contrast remedies for k=4 side by side.

  contrast : adds a term matching the output's per-sample spread to the target's
  stretch  : stretches every target to [0,15] so a low-contrast answer is unavailable

Both are children of this process, so they live and die with the tracked job.
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
SCRIPT = os.path.join(HERE, "dim_sweep.py")
OUT = os.path.join(ROOT, "results", "dimsweep")
os.makedirs(OUT, exist_ok=True)

JOBS = [
    ("contrast", ["--dims", "4", "--contrast-weight", "3.0", "--tag", "dim4_contrast"]),
    ("stretch", ["--dims", "4", "--stretch-targets", "--tag", "dim4_stretch"]),
]
STEPS = os.environ.get("STEPS", "12000")
THREADS = os.environ.get("THREADS", "10")

procs = []
for name, extra in JOBS:
    log = os.path.join(OUT, f"train_{name}.log")
    f = open(log, "w", encoding="utf-8", buffering=1)
    cmd = [PY, "-u", SCRIPT] + extra + ["--steps", STEPS, "--eval-every", "4000",
                                        "--threads", THREADS]
    p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
    procs.append({"name": name, "p": p, "f": f})
    print(f"  started {name:9s} pid={p.pid} -> {os.path.basename(log)}", flush=True)

t0 = time.time()
alive = {r["name"] for r in procs}
while alive:
    time.sleep(20)
    for r in procs:
        rc = r["p"].poll()
        if rc is not None and r["name"] in alive:
            alive.discard(r["name"])
            r["f"].close()
            print(f"  {r['name']} finished rc={rc} at {time.time()-t0:.0f}s", flush=True)
    if alive:
        print(f"  [{time.time()-t0:4.0f}s] training: {sorted(alive)}", flush=True)

print()
print("=" * 70)
for name, _ in JOBS:
    sp = os.path.join(OUT, f"dim4_{name}", "summary.json")
    if os.path.exists(sp):
        d = json.load(open(sp, encoding="utf-8"))
        last = d["history"][-1]
        print(f"  {name:9s} acc {last['acc']:.4f}  agree {last['agree']*100:5.1f}%")
print(f"total wall {time.time()-t0:.0f}s")
print("CONTRAST_VARIANTS_DONE")
