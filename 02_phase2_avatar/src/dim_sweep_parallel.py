#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Run the whole dimension ladder in parallel from ONE tracked process.

Why this exists
PowerShell's Start-Process was used first and every child died within seconds with an empty
log -- the same silent failure this project hit before. A managed background job is tracked
by the harness, so instead of spawning detached processes we spawn CHILDREN of this script
and wait for them. When the job ends, everything ends.

Each child writes its own log file (not a pipe: the sandbox blocks piped stdio for some
programs, and a file is easier to tail anyway).
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = os.path.join(os.path.join(ROOT, ".venv"), "Scripts", "python.exe")
SCRIPT = os.path.join(HERE, "dim_sweep.py")
OUT = os.path.join(ROOT, "results", "dimsweep")
os.makedirs(OUT, exist_ok=True)

LADDER = [int(x) for x in (os.environ.get("LADDER") or "1,2,4,8,16").split(",") if x.strip()]
STEPS = int(os.environ.get("STEPS", "12000"))
THREADS = int(os.environ.get("THREADS", "6"))

procs = []
for k in LADDER:
    log = os.path.join(OUT, f"train_dim{k}.log")
    f = open(log, "w", encoding="utf-8", buffering=1)
    cmd = [PY, "-u", SCRIPT, "--dims", str(k), "--steps", str(STEPS),
           "--eval-every", "4000", "--threads", str(THREADS)]
    p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
    procs.append({"k": k, "p": p, "f": f, "log": log})
    print(f"  started k={k}  pid={p.pid}  -> {os.path.basename(log)}", flush=True)

t0 = time.time()
alive = {r["k"] for r in procs}
while alive:
    time.sleep(20)
    for r in procs:
        rc = r["p"].poll()
        if rc is not None and r["k"] in alive:
            alive.discard(r["k"])
            r["f"].close()
            print(f"  k={r['k']} finished rc={rc} at {time.time()-t0:.0f}s", flush=True)
    if alive:
        print(f"  [{time.time()-t0:4.0f}s] still training: {sorted(alive)}", flush=True)

print()
print("=" * 70)
rows = []
for r in procs:
    sp = os.path.join(OUT, f"dim{r['k']}", "summary.json")
    if os.path.exists(sp):
        d = json.load(open(sp, encoding="utf-8"))
        last = d["history"][-1]
        rows.append({"k": d["k"], "multiplies": d["multiplies"],
                     "acc": last["acc"], "agree": last["agree"]})
print(f"  {'k':>3} {'multiplies':>11} {'accuracy':>9} {'agreement':>10}")
for r in rows:
    print(f"  {r['k']:3d} {r['multiplies']:11,d} {r['acc']:9.4f} {r['agree']*100:9.1f}%")
print(f"  {'22':>3} {74752:11,d} {0.7719:9.4f} {72.8:9.1f}%   (s32_tv5 reference)")
json.dump(rows, open(os.path.join(OUT, "ladder_all.json"), "w"), indent=2)
print(f"\nwrote {os.path.join(OUT, 'ladder_all.json')}")
print(f"total wall {time.time()-t0:.0f}s")
print("LADDER_PARALLEL_DONE")
