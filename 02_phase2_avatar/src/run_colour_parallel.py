import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(os.path.dirname(HERE), ".venv", "Scripts", "python.exe")
SCRIPT = os.path.join(HERE, "colour_stage2_train.py")
LOG = os.path.join(os.path.dirname(HERE), "results", "colour", "logs")
os.makedirs(LOG, exist_ok=True)

VARIANTS = ["E1", "E2", "E3", "E4", "E5", "E6"]
print(f"launching {len(VARIANTS)} variants, 4 threads each ({4*len(VARIANTS)} of 32 cores)")

procs = []
for v in VARIANTS:
    f = open(os.path.join(LOG, f"{v}.log"), "w", encoding="utf-8")
    p = subprocess.Popen([PY, SCRIPT, "--steps", "12000", "--threads", "4", "--only", v],
                         stdout=f, stderr=subprocess.STDOUT)
    procs.append((v, p, f))
    print(f"  {v} pid {p.pid}")

for v, p, f in procs:
    rc = p.wait()
    f.close()
    print(f"  {v} finished rc={rc}")

print("ALL_VARIANTS_DONE")
