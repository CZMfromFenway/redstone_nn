import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
SCRIPT = os.path.join(HERE, "dim_sweep.py")
LOG = os.path.join(ROOT, "results", "dimsweep", "compress_logs")
os.makedirs(LOG, exist_ok=True)

# The width sweep holds everything else fixed (k=4, 12000 steps, same recipe) and varies only
# the generator. h256 with two layers reproduces the current best model, so its score is the
# control: if it does not come back at the same accuracy, the comparison is invalid.
CONFIGS = [
    (32, 2), (64, 2), (96, 2), (128, 2), (192, 2), (256, 2),
    (64, 1), (128, 1), (256, 1),
]

procs = []
for hid, layers in CONFIGS:
    tag = f"dim4w{hid}L{layers}"
    d = os.path.join(ROOT, "results", "dimsweep", tag)
    if os.path.exists(os.path.join(d, "final.pt")):
        print(f"  {tag:14s} already exists, skipping")
        continue
    f = open(os.path.join(LOG, f"{tag}.log"), "w", encoding="utf-8")
    cmd = [PY, "-u", SCRIPT, "--dims", "4", "--hid", str(hid), "--layers", str(layers),
           "--steps", "12000", "--threads", "3", "--tag", tag]
    p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
    procs.append((tag, p, f))
    print(f"  {tag:14s} pid {p.pid}  ({4}->{hid}" + (f"->{hid}" if layers == 2 else "") + "->32)")

for tag, p, f in procs:
    rc = p.wait()
    f.close()
    print(f"  {tag} finished rc={rc}")

print("COMPRESS_ALL_DONE")
