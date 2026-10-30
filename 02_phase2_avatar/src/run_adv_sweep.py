import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
SCRIPT = os.path.join(HERE, "colour_gan.py")
LOG = os.path.join(ROOT, "results", "colour", "gan", "logs")
os.makedirs(LOG, exist_ok=True)

# ATTACKING THE EDGE GAP
# Measured on identical axes: the real data's 90th-percentile colour step between neighbouring
# pixels is 15.56, while every model sits near 9.5-10. Growing the run from 8000 to 20000 steps
# only moved one variant from 9.31 to 10.05, so more steps is not the lever.
# The loss balance is: MSE ~2.25 versus adversarial ~0.7, i.e. reconstruction outweighs the
# discriminator roughly 3:1. Edge sharpness is the discriminator's job, so it is being
# out-voted. These runs raise the adversarial weight to let it compete.
CONFIGS = [
    ("A3", ["--tv", "5.0", "--stats", "1.0", "--adv", "3.0"]),
    ("A5", ["--tv", "5.0", "--stats", "1.0", "--adv", "5.0"]),
    ("A3T2", ["--tv", "2.0", "--stats", "1.0", "--adv", "3.0"]),
]

procs = []
for tag, extra in CONFIGS:
    f = open(os.path.join(LOG, f"{tag}.log"), "w", encoding="utf-8")
    cmd = [PY, "-u", SCRIPT, "--tag", tag, "--steps", "20000", "--threads", "4"] + extra
    p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
    procs.append((tag, p, f))
    print(f"  {tag} pid {p.pid}  {' '.join(extra)}")

for tag, p, f in procs:
    rc = p.wait()
    f.close()
    print(f"  {tag} finished rc={rc}")
print("ADV_SWEEP_DONE")
