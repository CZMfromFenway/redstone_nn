import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
SCRIPT = os.path.join(HERE, "colour_gan.py")
LOG = os.path.join(ROOT, "results", "colour", "gan", "logs")
os.makedirs(LOG, exist_ok=True)

# C2  the grayscale recipe transplanted: adversarial + MSE + total variation
# C3  C2 plus explicit matching of mean saturation and colour separation to the real data's
# C4  same as C2 but a much stronger smoothness term, since TV still helped the grayscale line
CONFIGS = [
    ("C2", ["--tv", "5.0", "--stats", "0.0"]),
    ("C3", ["--tv", "5.0", "--stats", "1.0"]),
    ("C4", ["--tv", "12.0", "--stats", "0.0"]),
    ("C5", ["--tv", "5.0", "--stats", "2.0"]),
]

procs = []
for tag, extra in CONFIGS:
    f = open(os.path.join(LOG, f"{tag}.log"), "w", encoding="utf-8")
    cmd = [PY, "-u", SCRIPT, "--tag", tag, "--steps", "20000", "--threads", "3"] + extra
    p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
    procs.append((tag, p, f))
    print(f"  {tag} pid {p.pid}  {' '.join(extra)}")

for tag, p, f in procs:
    rc = p.wait()
    f.close()
    print(f"  {tag} finished rc={rc}")
print("COLOUR_GAN_ALL_DONE")
