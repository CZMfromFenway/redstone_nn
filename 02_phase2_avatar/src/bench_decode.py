import io, time, zipfile
import numpy as np
from PIL import Image

z = zipfile.ZipFile(r"E:\redstone_nn\02_phase2_avatar\archive_phase1_attempts\data\minecraft_skins_64x64.zip")
names = [n for n in z.namelist() if n.lower().endswith(".png")]
print("entries:", len(names))

for N in (200, 1000):
    t0 = time.time()
    crops = []
    for n in names[:N]:
        with z.open(n) as fh:
            raw = fh.read()
        im = Image.open(io.BytesIO(raw)).convert("RGBA")
        crops.append(np.asarray(im.crop((8, 8, 16, 16)), dtype=np.uint8)[:, :, :3])
    dt = time.time() - t0
    print(f"  decode {N:5d}: {dt:6.2f}s  = {N/dt:8.0f} img/s   -> 1.1M would take {1107411/(N/dt)/3600:5.2f} h")

# Is .convert('RGBA') the bottleneck? test without it
for N in (1000,):
    t0 = time.time()
    for n in names[:N]:
        with z.open(n) as fh:
            raw = fh.read()
        im = Image.open(io.BytesIO(raw))
        a = np.asarray(im)
    dt = time.time() - t0
    print(f"  raw read only {N}: {dt:6.2f}s = {N/dt:8.0f} img/s")
