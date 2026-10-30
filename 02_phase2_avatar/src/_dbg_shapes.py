import os, sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
HERE = r'E:\redstone_nn\02_phase2_avatar\src'
ROOT = os.path.dirname(HERE)
import importlib.util
sp = importlib.util.spec_from_file_location("layout", os.path.join(HERE, "layout.py"))
layout = importlib.util.module_from_spec(sp); sp.loader.exec_module(layout)
NLEV = 16
full = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_full_idx.npy"))
left = full[:, layout.POS_ROW, layout.POS_COL].astype(np.int64)

# replicate the half condition exactly as sweep_condition.py does
COND_DIM = 32
oh = np.zeros((len(left), 32, NLEV), dtype=np.float32)
oh[np.arange(len(left))[:, None], np.arange(32)[None, :], left] = 1.0
cond_all = oh.reshape(len(left), -1)
print("cond_all", cond_all.shape, "but COND_DIM variable says", COND_DIM)
print("  -> mismatch! the builder returns", cond_all.shape[1], "dims while COND_DIM =", COND_DIM)
print()
print("In sweep_condition.py the half branch does:")
print("    COND_DIM = 32          <- then returns a (N, 512) matrix")
print("  so Dis gets cdim=32 (the flag) but the real condition is 512 wide.")
print("  Gen uses 32*NLEV=512 for its input but cdim=32 for the discriminator tail.")
print()
# demonstrate the crash
class Dis(nn.Module):
    def __init__(self, c, emb, cdim):
        super().__init__()
        self.cdim = cdim
        self.conv = nn.Sequential(nn.Conv2d(emb, c, 3, padding=1), nn.LeakyReLU(0.2, True))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(c*64 + cdim, 64), nn.Linear(64, 1))
c = 64
for cdim in (32, 512):
    d = Dis(c, 8, cdim)
    f = torch.zeros(2, c*64)
    cond = torch.zeros(2, cdim)
    try:
        out = d.head(torch.cat([f, cond], 1))
        print(f"  cdim={cdim:4d} -> head OK, out {tuple(out.shape)}")
    except RuntimeError as e:
        print(f"  cdim={cdim:4d} -> RuntimeError: {e}")
