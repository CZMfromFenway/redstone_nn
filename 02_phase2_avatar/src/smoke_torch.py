#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
"""
Smoke test: torch works, and a real forward/backward pass runs on faces8 data.
Writes nothing except stdout.
"""
import os, sys, time
import numpy as np

print("python:", sys.version.split()[0], sys.executable)
import torch
print("torch :", torch.__version__)
print("cuda  :", torch.cuda.is_available())
print("threads:", torch.get_num_threads())

# 1. autograd
x = torch.randn(8, 4, requires_grad=True)
loss = (x @ x.T).sum()
loss.backward()
print("autograd: OK, grad norm =", float(x.grad.norm()))

# 2. conv + cross entropy on the real palette data
DATA = r"E:\redstone_nn\02_phase2_avatar\archive_phase1_attempts\data"
lab = np.load(os.path.join(DATA, "faces8.npy")).astype(np.int64)
print("faces8:", lab.shape, "unique:", np.unique(lab).size)

t = torch.from_numpy(lab).long()                 # (N,64) in 0..15
N = 20000
idx = torch.randperm(len(t))[:N]
t = t[idx]
# one-hot image (N,16,8,8) to feed a conv
img = torch.zeros(N, 16, 8, 8)
pos = torch.arange(64)
img.view(N, 16, 64).scatter_(1, t.unsqueeze(1), 1.0)

conv = torch.nn.Conv2d(16, 32, 3, padding=1)
head = torch.nn.Conv2d(32, 16, 1)
opt = torch.optim.Adam(list(conv.parameters()) + list(head.parameters()), lr=1e-3)

t0 = time.time()
for step in range(200):
    logits = head(torch.relu(conv(img)))
    loss = torch.nn.functional.cross_entropy(
        logits.permute(0, 2, 3, 1).reshape(-1, 16),
        t.reshape(-1),
    )
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 50 == 0:
        with torch.no_grad():
            acc = (logits.argmax(1) == t.view(N, 8, 8)).float().mean().item()
        print(f"  step {step:4d}  loss {loss.item():.4f}  pixel-acc {acc:.3f}  ({time.time()-t0:.1f}s)")
print("conv train smoke: OK")

# 3. can we persist and reload a model file
p = r"E:\redstone_nn\02_phase2_avatar\results\_smoke.pt"
torch.save({"w": conv.weight.detach()}, p)
d = torch.load(p, weights_only=True)
os.remove(p)
print("save/load: OK", tuple(d["w"].shape))
print("SMOKE_OK")
