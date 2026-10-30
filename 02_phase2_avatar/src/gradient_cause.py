# Measure the metric that actually explains the visible softness.
import json, os, sys
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, r"E:\redstone_nn\02_phase2_avatar\src")
import importlib.util as _ilu
_l = _ilu.spec_from_file_location("layout", r"E:\redstone_nn\02_phase2_avatar\src\layout.py")
layout = _ilu.module_from_spec(_l); _l.loader.exec_module(layout)
NLEV=16; LEVELS=np.linspace(0,255,16); R=r"E:\redstone_nn\02_phase2_avatar"
rgb=np.load(os.path.join(R,"data","faces8_rgb_v2.npy")); LUM=np.array([0.299,0.587,0.114])
gray=(rgb.astype(np.float64)*LUM).sum(-1)
gyi=np.abs(gray[:,:,:,None]-LEVELS[None,None,None,:]).argmin(-1)
kn=(gyi[:,layout.POS_ROW,layout.POS_COL]==gyi[:,layout.POS_ROW,layout.MIRROR_COL]).mean(1)>=0.95
Gg=gray[kn]; X=Gg.reshape(len(Gg),-1); mu=X.mean(0,keepdims=True)
sub=np.random.default_rng(0).choice(len(X),min(60000,len(X)),replace=False)
U,S,Vt=np.linalg.svd(X[sub]-mu,full_matrices=False); comps=Vt[:22]
sd=(((X-mu)@comps.T).std(0,keepdims=True)+1e-6)
COND=torch.tensor(((X-mu)@comps.T)/sd[0],dtype=torch.float32)
va=np.load(os.path.join(R,"data_v2_canonical","L0_split.npz"))["val"]
def load(path):
    c=torch.load(path,weights_only=False); st=c.get("G") or c.get("S")
    norm={k.replace("net.","",1) if k.startswith("net.") else k:v for k,v in st.items()}
    idx=sorted({int(k.split(".")[0]) for k in norm if k.endswith(".weight")})
    seq=[]
    for j in idx:
        W=norm[f"{j}.weight"]; seq.append(nn.Linear(int(W.shape[1]),int(W.shape[0])))
        if j!=idx[-1]: seq.append(nn.ReLU())
    m=nn.Sequential(*seq); m.load_state_dict(norm); m.eval(); return m
print(f"{'variant':34s} {'within 0.1 of a level':>22s} {'mean |frac|':>12s}")
print("-"*72)
for name,path in (("22->128->128->32 (23k)","results/fragment2/tv_cont_0p1_(control)/final.pt"),
                  ("22->32->32 (1.7k)","results/shrink5/ladder_32.pt"),
                  ("distill 22->256->256->32","results/distill/distill_plain/final.pt"),
                  ("hard+TV5 22->256->256->32","results/shrink/s32_tv5/final.pt")):
    p=os.path.join(R,path)
    if not os.path.exists(p): continue
    m=load(p)
    with torch.no_grad(): v=m(COND[va[:1000]]).clamp(0,15).numpy()
    frac=(v-v.round()).__abs__()
    print(f"  {name:32s} {float((frac<0.1).mean())*100:21.1f}% {float(frac.mean()):12.3f}")
print()
print("  The teacher has no such quantity: it always emits one of 16 levels, so 100% of")
print("  its output sits exactly on a level. Any fraction below 100% means the student is")
print("  producing in-between values, which render as gradients instead of steps.")
