from __future__ import annotations
import sys,time,argparse
from pathlib import Path
import numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).parent))
from legacy_primitives import D_apply

def libs():
 x=np.arange(16,dtype=np.int16);dm={}
 for m in range(4):
  for a in range(16):
   for b in range(16):
    ar=tuple(map(int,D_apply(x,m,a,b)));dm.setdefault(ar,(m,a,b))
 D=np.array(list(dm.keys()),dtype=np.int16);dw=list(dm.values()); hm={}
 for ri,row in enumerate(D):
  for op in range(2):
   for k in range(16):
    bits=tuple(map(int,(row<=k) if op else (row>k)));hm.setdefault(bits,(ri,k,op))
 H=np.array(list(hm.keys()),dtype=np.int8);hw=list(hm.values());return D,dw,H,hw
D,DW,H,HW=libs(); assert len(D)==739 and len(H)==242
ID_D=np.where([np.array_equal(r,np.arange(16)) for r in D])[0][0]
ZERO_D=np.where([np.all(r==0) for r in D])[0][0]
# final head that is 1 iff count>0
GT0=np.where([np.array_equal(r[:3],np.array([0,1,1])) for r in H])[0][0]

def flips(t):return int(np.sum(t[1:]!=t[:-1]))
def valid(t,K):return flips(t)<=K

def best_head(c,y,maxc):
 pos=np.bincount(c[y==1],minlength=maxc+1);neg=np.bincount(c[y==0],minlength=maxc+1)
 bits=H[:,:maxc+1]
 errs=(bits*neg[None,:]+(1-bits)*pos[None,:]).sum(1)
 i=int(np.argmin(errs));return int(errs[i]),i

def rand_table(rng,K):
 # sample random run sequence directly
 t=np.zeros(16,dtype=np.int8);t[0]=rng.integers(2)
 # random number of transitions biased broad
 k=int(rng.integers(K+1));loc=set(rng.choice(np.arange(1,16),size=k,replace=False).tolist()) if k else set()
 for i in range(1,16):t[i]=t[i-1]^(i in loc)
 return t

def one_forward(B,x1,x2): return B[0,x1]+B[1,x2]
def optimize_one(x1,x2,y,K,rng,restarts=24,passes=20):
 best=None;evals=0
 # curated seeds + random
 seeds=[]
 # threshold-ish and interval-ish starts
 for th1 in [4,8,12]:
  for th2 in [4,8,12]:
   b=np.zeros((2,16),np.int8);b[0,th1:]=1;b[1,th2:]=1;seeds.append(b)
 while len(seeds)<restarts:seeds.append(np.stack([rand_table(rng,K),rand_table(rng,K)]))
 for B in seeds[:restarts]:
  B=B.copy();c=one_forward(B,x1,x2);err,h=best_head(c,y,2);evals+=1
  for ps in range(passes):
   imp=False
   order=rng.permutation(32)
   for q in order:
    e=q//16;z=q%16;cand=B[e].copy();cand[z]^=1
    if not valid(cand,K):continue
    old=B[e,z];B[e,z]^=1;c2=one_forward(B,x1,x2);er2,h2=best_head(c2,y,2);evals+=1
    if er2<=err:
     if er2<err:imp=True
     err,h,c=er2,h2,c2
    else:B[e,z]=old
   if not imp:break
  rec=(err,B.copy(),h,evals)
  if best is None or err<best[0]:best=rec
  if err==0:break
 return best,evals

def forward2(B1,Gids,B2,x1,x2):
 h=B1.shape[0];N=len(x1);Hs=[];qs=[]
 for j in range(h):
  c=B1[j,0,x1]+B1[j,1,x2];hv=D[Gids[j],c];Hs.append(hv);qs.append(B2[j,hv])
 return np.stack(Hs,1),np.stack(qs,1).sum(1)

def embed_one(one,K,h=2):
 err,B,hid,_=one;B1=np.zeros((h,2,16),np.int8);G=np.full(h,ZERO_D,dtype=np.int16);B2=np.zeros((h,16),np.int8)
 B1[0]=B;G[0]=ID_D;B2[0]=H[hid].copy() # original head becomes 2nd-layer branch predicate on count 0..2
 # others remain zero. final head is count>0
 return B1,G,B2,GT0

def optimize_two(x1,x2,y,K,rng,restarts=10,passes=12,h=2):
 one,_=optimize_one(x1,x2,y,K,rng,restarts=18,passes=15)
 seed=embed_one(one,K,h);best=None;evals=0
 for rr in range(restarts):
  if rr==0:B1,Gids,B2,hid=[a.copy() if hasattr(a,'copy') else a for a in seed]
  else:
   B1=np.empty((h,2,16),np.int8);B2=np.empty((h,16),np.int8);Gids=np.empty(h,np.int16)
   # half restart near embedded, half random
   if rr<restarts//2:
    B1,Gids,B2,hid=[a.copy() if hasattr(a,'copy') else a for a in seed]
    for _ in range(2+rr):
     j=rng.integers(h);i=rng.integers(2);B1[j,i]=rand_table(rng,K)
    if rr%2:Gids[rng.integers(h)]=rng.integers(len(D))
   else:
    for j in range(h):
     for i in range(2):B1[j,i]=rand_table(rng,K)
     B2[j]=rand_table(rng,K);Gids[j]=rng.integers(len(D))
    hid=rng.integers(len(H))
  Hs,c=forward2(B1,Gids,B2,x1,x2);err,hid=best_head(c,y,h);evals+=1
  for ps in range(passes):
   start_err=err
   # optimize second-layer branch tables bitwise
   for j in rng.permutation(h):
    for z in rng.permutation(16):
     cand=B2[j].copy();cand[z]^=1
     if not valid(cand,K):continue
     B2[j,z]^=1;Hs2,c2=forward2(B1,Gids,B2,x1,x2);e2,hh=best_head(c2,y,h);evals+=1
     if e2<=err:err,hid,Hs,c=e2,hh,Hs2,c2
     else:B2[j,z]^=1
   # optimize G exactly over 739 maps, one node at a time
   for j in rng.permutation(h):
    other=np.zeros(len(y),dtype=np.int16)
    for k in range(h):
     if k!=j:other+=B2[k,Hs[:,k]]
    cnt=B1[j,0,x1]+B1[j,1,x2]
    be=err;bg=int(Gids[j]);bh=hid;bhv=Hs[:,j].copy();bc=c.copy()
    for gid in range(len(D)):
     hv=D[gid,cnt];cc=other+B2[j,hv];e2,hh=best_head(cc,y,h);evals+=1
     if e2<be:be,bg,bh,bhv,bc=e2,gid,hh,hv,cc
    if be<=err:Gids[j]=bg;err,hid=be,bh;Hs[:,j]=bhv;c=bc
   # optimize first-layer branch tables bitwise
   for j in rng.permutation(h):
    for i in rng.permutation(2):
     for z in rng.permutation(16):
      cand=B1[j,i].copy();cand[z]^=1
      if not valid(cand,K):continue
      B1[j,i,z]^=1;Hs2,c2=forward2(B1,Gids,B2,x1,x2);e2,hh=best_head(c2,y,h);evals+=1
      if e2<=err:err,hid,Hs,c=e2,hh,Hs2,c2
      else:B1[j,i,z]^=1
   if err==start_err:break
   if err==0:break
  rec=(err,B1.copy(),Gids.copy(),B2.copy(),hid,evals)
  if best is None or err<best[0]:best=rec
  if err==0:break
 return one,best,evals

def run_suite(suite,K,hidden,start,end):
 OUT=Path('/mnt/data/dt_duel_v03/results');OUT.mkdir(parents=True,exist_ok=True);path=OUT/f'count_func_cd_K{K}_h{hidden}_{suite}_v03.csv'
 jobs=[]
 if suite=='structured':
  for i,t in enumerate(['sum_threshold','compare','xor_hi','interval_or']):
   if start<=i<end:jobs.append((t,20260926,i))
 else:
  for i,s in enumerate(range(20261000,20261008)):
   if start<=i<end:jobs.append(('random_balanced',s,i))
 for task,seed,i in jobs:
  x1,x2,y=task_targets(task,seed=seed);rng=np.random.default_rng(2000000+i*139+hidden*17+K);t0=time.perf_counter()
  if hidden==0:
   one,ev=optimize_one(x1,x2,y,K,rng);err=one[0];base=err
  else:
   one,two,ev=optimize_two(x1,x2,y,K,rng,h=hidden);base=one[0];err=two[0]
  wall=time.perf_counter()-t0;row=dict(task=task if task!='random_balanced' else f'random_balanced_{seed}',K=K,hidden=hidden,base_one_errors=base,errors=err,improvement=base-err,hard_MAE=err/256,evaluations=ev,wall_s=wall)
  pd.DataFrame([row]).to_csv(path,mode='a',header=not path.exists(),index=False);print(row,flush=True)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--suite',choices=['structured','random'],required=True);ap.add_argument('--K',type=int,choices=[7,8],default=8);ap.add_argument('--hidden',type=int,choices=[0,2],default=0);ap.add_argument('--start',type=int,default=0);ap.add_argument('--end',type=int,default=8);a=ap.parse_args();run_suite(a.suite,a.K,a.hidden,a.start,a.end)
if __name__=='__main__':main()
