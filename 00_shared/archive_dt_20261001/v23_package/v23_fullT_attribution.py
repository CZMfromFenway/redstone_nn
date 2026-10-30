import sys, numpy as np, pandas as pd
sys.path.insert(0,'/mnt/data')
from DTF004_exact_depth2 import build_base, eval_T, compose
X=range(16)
D,_=build_base(False)

def T_by_pairs(pairs):
    out={}
    for A in ('C','S'):
      for p,q in pairs:
       for a in X:
        for b in X:
         f=eval_T(A,p,q,a,b); out.setdefault(f,('T',A,p,q,a,b))
    return out
Tmix=T_by_pairs((('c','s'),('s','c')))
Tcc=T_by_pairs((('c','c'),))
Tss=T_by_pairs((('s','s'),))
Tfull=T_by_pairs((('c','s'),('s','c'),('c','c'),('s','s')))
print('T unique mix cc ss full',*[len(z) for z in [Tmix,Tcc,Tss,Tfull]])

def pool(seq,T):
    base={'D':D,'T':T}; s=set()
    for fi in base[seq[0]]:
      for fo in base[seq[1]]:
        f=compose(fi,fo)
        if max(f)<=3: s.add(f)
    return s
rows=[]
for seq in ['DT','TD','TT']:
    sm=pool(seq,Tmix); sf=pool(seq,Tfull); scc=pool(seq,Tcc); sss=pool(seq,Tss)
    add=sf-sm
    rows.append(dict(sequence=seq,full_added=len(add),added_reachable_with_cc=len(add&scc),added_reachable_with_ss=len(add&sss),added_reachable_with_either_same=len(add&(scc|sss)),added_need_mixed_same_interaction=len(add-(scc|sss)),cc_pool=len(scc),ss_pool=len(sss)))
pd.DataFrame(rows).to_csv('/mnt/data/v23_fullT_attribution.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False))
