import sys,pandas as pd
sys.path.insert(0,'/mnt/data')
from DTF004_exact_depth2 import build_base, eval_T, compose
X=range(16); D,_=build_base(False)
def T(pairs):
 o={}
 for A in ('C','S'):
  for p,q in pairs:
   for a in X:
    for b in X:o.setdefault(eval_T(A,p,q,a,b),1)
 return o
mix=T((('c','s'),('s','c'))); cc=T((('c','c'),)); ss=T((('s','s'),)); same={**cc,**ss}; full={**mix,**same}
def comp(A,B):
 s=set()
 for a in A:
  for b in B:
   f=compose(a,b)
   if max(f)<=3:s.add(f)
 return s
sm=comp(mix,mix); sf=comp(full,full); add=sf-sm
cats={'mix->same':comp(mix,same),'same->mix':comp(same,mix),'same->same':comp(same,same),'cc->mix':comp(cc,mix),'ss->mix':comp(ss,mix),'mix->cc':comp(mix,cc),'mix->ss':comp(mix,ss)}
rows=[]
for k,v in cats.items():rows.append({'category':k,'pool_size':len(v),'new_fullT_functions_reached':len(add&v),'fraction_of_all_added':len(add&v)/len(add)})
pd.DataFrame(rows).to_csv('/mnt/data/v23_TT_fullT_attribution.csv',index=False)
print('TT mixed',len(sm),'full',len(sf),'added',len(add))
print(pd.DataFrame(rows).to_string(index=False))
