import os,itertools,numpy as np,pandas as pd,sys
sys.path.insert(0,'/mnt/data')
from DTF004_exact_depth2 import props
root='/mnt/data/v23_exact_pools'
rows=[]
for mode in ['mixedT','fullT']:
 pools={s:np.load(f'{root}/{mode}_{s}_03.npy') for s in ['DD','DT','TD','TT']}
 sets={s:{bytes(r) for r in a} for s,a in pools.items()}
 for k in [1,2,3,4]:
  for combo in itertools.combinations(['DD','DT','TD','TT'],k):
   u=set().union(*(sets[s] for s in combo))
   arr=np.frombuffer(b''.join(u),dtype=np.uint8).reshape(-1,16) if u else np.empty((0,16),dtype=np.uint8)
   cnt={'T_mode':mode,'families':'+'.join(combo),'num_families':k,'union_size':len(u)}
   # vectorized-ish loop
   fams=['nondecreasing','monotone_either','unimodal_peak','continuous','continuous_nondecreasing','continuous_unimodal_peak']
   cc={f:0 for f in fams}
   for r in arr:
    pr=props(tuple(map(int,r)))
    for f in fams:cc[f]+=pr[f]
   cnt.update(cc); rows.append(cnt)

df=pd.DataFrame(rows)
df.to_csv('/mnt/data/v23_pool_unions.csv',index=False)
print('mixed pairs sorted peak')
print(df[(df.T_mode=='mixedT')&(df.num_families==2)].sort_values('unimodal_peak',ascending=False).to_string(index=False))
print('\nfull pairs sorted nondecreasing')
print(df[(df.T_mode=='fullT')&(df.num_families==2)].sort_values('nondecreasing',ascending=False).to_string(index=False))
