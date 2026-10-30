import os,sys,numpy as np,pandas as pd
sys.path.insert(0,'/mnt/data')
from DTF004_exact_depth2 import props
root='/mnt/data/v23_exact_pools'; rows=[]
for seq in ['DT','TD','TT']:
 a=np.load(f'{root}/mixedT_{seq}_03.npy'); b=np.load(f'{root}/fullT_{seq}_03.npy')
 sa={bytes(r) for r in a}; added=[r for r in b if bytes(r) not in sa]
 fams=['nondecreasing','nonincreasing','monotone_either','unimodal_peak','continuous','continuous_nondecreasing','continuous_monotone_either','continuous_unimodal_peak']
 row={'sequence':seq,'added_total':len(added)}
 for f in fams:
  n=sum(props(tuple(map(int,r)))[f] for r in added); row[f]=n; row[f+'_pct']=100*n/len(added)
 rows.append(row)
pd.DataFrame(rows).to_csv('/mnt/data/v23_fullT_added_geometry.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False))
