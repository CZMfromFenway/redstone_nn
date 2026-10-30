import sys, os, numpy as np, pandas as pd
sys.path.insert(0,'/mnt/data')
from DTF004_exact_depth2 import build_base, compose, props

out='/mnt/data/v23_exact_pools'
os.makedirs(out,exist_ok=True)
rows=[]
for fullT,mode in [(False,'mixedT'),(True,'fullT')]:
    D,T=build_base(fullT)
    base={'D':D,'T':T}
    for seq in ['DD','DT','TD','TT']:
        funcs={}
        for fi,pi in base[seq[0]].items():
            for fo,po in base[seq[1]].items():
                f=compose(fi,fo)
                if min(f)>=0 and max(f)<=3 and f not in funcs:
                    funcs[f]=(pi,po)
        arr=np.asarray(list(funcs.keys()),dtype=np.uint8)
        np.save(os.path.join(out,f'{mode}_{seq}_03.npy'),arr)
        # Shape statistics for the exact 0..3 pool
        stat={'T_mode':mode,'sequence':seq,'pool_size':len(arr)}
        for fam in ['nondecreasing','nonincreasing','monotone_either','unimodal_peak','continuous','continuous_nondecreasing','continuous_monotone_either','continuous_unimodal_peak']:
            stat[fam]=sum(props(tuple(map(int,r)))[fam] for r in arr)
        # mean number of strict turns and mean total variation
        turns=[]; tv=[]; occupied=[]
        for r in arr.astype(int):
            d=np.diff(r); nz=np.sign(d[d!=0])
            turns.append(int(np.sum(nz[1:]!=nz[:-1])) if len(nz)>1 else 0)
            tv.append(int(np.abs(d).sum()))
            occupied.append(len(np.unique(r)))
        stat['mean_turns']=float(np.mean(turns)); stat['median_turns']=float(np.median(turns)); stat['max_turns']=int(np.max(turns))
        stat['mean_total_variation']=float(np.mean(tv)); stat['mean_levels_used']=float(np.mean(occupied))
        rows.append(stat)
        print(mode,seq,len(arr),flush=True)

pd.DataFrame(rows).to_csv('/mnt/data/v23_exact_pool_geometry.csv',index=False)

# Exact set gains from full-state T over mixed-state T.
gains=[]
for seq in ['DD','DT','TD','TT']:
    a=np.load(os.path.join(out,f'mixedT_{seq}_03.npy'))
    b=np.load(os.path.join(out,f'fullT_{seq}_03.npy'))
    sa={bytes(x) for x in a}; sb={bytes(x) for x in b}
    gains.append({'sequence':seq,'mixed_size':len(sa),'full_size':len(sb),'added_by_fullT':len(sb-sa),'lost':len(sa-sb),'mixed_subset_full':sa.issubset(sb),'gain_pct':100*(len(sb)-len(sa))/len(sa)})
pd.DataFrame(gains).to_csv('/mnt/data/v23_fullT_exact_gains.csv',index=False)
print(pd.DataFrame(gains).to_string(index=False))
