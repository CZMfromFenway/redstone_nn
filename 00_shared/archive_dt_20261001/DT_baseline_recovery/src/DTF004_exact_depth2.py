from itertools import product
from collections import defaultdict
import csv, json, math, time

X=range(16)

# ---------- primitives ----------
def c(x,a): return a if x<=a else 0
def s(x,a): return max(0,a-x)
def C(u,v): return u if u>=v else 0
def S(u,v): return max(0,u-v)

def eval_D(pm,am,a,b):
    out=[]
    for x in X:
        p = c(x,a) if pm=='c' else s(x,a)
        A = C(x,b) if am=='C' else S(x,b)
        out.append(max(p,A))
    return tuple(out)

def eval_T(A,p,q,a,b):
    out=[]
    for x in X:
        u = c(x,a) if p=='c' else s(x,a)
        v = c(x,b) if q=='c' else s(x,b)
        out.append(C(u,v) if A=='C' else S(u,v))
    return tuple(out)

def build_base(fullT=False):
    D={}
    for pm in ('c','s'):
      for am in ('C','S'):
       for a in X:
        for b in X:
         f=eval_D(pm,am,a,b)
         D.setdefault(f,('D',pm,am,a,b))
    T={}
    pairs=(('c','s'),('s','c')) if not fullT else tuple(product(('c','s'), repeat=2))
    for A in ('C','S'):
      for p,q in pairs:
       for a in X:
        for b in X:
         f=eval_T(A,p,q,a,b)
         T.setdefault(f,('T',A,p,q,a,b))
    return D,T

def compose(inner,outer):
    return tuple(outer[v] for v in inner)

# ---------- family predicates ----------
def nondec(f): return all(f[i]<=f[i+1] for i in range(15))
def noninc(f): return all(f[i]>=f[i+1] for i in range(15))
def unimodal_peak(f):
    # nondecreasing until the first strict decrease, then nonincreasing forever
    down=False
    for i in range(15):
        if not down:
            if f[i+1] < f[i]: down=True
        else:
            if f[i+1] > f[i]: return False
    return True

def continuous(f): return all(abs(f[i+1]-f[i])<=1 for i in range(15))

def props(f):
    nd=nondec(f); ni=noninc(f); up=unimodal_peak(f); ct=continuous(f)
    return {
      'all': True,
      'nondecreasing': nd,
      'nonincreasing': ni,
      'monotone_either': nd or ni,
      'unimodal_peak': up,
      'continuous': ct,
      'continuous_nondecreasing': ct and nd,
      'continuous_nonincreasing': ct and ni,
      'continuous_monotone_either': ct and (nd or ni),
      'continuous_unimodal_peak': ct and up,
    }

FAMILIES=['all','nondecreasing','nonincreasing','monotone_either','unimodal_peak',
          'continuous','continuous_nondecreasing','continuous_nonincreasing',
          'continuous_monotone_either','continuous_unimodal_peak']

# target-space sizes for q=4,n=16; all/continuous computed by DP

def count_target(family,n=16,q=4):
    # DP over all q-ary sequences, tracking enough state for each predicate.
    if family=='all': return q**n
    if family=='nondecreasing': return math.comb(n+q-1,q-1)
    if family=='nonincreasing': return math.comb(n+q-1,q-1)
    if family=='monotone_either': return 2*math.comb(n+q-1,q-1)-q
    # generic DP states: last, down_started, ever_up, ever_down (for continuous and unimodal)
    # brute DP for requested predicate only.
    from collections import defaultdict
    # state=(last, phase) for peak; for unrestricted continuous only last
    if family in ('unimodal_peak','continuous_unimodal_peak'):
        cont=family.startswith('continuous_')
        dp={(v,0):1 for v in range(q)}
        for _ in range(1,n):
            ndp=defaultdict(int)
            for (prev,phase),cnt in dp.items():
                for v in range(q):
                    if cont and abs(v-prev)>1: continue
                    if phase==0:
                        if v>=prev: ndp[(v,0)]+=cnt
                        else: ndp[(v,1)]+=cnt
                    elif v<=prev: ndp[(v,1)]+=cnt
            dp=ndp
        return sum(dp.values())
    if family=='continuous':
        dp={v:1 for v in range(q)}
        for _ in range(1,n):
            ndp={v:0 for v in range(q)}
            for prev,cnt in dp.items():
                for v in range(q):
                    if abs(v-prev)<=1: ndp[v]+=cnt
            dp=ndp
        return sum(dp.values())
    if family in ('continuous_nondecreasing','continuous_nonincreasing'):
        direction=1 if family.endswith('nondecreasing') else -1
        dp={v:1 for v in range(q)}
        for _ in range(1,n):
            ndp={v:0 for v in range(q)}
            for prev,cnt in dp.items():
                for v in range(q):
                    if abs(v-prev)>1: continue
                    if (direction==1 and v>=prev) or (direction==-1 and v<=prev): ndp[v]+=cnt
            dp=ndp
        return sum(dp.values())
    if family=='continuous_monotone_either':
        one=count_target('continuous_nondecreasing',n,q)
        return 2*one-q
    raise KeyError(family)

TARGET={fam:count_target(fam) for fam in FAMILIES}


def enumerate_depth2(fullT=False, prefix='mixed'):
    D,T=build_base(fullT)
    print(prefix,'base unique D,T',len(D),len(T),flush=True)
    assert len(D)==739
    assert len(T)==(618 if fullT else 409)
    base={'D':D,'T':T}
    seqs=['DD','DT','TD','TT']
    rows=[]
    rep_records=[]
    # store exact represented sets by seq/window for later union/containment and representatives
    store={}
    for seq in seqs:
        t0=time.time()
        inner=base[seq[0]]; outer=base[seq[1]]
        # per-window dict code -> representative configs
        wins=[{} for _ in range(13)]
        for fi,pi in inner.items():
            for fo,po in outer.items():
                f=compose(fi,fo)
                mn=min(f); mx=max(f)
                # fits [lo,lo+3] iff lo<=mn and mx<=lo+3
                lo_min=max(0,mx-3); lo_max=min(12,mn)
                if lo_min<=lo_max:
                    for lo in range(lo_min,lo_max+1):
                        if f not in wins[lo]: wins[lo][f]=(pi,po)
        for lo in range(13):
            famsets={fam:set() for fam in FAMILIES}
            for f in wins[lo]:
                pr=props(f)
                for fam in FAMILIES:
                    if pr[fam]: famsets[fam].add(f)
            store[(seq,lo)]={fam:famsets[fam] for fam in FAMILIES}
            for fam in FAMILIES:
                n=len(famsets[fam]); den=TARGET[fam]
                rows.append({'T_mode':prefix,'sequence':seq,'depth':2,'lo':lo,'hi':lo+3,'family':fam,
                             'represented':n,'target_space':den,'coverage':n/den})
        print(prefix,seq,'done',round(time.time()-t0,2),'s', [len(w) for w in wins],flush=True)
    # union across sequences + exclusive + inclusion
    unionrows=[]; inclusions=[]
    for lo in range(13):
      for fam in FAMILIES:
        sets={s:store[(s,lo)][fam] for s in seqs}
        uni=set().union(*sets.values())
        unionrows.append({'T_mode':prefix,'lo':lo,'hi':lo+3,'family':fam,'union_represented':len(uni),
                          'target_space':TARGET[fam],'union_coverage':len(uni)/TARGET[fam]})
        for a in seqs:
          others=set().union(*(sets[b] for b in seqs if b!=a))
          exclusive=len(sets[a]-others)
          unionrows.append({'T_mode':prefix,'lo':lo,'hi':lo+3,'family':fam,'sequence_exclusive':a,
                            'exclusive_count':exclusive})
          for b in seqs:
            if a==b: continue
            inclusions.append({'T_mode':prefix,'lo':lo,'hi':lo+3,'family':fam,'A':a,'B':b,
                               'A_size':len(sets[a]),'B_size':len(sets[b]),'A_subset_B':sets[a].issubset(sets[b]),
                               'intersection':len(sets[a]&sets[b])})
    # save representative configs for all functions in windows for later plotting, but compact only for mixed
    if prefix=='mixedT':
        for seq in seqs:
          inner=base[seq[0]]; outer=base[seq[1]]
          # rebuild representative dict only for functions that appear in any window; use first hit
          reps={}
          needed=set().union(*(store[(seq,lo)]['all'] for lo in range(13)))
          for fi,pi in inner.items():
            for fo,po in outer.items():
              f=compose(fi,fo)
              if f in needed and f not in reps:
                reps[f]=(pi,po)
                if len(reps)==len(needed): break
            if len(reps)==len(needed): break
          for f,(pi,po) in reps.items():
            rep_records.append({'sequence':seq,'function':','.join(map(str,f)),
                                'inner_params':repr(pi),'outer_params':repr(po)})
    return rows,unionrows,inclusions,rep_records,store

if __name__=='__main__':
    print('TARGET',TARGET,flush=True)
    all_rows=[]; all_union=[]; all_inc=[]; reps=[]
    for full,prefix in [(False,'mixedT'),(True,'fullT')]:
        rows,un,inc,rr,store=enumerate_depth2(full,prefix)
        all_rows += rows; all_union += un; all_inc += inc; reps += rr
    def write_csv(path,rows):
        keys=[]
        for r in rows:
            for k in r:
                if k not in keys: keys.append(k)
        with open(path,'w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    write_csv('/mnt/data/DTF004_depth2_exact.csv',all_rows)
    write_csv('/mnt/data/DTF004_depth2_union_exclusive.csv',all_union)
    write_csv('/mnt/data/DTF004_depth2_inclusions.csv',all_inc)
    write_csv('/mnt/data/DTF004_mixedT_representatives.csv',reps)
    with open('/mnt/data/DTF004_target_space.json','w',encoding='utf8') as f: json.dump(TARGET,f,ensure_ascii=False,indent=2)
    print('DONE')
