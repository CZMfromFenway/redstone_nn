from __future__ import annotations
import numpy as np
from pathlib import Path
import sys,time,math
sys.path.insert(0,'/mnt/data/duelv03src/src')
from count_func_cd_v03 import H, HW

# K-limited 16-point binary edge tables; guaranteed compilable by DT for K<=4,
# DTT for K<=5, DDTT for K<=8 according to current exact K_full results.
def flips(t):
    t=np.asarray(t,dtype=np.int8)
    return int(np.sum(t[1:]!=t[:-1]))

def rand_table(rng,K):
    t=np.zeros(16,dtype=np.int8); t[0]=rng.integers(2)
    k=int(rng.integers(K+1))
    if k:
        loc=set(rng.choice(np.arange(1,16),size=k,replace=False).tolist())
    else: loc=set()
    for i in range(1,16): t[i]=t[i-1]^(i in loc)
    return t

def rand_pred(rng,n,K):
    # predicate over count domain 0..n, extended constant afterward so flips <= K
    L=n+1
    t=np.zeros(L,dtype=np.int8); t[0]=rng.integers(2)
    k=int(rng.integers(min(K,n)+1))
    loc=set(rng.choice(np.arange(1,L),size=k,replace=False).tolist()) if k else set()
    for i in range(1,L): t[i]=t[i-1]^(i in loc)
    return t

def valid_table(t,K): return flips(t)<=K

def valid_pred(t,K): return flips(t)<=K

def best_head_from_hist(pos,neg,h):
    bits=H[:,:h+1]
    errs=(bits*neg[None,:]+(1-bits)*pos[None,:]).sum(1)
    i=int(np.argmin(errs)); return int(errs[i]),i

def hist_counts(total,y,h):
    pos=np.bincount(total[y==1],minlength=h+1).astype(np.int64)
    neg=np.bincount(total[y==0],minlength=h+1).astype(np.int64)
    return pos,neg

def update_hist_candidate(pos,neg,total,idx,new_total,y,h):
    # return candidate hist after changing total[idx] -> new_total
    cp=pos.copy(); cn=neg.copy()
    if len(idx)==0: return cp,cn
    old=total[idx]; yy=y[idx]
    # only changed positions expected but harmless otherwise
    m=(old!=new_total)
    if not np.any(m): return cp,cn
    old=old[m]; new=new_total[m]; yy=yy[m]
    if np.any(yy==1):
        op=np.bincount(old[yy==1],minlength=h+1); np_=np.bincount(new[yy==1],minlength=h+1)
        cp += np_-op
    if np.any(yy==0):
        on=np.bincount(old[yy==0],minlength=h+1); nn=np.bincount(new[yy==0],minlength=h+1)
        cn += nn-on
    return cp,cn

def make_indices(X):
    n=X.shape[1]; out=[]
    for i in range(n):
        out.append([np.where(X[:,i]==z)[0] for z in range(16)])
    return out

def seed_edges(rng,h,n,K):
    E=np.empty((h,n,16),np.int8)
    # mixture of thresholds/intervalish/random
    for j in range(h):
        for i in range(n):
            r=rng.random()
            if r<.35:
                th=int(rng.integers(2,15)); t=np.zeros(16,np.int8); t[th:]=1
            elif r<.55:
                th=int(rng.integers(2,15)); t=np.ones(16,np.int8); t[th:]=0
            elif r<.7 and K>=2:
                a=int(rng.integers(1,13)); b=int(rng.integers(a+1,16)); t=np.zeros(16,np.int8);t[a:b]=1
            else:t=rand_table(rng,K)
            E[j,i]=t
    return E

def init_state(X,y,h,K,rng):
    N,n=X.shape
    E=seed_edges(rng,h,n,K)
    P=np.empty((h,n+1),np.int8)
    C=np.zeros((N,h),np.int8)
    Q=np.zeros((N,h),np.int8)
    for j in range(h):
        for i in range(n): C[:,j]+=E[j,i,X[:,i]]
        # seed predicate broad, include >=half / ==1 / odd-like when valid
        if j%4==0:
            p=np.zeros(n+1,np.int8); p[(n+1)//2:]=1
        elif j%4==1:
            p=np.zeros(n+1,np.int8); p[min(1,n)]=1
        elif j%4==2:
            p=np.array([(c&1) for c in range(n+1)],np.int8)
            if flips(p)>K: p=rand_pred(rng,n,K)
        else:p=rand_pred(rng,n,K)
        P[j]=p; Q[:,j]=p[C[:,j]]
    total=Q.sum(1).astype(np.int16)
    pos,neg=hist_counts(total,y,h); err,head=best_head_from_hist(pos,neg,h)
    return E,P,C,Q,total,pos,neg,err,head

def optimize_count(X,y,K,h,seed=0,restarts=6,passes=10,whole_props=1,verbose=False):
    X=np.asarray(X,dtype=np.int8); y=np.asarray(y,dtype=np.int8); N,n=X.shape
    assert h<=15 and n<=15
    idxs=make_indices(X); rng0=np.random.default_rng(seed)
    best=None; total_evals=0
    for rr in range(restarts):
        rng=np.random.default_rng(int(rng0.integers(2**31-1)))
        E,P,C,Q,total,pos,neg,err,head=init_state(X,y,h,K,rng); evals=1
        for ps in range(passes):
            start=err
            # predicate bits first
            for j in rng.permutation(h):
                for z in rng.permutation(n+1):
                    Pnew=P[j].copy();Pnew[z]^=1
                    if not valid_pred(Pnew,K): continue
                    idx=np.where(C[:,j]==z)[0]
                    if len(idx)==0: P[j,z]^=1; continue
                    nq=np.full(len(idx),Pnew[z],dtype=np.int8); oq=Q[idx,j]
                    d=(nq-oq).astype(np.int16); ch=np.where(d!=0)[0]
                    if len(ch)==0: P[j,z]^=1; continue
                    ii=idx[ch]; nt=total[ii]+d[ch]
                    cp,cn=update_hist_candidate(pos,neg,total,ii,nt,y,h); e2,hd2=best_head_from_hist(cp,cn,h); evals+=1
                    if e2<=err:
                        P[j,z]^=1; Q[ii,j]=nq[ch]; total[ii]=nt; pos,neg=cp,cn; err,head=e2,hd2
            # edge bit flips
            for j in rng.permutation(h):
                for i in rng.permutation(n):
                    for z in rng.permutation(16):
                        Enew=E[j,i].copy();Enew[z]^=1
                        if not valid_table(Enew,K): continue
                        idx=idxs[i][z]
                        delta=1 if Enew[z] else -1
                        nc=(C[idx,j].astype(np.int16)+delta).astype(np.int8)
                        nq=P[j,nc]; oq=Q[idx,j]; d=(nq-oq).astype(np.int16); ch=np.where(d!=0)[0]
                        # even if q unchanged, C must change if accepted; objective ties accepted gives neutral drift.
                        if len(ch)==0:
                            # accept neutral C/E change to allow landscape movement
                            E[j,i,z]^=1; C[idx,j]=nc
                            continue
                        ii=idx[ch]; nt=total[ii]+d[ch]
                        cp,cn=update_hist_candidate(pos,neg,total,ii,nt,y,h); e2,hd2=best_head_from_hist(cp,cn,h); evals+=1
                        if e2<=err:
                            E[j,i,z]^=1; C[idx,j]=nc; Q[ii,j]=nq[ch]; total[ii]=nt; pos,neg=cp,cn; err,head=e2,hd2
            # whole-table proposals to escape bit-flip local minima
            for _ in range(whole_props):
                for j in rng.permutation(h):
                    # predicate replacement
                    pp=rand_pred(rng,n,K); idx=np.arange(N); nq=pp[C[:,j]]; d=(nq-Q[:,j]).astype(np.int16); ch=np.where(d!=0)[0]
                    if len(ch):
                        ii=idx[ch]; nt=total[ii]+d[ch]; cp,cn=update_hist_candidate(pos,neg,total,ii,nt,y,h);e2,hd2=best_head_from_hist(cp,cn,h);evals+=1
                        if e2<err:
                            P[j]=pp;Q[ii,j]=nq[ch];total[ii]=nt;pos,neg=cp,cn;err,head=e2,hd2
                    # one random edge replacement per node
                    i=int(rng.integers(n));tt=rand_table(rng,K); idx=np.arange(N); old=E[j,i,X[:,i]]; new=tt[X[:,i]]; dc=(new-old).astype(np.int16)
                    nc=(C[:,j].astype(np.int16)+dc).astype(np.int8); nq=P[j,nc]; d=(nq-Q[:,j]).astype(np.int16);ch=np.where(d!=0)[0]
                    if len(ch):
                        ii=ch;nt=total[ii]+d[ii];cp,cn=update_hist_candidate(pos,neg,total,ii,nt,y,h);e2,hd2=best_head_from_hist(cp,cn,h);evals+=1
                        if e2<err:
                            E[j,i]=tt;C[:,j]=nc;Q[ii,j]=nq[ii];total[ii]=nt;pos,neg=cp,cn;err,head=e2,hd2
            if verbose: print('rr',rr,'pass',ps,'err',err,'evals',evals,flush=True)
            if err==0 or err==start: break
        total_evals+=evals
        rec={'errors':int(err),'edge_tables':E.copy(),'predicates':P.copy(),'head_id':int(head),'evaluations':int(evals),'restart':rr}
        if best is None or rec['errors']<best['errors']: best=rec
        if err==0: break
    best['total_evaluations']=total_evals
    return best

def predict(model,X):
    E=model['edge_tables'];P=model['predicates'];h,n,_=E.shape;N=len(X); total=np.zeros(N,np.int16)
    for j in range(h):
        c=np.zeros(N,np.int8)
        for i in range(n):c+=E[j,i,X[:,i]]
        total+=P[j,c]
    return H[model['head_id'],total].astype(np.int8)
