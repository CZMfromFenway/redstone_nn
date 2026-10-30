from pathlib import Path
import sys,time,numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from count_general import optimize_count,predict
X=np.indices((16,)*4,dtype=np.int8).reshape(4,-1).T
y=np.load(ROOT/'teachers/N4_MLP_T1.npy',allow_pickle=False)
t=time.time()
m=optimize_count(X,y,4,12,seed=940000+4*10000+12*101,restarts=8,passes=12,whole_props=2)
pred=predict(m,X)
e=int(np.sum(pred!=y))
assert e==m['errors']==5257,(e,m['errors'])
print('Original MLP T1 / K4 / h12:',e,'errors; historical result matches; wall seconds:',time.time()-t)
