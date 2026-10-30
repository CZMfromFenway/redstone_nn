from pathlib import Path
import hashlib,json,numpy as np
ROOT=Path(__file__).resolve().parent
X=np.indices((16,)*4,dtype=np.int8).reshape(4,-1).T
H=np.load(ROOT/'legacy_H.npy',allow_pickle=False)
records=json.loads((ROOT/'teacher_validation.json').read_text())
for rec in records:
    name=rec['name'];p=ROOT/'teachers'/name
    y=np.load(str(p)+'.npy',allow_pickle=False);m=json.loads(Path(str(p)+'.json').read_text())
    assert y.shape==(65536,) and np.isin(y,[0,1]).all()
    assert hashlib.sha256(Path(str(p)+'.npy').read_bytes()).hexdigest()==rec['npy_sha256']
    assert hashlib.sha256(y.astype(np.uint8).tobytes()).hexdigest()==rec['label_sha256']
    if 'MLP' in name:
        yp=np.tanh(((X.astype(float)-7.5)/7.5)@np.array(m['W1']).T+np.array(m['b1']))@np.array(m['W2'])+m['b2']>=0
    else:
        E=np.array(m['E']);P=np.array(m['P']);Q=[]
        for j in range(m['h']):Q.append(P[j,sum(E[j,i,X[:,i]] for i in range(4))])
        yp=H[m['head_id'],np.stack(Q,1).sum(1)]
    delta=int(np.sum(y!=yp));assert delta==0
    print(name,'positives=',int(y.sum()),'JSON mismatches=',delta,'SHA256 OK')
print('All four original Teachers verified; x4 varies fastest.')
