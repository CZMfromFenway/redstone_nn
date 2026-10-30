import csv, os, math, struct, collections, json
BASE='/mnt/data'
seqs_by_depth={1:['D','T'],2:['DD','DT','TD','TT'],3:['DDD','DDT','DTD','DTT','TDD','TDT','TTD','TTT']}

def bpath(mode,s):
    if mode=='canonical':
        if len(s)<=2 or s=='DDD': return f'{BASE}/can_{s}_binary.tsv'
        return f'{BASE}/can_{s}3_binary.tsv'
    else:
        if s=='DDD': return f'{BASE}/can_DDD_binary.tsv'
        if len(s)<=2: return f'{BASE}/full_{s}_binary.tsv'
        return f'{BASE}/full_{s}3_binary.tsv'

def apath(mode,s):
    if mode=='canonical':
        if len(s)<=2 or s=='DDD': return f'{BASE}/can_{s}_analog.bin'
        return f'{BASE}/can_{s}3_analog.bin'
    else:
        if s=='DDD': return f'{BASE}/can_DDD_analog.bin'
        if len(s)<=2: return f'{BASE}/full_{s}_analog.bin'
        return f'{BASE}/full_{s}3_analog.bin'

def analog_count(mode,s):
    with open(apath(mode,s),'rb') as f:return struct.unpack('<Q',f.read(8))[0]

def read_binary(mode,s):
    d={}
    with open(bpath(mode,s)) as f:
        r=csv.DictReader(f,delimiter='\t')
        for row in r:d[int(row['mask'])]=row
    return d

def rev(mask):
    return sum(((mask>>i)&1)!=((mask>>(i-1))&1) for i in range(1,16))

def kfull(masks):
    c=collections.Counter(rev(m) for m in masks)
    k=-1
    for r in range(16):
        if c[r]==2*math.comb(15,r):k=r
        else:break
    return k,c

def decode_cfg(kind,cfgid,mode):
    mi=cfgid//256; rem=cfgid%256; a=rem//16; b=rem%16
    if kind=='D':
        labels=['D(C,c)','D(S,c)','D(C,s)','D(S,s)']
    elif mode=='canonical':
        labels=['T(C,s,c)','T(S,s,c)','T(C,c,s)','T(S,c,s)']
    else:
        labels=['T(C,c,c)','T(S,c,c)','T(C,c,s)','T(S,c,s)','T(C,s,c)','T(S,s,c)','T(C,s,s)','T(S,s,s)']
    return f'{labels[mi]}[a={a},b={b}]'

def decode_chain(seq,cfgpack,mode):
    parts=[]
    for i,k in enumerate(seq):
        cid=(cfgpack>>(11*i))&0x7FF
        parts.append(decode_cfg(k,cid,mode))
    return ' -> '.join(parts)

# metrics
metric_rows=[]
all_sets={}
for mode in ['canonical','full']:
    for d,seqs in seqs_by_depth.items():
        for s in seqs:
            bd=read_binary(mode,s); masks=set(bd)
            k,c=kfull(masks)
            metric_rows.append({'mode':mode,'depth':d,'sequence':s,'analog_functions':analog_count(mode,s),'binary_functions':len(masks),'k_full':k,'max_reversal':max(c) if c else -1,'volume_chain_est':8*d,'binary_per_volume':len(masks)/(8*d)})
            all_sets[(mode,s)]=masks
with open(f'{BASE}/DT_depth1-3_metrics.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=metric_rows[0].keys());w.writeheader();w.writerows(metric_rows)

# same-depth binary exclusive + subset relations canonical
exclusive=[]
subset=[]
for d,seqs in seqs_by_depth.items():
    if d<2: continue
    sets={s:all_sets[('canonical',s)] for s in seqs}
    for s in seqs:
        other=set().union(*(sets[t] for t in seqs if t!=s))
        exclusive.append((d,s,len(sets[s]),len(sets[s]-other)))
    for a in seqs:
        for b in seqs:
            if a!=b and sets[a] <= sets[b]: subset.append((d,a,b,len(sets[a]),len(sets[b])))
with open(f'{BASE}/DT_binary_subset_relations.tsv','w',newline='') as f:
    w=csv.writer(f,delimiter='\t');w.writerow(['depth','subset','superset','subset_n','superset_n']);w.writerows(subset)
with open(f'{BASE}/DT_binary_exclusive.tsv','w',newline='') as f:
    w=csv.writer(f,delimiter='\t');w.writerow(['depth','sequence','total_binary','exclusive_binary']);w.writerows(exclusive)

# minimal-depth compiler db
for mode in ['canonical','full']:
    candidates=collections.defaultdict(list)
    for d,seqs in seqs_by_depth.items():
        for s in seqs:
            bd=read_binary(mode,s)
            for mask,row in bd.items():
                candidates[mask].append((d,s,row))
    out=[]
    for mask,lst in sorted(candidates.items()):
        md=min(x[0] for x in lst)
        bests=[x for x in lst if x[0]==md]
        # tie: fewer T (lower latency), then lexical; merely representative, all structures retained
        chosen=min(bests,key=lambda x:(x[1].count('T'),x[1]))
        d,s,row=chosen
        cfg=int(row['cfgpack'])
        out.append({
            'mask_decimal':mask,
            'truth_bits_x0_to_x15':''.join(str((mask>>i)&1) for i in range(16)),
            'reversals':rev(mask),
            'min_depth':md,
            'min_volume_chain_est':8*md,
            'all_min_sequences':'|'.join(sorted(x[1] for x in bests)),
            'representative_sequence':s,
            'representative_config':decode_chain(s,cfg,mode),
            'threshold_op':row['op'],
            'threshold_k':row['k'],
            'analog_hex':row['analog_hex'],
            'cfgpack':cfg,
        })
    fn=f'{BASE}/DT_binary_compiler_{mode}_depth1-3.csv'
    with open(fn,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=out[0].keys());w.writeheader();w.writerows(out)

# make concise markdown report
metrics={(r['mode'],r['sequence']):r for r in metric_rows}
can3=seqs_by_depth[3]
full_added=[]
for s in sum(seqs_by_depth.values(),[]):
    a=all_sets[('canonical',s)];b=all_sets[('full',s)]
    full_added.append((s,len(a),len(b),len(b-a)))
union_can3=set().union(*(all_sets[('canonical',s)] for s in can3))
union_full3=set().union(*(all_sets[('full',s)] for s in can3))
kuc,cuc=kfull(union_can3)
report=[]
report.append('# D/T 嵌套函数研究：深度1–3精确枚举（2026-09-26）\n')
report.append('## 口径\n- Canonical-T：仅允许 `C(s,c), S(s,c), C(c,s), S(c,s)`。\n- Full-T：三个比较器状态全部开放，共8种。\n- 模拟函数：完整 `0..15 -> 0..15` 映射。\n- 二值逻辑：对模拟输出允许末端 `>k` 或 `<=k`。\n- `K_full`：从0次反转开始，所有不超过该反转次数的16位二值逻辑均被完整覆盖。\n')
for mode in ['canonical','full']:
    report.append(f'## {mode} 结果\n')
    report.append('|深度|结构|模拟函数|二值逻辑|K_full|最大反转|二值/V(体积约8d)|\n|---:|---|---:|---:|---:|---:|---:|\n')
    for d,seqs in seqs_by_depth.items():
        for s in seqs:
            r=metrics[(mode,s)]
            report.append(f"|{d}|{s}|{r['analog_functions']:,}|{r['binary_functions']:,}|{r['k_full']}|{r['max_reversal']}|{r['binary_per_volume']:.1f}|\n")
report.append('\n## Canonical 三层关键结论\n')
report.append(f'- 三层八结构二值并集：**{len(union_can3):,} / 65,536**，并集 `K_full={kuc}`。\n')
report.append('- `DDT`: 32,088 个二值逻辑，`K_full=5`；`DTT`: 32,338 个，`K_full=5`。\n')
report.append('- `DTD` 只有 15,844 个、`K_full=4`，因此“三层 DDT+DTD 最强”不成立；第一梯队是 DDT/DTT。\n')
report.append('- 二值集合存在严格包含：`DDD⊂DDT`、`DTD⊂DDT`、`DTD⊂DTT`、`TDD⊂TDT`、`TTD⊂TDT`、`TTD⊂TTT`。\n')
report.append('- 同深度独占二值逻辑：DDD=0，DDT=2572，DTD=0，DTT=2804，TDD=0，TDT=644，TTD=0，TTT=772。\n')
report.append('- 但模拟函数集合没有发现上述整集合包含；因此二值网络可以大胆剪弱结构，模拟中间层不能照搬。\n')
report.append('\n## Full-T 增益\n')
for s,a,b,add in full_added:
    report.append(f'- {s}: {a:,} -> {b:,} 二值逻辑，新增 {add:,}。\n')
report.append(f'- 三层总体并集仍为 **{len(union_full3):,}**；相对 Canonical 三层并集新增 **{len(union_full3-union_can3)}**。\n')
report.append('- 结论：额外T状态主要增加模拟函数；到深度3为止并未扩大整体二值函数并集。\n')
report.append('\n## 生成文件\n- `DT_depth1-3_metrics.csv`：完整指标表。\n- `DT_binary_compiler_canonical_depth1-3.csv`：Canonical 真值表 -> 最浅真实D/T配置。\n- `DT_binary_compiler_full_depth1-3.csv`：Full-T 对应数据库。\n- `DT_binary_subset_relations.tsv`：二值集合严格包含关系。\n- `DT_binary_exclusive.tsv`：同深度独占函数数。\n- `canonical_depth3_overlap.tsv`：三层模拟函数两两交集。\n- `canonical_depth3_binary_overlap.tsv`：三层二值逻辑两两交集。\n')
open(f'{BASE}/DT_depth1-3_report_20260926.md','w').write(''.join(report))
print('wrote files')
