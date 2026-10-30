#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D/T function toolkit (2026-09-28 archive version)

Functions:
  1) eval: evaluate a concrete D/T chain on x=0..15.
  2) lookup: look up a 16-bit binary truth table in the exact depth<=3 compiler DB.

Notation:
  c(x;a) = a if x<=a else 0
  s(x;a) = max(0,a-x)
  C(u;v) = u if u>=v else 0
  S(u;v) = max(0,u-v)

D syntax: D(A,p,a,b)
  A in C/S; p in c/s; a,b in 0..15
  D(x)=max(p(x;a), A(x;b))

T syntax: T(A,p,q,a,b)
  A in C/S; p,q in c/s; a,b in 0..15
  T(x)=A(p(x;a),q(x;b))
  Mixed-mode T requires p!=q. Full-state T also allows p==q.

Chain order is signal-flow order, separated by ';', e.g.
  D(S,c,15,0);T(C,s,c,12,7)
means input -> D -> T.
"""
from __future__ import annotations
import argparse, csv, os, re, sys

X = list(range(16))

def c(x,a): return a if x <= a else 0
def s(x,a): return max(0,a-x)
def C(u,v): return u if u >= v else 0
def S(u,v): return max(0,u-v)

def passive(kind,x,a):
    if kind == 'c': return c(x,a)
    if kind == 's': return s(x,a)
    raise ValueError(f'bad passive mode: {kind}')

def active(kind,u,v):
    if kind == 'C': return C(u,v)
    if kind == 'S': return S(u,v)
    raise ValueError(f'bad active mode: {kind}')

def D(A,p,a,b,x):
    return max(passive(p,x,a), active(A,x,b))

def T(A,p,q,a,b,x):
    return active(A, passive(p,x,a), passive(q,x,b))

PAT_D = re.compile(r'^D\(([CS]),([cs]),(\d+),(\d+)\)$')
PAT_T = re.compile(r'^T\(([CS]),([cs]),([cs]),(\d+),(\d+)\)$')

def parse_chain(text):
    nodes=[]
    for raw in [z.strip() for z in text.split(';') if z.strip()]:
        m=PAT_D.match(raw)
        if m:
            A,p,a,b=m.groups(); a=int(a); b=int(b)
            if not(0<=a<=15 and 0<=b<=15): raise ValueError(raw)
            nodes.append(('D',(A,p,a,b)))
            continue
        m=PAT_T.match(raw)
        if m:
            A,p,q,a,b=m.groups(); a=int(a); b=int(b)
            if not(0<=a<=15 and 0<=b<=15): raise ValueError(raw)
            nodes.append(('T',(A,p,q,a,b)))
            continue
        raise ValueError(f'cannot parse node: {raw}')
    if not nodes: raise ValueError('empty chain')
    return nodes

def eval_chain(nodes):
    out=[]
    for x0 in X:
        y=x0
        for typ,cfg in nodes:
            y = D(*cfg,y) if typ=='D' else T(*cfg,y)
        out.append(y)
    return out

def parse_threshold(spec):
    if spec is None: return None
    m=re.match(r'^(gt|le):(\d+)$',spec)
    if not m: raise ValueError('threshold must look like gt:7 or le:7')
    op,k=m.group(1),int(m.group(2))
    if not 0<=k<=15: raise ValueError('threshold k must be 0..15')
    return op,k

def binarize(vals,thr):
    if thr is None: return None
    op,k=thr
    return [1 if (v>k if op=='gt' else v<=k) else 0 for v in vals]

def bits_to_mask(bits):
    bits=''.join(ch for ch in bits.strip() if ch in '01')
    if len(bits)!=16: raise ValueError('need exactly 16 bits in x=0..15 order')
    return sum((1<<i) for i,ch in enumerate(bits) if ch=='1'), bits

def reversals(bits):
    return sum(bits[i]!=bits[i-1] for i in range(1,len(bits)))

def default_db(mode):
    root=os.path.dirname(os.path.abspath(__file__))
    cand = ('DT_binary_compiler_mixedT_depth1-3.csv' if mode=='mixed'
            else 'DT_binary_compiler_fullT_depth1-3.csv')
    paths=[os.path.join(root,'..','data',cand),os.path.join(root,cand),cand]
    for p in paths:
        if os.path.exists(p): return p
    return paths[0]

def cmd_eval(args):
    nodes=parse_chain(args.chain)
    vals=eval_chain(nodes)
    print('x      :', ' '.join(f'{x:2d}' for x in X))
    print('analog :', ' '.join(f'{v:2d}' for v in vals))
    thr=parse_threshold(args.threshold)
    if thr:
        bits=binarize(vals,thr)
        print('binary :', ''.join(map(str,bits)), f'(reversals={reversals(bits)})')

def cmd_lookup(args):
    mask,bits=bits_to_mask(args.bits)
    db=args.db or default_db(args.mode)
    if not os.path.exists(db):
        print(f'DB not found: {db}',file=sys.stderr); sys.exit(2)
    hit=None
    with open(db,'r',encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):
            if int(row['mask_decimal'])==mask:
                hit=row; break
    r=reversals([int(c) for c in bits])
    print(f'target bits x0..x15: {bits}')
    print(f'mask_decimal: {mask}; reversals: {r}')
    if hit:
        print('FOUND exact solution in depth<=3 database')
        for key in ['min_depth','all_min_sequences','representative_sequence','representative_config','threshold_op','threshold_k','analog_hex']:
            print(f'{key}: {hit.get(key)}')
    else:
        print('No exact solution in current depth<=3 compiler DB.')
        if r<=8:
            print('Known theorem/data point: every binary truth table with <=8 reversals has some depth-4 DDTT implementation (K_full=8), but this archive does not yet contain a depth-4 witness-parameter database.')
        else:
            print('Depth-4 union covers 65,100/65,536 binary tables, but without a depth-4 witness DB this tool cannot decide membership for this target.')

def main():
    ap=argparse.ArgumentParser(description='D/T redstone function toolkit')
    sub=ap.add_subparsers(dest='cmd',required=True)
    p=sub.add_parser('eval',help='evaluate a concrete D/T chain')
    p.add_argument('--chain',required=True,help='e.g. D(S,c,15,0);T(C,s,c,12,7)')
    p.add_argument('--threshold',help='optional gt:k or le:k')
    p.set_defaults(func=cmd_eval)
    p=sub.add_parser('lookup',help='lookup binary truth table in exact depth<=3 DB')
    p.add_argument('--bits',required=True,help='16 bits in x=0..15 order')
    p.add_argument('--mode',choices=['mixed','full'],default='mixed',help='T state set; mixed=one c and one s in two passive branches')
    p.add_argument('--db',help='override compiler CSV path')
    p.set_defaults(func=cmd_lookup)
    args=ap.parse_args(); args.func(args)

if __name__=='__main__': main()
