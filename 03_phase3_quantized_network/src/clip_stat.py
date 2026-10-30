#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""量化「碰壁」：原始高斯直接截断时，有多少分量被压到边界。"""
import json
import os
import sys
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
P2 = os.path.join(os.path.dirname(ROOT), "02_phase2_avatar")
styles = json.load(open(os.path.join(P2, "webapp", "styles_gauss.json"),
                        encoding="utf-8"))["styles"]

rng = np.random.default_rng(77)
NG = 200000

print("=" * 100)
print("碰壁的量化：原始 (mu, L) 直接截断到 ±3.5")
print("=" * 100)
print(f"  {'风格':10s} {'至少一维碰壁':>13s} {'平均碰壁维数':>14s} "
      f"{'逐维碰壁率（1,2,3,4）':>34s} {'低边界':>8s} {'高边界':>8s}")
print("  " + "-" * 90)
for e in styles:
    mu = np.array(e["mu"], dtype=float)
    L = np.array(e["chol"], dtype=float)
    x = rng.standard_normal((NG, 4)) @ L.T + mu
    hit = np.abs(x) > 3.5
    per_dim = hit.mean(0)
    lo = (x < -3.5).mean()
    hi = (x > 3.5).mean()
    print(f"  {e['name']:10s} {100*hit.any(1).mean():12.1f}% {hit.sum(1).mean():14.3f} "
          f"{'   '.join(f'{100*v:5.1f}%' for v in per_dim):>34s} "
          f"{100*lo:7.2f}% {100*hi:7.2f}%")

print()
print("=" * 100)
print("碰壁如何改变分布：截断后的各维边缘（均值 / 标准差 / 偏度）")
print("=" * 100)
print(f"  {'风格':10s} {'维':>2s} {'原始均值':>9s} {'截断后均值':>10s} {'原始std':>8s} "
      f"{'截断后std':>9s} {'原始偏度':>8s} {'截断后偏度':>10s}")
print("  " + "-" * 76)
for e in styles:
    mu = np.array(e["mu"], dtype=float)
    L = np.array(e["chol"], dtype=float)
    x = rng.standard_normal((NG, 4)) @ L.T + mu
    xc = np.clip(x, -3.5, 3.5)
    for d in range(4):
        a, b = x[:, d], xc[:, d]
        def sk(v):
            m, sd = v.mean(), v.std()
            return float((((v - m) ** 3).mean()) / sd ** 3) if sd > 0 else 0.0
        sk_a, sk_b = sk(a), sk(b)
        flag = "  ←" if abs(sk_b - sk_a) > 0.3 else ""
        print(f"  {e['name'] if d == 0 else '':10s} {d+1:2d} {a.mean():9.3f} "
              f"{b.mean():10.3f} {a.std():8.3f} {b.std():9.3f} "
              f"{sk_a:8.3f} {sk_b:10.3f}{flag}")
print()
print("  偏度变化大 = 分布被截得不对称，人脸一侧被压平。")
print("CLIP_STAT_DONE")
