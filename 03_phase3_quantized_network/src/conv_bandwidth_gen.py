# -*- coding: utf-8 -*-
"""卷积矩阵的带宽：一般化到任意核尺寸 k、步长 s、填充 p。

问题
    n×n 输入、k×k 核、stride=s、padding=p → 输出 m×m，
        m = (n + 2p - k)//s + 1
    矩阵 M[o][i]：行 = 输出 o=(a,b)，列 = 输入 i=(r,c)。
    非零 ⟺ r ∈ [a·s - p, a·s - p + k - 1] 且 c ∈ [b·s - p, ...]，且 0 <= r,c < n。

要压的数
    W = min over (行序 π, 列序 σ) of  max_{非零} |π(o) - σ(i)|
    平面里一行的横向跨度（"19"那个数）= 2W + 1。

三种排序都算
    A 行优先（朴素）           π = σ = 行优先
    B 中心映射（把输出按它在输入格点上的中心排）  π 由匹配给出，σ = 行优先
    C 搜索（在 B 基础上交换 σ，π 每步用区间匹配重算）

用法：python src/conv_bandwidth_gen.py [--n 8] [--iters 20000]
"""
import argparse
import itertools
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def neigh(n, k, s, p):
    """每个输出通道读哪些输入下标（行优先下标 r*n+c）。"""
    m = (n + 2 * p - k) // s + 1
    out = []
    for a in range(m):
        for b in range(m):
            lst = []
            for dr in range(k):
                for dc in range(k):
                    r = a * s - p + dr
                    c = b * s - p + dc
                    if 0 <= r < n and 0 <= c < n:
                        lst.append(r * n + c)
            out.append(lst)
    return m, out


def feasible(N, sigma, W, m):
    """固定列序 sigma 下，W 是否可行（行用区间匹配判定）。"""
    ivs = []
    for nb in N:
        if not nb:
            return False
        lo = min(sigma[j] for j in nb)
        hi = max(sigma[j] for j in nb)
        lo2, hi2 = max(0, hi - W), min(m - 1, lo + W)
        if lo2 > hi2:
            return False
        ivs.append((hi2, lo2))
    ivs.sort()
    used = [False] * m
    for (b, a) in ivs:
        t = a
        while t <= b and used[t]:
            t += 1
        if t > b:
            return False
        used[t] = True
    return True


def best_W(N, sigma, m):
    lo, hi = 0, m - 1
    if feasible(N, sigma, 0, m):
        return 0
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if feasible(N, sigma, mid, m):
            hi = mid
        else:
            lo = mid
    return hi


def span_sigma(N, sigma):
    """σ 给定时的最大行内列跨度：max over 行 of (max σ(N_row) - min σ(N_row))。
       这个量与行序 π 无关——每行的横向宽度只由列序决定。"""
    worst = 0
    for nb in N:
        s = [sigma[j] for j in nb]
        d = max(s) - min(s)
        if d > worst:
            worst = d
    return worst


def sigma_block(n, k, s):
    """块内连续排法：按 (块行, 块列, 块内行, 块内列) 排序输入。
       s >= k 时每个窗口整块落在同一块里，所以能压到 k^2。"""
    order = sorted(range(n * n),
                   key=lambda i: ((i // n) // s, (i % n) // s, (i // n) % s, (i % n) % s))
    sigma = [0] * (n * n)
    for lbl, idx in enumerate(order):
        sigma[idx] = lbl
    return sigma


def w_identity(N):
    """π = σ = 行优先时的带宽。"""
    m = len(N)
    W = 0
    for o, nb in enumerate(N):
        for j in nb:
            W = max(W, abs(o - j))
    return W


def sigma_center(n, k, s, p):
    """中心映射：输出 (a,b) 的中心在输入格点上的位置，按此排序输入。
    等价于"按输入格点坐标排"，使 |Δ| 尽量只含 δ_r n + δ_c。"""
    m = (n + 2 * p - k) // s + 1
    ctr = []
    for a in range(m):
        for b in range(m):
            r = a * s - p + (k - 1) // 2
            c = b * s - p + (k - 1) // 2
            ctr.append((max(0, min(n - 1, r)), max(0, min(n - 1, c))))
    order = sorted(range(n * n), key=lambda i: (i // n, i % n))
    return order, ctr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    n = a.n

    cases = [
        (1, 1, 0), (3, 1, 0), (3, 1, 1), (3, 1, 2),
        (3, 2, 0), (3, 2, 1), (3, 2, 2),
        (3, 3, 0), (3, 3, 1), (3, 4, 0),
        (5, 1, 0), (5, 1, 2), (5, 2, 2), (5, 1, 4),
        (7, 1, 3), (7, 1, 0),
    ]
    print("=" * 104)
    print(f"卷积矩阵的带宽与横向跨度（n={n}，行序/列序都可重排）")
    print("=" * 104)
    print()
    hdr = (f"  {'k':>2s} {'s':>2s} {'p':>2s} {'输出':>5s} {'矩阵':>9s} {'非零':>6s} "
           f"{'每行最多':>7s} {'行优先跨度':>10s} {'块内连续':>8s} {'最优':>5s} {'下界k^2':>8s} {'s=1公式':>9s}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 3))

    import random
    rng = random.Random(a.seed)
    rows = []
    for (k, s, p) in cases:
        m, N = neigh(n, k, s, p)
        NN = n * n
        M = m * m
        if M == 0 or M > NN:
            print(f"  {k:2d} {s:2d} {p:2d}   —— 输出数为 {M}，跳过")
            continue
        nz = sum(len(x) for x in N)

        # 行优先列序
        sigma = list(range(NN))
        sp0 = span_sigma(N, sigma)

        # 交换式局部搜索：直接压"最大行内列跨度"（与行序 π 无关）
        cur_s, cur = sigma[:], sp0
        best, best_s = sp0, sigma[:]
        for _ in range(a.iters):
            i, j = rng.randrange(NN), rng.randrange(NN)
            if i == j:
                continue
            cur_s[i], cur_s[j] = cur_s[j], cur_s[i]
            sp = span_sigma(N, cur_s)
            if sp <= cur:
                cur = sp
                if sp < best:
                    best, best_s = sp, cur_s[:]
            else:
                cur_s[i], cur_s[j] = cur_s[j], cur_s[i]

        kmax = max(len(x) for x in N)
        # 块内连续排法（s >= k 时应当压到 k^2）
        spb = span_sigma(N, sigma_block(n, k, s))
        best = min(best, spb)

        formula = (k - 1) * (n + 1) + 1 if s == 1 else None
        rows.append((k, s, p, m, M, nz, kmax, sp0 + 1, best + 1))
        print(f"  {k:2d} {s:2d} {p:2d} {m:5d} {m:3d}x{m:<5d} {nz:6d} "
              f"{kmax:7d} {sp0+1:10d} {spb+1:8d} {best+1:7d} {kmax*kmax:8d} "
              f"{(str(formula) if formula else '—'):>9s}")

    print()
    print("  列含义：k/s/p = 核/步长/填充；『每行最多』= k^2（内部行）；")
    print("  『行优先跨度』= 行优先列序下每行触及的列数；『块内连续』= s>=k 时把每个窗口整块排在连续标签上；")
    print("  『最优』= 两者与搜索取小；『下界 k^2』= 一行 k^2 个非零，跨度至少 k^2；『s=1 公式』= (k-1)(n+1)+1。")
    print()
    print("  校验 s=1：(k-1)(n+1)+1 =", ", ".join(
        f"k={k}:{(k-1)*(n+1)+1}" for k in (1, 3, 5, 7)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
