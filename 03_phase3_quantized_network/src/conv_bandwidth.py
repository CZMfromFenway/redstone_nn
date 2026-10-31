# -*- coding: utf-8 -*-
"""卷积矩阵的对角化极限：把 n×n 图上的 3×3 卷积写成 n^2 × n^2 矩阵后，
在"只重排输入"与"输入输出都可重排"两种自由度下，非零元能压到多窄的带里。

模型
    单元 = n×n 图像的一个像素 p=(r,c)，共 n^2 个。
    非零 (p,q) 当且仅当 q 落在 p 的 3×3 邻域内（padding=1, stride=1）：
        max(|r_p - r_q|, |c_p - c_q|) <= 1
    每条非零元是一个权重（一个容器）。

两种自由度
    A. 只重排输入（行序固定为同一个排序 π = σ）：
       W = max_{(p,q) 非零} |π(p) - σ(q)|
    B. 输入、输出都可重排（π ≠ σ）：
       固定列序 σ 后，行 p 的 π(p) 只要落在区间 [max σ(N(p)) - W, min σ(N(p)) + W] 即可，
       是一个"区间匹配"问题——用贪心判定可行性，二分出该 σ 下的最优 W。
       再对 σ 做局部搜索（交换两个标签）。

用法：python src/conv_bandwidth.py [--iters 200000] [--nmax 8]
"""
import argparse
import random
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def build(n):
    """返回 (cells, N[p] 邻域下标列表)。"""
    cells = [(r, c) for r in range(n) for c in range(n)]
    idx = {p: i for i, p in enumerate(cells)}
    N = []
    for (r, c) in cells:
        nb = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < n and 0 <= cc < n:
                    nb.append(idx[(rr, cc)])
        N.append(nb)
    return cells, N


def same_order_W(n):
    """π = σ = 行优先时的带宽，并给出该排序下的验证。"""
    cells, N = build(n)
    lab = {(r, c): r * n + c for r, c in cells}
    W = 0
    for i, p in enumerate(cells):
        for j in N[i]:
            W = max(W, abs(lab[p] - lab[cells[j]]))
    return W


def feasible(N, sigma, W, m):
    """固定列序 sigma（sigma[i] = 像素 i 的列标签），W 是否可行。
    每行 p 的可用标签是区间 [hi-W, lo+W]；区间族用"按右端点排序 + 取最小可用"贪心即可判定。"""
    ivs = []
    for i, nb in enumerate(N):
        lo = min(sigma[j] for j in nb)
        hi = max(sigma[j] for j in nb)
        # 行 p 的 π(p) 必须同时落在 [hi-W, lo+W] 与 [0, m-1] 里
        a, b = max(0, hi - W), min(m - 1, lo + W)
        if a > b:
            return False
        ivs.append((b, a))
    ivs.sort()
    used = [False] * m
    for (b, a) in ivs:
        # 取 >= a 的最小未用标签，且 <= b
        k = a
        while k <= b and used[k]:
            k += 1
        if k > b:
            return False
        used[k] = True
    return True


def best_W_for_sigma(N, sigma, m):
    """该 σ 下的最优 W（二分）。"""
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


def lower_bound(n):
    """两条独立下界：
       ① 计数：每行的 9 个邻域列必须落进 2W+1 个标签里 → 2W+1 >= 邻域大小
       ② 区间半宽：任何一行要求 2W >= (该行邻域的 σ 极差) >= 邻域大小-1
       ⇒ W >= ceil((邻域大小-1)/2)
    """
    cells, N = build(n)
    return max((len(nb) - 1 + 1) // 2 for nb in N), max(len(nb) for nb in N)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=120000)
    ap.add_argument("--nmax", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    print("=" * 96)
    print("卷积矩阵的带宽：n×n 图，3×3 核，padding=1，stride=1")
    print("=" * 96)
    print()
    print("  像素数 n^2，矩阵 n^2 x n^2；每行非零数 = |3x3 邻域|（角 4 / 边 6 / 内 9）")
    print()
    hdr = (f"  {'n':>2s} {'像素':>5s} {'矩阵':>9s} {'非零/行(内)':>10s} "
           f"{'π=σ 最优':>9s} {'计数下界':>7s} {'π≠σ 搜索到':>10s} {'带宽省':>7s}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for n in range(2, a.nmax + 1):
        cells, N = build(n)
        m = n * n
        w_same = same_order_W(n)
        # 计数下界：一行的 |N| 个非零必须落进 2W+1 个槽 → W >= ceil((|N|-1)/2)
        # 注意：不要把 n+1 当独立重排的下界——那个定理的证明要求 π=σ，换序后不成立。
        lb_count, kmax = lower_bound(n)

        # π ≠ σ：从行优先 σ 出发做局部搜索
        sigma = list(range(m))
        best = best_W_for_sigma(N, sigma, m)
        best_sigma = sigma[:]
        cur, cur_sigma = best, sigma[:]
        for it in range(a.iters):
            i, j = rng.randrange(m), rng.randrange(m)
            if i == j:
                continue
            cur_sigma[i], cur_sigma[j] = cur_sigma[j], cur_sigma[i]
            w = best_W_for_sigma(N, cur_sigma, m)
            if w <= cur:
                cur = w
                if w < best:
                    best = w
                    best_sigma = cur_sigma[:]
            else:
                cur_sigma[i], cur_sigma[j] = cur_sigma[j], cur_sigma[i]

        print(f"  {n:2d} {m:5d} {m:4d}x{m:<4d} {kmax:10d} "
              f"{w_same:9d} {lb_count:7d} {best:10d} "
              f"{100*(1-(2*best+1)/(2*w_same+1)):6.0f}%")

    print()
    print("  说明：")
    print("    π=σ 最优 = 行优先取到的带宽，且等于定理下界 (n^2-1)/(n-1) = n+1 —— 即同序下已经最优。")
    print("    π≠σ 搜索到 = 输入与输出都允许重排时，局部搜索找到的最小带宽（上界）。")
    print("    带宽省 = 平面内带面积 (2W+1)/(2W_same+1) 的节省比例。")

    # 平面占用对比（以 n=8 为例给出明细）
    print()
    print("=" * 96)
    print("平面占用：整方阵 vs 带宽带（按每条非零元占一个位置计）")
    print("=" * 96)
    print()
    print(f"  {'n':>2s} {'整方阵位置':>10s} {'真实非零':>9s} {'W=n+1 带':>10s} {'W=4 带':>8s} "
          f"{'带/方阵':>8s}")
    for n in range(2, a.nmax + 1):
        cells, N = build(n)
        m = n * n
        nz = sum(len(nb) for nb in N)
        w_same = same_order_W(n)
        band_same = m * (2 * w_same + 1)
        band4 = m * (2 * 4 + 1)
        print(f"  {n:2d} {m*m:10d} {nz:9d} {band_same:10d} {band4:8d} "
              f"{100*band_same/(m*m):7.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
