# -*- coding: utf-8 -*-
"""精确最小跨度：标签式 CSP（MRV + 区间可用性），带 D4 对称破缺。

约束
    每个 3×3 窗口内 9 个标签的跨度 ≤ T（等价于棋盘距离 ≤ k−1 的两格标签差 ≤ T，
    即 G_{k−1} 的带宽 ≤ T）。求最小可行 T，得 S = T + 1。

用法：
    python src/conv_span_exact.py --n 4 --k 3 --tmax 11 --budget 300
"""
import argparse
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def windows(n, k, p):
    W = []
    for a in range(n):
        for b in range(n):
            w = []
            for dr in range(k):
                for dc in range(k):
                    r = a - p + dr
                    c = b - p + dc
                    if 0 <= r < n and 0 <= c < n:
                        w.append(r * n + c)
            W.append(w)
    return W


class Solver:
    def __init__(self, n, k, p, T, budget, zero_cell):
        self.W = windows(n, k, p)
        self.N = n * n
        self.T = T
        self.budget = budget
        self.t0 = time.time()
        self.nodes = 0
        N = self.N
        self.inw = [[] for _ in range(N)]
        for wi, w in enumerate(self.W):
            for c in w:
                self.inw[c].append(wi)
        self.lab = [-1] * N
        self.used = [False] * N
        self.wmin = [None] * len(self.W)
        self.wmax = [None] * len(self.W)
        self.wleft = [len(w) for w in self.W]
        # 对称破缺：指定哪一格拿标签 0
        self.lab[zero_cell] = 0
        self.used[0] = True
        for wi in self.inw[zero_cell]:
            self.wmin[wi] = self.wmax[wi] = 0
            self.wleft[wi] -= 1
        self.filled = 1

    def avail(self, lo, hi):
        lo = max(0, lo)
        hi = min(self.N - 1, hi)
        return sum(1 for L in range(lo, hi + 1) if not self.used[L])

    def dfs(self):
        self.nodes += 1
        if self.nodes % 3000 == 0 and time.time() - self.t0 > self.budget:
            raise TimeoutError
        N, T = self.N, self.T
        if self.filled == N:
            return True
        best_c, best_cands = -1, None
        for c in range(N):
            if self.lab[c] != -1:
                continue
            lo, hi = 0, N - 1
            for wi in self.inw[c]:
                if self.wmin[wi] is not None:
                    lo = max(lo, self.wmax[wi] - T)
                    hi = min(hi, self.wmin[wi] + T)
            if lo > hi:
                return False
            cands = [L for L in range(lo, hi + 1) if not self.used[L]]
            if not cands:
                return False
            if best_cands is None or len(cands) < len(best_cands):
                best_c, best_cands = c, cands
                if len(cands) == 1:
                    break
        c = best_c
        for L in best_cands:
            self.lab[c] = L
            self.used[L] = True
            saved = []
            ok = True
            for wi in self.inw[c]:
                saved.append((wi, self.wmin[wi], self.wmax[wi], self.wleft[wi]))
                self.wleft[wi] -= 1
                self.wmin[wi] = L if self.wmin[wi] is None else min(self.wmin[wi], L)
                self.wmax[wi] = L if self.wmax[wi] is None else max(self.wmax[wi], L)
                if self.wmax[wi] - self.wmin[wi] > T:
                    ok = False
                    break
                if self.wleft[wi] > self.avail(self.wmin[wi] - T, self.wmax[wi] + T):
                    ok = False
                    break
            self.filled += 1
            if ok and self.dfs():
                return True
            self.filled -= 1
            for (wi, mn, mx, lf) in saved:
                self.wmin[wi], self.wmax[wi], self.wleft[wi] = mn, mx, lf
            self.lab[c] = -1
            self.used[L] = False
        return False


def solve(n, k, p, T, budget):
    """对三个 D4 轨道代表各试一次。"""
    reps = sorted({0, 1, n // 2 * n + n // 2} | {n * (n // 2) + 0})
    for zc in [0, 1, (n // 2) * n + n // 2]:
        s = Solver(n, k, p, T, budget, zc)
        try:
            if s.dfs():
                return s.lab
        except TimeoutError:
            return "timeout"
    return None


def span_of(n, k, p, lab):
    W = windows(n, k, p)
    return max(max(lab[c] for c in w) - min(lab[c] for c in w) for w in W)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--tmax", type=int, default=12)
    ap.add_argument("--budget", type=float, default=60.0)
    a = ap.parse_args()
    n, k = a.n, a.k
    p = (k - 1) // 2
    print("=" * 62)
    print(f"精确最小跨度：n={n}，k={k}，下界 S >= k^2 = {k*k}")
    print("=" * 62)
    for T in range(k * k - 1, a.tmax + 1):
        r = solve(n, k, p, T, a.budget)
        if r == "timeout":
            print(f"  T={T:2d}（S={T+1:2d}）  超时（{a.budget:.0f}s）—— 未知")
            return 2
        if r is None:
            print(f"  T={T:2d}（S={T+1:2d}）  ✗ 不存在")
            continue
        print(f"  T={T:2d}（S={T+1:2d}）  ✓ 最小跨度 {span_of(n,k,p,r)+1}")
        for rr in range(n):
            print("      " + " ".join(f"{r[rr*n+c]:3d}" for c in range(n)))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
