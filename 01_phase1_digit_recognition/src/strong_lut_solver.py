#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
强模 LUT 单列求解器
====================
给定目标 f: {0..15} -> Z (模拟信号, 不限于 0~15), 判断能否用
  lambda(x) = B  - S(x)                 (顺式)
  lambda(x) = Bt - (B - S(x))           (反式)
表示, 其中 S(x) = sum_k c_k(x),
  c_k(x) = a_k(x) 若 a_k(x) >= 15, 否则 c_k(x) = b_k
  a_k(x) = 14 + k - x            (比较器截断)
  a_k(x) = min(15, 14 + k - x)   (红石粉截断)
约束: b_0 = 0 (固定), b_t ∈ [0, 897].

反解公式 (已由前两轮测试验证):
  红石粉截断  ΔS(x) = b_{x+1} - 15      =>  b_{x+1} = 15 - Δλ(x)
  比较器截断  ΔS(x) = b_{x+1} - 29 + x  =>  b_{x+1} = (29 - x) - Δλ(x)
  (反式: 对 -f 用同样的式子)
"""
import numpy as np, json, os

B_MIN, B_MAX = 0, 897
B0 = 0

# ---------------- 两种物理机制 ----------------
def a_of(k, x, mode):
    return (14 + k - x) if mode == "comp" else min(15, 14 + k - x)

def c_of(b, k, x, mode):
    a = a_of(k, x, mode)
    return a if a >= 15 else int(b[k])

def S_of(b, x, mode):
    return sum(c_of(b, k, x, mode) for k in range(16))

# ---------------- 反解 b ----------------
def solve_b(target, mode):
    """
    尝试用顺式表示 target。
    返回 (ok, b, B, 说明)
    """
    t = np.asarray(target, dtype=np.int64)
    d = np.diff(t)                       # Δλ(x), x=0..14
    b = np.zeros(16, dtype=np.int64)
    b[0] = B0
    for x in range(15):
        if mode == "dust":
            bx = 15 - int(d[x])
        else:
            bx = (29 - x) - int(d[x])
        if not (B_MIN <= bx <= B_MAX):
            return False, None, None, f"b_{x+1}={bx} 超出 [{B_MIN},{B_MAX}]"
        b[x + 1] = bx
    # 用解出的 b 正向重建, 校验
    S = np.array([S_of(b, x, mode) for x in range(16)], dtype=np.int64)
    B = int(t[0]) + int(S[0])            # lambda = B - S  =>  B = lambda(0) + S(0)
    rec = B - S
    if not np.array_equal(rec, t):
        return False, None, None, f"重建不符 (最大偏差 {int(np.max(np.abs(rec-t)))})"
    return True, b, B, "ok"

def solve_forward(target, mode):
    return solve_b(target, mode)

def solve_reverse(target, mode):
    """反式: lambda = Bt - (B - S) = Bt - B + S"""
    t = np.asarray(target, dtype=np.int64)
    ok, b, B, msg = solve_b(-t, mode)
    if not ok:
        return False, None, None, None, msg
    S = np.array([S_of(b, x, mode) for x in range(16)], dtype=np.int64)
    Bt = int(t[0]) + B - int(S[0])       # t = Bt - B + S
    return True, b, B, Bt, "ok"

# ---------------- 总入口 ----------------
def solve(target, verbose=True):
    t = np.asarray(target, dtype=np.int64)
    results = []
    for form, fn in (("顺式", solve_forward), ("反式", solve_reverse)):
        for mode, mname in (("comp", "比较器截断"), ("dust", "红石粉截断")):
            r = fn(t, mode)
            if form == "顺式":
                ok, b, B, msg = r
                results.append({"form": form, "mode": mname, "ok": ok, "b": None if not ok else b.tolist(),
                                "B": B, "Bt": None, "msg": msg})
            else:
                ok, b, B, Bt, msg = r
                results.append({"form": form, "mode": mname, "ok": ok, "b": None if not ok else b.tolist(),
                                "B": B, "Bt": Bt, "msg": msg})

    if verbose:
        print("=" * 88)
        print("目标 f(x):", t.tolist())
        print("=" * 88)
        for r in results:
            tag = "可行" if r["ok"] else "不可行"
            print(f"\n[{r['form']} + {r['mode']}] {tag}   ({r['msg']})")
            if r["ok"]:
                print(f"   b_0..b_15 = {r['b']}")
                print(f"   b_1..b_15 = {r['b'][1:]}")
                if r["B"] is not None:
                    print(f"   B  = {r['B']}")
                if r["Bt"] is not None:
                    print(f"   Bt = {r['Bt']}")
    return results

# ---------------- 自检 ----------------
def self_test():
    print("#" * 88)
    print("自检 1: czm 第一组测试 (顺式, comp, B=330)")
    b1 = [0, 21, 21, 34, 26, 26, 24, 15, 30, 28, 5, 19, 33, 17, 8, 8]
    lam1 = [0, 8, 15, 8, 8, 7, 7, 15, 7, 0, 15, 15, 0, 0, 8, 15]
    # 用该 b 正向生成, 再交给求解器
    S = np.array([S_of(b1, x, "comp") for x in range(16)])
    tgt = 330 - S
    print("   用 b 生成的目标:", tgt.tolist())
    assert tgt.tolist() == lam1, (tgt.tolist(), lam1)
    r = solve(tgt, verbose=False)
    hit = [q for q in r if q["ok"]]
    print(f"   求解器找到 {len(hit)} 组可行解")
    for q in hit:
        print(f"      {q['form']}+{q['mode']}: b_1..15={q['b'][1:]}  B={q['B']} Bt={q['Bt']}")

    print("\n" + "#" * 88)
    print("自检 2: czm 第二组测试 (顺式, dust, B=225)")
    b2 = [0, 3, 24, 4, 21, 15, 18, 12, 19, 7, 15, 15, 15, 21, 19, 17]
    lam2 = [0, 12, 3, 14, 8, 8, 5, 8, 4, 12, 12, 12, 12, 6, 2, 0]
    S2 = np.array([S_of(b2, x, "dust") for x in range(16)])
    tgt2 = 225 - S2
    print("   用 b 生成的目标:", tgt2.tolist())
    assert tgt2.tolist() == lam2, (tgt2.tolist(), lam2)
    r2 = solve(tgt2, verbose=False)
    hit2 = [q for q in r2 if q["ok"]]
    print(f"   求解器找到 {len(hit2)} 组可行解")
    for q in hit2:
        print(f"      {q['form']}+{q['mode']}: b_1..15={q['b'][1:]}  B={q['B']} Bt={q['Bt']}")

def coverage_scan():
    """随机目标函数在不同跳跃幅度下的可表示率"""
    rng = np.random.default_rng(2026)
    print("\n" + "=" * 88)
    print("可表示率扫描: 随机 f(0..15), 相邻差均匀取自 [-J, J]")
    print("=" * 88)
    print(f"{'J':>6} | {'顺式comp':>9} {'顺式dust':>9} {'反式comp':>9} {'反式dust':>9} | {'至少一种':>9}")
    print("-" * 88)
    rows = []
    for J in (5, 15, 30, 100, 300, 900, 3000):
        cnt = {"顺式comp": 0, "顺式dust": 0, "反式comp": 0, "反式dust": 0, "any": 0}
        M = 300
        for _ in range(M):
            d = rng.integers(-J, J + 1, size=15)
            t = np.concatenate([[0], np.cumsum(d)]).astype(np.int64)
            got = False
            for q in solve(t, verbose=False):
                key = q["form"] + ("comp" if q["mode"] == "比较器截断" else "dust")
                if q["ok"]:
                    cnt[key] += 1
                    got = True
            cnt["any"] += int(got)
        print(f"{J:6d} | {cnt['顺式comp']/M*100:8.1f}% {cnt['顺式dust']/M*100:8.1f}% "
              f"{cnt['反式comp']/M*100:8.1f}% {cnt['反式dust']/M*100:8.1f}% | {cnt['any']/M*100:8.1f}%")
        rows.append((J, cnt, M))

    print("\n[解读]")
    print("   - 红石粉截断的可行条件就是『每个相邻差落在常数窗口内』, 所以 J<=15 时几乎必然可行;")
    print("   - 比较器截断的窗口是 [-(868+x), 29-x], 比红石粉截断略宽一点(多两侧各 14),")
    print("     因为死区项 a_k 在 x 较小时给出额外的正余量;")
    print("   - 真正做限制的是『同一步里顺式与反式各自只能覆盖一个方向』这件事:")
    print("     一个既有大正跳变又有大负跳变的函数, 必须靠某一种形式, 而该形式只容忍一个方向。")
    return rows


if __name__ == "__main__":
    self_test()
    coverage_scan()

