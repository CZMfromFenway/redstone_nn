# -*- coding: utf-8 -*-
"""复核 data/relu.schem 的函数：用 relu_probe 的锁定式求值，扫 (c, V) 全量。

输出 = 8 位，位 k 由减法器 (3,3k+1) 是否为非零给出（非零 = 1）。
参照 = floor((V-1)/c)，V<1 时为 0。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import relu_probe as RP  # noqa

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
ROOT = os.path.dirname(HERE)


def load_once():
    sch, inv, idx, W, H, L = RP.load(os.path.join(ROOT, "data", "relu.schem"))
    return RP.parse(sch, inv, idx, W, H, L), W, H, L


def run(cmps0, blocks, cval, V, L, src):
    """跑一次，返回 (8 位, 未解析数)。"""
    cmps = {p: dict(v) for p, v in cmps0.items()}
    for k in range(8):
        z0 = 3 * k + 1
        for pos in ((6, z0), (5, z0 + 1), (7, z0 + 1), (6, z0 + 2)):
            cmps[pos].update(state=cval * (1 << k), src=True, init=cval * (1 << k))
    cmps, charge, locked = RP.evaluate({}, cmps, blocks, extra_sources=src(V))
    state = [cmps[(3, 3 * k + 1)]["state"] for k in range(8)]
    unl = sum(1 for c in cmps.values() if c["state"] is None)
    bits = [1 if s not in (None, 0) else 0 for s in state]
    return bits, unl, state


def main():
    (cmps0, blocks, torches), W, H, L = load_once()
    src = lambda V: [(-1, L - 1, "west", V)]

    print("=" * 78)
    print("relu.schem 复核：输出 = floor((V-1)/c)，8 位")
    print("=" * 78)
    cval = 10
    V = 137
    bits, unl, state = run(cmps0, blocks, cval, V, L, src)
    print(f"\n单点 c={cval} V={V}: 未解析 {unl}")
    for k in range(8):
        print(f"  位{k}: 减法器(3,{3*k+1}) = {state[k]!s:>5s}  → 位 {bits[k]}")
    q = sum(b << k for k, b in enumerate(bits))
    print(f"  读得 {q}，floor((V-1)/c) = {(V - 1) // cval}")

    print("\n全量扫描：")
    total = bad = 0
    badlist = []
    for cval in (1, 2, 3, 5, 7, 10, 17, 33):
        top = min(255 * cval + 1, 3000)
        for V in range(0, top + 1):
            bits, unl, state = run(cmps0, blocks, cval, V, L, src)
            if unl:
                badlist.append((cval, V, "未解析 %d" % unl))
                bad += 1
                continue
            q = sum(b << k for k, b in enumerate(bits))
            want = (V - 1) // cval if V >= 1 else 0
            total += 1
            if q != want:
                bad += 1
                if len(badlist) < 8:
                    badlist.append((cval, V, "得 %d 期望 %d" % (q, want)))
        print(f"  c={cval:2d}: V 跑 0..{top} 完成")
    print(f"\n合计 {total} 例，不符 {bad}")
    for b in badlist[:8]:
        print("   ", b)
    return 0


if __name__ == "__main__":
    sys.exit(main())
