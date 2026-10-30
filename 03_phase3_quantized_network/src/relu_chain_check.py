# -*- coding: utf-8 -*-
"""端到端：矩阵乘法头部（白环已减 P−1）→ RELU 除法器电路 → 与编译器整数前向对账。

链路：
    N   = Σ (W·bit − min(0,W))                     矩阵乘法的累加量（公式，已验证 = 仿真头部）
    V   = max(0, N−P+1)                            白环烧 J·K−15nb−(P−1) 后头部直接给出
    q   = 除法器(data/relu.schem, 环烧 c·2^k) 的输出  取反读 8 个火把
    期望 = clip( floor((N−P)/c), 0, 255 )            compiler.py 的 integer_forward

用法：python src/relu_chain_check.py [每个通道的随机输入数]
"""
import copy
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import relu_probe as RP  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = os.path.join(HERE, "..")
B = 8
N = 4


def main():
    rng = __import__("random").Random(20261025)
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 8

    nl = json.load(open(os.path.join(ROOT, "results", "netlist_grayscale.json"), encoding="utf-8"))
    l1 = nl["layers"][0]
    W = l1["W"]            # [64][4][8]
    P = l1["P"]
    C = l1["c"]
    NCH = len(W)

    # 解析除法器一次，作为模板
    sch, inv, idx, SW, SH, SL = RP.load(os.path.join(ROOT, "data", "relu.schem"))
    tmpl = RP.parse(sch, inv, idx, SW, SH, SL)
    rings = [(6, 3 * k + 1), (5, 3 * k + 2), (7, 3 * k + 2), (6, 3 * k + 3)] if False else None

    def divider(V, cval):
        cmps, blocks, torches = tmpl[0].copy(), tmpl[1].copy(), tmpl[2]
        # 深拷贝比较器状态
        cmps = {p: dict(v) for p, v in cmps.items()}
        for k in range(8):
            z0 = 3 * k + 1
            for pos in [(6, z0), (5, z0 + 1), (7, z0 + 1), (6, z0 + 2)]:
                cmps[pos].update(state=cval * (1 << k), src=True, init=cval * (1 << k))
        cmps, charge, locked = RP.evaluate({}, cmps, blocks, extra_sources=[(-1, SL - 1, "west", V)])
        bits = [1 if cmps[(3, 3 * k + 1)]["state"] == 0 else 0 for k in range(8)]
        return sum((1 - b) << k for k, b in enumerate(bits))     # 取反读

    # 参照：编译器公式
    def expect(Nval, p, c):
        v = (Nval - p) // c
        return max(0, min(255, v))

    tot = bad = 0
    fails = []
    for ch in range(NCH):
        xs = [[0, 0, 0, 0], [255, 255, 255, 255]]
        for _ in range(reps):
            xs.append([rng.randrange(256) for _ in range(N)])
        for X in xs:
            nval = 0
            for i in range(N):
                for k in range(B):
                    nval += W[ch][i][k] * ((X[i] >> k) & 1) - min(0, W[ch][i][k])
            V = max(0, nval - P[ch] + 1)
            q = divider(V, C[ch]) if V >= 1 else 0
            want = expect(nval, P[ch], C[ch])
            tot += 1
            if q != want:
                bad += 1
                if len(fails) < 8:
                    fails.append((ch, X, nval, V, q, want))

    print(f"通道 {NCH} × 每组 {reps + 2} 个输入 = {tot} 次端到端")
    print(f"  除法器输出 vs 编译器 clip(floor((N−P)/c))：{tot - bad}/{tot} "
          f"{'✓' if bad == 0 else '✗'}")
    for f in fails:
        print("   ch%d X=%s N=%d V=%d 得 %d 期望 %d" % f)

    # 边界专项：N−P 恰好跨过 c 的整数倍
    edge = 0
    eb = 0
    for ch in range(NCH):
        c, p = C[ch], P[ch]
        for m in range(0, 256):
            for nval in (p + m * c - 1, p + m * c, p + m * c + 1):
                if nval < 0:
                    continue
                V = max(0, nval - p + 1)
                q = divider(V, c) if V >= 1 else 0
                want = expect(nval, p, c)
                edge += 1
                if q != want:
                    eb += 1
    print(f"  边界专项（每通道 N=P+mc−1/P+mc/P+mc+1，共 {edge} 例）："
          f"{edge - eb}/{edge} {'✓' if eb == 0 else '✗'}")
    return 0 if bad == 0 and eb == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
