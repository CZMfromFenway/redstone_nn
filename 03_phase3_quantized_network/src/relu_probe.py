# -*- coding: utf-8 -*-
"""反解 data/relu.schem：8 位除法器（restoring division）结构探针。

电路规则与 plane_sim.html 的 evalBoard 同源：
    comparator：facing = "输入来向"（rear 在 facing 方向），输出打到反方向那格
                compare: out = rear if rear>=side else 0
                subtract: out = max(0, rear-side)
    实心块：被比较器打到就带电，值 = 所有充能者的最大值（全部充能者锁定后才可读）
    墙上的火把：亮 ⟺ 支撑块不带电（支撑 = facing 的反方向）；它只做指示，不给别的东西供电
    环：4 个比较器 + 4 个角块构成自锁环，四个角恒等于烧录值

用法：python src/relu_probe.py [-c 10] [-V 137]
"""
import gzip
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from schem_verify_stack import R, varints  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = os.path.join(HERE, "..")
DV = {"north": (0, -1), "south": (0, 1), "east": (1, 0), "west": (-1, 0)}
COLOR_OF = {"red": 0, "orange": 1, "yellow": 2, "lime": 3, "cyan": 4,
            "blue": 5, "purple": 6, "magenta": 7}


def load(path):
    raw = gzip.decompress(open(path, "rb").read())
    r = R(raw); r.u1(); r.name()
    sch = r.payload(10)["Schematic"]
    W, H, L = sch["Width"], sch["Height"], sch["Length"]
    blk = sch["Blocks"]
    inv = {v: k for k, v in blk["Palette"].items()}
    idx = varints(blk["Data"])
    return sch, inv, idx, W, H, L


def parse(sch, inv, idx, W, H, L, y=0):
    cmps, blocks, torches = {}, {}, {}
    for z in range(L):
        for x in range(W):
            st = inv[idx[(y * L + z) * W + x]]
            if st == "minecraft:air":
                continue
            if "comparator" in st:
                f = st.split("facing=")[1].split(",")[0]
                cmps[(x, z)] = {
                    "facing": f,
                    "mode": "subtract" if "subtract" in st else "compare",
                    "state": None, "src": False, "init": 0,
                }
            elif "torch" in st:
                f = st.split("facing=")[1].split(",")[0] if "facing=" in st else None
                torches[(x, z)] = {"facing": f, "lit": "lit=true" in st}
            else:
                c = st.replace("minecraft:", "").replace("_concrete", "")
                blocks[(x, z)] = {"color": COLOR_OF.get(c, -1)}
    return cmps, blocks, torches


def evaluate(cells, cmps, blocks, extra_sources=()):
    """拓扑推进求值（与 evalBoard 同构）。extra_sources: [(x,z,facing,value)] 外部注入。"""
    for (x, z, f, v) in extra_sources:
        cmps[(x, z)] = {"facing": f, "mode": "compare", "state": v, "src": True, "init": v}

    # 每个块被哪些比较器充能
    chargers = {}
    for pos, c in cmps.items():
        fv = DV[c["facing"]]
        tgt = (pos[0] - fv[0], pos[1] - fv[1])
        chargers.setdefault(tgt, []).append(pos)

    charge = {}
    locked = {pos for pos, c in cmps.items() if c["src"]}
    for pos in locked:
        c = cmps[pos]
        fv = DV[c["facing"]]
        tgt = (pos[0] - fv[0], pos[1] - fv[1])
        if tgt in blocks:
            charge[tgt] = max(charge.get(tgt, 0), c["init"])

    def rear_val(pos, c):
        fv = DV[c["facing"]]
        r_pos = (pos[0] + fv[0], pos[1] + fv[1])
        if r_pos in cmps:
            r = cmps[r_pos]
            if r["state"] is None:
                return None
            return r["state"]
        if r_pos in blocks:
            ch = chargers.get(r_pos, [])
            if not ch:
                return 0
            if any(p not in locked for p in ch):
                return None
            return charge.get(r_pos, 0)
        return 0

    def side_cmps(pos, c):
        fv = DV[c["facing"]]
        out = []
        for s in ((-fv[1], fv[0]), (fv[1], -fv[0])):
            n = (pos[0] + s[0], pos[1] + s[1])
            if n in cmps:
                nf = DV[cmps[n]["facing"]]
                if (n[0] - nf[0], n[1] - nf[1]) == pos:
                    out.append(n)
        return out

    changed = True
    guard = 0
    while changed and guard < 10000:
        changed = False
        guard += 1
        for pos, c in cmps.items():
            if pos in locked:
                continue
            rv = rear_val(pos, c)
            if rv is None:
                continue
            sides = side_cmps(pos, c)
            if any(s not in locked for s in sides):
                continue
            sv = max([cmps[s]["state"] for s in sides], default=0)
            v = max(0, rv - sv) if c["mode"] == "subtract" else (rv if rv >= sv else 0)
            c["state"] = v
            locked.add(pos)
            fv = DV[c["facing"]]
            tgt = (pos[0] - fv[0], pos[1] - fv[1])
            if tgt in blocks:
                charge[tgt] = max(charge.get(tgt, 0), v)
            changed = True
    return cmps, charge, locked


def main():
    cval = 10
    V = 137
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a == "-c":
            cval = int(argv[i + 1])
        if a == "-V":
            V = int(argv[i + 1])

    sch, inv, idx, W, H, L = load(os.path.join(ROOT, "data", "relu.schem"))
    cmps, blocks, torches = parse(sch, inv, idx, W, H, L)
    print(f"relu.schem  {W}x{H}x{L}  比较器 {len(cmps)}  实心块 {len(blocks)}  火把 {len(torches)}")

    # 环：每个位 4 个比较器（x=5..7），按颜色分组
    rings = []
    for k in range(8):
        z0 = 3 * k + 1
        ring = [(6, z0), (5, z0 + 1), (7, z0 + 1), (6, z0 + 2)]
        assert all(p in cmps for p in ring), (k, [p for p in ring if p not in cmps])
        rings.append(ring)
    print(f"识别到 8 个环，每个 4 个比较器；宽度 3 列 × 3 行")
    print(f"烧录：环 k = c·2^k，c={cval}  ->  {[cval * (1 << k) for k in range(8)]}")
    print(f"输入：白块 (0,{L - 1}) 注入 V={V}\n")

    # 注入环的烧录值 + 外部输入
    for k, ring in enumerate(rings):
        for pos in ring:
            cmps[pos].update(state=cval * (1 << k), src=True, init=cval * (1 << k))
    cmps, charge, locked = evaluate({}, cmps, blocks,
                                    extra_sources=[(-1, L - 1, "west", V)])

    unl = [p for p, c in cmps.items() if c["state"] is None]
    print(f"求值：{len(cmps) - len(unl)}/{len(cmps)} 个比较器有值"
          + (f"，未解析 {unl[:6]}" if unl else "（全部解析）"))

    # 输出火把：亮 ⟺ 支撑块(2,3k+1) 的充能者(3,3k+1)输出为 0
    bits = []
    for k in range(8):
        sub = cmps[(3, 3 * k + 1)]["state"]
        bits.append(1 if (sub == 0) else 0)
        print(f"  位{k}: 减法器(3,{3*k+1}) = {sub:5d}   火把(2,{3*k}) "
              f"{'亮' if sub == 0 else '灭'}")
    print(f"\n火把读作二进制（位7..0）= {''.join(str(b) for b in reversed(bits))} = "
          f"{sum(b << k for k, b in enumerate(bits))}")
    print(f"取反读 = {sum((1 - b) << k for k, b in enumerate(bits))}")
    print(f"\n参照：floor((V-1)/c) = floor({V - 1}/{cval}) = {(V - 1) // cval}"
          f" = 0b{(V - 1) // cval:08b}；floor(V/c) = {V // cval} = 0b{V // cval:08b}")


if __name__ == "__main__":
    main()
