# -*- coding: utf-8 -*-
"""重新学习 data/relu.schem：强模 → 数电 的转换器。

第一步只做"普查"：尺寸、调色板、方块数、方块实体、以及逐层的三维分布，
不预设它是除法器还是量化器——先看清里面有什么。

用法：python src/clip_census.py [path]
"""
import gzip
import os
import sys
from collections import Counter

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


class R:
    """最小 NBT 读取器（Sponge v3）。"""

    def __init__(self, b):
        self.b, self.i = b, 0

    def u1(self):
        v = self.b[self.i]
        self.i += 1
        return v

    def i2(self):
        v = int.from_bytes(self.b[self.i:self.i + 2], "big", signed=True)
        self.i += 2
        return v

    def i4(self):
        v = int.from_bytes(self.b[self.i:self.i + 4], "big", signed=True)
        self.i += 4
        return v

    def i8(self):
        v = int.from_bytes(self.b[self.i:self.i + 8], "big", signed=True)
        self.i += 8
        return v

    def name(self):
        n = self.i2()
        s = self.b[self.i:self.i + n].decode("utf-8", "replace")
        self.i += n
        return s

    def payload(self, t):
        if t == 1:
            return self.u1()
        if t == 2:
            return self.i2()
        if t == 3:
            return self.i4()
        if t == 4:
            return self.i8()
        if t == 5:
            return self.i4() / 2.0 if False else ("F", self.i4())
        if t == 6:
            return ("D", self.i8())
        if t == 7:
            n = self.i4()
            v = self.b[self.i:self.i + n]
            self.i += n
            return v
        if t == 8:
            return self.name()
        if t == 9:
            et = self.u1()
            n = self.i4()
            return [self.payload(et) for _ in range(n)]
        if t == 10:
            d = {}
            while True:
                t2 = self.u1()
                if t2 == 0:
                    return d
                k = self.name()
                d[k] = self.payload(t2)
        if t == 11:
            n = self.i4()
            v = [self.i4() for _ in range(n)]
            return v
        if t == 12:
            n = self.i4()
            v = [self.i8() for _ in range(n)]
            return v
        raise ValueError("tag %d" % t)


def varints(b):
    out = []
    for x in b:
        if x & 0x80:
            out.append(x & 0x7F)
        else:
            out.append((out[-1] << 8 | x) if out and False else x)
    return out


def read_varints(data):
    """Sponge v3 的 Data 是 varint（无符号低位组），逐字节：高位=1 表示继续。"""
    out, cur, shift = [], 0, 0
    for x in data:
        cur |= (x & 0x7F) << shift
        if x & 0x80:
            shift += 7
        else:
            out.append(cur)
            cur, shift = 0, 0
    return out


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "data", "relu.schem")
    raw = gzip.decompress(open(path, "rb").read())
    r = R(raw)
    r.u1()
    r.name()
    top = r.payload(10)
    key = [k for k in top if k != ""][0]
    sch = top[key]

    W, H, L = sch["Width"], sch["Height"], sch["Length"]
    pal = sch["Blocks"]["Palette"]
    idx = read_varints(sch["Blocks"]["Data"])
    ents = sch["Blocks"].get("BlockEntities", [])

    print("=" * 78)
    print("文件:", os.path.basename(path))
    print(f"尺寸 {W} x {H} x {L}（x, y, z），方块总数 {W*H*L}，Data 长度 {len(idx)}")
    print("=" * 78)
    print()
    print("一、调色板（共 %d 项）" % len(pal))
    for k, v in sorted(pal.items(), key=lambda kv: kv[1]):
        print(f"  {v:3d}  {k}")
    print()
    cnt = Counter(idx)
    print("二、方块统计")
    tot = 0
    for k, v in sorted(pal.items(), key=lambda kv: kv[1]):
        c = cnt.get(v, 0)
        if c:
            tot += c
            print(f"  {c:5d}  {k}")
    print(f"  {tot:5d}  合计（非空气）")
    print()
    print("三、方块实体 %d 个" % len(ents))
    kinds = Counter(e["Id"] for e in ents)
    for k, v in kinds.items():
        print(f"  {v:5d}  {k}")
    print()
    print("四、逐层（y）分布")
    print(f"  {'y':>2s} {'非空':>5s} {'比较器':>6s} {'容器':>5s} {'火把':>5s} {'环4块':>6s}")
    def is_cmp(k):
        return "comparator" in k
    def is_bar(k):
        return any(s in k for s in ("barrel", "hopper", "chest", "dropper"))
    def is_torch(k):
        return "torch" in k
    for y in range(H):
        cells = [(x, z) for z in range(L) for x in range(W)]
        n = ncmp = nbar = ntor = 0
        for z in range(L):
            for x in range(W):
                k = [kk for kk, vv in pal.items() if vv == idx[(y * L + z) * W + x]][0]
                if k == "minecraft:air":
                    continue
                n += 1
                if is_cmp(k):
                    ncmp += 1
                elif is_bar(k):
                    nbar += 1
                elif is_torch(k):
                    ntor += 1
        print(f"  {y:2d} {n:5d} {ncmp:6d} {nbar:5d} {ntor:5d} {'':>6s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
