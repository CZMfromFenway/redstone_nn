#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""最小 NBT 解析器 + Sponge schematic 读取（纯 Python，无第三方依赖）。"""
import gzip
import json
import os
import struct
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

P = r"E:\redstone_nn\03_phase3_quantized_network\data\rsnn.schem"


class R:
    def __init__(self, b):
        self.b, self.i = b, 0

    def u1(self):
        v = self.b[self.i]
        self.i += 1
        return v

    def i1(self):
        v = struct.unpack_from(">b", self.b, self.i)[0]
        self.i += 1
        return v

    def i2(self):
        v = struct.unpack_from(">h", self.b, self.i)[0]
        self.i += 2
        return v

    def i4(self):
        v = struct.unpack_from(">i", self.b, self.i)[0]
        self.i += 4
        return v

    def i8(self):
        v = struct.unpack_from(">q", self.b, self.i)[0]
        self.i += 8
        return v

    def f4(self):
        v = struct.unpack_from(">f", self.b, self.i)[0]
        self.i += 4
        return v

    def f8(self):
        v = struct.unpack_from(">d", self.b, self.i)[0]
        self.i += 8
        return v

    def s(self):
        n = struct.unpack_from(">H", self.b, self.i)[0]
        self.i += 2
        v = self.b[self.i:self.i + n].decode("utf-8", "replace")
        self.i += n
        return v


def payload(r, t):
    if t == 1:
        return r.i1()
    if t == 2:
        return r.i2()
    if t == 3:
        return r.i4()
    if t == 4:
        return r.i8()
    if t == 5:
        return r.f4()
    if t == 6:
        return r.f8()
    if t == 7:
        n = r.i4()
        v = list(r.b[r.i:r.i + n])
        r.i += n
        return v
    if t == 8:
        return r.s()
    if t == 9:
        it = r.u1()
        n = r.i4()
        return [payload(r, it) for _ in range(n)]
    if t == 10:
        d = {}
        while True:
            tt = r.u1()
            if tt == 0:
                break
            nm = r.s()
            d[nm] = payload(r, tt)
        return d
    if t == 11:
        n = r.i4()
        v = list(struct.unpack_from(">" + "i" * n, r.b, r.i))
        r.i += 4 * n
        return v
    if t == 12:
        n = r.i4()
        v = list(struct.unpack_from(">" + "q" * n, r.b, r.i))
        r.i += 8 * n
        return v
    raise ValueError(f"unknown tag {t}")


raw = gzip.decompress(open(P, "rb").read())
r = R(raw)
t = r.u1()
assert t == 10, t
name = r.s()
root = payload(r, 10)
print("根标签名:", repr(name))
print("顶层键:", list(root.keys()))
print()
for k, v in root.items():
    if isinstance(v, (list, dict)):
        print(f"  {k}: {type(v).__name__} len={len(v)}")
    else:
        print(f"  {k}: {v!r}")

json.dump({"root_name": name, "keys": list(root.keys())},
          open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "results", "schem_keys.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("\n已写出 results/schem_keys.json")
print("NBT_DONE")
