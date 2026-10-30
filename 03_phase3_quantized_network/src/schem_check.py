# -*- coding: utf-8 -*-
"""读回 .schem 验证结构，并生成测试样例文档。"""
import gzip
import io
import json
import os
import struct
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = r"E:\redstone_nn\03_phase3_quantized_network"
SCH = os.path.join(ROOT, "results", "schem")


class R:
    def __init__(self, b):
        self.b, self.i = b, 0

    def u1(self):
        v = self.b[self.i]; self.i += 1; return v

    def i2(self):
        v = struct.unpack(">h", self.b[self.i:self.i + 2])[0]; self.i += 2; return v

    def i4(self):
        v = struct.unpack(">i", self.b[self.i:self.i + 4])[0]; self.i += 4; return v

    def i8(self):
        v = struct.unpack(">q", self.b[self.i:self.i + 8])[0]; self.i += 8; return v

    def f4(self):
        v = struct.unpack(">f", self.b[self.i:self.i + 4])[0]; self.i += 4; return v

    def f8(self):
        v = struct.unpack(">d", self.b[self.i:self.i + 8])[0]; self.i += 8; return v

    def s(self):
        n = struct.unpack(">H", self.b[self.i:self.i + 2])[0]; self.i += 2
        v = self.b[self.i:self.i + n].decode("utf-8"); self.i += n; return v

    def payload(self, t):
        if t == 1: return self.u1()
        if t == 2: return self.i2()
        if t == 3: return self.i4()
        if t == 4: return self.i8()
        if t == 5: return self.f4()
        if t == 6: return self.f8()
        if t == 7:
            n = self.i4(); v = self.b[self.i:self.i + n]; self.i += n; return v
        if t == 8: return self.s()
        if t == 9:
            et = self.u1(); n = self.i4()
            return [self.payload(et) for _ in range(n)]
        if t == 10:
            d = {}
            while True:
                tt = self.u1()
                if tt == 0: return d
                nm = self.s()
                d[nm] = self.payload(tt)
        if t == 11:
            n = self.i4()
            return [self.i4() for _ in range(n)]
        if t == 12:
            n = self.i4()
            return [self.i8() for _ in range(n)]
        raise ValueError(f"tag {t}")


def check(path):
    raw = gzip.decompress(open(path, "rb").read())
    r = R(raw)
    assert r.u1() == 10, "根不是 compound"
    _ = r.s()
    root = r.payload(10)
    W = root["Width"]; H = root["Height"]; L = root["Length"]
    pal = root["Palette"]
    bd = root["BlockData"]
    # 展开 varint
    idx = []
    val = 0; shift = 0; pos = 0
    while pos < len(bd):
        b = bd[pos]; pos += 1
        val |= (b & 0x7F) << shift
        shift += 7
        if not (b & 0x80):
            idx.append(val); val = 0; shift = 0
    assert len(idx) == W * H * L, f"BlockData {len(idx)} != {W*H*L}"
    inv = {v: k for k, v in pal.items()}
    counts = {}
    for i in idx:
        nm = inv[i]
        counts[nm] = counts.get(nm, 0) + 1
    return dict(W=W, H=H, L=L, npal=len(pal), ncell=len(idx),
                counts=counts, version=root.get("Version"),
                dataversion=root.get("DataVersion"))


def main():
    d = json.load(open(os.path.join(ROOT, "results", "board_dump.json"), encoding="utf-8"))
    print("=== .schem 文件校验 ===")
    for case in d["cases"]:
        p = os.path.join(SCH, f"layer1_ch0_{case['name']}.schem")
        info = check(p)
        nc = info["counts"]
        cmp_n = sum(v for k, v in nc.items() if "comparator" in k)
        bar_n = sum(v for k, v in nc.items() if "barrel" in k)
        tor_n = sum(v for k, v in nc.items() if "redstone_torch" in k)
        print(f"  {case['name']:10s} {info['W']}×{info['H']}×{info['L']}  "
              f"Version={info['version']}  格数={info['ncell']}  "
              f"比较器={cmp_n}  容器={bar_n}  火把={tor_n}  "
              f"head={case['head']} N={case['N']}  {'✓' if case['head']==case['N'] else '✗'}")

    # ---- 测试样例文档
    md = io.StringIO()
    md.write("# 层1 通道0 测试样例（b=8，灰/白朝低位权方向，无输入反转，无位权镜像）\n\n")
    md.write(f"平面尺寸 14×31，原点列 3、原点行 21；"
             f"$K={d['cases'][0]['K']}$，$W_0={d['cases'][0]['W0']}$。\n\n")
    md.write("## 火把规则\n\n")
    md.write("每一格 $(i,k)$ 的火把状态按\n\n")
    md.write("$$\\text{火把亮} \\iff (x_i == 1) \\ne (W^{(k)}_i > 0)$$\n\n")
    md.write("即：$W>0$ 的格子在 $x=0$ 时亮、$x=1$ 时灭；$W<0$ 的格子在 $x=1$ 时亮、$x=0$ 时灭。\n\n")

    for case in d["cases"]:
        md.write(f"## {case['name']}　$x=({', '.join(map(str, case['x']))})$\n\n")
        md.write(f"- 头部输出 $N = {case['N']}$，仿真器 {case['head']}，"
                 f"{'一致 ✓' if case['head']==case['N'] else '不一致 ✗'}\n")
        md.write(f"- 头部位置：绝对 `(x={case['headPos'][0]}, z={case['headPos'][1]})`\n\n")
        md.write("| 位 k | 颜色 | 各输入的 W | 火把（亮=15） |\n|---:|---|---|---|\n")
        S = case["S"]; WS = case["WS"]
        COLORS = ["红", "橙", "黄", "绿", "青", "蓝", "紫", "品红"]
        NB = d["meta"]["b"]
        for k in range(NB - 1, -1, -1):
            ws = " / ".join(str(WS[i][k]) for i in range(4))
            ts = " / ".join(
                "亮" if (((case["x"][i] == 1) != (WS[i][k] > 0))) else "灭"
                for i in range(4))
            md.write(f"| {k} | {COLORS[k % 8]} | {ws} | {ts} |\n")
        md.write("\n")
        # 比较器全局值
        md.write("<details><summary>全部比较器信号（按 z 分行）</summary>\n\n")
        byz = {}
        for c in case["cells"]:
            if c["t"] == "cmp":
                byz.setdefault(c["z"], []).append((c["x"], c["st"], c["mode"]))
        for z in sorted(byz):
            row = "  ".join(f"x{x}:{st}" + ("(减)" if m == "subtract" else "")
                            for x, st, m in sorted(byz[z]))
            md.write(f"- $z={z}$  {row}\n")
        md.write("\n</details>\n\n")

    p = os.path.join(ROOT, "results", "schem_test_cases.md")
    io.open(p, "w", encoding="utf-8").write(md.getvalue())
    print(f"\n测试样例文档 -> {p}")


if __name__ == "__main__":
    main()
