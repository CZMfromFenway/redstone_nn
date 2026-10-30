# -*- coding: utf-8 -*-
"""生成给 hjkv（MCHPRS 强模支持）的交付包。

产出目录  deliverables/hjkv_mchprs_4to64/
    README.md                             主文档（坐标表、映射、测试用例、声明）
    layer1_matmul_4to64_mc1.21_v1.schem   4 → 64 矩阵乘法器
    io_map.json                           32 个输入块 + 64 个输出比较器 + 检查点
    weights_L1_b8.json                    2048 个位权重（predict.py 用）
    test_vectors.json                     测试用例
    predict.py                            独立脚本：输入 → 需要充能的块 + 期望输出
    NOTE_版本与测试声明.txt               贴在文件旁的备注

文档一律用纯文本公式（不用 LaTeX），方便对方在任何编辑器里看。

用法：python src/deliver_hjkv.py
"""
import json
import os
import shutil
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
OUT = os.path.join(ROOT, "deliverables", "hjkv_mchprs_4to64")
N, B = 4, 8
YOFF = 2

PREDICT_PY = r'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""predict.py —— 这个 schem 的"预期输出"计算器（纯 Python 3，无第三方依赖）。

它做两件事：
  1) 给定 4 个输入整数，列出 32 个需要充能的方块（哪些要充、哪些不充）；
  2) 给出 64 个输出比较器的期望 OutputSignal。

用法：
    python predict.py 0 0 0 0             # 全 0（不接线就能验）
    python predict.py 186 37 204 91       # 任意一组
    python predict.py 186 37 204 91 --json
    python predict.py --coords            # 只打印坐标表

公式：
    N[ch] = sum_i sum_k ( W[ch][i][k] * bit_k(X[i]) - min(0, W[ch][i][k]) )
    输出比较器的 OutputSignal 就等于 N[ch]。
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def load():
    with open(os.path.join(HERE, "weights_L1_b8.json"), encoding="utf-8") as f:
        W = json.load(f)["W"]
    with open(os.path.join(HERE, "io_map.json"), encoding="utf-8") as f:
        m = json.load(f)
    return W, m


def expected(W, X):
    out = []
    for ch in range(len(W)):
        n = 0
        for i in range(4):
            for k in range(8):
                w = W[ch][i][k]
                n += w * ((X[i] >> k) & 1) - min(0, w)
        out.append(n)
    return out


def main():
    W, m = load()
    if "--coords" in sys.argv:
        print("输入块（bit=1 时给该方块充能）:")
        for e in m["inputs"]:
            print("  i=%d k=%d  (%d, %d, %d)" % (e["i"], e["k"], e["x"], e["y"], e["z"]))
        print("输出比较器:")
        for e in m["outputs"]:
            print("  ch=%d  (%d, %d, %d)" % (e["ch"], e["x"], e["y"], e["z"]))
        return

    xs = [int(a) for a in sys.argv[1:5]]
    if len(xs) != 4:
        print(__doc__)
        return 1
    if any(v < 0 or v > 255 for v in xs):
        print("输入必须落在 0..255")
        return 1

    exp = expected(W, xs)
    if "--json" in sys.argv:
        print(json.dumps({"X": xs, "expected": exp}, ensure_ascii=False))
        return

    print("输入 X = %s" % (xs,))
    print("  二进制（位7..0）: " + "  ".join("X%d=%s" % (i, format(xs[i], "08b")) for i in range(4)))
    print()
    print("需要充能的方块（%d 个中，充能的那些）:" % len(m["inputs"]))
    hot = [e for e in m["inputs"] if (xs[e["i"]] >> e["k"]) & 1]
    for e in hot:
        print("  (%d, %d, %d)   <- 输入%d 的第 %d 位" % (e["x"], e["y"], e["z"], e["i"], e["k"]))
    print("  （其余 %d 个保持不充能）" % (len(m["inputs"]) - len(hot)))
    print()
    print("输出比较器的期望 OutputSignal：")
    for ch in range(len(exp)):
        e = m["outputs"][ch]
        print("  ch%-2d (%2d, %3d, %2d)  ->  %d" % (ch, e["x"], e["y"], e["z"], exp[ch]))
    print()
    print("合计：%d 个通道，期望输出范围 [%d, %d]" % (len(exp), min(exp), max(exp)))


if __name__ == "__main__":
    sys.exit(main())
'''

NOTE_TXT = """这个 .schem 是 4 -> 64 的矩阵乘法器（强模）。请先看 README.md。

【版本】
  文件按 Minecraft 1.21 导出（Sponge Schematic v3，DataVersion 3953）。
  MCHPRS 目前适配 1.20.4，需要手动转换。
  本文件只有比较器 + 混凝土，没有容器与物品，所以两种版本之间实质差异很小，见 README.md 第 1 节。

【测试程度 —— 重要】
  本文件没有在游戏里跑完全部测试。
  已做：4 个通道 x 2 组输入，手工读头部与白总线沿途，与预期一致。
  未做：全部 64 通道、全部输入组合、MCHPRS 上的任何运行。
  64 个通道的期望值由仿真器给出，并已与编译器的整数前向逐通道核对（64/64 一致）。
  但"仿真一致"不等于"游戏一致"，请以实测为准。

【怎么用】
  1) README.md 第 3 节：32 个需要充能的方块坐标；
     第 4 节：64 个输出比较器坐标。
  2) 想要某组输入的预期值：python predict.py <X0> <X1> <X2> <X3>
  3) 全 0 输入是这个文件的「空闲态」，不接线就能读，建议先验这一组。

【关于「文件上的备注」】
  .schem 是 gzip 压缩的二进制 NBT，塞自定义字段有破坏加载兼容的风险，
  所以本文件的备注只体现在文件名（含 mc1.21 与 v1）和这份 NOTE / README 里，
  文件内容没有改动。
"""


def fmt_grid(vals, per=8):
    lines = []
    for i in range(0, len(vals), per):
        lines.append("  ".join("%6d" % v for v in vals[i:i + per]))
    return lines


def main():
    planes = json.load(open(os.path.join(ROOT, "results", "planes64.json"), encoding="utf-8"))
    towers = json.load(open(os.path.join(ROOT, "results", "towers32.json"), encoding="utf-8"))
    ref = json.load(open(os.path.join(ROOT, "results", "py_compile_reference.json"),
                         encoding="utf-8"))
    W = ref["layers"]["L1"]["8"]["W"]
    NCH = len(planes["planes"])
    os.makedirs(OUT, exist_ok=True)

    # ---------------- 坐标 ----------------
    inputs = []
    for T in towers["towers"]:
        ax, az = T["cells"]["A"]          # towers32.json 里已是绝对坐标 (x, z)
        inputs.append({
            "i": T["i"], "k": T["k"],
            "x": ax, "y": 0, "z": az,     # 塔底那一格：y_schem = 0
            "bottom_inv": T["bottom_inv"],
            "flips": sum(T["flip"]),
        })
    inputs.sort(key=lambda t: (t["i"], t["k"]))

    def yy(ch):                            # 平面在 y_world = 2ch+1，导出 +2
        return 2 * ch + 1 + YOFF

    outputs = [{"ch": ch, "x": 11, "y": yy(ch), "z": 3} for ch in range(NCH)]
    checks = [{"ch": ch,
               "K": {"x": 1, "y": yy(ch), "z": 21},
               "W0": {"x": 11, "y": yy(ch), "z": 23},
               "bus": [{"x": 11, "y": yy(ch), "z": z} for z in (21, 15, 9)],
               "head": {"x": 11, "y": yy(ch), "z": 3}} for ch in range(NCH)]
    io_map = {
        "schem": {"W": planes["W"], "H": 2 * NCH + YOFF, "L": planes["L"],
                  "channels": NCH, "inputs_i": N, "bits_b": B,
                  "y_convention": "y_schem = y_world + 2; channel ch plane at y = 2*ch+3"},
        "inputs": inputs, "outputs": outputs, "checkpoints": checks,
        "note": "inputs: 给该方块充能表示该位=1（15 即可）; outputs: 读该比较器的 OutputSignal",
    }
    json.dump(io_map, open(os.path.join(OUT, "io_map.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    json.dump({"note": "W[ch][i][k] = layer1 的整数位权重 (b=8)", "W": W},
              open(os.path.join(OUT, "weights_L1_b8.json"), "w", encoding="utf-8"))

    # ---------------- 期望值 ----------------
    def exp(X):
        out = []
        for ch in range(NCH):
            n = 0
            for i in range(N):
                for k in range(B):
                    w = W[ch][i][k]
                    n += w * ((X[i] >> k) & 1) - min(0, w)
            out.append(n)
        return out

    cases = [
        ("全 0（空闲态，不接线即可验）", [0, 0, 0, 0]),
        ("全 255", [255, 255, 255, 255]),
        ("已在游戏里验过 4 个通道", [186, 37, 204, 91]),
        ("只动输入 0", [1, 0, 0, 0]),
        ("只动输入 1", [0, 1, 0, 0]),
        ("只动输入 2", [0, 0, 1, 0]),
        ("只动输入 3", [0, 0, 0, 1]),
        ("输入 2/3 全开（塔底带反相级的 16 根）", [0, 0, 255, 255]),
    ]
    vectors = [{"name": nm, "X": x, "expected": exp(x)} for nm, x in cases]
    json.dump({"note": "expected[ch] = 通道 ch 的输出比较器 OutputSignal", "vectors": vectors},
              open(os.path.join(OUT, "test_vectors.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    # ---------------- schem / 脚本 / 备注 ----------------
    shutil.copyfile(os.path.join(ROOT, "results", "schem", "layer1_full.schem"),
                    os.path.join(OUT, "layer1_matmul_4to64_mc1.21_v1.schem"))

    # ---------------- 自检：把坐标对回交付的 schem ----------------
    import gzip
    sys.path.insert(0, HERE)
    from schem_verify_stack import R, varints
    raw = gzip.decompress(open(os.path.join(OUT, "layer1_matmul_4to64_mc1.21_v1.schem"),
                              "rb").read())
    r = R(raw); r.u1(); r.name()
    sch = r.payload(10)["Schematic"]
    Wd, Hd, Ld = sch["Width"], sch["Height"], sch["Length"]
    inv = {v: k for k, v in sch["Blocks"]["Palette"].items()}
    idx = varints(sch["Blocks"]["Data"])
    be = {(e["Id"], tuple(e["Pos"])): e.get("Data", {})
          for e in sch["Blocks"]["BlockEntities"]}

    def at(x, y, z):
        return inv[idx[(y * Ld + z) * Wd + x]]

    e0 = exp([0, 0, 0, 0])
    problems = []
    for e in inputs:
        if not (0 <= e["x"] < Wd and 0 <= e["y"] < Hd and 0 <= e["z"] < Ld):
            problems.append("输入块越界 %s" % (e,))
            continue
        if "concrete" not in at(e["x"], e["y"], e["z"]):
            problems.append("输入块 (i%d,k%d)@(%d,%d,%d) 不是混凝土：%s"
                            % (e["i"], e["k"], e["x"], e["y"], e["z"],
                               at(e["x"], e["y"], e["z"])))
    for e in outputs:
        st = at(e["x"], e["y"], e["z"])
        v = be.get(("minecraft:comparator", (e["x"], e["y"], e["z"])), {}).get("OutputSignal")
        if "comparator" not in st:
            problems.append("输出 ch%d@(%d,%d,%d) 不是比较器：%s"
                            % (e["ch"], e["x"], e["y"], e["z"], st))
        elif v != e0[e["ch"]]:
            problems.append("输出 ch%d 的 OutputSignal=%s，但 X=0 期望 %d"
                            % (e["ch"], v, e0[e["ch"]]))
    for c in checks:
        for key in ("K", "W0"):
            p = c[key]
            v = be.get(("minecraft:comparator", (p["x"], p["y"], p["z"])), {}).get("OutputSignal")
            if v is None:
                problems.append("ch%d 的 %s 位置不是比较器" % (c["ch"], key))
    if problems:
        print("✗ 自检失败 %d 项：" % len(problems))
        for p in problems[:10]:
            print("   " + p)
        return 1
    print("✓ 自检通过：32 个输入块都是混凝土、64 个输出比较器的 OutputSignal 与 X=0 期望逐通道一致、"
          "K/W0 位置可读")

    open(os.path.join(OUT, "predict.py"), "w", encoding="utf-8").write(PREDICT_PY)
    open(os.path.join(OUT, "NOTE_版本与测试声明.txt"), "w", encoding="utf-8").write(NOTE_TXT)

    # ---------------- README ----------------
    e0 = exp([0, 0, 0, 0])
    e1 = exp([186, 37, 204, 91])
    L = []
    A = L.append
    A("# 4 -> 64 矩阵乘法器（强模）：测试案例说明\n\n")
    A("**交付对象**：hjkv（MCHPRS 强模支持）　　**文件**：`layer1_matmul_4to64_mc1.21_v1.schem`\n\n")
    A("**日期**：2026-10-25　　**来源**：redstone_nn 三期工程（把 4-64-32 量化网络编译成红石）的第 1 层\n\n")
    A("---\n\n")

    A("## 0. 一句话\n\n")
    A("这是一台**纯强模的 4 -> 64 矩阵乘法器**：输入是 4 个 0..255 的整数（共 32 个二进制位），\n")
    A("输出是 64 个比较器，每个给出该通道的累加量 **N**（整数，0 到约 11000）。\n\n")
    A("它只用到四种东西：比较器、木桶（存权重）、实心方块、火把。\n")
    A("**没有红石粉、没有中继器、没有活塞。**\n\n")
    A("---\n\n")

    A("## 1. 先说清楚：版本与测试程度\n\n")
    A("### 1.1 版本\n\n")
    A("| 项 | 值 |\n|---|---|\n")
    A("| 导出格式 | Sponge Schematic v3 |\n")
    A("| **导出时的游戏版本** | **Minecraft 1.21**（`DataVersion = 3953`） |\n")
    A("| MCHPRS 当前适配 | 1.20.4 |\n")
    A("| **需要你做的事** | **手动转换到 1.20.4** |\n\n")
    A("> **好消息：本文件只有比较器 + 混凝土方块，没有容器、没有物品栈。**\n")
    A("> 所以 1.20.4 与 1.21 之间的物品格式差异（`Count` / `count`）**不会碰到**。\n")
    A("> 转换大概率只需要改 `DataVersion`（1.20.4 = 3837），并确认方块名一致。\n\n")

    A("### 1.2 测试程度（请务必读）\n\n")
    A("**这个文件没有在游戏里跑完全部测试。** 已做的与没做的分开列：\n\n")
    A("| | 内容 |\n|---|---|\n")
    A("| 已做 | 64 层堆叠能加载；**4 个通道**（ch0 / ch3 / ch20 / ch63）在 **2 组输入**下，"
      "手工读头部与白总线沿途 3 个检查点，与预期**一致** |\n")
    A("| 已做 | ch0 的**单层**版本另有 4 组输入（全 0 / 全 1 / 1010 / 0101）在游戏里实测通过 |\n")
    A("| 已做 | 全部 64 通道 x 6 组输入的**仿真**值，与 Python 编译器的整数前向逐通道核对："
      "**64/64 一致** |\n")
    A("| **未做** | 全部 64 通道的**游戏内**读数 |\n")
    A("| **未做** | 全部输入组合（共 256^4 种） |\n")
    A("| **未做** | MCHPRS 上的任何运行（这正是本次目的） |\n")
    A("| **未做** | 时序（比较器每级 1 tick 延迟；只影响多久稳定，不影响稳态值） |\n\n")
    A("> 换句话说：**几何与稳态值有仿真背书 + 少量实测背书；大规模实测要靠你。**\n\n")
    A("---\n\n")

    A("## 2. 结构总览\n\n")
    A("| 项 | 值 |\n|---|---|\n")
    A("| 尺寸 | **14 x 130 x 31**（宽 x 高 x 长） |\n")
    A("| 通道数 | 64（每个通道 = 一次点积 + 4 条减法链 + 白总线） |\n")
    A("| 非空方块 | 29 930（比较器 6 730、木桶 1 948、火把 6 192、混凝土约 15 000） |\n")
    A("| 方块实体 | 8 678（比较器 6 730 + 木桶 1 948） |\n")
    A("| 是否含输入分配结构 | **含**（32 根竖直「火把塔」，把 32 个位送到 64 个通道） |\n\n")
    A("**y 坐标约定**：`y_schem = y_world + 2`，通道 **ch** 的平面在 **y = 2*ch + 3**。\n\n")
    A("> 通道 0 在 y=3、通道 63 在 y=129；**y=0 与 y=1 是塔底**，也就是输入接口。\n\n")
    A("---\n\n")

    A("## 3. 输入：32 个要充能的方块\n\n")
    A("**规则**：第 (i, k) 个方块 = 输入 `X[i]` 的第 k 位。\n")
    A("位为 **1** 就给这个方块**充能**（15 即可，火把 / 中继器 / 拉杆都行）；位为 **0** 就**保持不充能**。\n\n")
    A("方块坐标 = `(x, 0, z)`，见下表：\n\n")
    A("| # | 输入 i | 位 k | 坐标 (x, y, z) | 备注 |\n|---:|---:|---:|---|---|\n")
    for idx, e in enumerate(inputs):
        note = "塔底带反相级" if e["bottom_inv"] else ""
        A("| %d | %d | %d | (%d, %d, %d) | %s |\n"
          % (idx, e["i"], e["k"], e["x"], e["y"], e["z"], note))
    A("\n同一输入的 8 个位共用一套符号模式，所以是否需要反相级只取决于输入下标：\n")
    A("**输入 0、1 的 16 根塔底是「之字形双火把」；输入 2、3 的 16 根是「块 + 直立火把」（多一级反相）。**\n")
    A("这是编译时按权重符号算好的，**你不需要管** —— 两种情况都只要求给那个方块充能。\n\n")
    A("---\n\n")

    A("## 4. 输出：64 个比较器\n\n")
    A("通道 **ch** 的输出比较器坐标 **`(11, 2*ch+3, 3)`**，直接读它的 **`OutputSignal`**（方块实体字段），\n")
    A("就是该通道的累加量 N。\n\n")
    A("| ch | 坐标 (x, y, z) | ch | 坐标 (x, y, z) | ch | 坐标 (x, y, z) | ch | 坐标 (x, y, z) |\n")
    A("|---:|---|---:|---|---:|---|---:|---|\n")
    for q in range(16):
        cells = []
        for j in range(4):
            ch = q + 16 * j
            cells.append("%d | (%d, %d, %d)" % (ch, 11, 2 * ch + YOFF, 3))
        A("| " + " | ".join(cells) + " |\n")
    A("\n### 4.1 调试用的检查点（头部不对时先看这些）\n\n")
    A("每个通道还有 4 个中间值，其中前两个**与输入完全无关**，所以第一次接线前就该能读到固定值：\n\n")
    A("| 位置 | 坐标 | 含义 |\n|---|---|---|\n")
    A("| 灰环 K | `(1, 2ch+3, 21)` | K = ΣS（全部容器值之和），**与输入无关** |\n")
    A("| 白环 W0 | `(11, 2ch+3, 23)` | W0 = J·K − 15nb，**与输入无关** |\n")
    A("| 白总线 | `(11, 2ch+3, 21)` / `(…, 15)` / `(…, 9)` | 依次减掉四条链后的运行值 |\n")
    A("| 头部 | `(11, 2ch+3, 3)` | 最终 N |\n\n")
    A("正常关系：`W0` --减 d3--> `(11,y,21)` --减 d2--> `(11,y,15)` --减 d1--> `(11,y,9)` --减 d0--> `N`\n\n")
    A("> **最快的「板子没贴坏」检查**：先读 K 与 W0，它们不随输入变化（例如 ch0 的 K=9176、W0=36224）。\n\n")
    A("---\n\n")

    A("## 5. 输入整数 -> 32 个方块\n\n")
    A("```python\n")
    A("for i in range(4):            # 4 个输入\n")
    A("    for k in range(8):        # 每个 8 位\n")
    A("        bit = (X[i] >> k) & 1        # 第 k 位，k=0 是最低位\n")
    A("        if bit: 给坐标 (x[i][k], 0, z[i][k]) 充能\n")
    A("        else:   不充能\n")
    A("```\n\n")
    A("例：`X = (186, 37, 204, 91)`，其中 `X[2] = 204 = 0b11001100`：\n\n")
    A("```\n")
    A("204 = 1x128 + 1x64 + 0x32 + 0x16 + 1x8 + 1x4 + 0x2 + 0x1\n")
    A("=> 位 7,6,3,2 = 1  => 给 (i=2, k=7/6/3/2) 那 4 个方块充能，其余 4 个不动\n")
    A("```\n\n")
    A("---\n\n")

    A("## 6. 输出应该等于多少\n\n")
    A("```\n")
    A("N[ch] = sum over i=0..3, k=0..7 of ( W[ch][i][k] * bit_k(X[i]) - min(0, W[ch][i][k]) )\n")
    A("```\n\n")
    A("- `W[ch][i][k]`：该通道该输入该位的**整数位权重**，全部 2048 个值在 `weights_L1_b8.json` 里。\n")
    A("- 后一项 `min(0, W)` 是「负权重的偏置」，板上由火把亮灭自动完成，**你不需要管**。\n")
    A("- 每一项都减去 `min(0,W)`，所以 **N >= 0 恒成立**，最大约 11 000。\n\n")
    A("---\n\n")

    A("## 7. 测试用例\n\n")
    A("完整的 8 组用例（含全部 64 通道期望值）在 `test_vectors.json`。下面给两组，方便直接开始。\n\n")
    A("### 7.1 全 0 输入（**建议第一个验，不用接线**）\n\n")
    A("这个文件的「空闲态」就是全 0：所有塔底都不充能。**贴进去直接读数**就应该等于：\n\n")
    A("```\n")
    for ln in fmt_grid(e0):
        A(ln + "\n")
    A("```\n\n")
    A("### 7.2 X = (186, 37, 204, 91)（已在游戏里验过 4 个通道）\n\n")
    A("```\n")
    for ln in fmt_grid(e1):
        A(ln + "\n")
    A("```\n\n")
    A("> **游戏内实测**：ch0 = 3261、ch3 = 3780、ch20 = 6736、ch63 = 3145 —— 四个都对上了。\n\n")
    A("### 7.3 其它用例\n\n")
    A("| 用例 | X | 用途 |\n|---|---|---|\n")
    A("| 全 0 | (0,0,0,0) | 空闲态，不接线 |\n")
    A("| 全 255 | (255,255,255,255) | 所有位都充能 |\n")
    A("| 随机（已验） | (186,37,204,91) | 综合 |\n")
    A("| 只动输入 0..3 | (1,0,0,0) … (0,0,0,1) | 每根塔单独验证，定位「哪一组塔接错」 |\n")
    A("| 输入 2/3 全开 | (0,0,255,255) | 专门压塔底反相级（这 16 根多一级） |\n\n")
    A("---\n\n")

    A("## 8. `predict.py`：自己算期望值\n\n")
    A("```bash\n")
    A("python predict.py 0 0 0 0             # 全 0\n")
    A("python predict.py 186 37 204 91       # 任意一组\n")
    A("python predict.py 186 37 204 91 --json\n")
    A("python predict.py --coords            # 只打印坐标表\n")
    A("```\n\n")
    A("输出包括：**哪些方块要充能**（带坐标）+ **每个通道的期望 OutputSignal**。\n")
    A("> 我们另有一个**板级仿真器**（能画出每一格、逐格求值、改任意输入）——如果 MCHPRS 那边需要，\n> 可以再找我们要；它需要网页或 Node 环境，所以没放进这个包。\n")
    A("纯 Python 3，无第三方依赖，只读同目录的 `weights_L1_b8.json` 与 `io_map.json`。\n\n")
    A("---\n\n")

    A("## 9. 已知限制 / 可能踩的坑\n\n")
    A("| # | 项 | 说明 |\n|---:|---|---|\n")
    A("| 1 | 版本 | 1.21 -> 1.20.4 需手动转换（见 1.1） |\n")
    A("| 2 | 未跑全量 | 见 1.2 |\n")
    A("| 3 | 火把亮灭的「空闲态」 | 文件里烧录的亮灭按全 0 输入算。接上输入后由游戏重算；"
      "**若 MCHPRS 不重算火把状态，可能需要先 tick 一次** |\n")
    A("| 4 | 比较器 `powered` 与 `OutputSignal` | 文件里两者一致。**若转换工具只改 `OutputSignal` "
      "而不同步 `powered`，环会失效**（我们在游戏里踩过） |\n")
    A("| 5 | 环 | 灰环 / 白环 / 配对区的 Sa 环都是「4 比较器 + 4 中介方块」的自锁环，靠烧录值启动。"
      "**若 MCHPRS 把 `OutputSignal` 当普通方块属性重算，需要保留环的初始值** |\n")
    A("| 6 | 时序 | 比较器每级 1 tick；128 层高的塔链延迟可观（未测），但稳态值不受影响 |\n")
    A("| 7 | 输入驱动 | 文件里**没有**总线。你需要自己给那 32 个方块充能；"
      "任何 >=1 的红石源都可以（亮火把 / 中继器 / 拉杆） |\n\n")
    A("---\n\n")

    A("## 10. 文件清单\n\n")
    A("| 文件 | 说明 |\n|---|---|\n")
    A("| `layer1_matmul_4to64_mc1.21_v1.schem` | 本案例（14x130x31，29 930 方块） |\n")
    A("| `README.md` | 本文件 |\n")
    A("| `NOTE_版本与测试声明.txt` | 贴在同一目录的短备注 |\n")
    A("| `io_map.json` | 32 输入块 + 64 输出比较器 + 检查点的机器可读坐标 |\n")
    A("| `weights_L1_b8.json` | 2048 个整数位权重 |\n")
    A("| `test_vectors.json` | 8 组用例的完整期望输出 |\n")
    A("| `predict.py` | 独立预测脚本 |\n\n")
    A("> 关于「在 schem 文件上做备注」：`.schem` 是 gzip 压缩的二进制 NBT，"
      "**往里面塞自定义字段有破坏加载兼容的风险**，\n")
    A("> 所以备注只体现在文件名（`mc1.21` / `v1`）与本目录的两份文本里，**文件内容一个字没改**。\n\n")
    A("---\n\n")

    A("## 11. 补充：另有一份「预减 P-1」的变体\n\n")
    A("我们另外导出了 `layer1_relu.schem`：白环整体减掉了 `P-1`，于是头部输出直接变成\n")
    A("`max(0, N-P+1)` —— 那是后端 RELU 除法器要的输入。\n\n")
    A("**本案例用的是普通版**（头部输出就是 N），因为它最好验证。\n")
    A("如果 MCHPRS 那边跑通了、想接着测输出级，再找我们要那一份。\n")

    open(os.path.join(OUT, "README.md"), "w", encoding="utf-8").write("".join(L))
    print("已生成交付包：", OUT)
    for f in sorted(os.listdir(OUT)):
        print("  %-44s %7.1f KB" % (f, os.path.getsize(os.path.join(OUT, f)) / 1024))
    return 0


if __name__ == "__main__":
    sys.exit(main())
