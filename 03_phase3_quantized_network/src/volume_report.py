# -*- coding: utf-8 -*-
"""体积核算：实际 schem vs 文档估计。

对每一层输出：
    包围盒体积、非空方块数、密度、按 y 奇偶分开的占用
    与 docs/39 §二 的估计（V_bits = 3b·2n·2m）对比，并给出膨胀倍数
用法：python src/volume_report.py
"""
import gzip
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from schem_verify_stack import R, varints  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = os.path.join(HERE, "..")


def load(path):
    raw = gzip.decompress(open(path, "rb").read())
    r = R(raw); r.u1(); r.name()
    sch = r.payload(10)["Schematic"]
    W, H, L = sch["Width"], sch["Height"], sch["Length"]
    blk = sch["Blocks"]
    inv = {v: k for k, v in blk["Palette"].items()}
    idx = varints(blk["Data"])
    vol = W * H * L
    nonair = sum(1 for i in idx if inv[i] != "minecraft:air")
    per_y = Counter()
    for y in range(H):
        base = y * L * W
        per_y[y] = sum(1 for i in idx[base:base + L * W] if inv[i] != "minecraft:air")
    return dict(W=W, H=H, L=L, vol=vol, nonair=nonair, per_y=per_y,
                palette=len(blk["Palette"]))


def main():
    # 文档估计（docs/39 §二，2026-10-20 的"b=8/10 单环(定稿)"那一行）
    DOC = {"L1": 27648, "L2": 270336}
    DOC_LUT = 811008
    # 按最终定案 b=8 套同一个公式 V_bits = 3b·2n·2m 重算
    def V(n, m, b=8):
        return 3 * b * 2 * n * 2 * m

    print("=" * 78)
    print("体积核算：矩阵乘法器（b=8 定案）")
    print("=" * 78)
    rows = []
    for tag, fn, n, m in (("层1", "layer1_full.schem", 4, 64),
                          ("层2", "layer2_full.schem", 64, 32)):
        p = os.path.join(ROOT, "results", "schem", fn)
        s = load(p)
        rows.append((tag, n, m, s))

    print(f"{'':6s}{'尺寸 W×H×L':>18s}{'包围盒':>10s}{'非空方块':>10s}{'密度':>8s}"
          f"{'文档估计':>10s}{'膨胀':>7s}{'b=8 公式':>10s}{'膨胀':>7s}")
    tot_vol = tot_blk = tot_doc = tot_f = 0
    for tag, n, m, s in rows:
        doc = DOC["L1"] if tag == "层1" else DOC["L2"]
        f8 = V(n, m)
        tot_vol += s["vol"]; tot_blk += s["nonair"]; tot_doc += doc; tot_f += f8
        print(f"{tag:6s}{'%d×%d×%d' % (s['W'], s['H'], s['L']):>18s}"
              f"{s['vol']:>10,}{s['nonair']:>10,}{s['nonair']/s['vol']*100:>7.1f}%"
              f"{doc:>10,}{s['vol']/doc:>6.2f}×{f8:>10,}{s['vol']/f8:>6.2f}×")
    print(f"{'合计':6s}{'':>18s}{tot_vol:>10,}{tot_blk:>10,}{tot_blk/tot_vol*100:>7.1f}%"
          f"{tot_doc:>10,}{tot_vol/tot_doc:>6.2f}×{tot_f:>10,}{tot_vol/tot_f:>6.2f}×")
    print()
    print(f"参照：同文档的强模 LUT 方案 {DOC_LUT:,} → 实际是它的 "
          f"{tot_vol/DOC_LUT*100:.1f}%（小 {DOC_LUT/tot_vol:.2f} 倍）")
    print()

    # 逐层占用
    for tag, n, m, s in rows:
        odd = sum(v for y, v in s["per_y"].items() if y % 2 == 1)
        even = s["nonair"] - odd
        n_odd = sum(1 for y in s["per_y"] if y % 2 == 1)
        n_even = s["H"] - n_odd
        cap = s["W"] * s["L"]
        print(f"{tag}：{n_odd} 个平面层（奇 y）+ {n_even} 个支撑层（偶 y），每层容量 {cap:,}")
        print(f"     平面层  {odd:>7,} 块   占 {odd/(n_odd*cap)*100:5.1f}%   （比较器/容器/位块/导线）")
        print(f"     支撑层  {even:>7,} 块   占 {even/(n_even*cap)*100:5.1f}%   "
              f"（比较器支撑 + 塔的 A/tap/relay + 塔底）")
        print(f"     空格    {s['vol']-s['nonair']:>7,}        每通道还有 "
              f"{(s['vol']-s['nonair'])/((s['H']-2)//2):,.0f} 格空位可供输出级使用")
        print()

    # 塔占了多少
    print("火把塔占的方块（在包围盒之内，不额外增加体积）：")
    for tag, tw_file, nch, ntw in (("层1", "towers32.json", 64, 32),
                                   ("层2", "towers_L2.json", 32, 512)):
        tw = json.load(open(os.path.join(ROOT, "results", tw_file), encoding="utf-8"))["towers"]
        nstand = sum(sum(t["flip"]) for t in tw)
        ninv = sum(1 for t in tw if t["bottom_inv"])
        nb = 2 * ntw * nch                     # A/B
        pad = ninv * 2 + (ntw - ninv) * 4      # 塔底
        tot = nb + pad + 3 * ntw * nch         # + 三个火把/通道
        print(f"  {tag}: {ntw} 塔 × {nch} 通道 × 5 + 塔底 {pad} = {tot:,} 块"
              f"（直立 {nstand} + 塔底反相 {ninv}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
