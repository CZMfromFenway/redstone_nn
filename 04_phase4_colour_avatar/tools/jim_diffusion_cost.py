"""Jim_Sama 的 8x8 diffusion 页面（index_v6.3u_diffusion.html）—— 结构与成本核算。

只读那个 HTML 的 `<script id="payload">`，把 meta.layout 里的张量形状翻成
参数数与乘加数（MAC），用来和四期现役管线（88,380 乘法）对账。

口径
----
乘法（MAC）：
  conv4d  (CO,CI,KH,KW)  ->  P_out * KH*KW * CI * CO      （含 3x3 与 1x1）
  linear  (OUT,IN)       ->  OUT * IN
  attn    qkv / out 已是 1x1 conv，计入上面两类；注意力本身另算
  GroupNorm / SiLU / 加法 / 上采样 / 拼接：不算乘法（常数次或零次）
参数：张量元素个数直接相加。

不做任何前向，只读形状；所以数与模型文件是否 fp16、WASM 还是 JS 内核无关。

用法： python jim_diffusion_cost.py [--html 路径] [--steps 10] [--cfg 1.2]
"""
import argparse
import json
import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_HTML = os.path.join(ROOT, "shared_by_jim_sama", "index_v6.3u_diffusion.html")


def load_meta(path):
    """取出 payload 里的 meta 对象（7.6 MB 的 base64 权重不动）。"""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if '"meta":' in line:
                i = line.index('"meta":')
                obj, _ = json.JSONDecoder().raw_decode(line[i + len('"meta":'):])
                return obj
    raise SystemExit("没找到 meta")


def spatial(shape):
    """布局里的 shape -> (H, W)。只对卷积有意义。"""
    return None


def mac_of(entry):
    """按名字与形状推一个张量的乘法数。

    分辨率只有两档：8x8（主干 + 上采样后）与 4x4（下采样后的瓶颈）。
    - down.weight 是 stride 2，输出 4x4；
    - d2.* / attn2.* / mid.* 都在 4x4；
    - 其余（inp/d1.*/attn.*/up.*/u2.*/u1.*/outc.*）都在 8x8。
    唯一例外：up2 之后的 u2.0 吃 cat(h, h0)，仍是 8x8。
    """
    name, shp = entry["name"], entry["shape"]
    res = 4 if name.startswith(("d2.", "attn2.", "mid.", "down.")) else 8
    P = res * res
    if len(shp) == 4:
        CO, CI, KH, KW = shp
        if CO == 1 and KH * KW == P:
            return 0, "pos-embed %s (%dx%d 表)" % (name, KH, KW)   # 位置表，纯查表
        return P * KH * KW * CI * CO, "conv%dx%d %d->%d @%d^2" % (KH, KW, CI, CO, res)
    if len(shp) == 2:
        OUT, IN = shp
        return OUT * IN, "linear %d->%d" % (IN, OUT)
    return 0, "vec %d" % shp[0]


def attn_mac(meta, prefix, heads):
    """自注意力的 QK^T 与 AV：每头 2*P*P*hd。"""
    P = 64 if prefix == "attn" else 16
    C = 64 if prefix == "attn" else 128
    hd = C // heads
    return 2 * P * P * C, "attn %s P=%d heads=%d hd=%d" % (prefix, P, heads, hd)


def ladder(out, ref):
    """把 conv-free 的 MLP 去噪器按 (空间, 宽度, 层数, 步数) 列成成本梯子。

    MAC/步 = (D + E + C)*h + (L-1)*h*h + h*D
      D = 输出维数（= 输入维数，去噪器逐像素回归）
      E = 时间条件的输入维数（取 16；若做查表则是 0，见 §注）
      C = 条件旋钮数（PCA4 取 4；无条件取 0）
      L = 隐层数（L=1 即 单隐层  D -> h -> D）
    """
    cases = [
        # 名字, D, h, L, E, C
        ("rgb96 镜像 32 位 · h=64 · 单隐层", 96, 64, 1, 16, 4),
        ("rgb96 镜像 32 位 · h=128 · 单隐层", 96, 128, 1, 16, 4),
        ("rgb96 镜像 32 位 · h=128 · 双隐层", 96, 128, 2, 16, 4),
        ("rgb96 镜像 32 位 · h=256 · 双隐层", 96, 256, 2, 16, 4),
        ("rgb192 全 8x8 · h=128 · 双隐层", 192, 128, 2, 16, 4),
        ("rgb192 全 8x8 · h=256 · 双隐层", 192, 256, 2, 16, 4),
        ("32 位置 x 16 色 logits(512) · h=128 · 双隐层", 512, 128, 2, 16, 4),
    ]
    out.append("")
    out.append("=== conv-free（纯 MLP）去噪器成本梯子 ===")
    out.append("  每步 MAC = (D+E+C)*h + (L-1)*h*h + h*D")
    out.append("")
    head = "%-46s %9s %9s %9s %9s %9s %9s" % ("配置", "MAC/步", "1步", "2步", "4步", "8步", "16步")
    out.append(head)
    for name, D, h, L, E, C in cases:
        per = (D + E + C) * h + (L - 1) * h * h + h * D
        cells = []
        for s in (1, 2, 4, 8, 16):
            t = per * s
            cells.append("%d(%.1fx)" % (t, t / ref))
        out.append("%-46s %9d %9s %9s %9s %9s %9s" % (name, per, cells[0], cells[1], cells[2],
                                                     cells[3], cells[4]))
    out.append("")
    out.append("参照：四期现役管线 = %d 乘法（灰度 2,304 + 颜色头 86,016 + 筛选 60）" % ref)
    out.append("注：时间条件若做成「查表 → 加偏置」则 E 的乘法为 0，上表 E=16 是把它当线性层算的保守口径。")
    out.append("注：采样器本身的更新（x0 = a*x_t + b*eps）是每维 1 次常数乘，192 维 = 192 MAC/步，可忽略。")
    out.append("注：CFG 若保留则每步 2 次前向；这里全部按 1 次前向算（即把引导尺度烘进权重）。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--html", default=DEFAULT_HTML)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--cfg", type=float, default=1.2)
    ap.add_argument("--all", action="store_true", help="打印全部张量")
    ap.add_argument("--ladder", action="store_true", default=True, help="打印 MLP 去噪器成本梯子")
    ap.add_argument("--no-ladder", dest="ladder", action="store_false")
    ap.add_argument("--dump", default="", help="把结果另存为 UTF-8 文本")
    a = ap.parse_args()

    meta = load_meta(a.html)
    lay = meta["layout"]
    rows, total_mac, total_p = [], 0, 0
    for e in lay:
        m, desc = mac_of(e)
        total_mac += m
        total_p += e["n"]
        rows.append((m, e["n"], desc))

    heads, heads2 = meta.get("heads", 4), meta.get("heads2", 8)
    am1, d1 = attn_mac(meta, "attn", heads)
    am2, d2 = attn_mac(meta, "attn2", heads2)
    total_mac += am1 + am2
    total_p += 0

    rows.sort(reverse=True)
    out = []
    out.append("=== Jim_Sama 8x8 diffusion (v6.3u) ===")
    out.append("T=%s  pred=%s  base=%s  params(meta)=%s" %
               (meta.get("T"), meta.get("pred"), meta.get("base"), meta.get("params")))
    out.append("params 实数（按 layout 求和） = %d" % total_p)
    out.append("条件档位数 ncond = %d（其中 null = %s）" % (meta.get("ncond", 9), meta.get("null")))
    out.append("注意力（不进 layout）    : %s / %s" % (d1, d2))
    out.append("")
    out.append("-- 最贵的 12 个张量（MAC / 元素数 / 说明）--")
    for m, n, desc in rows[:12]:
        out.append("  %12d / %8d  %s" % (m, n, desc))
    out.append("  ... 全部 %d 个张量见 --all" % len(lay))
    if a.all:
        out.append("")
        out.append("-- 全表 --")
        for m, n, desc in rows:
            if m:
                out.append("  %12d / %8d  %s" % (m, n, desc))
    out.append("")
    out.append("单次前向 MAC（含注意力） = %d" % total_mac)
    cfg_forwards = 2 if a.cfg and a.cfg > 0 else 1
    per_face = total_mac * cfg_forwards * a.steps
    out.append("一张脸：%d 步 × %d 次前向(CFG w=%s) = %d MAC" %
               (a.steps, cfg_forwards, a.cfg, per_face))
    ref = 88380
    out.append("")
    out.append("-- 三种走法 --")
    for steps, use_cfg in ((a.steps, True), (a.steps, False), (1, False)):
        fwd = 2 if use_cfg else 1
        tot = total_mac * fwd * steps
        out.append("  %2d 步 × %d 次前向 %-6s = %13d MAC  (%8.0fx 四期现役 %d)"
                   % (steps, fwd, "(CFG)" if use_cfg else "(无 CFG)", tot, tot / ref, ref))
    out.append("")
    out.append("对照二期灰度 2,304 / 三期单层 18,432 位权重 -> %d 步 CFG 版分别是 %.0fx / %.0fx"
               % (a.steps, per_face / 2304, per_face / 18432))
    if a.ladder:
        ladder(out, ref)
    txt = "\n".join(out)
    print(txt)
    if a.dump:
        with open(a.dump, "w", encoding="utf-8") as f:
            f.write(txt + "\n")


if __name__ == "__main__":
    main()
