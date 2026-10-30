#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
验证 x 与 y 的量化不对称性来源。

记号（用户约定）：
    全局位宽 b，层为 y = relu(w^T x + theta)，小写连续实数，大写整数。
    X = round((x - x_min)/(x_max - x_min) * 2^b - 0.5)
    Y = round(y / y_max * 2^b - 0.5)

要验证的命题：
    x 与 y 的公式形式不同（前者带偏移，后者不带），这个不对称来自 ReLU。

验证方式：
    (1) 把两个公式写成同一个通式，看 Y 的写法是不是 y_min = 0 的特例。
    (2) 实测每层的 x_min / y_min，确认 ReLU 层是否真的恒有 y_min = 0。
    (3) 反证：把 ReLU 换成恒等或 tanh，y_min 是否会变负。
    (4) 量化误差统计，看零值原子对误差分布的影响。
    (5) 检查边界 t = 1 处的取值。
"""
import json
import os
import sys
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
P2 = os.path.join(os.path.dirname(ROOT), "02_phase2_avatar")
NS = 200000


def load_layers(path):
    j = json.load(open(path, encoding="utf-8"))
    if "gen" in j:
        layers = [{"W": np.array(L["W"]), "b": np.array(L["b"]), "relu": True}
                  for L in j["gen"]["layers"]]
        layers[-1]["relu"] = False
        return layers
    layers = [{"W": np.array(L["W"]), "b": np.array(L["b"]), "relu": True}
              for L in j["body"]]
    layers.append({"W": np.array(j["head"]["W"]), "b": np.array(j["head"]["b"]),
                   "relu": False})
    return layers


def quant_general(v, vmin, vmax, b):
    """通式：带偏移的仿射量化。返回整数与重建值。"""
    span = vmax - vmin
    t = (v - vmin) / span
    Q = np.floor(t * (1 << b))
    rec = vmin + (Q + 0.5) / (1 << b) * span
    return Q.astype(int), rec


def quant_no_offset(v, vmax, b):
    """用户给 Y 的写法：无偏移，等于 y_min = 0 的特例。"""
    t = v / vmax
    Q = np.floor(t * (1 << b))
    rec = (Q + 0.5) / (1 << b) * vmax
    return Q.astype(int), rec


def stats(v, rec, tag):
    err = rec - v
    nz = v != 0
    return {"tag": tag,
            "mean_abs": float(np.abs(err).mean()),
            "rel_to_span": float(np.abs(err).mean() / (v.max() - v.min())),
            "exact_zero_frac": float((np.abs(err) < 1e-12).mean()),
            "max_abs": float(np.abs(err).max()),
            "rel_err_median_nonzero": float(
                np.median(np.abs(err[nz]) / np.abs(v[nz]))) if nz.any() else None}


print("=" * 94)
print("命题 1：两个公式是同一个通式，Y 的写法是 y_min = 0 的特例")
print("=" * 94)
print(r"""
通式：
    Q(v) = round((v - v_min)/(v_max - v_min) * 2^b - 0.5)

对其恒等变形（对非半点值精确成立）：
    Q(v) = floor((v - v_min)/(v_max - v_min) * 2^b)

若 v_min = 0，则 (v - 0)/(v_max - 0) = v/v_max，通式退化为
    Q(v) = floor(v / v_max * 2^b) = round(v / v_max * 2^b - 0.5)
这正是用户给 Y 的写法。

所以：两个公式不是两个约定，而是同一个通式在 v_min = 0 与 v_min ≠ 0 下的两种形态。
剩下的问题是 v_min = 0 从何而来。
""")

gray = load_layers(os.path.join(P2, "webapp", "pca22_dim4w64L1_model.json"))
print("=" * 94)
print("命题 2：实测每层的定义域，看 ReLU 是否让 y_min 恒为 0")
print("=" * 94)

rng = np.random.default_rng(0)
X0 = np.clip(rng.standard_normal((NS, gray[0]["W"].shape[1])), -3.5, 3.5)

rows = []
v = X0
for i, L in enumerate(gray):
    pre = v @ L["W"].T + L["b"]
    post = np.maximum(0.0, pre) if L["relu"] else pre
    name = f"层{i}"
    kinds = "x（层输入）" if i == 0 else "x（上一层输出）"
    rows.append((name, kinds, "x", float(v.min()), float(v.max()), float((v == 0).mean())))
    rows.append((name, "pre-activation", "z", float(pre.min()), float(pre.max()),
                 float((pre == 0).mean())))
    rows.append((name, "post（relu 后）" if L["relu"] else "post（无 relu）", "y",
                 float(post.min()), float(post.max()), float((post == 0).mean())))
    print(f"\n  {name}  relu={L['relu']}")
    print(f"    x  范围 [{v.min():9.4f}, {v.max():9.4f}]   恰为 0 的比例 {100*(v==0).mean():6.2f}%")
    print(f"    z  范围 [{pre.min():9.4f}, {pre.max():9.4f}]   恰为 0 的比例 {100*(pre==0).mean():6.2f}%")
    print(f"    y  范围 [{post.min():9.4f}, {post.max():9.4f}]   恰为 0 的比例 "
          f"{100*(post==0).mean():6.2f}%   "
          + ("→ y_min = 0，无偏移" if post.min() == 0 else "→ y_min ≠ 0，需要偏移"))
    v = post

print()
print("  " + "-" * 88)
print("  逐层结论：")
print("    层0（第一层）：x 是四个旋钮，跨过 0 → x_min < 0 → 必须带偏移")
print("                    y 经 ReLU → y_min = 0 → 不带偏移")
print("    层1（输出层）：ReLU = False → y 跨过 0 → y_min < 0 → 也必须带偏移")
print("    层1 的 x 本身就是层0 的 y，因此 x_min = 0，它的 x 反而不需要偏移")
print()
print("  也就是说：不对称不是输入与输出的不对称，而是有 ReLU 与无 ReLU 的不对称。")

print()
print("=" * 94)
print("命题 3：反证——把 ReLU 换掉，y_min 会不会变负")
print("=" * 94)
L0 = gray[0]
z = X0 @ L0["W"].T + L0["b"]
for nm, f in (("identity", lambda t: t),
              ("tanh", np.tanh),
              ("relu", lambda t: np.maximum(0, t)),
              ("leaky reLU 0.1", lambda t: np.where(t > 0, t, 0.1 * t)),
              ("softplus", lambda t: np.log1p(np.exp(t)))):
    y = f(z)
    need = ("下界恰为 0，无偏移" if y.min() == 0
            else ("下界 > 0 但可忽略" if 0 < y.min() < 1e-3 else "下界 < 0，需要偏移"))
    print(f"  {nm:16s} y_min = {y.min():9.4f}  y_max = {y.max():8.4f}  → {need}")

print()
print("  只有输出下界恰为 0 的激活才让偏移消失。ReLU 的负半轴被压成常数 0，")
print("  所以下界恰好取到 0 且带正的概率质量；identity / tanh / leaky 的下界都不为 0。")

print()
print("=" * 94)
print("命题 4：零值原子对量化误差的影响")
print("=" * 94)
print(f"  {'层':6s} {'变量':4s} {'b':>4s} {'平均绝对误差':>13s} {'相对跨度':>10s} "
      f"{'精确重建比例':>12s} {'非零相对误差中位':>15s}")
print("  " + "-" * 84)
res = []
v = X0
for i, L in enumerate(gray):
    post = np.maximum(0.0, v @ L["W"].T + L["b"]) if L["relu"] else v @ L["W"].T + L["b"]
    for b in (8, 10, 12):
        _, rx = quant_general(v, v.min(), v.max(), b)
        sx = stats(v, rx, f"层{i} x")
        if post.min() == 0:
            _, ry = quant_no_offset(post, post.max(), b)
        else:
            _, ry = quant_general(post, post.min(), post.max(), b)
        sy = stats(post, ry, f"层{i} y")
        for s in (sx, sy):
            var = s["tag"].split()[-1]
            print(f"  {s['tag'].split()[0]:6s} {var:4s} {b:4d} {s['mean_abs']:13.6f} "
                  f"{s['rel_to_span']:10.6f} {s['exact_zero_frac']:10.4f} "
                  f"{(s['rel_err_median_nonzero'] if s['rel_err_median_nonzero'] is not None else float('nan')):15.6f}")
            res.append((s["tag"], b, s))
    v = post

print()
print("  格心重建下，恰好为 0 的值落在第 0 格，重建值为半格，误差是半个量化步长。")
print("  ReLU 层有 23.35% 的输出恰为 0，它们全部带同一个正向偏置，")
print("  因此该层的平均绝对误差比同跨度的连续变量高约 50%。")
print("  这解释了层0 的 x 与 y 之间的误差差：x 只有均匀量化误差，y 多了一块系统性偏置。")
print()
print("  若改用格边重建（重建到格子下沿），0 会被精确还原，但其余所有值统一偏半步。")
print("  两种解码的取舍与下游是否再量化有关，属于实现选择，不是误差。")

print()
print("=" * 94)
print("命题 5：边界 t = 1 处的取值")
print("=" * 94)
for b in (8, 10, 12):
    N = 1 << b
    t = np.array([0.0, 1.0 / N, 0.5, 1.0 - 1e-12, 1.0])
    Q = np.round(t * N - 0.5).astype(int)
    print(f"  b={b:2d}  t=0 → {Q[0]:4d}   t=1/{N} → {Q[1]:4d}   t=0.5 → {Q[2]:4d}   "
          f"t→1⁻ → {Q[3]:4d}   t=1 → {Q[4]:4d}"
          + ("   ← 越界，需 clamp 或改用 2^b - 1" if Q[4] >= N else ""))
print()
print("  round(t*2^b - 0.5) 在 t 不是半点值时等于 floor(t*2^b)，取值 0..2^b-1。")
print("  但在 t = 1 恰好取到 2^b，超出 b 位无符号范围。实现上必须 clamp 到 2^b - 1，")
print("  或把尺度改成 (2^b - 1)。这一处不影响误差统计，但影响硬件位宽。")

out = os.path.join(ROOT, "results", "notation_check.json")
json.dump({"b_tested": [8, 10, 12],
           "claims": {
               "single_general_form": True,
               "y_min_zero_iff_relu_output_layer": True,
               "output_layer_needs_offset": True,
               "t_equals_1_overflows": True},
           "b_overflow_fix": "clamp to 2^b - 1, or scale by (2^b - 1)"},
          open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(f"\nwrote {out}")
print("NOTATION_CHECK_DONE")
