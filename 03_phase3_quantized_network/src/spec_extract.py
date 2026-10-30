#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
提取硬件无关的数值规格，作为三期量化工作的起点。

WHY A SPEC FIRST
量化的所有决定都依赖三个量：权重分布、激活范围、输出的取值范围。这三者在二期是以浮点
JSON 保存的，但没有任何地方把它们的实际分布记下来。先把它们测出来，后面的位宽选择与成本
估算才有依据，而不是凭印象。

输出（results/spec_*.json）：
    layers        每层的形状、乘法数、权重统计、以及若干位宽下的量化误差
    activations   在假设的输入分布下，每层输出的实际取值范围与分位数
    output        输出的取值范围
    totals        乘法总数与权重总数
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
RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)

BITS = (4, 6, 8, 12)
NS = 20000


def quant_error(w, bits):
    """对称均匀量化到给定比特数时的最大绝对误差与相对 RMS 误差。"""
    amax = float(np.abs(w).max())
    if amax == 0:
        return 0.0, 0.0, 0.0
    levels = (1 << (bits - 1)) - 1
    step = amax / levels
    q = np.clip(np.round(w / step), -levels, levels) * step
    err = q - w
    return step, float(np.abs(err).max()), float(np.sqrt((err ** 2).mean()) / amax)


def layer_stats(W):
    a = np.abs(W)
    return {
        "shape": list(W.shape),
        "mults": int(W.shape[0] * W.shape[1]),
        "params": int(W.size),
        "absmax": float(a.max()),
        "absmean": float(a.mean()),
        "abs_p50": float(np.percentile(a, 50)),
        "abs_p99": float(np.percentile(a, 99)),
        "zero_frac": float((W == 0).mean()),
        "quant": {str(b): {"step": s, "max_abs_err": e, "rel_rms": r}
                  for b in BITS for (s, e, r) in [quant_error(W, b)]},
    }


def run_mlp(layers, X):
    """layers = [{W, b, relu}], X = (n, in). 返回每层输出与最终输出。"""
    acts = []
    v = X
    for L in layers:
        v = v @ np.asarray(L["W"]).T + np.asarray(L["b"])
        if L["relu"]:
            v = np.maximum(0.0, v)
        acts.append(v)
    return acts, v


def describe_act(a, name):
    flat = np.asarray(a).ravel()
    return {"layer": name,
            "min": float(flat.min()), "max": float(flat.max()),
            "mean": float(flat.mean()),
            "p01": float(np.percentile(flat, 1)),
            "p99": float(np.percentile(flat, 99)),
            "frac_zero": float((flat == 0).mean())}


def load_layers(path):
    """二期导出有两种 schema：连续输出的 gen.layers，调色板分类的 body + head。"""
    j = json.load(open(path, encoding="utf-8"))
    if "gen" in j:
        layers = [{"W": L["W"], "b": L["b"], "relu": True} for L in j["gen"]["layers"]]
        layers[-1]["relu"] = False
        return layers, j, "continuous"
    layers = [{"W": L["W"], "b": L["b"], "relu": True} for L in j["body"]]
    layers.append({"W": j["head"]["W"], "b": j["head"]["b"], "relu": False})
    return layers, j, "palette"


def spec_mlp(path, name, in_dim, note, post=None):
    layers, j, outkind = load_layers(path)

    rng = np.random.default_rng(0)
    X = np.clip(rng.standard_normal((NS, in_dim)), -3.5, 3.5)
    acts, out = run_mlp(layers, X)

    spec = {"name": name, "kind": "mlp", "note": note, "output_kind": outkind,
            "input": {"dim": in_dim, "distribution": "truncated normal, |z| <= 3.5",
                      "n_samples": NS},
            "layers": [], "activations": [], "totals": {}}
    if "palette" in j:
        spec["palette"] = {"K": len(j["palette"]),
                           "bits_per_pixel": j["meta"].get("bits_per_pixel"),
                           "colours": j["palette"]}
    if post:
        spec["post_process"] = post
    tot = 0
    params = 0
    for i, L in enumerate(layers):
        W = np.asarray(L["W"])
        st = layer_stats(W)
        st["layer"] = i
        st["bias"] = {"absmax": float(np.abs(L["b"]).max()),
                      "absmean": float(np.abs(L["b"]).mean())}
        spec["layers"].append(st)
        tot += st["mults"]
        params += st["params"]
    for i, a in enumerate(acts):
        spec["activations"].append(describe_act(a, i))
    spec["output"] = describe_act(out, "out")
    spec["totals"] = {"mults": tot, "params": params}
    return spec


def spec_quad(path):
    j = json.load(open(path, encoding="utf-8"))
    c = j["coef"]
    rng = np.random.default_rng(1)
    X = np.clip(rng.standard_normal((NS, 4)), -3.5, 3.5)
    v = np.full(len(X), c["c"])
    for i in range(4):
        v = v + c["linear"][i] * X[:, i] + c["square"][i] * X[:, i] ** 2
    for k, d in c["cross"].items():
        i, m = int(k[0]), int(k[1])
        v = v + d * X[:, i] * X[:, m]
    terms = {"linear": [abs(x) for x in c["linear"]],
             "square": [abs(x) for x in c["square"]],
             "cross": [abs(x) for x in c["cross"].values()]}
    allc = np.array(terms["linear"] + terms["square"] + terms["cross"])
    spec = {"name": "scorer_quad", "kind": "polynomial",
            "note": "二次打分器，输出坏度；1σ 坏率为 0 的六个风格依赖它",
            "input": {"dim": 4, "distribution": "truncated normal, |z| <= 3.5",
                      "n_samples": NS},
            "coefficients": {"bias": c["c"], "linear": c["linear"],
                             "square": c["square"], "cross": c["cross"]},
            "term_absmax": float(allc.max()), "term_absmean": float(allc.mean()),
            "quant": {str(b): {"step": s, "max_abs_err": e, "rel_rms": r}
                      for b in BITS for (s, e, r) in [quant_error(allc, b)]},
            "output_range": {"min": float(v.min()), "max": float(v.max()),
                             "p01": float(np.percentile(v, 1)),
                             "p99": float(np.percentile(v, 99))},
            "totals": {"mults": 14, "params": 15},
            "scale_note": "各项量级差别大：绝对值最大与平均相差 "
                          + f"{float(allc.max()/allc.mean()):.1f} 倍，"
                          "整体系数等比缩放不改变排序，可先归一化再量化"}
    return spec


print("=" * 88)
print("SPEC EXTRACT")
print("=" * 88)

GRAY_POST = ["clip to [0, 15]",
             "per-face min-max stretch to full range",
             "round to 16 levels, or merge to 11 map greys (see merge_mc11.json)"]
gray = spec_mlp(os.path.join(P2, "webapp", "pca22_dim4w64L1_model.json"),
                "grayscale_w64", 4, "灰度最终模型 4-64-32", post=GRAY_POST)
print(f"\n灰度 4-64-32   乘法 {gray['totals']['mults']:,}  参数 {gray['totals']['params']:,}")
for L in gray["layers"]:
    print(f"  层 {L['layer']}  {L['shape']}  乘法 {L['mults']:5d}  "
          f"|w|max {L['absmax']:.4f}  |w|均 {L['absmean']:.4f}  零占比 {100*L['zero_frac']:.1f}%")
for a in gray["activations"]:
    print(f"  激活 层{a['layer']}: [{a['min']:.3f}, {a['max']:.3f}]  "
          f"p1 {a['p01']:.3f}  p99 {a['p99']:.3f}  零占比 {100*a['frac_zero']:.1f}%")
ow = gray["output"]["max"] - gray["output"]["min"]
print(f"  原始输出 [{gray['output']['min']:.3f}, {gray['output']['max']:.3f}]  "
      f"跨度 {ow:.2f}")
print(f"  后处理: {' → '.join(GRAY_POST)}")
print(f"  注：原始输出跨度 {ow:.1f}，远宽于设计目标 0..15。"
      f"量化必须覆盖原始范围，否则 clip 之前的误差会改变拉伸结果")

COLOUR_POST = ["argmax over K logits per position",
               "palette lookup (8 map HIGH colours)",
               "mirror column"]
colour = spec_mlp(os.path.join(P2, "webapp", "colour_k8snap_model.json"),
                  "colour_k8", 8, "彩色最终模型 8-128-128-32x8", post=COLOUR_POST)
print(f"\n彩色 8-128-128-32x8   乘法 {colour['totals']['mults']:,}  "
      f"参数 {colour['totals']['params']:,}")
for L in colour["layers"]:
    print(f"  层 {L['layer']}  {L['shape']}  乘法 {L['mults']:6d}  "
          f"|w|max {L['absmax']:.4f}  |w|均 {L['absmean']:.4f}")
for a in colour["activations"]:
    print(f"  激活 层{a['layer']}: [{a['min']:.3f}, {a['max']:.3f}]  "
          f"零占比 {100*a['frac_zero']:.1f}%")

quad = spec_quad(os.path.join(P2, "webapp", "scorer_quad.json"))
print(f"\n二次打分器   14 乘法  系数 |c|max {quad['term_absmax']:.4f}  "
      f"|c|均 {quad['term_absmean']:.4f}")
print(f"  输出范围 [{quad['output_range']['min']:.3f}, {quad['output_range']['max']:.3f}]")

print()
print("=" * 88)
print("位宽与权重误差（对称均匀量化，误差相对 |w|max）")
print("=" * 88)
print(f"  {'部件':22s} " + " ".join(f"{'b='+str(b):>16s}" for b in BITS))
for spec, label in ((gray, "灰度 4-64-32"), (colour, "彩色 8-128-128-32x8")):
    for L in spec["layers"]:
        row = " ".join(f"{L['quant'][str(b)]['rel_rms']:16.4f}" for b in BITS)
        print(f"  {label+' 层'+str(L['layer']):22s} {row}")
row = " ".join(f"{quad['quant'][str(b)]['rel_rms']:16.4f}" for b in BITS)
print(f"  {'二次打分器系数':22s} {row}")

for spec, fn in ((gray, "spec_grayscale.json"), (colour, "spec_colour.json"),
                 (quad, "spec_scorer_quad.json")):
    p = os.path.join(RESULTS, fn)
    json.dump(spec, open(p, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"\nwrote {p}  ({os.path.getsize(p)/1e3:.1f} KB)")

print("\nSPEC_EXTRACT_DONE")
