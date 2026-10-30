#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
把 ReLU / CLIP 型 MLP 编译成整数管线。

只支持两种层类型
    "relu"  输出压成 out_bits 位整数，带零点 P_i 与步长 c_i
    "clip"  输出按实数上界 lambda_i 截断，带零点 P_i 与整数上界 T_i

不负责后处理。只输出最后一层输出的整数理论范围。

三个开关
    same_span   由输入分辨率是否一致自动判定（一致则 delta 是标量）
    comparable  逐层显式指定；决定 sigma 是整层一个还是逐通道
    kind        "relu" 或 "clip"

记号
    sigma / sigma_i        权重缩放因子
    W^(k)_{i,j}            位权重
    C_i, C_{i,min}         原始累加及其最小值
    N_i = C_i - C_{i,min}  非负累加量
    P_i                    预激活整数零点
    c_i                    输出量化步长（仅 relu）
    T_i                    整数截断上限，lambda_i 是实数截断上限（仅 clip）
    delta_j                输入实数分辨率
"""
import json
import os
import sys
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

LIMIT = 882
TOP = (1 << 31) - 1


class Layer:
    """一层的浮点定义 + 设计选择。"""

    def __init__(self, w, theta, kind, comparable=False, out_bits=None,
                 lam=None, name="layer"):
        self.w = np.asarray(w, dtype=np.float64)
        self.theta = np.asarray(theta, dtype=np.float64)
        self.kind = kind
        self.comparable = comparable
        self.out_bits = out_bits
        self.lam = None if lam is None else np.asarray(lam, dtype=np.float64)
        self.name = name
        assert self.w.ndim == 2 and self.w.shape[0] == len(self.theta)
        assert kind in ("relu", "clip")
        if kind == "relu":
            assert out_bits is not None, "relu 层要给 out_bits"
        if kind == "clip":
            assert lam is not None, "clip 层要给 lam"


def compile_network(layers, xmin, xmax, in_bits, verbose=True, w_ref=LIMIT):
    """编译整条链，返回每层的整数常数与最后一层的输出范围。

    w_ref 决定把次高位位权重推到多大：
        w_ref = 882   次高位满 882，最高位到 1764（需要"最高位可超"的许可）
        w_ref = 441   次高位 441，最高位 882（全部不超过单容器上限）
    """
    m0 = layers[0].w.shape[1]
    delta = np.full(m0, (xmax - xmin) / (1 << in_bits))     # 第一层：四个输入同分辨率
    a0 = np.full(m0, xmin)                                  # 第一层：A=0 对应 x_min
    in_bits_cur = in_bits
    out = []
    for li, L in enumerate(layers):
        m, n = L.w.shape
        same_span = bool(np.ptp(delta) == 0)

        # ---- 步 1：权重缩放因子
        g = np.abs(L.w) * delta[None, :]
        if L.comparable:
            sig = np.full(m, w_ref / (g.max() * (1 << (in_bits_cur - 2))))
        else:
            sig = w_ref / (g.max(axis=1) * (1 << (in_bits_cur - 2)))

        # ---- 步 2：位权重与非负累加量
        W = np.rint(sig[:, None, None] * L.w[:, :, None] * delta[None, :, None]
                    * (1 << np.arange(in_bits_cur))[None, None, :]).astype(np.int64)
        over = int(np.abs(W[:, :, :-1]).max()) if in_bits_cur > 1 else 0
        topv = int(np.abs(W[:, :, -1]).max()) if in_bits_cur > 1 else 0
        assert over <= LIMIT, f"{L.name}: 非最高位权重 {over} 超过单容器上限 {LIMIT}"
        assert topv <= TOP, f"{L.name}: 最高位权重 {topv} 超过强模上限"
        Cmin = np.where(W < 0, W, 0).sum(axis=(1, 2))
        Cmax = np.where(W > 0, W, 0).sum(axis=(1, 2))
        Nmax = Cmax - Cmin

        # ---- 步 3：预激活整数零点
        kappa = L.theta + (L.w * (a0 + 0.5 * delta)[None, :]).sum(axis=1)
        P = np.rint(-sig * kappa - Cmin).astype(np.int64)

        rec = dict(name=L.name, kind=L.kind, in_dim=n, out_dim=m,
                   in_bits=in_bits_cur, same_span=same_span,
                   comparable=L.comparable, sigma=sig, W=W,
                   Cmin=Cmin, Cmax=Cmax, Nmax=Nmax, P=P,
                   delta_in=delta, a0=a0, w=L.w.copy(), theta=L.theta.copy())

        # ---- 步 4：输出
        if L.kind == "relu":
            ob = L.out_bits
            c = np.maximum(np.ceil((Nmax - P) / ((1 << ob) - 1)), 1).astype(np.int64)
            ymax = np.clip(np.floor_divide(Nmax - P, c), 0, (1 << ob) - 1)
            rec.update(out_bits=ob, c=c, out_min=np.zeros(m, dtype=np.int64),
                       out_max=ymax)
            d_out = c / sig
        else:
            T = np.rint(L.lam * sig).astype(np.int64)
            ob = int(np.ceil(np.log2(T.max() + 1)))
            rec.update(out_bits=ob, T=T, lam=L.lam, out_min=np.zeros(m, dtype=np.int64),
                       out_max=T.copy())
            d_out = 1.0 / sig

        rec["delta_out"] = d_out
        out.append(rec)

        # 下一层
        delta = d_out
        a0 = np.zeros(m)
        in_bits_cur = rec["out_bits"]

        if verbose:
            print(f"  [{L.name}] {n}×{m}  kind={L.kind:4s}  "
                  f"same_span={'Y' if same_span else 'N'}  "
                  f"comparable={'Y' if L.comparable else 'N'}")
            print(f"        σ {'整层 ' + format(sig[0], '.2f') if L.comparable else '逐通道 [' + format(sig.min(), '.1f') + ', ' + format(sig.max(), '.1f') + ']'}"
                  f"   |W| 非最高位 {over}，最高位 {topv}")
            print(f"        N_i ∈ [0, {int(Nmax.max()):,}]   P_i ∈ [{int(P.min())}, {int(P.max())}]"
                  if L.comparable else
                  f"        N_i ∈ [0, {int(Nmax.max()):,}]   P_i ∈ [{int(P.min())}, {int(P.max())}]")
            if L.kind == "relu":
                print(f"        c_i ∈ [{int(c.min())}, {int(c.max())}]   "
                      f"输出 {ob} 位，实测上界 [{int(ymax.min())}, {int(ymax.max())}]")
            else:
                print(f"        T_i ∈ [{int(T.min())}, {int(T.max())}]   "
                      f"λ_i ∈ [{L.lam.min():g}, {L.lam.max():g}]   输出 {ob} 位")
            print(f"        δ_out ∈ [{d_out.min():.6g}, {d_out.max():.6g}]")

    return out


def integer_forward(comp, X):
    """按编译结果跑整数前向。X 是 (N, in_dim) 的第一层输入整数。"""
    A = np.asarray(X, dtype=np.int64)
    for r in comp:
        b = r["in_bits"]
        bits = ((A[:, :, None] >> np.arange(b)[None, None, :]) & 1).astype(np.int64)
        C = np.einsum("njk,ijk->ni", bits, r["W"])
        N = C - r["Cmin"][None, :]
        S = N - r["P"][None, :]
        if r["kind"] == "relu":
            A = np.clip(np.floor_divide(S, r["c"][None, :]), 0, (1 << r["out_bits"]) - 1)
        else:
            A = np.clip(S, 0, r["T"][None, :])
    return A


def float_forward(layers, x, xmin, xmax, in_bits):
    """按编译时同样的约定跑浮点：第一层吃格心重建值。"""
    delta = (xmax - xmin) / (1 << in_bits)
    a = xmin + (np.asarray(x, dtype=np.float64) + 0.5) * delta
    for L in layers:
        c = a @ L.w.T + L.theta
        if L.kind == "relu":
            a = np.maximum(0.0, c)
        else:
            a = np.clip(c, 0, L.lam)
    return a


# ======================================================================
# 演示：编译二期头像模型
# ======================================================================
if __name__ == "__main__":
    HERE = os.path.dirname(os.path.abspath(__file__))
    ROOT = os.path.dirname(HERE)
    P2 = os.path.join(os.path.dirname(ROOT), "02_phase2_avatar")
    mj = json.load(open(os.path.join(P2, "webapp", "pca22_dim4w64L1_model.json"),
                        encoding="utf-8"))

    XMIN, XMAX, INB = -3.5, 3.5, 8

    print("=" * 96)
    print("编译二期灰度头像模型")
    print("=" * 96)
    layers = [
        Layer(np.array(mj["gen"]["layers"][0]["W"]),
              np.array(mj["gen"]["layers"][0]["b"]),
              "relu", comparable=False, out_bits=8, name="层1 4→64"),
        Layer(np.array(mj["gen"]["layers"][1]["W"]),
              np.array(mj["gen"]["layers"][1]["b"]),
              "clip", comparable=True, lam=np.full(32, 15.0), name="层2 64→32"),
    ]
    comp = compile_network(layers, XMIN, XMAX, INB)

    last = comp[-1]
    print()
    print("=" * 96)
    print("最后一层的整数理论范围")
    print("=" * 96)
    print(f"  输出个数 {last['out_dim']}，位宽 {last['out_bits']} 位")
    print(f"  每通道范围 [{int(last['out_min'].min())}, {int(last['out_max'].min())}]"
          f" ~ [{int(last['out_min'].max())}, {int(last['out_max'].max())}]")
    print(f"  全局范围 [0, {int(last['out_max'].max())}]"
          f"   ← 需要 {int(np.ceil(np.log2(int(last['out_max'].max()) + 1)))} 位")
    print(f"  非负累加量 N_i 的上界 {int(last['Nmax'].max()):,}"
          f"   ← 需要 {int(np.ceil(np.log2(int(last['Nmax'].max()) + 1)))} 位（中间量）")

    # ---------------- 验证：物理单位比较 ----------------
    def decode(r, B):
        """把整数输出解码回该层的物理预激活值（格心）。"""
        sg = r["sigma"][None, :]
        if r["kind"] == "relu":
            return (B + 0.5) * (r["c"][None, :] / sg)
        return (B + 0.5) / sg

    def layer_ideal(r, A_int):
        """同一批整数输入走浮点，得到该层的物理预激活。"""
        a_hat = r["a0"][None, :] + (A_int + 0.5) * r["delta_in"][None, :]
        return a_hat @ r["w"].T + r["theta"]

    rng = np.random.default_rng(41)
    NF = 20000
    xs = np.clip(rng.standard_normal((NF, 4)), XMIN, XMAX)
    Xq = np.clip(np.floor((xs - XMIN) / (XMAX - XMIN) * (1 << INB)), 0, (1 << INB) - 1)

    print()
    print("=" * 96)
    print(f"逐层验证（{NF:,} 张）：整数解码值 vs 同一整数输入的浮点预激活")
    print("=" * 96)
    A = Xq.astype(np.int64)
    for r in comp:
        b = r["in_bits"]
        bits = ((A[:, :, None] >> np.arange(b)[None, None, :]) & 1).astype(np.int64)
        C = np.einsum("njk,ijk->ni", bits, r["W"])
        S = (C - r["Cmin"][None, :]) - r["P"][None, :]
        if r["kind"] == "relu":
            B = np.clip(np.floor_divide(S, r["c"][None, :]), 0, (1 << r["out_bits"]) - 1)
        else:
            B = np.clip(S, 0, r["T"][None, :])
        pre_ideal = layer_ideal(r, A)
        pre_int = decode(r, B)
        # 只看未被截断的区域（截断处的差是下界的定义，不算误差）
        mask = (pre_ideal > 0) & (pre_ideal < (r["lam"].max() if r["kind"] == "clip" else np.inf))
        if r["kind"] == "relu":
            mask = pre_ideal > 0
        d = (pre_int - pre_ideal)[mask]
        span = float(pre_ideal.max() - pre_ideal.min())   # 物理跨度，不是整数跨度
        print(f"  [{r['name']}] 解码步长 {float(np.median(r['c']/r['sigma'] if r['kind']=='relu' else 1/r['sigma'])):.6f}")
        print(f"        误差 平均 {np.abs(d).mean():.6f}  最大 {np.abs(d).max():.6f}  "
              f"（物理跨度 {span:.3f}，占 {100*np.abs(d).max()/span:.4f}%）")
        A = B

    # 端到端：最后一层解码后与浮点整链比较
    Zi = integer_forward(comp, Xq)
    Zf = float_forward(layers, Xq, XMIN, XMAX, INB)
    last = comp[-1]
    phys_int = (Zi + 0.5) / last["sigma"][0]          # 解码回物理量
    phys_flt = np.clip(Zf, 0, 15.0)
    e = phys_int - phys_flt
    print()
    print("=" * 96)
    print(f"端到端（{NF:,} 张）：最后一层解码后的物理量 vs 浮点整链")
    print("=" * 96)
    print(f"  误差 平均 {np.abs(e).mean():.4f}   最大 {np.abs(e).max():.4f}   "
          f"（物理跨度 15，占 {100*np.abs(e).max()/15:.2f}%）")
    print(f"  换算成 16 级：平均 {np.abs(e).mean()/15*16:.3f} 级，"
          f"最大 {np.abs(e).max()/15*16:.3f} 级")
    print(f"  整数输出实测范围 [{Zi.min()}, {Zi.max()}]（理论 [0, {int(last['out_max'].max())}]）")

    # 导出网表
    nl = {"meta": {"in_dim": 4, "in_bits": INB, "xmin": XMIN, "xmax": XMAX,
                   "layers": len(comp)},
          "layers": []}
    for r in comp:
        nl["layers"].append({
            "name": r["name"], "kind": r["kind"],
            "in_dim": int(r["in_dim"]), "out_dim": int(r["out_dim"]),
            "in_bits": int(r["in_bits"]), "out_bits": int(r["out_bits"]),
            "same_span": bool(r["same_span"]), "comparable": bool(r["comparable"]),
            "sigma": [float(v) for v in np.atleast_1d(r["sigma"])][:1]
                     if r["comparable"] else [float(v) for v in r["sigma"]],
            "W": r["W"].tolist(),
            "Cmin": r["Cmin"].tolist(), "Nmax": r["Nmax"].tolist(),
            "P": r["P"].tolist(),
            ("c" if r["kind"] == "relu" else "T"):
                [int(v) for v in (r["c"] if r["kind"] == "relu" else r["T"])],
            "out_min": [int(v) for v in r["out_min"]],
            "out_max": [int(v) for v in r["out_max"]],
            "delta_out": [float(v) for v in r["delta_out"]],
        })
    out = os.path.join(ROOT, "results", "netlist_grayscale.json")
    json.dump(nl, open(out, "w", encoding="utf-8"))
    print(f"\n  网表已导出 {out}  ({os.path.getsize(out)/1e6:.2f} MB)")
    print("COMPILER_DONE")
