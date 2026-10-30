# -*- coding: utf-8 -*-
"""导出 Python 侧编译结果，供与 JS 对比。

严格照抄 03_phase3_quantized_network/src/compiler.py 的公式：
    delta = (xmax-xmin)/2^in_bits
    sig   = w_ref / (|w|*delta 的逐通道最大值 * 2^(in_bits-2))
    W     = rint(sig * w * delta * 2^j)
"""
import json
import os
import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = r"E:\redstone_nn"
MODEL = os.path.join(ROOT, "02_phase2_avatar", "webapp",
                     "pca22_dim4w64L1_model.json")
XMIN, XMAX, INB = -3.5, 3.5, 8
LIMIT = 882


def compile_layer(w, in_bits, delta, w_ref=LIMIT):
    """返回 (sig, W)，W 形状 (m, n, b)。"""
    w = np.asarray(w, dtype=np.float64)
    g = np.abs(w) * delta[None, :]
    sig = w_ref / (g.max(axis=1) * (1 << (in_bits - 2)))
    bits = np.arange(in_bits)
    W = np.rint(sig[:, None, None] * w[:, :, None]
                * delta[None, :, None]
                * (1 << bits)[None, None, :]).astype(np.int64)
    return sig, W


def main():
    mj = json.load(open(MODEL, encoding="utf-8"))
    L1 = np.array(mj["gen"]["layers"][0]["W"])   # (64,4)
    L2 = np.array(mj["gen"]["layers"][1]["W"])   # (32,64)

    delta1 = np.full(4, (XMAX - XMIN) / (1 << INB))

    out = {"meta": {"xmin": XMIN, "xmax": XMAX, "in_bits": INB, "limit": LIMIT},
           "layers": {}}

    # ---- 层1：每个 b 都算一遍
    out["layers"]["L1"] = {}
    for b in (8, 9, 10, 11, 12):
        sig, W = compile_layer(L1, b, delta1)
        out["layers"]["L1"][str(b)] = {
            "sig": sig.tolist(),
            "W": W.tolist(),
            "maxAbsNonTop": int(np.abs(W[:, :, :-1]).max()) if b > 1 else 0,
            "maxAbsTop": int(np.abs(W[:, :, -1]).max()) if b > 1 else 0,
        }
        print(f"  层1 b={b}: |W| 非最高位最大 {out['layers']['L1'][str(b)]['maxAbsNonTop']}"
              f"  最高位最大 {out['layers']['L1'][str(b)]['maxAbsTop']}")

    # ---- 层2：in_bits 由层1 的 out_bits=8 决定，delta 逐通道
    # 需要先算出层1 的 delta_out = c/sig；这里用 b=8 的层1 结果
    sig1, W1 = compile_layer(L1, 8, delta1)
    Cmin1 = np.where(W1 < 0, W1, 0).sum(axis=(1, 2))
    Cmax1 = np.where(W1 > 0, W1, 0).sum(axis=(1, 2))
    Nmax1 = Cmax1 - Cmin1
    th1 = np.array(mj["gen"]["layers"][0]["b"], dtype=np.float64)
    kappa1 = th1 + (L1 * (XMIN + 0.5 * delta1)[None, :]).sum(axis=1)
    P1 = np.rint(-sig1 * kappa1 - Cmin1).astype(np.int64)
    c1 = np.maximum(np.ceil((Nmax1 - P1) / ((1 << 8) - 1)), 1).astype(np.int64)
    dout1 = c1 / sig1

    out["layers"]["L2"] = {}
    for b in (8, 10):
        sig2, W2 = compile_layer(L2, b, dout1)
        out["layers"]["L2"][str(b)] = {
            "sig": sig2.tolist(),
            "W": W2.tolist(),
            "maxAbsNonTop": int(np.abs(W2[:, :, :-1]).max()) if b > 1 else 0,
            "maxAbsTop": int(np.abs(W2[:, :, -1]).max()) if b > 1 else 0,
            "delta_in": dout1.tolist(),
        }
        print(f"  层2 b={b}: |W| 非最高位最大 {out['layers']['L2'][str(b)]['maxAbsNonTop']}"
              f"  最高位最大 {out['layers']['L2'][str(b)]['maxAbsTop']}")

    p = os.path.join(ROOT, "03_phase3_quantized_network", "results",
                     "py_compile_reference.json")
    json.dump(out, open(p, "w", encoding="utf-8"))
    print(f"\n  写出 {p}  ({os.path.getsize(p)/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
