#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
纯整数管线。所有编译期常数都是整数，运行期只做整数加、减、比较、右移。

关键约定
    1. 位权重 Tb[p][i][j] 是整数；j < b-1 的受 882 约束，j = b-1 可到 2^31-1
    2. F' = sum Tb * Xbit 是整数；F = F' - F'lo，无符号
    3. A（锚点）与 c（步长）都是整数，Y = clip(floor((F - A) / c), 0, 2^b-1)
    4. 输出层 32 个神经元共用同一个 scale 与同一个 c，
       只让 A 逐神经元不同，这样 32 个 Y 才是同一把尺子量出来的
    5. 后处理全程整数：span = Ymax - Ymin, step = span >> 4,
       level = 15 - clip((Ymax - Y) // step, 0, 15)
"""
import numpy as np

LIMIT = 882
TOP_LIMIT = (1 << 31) - 1


def compile_layer(Wf, th, cunit, b, relu, xbase=0.0, ymax_in=None,
                  ztop=None, shared_scale=False):
    """编译一层。返回全部为整数的常数。

    Wf       (out, in) 浮点权重
    cunit    每个输入一个整数级对应的物理量（标量或长度 in 的向量）
    xbase    输入值为 0 时对应的物理量（层 1 是 -3.5，层 2 是 0）
    ymax_in  输入整数的可达上界（长度 in，默认 2^b-1）
    ztop     该层输出对应的物理上界。ReLU 层为 None（按各神经元的解析上界），
             输出层给 15.0，表示只在 [0, 15] 上量化
    shared_scale  True 时整层共用一个缩放，这是输出层必须的
    """
    out, nin = Wf.shape
    N = 1 << b
    cu = np.full(nin, float(cunit)) if np.isscalar(cunit) else np.asarray(cunit, float)
    xb = np.full(nin, float(xbase)) if np.isscalar(xbase) else np.asarray(xbase, float)
    top = np.full(nin, N - 1, float) if ymax_in is None else np.asarray(ymax_in, float)

    # 缩放：把 j = b-2 位的最大权重推到 882
    amax = np.abs(Wf * cu[None, :]).max(axis=1)
    if shared_scale:
        s = np.full(out, LIMIT / (amax.max() * (1 << (b - 2))))
    else:
        s = LIMIT / (amax * (1 << (b - 2)))

    Tb = np.rint(s[:, None, None] * Wf[:, :, None] * cu[None, :, None]
                 * (1 << np.arange(b))[None, None, :]).astype(np.int64)
    # 硬校验：只有最高位可以越过 882
    over = np.abs(Tb[:, :, :-1]).max() if b > 1 else 0
    assert over <= LIMIT, f"非最高位权重 {over} 超过 {LIMIT}"
    assert np.abs(Tb).max() <= TOP_LIMIT, "最高位权重超过强模上限"

    Flo = np.where(Tb < 0, Tb, 0).sum(axis=(1, 2))          # F' 的最小可达值
    Fhi = np.where(Tb > 0, Tb, 0).sum(axis=(1, 2))          # F' 的最大可达值

    # C0：z = C0 + cu . (w * X)
    C0 = th + (xb * Wf).sum(axis=1) + 0.5 * (Wf * cu[None, :]).sum(axis=1)

    # A = z = 0 处的 F，取整到整数
    A = np.rint(-s * C0 - Flo).astype(np.int64)

    # c：从 z = 0 到该层输出上界，共 2^b 格
    if ztop is not None:
        zhi = np.full(out, float(ztop))
    else:
        zhi = C0 + ((Wf.clip(min=0) * cu[None, :]) * top[None, :]).sum(axis=1)
    c = np.maximum(np.ceil(s * zhi / N), 1).astype(np.int64)
    if shared_scale:
        c = np.full(out, int(c.max()))                      # 输出层：全局同一步长

    return dict(Tb=Tb, s=s, Flo=Flo, Fhi=Fhi, C0=C0, A=A, c=c,
                b=b, N=N, cu=cu, relu=relu, shared_scale=shared_scale)


def bits_of(X, b):
    return ((X[:, :, None] >> np.arange(b)[None, None, :]) & 1).astype(np.int64)


def run(X, L):
    """一层整数前向。返回 F' 与 Y，全部整数。"""
    Fp = np.einsum("nij,kij->nk", bits_of(X, L["b"]), L["Tb"])
    F = Fp - L["Flo"][None, :]
    Y = np.clip(np.floor_divide(F - L["A"][None, :], L["c"][None, :]),
                0, L["N"] - 1).astype(np.int64)
    return Fp, F, Y


def to_levels(Y, bits=4):
    """后处理：整形的逐张 min-max，右移取步长，自顶向下数级。

    返回 0..15 的整数级。全部整数运算，只有一次右移。
    """
    NLEV = (1 << bits) - 1
    ymax = Y.max(axis=1, keepdims=True)
    ymin = Y.min(axis=1, keepdims=True)
    span = ymax - ymin
    step = span >> bits
    step = np.maximum(step, 1)                              # s=0 时退化为 1
    up = np.floor_divide(ymax - Y, step)                    # 距最高值的步数
    up = np.clip(up, 0, NLEV)
    return (NLEV - up).astype(np.int64)


def to_float_face(z, bits=4):
    NLEV = (1 << bits) - 1
    v = np.clip(z, 0, 15)
    lo, hi = v.min(axis=1, keepdims=True), v.max(axis=1, keepdims=True)
    r = np.clip((v - lo) / np.maximum(hi - lo, 1e-6), 0, 1)
    return np.rint(r * NLEV).astype(np.int64)
