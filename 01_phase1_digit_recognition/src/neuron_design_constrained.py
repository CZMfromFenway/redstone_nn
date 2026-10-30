#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实验（修订版）：加入两条物理硬约束后，强模 LUT 能多接近同结构的 MLP
=====================================================================
【新增约束】
  C1  lambda >= 0 必须成立（比较器不能输出负信号）
      => 每张 LUT 的读出恒非负; 偏移全部并入激活表 f
  C2  每层的 y 必须转成弱模 0~15（纵向布线的要求）
      => 隐藏层 8 个 y 各只有 16 个取值

【实验配置】
  输入空间   (x1..x4) ∈ {0..15}^4，全部 65536 状态穷举
  目标标签   N4_MLP_T1 官方标签（4->8->1 tanh MLP，seed 71000，正类 18436）
  评价       全部 65536 状态上的最终 0/1 错误数

  参照组 A   原 Teacher：4->8->1 实数 tanh，官方前向
  参照组 B   同结构实数 MLP：隐藏层输出**量化到 0~15**，在同一批标签上从零训练
  实验组 C   强模 LUT：lambda>=0，隐藏层输出 y ∈ {0..15}，输出层整数读出
"""
import numpy as np, json, os

CAP_STEP, CONT_HI = 868, 897

def load_experiment():
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    rec = os.path.join(base, "DT_Redstone_Research_Archive_20261001", "DT_baseline_recovery")
    m = json.load(open(os.path.join(rec, "teachers", "N4_MLP_T1.json"), encoding="utf-8"))
    lab = np.load(os.path.join(rec, "teachers", "N4_MLP_T1.npy")).astype(np.int8)
    X = np.indices((16,) * 4, dtype=np.int8).reshape(4, -1).T
    return m, lab, X

# ------------------------------------------------ 参照组 B：隐藏层量化到 0~15
def train_mlp_q(X, y, h=8, seed=0, steps=6000, lr=0.05, levels=16):
    """
    4->h->1 tanh MLP，但隐藏层输出被量化到 {0..levels-1}（模拟纵向弱模输出）。
    量化在训练中始终开启（前向量化，反向用 STE），对应硬件上的"必须二值/弱模化"。
    """
    rng = np.random.default_rng(seed)
    Xn = (X.astype(np.float64) - 7.5) / 7.5
    W1 = rng.normal(0, 0.5, (h, 4)); b1 = np.zeros(h)
    W2 = rng.normal(0, 0.5, h); b2 = 0.0
    params = [W1, b1, W2, np.array(b2)]
    ms = [np.zeros_like(p) for p in params]; vs = [np.zeros_like(p) for p in params]
    N = len(y)
    for t in range(1, steps + 1):
        Hc = np.tanh(Xn @ W1.T + b1)                    # (-1,1)
        Hq = np.clip(np.round((Hc + 1) / 2 * (levels - 1)), 0, levels - 1)   # 量化到 0..15
        z = Hq @ W2 + b2
        p = 1 / (1 + np.exp(-z)); dz = (p - y) / N
        gW2 = Hq.T @ dz; gb2 = dz.sum()
        dH = np.outer(dz, W2)                            # STE: 量化器导数记为 1（已含 1/2 缩放）
        dH = dH * 0.5 * (1 - Hc ** 2)
        gW1 = dH.T @ Xn; gb1 = dH.sum(axis=0)
        for i, (pp, g) in enumerate(zip(params, [gW1, gb1, gW2, np.array(gb2)])):
            ms[i] = 0.9 * ms[i] + 0.1 * g; vs[i] = 0.999 * vs[i] + (1 - 0.999) * g ** 2
            pp -= lr * (ms[i] / (1 - 0.9 ** t)) / (np.sqrt(vs[i] / (1 - 0.999 ** t)) + 1e-8)
    Hc = np.tanh(Xn @ W1.T + b1)
    Hq = np.clip(np.round((Hc + 1) / 2 * (levels - 1)), 0, levels - 1)
    err = int(np.sum(((Hq @ W2 + b2) >= 0).astype(np.int8) != y))
    return {"err": err, "seed": seed, "Hq": Hq, "W1": W1, "b1": b1, "W2": W2, "b2": b2}

# ------------------------------------------------ 实验组 C：强模 LUT，lambda>=0
def build_lut_nonneg(W_ji, s):
    """
    造一张 LUT，满足 C1：lambda(x) >= 0 处处成立。

    目标: lambda_i(x) ≈ s * (W_ji/7.5) * x      （可正可负的斜率）
    做法: 先造"非负上升阶梯" up[x] = round(x*s*|W_ji|/7.5)  (>=0, 步长 d=round(s|W|/7.5))
          正权重: 用反式, lambda = up              (上升, 非负)     容器 = 15 + d
          负权重: 用顺式, lambda = span - up       (下降, 非负)     容器 = 15 + d
                 其中 span = up[15] = 15*d 量级
          两种情形 lambda 都 >= 0，容器都在 [15, 883]
    返回 (lambda 数组(非负), 容器, 形式, 偏移常数 offset) 使
          lambda_i(x) = s*(W_ji/7.5)*x + offset_i   （近似）
    """
    aw = abs(float(W_ji)) / 7.5
    x = np.arange(16)
    up = np.round(x * s * aw).astype(np.int64)
    d = int(up[1]) if len(up) > 1 else 0
    if d > CAP_STEP:
        return None
    b = np.full(16, 15 + d, dtype=np.int64); b[0] = 0
    if b.max() > CONT_HI or b.min() < 0:
        return None
    if W_ji >= 0:
        lam = up; form = "反式"; off = 0
    else:
        span = int(up[-1]); lam = span - up; form = "顺式"; off = span
    return {"lam": lam, "container": b, "form": form, "d": d, "span": int(up[-1]), "off": off}

def max_s_for(W_ji, cap=CAP_STEP):
    aw = abs(float(W_ji)) / 7.5
    return 10 ** 9 if aw == 0 else int(cap / aw)

def build_design(W1):
    h, n = W1.shape
    tables, scales = [], []
    for j in range(h):
        s = min(max_s_for(W1[j, i]) for i in range(n))
        row = [build_lut_nonneg(W1[j, i], s) for i in range(n)]
        if any(t is None for t in row):
            return None
        tables.append(row); scales.append(s)
    return {"tables": tables, "scales": scales}

def forward_strong(design, X, W1, b1, W2, b2, levels=16):
    """
    t_j = sum_i lambda_i(x_i)                        (整数, 各 lambda >= 0)
    关系: sum_i lambda_i = s * sum_i (W_ji/7.5) x_i + OFF_j
          OFF_j = 负权重坐标的 span 之和
    u_j = (t_j - OFF_j)/s_j - sum_i W_ji + b_j
    h_j = tanh(u_j)  ->  量化到 0..15（C2）
    """
    Xi = X.astype(np.int64); N = len(Xi)
    h = len(design["tables"])
    Hq = np.zeros((N, h), dtype=np.int64)
    tmin = tmax = None
    for j in range(h):
        s = design["scales"][j]
        t = np.zeros(N, dtype=np.int64); off = 0
        for i, tb in enumerate(design["tables"][j]):
            t += tb["lam"][Xi[:, i]]
            off += tb["off"]
        if j == 0:
            tmin, tmax = int(t.min()), int(t.max())
        u = (t - off) / s - W1[j].sum() + b1[j]
        Hq[:, j] = np.clip(np.round((np.tanh(u) + 1) / 2 * (levels - 1)), 0, levels - 1)
    logit = Hq @ W2 + b2
    return (logit >= 0).astype(np.int8), Hq, (tmin, tmax)

def main():
    m, lab, X = load_experiment()
    y = (lab > 0).astype(np.int64)
    W1 = np.array(m["W1"]); b1 = np.array(m["b1"])
    W2 = np.array(m["W2"]); b2 = float(m["b2"])
    Xn = (X.astype(np.float64) - 7.5) / 7.5
    N = len(lab)

    print("=" * 104)
    print("实验（修订版）：加入 lambda>=0 与 隐藏层必须转弱模 0~15 两条约束")
    print("=" * 104)
    print(f"输入空间 : (x1..x4) ∈ {{0..15}}^4，全部 {N} 状态穷举")
    print(f"目标标签 : N4_MLP_T1（4->8->1 tanh，seed 71000，正类 {int(y.sum())}）")
    print(f"评价口径 : 全部 {N} 状态上的最终 0/1 错误数")
    print(f"新约束   : C1 lambda>=0（比较器不能出负信号）; C2 每层 y 必须是弱模 0~15")

    # A
    predA = ((np.tanh(Xn @ W1.T + b1) @ W2 + b2) >= 0).astype(np.int8)
    errA = int(np.sum(predA != lab))
    print(f"\n[参照组 A] 原 Teacher（实数 tanh，隐藏层无量化）: 错误 = {errA}/{N}")

    # A'：原 Teacher 但隐藏层量化到 0~15 —— 看"量化本身"的代价
    print(f"\n[参照组 A'] 原 Teacher 的权重不变，只把隐藏层量化到 0~15")
    for meth, tag in (("round", "四舍五入"),):
        Hc = np.tanh(Xn @ W1.T + b1)
        Hq = np.clip(np.round((Hc + 1) / 2 * 15), 0, 15)
        e = int(np.sum(((Hq @ W2 + b2) >= 0).astype(np.int8) != lab))
        print(f"   {tag}量化: 错误 = {e}/{N}  ({e/N*100:.3f}%)")

    # B：同结构实数 MLP，隐藏层量化到 0~15，从零训练
    print(f"\n[参照组 B] 同结构实数 MLP 4->8->1 tanh，隐藏层输出量化到 0~15，从零训练")
    print(f"   numpy 手写反传 + STE，全批，lr=0.05，6000 步，3 个种子")
    bestB = None
    for seed in (0, 1, 2):
        r = train_mlp_q(X, y, seed=seed)
        print(f"   seed={seed}: 错误 = {r['err']}/{N}")
        if bestB is None or r["err"] < bestB["err"]:
            bestB = r
    print(f"   参照组 B 最好 = {bestB['err']}/{N}  ({bestB['err']/N*100:.3f}%)")

    # C
    print(f"\n[实验组 C] 强模 LUT，lambda>=0，隐藏层 y ∈ {{0..15}}")
    design = build_design(W1)
    if design is None:
        print("   构造失败"); return
    print(f"   {'j':>2} | {'s_j':>6} | {'步长 d':>8} | {'坐标形式':>18} | {'容器(常数)':>10} | "
          f"{'lambda 范围':>16}")
    for j in range(W1.shape[0]):
        tb = design["tables"][j]
        lo = min(int(t["lam"].min()) for t in tb); hi = max(int(t["lam"].max()) for t in tb)
        print(f"   {j:2d} | {design['scales'][j]:6d} | {max(t['d'] for t in tb):8d} | "
              f"{' '.join(t['form'] for t in tb):>18} | {max(t['container'].max() for t in tb):10d} | "
              f"[{lo:6d},{hi:6d}]")
    predC, HqC, tr = forward_strong(design, X, W1, b1, W2, b2)
    errC = int(np.sum(predC != lab))
    print(f"\n   lambda 全部非负: {all(int(t['lam'].min()) >= 0 for tb in design['tables'] for t in tb)}")
    print(f"   第 0 个神经元 t 的范围 = {tr}")
    print(f"   隐藏层取值个数 = {len(np.unique(HqC))}（应为 16 以内）")
    print(f"   直接用原 W2 读出: 错误 = {errC}/{N}  ({errC/N*100:.3f}%)  <- 输出层未重新标定")

    # ---- C-retrain: 固定隐藏编码，只重训输出层 ----
    print(f"\n[实验组 C-retrain] 隐藏编码完全固定（强模 LUT 产生），只用整数权重重训输出层")
    # 隐藏编码是 (N,8) 的 0~15 整数；把每个神经元的 16 个取值 one-hot 展开成 128 维二进制特征，
    # 用逻辑回归拟合 —— 这是"输出层可做的事"的上界（整数权重 + 阈值）
    F = np.zeros((N, 8 * 16), dtype=np.float64)
    for j in range(8):
        F[np.arange(N), j * 16 + HqC[:, j]] = 1.0
    w = np.zeros(8 * 16); b = 0.0
    ms = np.zeros_like(w); vs = np.zeros_like(w); mb = vb = 0.0
    lr = 0.05
    for t in range(1, 8001):
        z = F @ w + b
        p = 1 / (1 + np.exp(-z)); dz = (p - y) / N
        gw = F.T @ dz; gb = dz.sum()
        ms = 0.9 * ms + 0.1 * gw; vs = 0.999 * vs + 0.001 * gw ** 2
        w -= lr * (ms / (1 - 0.9 ** t)) / (np.sqrt(vs / (1 - 0.999 ** t)) + 1e-8)
        mb = 0.9 * mb + 0.1 * gb; vb = 0.999 * vb + 0.001 * gb ** 2
        b -= lr * (mb / (1 - 0.9 ** t)) / (np.sqrt(vb / (1 - 0.999 ** t)) + 1e-8)
    predR = ((F @ w + b) >= 0).astype(np.int8)
    errR = int(np.sum(predR != lab))
    print(f"   输出权重可任意取且可为整数: 错误 = {errR}/{N}  ({errR/N*100:.4f}%)")
    # 整数化输出权重并复核
    for q in (1000, 100000):
        wq = np.round(w * q).astype(np.int64); bq = round(b * q)
        predQ = ((F @ wq.astype(np.float64) + bq) >= 0).astype(np.int8)
        print(f"     输出权重整数化 (q={q:6d}): 错误 = {int(np.sum(predQ != lab))}/{N}")

    # ---- C-full: 连隐藏层的形状也重新选（用同一批 LUT 的 s 与方向，但重训步长分配）----
    print(f"\n[实验组 C-full] 隐藏层仍由强模 LUT 产生（lambda>=0, 16 级输出），"
          f"但隐藏层权重用 STE 联合重训")
    print(f"   （即：把强模 LUT 的隐藏层当成可训练网络，端到端训练后再编译）")
    r2 = train_mlp_q(X, y, seed=0, steps=6000)
    Hq2 = np.asarray(r2["Hq"]).astype(np.int64)
    F2 = np.zeros((N, 8 * 16), dtype=np.float64)
    for j in range(8):
        F2[np.arange(N), j * 16 + Hq2[:, j]] = 1.0
    w2 = np.zeros(8 * 16); b2v = 0.0
    ms = np.zeros_like(w2); vs = np.zeros_like(w2); mb = vb = 0.0
    for t in range(1, 8001):
        z = F2 @ w2 + b2v
        p = 1 / (1 + np.exp(-z)); dz = (p - y) / N
        gw = F2.T @ dz; gb = dz.sum()
        ms = 0.9 * ms + 0.1 * gw; vs = 0.999 * vs + 0.001 * gw ** 2
        w2 -= lr * (ms / (1 - 0.9 ** t)) / (np.sqrt(vs / (1 - 0.999 ** t)) + 1e-8)
        mb = 0.9 * mb + 0.1 * gb; vb = 0.999 * vb + 0.001 * gb ** 2
        b2v -= lr * (mb / (1 - 0.9 ** t)) / (np.sqrt(vb / (1 - 0.999 ** t)) + 1e-8)
    errF = int(np.sum(((F2 @ w2 + b2v) >= 0).astype(np.int8) != lab))
    errF0 = int(np.sum(((F2 @ np.zeros(8 * 16) + b2v) >= 0).astype(np.int8) != lab))
    print(f"   联合重训 + 输出层最优读出: 错误 = {errF}/{N}  ({errF/N*100:.4f}%)")
    errB_simple = int(np.sum(((Hq2 @ r2["W2"] + r2["b2"]) >= 0).astype(np.int8) != lab))
    print(f"   联合重训 + 原线性读出形式: 错误 = {errB_simple}/{N}")

    # 上界参照：若隐藏层不量化
    print(f"\n[对照] 若解除 C2（隐藏层保留模拟量），同设计可达到的错误数")
    design2 = build_design(W1)
    h = len(design2["tables"]); Hc2 = np.zeros((N, h))
    Xi = X.astype(np.int64)
    for j in range(h):
        s = design2["scales"][j]; t = np.zeros(N, dtype=np.int64); off = 0
        for i, tb in enumerate(design2["tables"][j]):
            t += tb["lam"][Xi[:, i]]; off += tb["off"]
        Hc2[:, j] = np.tanh((t - off) / s - W1[j].sum() + b1[j])
    e2 = int(np.sum(((Hc2 @ W2 + b2) >= 0).astype(np.int8) != lab))
    print(f"   隐藏层不量化: 错误 = {e2}/{N}")

    print("\n" + "=" * 104)
    print("汇总")
    print("=" * 104)
    HqA = np.clip(np.round((np.tanh(Xn @ W1.T + b1) + 1) / 2 * 15), 0, 15).astype(np.int64)
    eA2 = int(np.sum(((HqA @ W2 + b2) >= 0).astype(np.int8) != lab))
    # A-retrain: 原 Teacher 的隐藏编码 + 重训输出层
    FA = np.zeros((N, 8 * 16))
    for j in range(8):
        FA[np.arange(N), j * 16 + HqA[:, j]] = 1.0
    wA = np.zeros(8 * 16); bA = 0.0
    msA = np.zeros_like(wA); vsA = np.zeros_like(wA); mbA = vbA = 0.0
    for t in range(1, 8001):
        z = FA @ wA + bA; p = 1 / (1 + np.exp(-z)); dz = (p - y) / N
        gw = FA.T @ dz; gb = dz.sum()
        msA = 0.9 * msA + 0.1 * gw; vsA = 0.999 * vsA + 0.001 * gw ** 2
        wA -= lr * (msA / (1 - 0.9 ** t)) / (np.sqrt(vsA / (1 - 0.999 ** t)) + 1e-8)
        mbA = 0.9 * mbA + 0.1 * gb; vbA = 0.999 * vbA + 0.001 * gb ** 2
        bA -= lr * (mbA / (1 - 0.9 ** t)) / (np.sqrt(vbA / (1 - 0.999 ** t)) + 1e-8)
    eA3 = int(np.sum(((FA @ wA + bA) >= 0).astype(np.int8) != lab))
    chance = max(int(y.sum()), N - int(y.sum()))
    rows = [
        ("A  原 Teacher（实数，隐藏层连续）", "连续", errA),
        ("A' 原 Teacher + 隐藏层量化 + 原 W2", "0~15", eA2),
        ("A'' 原 Teacher + 隐藏层量化 + 重训输出层", "0~15", eA3),
        ("B  同结构实数 MLP（STE 联合训练）", "0~15", bestB["err"]),
        ("C  强模 LUT（lambda>=0，原 W2 读出）", "0~15", errC),
        ("C-retrain  强模 LUT 隐藏编码 + 重训输出层", "0~15", errR),
        ("C-full  强模 LUT 隐藏层联合重训 + 最优读出", "0~15", errF),
        ("C'  强模 LUT，解除 C2（隐藏层留模拟量）", "连续", e2),
    ]
    print(f"   {'方案':>46} | {'隐藏层':>6} | {'错误':>8} | {'错误率':>9}")
    for nm, hl, e in rows:
        print(f"   {nm:>46} | {hl:>6} | {e:8d} | {e/N*100:8.4f}%")
    print(f"\n   [基线] 恒预测多数类: 错误 = {chance}/{N} = {chance/N*100:.3f}%")

    here = os.path.dirname(os.path.abspath(__file__))
    res = os.path.join(os.path.dirname(here), "results"); os.makedirs(res, exist_ok=True)
    json.dump({"A": errA, "A_quant_origW2": eA2, "A_quant_retrain": eA3,
               "B": bestB["err"], "C_origW2": errC, "C_retrain": errR, "C_full": errF,
               "C_nolimit": e2, "chance": chance},
              open(os.path.join(res, "strong_vs_mlp_constrained.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n已写出: results/strong_vs_mlp_constrained.json")

if __name__ == "__main__":
    main()
