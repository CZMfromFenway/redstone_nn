#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
强模 LUT 网络 vs MLP 保真度实验  (v6, 向量化)
==============================================
前向口径照搬恢复包 verify_teachers.py。

逐级约束, 测量每一级损失:
  L0  官方前向复现 (必须 100%)
  L1  权重/激活整体量化到 k 格点           -> 量化损失
  L2  每神经元激活换成单调阶跃 + 自由阈值    -> 激活约束损失
  L3  g 份共享每坐标表格 (g=1..h) + 每神经元自由阈值 + 计数读出
                                            -> 表格共享的代价 (本实验核心)
"""
import numpy as np
import json, os, time, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RECOV = os.path.join(os.path.dirname(ROOT),
                     "DT_Redstone_Research_Archive_20261001", "DT_baseline_recovery")
TEACH = os.path.join(RECOV, "teachers")
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)

X = np.indices((16,) * 4, dtype=np.int8).reshape(4, -1).T
Xn = (X.astype(np.float64) - 7.5) / 7.5
H_TABLE = np.load(os.path.join(RECOV, "legacy_H.npy"))
CI = [X[:, i].astype(np.int64) for i in range(4)]
N = 65536

def load(name):
    y = np.load(os.path.join(TEACH, name + ".npy"))
    with open(os.path.join(TEACH, name + ".json"), encoding="utf-8") as f:
        m = json.load(f)
    return y, m

def mlp_forward(m):
    return np.tanh(Xn @ np.array(m["W1"]).T + np.array(m["b1"])) @ np.array(m["W2"]) + m["b2"]

def count_forward(m):
    E = np.array(m["E"]); P = np.array(m["P"])
    Q = np.stack([P[j, sum(E[j, i, X[:, i]] for i in range(4))] for j in range(m["h"])], 1)
    return H_TABLE[m["head_id"], Q.sum(1)]

def err_of_threshold(score, y, t):
    return int(np.sum((score >= t) != y))

# ---------------------------------------------------------------
# 向量化共享表格模型
# ---------------------------------------------------------------
def make_cell_index(max_step):
    """cell[P, c] = 若表格第 P 个条目取值 <= c 则为 1"""
    P = max_step + 1
    cell = np.zeros((P, 16), dtype=np.int64)
    for p in range(P):
        cell[p, p:] = 1
    return cell

def eval_tables_bits(T, cnt_upto, cell, n_units):
    """
    T: (4,16) 表格值
    cnt_upto: (4,16) cnt_upto[i,j] = #{k<=j, k<=15}  -> 这里直接用 j (贡献的项数)
    返回按"逐神经元阈值扫描"选出的最优计数读出错误。
    做法: 用一维直方图最优分割近似 —— 对每个神经元用不同的分位阈值扫描, 选整体错误最小的组合。
    """
    # lam = sum_i T_i(x_i)
    lam = T[0][CI[0]] + T[1][CI[1]] + T[2][CI[2]] + T[3][CI[3]]
    return lam

def best_count_readout(lam, y, n_units, n_thr=9):
    """贪心: 依次为每个神经元选阈值, 使最终计数读出 + 全局阈值的错误最小。"""
    ys = y
    votes = np.zeros(N, dtype=np.int64)
    thetas = []
    for u in range(n_units):
        qs = np.percentile(lam, np.linspace(0, 100, n_thr + 2)[1:-1]).astype(np.int64)
        best = None
        for t in np.unique(qs):
            v = votes + (lam >= t)
            e, Tf = best_threshold(v, ys)
            if best is None or e < best[0]:
                best = (e, int(t), int(Tf), v)
        thetas.append(best[1])
        votes = best[3]
    e, Tf = best_threshold(votes, ys)
    return int(e), thetas, int(Tf), votes

def best_threshold(score, y):
    """O(N) 整数阈值搜索: 找 t 使 (score>=t) 与 y 的不一致最小。"""
    s = np.asarray(score).astype(np.int64)
    lo = int(s.min()); hi = int(s.max())
    idx = s - lo
    L = hi - lo + 1
    tot = np.bincount(idx, minlength=L)
    pos = np.bincount(idx, weights=y.astype(np.float64), minlength=L).astype(np.int64)
    cpos = np.concatenate([[0], np.cumsum(pos)])
    ctot = np.concatenate([[0], np.cumsum(tot)])
    P = int(y.sum())
    # 预测: idx < k 判 0, idx >= k 判 1
    err = (ctot - cpos) + (P - cpos)          # 前半的负类 + 后半的正类
    k = int(np.argmin(err))
    t = lo + k
    return int(err[k]), int(t)

def train_shared(y, n_units, n_tables, max_step, iters, seed):
    """
    n_tables 张共享表格 (n_tables = 4 * g, 每坐标 g 张), 所有神经元共用。
    神经元 u: bit_u = 1[ sum_{i,l} T_{i,l}(x_i) >= theta_u ]  (等价于对共享线性型取不同阈值)
    读出: sum_u bit_u >= Tfinal
    """
    rng = np.random.default_rng(seed)
    T = np.sort(rng.integers(0, max_step + 1, size=(n_tables, 16)), axis=1)
    # 每个坐标有 g 张表
    g = n_tables // 4
    groups = [np.arange(i * g, (i + 1) * g) for i in range(4)]

    def lam_of(T):
        s = np.zeros(N, dtype=np.int64)
        for i in range(4):
            for l in groups[i]:
                s += T[l][CI[i]]
        return s

    lam = lam_of(T)
    e, thetas, Tf, votes = best_count_readout(lam, y, n_units)
    best = (e, T.copy(), list(thetas), Tf)
    for it in range(iters):
        l = int(rng.integers(n_tables)); k = int(rng.integers(16))
        old = int(T[l, k])
        T[l, k] = int(rng.integers(0, max_step + 1))
        T[l] = np.sort(T[l])
        lam = lam_of(T)
        e2, th2, Tf2, _ = best_count_readout(lam, y, n_units)
        if e2 <= best[0]:
            if e2 < best[0]:
                best = (e2, T.copy(), list(th2), Tf2)
        else:
            T[l, k] = old; T[l] = np.sort(T[l])
    return best

def main():
    out = {}
    only_l12 = "--no-l3" in sys.argv
    print("=" * 96)
    print("强模 LUT 网络 vs MLP 保真度实验 v6   (官方前向口径, 65536 状态全枚举)")
    print("=" * 96)

    for name in ("N4_MLP_T1", "N4_MLP_T2", "N4_COUNT_T1", "N4_COUNT_T2"):
        y, m = load(name); yb = (y > 0)
        print(f"\n{'#'*96}\n### {name}  family={m.get('family')} seed={m.get('seed')} "
              f"正类={int(yb.sum())}\n{'#'*96}")
        rec = {"family": m.get("family"), "seed": m.get("seed"), "positives": int(yb.sum())}
        yi = yb.astype(np.int64)

        f = mlp_forward(m) if "MLP" in name else count_forward(m)
        ag0 = float(((f >= 0) == yb).mean())
        print(f"[L0] 官方前向复现: 一致率 = {ag0*100:.4f}%  "
              f"(不一致 {int(((f>=0)!=yb).sum())}/65536)")
        rec["L0_agreement"] = ag0
        out[name] = rec

        if "MLP" in name:
            W1 = np.array(m["W1"]); b1 = np.array(m["b1"])
            W2 = np.array(m["W2"]); b2 = float(m["b2"])
            Htrue = np.tanh(Xn @ W1.T + b1)
            targets = (Htrue >= 0).astype(np.int64)

            print("\n[L1] 权重/偏置整体乘 k 取整, tanh 量化到同格点, 线性整数读出")
            rows = []
            for k in (1, 4, 8, 16, 32, 64, 128, 256):
                kk = float(k)
                xi = np.round(Xn * kk).astype(np.int64)
                W1i = np.round(W1 * kk).astype(np.int64)
                b1i = np.round(b1 * kk).astype(np.int64)
                W2i = np.round(W2 * kk).astype(np.int64)
                b2i = np.round(b2 * kk)
                pre = xi @ W1i.T + b1i
                hq = np.round(np.tanh(pre / kk) * kk).astype(np.int64)
                agg = hq @ W2i + b2i
                pred = (agg >= 0)
                ag = float((pred == yb).mean()); err = int((pred != yb).sum())
                rows.append({"k": k, "agreement": ag, "errors": err,
                             "max_abs_weight": int(np.abs(W1i).max()),
                             "lut_height": int(np.abs(W1i).sum(axis=1).max())})
                print(f"   k={k:4d}: 一致率={ag*100:9.5f}%  错={err:6d}  "
                      f"|W1|max={int(np.abs(W1i).max()):5d}  "
                      f"LUT高度(最坏)={int(np.abs(W1i).sum(axis=1).max()):7d}")
            rec["L1_quantization"] = rows

            for k in (16, 64):
                print(f"\n[L2] k={k}: 每神经元激活换成'单调阶跃 + 自由阈值'")
                kk = float(k)
                pre = np.round(Xn * kk).astype(np.int64) @ np.round(W1 * kk).astype(np.int64).T \
                      + np.round(b1 * kk).astype(np.int64)
                W2i = np.round(W2 * kk).astype(np.int64); b2i = np.round(b2 * kk)
                bits = []; se = []
                for j in range(pre.shape[1]):
                    e, t = best_threshold(pre[:, j], targets[:, j])
                    bits.append((pre[:, j] >= t).astype(np.int64)); se.append(e)
                votes = np.stack(bits, 1).sum(1)
                e_cnt, t_cnt = best_threshold(votes, yi)
                agg = np.stack(bits, 1) @ W2i + b2i
                ag2 = float(((agg >= 0) == yb).mean())
                print(f"   各神经元 1bit 化误差 = {se}  (合计 {sum(se)})")
                print(f"   原输出权重读出: 一致率={ag2*100:.4f}%  错={int(((agg>=0)!=yb).sum())}")
                print(f"   纯计数读出: 错={e_cnt}/65536 = {e_cnt/65536*100:.3f}% "
                      f"(阈值 {t_cnt}/{pre.shape[1]})")
                rec[f"L2_k{k}"] = {"agreement_with_W2": ag2, "count_readout_errors": e_cnt,
                                   "self_bit_errors": se}

        # ---------- L3 ----------
        if only_l12:
            out[name] = rec
            continue
        print("\n[L3] g 份共享每坐标表格 (n_tables = 4g) + 每神经元自由阈值 + 计数读出")
        print(f"     (共享构型下表格总数 n_tables = 4g;  g=h 即每神经元独立表格 = 完整整数 MLP)")
        h_units = 15 if "MLP" not in name else int(np.array(m["W1"]).shape[0]) * 2
        for g, n_units, mes, iters in ((1, 15, 15, 1200), (2, 15, 15, 1200),
                                       (4, 15, 15, 1200), (8, 15, 15, 1200),
                                       (15, 15, 15, 1200)):
            n_tables = 4 * g
            t0 = time.time(); best = None
            for rs in range(2):
                r = train_shared(yi, n_units, n_tables, mes, iters, seed=200 + rs)
                if best is None or r[0] < best[0]:
                    best = r
                if time.time() - t0 > 120:
                    break
            print(f"   g={g:3d} (表格数 {n_tables:3d}, 神经元 {n_units}): "
                  f"最佳错误={best[0]}/65536 = {best[0]/65536*100:.3f}%  "
                  f"用时 {time.time()-t0:.0f}s")
            rec[f"L3_g{g}_errors"] = int(best[0])
        out[name] = rec

    path = os.path.join(RES, "mlp_fidelity.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n已写出:", path)

if __name__ == "__main__":
    main()
