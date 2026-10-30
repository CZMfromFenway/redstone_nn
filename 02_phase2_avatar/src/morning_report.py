#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Morning report: aggregate everything the night produced into one readable summary.

Defensive by design: the previous version crashed on a single wrong key
(lam_sweep.json uses "rows", not "results") and therefore produced no report at all.
Every read here goes through .get() with a default, and a failure in one section can no
longer prevent the other sections or the file itself.
"""
import glob, json, os, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
R = os.path.join(ROOT, "results")
OUT = os.path.join(R, "overnight", "MORNING_REPORT.md")
os.makedirs(os.path.dirname(OUT), exist_ok=True)


def jload(p):
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def jlload(p):
    out = []
    try:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except Exception:
                        pass
    except Exception:
        pass
    return out


def fmt(v, nd=4, pct=False, default="—"):
    try:
        if v is None:
            return default
        return f"{v*100:.1f}%" if pct else f"{float(v):.{nd}f}"
    except Exception:
        return default


L = []
try:
    import shutil, time
    L.append("# 夜间实验结果汇总（自动生成）")
    L.append("")
    L.append(f"- 生成时刻：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    try:
        L.append(f"- 剩余磁盘：{shutil.disk_usage(ROOT).free/1e9:.1f} GB")
    except Exception:
        pass
    L.append("")

    # ------------------------------------------------------------ 0 runner log
    L.append("## 0. 调度器日志")
    L.append("")
    log = jlload(os.path.join(R, "overnight", "runner_log.jsonl"))
    if log:
        L.append(f"共 {len(log)} 条记录。关键事件：")
        L.append("")
        L.append("| 时刻 | 事件 | 说明 |")
        L.append("|---|---|---|")
        for e in log:
            ev = e.get("event", "")
            if ev in ("start", "finish", "stop", "skip", "health", "chain", "error", "pilot"):
                L.append(f"| {e.get('clock','')} | {ev} | {e.get('detail','')} |")
    else:
        L.append("（无日志）")
    L.append("")

    # ------------------------------------------------------------ 1 coherence
    L.append("## 1. 空间一致性（杂色治理）")
    L.append("")
    L.append("真实脸参考：相邻同色率约 **60%**，孤立像素约 **6.8**，连通色块约 **13.8**。")
    L.append("")
    rows = []
    for d in sorted(glob.glob(os.path.join(R, "coherence", "*"))):
        if not os.path.isdir(d):
            continue
        tag = os.path.basename(d)
        if tag.startswith("pilot_"):
            continue
        h = jlload(os.path.join(d, "metrics.jsonl"))
        s = jload(os.path.join(d, "summary.json")) or {}
        if not h:
            continue
        last = h[-1]
        best = max(h, key=lambda x: x.get("agree_mode", 0))
        rows.append({
            "tag": tag,
            "steps": last.get("step"),
            "lam": last.get("lam_now", (s.get("args") or {}).get("lam")),
            "acc": last.get("acc_mode"),
            "agree": last.get("agree_mode"),
            "iso": last.get("iso_mode"),
            "blobs": last.get("blobs_mode"),
            "ent": last.get("entropy"),
            "maxp": last.get("maxprob"),
            "best_agree": best.get("agree_mode"),
            "best_acc": best.get("acc_mode"),
        })
    if rows:
        L.append("| 运行 | 步数 | λ | 准确率 | 相邻同色率 | 孤立像素 | 色块 | 熵 |")
        L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for r in sorted(rows, key=lambda x: -(x["agree"] or 0)):
            L.append(f"| {r['tag']} | {r['steps']} | {fmt(r['lam'],2)} | {fmt(r['acc'])} | "
                     f"{fmt(r['agree'],pct=True)} | {fmt(r['iso'],2)} | {fmt(r['blobs'],1)} | "
                     f"{fmt(r['ent'],3)} |")
        L.append("")
        ok = [r for r in rows if (r["acc"] or 0) > 0.55]
        if ok:
            best_bal = max(ok, key=lambda x: (x["agree"] or 0))
            L.append(f"**在准确率仍高于 0.55 的运行中，最平滑的是 `{best_bal['tag']}`：** "
                     f"相邻同色率 {fmt(best_bal['agree'],pct=True)}、孤立 {fmt(best_bal['iso'],2)}、"
                     f"色块 {fmt(best_bal['blobs'],1)}，同时准确率 {fmt(best_bal['acc'])}")
            L.append("")
        most = max(rows, key=lambda x: (x["agree"] or 0))
        L.append(f"**绝对最平滑：** `{most['tag']}` 相邻同色率 {fmt(most['agree'],pct=True)}，"
                 f"准确率 {fmt(most['acc'])}")
    else:
        L.append("（无）")
    L.append("")

    # ------------------------------------------------------------ 2 lam sweeps
    L.append("## 2. λ 扫描")
    L.append("")
    any_sw = False
    for d in sorted(glob.glob(os.path.join(R, "coherence", "*"))):
        p = os.path.join(d, "lam_sweep.json")
        if not os.path.isfile(p):
            continue
        s = jload(p) or {}
        rs = s.get("rows") or s.get("results") or []
        if not rs:
            continue
        any_sw = True
        L.append(f"### {os.path.basename(d)}（惩罚 {s.get('penalty','?')}，解码 {s.get('decode','?')}）")
        L.append("")
        L.append("| λ | 同色率(众数) | 孤立(众数) | 准确率(众数) | 同色率(采样) |")
        L.append("|---:|---:|---:|---:|---:|")
        for r in rs:
            L.append(f"| {fmt(r.get('lam'),2)} | {fmt(r.get('agree_mode'),pct=True)} | "
                     f"{fmt(r.get('iso_mode'),2)} | {fmt(r.get('acc_mode'))} | "
                     f"{fmt(r.get('agree_sample'),pct=True)} |")
        L.append("")
    if not any_sw:
        L.append("（无）")
    L.append("")

    # ------------------------------------------------------------ 3 style clustering
    L.append("## 3. 风格聚类")
    L.append("")
    l4 = jload(os.path.join(R, "style", "loo_4class_results.json"))
    if l4:
        L.append(f"用你标注的 24 格、4 类（含 4 张未命名）做留一法；多数类基线 "
                 f"{fmt(l4.get('majority_baseline'))}，置换检验 p = **{l4.get('perm_p')}**")
        L.append("")
        L.append("| 特征空间 | 最近质心 | 3-NN | 逻辑回归 |")
        L.append("|---|---:|---:|---:|")
        for r in (l4.get("table") or []):
            L.append(f"| {r.get('space')} | {fmt(r.get('centroid'),3)} | "
                     f"{fmt(r.get('knn3'),3)} | {fmt(r.get('logreg'),3)} |")
    else:
        L.append("（无）")
    L.append("")

    # ------------------------------------------------------------ 4 style-specialised
    L.append("## 4. 按风格分别训练")
    L.append("")
    found = False
    for d in sorted(glob.glob(os.path.join(R, "styled", "*"))):
        if os.path.basename(d).startswith("pilot_") or os.path.basename(d) == "smoke":
            continue
        s = jload(os.path.join(d, "summary.json"))
        if not s:
            continue
        res = s.get("results")
        if not res:
            continue
        found = True
        args = s.get("args") or {}
        L.append(f"### {os.path.basename(d)}（k={args.get('k')}，{args.get('steps')} 步，"
                 f"λ {args.get('lam')}→{args.get('lam_final')}）")
        L.append("")
        L.append(f"训练集簇大小：{s.get('cluster_sizes_train')}")
        L.append("")
        g = res.get("global") or {}
        L.append(f"- 全局模型留出准确率 **{fmt(g.get('acc_all'))}**")
        pc = res.get("per_cluster") or {}
        if pc:
            L.append("")
            L.append("| 簇 | 训练样本 | 专家准确率 | 全局模型在同簇 | 增益 |")
            L.append("|---:|---:|---:|---:|---:|")
            gains = []
            for j, m in pc.items():
                gain = m.get("specialist_gain")
                if gain is not None:
                    gains.append(gain)
                L.append(f"| {j} | {m.get('n_train')} | {fmt(m.get('acc_all'))} | "
                         f"{fmt(m.get('global_on_this_cluster'))} | {fmt(gain,4)} |")
            if gains:
                L.append("")
                L.append(f"平均增益 **{sum(gains)/len(gains):+.4f}**"
                         f"（负值表示专门化没有帮助）")
        L.append("")
    if not found:
        L.append("（无）")
    L.append("")

    L.append("## 5. 原始数据位置")
    L.append("")
    L.append(f"- 全部结果：`{R}`")
    L.append(f"- 调度日志：`{os.path.join(R,'overnight','runner_log.jsonl')}`")
    L.append(f"- 实测耗时：`{os.path.join(R,'overnight','timings.json')}`")

except Exception:
    L.append("")
    L.append("## 报告生成时发生异常（已捕获，下面是堆栈）")
    L.append("")
    L.append("```")
    L.append(traceback.format_exc())
    L.append("```")

open(OUT, "w", encoding="utf-8").write("\n".join(L))
print(f"wrote {OUT}  ({len(L)} lines)")
print("MORNING_REPORT_DONE")
