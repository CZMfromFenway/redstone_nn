#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
把调色板编辑器导出的 `palette_scheme.json` 变成可直接用的模型文件。

方案 v2 允许「一个槽位有多个候选、各有权重、并带被换概率 p」。于是有两种产物：

  单候选方案（每槽 ≤1 个候选且 p=1）→ 普通 `palette`，任何工具都能直接用
  多候选方案                        → 额外写 `palette_sets`（每槽的候选集合）
                                      与 `slot_p` / `slot_w`，由页面逐张抽样

用法：
    python apply_palette_scheme.py my_scheme.json [--tag my1]
"""
import argparse
import json
import os
import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
P4 = os.path.dirname(HERE)
REPO = os.path.dirname(P4)


def lum(c):
    return 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scheme")
    ap.add_argument("--tag", default=None)
    A = ap.parse_args()

    with open(A.scheme, encoding="utf-8-sig") as f:      # 容忍 Windows 记事本的 BOM
        sc = json.load(f)
    if sc.get("kind") != "palette_scheme":
        print("警告：缺 kind=palette_scheme，仍尝试解析")

    base = json.load(open(os.path.join(P4, "results", "twostage", "pal16.json"),
                          encoding="utf-8"))
    usage = json.load(open(os.path.join(P4, "results", "pca4map", "slot_usage.json"),
                           encoding="utf-8"))["deploy"]["s"]
    old = np.array(base["palette"], dtype=float)

    slots = sc.get("slots") or []
    multi = any(len(s.get("alts", [])) > 1 for s in slots)
    p_any = any(float(s.get("p", 1)) < 0.999 for s in slots)

    sets, probs, weights = [], [], []
    for i in range(16):
        ent = next((s for s in slots if int(s["slot"]) == i), None)
        if not ent or not ent.get("alts"):
            sets.append([[round(float(v), 3) for v in old[i]]])
            probs.append(0.0)
            weights.append([1.0])
            continue
        alts = ent["alts"]
        sets.append([[round(float(v), 3) for v in a["rgb"]] for a in alts])
        probs.append(float(ent.get("p", 1.0)))
        weights.append([max(0.0, float(a.get("weight", 1))) for a in alts])

    touched = [i for i in range(16)
               if len(sets[i]) > 1 or np.linalg.norm(np.array(sets[i][0]) - old[i]) > 0.6]
    dl = [abs(lum(np.array(sets[i][0])) - lum(old[i])) for i in touched]
    share = sum(float(usage.get(str(i), 0)) * probs[i] for i in touched)
    distinct = len({tuple(np.round(c, 2)) for s in sets for c in s})

    print(f"方案：{sc.get('created', '?')}"
          f"（{'多候选' if multi else '单候选'}{'、带概率' if p_any else ''}）")
    print(f"  有候选的槽 {len(touched)}、候选总数 {sum(len(s) for s in sets)}")
    if dl:
        print(f"  平均 |ΔL| = {np.mean(dl):.2f} 级（最大 {np.max(dl):.2f}）")
    print(f"  预计被改动的像素 = {share*100:.1f}%（Σ 用量×p）")
    print(f"  全部候选去重后 {distinct} 个不同颜色")

    tag = A.tag or "scheme"
    out = os.path.join(P4, "results", "twostage", f"pal16_{tag}.json")
    mdl = json.loads(json.dumps(base))
    if multi or p_any:
        # 多候选：palette 保持**原色板**，"不替换"时回退到原色（这才是 p<1 的语义）；
        # 候选全放 palette_sets，由页面逐张抽样。
        mdl["palette_sets"] = sets
        mdl["slot_p"] = [round(p, 3) for p in probs]
        mdl["slot_w"] = weights
        mdl["meta"] = dict(base.get("meta", {}), palette_scheme=tag, scheme_type="multi",
                           note="多候选调色板：每槽按 slot_p 决定是否替换，"
                                "替换时按 slot_w 在 palette_sets 里抽")
        print("\n这是**多候选**方案：写入了 palette_sets / slot_p / slot_w，逐张抽样出图。")
        print("  支持它的工具：webapp/sandbox.html、webapp/palette_editor.html 的预览")
        print("  palette 字段保持原色板（不替换时回退到原色）；旧工具会退化成原色板")
    else:
        # 单候选：直接替换 palette，任何工具都能用
        mdl["palette"] = [[round(float(v), 3) for v in c] for c in
                          np.array([s[0] for s in sets], dtype=float)]
        mdl["meta"] = dict(base.get("meta", {}), palette_scheme=tag,
                           note="单候选调色板方案")
    json.dump(mdl, open(out, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"\nwrote {os.path.relpath(out, REPO)}")
    print("下一步：把这个臂加进 tools/make_manifest.py 的某一轮，"
          "或把 scheme 文件给我，我直接接进盲测。")


if __name__ == "__main__":
    main()
