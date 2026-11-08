#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
生成「可玩的生成器」单文件 HTML：把目前所有开关都做成可调的。

开关：
    上色 / 灰度半成品（看颜色头读到的输入）
    先验校正（revive bias）开/关
    调色板：原色板 / 作者方案（多候选） / 精选 64 组合
    WOOL→QUARTZ（画面没有 QUARTZ 时）开/关
    归一化强度 α（0=不归一化，1=拉满）
    筛选开/关 + 力度（留 N 选 1）
    批量张数

产物：04_phase4_colour_avatar/webapp/playground.html
用法：python build_playground.py
"""
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
P4 = os.path.dirname(HERE)
REPO = os.path.dirname(P4)
WEB = os.path.join(P4, "webapp")


def r6(v):
    return round(float(v), 6)


def main():
    g = json.load(open(os.path.join(P4, "results", "twostage", "gray_pca4.json"),
                       encoding="utf-8"))
    gray = {"layers": [{"W": [[r6(v) for v in row] for row in L["W"]],
                        "b": [r6(v) for v in L["b"]]} for L in g["layers"]]}

    base = json.load(open(os.path.join(P4, "results", "twostage", "pal16.json"),
                          encoding="utf-8"))
    rev = json.load(open(os.path.join(P4, "results", "twostage", "pal16_revive.json"),
                         encoding="utf-8"))
    body = [{"W": [[r6(v) for v in row] for row in L["W"]],
             "b": [r6(v) for v in L["b"]]} for L in base["body"]]
    colour = {"body": body,
              "headW": [[r6(v) for v in row] for row in base["head"]["W"]],
              "bOrig": [r6(v) for v in base["head"]["b"]],
              "bRev": [r6(v) for v in rev["head"]["b"]]}

    def variant(path, mode):
        j = json.load(open(path, encoding="utf-8"))
        out = {"mode": mode,
               "palette": [[r6(v) for v in c] for c in j["palette"]]}
        if j.get("palette_sets"):
            out["palette_sets"] = [[[r6(v) for v in c] for c in s]
                                   for s in j["palette_sets"]]
            out["slot_p"] = [float(p) for p in j.get("slot_p", [1.0] * 16)]
            out["slot_w"] = [[float(x) for x in w] for w in
                             j.get("slot_w", [[1.0] * len(s) for s in j["palette_sets"]])]
        if j.get("configs"):
            out["configs"] = [{"score": c["score"],
                               "cfg": {k: [r6(v) for v in val] for k, val in c["cfg"].items()}}
                              for c in j["configs"]]
        return out

    variants = {
        "base": variant(os.path.join(P4, "results", "twostage", "pal16.json"), "fixed"),
        "user": variant(os.path.join(P4, "results", "twostage", "pal16_user.json"), "sets"),
        "curated": variant(os.path.join(P4, "results", "twostage", "pal16_curated64.json"),
                           "configs"),
    }
    sc = json.load(open(os.path.join(P4, "results", "labels", "scorers_round5.json"),
                        encoding="utf-8"))["families"]["two_stage"]
    dp = json.load(open(os.path.join(P4, "results", "fit", "deploy_palette.json"),
                        encoding="utf-8"))["K16_w128_k8"]

    data = {"gray": gray, "colour": colour, "variants": variants,
            "scorer": {"weights": [r6(v) for v in sc["weights"]], "cv_auc": sc["cv_auc"]},
            "names": dp["names"],
            "cost": {"gray": 2304, "head": 86016, "scorer": 15}}

    tpl = open(os.path.join(HERE, "playground_template.html"), encoding="utf-8").read()
    out = tpl.replace("/*__DATA__*/null",
                      json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    p = os.path.join(WEB, "playground.html")
    open(p, "w", encoding="utf-8").write(out)
    print(f"wrote {os.path.relpath(p, REPO)}  ({os.path.getsize(p)/1e6:.2f} MB)")
    for k, v in variants.items():
        print(f"  {k}: mode={v['mode']} palette_sets={len(v.get('palette_sets', []))} "
              f"configs={len(v.get('configs', []))}")


if __name__ == "__main__":
    main()
