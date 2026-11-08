#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
把"目前最好的模型"打成一个离线单文件 HTML（双击即用），参考二期 sandbox_standalone.html 的做法。

嵌进去的东西：
    灰度生成器  results/twostage/gray_pca4.json     4→64→32，2,304 乘法
    颜色头      results/twostage/pal16.json         32→128→128→32×16，86,016 乘法
    调色板      results/twostage/pal16.json 内置     16 槽吸附到 61 HIGH（15 个不同颜色）
    打分器      results/labels/scorers_round5.json  4 旋钮的 15 项二次式

产物：04_phase4_colour_avatar/webapp/sandbox.html（约 2 MB，无外部依赖）
用法：python build_sandbox.py
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
os.makedirs(WEB, exist_ok=True)


def r6(x):
    return round(float(x), 6)


def compact_gray(path):
    j = json.load(open(path, encoding="utf-8"))
    return {"layers": [{"W": [[r6(v) for v in row] for row in L["W"]],
                        "b": [r6(v) for v in L["b"]]} for L in j["layers"]]}


def compact_palette(path):
    j = json.load(open(path, encoding="utf-8"))
    return {"body": [{"W": [[r6(v) for v in row] for row in L["W"]],
                      "b": [r6(v) for v in L["b"]]} for L in j["body"]],
            "head": {"W": [[r6(v) for v in row] for row in j["head"]["W"]],
                     "b": [r6(v) for v in j["head"]["b"]]},
            "palette": [[r6(v) for v in c] for c in j["palette"]],
            "meta": {k: j["meta"][k] for k in ("K", "hid", "mults", "steps")
                     if k in j["meta"]}}


def main():
    gray = compact_gray(os.path.join(P4, "results", "twostage", "gray_pca4.json"))
    pal = compact_palette(os.path.join(P4, "results", "twostage", "pal16.json"))
    sc = json.load(open(os.path.join(P4, "results", "labels", "scorers_round5.json"),
                        encoding="utf-8"))["families"]["two_stage"]
    dp = json.load(open(os.path.join(P4, "results", "fit", "deploy_palette.json"),
                        encoding="utf-8"))["K16_w128_k8"]
    data = {"gray": gray, "pal": pal,
            "names": dp["names"],
            "scorer": {"kind": sc["kind"], "weights": [r6(v) for v in sc["weights"]],
                       "cv_auc": sc["cv_auc"], "n": sc["n"]},
            "palette_meta": {"distinct": dp["distinct_colours"],
                             "snap_err_mean": dp["snap_err_mean"]},
            "cost": {"gray": 2304, "colour_head": 86016, "scorer_per_cand": 15}}

    tpl = open(os.path.join(HERE, "sandbox_template.html"), encoding="utf-8").read()
    out = tpl.replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    p = os.path.join(WEB, "sandbox.html")
    open(p, "w", encoding="utf-8").write(out)
    print(f"wrote {os.path.relpath(p, REPO)}  ({os.path.getsize(p)/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
