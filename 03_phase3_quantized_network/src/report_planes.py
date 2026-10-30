# -*- coding: utf-8 -*-
"""把一份 planes_*.json 里 X=0 的检查点整理成游戏内可对照的表。

坐标：灰环 K (1,21)、白环 W0 (xw,23)、白总线 (xw,21/15/9)、头部 (xw,3)，y = 2ch+3。

用法：python src/report_planes.py [planes.json]
"""
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def main():
    p = (sys.argv[1] if len(sys.argv) > 1
         else os.path.join(ROOT, "results", "planes_L2.json"))
    d = json.load(open(p, encoding="utf-8"))
    planes = d["planes"]
    W, L, b = d["W"], d["L"], d["b"]
    n = (W - 6) // 2
    print(f"{os.path.basename(p)}  宽 {W}（n={n} b={b}）  通道 {len(planes)}")

    rows = []
    for pl in planes:
        cells = {(c["x"], c["z"]): c for c in pl["cells"]}
        xw = pl.get("xw")
        if xw is None:
            xw = 2 * n + 3
        K = cells.get((1, 21), {})
        K = K.get("st")
        W0 = cells.get((xw, 23), {})
        W0 = W0.get("st")
        bus = [cells.get((xw, z), {}).get("st") for z in (21, 15, 9)]
        headc = cells.get((xw, 3), {})
        rows.append(dict(ch=pl["ch"], K=K, W0=W0, bus=bus, head=headc.get("st"),
                         N=pl.get("N"), xw=xw))

    print(f"头部格子：x={rows[0]['xw']}  z=3   （白环 W0 在 x={rows[0]['xw']}, z=23；灰环 K 在 x=1, z=21）")
    print()
    hdr = ("ch | y    |   灰环K |   白环W0 |  x=%d,z21  x=%d,z15  x=%d,z9 | 头部 x=%d,z3 | 公式N"
           % (rows[0]["xw"], rows[0]["xw"], rows[0]["xw"], rows[0]["xw"]))
    print(hdr)
    print("-" * len(hdr))
    bad = 0
    for r in rows:
        y = 2 * r["ch"] + 3
        ok = (r["head"] == r["N"])
        bad += 0 if ok else 1
        print("%2d | %3d  | %7s | %8s | %8s  %8s  %6s | %11s | %s%s"
              % (r["ch"], y, r["K"], r["W0"], r["bus"][0], r["bus"][1], r["bus"][2],
                 r["head"], r["N"], "" if ok else "  ✗"))
    print()
    if bad:
        print(f"✗ {bad} 个通道头部 ≠ 公式 N")
    else:
        print(f"✓ {len(rows)}/{len(rows)} 通道头部 = 公式 N（X 全 0）")


if __name__ == "__main__":
    sys.exit(main())
