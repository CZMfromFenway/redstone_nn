#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
四期第一轮盲测的本地服务（静态文件 + 收标签）。

用法：
    python label_server.py            # 默认 http://127.0.0.1:8771/
    python label_server.py --port 9000

页面：/04_phase4_colour_avatar/tools/label.html
接口：
    POST /label   {"round":"1","model":"k8","pca":[...],"grade":2}
                  → 追加到 04_phase4_colour_avatar/results/labels/colour_round1.jsonl
    GET  /stats   → 当前轮的各臂计数与接受率
    GET  /selftest.json → 前向自检参考值（页面加载时自动对账）

只写 `04_phase4_colour_avatar/results/labels/`，不碰任何原始数据。
"""
import argparse
import json
import os
import sys
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
P4 = os.path.dirname(HERE)
REPO = os.path.dirname(P4)
LABELDIR = os.path.join(P4, "results", "labels")
os.makedirs(LABELDIR, exist_ok=True)


class Handler(SimpleHTTPRequestHandler):
    root = REPO

    def __init__(self, *a, **kw):
        super().__init__(*a, directory=Handler.root, **kw)

    def log_message(self, fmt, *args):
        pass

    def _send(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path.split("?")[0] != "/label":
            self._send({"ok": False, "error": "unknown endpoint"}, 404)
            return
        n = int(self.headers.get("Content-Length", "0"))
        try:
            rec = json.loads(self.rfile.read(n).decode("utf-8"))
        except Exception as e:
            self._send({"ok": False, "error": str(e)}, 400)
            return
        rnd = str(rec.get("round", "1"))
        rec["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        rec.setdefault("scheme", "style")
        path = os.path.join(LABELDIR, f"colour_round{rnd}.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._send({"ok": True, "file": os.path.relpath(path, REPO), "n": self._count(rnd)})

    def _count(self, rnd):
        path = os.path.join(LABELDIR, f"colour_round{rnd}.jsonl")
        if not os.path.exists(path):
            return 0
        with open(path, encoding="utf-8") as f:
            return sum(1 for _ in f)

    def do_GET(self):
        p = self.path.split("?")[0]
        if p == "/stats":
            out = {}
            for fn in sorted(os.listdir(LABELDIR)):
                if not fn.startswith("colour_round") or not fn.endswith(".jsonl"):
                    continue
                rows = [json.loads(l) for l in open(os.path.join(LABELDIR, fn), encoding="utf-8")
                        if l.strip()]
                by = {}
                for r in rows:
                    e = by.setdefault(r.get("model", "?"), {"n": 0, "acc": 0})
                    e["n"] += 1
                    e["acc"] += int(r.get("grade", 0) >= 2)
                for k, e in by.items():
                    e["yield"] = round(e["acc"] / max(1, e["n"]), 3)
                out[fn] = {"n": len(rows), "arms": by}
            self._send(out)
            return
        return super().do_GET()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8771)
    ap.add_argument("--root", default=REPO)
    A = ap.parse_args()
    Handler.root = os.path.abspath(A.root)
    srv = ThreadingHTTPServer(("127.0.0.1", A.port), Handler)
    print(f"serving {Handler.root}")
    print(f"盲测页： http://127.0.0.1:{A.port}/04_phase4_colour_avatar/tools/label.html")
    print(f"统计：   http://127.0.0.1:{A.port}/stats")
    print("Ctrl+C 结束")
    srv.serve_forever()


if __name__ == "__main__":
    main()
