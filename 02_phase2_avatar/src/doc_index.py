import datetime
import io
import json
import os
import re

ROOT = r"E:\redstone_nn\02_phase2_avatar"
OUT = os.path.join(ROOT, "results", "DOC_INDEX.md")

paths = []
for dirpath, dirnames, filenames in os.walk(ROOT):
    dirnames[:] = [d for d in dirnames
                   if d not in (".venv", "node_modules", "__pycache__", ".piptmp")]
    for fn in filenames:
        if fn.lower().endswith(".md"):
            paths.append(os.path.join(dirpath, fn))
paths.sort(key=lambda p: os.path.getmtime(p))

rows = []
for p in paths:
    rel = os.path.relpath(p, ROOT)
    ts = datetime.datetime.fromtimestamp(os.path.getmtime(p)).strftime("%Y-%m-%d %H:%M")
    try:
        text = io.open(p, encoding="utf-8").read()
    except Exception:
        text = io.open(p, encoding="gbk", errors="replace").read()
    lines = [l.rstrip() for l in text.splitlines()]
    title = ""
    for l in lines:
        if l.startswith("#"):
            title = l.lstrip("# ").strip()
            break
    body = ""
    for l in lines:
        s = l.strip()
        if not s or s.startswith(("#", ">", "|", "-", "*", "```", "---", "===")):
            continue
        body = s
        break
    heads = [l.lstrip("# ").strip() for l in lines if l.startswith("## ")]
    rows.append({"ts": ts, "rel": rel, "kb": os.path.getsize(p) / 1e3,
                 "title": title, "head": body[:190], "heads": heads[:10],
                 "chars": len(text)})

buf = ["# 二期工程文档索引", "",
       f"共 {len(rows)} 份 Markdown，按修改时间排序。时间取自文件 mtime。", ""]
day = None
for r in rows:
    d = r["ts"][:10]
    if d != day:
        buf.append(f"\n## {d}\n")
        day = d
    buf.append(f"### {r['ts'][11:]}  `{r['rel']}`  ({r['kb']:.1f} KB, {r['chars']:,} 字符)")
    buf.append(f"- **标题**：{r['title']}")
    buf.append(f"- **开头**：{r['head']}")
    if r["heads"]:
        buf.append("- **小标题**：" + " / ".join(r["heads"]))
    buf.append("")

io.open(OUT, "w", encoding="utf-8").write("\n".join(buf))
print("wrote", OUT, len(rows), "entries")
