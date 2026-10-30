#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
Build a single-file version of the sandbox that works by double-clicking.

WHY THIS INSTEAD OF AN .exe
The page does nothing but integer arithmetic in JavaScript -- the models are plain weight arrays
and the inference is a few nested loops. An .exe would add an Electron or PyInstaller runtime
(hundreds of MB, plus an unsigned-binary warning on every machine) for no functionality.

The only reason the current page needs a server is that browsers block fetch() on file:// URLs.
Inlining the weights and shimming fetch removes that need entirely: one .html, no install, works
offline, opens in any browser, and can be emailed or dropped in a shared folder.

The shim is deliberately installed BEFORE the page script, so the page code is byte-identical to
the tested version -- nothing about the working logic changes.
"""
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WEB = os.path.join(ROOT, "webapp")
OUT = os.path.join(WEB, "sandbox_standalone.html")

fam = json.load(open(os.path.join(WEB, "grayscale_family.json"), encoding="utf-8"))
data = {"grayscale_family.json": fam}
total = 0
for m in fam["models"]:
    p = os.path.join(WEB, m["file"])
    if not os.path.exists(p):
        print(f"  skip missing {m['file']}")
        continue
    j = json.load(open(p, encoding="utf-8"))
    # keep only what the page reads: gen.layers, plus meta for nothing
    data[m["file"]] = {"gen": {"layers": j["gen"]["layers"]}}
    total += os.path.getsize(p)
for f in ("scorer_accept.json", "scorer_anime.json", "merge_mc11.json",
          "scorer_quad.json", "styles_gauss.json"):
    data[f] = json.load(open(os.path.join(WEB, f), encoding="utf-8"))

src = open(os.path.join(WEB, "sandbox.html"), encoding="utf-8").read()

blob = json.dumps(data, separators=(",", ":"))
shim = ("<script>\n"
        "/* Inlined weights so the page works from file:// with no server.\n"
        "   fetch() is shimmed to serve them by filename; the page code below is unchanged. */\n"
        "const __DATA = " + blob + ";\n"
        "window.fetch = async function(u){\n"
        "  const k = String(u).split('?')[0].split('/').pop();\n"
        "  if (Object.prototype.hasOwnProperty.call(__DATA, k))\n"
        "    return { ok: true, status: 200, json: async function(){ return __DATA[k]; } };\n"
        "  return { ok: false, status: 404, json: async function(){ return {}; } };\n"
        "};\n"
        "</script>\n")

# install the shim immediately before the page's own script
marker = "<script>\n'use strict';"
if marker not in src:
    raise SystemExit("could not find the page script marker")
src = src.replace(marker, shim + marker, 1)

sniff = ('<!-- Saved as a single file: models are inlined, no server needed. '
         'Just open it. -->\n')
if "<!DOCTYPE html>" in src:
    src = src.replace("<!DOCTYPE html>", "<!DOCTYPE html>\n" + sniff, 1)

open(OUT, "w", encoding="utf-8").write(src)
print(f"wrote {OUT}")
print(f"  inlined {len([k for k in data if k.endswith('.json')])} json files from {total/1e6:.2f} MB of weights")
print(f"  output size {os.path.getsize(OUT)/1e6:.2f} MB")
print("  open it directly: no server, no install, works offline")
print("STANDALONE_DONE")
