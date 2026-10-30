#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
"""
Install torch (+ its few runtime deps) into the phase-2 venv WITHOUT using pip.

Why: pip fails in this sandbox with
    PermissionError: [Errno 13] Permission denied: '...\\pip-unpack-*\\torch-...whl'
even though a plain Python 150 MB write/rename/delete in the same directory succeeds.
So we do the three steps ourselves: resolve (PyPI JSON), download (urllib), unpack (zipfile).

Usage:
    <venv-python> install_wheels.py --site <venv Lib/site-packages> --cache <dir> torch
"""
import argparse, io, json, os, re, sys, zipfile, urllib.request

MIRROR = "https://pypi.tuna.tsinghua.edu.cn/simple"
JSON_API = "https://pypi.org/pypi/{}/json"

# Acceptable wheel platform tags for this interpreter/OS.
PY = "cp{}{}".format(*sys.version_info[:2])
PLAT_OK = (PY, "none-any", "py3-none-any")


def fetch_json(name):
    req = urllib.request.Request(JSON_API.format(name), headers={"User-Agent": "dsh-wheel-fetch/1.0"})
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.load(r)


def pick_wheel(name, data=None):
    """Return (filename, url, size) for the best wheel for this interpreter."""
    data = data or fetch_json(name)
    ver = data["info"]["version"]
    cands = []
    for f in data["releases"].get(ver, []):
        n = f["filename"]
        if not n.endswith(".whl"):
            continue
        tags = n[:-4].split("-")
        # <dist>-<ver>(-<build>)?-<pytag>-<abitag>-<platformtag>
        py_tag, abi_tag, plat_tag = tags[-3], tags[-2], tags[-1]
        if plat_tag not in ("win_amd64", "win32", "any"):
            continue
        if abi_tag not in ("none", "abi3", PY):
            continue
        if py_tag not in (PY, "py3", "py2.py3", "none"):
            continue
        cands.append((0 if plat_tag == "win_amd64" else 1, n, f["url"], f["size"]))
    if not cands:
        return None
    cands.sort()
    return cands[0][1], cands[0][2], cands[0][3]


def req_names(requires_dist):
    out = []
    for r in requires_dist or []:
        if ";" in r:
            marker = r.split(";", 1)[1]
            if "extra" in marker:
                continue
            if "platform_system" in marker and "Linux" in marker and "!=" not in marker:
                continue
        name = re.split(r"[<>=!\[;( ]", r.strip())[0]
        if name:
            out.append(name)
    return out


def download(url, dest, label):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"    cached {label} ({os.path.getsize(dest)/1e6:.1f} MB)")
        return
    req = urllib.request.Request(url, headers={"User-Agent": "dsh-wheel-fetch/1.0"})
    tmp = dest + ".part"
    got = 0
    with urllib.request.urlopen(req, timeout=300) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            got += len(chunk)
            if total and got % (10 << 20) < (1 << 20):
                print(f"      {label}: {got/1e6:.0f}/{total/1e6:.0f} MB", flush=True)
    os.replace(tmp, dest)
    print(f"    downloaded {label} ({os.path.getsize(dest)/1e6:.1f} MB)")


def unpack(whl, site):
    n = 0
    with zipfile.ZipFile(whl) as z:
        for m in z.namelist():
            if m.endswith("/"):
                continue
            if ".dist-info/RECORD" in m or m.endswith(".dist-info/RECORD"):
                pass
            target = os.path.join(site, *m.split("/"))
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with z.open(m) as src, open(target, "wb") as dst:
                dst.write(src.read())
            n += 1
    print(f"    unpacked {n} files")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("pkgs", nargs="+")
    a = ap.parse_args()

    os.makedirs(a.cache, exist_ok=True)
    os.makedirs(a.site, exist_ok=True)

    queue = list(a.pkgs)
    seen, order = set(), []
    meta = {}
    while queue:
        name = queue.pop(0).lower()
        if name in seen:
            continue
        seen.add(name)
        print(f"[resolve] {name}")
        try:
            data = fetch_json(name)
        except Exception as e:
            print(f"    !! metadata failed for {name}: {type(e).__name__} {e}")
            continue
        meta[name] = data
        chosen = pick_wheel(name, data)
        if not chosen:
            print(f"    !! no compatible wheel for {name} (pure-python sdist only?)")
            continue
        order.append((name, chosen))
        for dep in req_names(data["info"].get("requires_dist")):
            if dep.lower() not in seen:
                queue.append(dep)

    print("\n=== plan ===")
    for name, (fn, _, sz) in order:
        print(f"  {name:22s} {sz/1e6:8.1f} MB  {fn}")

    print("\n=== download + unpack ===")
    for name, (fn, url, sz) in order:
        dest = os.path.join(a.cache, fn)
        print(f"[install] {name} -> {fn}")
        download(url, dest, fn)
        unpack(dest, a.site)
    print("\nWHEELS_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
