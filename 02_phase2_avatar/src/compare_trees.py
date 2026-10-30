#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
"""
Full content comparison: D: source tree  vs  E: migrated tree.

Why: DELETE_OLD_D_CHECKLIST.md claims a per-file verification was done before migration,
but that script (a) still lived only in the checklist, (b) compared by NAME+SIZE, and (c)
could not be re-run because I wrongly believed the source was gone. It is not gone.
This does a real content comparison (SHA-256) and produces a machine-readable report.

What it can and cannot tell you:
  * MIGRATION FIDELITY: every D: file whose content exists somewhere in E: -> accounted for.
  * DIVERGENCE: D: files whose content is NOT in E: -> either never migrated, or E: was
    edited after migration (in which case the E: copy is the newer, intended one).
  * UNIQUENESS: content that exists only on D: is the part that would actually be lost.

Reports are written as UTF-8 JSON; stdout stays ASCII so a console codepage mismatch
cannot corrupt the numbers.

Usage: <python> compare_trees.py --src "<D: source>" --dst "<E: repo>" --out <dir>
"""
import argparse, hashlib, json, os, sys, time
from collections import defaultdict

AP = argparse.ArgumentParser()
AP.add_argument("--src", required=True)
AP.add_argument("--dst", required=True)
AP.add_argument("--out", required=True)
AP.add_argument("--maxhash", type=int, default=0, help="0 = hash everything")
A = AP.parse_args()

os.makedirs(A.out, exist_ok=True)


def walk(root):
    """Yield (relpath, abspath, size). Skips reparse points to avoid loops."""
    out = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        # prune junctions/symlinks
        keep = []
        for d in dirnames:
            p = os.path.join(dirpath, d)
            try:
                if os.path.islink(p):
                    continue
                attrs = os.stat(p, follow_symlinks=False)
                if getattr(attrs, "st_file_attributes", 0) & 0x400:  # REPARSE_POINT
                    continue
            except OSError:
                continue
            keep.append(d)
        dirnames[:] = keep
        for f in filenames:
            p = os.path.join(dirpath, f)
            try:
                st = os.stat(p, follow_symlinks=False)
                if getattr(st, "st_file_attributes", 0) & 0x400:
                    continue
                out.append((os.path.relpath(p, root), p, st.st_size))
            except OSError:
                pass
    return out


def sha256(p, limit=0):
    h = hashlib.sha256()
    n = 0
    with open(p, "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
            n += len(b)
            if limit and n >= limit:
                break
    return h.hexdigest()


def build(root, label):
    t0 = time.time()
    files = walk(root)
    total = sum(s for _, _, s in files)
    print(f"[{label}] {len(files)} files, {total/1e9:.3f} GB -> {root}")
    by_ext = defaultdict(lambda: [0, 0])
    for rel, p, s in files:
        e = os.path.splitext(rel)[1].lower() or "(none)"
        by_ext[e][0] += 1
        by_ext[e][1] += s
    print(f"[{label}] hashing ({time.time()-t0:.0f}s so far)...")
    man = []
    for i, (rel, p, s) in enumerate(files, 1):
        try:
            man.append({"rel": rel, "size": s, "sha256": sha256(p), "name": os.path.basename(rel)})
        except OSError as e:
            man.append({"rel": rel, "size": s, "sha256": None, "name": os.path.basename(rel), "error": str(e)})
        if i % 500 == 0:
            print(f"  [{label}] {i}/{len(files)}  ({time.time()-t0:.0f}s)", flush=True)
    print(f"[{label}] done in {time.time()-t0:.0f}s")
    return man, by_ext


src, src_ext = build(A.src, "SRC D:")
dst, dst_ext = build(A.dst, "DST E:")

src_h = defaultdict(list)
dst_h = defaultdict(list)
for m in src:
    if m["sha256"]:
        src_h[m["sha256"]].append(m["rel"])
for m in dst:
    if m["sha256"]:
        dst_h[m["sha256"]].append(m["rel"])

src_hashes = set(src_h)
dst_hashes = set(dst_h)

only_src = sorted(src_hashes - dst_hashes)
only_dst = sorted(dst_hashes - src_hashes)
shared = src_hashes & dst_hashes

# How many D: FILES are accounted for (a shared hash may cover several D: paths)
acc = sum(len(src_h[h]) for h in shared)
only_src_files = sum(len(src_h[h]) for h in only_src)
only_dst_files = sum(len(dst_h[h]) for h in only_dst)

print()
print("================ CONTENT COMPARISON ================")
print(f"SRC distinct contents : {len(src_hashes):>7}   ({len(src)} files)")
print(f"DST distinct contents : {len(dst_hashes):>7}   ({len(dst)} files)")
print(f"shared contents       : {len(shared):>7}")
print(f"only on SRC (D:)      : {len(only_src):>7} contents, {only_src_files} files")
print(f"only on DST (E:)      : {len(only_dst):>7} contents, {only_dst_files} files")
print(f"SRC files accounted   : {acc}/{len(src)}  ({100.0*acc/max(len(src),1):.2f}%)")

sizes_src = {m["rel"]: m["size"] for m in src}
sizes_dst = {m["rel"]: m["size"] for m in dst}
only_src_bytes = sum(sizes_src[src_h[h][0]] for h in only_src)
print(f"bytes only on SRC     : {only_src_bytes/1e6:.2f} MB  <- what would actually be LOST")

# duplicate detection inside D:
dup_d = [(h, v) for h, v in src_h.items() if len(v) > 1]
dup_e = [(h, v) for h, v in dst_h.items() if len(v) > 1]
bytes_dup_d = sum(sizes_src[v[0]] * (len(v) - 1) for h, v in dup_d)
bytes_dup_e = sum(sizes_dst[v[0]] * (len(v) - 1) for h, v in dup_e)
print(f"SRC internal duplicates: {len(dup_d)} contents, {sum(len(v)-1 for _,v in dup_d)} extra files, {bytes_dup_d/1e6:.1f} MB")
print(f"DST internal duplicates: {len(dup_e)} contents, {sum(len(v)-1 for _,v in dup_e)} extra files, {bytes_dup_e/1e6:.1f} MB")

rep = {
    "src_root": A.src,
    "dst_root": A.dst,
    "counts": {"src_files": len(src), "dst_files": len(dst),
               "src_contents": len(src_hashes), "dst_contents": len(dst_hashes),
               "shared_contents": len(shared),
               "only_src_contents": len(only_src), "only_src_files": only_src_files,
               "only_dst_contents": len(only_dst), "only_dst_files": only_dst_files,
               "src_files_accounted": acc, "only_src_bytes": only_src_bytes},
    "src_ext_histogram": {k: v for k, v in sorted(src_ext.items(), key=lambda kv: -kv[1][1])},
    "dst_ext_histogram": {k: v for k, v in sorted(dst_ext.items(), key=lambda kv: -kv[1][1])},
    "only_on_src": [{"sha256": h, "size": sizes_src[src_h[h][0]], "paths": src_h[h]} for h in only_src],
    "only_on_dst": [{"sha256": h, "size": sizes_dst[dst_h[h][0]], "paths": dst_h[h]} for h in only_dst],
    "src_duplicates": [{"sha256": h, "size": sizes_src[v[0]], "count": len(v), "paths": v} for h, v in sorted(dup_d, key=lambda kv: -len(kv[1]))],
    "dst_duplicates": [{"sha256": h, "size": sizes_dst[v[0]], "count": len(v), "paths": v} for h, v in sorted(dup_e, key=lambda kv: -len(kv[1]))],
    "src_manifest": src,
    "dst_manifest": dst,
}
p = os.path.join(A.out, "tree_compare.json")
json.dump(rep, open(p, "w", encoding="utf-8"), indent=1)
print(f"\nreport -> {p}")
print("COMPARE_DONE")
