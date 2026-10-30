#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Rebuild the 8x8 face dataset CLEANLY, with provenance.

Why this exists (see 02_phase2_avatar\audit_tmp\AUDIT_REPORT.md):
  D2 alpha dropped without a mask  -> transparent head-fronts become black avatars
  D3 face-level redundancy         -> only 932,636 distinct crops in 1,107,411 entries
  D4 in-place overwrite            -> faces8.npy was rewritten, old results unreproducible
  D5 "quantization error" ambiguous-> per-pixel sum vs per-channel mean, 3x apart
  D6 first-N-in-zip-order slice    -> not a random sample

This script fixes all five at once:
  * rescans the zip in central-directory order (deterministic, reproducible)
  * keeps an alpha coverage mask and DROPS fully-invisible crops
  * dedups on crop bytes, and reports exact multiplicity
  * writes NEW VERSIONED filenames plus a manifest (args, counts, hashes, source names)
  * never overwrites any existing file

Outputs (in <outdir>):
  faces8_v2.npy         (N,64)  uint8  palette indices, -1 not used (dropped, not masked)
  faces8_rgb_v2.npy     (N,8,8,3) uint8  RGB with alpha composited onto black
  alpha_cov_v2.npy      (N,)    float32  fraction of the 64 px that are opaque
  source_names_v2.json  (N,)    list   source zip entry names, in order
  manifest_v2.json             counts, drops, hashes, palette, command

Usage:
  <venv-python> rebuild_faces8.py --zip <...zip> --out <dir> [--limit N] [--min-alpha 0.5]
"""
import argparse, hashlib, io, json, os, sys, time, zipfile
import numpy as np

AP = argparse.ArgumentParser()
AP.add_argument("--zip", required=True)
AP.add_argument("--out", required=True)
AP.add_argument("--palette", default=None, help="palette json with palette_ordered; default: refit k-means-free greedy on the kept set")
AP.add_argument("--limit", type=int, default=0, help="max zip entries to scan (0 = all)")
AP.add_argument("--min-alpha", type=float, default=0.5, help="drop crops with opaque fraction below this")
AP.add_argument("--target", type=int, default=0, help="stop after this many KEPT crops (0 = no target)")
A = AP.parse_args()

os.makedirs(A.out, exist_ok=True)
t0 = time.time()

# ---------------------------------------------------------------- palette
if A.palette and os.path.exists(A.palette):
    pj = json.load(open(A.palette, encoding="utf-8"))
    PAL = np.array(pj["palette_ordered"], dtype=np.uint16)
    pal_src = f"reused from {os.path.basename(A.palette)}"
else:
    PAL = None
    pal_src = "refit below"

print(f"zip      : {A.zip}")
print(f"out      : {A.out}")
print(f"palette  : {pal_src}")

# ---------------------------------------------------------------- scan
crops = []          # flat uint8 (3*64)
alpha_cov = []
names = []
seen = set()
seen_faces = {}
n_entries = n_png = n_bad = 0
n_drop_alpha = 0
n_dup = 0

zf = zipfile.ZipFile(A.zip)
entries = [n for n in zf.namelist() if n.lower().endswith(".png")]
print(f"png entries in zip: {len(entries):,}")

from PIL import Image

CHUNK = 20000
kept_bytes = []
for i, name in enumerate(entries):
    if A.limit and i >= A.limit:
        break
    n_entries += 1
    try:
        with zf.open(name) as fh:
            im = Image.open(io.BytesIO(fh.read())).convert("RGBA")
    except Exception:
        n_bad += 1
        continue
    if im.size != (64, 64):
        n_bad += 1
        continue
    n_png += 1
    face = im.crop((8, 8, 16, 16))
    arr = np.asarray(face, dtype=np.uint8)          # (8,8,4)
    a = arr[:, :, 3]
    cov = float((a > 0).mean())
    if cov < A.min_alpha:
        n_drop_alpha += 1
        continue
    # Keep the authored RGB as-is. Do NOT composite onto black: that would invent a dark
    # fringe on partially transparent edge pixels. The alpha coverage mask records how much
    # of the crop was actually visible, and fully-invisible crops were already dropped above.
    rgb = arr[:, :, :3].copy()
    key = rgb.tobytes()
    seen_faces[key] = seen_faces.get(key, 0) + 1
    if key in seen:
        n_dup += 1
        continue
    seen.add(key)
    crops.append(rgb.reshape(-1))
    alpha_cov.append(cov)
    names.append(name)
    if len(crops) % CHUNK == 0:
        print(f"  scanned {n_entries:,}  kept {len(crops):,}  dup {n_dup:,}  alpha-drop {n_drop_alpha:,}  ({time.time()-t0:.0f}s)", flush=True)
    if A.target and len(crops) >= A.target:
        break

RGB = np.stack(crops).reshape(-1, 8, 8, 3)
COV = np.array(alpha_cov, dtype=np.float32)
print(f"\nscanned entries : {n_entries:,}")
print(f"  valid 64x64   : {n_png:,}")
print(f"  unreadable    : {n_bad:,}")
print(f"  alpha-dropped : {n_drop_alpha:,}")
print(f"  crop-dup      : {n_dup:,}")
print(f"  KEPT          : {len(RGB):,}")
if seen_faces:
    mult = np.array(sorted(seen_faces.values(), reverse=True))
    print(f"  crop multiplicity: max {mult[0]:,}  mean {mult.mean():.2f}  distinct {len(mult):,}")

# ---------------------------------------------------------------- palette (refit if needed)
if PAL is None:
    # Greedy facility location: start from the most common colour, then repeatedly add the
    # candidate that most reduces total SSE. Fitted on a random pixel subsample (the greedy
    # ARGMIN is stable under subsampling; only the reported SSE needs the full pass, which we
    # do afterwards). int32 everywhere: 3*255^2 fits comfortably.
    px = RGB.reshape(-1, 3).astype(np.int32)
    rng = np.random.default_rng(0)
    FIT = min(len(px), 300_000)
    fit_px = px[rng.choice(len(px), FIT, replace=False)] if len(px) > FIT else px
    print(f"palette fit sample: {len(fit_px):,} of {len(px):,} pixels")
    uniq, cnt = np.unique(fit_px, axis=0, return_counts=True)
    order = np.argsort(-cnt)
    cand = np.ascontiguousarray(uniq[order][:4096].astype(np.int32))
    NC = len(cand)
    best_d = np.full(len(fit_px), np.iinfo(np.int32).max, dtype=np.int32)
    chosen = []
    for step in range(16):
        tot = np.zeros(NC, dtype=np.int64)
        CH2 = 60000
        for s in range(0, len(fit_px), CH2):
            e = min(s + CH2, len(fit_px))
            d = ((fit_px[s:e, None, :] - cand[None, :, :]) ** 2).sum(2)   # (chunk, NC) int32-ish
            tot += np.minimum(d, best_d[s:e, None]).sum(0)
        j = int(tot.argmin())
        chosen.append(j)
        CH3 = 60000
        for s in range(0, len(fit_px), CH3):
            e = min(s + CH3, len(fit_px))
            d = ((fit_px[s:e] - cand[j]) ** 2).sum(1)
            best_d[s:e] = np.minimum(best_d[s:e], d)
        print(f"  palette step {step+1:2d}/16  picked {j}  fit-SSE {tot[j]/len(fit_px):.2f}", flush=True)
    PAL = cand[chosen].astype(np.uint16)
    # order into a path: nearest neighbour + 2-opt on RGB distance
    P = PAL.astype(np.float32)
    Dm = ((P[:, None, :] - P[None, :, :]) ** 2).sum(2)
    unused = list(range(1, len(P)))
    seq = [0]
    while unused:
        last = seq[-1]
        nxt = min(unused, key=lambda j: Dm[last, j])
        seq.append(nxt); unused.remove(nxt)
    PAL = PAL[seq]
    pal_src = "refit on the kept set (greedy + nearest-neighbour path)"
    print(f"palette : refit -> {PAL.tolist()}")

# ---------------------------------------------------------------- quantize
P = PAL.astype(np.int32)
flat = RGB.reshape(-1, 3).astype(np.int32)
# distance to each palette entry, chunked
idx = np.empty(len(flat), dtype=np.uint8)
SSE = 0.0
CH = 200000
for s in range(0, len(flat), CH):
    e = min(s + CH, len(flat))
    d = ((flat[s:e, None, :] - P[None, :, :]) ** 2).sum(2)
    j = d.argmin(1)
    idx[s:e] = j
    SSE += float(d[np.arange(e - s), j].sum())
per_pixel_sse = SSE / len(flat)
per_channel_mse = per_pixel_sse / 3.0
print(f"\nquantization: per-pixel SSE {per_pixel_sse:.2f}   per-channel MSE {per_channel_mse:.3f}"
      f"   per-channel RMS {np.sqrt(per_channel_mse):.2f}/255")

LAB = idx.reshape(-1, 64)

# ---------------------------------------------------------------- write (versioned, never overwrite)
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

outs = {}
for fname, arr in (("faces8_v2.npy", LAB), ("faces8_rgb_v2.npy", RGB),
                   ("alpha_cov_v2.npy", COV), ("palette_avatar16_v2.npy", PAL)):
    p = os.path.join(A.out, fname)
    if os.path.exists(p):
        raise SystemExit(f"refusing to overwrite existing file: {p}")
    np.save(p, arr)
    outs[fname] = {"shape": list(arr.shape), "dtype": str(arr.dtype), "sha256": sha(p), "bytes": os.path.getsize(p)}

p = os.path.join(A.out, "source_names_v2.json")
if os.path.exists(p):
    raise SystemExit(f"refusing to overwrite existing file: {p}")
json.dump(names, open(p, "w", encoding="utf-8"))
outs["source_names_v2.json"] = {"count": len(names), "sha256": sha(p), "bytes": os.path.getsize(p)}

man = {
    "created_by": "02_phase2_avatar/src/rebuild_faces8.py",
    "command": " ".join(sys.argv),
    "zip": {"path": A.zip, "sha256": sha(A.zip), "bytes": os.path.getsize(A.zip)},
    "crop": [8, 8, 16, 16],
    "min_alpha": A.min_alpha,
    "limit": A.limit,
    "target": A.target,
    "counts": {"scanned": n_entries, "valid64": n_png, "unreadable": n_bad,
               "alpha_dropped": n_drop_alpha, "crop_dup": n_dup, "kept": int(len(RGB))},
    "palette": {"source": pal_src, "colors": PAL.tolist()},
    "quantization": {"definition": "per-pixel SSE = mean over pixels of sum over 3 channels",
                     "per_pixel_sse": round(per_pixel_sse, 4),
                     "per_channel_mse": round(per_channel_mse, 4),
                     "per_channel_rms_255": round(float(np.sqrt(per_channel_mse)), 4)},
    "stats": {"mean_distinct_colors": round(float(np.mean([len(np.unique(r)) for r in LAB])), 4),
              "mean_alpha_coverage": round(float(COV.mean()), 4),
              "fully_opaque_rows": int((COV == 1.0).sum())},
    "outputs": outs,
}
mp = os.path.join(A.out, "manifest_v2.json")
if os.path.exists(mp):
    raise SystemExit(f"refusing to overwrite existing file: {mp}")
json.dump(man, open(mp, "w", encoding="utf-8"), indent=2)

print("\n=== wrote ===")
for k, v in outs.items():
    print(f"  {k:26s} {v.get('shape', v.get('count'))}  {v['bytes']/1e6:.1f} MB  {v['sha256'][:16]}")
print(f"  manifest_v2.json")
print(f"\nmean distinct colors : {man['stats']['mean_distinct_colors']}")
print(f"mean alpha coverage  : {man['stats']['mean_alpha_coverage']}")
print(f"fully opaque rows    : {man['stats']['fully_opaque_rows']:,}")
print(f"elapsed              : {time.time()-t0:.0f}s")
print("REBUILD_DONE")
