#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Find 16 REAL Minecraft blocks whose own textures are neutral greys spanning black to white,
so the palette no longer depends on recolouring concrete.

IMPROVEMENT OVER THE FIRST ATTEMPT
The first version measured one texture per file, so any block whose art is split across
directional faces was skipped entirely -- quartz_block, bone_block and basalt all vanished
because they only ship `_side` / `_top` / `_bottom` files. That is wrong: the block exists
and a builder can place it.

This version groups every texture by the block it belongs to (stripping face suffixes when
the stripped name is a real block id) and averages ALL of that block's faces, alpha-weighted,
so a block's colour reflects the whole block rather than one arbitrary face.

Then the 16 blocks are chosen by an OPTIMAL ORDERED assignment: because both the target
levels and the candidate greys are ordered, the best 16 distinct blocks for the 16 levels
must preserve that order, so a small dynamic program finds the exact optimum instead of
being greedy level by level.
"""
import io
import json
import os
import sys
import zipfile
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
JAR = os.path.normpath(os.path.join(ROOT, "..", "99_mc_texture",
                                    "1.21.11-Fabric 0.19.5.jar"))
OUT = os.path.join(ROOT, "results", "palette16")
os.makedirs(OUT, exist_ok=True)
LEVELS = np.linspace(0, 255, 16)
SAT_WEIGHT = 90.0
MAX_SAT = 0.25

# Shapes that are NOT a full cube. A door or a slab cannot be used as one pixel: it does
# not fill the cell, so its visible colour depends on what is behind it and on the viewing
# angle. The first run picked pale_oak_door and smooth_stone_slab, which looked fine in a
# swatch and would be wrong in a build.
NOT_FULL_CUBE = (
    "_door", "_trapdoor", "_slab", "_stairs", "_fence", "_fence_gate", "_wall",
    "_pane", "_carpet", "_button", "_pressure_plate", "_sign", "_torch",
    "_lantern", "_chain", "_bars", "_rail", "_bed", "_cake", "_candle",
    "_banner", "_skull", "_head", "_flower", "_sapling", "_coral", "_vine",
    "_lily_pad", "_sugar_cane", "_pot", "_cauldron", "_hopper", "_repeater",
    "_comparator", "_daylight_detector", "_tripwire", "_lever", "_redstone_wire",
    "_piston_head", "_moving_piston", "_structure_void", "_light", "_air",
    "_ladder", "_scaffolding", "_bell", "_campfire", "_anvil_top", "_composter",
    "_grindstone", "_lectern", "_loom", "_smithing_table", "_stonecutter",
    "_cartography_table", "_fletching_table", "_barrel_top", "_beacon", "_conduit",
    "_sniffer_egg", "_turtle_egg", "_dragon_egg", "_frosted_ice", "_soul_fire",
)


def is_full_cube(bid):
    return not any(bid.endswith(s2) or s2.strip("_") in bid.split("_")[-1:] for s2 in NOT_FULL_CUBE)


FACE_SUFFIX = (
    "_front_vertical", "_side_end", "_inner", "_outer", "_vertical", "_horizontal",
    "_side", "_top", "_bottom", "_front", "_back", "_end", "_up", "_down",
    "_east", "_west", "_north", "_south", "_tip", "_base", "_corner", "_edge",
    "_overlay", "_stage0", "_stage1", "_stage2", "_stage3", "_stage4",
)


def font(sz):
    for fn in ("msyh.ttc", "simhei.ttf", "simsun.ttc"):
        p = os.path.join("C:\\Windows\\Fonts", fn)
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, sz)
            except Exception:
                pass
    return ImageFont.load_default()


z = zipfile.ZipFile(JAR)
lang = json.loads(z.read("assets/minecraft/lang/en_us.json").decode("utf-8"))
names = {k[len("block.minecraft."):]: v for k, v in lang.items()
         if k.startswith("block.minecraft.")}
block_ids = set(names)

# ---------------------------------------------------------------- group textures by block
by_block = {}
for path in z.namelist():
    if not (path.startswith("assets/minecraft/textures/block/") and path.endswith(".png")):
        continue
    stem = os.path.basename(path)[:-4]
    target = None
    if stem in block_ids:
        target = stem
    else:
        for suf in FACE_SUFFIX:
            if stem.endswith(suf) and stem[:-len(suf)] in block_ids:
                target = stem[:-len(suf)]
                break
    if target:
        by_block.setdefault(target, []).append(path)

print(f"blocks with at least one flat texture: {len(by_block)}")


def block_colour(paths):
    """Alpha-weighted mean over every face texture of the block."""
    tot = np.zeros(3)
    wsum = 0.0
    opaque_px = 0
    total_px = 0
    for p in paths:
        try:
            im = Image.open(io.BytesIO(z.read(p))).convert("RGBA")
        except Exception:
            continue
        if im.width != im.height:
            continue
        a = np.asarray(im, dtype=np.float64)
        rgb, al = a[..., :3], a[..., 3:4] / 255.0
        w = float(al.sum())
        if w <= 1e-6:
            continue
        tot += (rgb * al).sum((0, 1))
        wsum += w
        opaque_px += int((al > 0.99).sum())
        total_px += al.size
    if wsum <= 1e-6:
        return None
    mean = tot / wsum
    high = float(mean.max())
    sat = float((high - mean.min()) / high) if high > 1e-9 else 0.0
    return {"rgb": [round(float(v), 1) for v in mean],
            "grey": float(mean.mean()), "sat": sat,
            "lum": float(0.299 * mean[0] + 0.587 * mean[1] + 0.114 * mean[2]),
            "opaque": opaque_px / max(total_px, 1), "n_faces": len(paths)}


rows = []
for bid, paths in by_block.items():
    m = block_colour(paths)
    if m is None:
        continue
    if m["opaque"] < 0.95:
        continue                       # a see-through block is not a flat surface
    m.update({"id": bid, "name": names.get(bid, bid), "paths": paths})
    rows.append(m)
print(f"blocks with a solid, measurable appearance: {len(rows)}")

rows.sort(key=lambda r: r["sat"])
print()
print("  18 most neutral blocks:")
print(f"  {'block':30s} {'grey':>6s} {'sat':>6s} {'rgb':>24s} {'faces':>5s}")
for r in rows[:18]:
    print(f"  {r['id']:30s} {r['grey']:6.1f} {r['sat']:6.3f} {str(r['rgb']):>24s} "
          f"{r['n_faces']:5d}")

rows = [r for r in rows if is_full_cube(r["id"])]
print(f"  full-cube blocks only: {len(rows)}")
pool = sorted([r for r in rows if r["sat"] < MAX_SAT], key=lambda r: r["grey"])
print()
print(f"  candidates passing sat<{MAX_SAT}: {len(pool)}")
print("  the darkest 14 candidates:")
for r in pool[:14]:
    print(f"    {r['grey']:6.1f}  sat {r['sat']:.3f}  {r['id']}")
print("  the brightest 10 candidates:")
for r in pool[-10:]:
    print(f"    {r['grey']:6.1f}  sat {r['sat']:.3f}  {r['id']}")


# ---------------------------------------------------------------- optimal ordered assignment
def assign(cands, levels, sat_w):
    B, L = len(cands), len(levels)
    INF = float("inf")
    cost = [[abs(cands[b]["grey"] - levels[l]) + sat_w * cands[b]["sat"]
             for b in range(B)] for l in range(L)]
    dp = [[INF] * B for _ in range(L)]
    back = [[-1] * B for _ in range(L)]
    for b in range(B):
        dp[0][b] = cost[0][b]
    for l in range(1, L):
        best_prev, best_b = INF, -1
        for b in range(B):
            if b - 1 >= 0 and dp[l - 1][b - 1] < best_prev:
                best_prev, best_b = dp[l - 1][b - 1], b - 1
            if best_b >= 0:
                dp[l][b] = best_prev + cost[l][b]
                back[l][b] = best_b
    end = min(range(B), key=lambda b: dp[L - 1][b])
    if dp[L - 1][end] == INF:
        raise SystemExit("infeasible")
    seq = [0] * L
    b = end
    for l in range(L - 1, -1, -1):
        seq[l] = b
        b = back[l][b]
    return [cands[i] for i in seq]


chosen = []
for i, (blk, target) in enumerate(zip(assign(pool, LEVELS, SAT_WEIGHT), LEVELS)):
    chosen.append({"level": i, "target_grey": float(target), "id": blk["id"],
                   "name": blk["name"], "grey": blk["grey"], "sat": blk["sat"],
                   "rgb": blk["rgb"], "lum": blk["lum"], "paths": blk["paths"]})

print()
print("=" * 106)
print("THE 16 BLOCKS -- optimal ordered assignment of unmodified vanilla textures")
print("=" * 106)
print(f"  {'lvl':>3} {'target':>7} {'block id':30s} {'grey':>7s} {'err':>6s} "
      f"{'sat':>6s}  name")
print("  " + "-" * 100)
errs = []
for c in chosen:
    e = c["grey"] - c["target_grey"]
    errs.append(abs(e))
    print(f"  {c['level']:3d} {c['target_grey']:7.1f} {c['id']:30s} {c['grey']:7.1f} "
          f"{e:+6.1f} {c['sat']:6.3f}  {c['name']}")
print("  " + "-" * 100)
print(f"  mean |error| {np.mean(errs):.2f}   max {np.max(errs):.1f}")
print(f"  mean saturation {np.mean([c['sat'] for c in chosen]):.4f}   "
      f"max {np.max([c['sat'] for c in chosen]):.3f}")
print(f"  16 distinct blocks: {len({c['id'] for c in chosen}) == 16}")
print()
print(f"  NOTE the deepest level cannot reach 0: the darkest neutral solid block is "
      f"{pool[0]['id']} at {pool[0]['grey']:.1f}.")
print(f"  The achievable neutral range is about {pool[0]['grey']:.0f}..{pool[-1]['grey']:.0f}, "
      f"not 0..255. If exact spacing matters,")
print("  re-quantise the faces onto THESE 16 greys instead of the uniform ones -- then the")
print("  palette error is zero by construction and only the spacing is uneven.")

# ---------------------------------------------------------------- swatch sheet
TILE, PAD = 64, 6
W = 16 * (TILE + PAD) + PAD
H = 3 * (TILE + 24) + 74
cv = Image.new("RGB", (W, H), (20, 22, 26))
dr = ImageDraw.Draw(cv)
f_t, f_l, f_s = font(17), font(11), font(10)
dr.text((PAD, 8), "用 16 个原版方块自身的材质做灰阶（未修改任何贴图）",
        font=f_t, fill=(232, 236, 242))
y0 = 38
for i, c in enumerate(chosen):
    im = Image.open(io.BytesIO(z.read(c["paths"][0]))).convert("RGB")
    im = im.resize((TILE, TILE), Image.NEAREST)
    cv.paste(im, (PAD + i * (TILE + PAD), y0))
    dr.text((PAD + i * (TILE + PAD) + 1, y0 + TILE + 2), c["id"][:12], font=f_s,
            fill=(160, 172, 188))
y1 = y0 + TILE + 26
for i, c in enumerate(chosen):
    cv.paste(Image.new("RGB", (TILE, TILE), tuple(int(v) for v in c["rgb"])),
             (PAD + i * (TILE + PAD), y1))
    dr.text((PAD + i * (TILE + PAD) + 1, y1 + TILE + 2), f"{c['grey']:.0f}", font=f_s,
            fill=(160, 172, 188))
y2 = y1 + TILE + 26
for i, t in enumerate(LEVELS):
    cv.paste(Image.new("RGB", (TILE, TILE), (int(t), int(t), int(t))),
             (PAD + i * (TILE + PAD), y2))
    dr.text((PAD + i * (TILE + PAD) + 1, y2 + TILE + 2), f"{t:.0f}", font=f_s,
            fill=(160, 172, 188))
dr.text((PAD, y2 + TILE + 22),
        "第1行＝方块原贴图　第2行＝该方块所有分面的平均色　第3行＝目标的 16 度明度分级",
        font=f_l, fill=(150, 162, 178))
dr.text((PAD, y2 + TILE + 38),
        f"平均误差 {np.mean(errs):.2f} 灰阶 · 平均饱和度 "
        f"{np.mean([c['sat'] for c in chosen]):.4f} · 16 个方块互不相同",
        font=f_l, fill=(150, 162, 178))
p = os.path.join(OUT, "PALETTE16.png")
cv.save(p)
print(f"\nwrote {p}  {cv.size}")

json.dump({"jar": os.path.basename(JAR), "sat_weight": SAT_WEIGHT, "max_sat": MAX_SAT,
           "levels": [float(v) for v in LEVELS], "blocks": chosen,
           "mean_abs_error": float(np.mean(errs)),
           "max_abs_error": float(np.max(errs)),
           "mean_sat": float(np.mean([c["sat"] for c in chosen])),
           "max_sat_used": float(np.max([c["sat"] for c in chosen])),
           "achievable_range": [float(pool[0]["grey"]), float(pool[-1]["grey"])],
           "method": "optimal ordered assignment (DP) over vanilla solid blocks with "
                     "saturation < 0.25, minimising |grey-target| + 90*saturation; block "
                     "colour is the alpha-weighted mean of all its face textures"},
          open(os.path.join(OUT, "palette16.json"), "w"), indent=2, ensure_ascii=False)
print(f"wrote {os.path.join(OUT, 'palette16.json')}")
print("PALETTE16_DONE")
