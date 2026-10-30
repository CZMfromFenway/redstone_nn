#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
16 greyscale blocks from the vanilla jar -- fourth pass, after the user's review.

WHAT THE USER ASKED FOR, AND WHAT THIS DOES
1. "明度越极端，颜色对饱和度就越不敏感"
       effective_sat = sat * (0.25 + 0.75 * (1 - extremity))
   At grey 11 a blue tint cannot be seen, so black_concrete becomes usable; in the
   mid-tones a tint is still punished fully.

2. "多选一些候选整体对比"
   Instead of one answer per level, this renders a SHORTLIST of 5 candidates per level so
   the remaining choices can be compared directly.

3. "基岩和高炉排除：活塞推不动，颜色方差也大"
   Blocks that pistons refuse (flatly immovable, and in Java anything with a block entity)
   are excluded outright. Within-texture luminance std is penalised as well.

4. "石台阶某种意义上可以完整，门不行"
   Slabs are allowed (two stack into a full cube); doors, trapdoors, fences, panes are not.

5. The user explicitly approved ten blocks. Those get a COST BONUS so they win whenever
   they are competitive, and they are ticked in every output.

WHY THE THIRD PASS WENT WRONG (fixed here)
The third pass weighted texture uniformity so heavily that it preferred flat but tinted
blocks -- concrete powder and terracotta -- over the approved wool and andesite, and its
mean grey error rose from 3.3 to 4.9. The weights are now calibrated so that 1 grey unit of
error, 0.013 units of effective saturation, and 0.4 units of texture std are worth the same.
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

SAT_W = 75.0         # one unit of effective saturation is worth 75 grey units of error
UNIFORM_W = 2.6      # one unit of within-texture std is worth 2.6 grey units of error
APPROVED_BONUS = 18.0   # lower the effective cost of a user-approved block
MAX_SAT = 0.60
MAX_STD = 26.0
SHORT = 5

APPROVED = {
    "snow", "white_wool", "iron_block", "white_concrete", "diorite", "andesite",
    "black_wool", "coal_block", "cracked_stone_bricks", "chiseled_stone_bricks",
    "cracked_deepslate_bricks", "cracked_deepslate_tiles",
}
APPROVED_CN = {
    "snow": "雪块", "white_wool": "白色羊毛", "iron_block": "铁块",
    "white_concrete": "白色混凝土", "diorite": "闪长岩", "andesite": "安山岩",
    "black_wool": "黑色羊毛", "coal_block": "煤炭块",
    "cracked_stone_bricks": "开裂的石砖", "chiseled_stone_bricks": "雕纹石砖",
    "cracked_deepslate_bricks": "开裂的深板岩砖", "cracked_deepslate_tiles": "开裂的深板岩瓦",
}

IMMOVABLE = {
    "bedrock", "obsidian", "crying_obsidian", "respawn_anchor", "ancient_debris",
    "reinforced_deepslate", "end_portal", "end_portal_frame", "end_gateway",
    "nether_portal", "command_block", "chain_command_block", "repeating_command_block",
    "structure_block", "structure_void", "jigsaw", "barrier", "light", "moving_piston",
    "piston_head", "enchanting_table", "beacon", "spawner", "trial_spawner", "vault",
    "budding_amethyst",
}
BLOCK_ENTITY = {
    "furnace", "blast_furnace", "smoker", "chest", "trapped_chest", "barrel", "hopper",
    "dropper", "dispenser", "brewing_stand", "jukebox", "lectern", "campfire",
    "soul_campfire", "bee_nest", "beehive", "decorated_pot", "chiseled_bookshelf",
    "crafter", "conduit", "bell",
}
for c in ("white", "orange", "magenta", "light_blue", "yellow", "lime", "pink", "gray",
          "light_gray", "cyan", "purple", "blue", "brown", "green", "red", "black"):
    BLOCK_ENTITY.add(f"{c}_shulker_box")

NOT_FULL = ("_door", "_trapdoor", "_fence", "_fence_gate", "_wall", "_pane", "_carpet",
            "_button", "_pressure_plate", "_sign", "_torch", "_lantern", "_chain",
            "_bars", "_rail", "_bed", "_cake", "_candle", "_banner", "_skull", "_head",
            "_flower", "_sapling", "_coral", "_vine", "_lily_pad", "_sugar_cane",
            "_pot", "_cauldron", "_repeater", "_comparator", "_daylight_detector",
            "_tripwire", "_lever", "_ladder", "_scaffolding", "_campfire", "_composter",
            "_grindstone", "_loom", "_stonecutter", "_cartography_table",
            "_fletching_table", "_smithing_table", "_beacon", "_conduit", "_egg",
            "_frosted_ice", "_air", "_water", "_lava", "_fire", "_portal")

FACE_SUFFIX = ("_front_vertical", "_side_end", "_inner", "_outer", "_vertical",
               "_horizontal", "_double", "_side", "_top", "_bottom", "_front", "_back",
               "_end", "_up", "_down", "_east", "_west", "_north", "_south")


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


def excluded(bid):
    if bid in IMMOVABLE or bid in BLOCK_ENTITY:
        return True
    return any(s in bid for s in NOT_FULL)


by_block = {}
for path in z.namelist():
    if not (path.startswith("assets/minecraft/textures/block/") and path.endswith(".png")):
        continue
    stem = os.path.basename(path)[:-4]
    tgt = None
    if stem in block_ids:
        tgt = stem
    else:
        for suf in FACE_SUFFIX:
            if stem.endswith(suf) and stem[:-len(suf)] in block_ids:
                tgt = stem[:-len(suf)]
                break
    if tgt:
        by_block.setdefault(tgt, []).append(path)


def measure(paths):
    tot = np.zeros(3); wsum = 0.0; stds = []; opaque = 0; total = 0
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
        tot += (rgb * al).sum((0, 1)); wsum += w
        lum = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        m = al[..., 0] > 0.99
        if m.sum() > 1:
            stds.append(float(lum[m].std()))
        opaque += int(m.sum()); total += al.size
    if wsum <= 1e-6:
        return None
    mean = tot / wsum
    hi = float(mean.max())
    sat = float((hi - mean.min()) / hi) if hi > 1e-9 else 0.0
    grey = float(mean.mean())
    ext = abs(grey - 127.5) / 127.5
    return {"rgb": [round(float(v), 1) for v in mean], "grey": grey, "sat": sat,
            "eff_sat": sat * (0.25 + 0.75 * (1 - ext)),
            "std": float(np.mean(stds)) if stds else 0.0,
            "opaque": opaque / max(total, 1), "n_faces": len(paths)}


rows = []
for bid, paths in by_block.items():
    if excluded(bid):
        continue
    m = measure(paths)
    if m is None or m["opaque"] < 0.95 or m["sat"] > MAX_SAT or m["std"] > MAX_STD:
        continue
    m.update({"id": bid, "name": names.get(bid, bid), "paths": paths,
              "approved": bid in APPROVED})
    rows.append(m)

rows.sort(key=lambda r: r["grey"])
print(f"candidates after all filters: {len(rows)}")
print("  (excluded: piston-immovable, block-entity, non-full shapes, "
      f"sat>{MAX_SAT}, texture std>{MAX_STD})")


def cost(r, target):
    c = abs(r["grey"] - target) + SAT_W * r["eff_sat"] + UNIFORM_W * r["std"]
    if r["approved"]:
        c -= APPROVED_BONUS
    return c


print()
print("=" * 112)
print(f"SHORTLIST -- best {SHORT} candidates per level "
      f"(cost = |grey-target| + {SAT_W:.0f}*eff_sat + {UNIFORM_W}*std - {APPROVED_BONUS:.0f} if approved)")
print("=" * 112)
short = []
for i, target in enumerate(LEVELS):
    ranked = sorted(rows, key=lambda r: cost(r, target))[:SHORT]
    short.append(ranked)
    print(f"  level {i:2d}  target {target:5.1f}")
    for r in ranked:
        tag = "  ★认可" if r["approved"] else ""
        print(f"     {r['grey']:6.1f}  sat {r['sat']:5.3f} eff {r['eff_sat']:5.3f} "
              f"std {r['std']:5.1f}  cost {cost(r, target):6.1f}  {r['id']}{tag}")


def assign(cands, levels):
    B, L = len(cands), len(levels)
    INF = float("inf")
    c = [[cost(cands[b], levels[l]) for b in range(B)] for l in range(L)]
    dp = [[INF] * B for _ in range(L)]
    bk = [[-1] * B for _ in range(L)]
    for b in range(B):
        dp[0][b] = c[0][b]
    for l in range(1, L):
        bp, bb = INF, -1
        for b in range(B):
            if b - 1 >= 0 and dp[l - 1][b - 1] < bp:
                bp, bb = dp[l - 1][b - 1], b - 1
            if bb >= 0:
                dp[l][b] = bp + c[l][b]
                bk[l][b] = bb
    end = min(range(B), key=lambda b: dp[L - 1][b])
    seq = [0] * L
    b = end
    for l in range(L - 1, -1, -1):
        seq[l] = b
        b = bk[l][b]
    return [cands[i] for i in seq]


chosen = []
for i, (blk, target) in enumerate(zip(assign(rows, LEVELS), LEVELS)):
    chosen.append({"level": i, "target_grey": float(target), "id": blk["id"],
                   "name": blk["name"], "grey": blk["grey"], "sat": blk["sat"],
                   "eff_sat": blk["eff_sat"], "std": blk["std"], "rgb": blk["rgb"],
                   "approved": blk["approved"], "paths": blk["paths"]})

print()
print("=" * 112)
print("RECOMMENDED 16")
print("=" * 112)
print(f"  {'lvl':>3} {'target':>7} {'block id':30s} {'grey':>7s} {'err':>6s} "
      f"{'sat':>6s} {'std':>5s}  name")
print("  " + "-" * 104)
errs = []
for c in chosen:
    e = c["grey"] - c["target_grey"]
    errs.append(abs(e))
    tag = " ★" if c["approved"] else ""
    print(f"  {c['level']:3d} {c['target_grey']:7.1f} {c['id']:30s} {c['grey']:7.1f} "
          f"{e:+6.1f} {c['sat']:6.3f} {c['std']:5.1f}  {c['name']}{tag}")
print("  " + "-" * 104)
print(f"  mean |error| {np.mean(errs):.2f}   max {np.max(errs):.1f}")
print(f"  mean sat {np.mean([c['sat'] for c in chosen]):.4f}   "
      f"mean std {np.mean([c['std'] for c in chosen]):.1f}")
print(f"  approved blocks used: {sum(1 for c in chosen if c['approved'])}/16")

# ---------------------------------------------------------------- render shortlist
TILE, PAD, LAB = 58, 5, 30
W = 16 * (TILE + PAD) + PAD
H = 3 * (TILE + 20) + SHORT * (TILE + LAB) + 80
cv = Image.new("RGB", (W, H), (20, 22, 26))
dr = ImageDraw.Draw(cv)
f_t, f_l, f_s = font(17), font(11), font(9)
dr.text((PAD, 8), "16 级灰阶方块 · 每级 5 个候选（可整体对比择优）",
        font=f_t, fill=(232, 236, 242))

y = 34
for i, c in enumerate(chosen):
    im = Image.open(io.BytesIO(z.read(c["paths"][0]))).convert("RGB")
    cv.paste(im.resize((TILE, TILE), Image.NEAREST), (PAD + i * (TILE + PAD), y))
    dr.text((PAD + i * (TILE + PAD) + 1, y + TILE + 1), f"{c['grey']:.0f}",
            font=f_s, fill=(160, 172, 188))
    if c["approved"]:
        dr.text((PAD + i * (TILE + PAD) + 1, y + TILE + 10), "★", font=f_s,
                fill=(110, 200, 130))
dr.text((PAD, y + TILE + 19), "推荐 16（★＝你认可的）", font=f_l, fill=(150, 162, 178))

y += TILE + 24
for i, c in enumerate(chosen):
    cv.paste(Image.new("RGB", (TILE, TILE), tuple(int(v) for v in c["rgb"])),
             (PAD + i * (TILE + PAD), y))
dr.text((PAD, y + TILE + 2), "推荐 16 的平均色（点击式核对）", font=f_l, fill=(150, 162, 178))

y += TILE + 20
for i, t in enumerate(LEVELS):
    cv.paste(Image.new("RGB", (TILE, TILE), (int(t), int(t), int(t))),
             (PAD + i * (TILE + PAD), y))
    dr.text((PAD + i * (TILE + PAD) + 1, y + TILE + 1), f"{t:.0f}", font=f_s,
            fill=(160, 172, 188))
dr.text((PAD, y + TILE + 19), "目标 16 级", font=f_l, fill=(150, 162, 178))

y += TILE + 24
for rank in range(SHORT):
    for i in range(16):
        r = short[i][rank]
        im = Image.open(io.BytesIO(z.read(r["paths"][0]))).convert("RGB")
        cv.paste(im.resize((TILE, TILE), Image.NEAREST), (PAD + i * (TILE + PAD), y))
        dr.text((PAD + i * (TILE + PAD) + 1, y + TILE + 1),
                f"{r['grey']:.0f}", font=f_s, fill=(150, 162, 178))
        nm = r["id"][:10] + ("★" if r["approved"] else "")
        dr.text((PAD + i * (TILE + PAD) + 1, y + TILE + 10), nm, font=f_s,
                fill=(120, 132, 148) if not r["approved"] else (110, 200, 130))
        dr.text((PAD + i * (TILE + PAD) + 1, y + TILE + 19),
                f"s{r['sat']:.2f}", font=f_s, fill=(100, 110, 126))
    dr.text((PAD, y + TILE + 21), f"候选 {rank+1}", font=f_l, fill=(150, 162, 178))
    y += TILE + LAB

p = os.path.join(OUT, "PALETTE16_V4.png")
cv.save(p)
print(f"\nwrote {p}  {cv.size}")

json.dump({"jar": os.path.basename(JAR), "weights": {"sat": SAT_W, "std": UNIFORM_W,
                                                     "approved_bonus": APPROVED_BONUS},
           "levels": [float(v) for v in LEVELS], "recommended": chosen,
           "shortlist": [{"level": i, "target": float(LEVELS[i]),
                          "candidates": [{k: r[k] for k in
                                          ("id", "name", "grey", "sat", "eff_sat", "std",
                                           "rgb", "approved", "paths")} for r in rk]}
                         for i, rk in enumerate(short)],
           "mean_abs_error": float(np.mean(errs)),
           "mean_sat": float(np.mean([c["sat"] for c in chosen])),
           "mean_std": float(np.mean([c["std"] for c in chosen])),
           "approved_used": sum(1 for c in chosen if c["approved"]),
           "filters": {"piston_immovable": "excluded", "block_entity": "excluded",
                       "slabs": "allowed (two stack to a full cube)",
                       "doors_trapdoors_fences_panes": "excluded",
                       "max_sat": MAX_SAT, "max_texture_std": MAX_STD}},
          open(os.path.join(OUT, "palette16_v4.json"), "w"), indent=2, ensure_ascii=False)
print(f"wrote {os.path.join(OUT, 'palette16_v4.json')}")
print("PALETTE_V4_DONE")
