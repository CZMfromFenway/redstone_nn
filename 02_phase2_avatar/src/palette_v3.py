#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
16 greyscale blocks from the vanilla jar -- third pass, with the user's four corrections.

WHAT CHANGED AND WHY
1. SATURATION MATTERS LESS AT THE EXTREMES.
   The user's point: near black and near white, a hue tint is barely visible, so judging a
   dark block by raw saturation over-penalises it. Saturation is now discounted by how
   extreme the block's lightness is:
       effective_sat = sat * (0.25 + 0.75 * (1 - extremity))
   where extremity = |grey-127.5|/127.5. In the mid-tones a tint is still punished fully.
   This is what makes black_concrete (grey 11.4, sat 0.454, blue-tinted) usable at the
   darkest level -- at grey 11 the tint cannot be perceived.

2. TEXTURE UNIFORMITY IS NOW A CRITERION.
   The user rejected bedrock and blast_furnace partly because their colour VARIANCE is
   high. A block used as one pixel should be flat, so the within-texture standard deviation
   of luminance is measured and penalised.

3. SLABS ARE ALLOWED, DOORS ARE NOT.
   Two slabs stack into a full cube, so a slab can represent a full pixel. Doors,
   trapdoors, fences, panes and the like never fill the cell and stay excluded.

4. NON-PISTON-MOVABLE BLOCKS ARE EXCLUDED.
   Bedrock cannot be pushed at all. Blocks with BLOCK ENTITIES (furnaces, chests, hoppers,
   shulker boxes, ...) cannot be pushed in Java Edition either, which rules out
   blast_furnace and black_shulker_box. Both constraints come from the user and are
   correct for Java Edition.

OUTPUT
Rather than a single answer, this prints and renders a SHORTLIST: the best few candidates
per level, so the remaining choices can be compared side by side.
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

SAT_W = 110.0        # penalty per unit of effective saturation (in grey units)
UNIFORM_W = 45.0     # penalty per unit of within-texture luminance std
MAX_SAT = 0.60       # hard cap, looser now that extremes are discounted
MAX_STD = 26.0       # a texture noisier than this cannot read as one flat pixel

# The user explicitly accepted these; they are marked in the output.
APPROVED = {
    "snow": "雪块", "white_wool": "白色羊毛", "iron_block": "铁块",
    "white_concrete": "白色混凝土", "diorite": "闪长岩", "andesite": "安山岩",
    "black_wool": "黑色羊毛", "coal_block": "煤炭块",
    "cracked_stone_bricks": "开裂的石砖", "cracked_deepslate_bricks": "开裂的深板岩砖",
    "cracked_deepslate_tiles": "开裂的深板岩瓦", "chiseled_stone_bricks": "雕纹石砖",
}

# Blocks that cannot be moved by a piston in Java Edition. Two groups:
#   (a) flatly immovable
#   (b) blocks with a block entity -- pistons refuse these in Java
IMMOVABLE = {
    "bedrock", "obsidian", "crying_obsidian", "respawn_anchor", "ancient_debris",
    "reinforced_deepslate", "end_portal", "end_portal_frame", "end_gateway",
    "nether_portal", "command_block", "chain_command_block", "repeating_command_block",
    "structure_block", "structure_void", "jigsaw", "barrier", "light", "moving_piston",
    "piston_head", "enchanting_table", "beacon", "spawner", "trial_spawner", "vault",
    "reinforced_deepslate", "budding_amethyst",
}
BLOCK_ENTITY = {
    "furnace", "blast_furnace", "smoker", "chest", "trapped_chest", "barrel", "hopper",
    "dropper", "dispenser", "brewing_stand", "jukebox", "lectern", "campfire",
    "soul_campfire", "bee_nest", "beehive", "decorated_pot", "chiseled_bookshelf",
    "crafter", "conduit", "bell", "enchanting_table", "beacon", "spawner",
}
for c in ("white", "orange", "magenta", "light_blue", "yellow", "lime", "pink", "gray",
          "light_gray", "cyan", "purple", "blue", "brown", "green", "red", "black"):
    BLOCK_ENTITY.add(f"{c}_shulker_box")

# Shapes that never fill a cell. Slabs are deliberately absent: two slabs stack.
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
        return "immovable"
    if any(s in bid for s in NOT_FULL):
        return "not a full cell"
    return None


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
    tot = np.zeros(3)
    wsum = 0.0
    stds = []
    opaque = 0
    total = 0
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
        lum = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        m = al[..., 0] > 0.99
        if m.sum() > 1:
            stds.append(float(lum[m].std()))
        opaque += int(m.sum())
        total += al.size
    if wsum <= 1e-6:
        return None
    mean = tot / wsum
    hi = float(mean.max())
    sat = float((hi - mean.min()) / hi) if hi > 1e-9 else 0.0
    grey = float(mean.mean())
    extrem = abs(grey - 127.5) / 127.5
    eff = sat * (0.25 + 0.75 * (1 - extrem))
    return {"rgb": [round(float(v), 1) for v in mean], "grey": grey, "sat": sat,
            "eff_sat": eff, "std": float(np.mean(stds)) if stds else 0.0,
            "opaque": opaque / max(total, 1), "n_faces": len(paths)}


rows = []
for bid, paths in by_block.items():
    why = excluded(bid)
    if why:
        continue
    m = measure(paths)
    if m is None or m["opaque"] < 0.95:
        continue
    if m["sat"] > MAX_SAT or m["std"] > MAX_STD:
        continue
    m.update({"id": bid, "name": names.get(bid, bid), "paths": paths,
              "approved": bid in APPROVED})
    rows.append(m)

print(f"candidates after all filters: {len(rows)}")
print(f"  (excluded: piston-immovable, block-entity, non-full shapes, "
      f"sat>{MAX_SAT}, texture std>{MAX_STD})")

rows.sort(key=lambda r: r["grey"])
print()
print("  legend: grey  sat  eff_sat  std  id")
for r in rows[:6]:
    print(f"    {r['grey']:6.1f} {r['sat']:5.3f} {r['eff_sat']:6.3f} {r['std']:5.1f}  {r['id']}")
print("    ...")
for r in rows[-6:]:
    print(f"    {r['grey']:6.1f} {r['sat']:5.3f} {r['eff_sat']:6.3f} {r['std']:5.1f}  {r['id']}")

# ---------------------------------------------------------------- shortlist per level
def cost(r, target):
    return abs(r["grey"] - target) + SAT_W * r["eff_sat"] + UNIFORM_W * (r["std"] / 26.0)


SHORT = 4
print()
print("=" * 112)
print("SHORTLIST PER LEVEL (best %d candidates, cost = |grey-target| + %.0f*eff_sat + %.0f*std)"
      % (SHORT, SAT_W, UNIFORM_W))
print("=" * 112)
short = []
for i, target in enumerate(LEVELS):
    ranked = sorted(rows, key=lambda r: cost(r, target))[:SHORT]
    short.append(ranked)
    print(f"  level {i:2d}  target {target:5.1f}")
    for r in ranked:
        tag = "  [认可]" if r["approved"] else ""
        print(f"     {r['grey']:6.1f}  sat {r['sat']:5.3f} eff {r['eff_sat']:5.3f} "
              f"std {r['std']:5.1f}  cost {cost(r, target):6.1f}  {r['id']}{tag}")

# a global ordered assignment over the shortlist, so the 16 are distinct and ordered
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


uniq = {}
for r in rows:
    uniq.setdefault(r["id"], r)
pool = sorted(uniq.values(), key=lambda r: r["grey"])
chosen = []
for i, (blk, target) in enumerate(zip(assign(pool, LEVELS), LEVELS)):
    chosen.append({"level": i, "target_grey": float(target), "id": blk["id"],
                   "name": blk["name"], "grey": blk["grey"], "sat": blk["sat"],
                   "eff_sat": blk["eff_sat"], "std": blk["std"], "rgb": blk["rgb"],
                   "approved": blk["approved"], "paths": blk["paths"]})

print()
print("=" * 112)
print("FINAL 16")
print("=" * 112)
print(f"  {'lvl':>3} {'target':>7} {'block id':30s} {'grey':>7s} {'err':>6s} "
      f"{'sat':>6s} {'std':>5s}  name")
print("  " + "-" * 104)
errs = []
for c in chosen:
    e = c["grey"] - c["target_grey"]
    errs.append(abs(e))
    tag = " [认可]" if c["approved"] else ""
    print(f"  {c['level']:3d} {c['target_grey']:7.1f} {c['id']:30s} {c['grey']:7.1f} "
          f"{e:+6.1f} {c['sat']:6.3f} {c['std']:5.1f}  {c['name']}{tag}")
print("  " + "-" * 104)
print(f"  mean |error| {np.mean(errs):.2f}   max {np.max(errs):.1f}")
print(f"  mean sat {np.mean([c['sat'] for c in chosen]):.4f}   "
      f"mean texture std {np.mean([c['std'] for c in chosen]):.1f}")
print(f"  user-approved blocks used: {sum(1 for c in chosen if c['approved'])}/16")

# ---------------------------------------------------------------- render
TILE, PAD = 62, 6
W = 16 * (TILE + PAD) + PAD
H = 3 * (TILE + 24) + 80
cv = Image.new("RGB", (W, H), (20, 22, 26))
dr = ImageDraw.Draw(cv)
f_t, f_l, f_s = font(17), font(11), font(10)
dr.text((PAD, 8), "16 个原版灰阶方块（第三版：极端明度放宽饱和度 + 增加纹理均匀度）",
        font=f_t, fill=(232, 236, 242))
y0 = 38
for i, c in enumerate(chosen):
    im = Image.open(io.BytesIO(z.read(c["paths"][0]))).convert("RGB")
    cv.paste(im.resize((TILE, TILE), Image.NEAREST), (PAD + i * (TILE + PAD), y0))
    dr.text((PAD + i * (TILE + PAD) + 1, y0 + TILE + 2), c["id"][:12], font=f_s,
            fill=(160, 172, 188))
    if c["approved"]:
        dr.text((PAD + i * (TILE + PAD) + 1, y0 + TILE + 13), "认可", font=f_s,
                fill=(110, 200, 130))
y1 = y0 + TILE + 26
for i, c in enumerate(chosen):
    cv.paste(Image.new("RGB", (TILE, TILE), tuple(int(v) for v in c["rgb"])),
             (PAD + i * (TILE + PAD), y1))
    dr.text((PAD + i * (TILE + PAD) + 1, y1 + TILE + 2), f"{c['grey']:.0f}", font=f_s,
            fill=(160, 172, 188))
y2 = y1 + TILE + 22
for i, t in enumerate(LEVELS):
    cv.paste(Image.new("RGB", (TILE, TILE), (int(t), int(t), int(t))),
             (PAD + i * (TILE + PAD), y2))
    dr.text((PAD + i * (TILE + PAD) + 1, y2 + TILE + 2), f"{t:.0f}", font=f_s,
            fill=(160, 172, 188))
dr.text((PAD, y2 + TILE + 20),
        f"第1行＝原贴图  第2行＝全分面平均色  第3行＝目标 16 级  ·  "
        f"平均误差 {np.mean(errs):.2f}  平均饱和度 {np.mean([c['sat'] for c in chosen]):.4f}  "
        f"平均纹理标准差 {np.mean([c['std'] for c in chosen]):.1f}",
        font=f_l, fill=(150, 162, 178))
p = os.path.join(OUT, "PALETTE16_V3.png")
cv.save(p)
print(f"\nwrote {p}  {cv.size}")

json.dump({"jar": os.path.basename(JAR), "sat_weight": SAT_W, "uniform_weight": UNIFORM_W,
           "levels": [float(v) for v in LEVELS], "blocks": chosen,
           "shortlist": [{"level": i, "target": float(LEVELS[i]),
                          "candidates": [{k: r[k] for k in
                                          ("id", "name", "grey", "sat", "eff_sat", "std",
                                           "rgb", "paths")} for r in rk]}
                         for i, rk in enumerate(short)],
           "mean_abs_error": float(np.mean(errs)),
           "filters": {"piston_immovable_excluded": True, "block_entity_excluded": True,
                       "slabs_allowed": True, "max_sat": MAX_SAT, "max_texture_std": MAX_STD},
           "method": "lightness-discounted saturation + texture-uniformity penalty + "
                     "optimal ordered assignment"},
          open(os.path.join(OUT, "palette16_v3.json"), "w"), indent=2, ensure_ascii=False)
print(f"wrote {os.path.join(OUT, 'palette16_v3.json')}")
print("PALETTE_V3_DONE")
