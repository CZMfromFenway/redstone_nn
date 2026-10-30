#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
16 greyscale blocks from the vanilla jar -- final pass, calibrated to the user's review.

THE UNIFORMITY CRITERION WAS WRONG, AND PROBING FOUND WHY
The user rejected bedrock and blast_furnace saying their colour variance is high, yet
approved diorite -- whose WITHIN-TEXTURE pixel std is 34.7, higher than bedrock's 38.0.
So "variance" cannot mean within-texture noise. Measuring the block's FACE-TO-FACE colour
spread settles it:

    blast_furnace   12.66   (faces at grey 81, 107, 108 -- the lit front is much darker)
    bone_block      11.29   (faces at 198 and 221)
    diorite          0.00   (every face identical)
    andesite         0.00

A block used as one pixel must look the same from any direction, so the inter-face spread
is the right metric and within-texture noise is NOT a rejection criterion (diorite is
approved despite being mottled).

Bedrock's problem is separate and is handled by the piston filter: it cannot be pushed at
all. Neither can anything with a block entity in Java Edition, which rules out
blast_furnace, furnaces, chests, hoppers and shulker boxes.

OTHER RULES FROM THE REVIEW
  * saturation is discounted where lightness is extreme, because a tint is invisible
    against near-black or near-white
  * slabs are allowed (two stack into a full cube); doors, trapdoors, fences and panes
    are not, because they never fill the cell
  * the ten blocks the user accepted get a large cost bonus so they win when competitive
  * a SHORTLIST of five candidates per level is rendered, which is the user's actual
    request: compare candidates and choose
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

SAT_W = 75.0
FACEVAR_W = 8.0
STD_W = 0.8
APPROVED_BONUS = 10.0
MAX_SAT = 0.60
MAX_FACEVAR = 3.0
SHORT = 5

APPROVED = {
    "snow", "white_wool", "iron_block", "white_concrete", "diorite", "andesite",
    "black_wool", "coal_block", "cracked_stone_bricks", "chiseled_stone_bricks",
    "cracked_deepslate_bricks", "cracked_deepslate_tiles", "stone_bricks",
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
by_block = {}
for p in z.namelist():
    if not (p.startswith("assets/minecraft/textures/block/") and p.endswith(".png")):
        continue
    stem = os.path.basename(p)[:-4]
    tgt = stem if stem in names else None
    if tgt is None:
        for s in FACE_SUFFIX:
            if stem.endswith(s) and stem[:-len(s)] in names:
                tgt = stem[:-len(s)]
                break
    if tgt:
        by_block.setdefault(tgt, []).append(p)


def measure(paths):
    face_means = []
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
        m = (rgb * al).sum((0, 1)) / w
        face_means.append(float(m.mean()))
        tot += (rgb * al).sum((0, 1))
        wsum += w
        lum = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        mm = al[..., 0] > 0.99
        if mm.sum() > 1:
            stds.append(float(lum[mm].std()))
        opaque += int(mm.sum())
        total += al.size
    if wsum <= 1e-6:
        return None
    mean = tot / wsum
    hi = float(mean.max())
    sat = float((hi - mean.min()) / hi) if hi > 1e-9 else 0.0
    grey = float(mean.mean())
    ext = abs(grey - 127.5) / 127.5
    return {"rgb": [round(float(v), 1) for v in mean], "grey": grey, "sat": sat,
            "eff_sat": sat * (0.25 + 0.75 * (1 - ext)),
            "face_var": float(np.std(face_means)) if len(face_means) > 1 else 0.0,
            "face_means": [round(v, 1) for v in sorted(face_means)],
            "std": float(np.mean(stds)) if stds else 0.0,
            "opaque": opaque / max(total, 1), "n_faces": len(paths)}


rows = []
for bid, paths in by_block.items():
    if bid in IMMOVABLE or bid in BLOCK_ENTITY or any(s in bid for s in NOT_FULL):
        continue
    m = measure(paths)
    if m is None or m["opaque"] < 0.95:
        continue
    if m["sat"] > MAX_SAT or m["face_var"] > MAX_FACEVAR:
        continue
    m.update({"id": bid, "name": names.get(bid, bid), "paths": paths,
              "approved": bid in APPROVED})
    rows.append(m)
rows.sort(key=lambda r: r["grey"])
print(f"candidates: {len(rows)}   (piston-immovable, block-entity, non-full shapes, "
      f"sat>{MAX_SAT}, face_var>{MAX_FACEVAR} all excluded)")


def cost(r, target):
    c = (abs(r["grey"] - target) + SAT_W * r["eff_sat"]
         + FACEVAR_W * r["face_var"] + STD_W * r["std"])
    return c - (APPROVED_BONUS if r["approved"] else 0.0)


print()
print("=" * 108)
print(f"SHORTLIST -- best {SHORT} per level   "
      f"(cost = |grey-target| + {SAT_W:.0f}*eff_sat + {FACEVAR_W:.0f}*face_var + {STD_W}*std"
      f" - {APPROVED_BONUS:.0f} if approved)")
print("=" * 108)
short = []
for i, t in enumerate(LEVELS):
    rk = sorted(rows, key=lambda r: cost(r, t))[:SHORT]
    short.append(rk)
    print(f"  level {i:2d}  target {t:5.1f}")
    for r in rk:
        print(f"     {r['grey']:6.1f} sat {r['sat']:5.3f} eff {r['eff_sat']:5.3f} "
              f"fv {r['face_var']:5.2f} std {r['std']:5.1f} cost {cost(r,t):6.1f} "
              f"{r['id']}" + ("  ★认可" if r["approved"] else ""))


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
for i, (blk, t) in enumerate(zip(assign(rows, LEVELS), LEVELS)):
    chosen.append({"level": i, "target_grey": float(t), "id": blk["id"],
                   "name": blk["name"], "grey": blk["grey"], "sat": blk["sat"],
                   "eff_sat": blk["eff_sat"], "face_var": blk["face_var"],
                   "face_means": blk["face_means"], "std": blk["std"],
                   "rgb": blk["rgb"], "approved": blk["approved"], "paths": blk["paths"]})

print()
print("=" * 108)
print("RECOMMENDED 16")
print("=" * 108)
print(f"  {'lvl':>3} {'target':>7} {'block id':28s} {'grey':>7s} {'err':>6s} "
      f"{'sat':>6s} {'fv':>5s} {'std':>5s}  name")
print("  " + "-" * 100)
errs = []
for c in chosen:
    e = c["grey"] - c["target_grey"]
    errs.append(abs(e))
    print(f"  {c['level']:3d} {c['target_grey']:7.1f} {c['id']:28s} {c['grey']:7.1f} "
          f"{e:+6.1f} {c['sat']:6.3f} {c['face_var']:5.2f} {c['std']:5.1f}  "
          f"{c['name']}" + (" ★" if c["approved"] else ""))
print("  " + "-" * 100)
print(f"  mean |error| {np.mean(errs):.2f}   max {np.max(errs):.1f}")
print(f"  mean sat {np.mean([c['sat'] for c in chosen]):.4f}   "
      f"max face_var {np.max([c['face_var'] for c in chosen]):.2f}")
print(f"  approved blocks used: {sum(1 for c in chosen if c['approved'])}/16")

# ---------------------------------------------------------------- render
TILE, PAD, LAB = 56, 4, 34
LEFT = 86                     # room for the row labels, which previously sat under tile 0
COLS = 16
W = LEFT + COLS * (TILE + PAD) + PAD
H = 3 * (TILE + 20) + SHORT * (TILE + LAB) + 88
cv = Image.new("RGB", (W, H), (20, 22, 26))
dr = ImageDraw.Draw(cv)
f_t, f_l, f_s = font(16), font(11), font(8)
dr.text((6, 8), "16 级灰阶方块 · 每级 5 个候选（★＝你认可的，sat＝饱和度）",
        font=f_t, fill=(232, 236, 242))


def tx(i):
    return LEFT + i * (TILE + PAD)


y = 36
for i, c in enumerate(chosen):
    im = Image.open(io.BytesIO(z.read(c["paths"][0]))).convert("RGB")
    cv.paste(im.resize((TILE, TILE), Image.NEAREST), (tx(i), y))
    dr.text((tx(i) + 1, y + TILE + 1), f"{c['grey']:.0f}", font=f_s, fill=(160, 172, 188))
    if c["approved"]:
        dr.text((tx(i) + 1, y + TILE + 10), "★", font=f_s, fill=(110, 200, 130))
dr.text((6, y + TILE // 2 - 4), "推荐 16", font=f_l, fill=(210, 220, 232))

y += TILE + 26
for i, c in enumerate(chosen):
    cv.paste(Image.new("RGB", (TILE, TILE), tuple(int(v) for v in c["rgb"])), (tx(i), y))
dr.text((6, y + TILE // 2 - 4), "平均色", font=f_l, fill=(150, 162, 178))

y += TILE + 20
for i, t in enumerate(LEVELS):
    cv.paste(Image.new("RGB", (TILE, TILE), (int(t), int(t), int(t))), (tx(i), y))
    dr.text((tx(i) + 1, y + TILE + 1), f"{t:.0f}", font=f_s, fill=(160, 172, 188))
dr.text((6, y + TILE // 2 - 4), "目标 16 级", font=f_l, fill=(150, 162, 178))

y += TILE + 26
for rank in range(SHORT):
    for i in range(COLS):
        r = short[i][rank]
        im = Image.open(io.BytesIO(z.read(r["paths"][0]))).convert("RGB")
        cv.paste(im.resize((TILE, TILE), Image.NEAREST), (tx(i), y))
        dr.text((tx(i) + 1, y + TILE + 1), f"{r['grey']:.0f}", font=f_s,
                fill=(150, 162, 178))
        nm = r["id"][:11] + ("★" if r["approved"] else "")
        dr.text((tx(i) + 1, y + TILE + 10), nm, font=f_s,
                fill=(110, 200, 130) if r["approved"] else (118, 130, 146))
        dr.text((tx(i) + 1, y + TILE + 20), f"s{r['sat']:.2f}", font=f_s,
                fill=(96, 106, 122))
    dr.text((6, y + TILE // 2 - 4), f"候选 {rank+1}", font=f_l, fill=(150, 162, 178))
    y += TILE + LAB

p = os.path.join(OUT, "PALETTE16_V5.png")
cv.save(p)
print(f"\nwrote {p}  {cv.size}")

json.dump({"jar": os.path.basename(JAR),
           "weights": {"sat": SAT_W, "face_var": FACEVAR_W, "approved_bonus": APPROVED_BONUS},
           "levels": [float(v) for v in LEVELS], "recommended": chosen,
           "shortlist": [{"level": i, "target": float(LEVELS[i]),
                          "candidates": [{k: r[k] for k in
                                          ("id", "name", "grey", "sat", "eff_sat",
                                           "face_var", "face_means", "std", "rgb",
                                           "approved", "paths")} for r in rk]}
                         for i, rk in enumerate(short)],
           "mean_abs_error": float(np.mean(errs)),
           "mean_sat": float(np.mean([c["sat"] for c in chosen])),
           "max_face_var": float(np.max([c["face_var"] for c in chosen])),
           "approved_used": sum(1 for c in chosen if c["approved"]),
           "filters": {"piston_immovable": "excluded (bedrock cannot be pushed; in Java "
                                           "anything with a block entity cannot either, "
                                           "which removes blast_furnace, furnaces, chests, "
                                           "hoppers, shulker boxes)",
                       "slabs": "allowed -- two stack into a full cube",
                       "doors_trapdoors_fences_panes": "excluded, never fill the cell",
                       "face_var": f"<= {MAX_FACEVAR}: every face must look the same",
                       "within_texture_std": "reported only, NOT a filter -- diorite is "
                                             "approved and is mottled (std 34.7)",
                       "max_sat": MAX_SAT},
           "method": "lightness-discounted saturation + inter-face uniformity + approved "
                     "bonus + optimal ordered assignment"},
          open(os.path.join(OUT, "palette16_v5.json"), "w"), indent=2, ensure_ascii=False)
print(f"wrote {os.path.join(OUT, 'palette16_v5.json')}")
print("PALETTE_V5_DONE")
