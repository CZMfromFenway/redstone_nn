#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Find 16 REAL Minecraft blocks whose own textures are neutral greys spanning black to
white, so the 16-level palette does not depend on recolouring concrete.

WHY THIS IS NEEDED
The current palette was derived from a modified concrete set, and the complaint is fair:
hand-tinting a block's texture is not "authentic". What is wanted instead is 16 blocks that
already look like the 16 levels, unmodified.

HOW A BLOCK'S COLOUR IS MEASURED
Each block texture is a 16x16 PNG, some with transparency. A block's perceived colour is
taken as the ALPHA-WEIGHTED MEAN of its pixels, so a partly transparent texture is judged
by the part you actually see rather than by its transparent border. From that mean colour:
    grey  = mean of R,G,B          (the value a greyscale pipeline would assign)
    sat   = (max - min) / max      (0 = perfectly neutral)
    lum   = 0.299R + 0.587G + 0.114B   (perceptual, reported for reference)

HOW THE 16 ARE CHOSEN
For each of the 16 target levels (0,17,34,...,255) pick the block minimising
    |grey - target| + SAT_WEIGHT * sat
subject to every block being DISTINCT, so the palette has 16 different blocks. Textures
that are not simple full blocks (animated strips, overlays, cross-shapes, directional
faces) are excluded because they cannot tile one flat pixel.
"""
import json, os, re, sys, zipfile
import numpy as np
from PIL import Image

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
JAR = os.path.join(ROOT, "..", "99_mc_texture", "1.21.11-Fabric 0.19.5.jar")
JAR = os.path.normpath(JAR)
OUT = os.path.join(ROOT, "results", "palette16")
os.makedirs(OUT, exist_ok=True)
LEVELS = np.linspace(0, 255, 16)
SAT_WEIGHT = 90.0          # how many grey units one unit of saturation is worth

print(f"jar: {JAR}")
z = zipfile.ZipFile(JAR)

# block id -> English name, from the vanilla lang file
names = {}
try:
    lang = json.loads(z.read("assets/minecraft/lang/en_us.json").decode("utf-8"))
    for k, v in lang.items():
        if k.startswith("block.minecraft."):
            names[k[len("block.minecraft."):]] = v
except Exception as e:
    print("lang unavailable:", e)

# a texture name maps to a block id when the block exists
block_ids = set(names)

# suffixes that mean "a face/variant/stage", not a standalone flat block
BAD_SUFFIX = re.compile(
    r"(_top|_bottom|_side|_front|_back|_inner|_outer|_end|_vertical|_horizontal"
    r"|_stage\d+|_age\d+|_powered|_lit|_on|_off|_east|_west|_north|_south|_up|_down"
    r"|_corner|_edge|_tip|_base|_head|_body|_stem|_leaves|_overlay|_cracked|_open|_closed"
    r"|_singular|_plural|_active|_inactive|_bloom|_bud|_dead|_dry|_wet|_full|_empty"
    r"|_connected|_alt|_mirrored|_rotated|_eye|_mouth|_inside|_outside|_casing)$")

SKIP = {"destroy_stage_0", "destroy_stage_1", "destroy_stage_2", "destroy_stage_3",
        "destroy_stage_4", "destroy_stage_5", "destroy_stage_6", "destroy_stage_7",
        "destroy_stage_8", "destroy_stage_9", "debug", "debug2",
        "pumpkin_face_off", "pumpkin_face_on", "sun", "moon_phases", "fire_0", "fire_1",
        "water_still", "water_flow", "lava_still", "lava_flow"}


def measure(data):
    im = Image.open(__import__("io").BytesIO(data))
    if im.mode not in ("RGB", "RGBA"):
        im = im.convert("RGBA")
    else:
        im = im.convert("RGBA")
    a = np.asarray(im, dtype=np.float64)
    rgb, alpha = a[..., :3], a[..., 3:4] / 255.0
    w = alpha.sum()
    if w <= 1e-6:
        return None
    mean = (rgb * alpha).sum((0, 1)) / w
    mx, mn = mean.max(), mean.min()
    sat = float((mx - mn) / mx) if mx > 1e-9 else 0.0
    grey = float(mean.mean())
    lum = float(0.299 * mean[0] + 0.587 * mean[1] + 0.114 * mean[2])
    opaque = float((alpha > 0.99).mean())
    return {"rgb": [round(float(v), 1) for v in mean], "grey": grey, "sat": sat,
            "lum": lum, "opaque": opaque,
            "w": im.width, "h": im.height}


rows = []
for name in z.namelist():
    if not (name.startswith("assets/minecraft/textures/block/") and name.endswith(".png")):
        continue
    stem = os.path.basename(name)[:-4]
    if stem in SKIP or BAD_SUFFIX.search(stem):
        continue
    if stem not in block_ids:
        continue                       # texture does not correspond to a standalone block
    try:
        m = measure(z.read(name))
    except Exception:
        continue
    if m is None:
        continue
    if m["w"] % 16 or m["h"] % 16:
        continue                       # not a clean 16-multiple texture
    if m["h"] != m["w"]:
        continue                       # animated strip
    if m["opaque"] < 0.98:
        continue                       # partly transparent: not a solid surface
    m.update({"id": stem, "name": names.get(stem, stem), "path": name})
    rows.append(m)

print(f"candidate solid blocks with a neutral-capable texture: {len(rows)}")

# how neutral is the field? show the most neutral ones as a sanity check
rows.sort(key=lambda r: r["sat"])
print()
print("  20 most neutral blocks (lowest saturation):")
print(f"  {'block':34s} {'grey':>6s} {'sat':>6s} {'rgb':>20s}")
for r in rows[:20]:
    print(f"  {r['id']:34s} {r['grey']:6.1f} {r['sat']:6.3f} {str(r['rgb']):>20s}")

# ---------------------------------------------------------------- availability report
pool_all = [r for r in rows if r["sat"] < 0.22]
pool_all.sort(key=lambda r: r["grey"])
print()
print("  availability in the dark end (grey < 80):")
print(f"  {'grey':>6s} {'sat':>6s} {'rgb':>22s}  id")
for r in pool_all:
    if r["grey"] < 80:
        print(f"  {r['grey']:6.1f} {r['sat']:6.3f} {str(r['rgb']):>22s}  {r['id']}")
print()
print("  availability in the bright end (grey > 215):")
for r in pool_all:
    if r["grey"] > 215:
        print(f"  {r['grey']:6.1f} {r['sat']:6.3f} {str(r['rgb']):>22s}  {r['id']}")

# ---------------------------------------------------------------- optimal ordered assignment
# A per-level greedy can be wasteful when the field is sparse: picking the closest block for
# level 2 may consume the only block that suited level 3. Because both the levels and the
# blocks are ORDERED (brightness), the optimal assignment of 16 distinct blocks to the 16
# levels must preserve that order, so a small dynamic program solves it exactly.
def assign(cands, levels, sat_w):
    cands = sorted(cands, key=lambda r: r["grey"])
    B, L = len(cands), len(levels)
    INF = float("inf")
    # cost[l][b] = cost of giving level l the candidate b
    cost = [[abs(cands[b]["grey"] - levels[l]) + sat_w * cands[b]["sat"] for b in range(B)]
            for l in range(L)]
    # dp[l][b] = best total cost using levels 0..l with level l assigned to candidate b
    dp = [[INF] * B for _ in range(L)]
    back = [[-1] * B for _ in range(L)]
    for b in range(B):
        dp[0][b] = cost[0][b]
    for l in range(1, L):
        best_prev, best_b = INF, -1
        for b in range(B):
            # candidate for level l-1 must have index < b
            if b - 1 >= 0 and dp[l - 1][b - 1] < best_prev:
                best_prev = dp[l - 1][b - 1]
                best_b = b - 1
            if best_b >= 0:
                dp[l][b] = best_prev + cost[l][b]
                back[l][b] = best_b
    end_b = min(range(B), key=lambda b: dp[L - 1][b])
    if dp[L - 1][end_b] == INF:
        raise SystemExit("no feasible ordered assignment")
    seq = [0] * L
    b = end_b
    for l in range(L - 1, -1, -1):
        seq[l] = b
        b = back[l][b]
    return [cands[i] for i in seq]


chosen_blocks = assign(pool_all, LEVELS, SAT_WEIGHT)
chosen = []
for i, (blk, target) in enumerate(zip(chosen_blocks, LEVELS)):
    chosen.append({"level": i, "target_grey": float(target), "id": blk["id"],
                   "name": blk["name"], "grey": blk["grey"], "sat": blk["sat"],
                   "rgb": blk["rgb"], "lum": blk["lum"], "path": blk["path"]})

print()
print("=" * 104)
print("THE 16 BLOCKS -- optimal ordered assignment of unmodified vanilla textures")
print("=" * 104)
print(f"  {'lvl':>3} {'target':>7} {'block id':32s} {'actual grey':>11s} {'err':>6s} "
      f"{'sat':>6s}  name")
print("  " + "-" * 98)
errs = []
for c in chosen:
    err = c["grey"] - c["target_grey"]
    errs.append(abs(err))
    print(f"  {c['level']:3d} {c['target_grey']:7.1f} {c['id']:32s} {c['grey']:11.1f} "
          f"{err:+6.1f} {c['sat']:6.3f}  {c['name']}")
print("  " + "-" * 98)
print(f"  mean |error| {np.mean(errs):.2f} grey units, max {np.max(errs):.1f}")
print(f"  mean saturation {np.mean([c['sat'] for c in chosen]):.3f}, "
      f"max {np.max([c['sat'] for c in chosen]):.3f}")
print(f"  all 16 are distinct blocks: {len({c['id'] for c in chosen}) == 16}")

# ---------------------------------------------------------------- render swatches
TILE = 64
PAD = 6
W = 16 * (TILE + PAD) + PAD
H = 3 * (TILE + 20) + 64
cv = Image.new("RGB", (W, H), (20, 22, 26))
from PIL import ImageDraw, ImageFont


def font(sz):
    for fn in ("msyh.ttc", "simhei.ttf", "simsun.ttc"):
        p2 = os.path.join("C:\\Windows\\Fonts", fn)
        if os.path.exists(p2):
            try:
                return ImageFont.truetype(p2, sz)
            except Exception:
                pass
    return ImageFont.load_default()


dr = ImageDraw.Draw(cv)
f_t, f_l, f_s = font(17), font(11), font(10)
dr.text((PAD, 8), "\u7528 16 \u4e2a\u539f\u7248\u65b9\u5757\u7684\u6750\u8d28\u672c\u8eab\u505a\u7070\u9636\uff08\u672a\u6539\u4efb\u4f55\u8d34\u56fe\uff09",
        font=f_t, fill=(232, 236, 242))
y0 = 38
import io as _io
for i, c in enumerate(chosen):
    im = Image.open(_io.BytesIO(z.read(c["path"]))).convert("RGB")
    im = im.resize((TILE, TILE), Image.NEAREST)
    cv.paste(im, (PAD + i * (TILE + PAD), y0))
    dr.text((PAD + i * (TILE + PAD) + 1, y0 + TILE + 2), c["id"][:11], font=f_s,
            fill=(160, 172, 188))
y1 = y0 + TILE + 22
for i, c in enumerate(chosen):
    sw = Image.new("RGB", (TILE, TILE), tuple(int(v) for v in c["rgb"]))
    cv.paste(sw, (PAD + i * (TILE + PAD), y1))
    dr.text((PAD + i * (TILE + PAD) + 1, y1 + TILE + 2), f"{c['grey']:.0f}", font=f_s,
            fill=(160, 172, 188))
y2 = y1 + TILE + 22
for i, t in enumerate(LEVELS):
    sw = Image.new("RGB", (TILE, TILE), (int(t), int(t), int(t)))
    cv.paste(sw, (PAD + i * (TILE + PAD), y2))
    dr.text((PAD + i * (TILE + PAD) + 1, y2 + TILE + 2), f"{t:.0f}", font=f_s,
            fill=(160, 172, 188))
dr.text((PAD, y2 + TILE + 18),
        "\u7b2c1\u884c\uff1d\u65b9\u5757\u539f\u8d34\u56fe\uff0816\u00d716 \u653e\u5927\uff09   "
        "\u7b2c2\u884c\uff1d\u8be5\u8d34\u56fe\u7684\u5e73\u5747\u8272   "
        "\u7b2c3\u884c\uff1d\u76ee\u6807\u7684 16 \u5ea6\u660e\u5ea6\u5206\u7ea7",
        font=f_l, fill=(150, 162, 178))
p = os.path.join(OUT, "PALETTE16.png")
cv.save(p)
print(f"\nwrote {p}  {cv.size}")

json.dump({"jar": os.path.basename(JAR), "sat_weight": SAT_WEIGHT,
           "levels": [float(v) for v in LEVELS],
           "blocks": chosen,
           "mean_abs_error": float(np.mean(errs)),
           "max_abs_error": float(np.max(errs)),
           "mean_sat": float(np.mean([c["sat"] for c in chosen])),
           "max_sat": float(np.max([c["sat"] for c in chosen])),
           "method": "optimal ordered assignment (dynamic program) over solid 16x16 vanilla "
                     "block textures with saturation < 0.22, minimising "
                     "|grey-target| + 90*saturation"},
          open(os.path.join(OUT, "palette16.json"), "w"), indent=2, ensure_ascii=False)
print(f"wrote {os.path.join(OUT, 'palette16.json')}")
print("PALETTE16_DONE")
