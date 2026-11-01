# -*- coding: utf-8 -*-
"""把 dt_demo.schem 画成"具体"的俯视图：贴 MC 材质 + 红石粉连接 + 比较器指向。

材质从 99_mc_texture 的 jar 里现取（不落库，解到临时目录）。
比较器：箭头从**输入端（facing 侧）**指向**它充能的那一格**，并标 C（比较）/ S（减法）。
红石粉：按方块状态里的 east/north/south/west 逐边画线——这正是它在游戏里的连接形状。

用法： python dt_render.py
输出： kb/figures/dt_demo_map.png
"""
import os
import sys
import tempfile
import zipfile

from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dt_sim as sim

JAR = r'E:\redstone_nn\99_mc_texture\1.21.11-Fabric 0.19.5.jar'
OUT = r'E:\redstone_nn\kb\figures\dt_demo_map.png'
TEXDIR = os.path.join(tempfile.gettempdir(), 'dt_tex')

TEX = {
    'light_blue_concrete': 'assets/minecraft/textures/block/light_blue_concrete.png',
    'orange_concrete': 'assets/minecraft/textures/block/orange_concrete.png',
    'white_concrete': 'assets/minecraft/textures/block/white_concrete.png',
    'black_concrete': 'assets/minecraft/textures/block/black_concrete.png',
    'barrel': 'assets/minecraft/textures/block/barrel_top.png',
    'comparator': 'assets/minecraft/textures/block/comparator.png',
}


def get_tex():
    os.makedirs(TEXDIR, exist_ok=True)
    out = {}
    with zipfile.ZipFile(JAR) as z:
        for name, path in TEX.items():
            dst = os.path.join(TEXDIR, os.path.basename(path))
            if not os.path.exists(dst):
                with z.open(path) as f, open(dst, 'wb') as g:
                    g.write(f.read())
            out[name] = Image.open(dst).convert('RGBA')
    return out


def main():
    W, H, L, grid = sim.load_schem(sim.SRC)
    tex = get_tex()
    CELL = 64
    pad, head = 16, 30
    right = 190
    cw = pad * 2 + W * CELL + right
    chh = head + pad * 2 + L * CELL
    im = Image.new('RGB', (cw, chh), (18, 18, 22))
    dr = ImageDraw.Draw(im, 'RGBA')

    dr.text((pad, 8), 'dt_demo.schem   top view  7 x 30  (MC 1.21 textures)', fill=(235, 235, 245))

    # 部分框
    for nm, z0, z1, col in sim.PARTS:
        y0 = head + pad + z0 * CELL
        y1 = head + pad + (z1 + 1) * CELL
        dr.rectangle([pad - 8, y0 - 3, pad + W * CELL + 6, y1 - 3], outline=col, width=2)
        dr.text((pad + W * CELL + 14, y0 + 10), nm, fill=col)

    for (x, y, z), name in sorted(grid.items(), key=lambda kv: (kv[0][2], kv[0][0])):
        if name.endswith('air'):
            continue
        base = name.split('[')[0].replace('minecraft:', '')
        px = pad + x * CELL
        py = head + pad + z * CELL
        key = 'barrel' if base == 'barrel' else base
        if key in tex:
            t = tex[key].resize((CELL, CELL), Image.NEAREST)
            im.paste(t, (px, py), t)
        else:
            dr.rectangle([px, py, px + CELL - 1, py + CELL - 1], fill=(60, 60, 66))

        props = {}
        if '[' in name:
            for kv in name[name.index('[') + 1:-1].split(','):
                a, b = kv.split('=', 1); props[a] = b

        if base == 'comparator':
            # facing = 输入端；输出 = 反方向，箭头指向"它充能的那一格"
            fin = props['facing']
            out = sim.FLIP[fin]
            dx, dz = sim.D[out]
            cx, cy = px + CELL / 2, py + CELL / 2
            # 输入端小勾
            ix, iz = sim.D[fin]
            dr.line([cx - ix * 26, cy - iz * 26, cx - ix * 8, cy - iz * 8],
                    fill=(20, 20, 30, 220), width=4)
            # 输出箭头
            dr.line([cx, cy, cx + dx * 26, cy + dz * 26], fill=(15, 15, 25, 230), width=5)
            ax, ay = cx + dx * 26, cy + dz * 26
            perp = (-dz, dx)
            dr.polygon([(ax + dx * 9, ay + dz * 9),
                        (ax + perp[0] * 7, ay + perp[1] * 7),
                        (ax - perp[0] * 7, ay - perp[1] * 7)], fill=(15, 15, 25, 235))
            mode = 'C' if props['mode'] == 'compare' else 'S'
            dr.rectangle([px + 2, py + 2, px + 18, py + 18], fill=(255, 255, 255, 210))
            dr.text((px + 6, py + 4), mode, fill=(20, 20, 20))

        elif base == 'redstone_wire':
            cx, cy = px + CELL / 2, py + CELL / 2
            col = (200, 30, 30, 255)
            n = 0
            for side, d in (('north', (0, -1)), ('south', (0, 1)),
                            ('east', (1, 0)), ('west', (-1, 0))):
                if props.get(side, 'none') in ('side', 'up'):
                    n += 1
                    dr.line([cx, cy, cx + d[0] * CELL / 2, cy + d[1] * CELL / 2],
                            fill=col, width=8)
            dr.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], fill=col)
            if n == 0:
                dr.ellipse([cx - 10, cy - 10, cx + 10, cy + 10], fill=col)

        elif base == 'white_concrete':
            dr.text((px + 16, py + 24), 'IN', fill=(20, 20, 20))
        elif base == 'black_concrete':
            dr.text((px + 12, py + 24), 'OUT', fill=(245, 245, 245))

    for z in range(0, L, 5):
        dr.text((pad - 14, head + pad + z * CELL + 26), f'{z}', fill=(160, 160, 170))

    # 图例
    lx = pad + W * CELL + 14
    ly = head + pad + L * CELL - 210
    dr.text((lx, ly), 'barrel   = constant branch', fill=(210, 210, 220))
    dr.text((lx, ly + 16), 'C = compare   S = subtract', fill=(210, 210, 220))
    dr.text((lx, ly + 32), 'arrow: input side -> charged block', fill=(210, 210, 220))
    dr.text((lx, ly + 48), 'red lines = real dust connections', fill=(230, 130, 130))
    dr.text((lx, ly + 64), 'textures: MC 1.21 jar', fill=(150, 150, 165))
    im.save(OUT)
    print('wrote', OUT, im.size)


if __name__ == '__main__':
    main()
