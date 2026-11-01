# -*- coding: utf-8 -*-
"""把 dt_demo.schem 按部分画成 4 张俯视图（贴 MC 1.21 材质）。

材质从 99_mc_texture 的 jar 里现取（不落库，解到临时目录）。
比较器材质按 `facing` 旋转：材质本身的"前"要对上输出侧。
红石粉按方块状态逐边画线，L 形/十字形即游戏里的实际连接。
箭头：短尾在输入端（facing 侧），长箭头指向它充能的那一格，左上角 C/S 是模式。

用法： python dt_render.py
输出： kb/figures/dt_map_{D,DD,T,DDT}.png
"""
import os
import sys
import tempfile
import zipfile

from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dt_sim as sim

JAR = r'E:\redstone_nn\99_mc_texture\1.21.11-Fabric 0.19.5.jar'
FIG = r'E:\redstone_nn\kb\figures'
TEXDIR = os.path.join(tempfile.gettempdir(), 'dt_tex')

TEX = {
    'light_blue_concrete': 'assets/minecraft/textures/block/light_blue_concrete.png',
    'orange_concrete': 'assets/minecraft/textures/block/orange_concrete.png',
    'white_concrete': 'assets/minecraft/textures/block/white_concrete.png',
    'black_concrete': 'assets/minecraft/textures/block/black_concrete.png',
    'barrel': 'assets/minecraft/textures/block/barrel_top.png',
    'comparator': 'assets/minecraft/textures/block/comparator.png',
}
# 材质默认朝"南"（输出朝南）；按 facing 转到对应方向（逆时针为正）
ROT = {'north': 180, 'east': 90, 'south': 0, 'west': 270}


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


def draw_part(name, z0, z1, grid, tex, cell=72, modes=None, sub=''):
    xs = [x for (x, y, z) in grid if z0 <= z <= z1]
    x0, x1 = min(xs), max(xs)
    W = x1 - x0 + 1
    Hh = z1 - z0 + 1
    pad, head = 14, 26
    im = Image.new('RGB', (pad * 2 + W * cell + 8, head + pad * 2 + Hh * cell), (18, 18, 22))
    dr = ImageDraw.Draw(im, 'RGBA')
    dr.text((pad, 6), f'{name}   top view   {W} x {Hh}   (MC 1.21 textures)',
            fill=(235, 235, 245))

    for (x, y, z), nm in sorted(grid.items(), key=lambda kv: (kv[0][2], kv[0][0])):
        if not (z0 <= z <= z1) or nm.endswith('air'):
            continue
        base = nm.split('[')[0].replace('minecraft:', '')
        px = pad + (x - x0) * cell
        py = head + pad + (z - z0) * cell
        props = {}
        if '[' in nm:
            for kv in nm[nm.index('[') + 1:-1].split(','):
                a, b = kv.split('=', 1); props[a] = b

        key = 'barrel' if base == 'barrel' else base
        if key in tex:
            t = tex[key]
            if key == 'comparator':
                t = t.rotate(ROT.get(props.get('facing', 'south'), 0), expand=False)
            t = t.resize((cell, cell), Image.NEAREST)
            im.paste(t, (px, py), t)
        else:
            dr.rectangle([px, py, px + cell - 1, py + cell - 1], fill=(60, 60, 66))

        if base == 'comparator':
            fin, out = props['facing'], sim.FLIP[props['facing']]
            cx, cy = px + cell / 2, py + cell / 2
            ix, iz = sim.D[fin]
            dx, dz = sim.D[out]
            dr.line([cx - ix * cell * .42, cy - iz * cell * .42,
                     cx - ix * cell * .12, cy - iz * cell * .12],
                    fill=(240, 240, 250, 235), width=5)
            dr.line([cx + dx * cell * .02, cy + dz * cell * .02,
                     cx + dx * cell * .40, cy + dz * cell * .40],
                    fill=(30, 30, 40, 245), width=6)
            ax, ay = cx + dx * cell * .40, cy + dz * cell * .40
            perp = (-dz, dx)
            dr.polygon([(ax + dx * 10, ay + dz * 10),
                        (ax + perp[0] * 8, ay + perp[1] * 8),
                        (ax - perp[0] * 8, ay - perp[1] * 8)], fill=(30, 30, 40, 245))
            md = (modes or {}).get((x, y, z), props['mode'])
            mode = 'C' if md == 'compare' else 'S'
            dr.rectangle([px + 2, py + 2, px + 20, py + 20], fill=(255, 255, 255, 215))
            dr.text((px + 7, py + 5), mode, fill=(20, 20, 20))

        elif base == 'redstone_wire':
            cx, cy = px + cell / 2, py + cell / 2
            col = (200, 30, 30, 255)
            for side, d in (('north', (0, -1)), ('south', (0, 1)),
                            ('east', (1, 0)), ('west', (-1, 0))):
                if props.get(side, 'none') in ('side', 'up'):
                    dr.line([cx, cy, cx + d[0] * cell / 2, cy + d[1] * cell / 2],
                            fill=col, width=9)
            dr.ellipse([cx - 7, cy - 7, cx + 7, cy + 7], fill=col)

        elif base == 'white_concrete':
            dr.text((px + cell * .30, py + cell * .40), 'IN', fill=(20, 20, 20))
        elif base == 'black_concrete':
            dr.text((px + cell * .24, py + cell * .40), 'OUT', fill=(245, 245, 245))

    return im


def main():
    W, H, L, grid = sim.load_schem(sim.SRC)
    tex = get_tex()
    for nm, z0, z1, col in sim.PARTS:
        im = draw_part(nm, z0, z1, grid, tex)
        p = os.path.join(FIG, f'dt_map_{nm}.png')
        im.save(p)
        print('wrote', p, im.size)


if __name__ == '__main__':
    main()
