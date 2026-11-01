# -*- coding: utf-8 -*-
"""dt_demo.schem 的读取与模拟（三期 Board 的改造版：加了红石粉）。

规则集（本图只用到这几种元件）：
  比较器  facing = 输入侧；输出方向 = facing 的反向。rear = 背后那格；侧输入 = 左右两格。
          比较模式 out = rear 若 side<=rear 否则 0；减法模式 out = max(rear-side,0)。
  读一格  顺序：粉 > 容器 > 比较器 > 导体块（充能 / 穿透读后面的容器）> 空气
  红石粉  power = max(相邻粉 -1, 相邻充能导体, 指向它的比较器输出)
  导体块  充能 = max(指向它的比较器输出) + 白块上的外部注入
  容器    信号由内容物给定（本图当常数用）

用法： python dt_sim.py
"""
import gzip
import os
import struct

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = r'E:\redstone_nn\03_phase3_quantized_network\data\dt_demo.schem'
OUT_TXT = os.path.join(HERE, '_dt_demo_trace.txt')

D = {'north': (0, -1), 'south': (0, 1), 'east': (1, 0), 'west': (-1, 0)}
FLIP = {'north': 'south', 'south': 'north', 'east': 'west', 'west': 'east'}
DIRS4 = ((0, -1), (0, 1), (1, 0), (-1, 0))


class R:
    def __init__(self, b):
        self.b, self.i = b, 0

    def u1(self):
        v = self.b[self.i]; self.i += 1; return v

    def i1(self):
        v = struct.unpack_from('>b', self.b, self.i)[0]; self.i += 1; return v

    def i2(self):
        v = struct.unpack_from('>h', self.b, self.i)[0]; self.i += 2; return v

    def i4(self):
        v = struct.unpack_from('>i', self.b, self.i)[0]; self.i += 4; return v

    def i8(self):
        v = struct.unpack_from('>q', self.b, self.i)[0]; self.i += 8; return v

    def f4(self):
        v = struct.unpack_from('>f', self.b, self.i)[0]; self.i += 4; return v

    def f8(self):
        v = struct.unpack_from('>d', self.b, self.i)[0]; self.i += 8; return v

    def s(self):
        n = struct.unpack_from('>H', self.b, self.i)[0]; self.i += 2
        v = self.b[self.i:self.i + n].decode('utf-8', 'replace'); self.i += n
        return v


def _t(R, t):
    if t == 1: return R.i1()
    if t == 2: return R.i2()
    if t == 3: return R.i4()
    if t == 4: return R.i8()
    if t == 5: return R.f4()
    if t == 6: return R.f8()
    if t == 7:
        n = R.i4(); v = R.b[R.i:R.i + n]; R.i += n; return v
    if t == 8: return R.s()
    if t == 9:
        et = R.u1(); n = R.i4(); return [_t(R, et) for _ in range(n)]
    if t == 10:
        d = {}
        while True:
            tt = R.u1()
            if tt == 0: return d
            k = R.s(); d[k] = _t(R, tt)
    if t == 11:
        n = R.i4(); return [_t(R, 3) for _ in range(n)]
    if t == 12:
        n = R.i4(); return [_t(R, 4) for _ in range(n)]
    raise ValueError(t)


def varints(buf):
    out, v, sh = [], 0, 0
    for b in buf:
        v |= (b & 0x7F) << sh
        if not (b & 0x80):
            out.append(v); v = 0; sh = 0
        else:
            sh += 7
    return out


def load_schem(path):
    r = R(gzip.open(path, 'rb').read())
    t = r.u1()
    r.s()                      # 根名字（空）
    root = _t(r, t)['Schematic']
    W, H, L = root['Width'], root['Height'], root['Length']
    blk = root['Blocks']
    pal = {v: k for k, v in blk['Palette'].items()}
    idx = varints(blk['Data'])
    grid, k = {}, 0
    for y in range(H):
        for z in range(L):
            for x in range(W):
                grid[(x, y, z)] = pal[idx[k]]
                k += 1
    return W, H, L, grid


class Circuit:
    def __init__(self):
        self.cmp, self.dust, self.solid, self.barrel, self.ext = {}, {}, {}, {}, {}

    def add(self, pos, name):
        base = name.split('[')[0].replace('minecraft:', '')
        props = {}
        if '[' in name:
            for kv in name[name.index('[') + 1:-1].split(','):
                a, b = kv.split('=', 1); props[a] = b
        if base == 'comparator':
            self.cmp[pos] = dict(facing=props['facing'], out=FLIP[props['facing']],
                                 mode=props['mode'], state=0)
        elif base == 'redstone_wire':
            self.dust[pos] = 0
        elif base == 'barrel':
            self.barrel[pos] = 0
        else:
            self.solid[pos] = base

    def charge(self, p):
        if p in self.ext:
            return self.ext[p]
        v = 0
        for q, c in self.cmp.items():
            dx, dz = D[c['out']]
            if (q[0] + dx, q[1], q[2] + dz) == p:
                v = max(v, c['state'])
        # 读数点（黑块）额外认相邻红石粉——见 README/正文里的建模说明
        if self.solid.get(p) == 'black_concrete':
            for s in DIRS4:
                q = (p[0] + s[0], p[1], p[2] + s[1])
                if q in self.dust:
                    v = max(v, self.dust[q])
        return v

    def read(self, q, toward, side=False):
        if side:
            if q in self.dust:
                return self.dust[q]
            if q in self.cmp:
                c = self.cmp[q]
                return c['state'] if D[c['out']] == toward else 0
            return 0
        if q in self.dust:
            return self.dust[q]
        if q in self.barrel:
            return self.barrel[q]
        if q in self.cmp:
            c = self.cmp[q]
            return c['state'] if D[c['out']] == toward else 0
        if q in self.solid:
            v = self.charge(q)
            if v:
                return v
            back = (q[0] - toward[0], q[1], q[2] - toward[1])
            if back in self.barrel:
                return self.barrel[back]
            return 0
        return 0

    def step_cmp(self):
        ch = False
        for p, c in self.cmp.items():
            dx, dz = D[c['out']]
            rear = (p[0] - dx, p[1], p[2] - dz)
            rv = self.read(rear, (dx, dz))
            sv = 0
            for s in DIRS4:
                if s in ((dx, dz), (-dx, -dz)):
                    continue
                sv = max(sv, self.read((p[0] + s[0], p[1], p[2] + s[1]),
                                       (-s[0], -s[1]), side=True))
            v = (rv if rv >= sv else 0) if c['mode'] == 'compare' else max(0, rv - sv)
            if v != c['state']:
                c['state'] = v; ch = True
        return ch

    def step_dust(self):
        ch = False
        new = {}
        for p in self.dust:
            v = 0
            for s in DIRS4:
                q = (p[0] + s[0], p[1], p[2] + s[1])
                if q in self.dust:
                    v = max(v, self.dust[q] - 1)
                elif q in self.solid:
                    v = max(v, self.charge(q))
                elif q in self.cmp:
                    c = self.cmp[q]
                    dx, dz = D[c['out']]
                    if (q[0] + dx, q[1], q[2] + dz) == p:
                        v = max(v, c['state'])
            new[p] = max(0, min(15, v))
        for p, v in new.items():
            if self.dust[p] != v:
                self.dust[p] = v; ch = True
        return ch

    def run(self, maxit=400):
        for i in range(maxit):
            a = self.step_cmp()
            b = self.step_dust()
            if not (a or b):
                return i + 1, True
        return maxit, False


PARTS = [('D', 0, 6, '#8ecae6'), ('DD', 7, 14, '#8ecae6'),
         ('T', 15, 20, '#f4a261'), ('DDT', 21, 29, '#f4a261')]


def build(W, H, L, grid):
    c = Circuit()
    for pos, name in grid.items():
        if name.endswith('air'):
            continue
        c.add(pos, name)
    return c


def render(W, H, L, grid, path):
    """俯视图：一层平面，每格一个方块。"""
    from PIL import Image, ImageDraw
    cell, pad, head = 46, 10, 34
    cw = pad * 2 + W * cell + 150
    chh = head + pad * 2 + L * cell
    im = Image.new('RGB', (cw, chh), (24, 24, 28))
    dr = ImageDraw.Draw(im)
    COL = {
        'light_blue_concrete': (120, 190, 230), 'orange_concrete': (240, 160, 90),
        'white_concrete': (248, 248, 248), 'black_concrete': (18, 18, 18),
        'barrel': (150, 110, 60), 'comparator': (150, 150, 165), 'redstone_wire': (200, 40, 40),
    }
    dr.text((pad, 8), 'dt_demo.schem   top view  7 x 30', fill=(235, 235, 245))
    for nm, z0, z1, col in PARTS:
        y0 = head + pad + z0 * cell
        y1 = head + pad + (z1 + 1) * cell
        dr.rectangle([pad - 6, y0 - 2, pad + W * cell + 4, y1 - 2], outline=col, width=2)
        dr.text((pad + W * cell + 12, y0 + 8), nm, fill=col)
    for (x, y, z), name in grid.items():
        if name.endswith('air'):
            continue
        base = name.split('[')[0].replace('minecraft:', '')
        px = pad + x * cell
        py = head + pad + z * cell
        dr.rectangle([px + 2, py + 2, px + cell - 2, py + cell - 2],
                     fill=COL.get(base, (90, 90, 90)), outline=(40, 40, 40))
        if base == 'comparator':
            fac = name[name.index('facing=') + 7:].split(',')[0]
            dx, dz = D[FLIP[fac]]
            cx, cy = px + cell // 2, py + cell // 2
            dr.line([cx, cy, cx + dx * 16, cy + dz * 16], fill=(30, 30, 40), width=3)
            dr.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=(30, 30, 40))
        elif base == 'barrel':
            dr.text((px + 12, py + 14), 'B', fill=(255, 240, 200))
        elif base == 'redstone_wire':
            dr.text((px + 14, py + 14), 'r', fill=(255, 220, 220))
        elif base == 'white_concrete':
            dr.text((px + 12, py + 14), 'IN', fill=(20, 20, 20))
        elif base == 'black_concrete':
            dr.text((px + 8, py + 14), 'OUT', fill=(250, 250, 250))
    for z in range(0, L, 5):
        dr.text((pad - 26, head + pad + z * cell + 14), f'z{z}', fill=(150, 150, 160))
    im.save(path)
    print('wrote', path)
def main():
    W, H, L, grid = load_schem(SRC)
    c = build(W, H, L, grid)
    L_ = [f'尺寸 {W}x{H}x{L}   块 {len(grid)}',
          f'比较器 {len(c.cmp)}  粉 {len(c.dust)}  导体 {len(c.solid)}  容器 {len(c.barrel)}', '']
    whites = sorted(p for p, k in c.solid.items() if k == 'white_concrete')
    blacks = sorted(p for p, k in c.solid.items() if k == 'black_concrete')

    L_.append('=== 各部分的元件 ===')
    for nm, z0, z1, _ in PARTS:
        L_.append(f'[{nm}] 白{[p for p in whites if z0 <= p[2] <= z1]} '
                  f'黑{[p for p in blacks if z0 <= p[2] <= z1]}')
        for p, v in sorted(c.cmp.items()):
            if z0 <= p[2] <= z1:
                dx, dz = D[v['out']]
                L_.append(f'    cmp{p} facing={v["facing"]:5s} 读{(p[0]-dx,0,p[2]-dz)} '
                          f'写{(p[0]+dx,0,p[2]+dz)} {v["mode"]}')
        L_.append(f'    桶{[p for p in sorted(c.barrel) if z0 <= p[2] <= z1]}')
        L_.append(f'    粉{[p for p in sorted(c.dust) if z0 <= p[2] <= z1]}')

    def run_one(barrel_vals, x):
        cc = build(W, H, L, grid)
        for b, v in barrel_vals.items():
            cc.barrel[b] = v
        cc.ext = {w: x for w in whites}
        cc.run()
        out = {}
        for nm, z0, z1, _ in PARTS:
            outs = [p for p in blacks if z0 <= p[2] <= z1]
            out[nm] = max([cc.charge(o) for o in outs] or [0])
        return out

    # 单一常数 K：四个部分一起看
    for K in (0, 4, 8, 12, 15):
        L_.append('')
        L_.append(f'=== 所有桶 = {K}；白块注入 x；读黑块 ===')
        L_.append('   x |' + ''.join(f'{nm:>6}' for nm, *_ in PARTS))
        for x in range(16):
            o = run_one({b: K for b in c.barrel}, x)
            L_.append(f'  {x:2d} |' + ''.join(f'{o[nm]:6d}' for nm, *_ in PARTS))

    # D 的两个桶分别是 b（被动支路）与 a（A 支路）
    bD, aD = (1, 0, 0), (5, 0, 3)
    L_.append('')
    L_.append('=== 单 D：b=桶(1,0,0)，a=桶(5,0,3) ===')
    L_.append('   b  a |' + ''.join(f'{x:>4}' for x in range(16)))
    for bv in (0, 5, 10):
        for av in (0, 5, 10):
            row = []
            for x in range(16):
                o = run_one({q: (bv if q == bD else av if q == aD else 0) for q in c.barrel}, x)
                row.append(o['D'])
            L_.append(f'  {bv:2d} {av:2d} |' + ''.join(f'{v:4d}' for v in row))

    # T 的两个桶
    bT, aT = (1, 0, 15), (3, 0, 18)
    L_.append('')
    L_.append('=== 单 T：b=桶(1,0,15)（rear 侧），a=桶(3,0,18)（侧输入支路） ===')
    L_.append('   b  a |' + ''.join(f'{x:>4}' for x in range(16)))
    for bv in (0, 5, 10):
        for av in (0, 5, 10):
            row = []
            for x in range(16):
                o = run_one({q: (bv if q == bT else av if q == aT else 0) for q in c.barrel}, x)
                row.append(o['T'])
            L_.append(f'  {bv:2d} {av:2d} |' + ''.join(f'{v:4d}' for v in row))

    open(OUT_TXT, 'w', encoding='utf-8').write('\n'.join(L_))
    print('\n'.join(L_))


if __name__ == '__main__':
    main()
