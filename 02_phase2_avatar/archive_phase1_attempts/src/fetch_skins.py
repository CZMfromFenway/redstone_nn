#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从远程 zip 按需提取 N 张皮肤
==============================
已拿到中央目录（110 万条）。现在:
  对每个选中条目，Range 取本地文件头(30 字节 + nlen + elen) + 压缩数据，解压。
每张皮肤约 1~5 KB，2000 张约 5 MB。
"""
import urllib.request, struct, os, json, time, zlib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
URL = ("https://hf-mirror.com/datasets/MihaiPopa-1/minecraft-skins-1.1m-deduped-64x64"
       "/resolve/main/minecraft_skins_64x64.zip")
UA = {'User-Agent': 'Mozilla/5.0'}


def rng(a, b):
    req = urllib.request.Request(URL, headers={**UA, 'Range': f'bytes={a}-{b}'})
    with urllib.request.urlopen(req, timeout=180) as r:
        return r.read()


def full_index():
    N = 2714488499
    cd_off, cd_size = 2642232996, 72255405
    cd = rng(cd_off, cd_off + cd_size - 1)
    ents = []
    p = 0
    while p + 46 <= len(cd) and cd[p:p + 4] == b'PK\x01\x02':
        (ver, vneed, flag, method, mt, md, crc, csz, usz, nlen, elen, clen,
         dsk, iattr, eattr, lho) = struct.unpack('<HHHHHHIIIHHHHHII', cd[p + 4:p + 46])
        name = cd[p + 46:p + 46 + nlen].decode('utf-8', 'replace')
        ents.append((name, method, csz, usz, lho, nlen, elen))
        p += 46 + nlen + elen + clen
    return ents


def extract(e):
    name, method, csz, usz, lho, nlen, elen = e
    hdr = rng(lho, lho + 30 + nlen + elen - 1)
    if hdr[:4] != b'PK\x03\x04':
        raise ValueError('bad LFH')
    data = rng(lho + 30 + nlen + elen, lho + 30 + nlen + elen + csz - 1)
    if method == 0:
        return data
    return zlib.decompress(data, -15)


def main():
    print("=" * 88)
    print("从远程 zip 提取皮肤样本")
    print("=" * 88)
    t0 = time.time()
    ents = full_index()
    print(f"索引: {len(ents)} 条  [{time.time()-t0:.0f}s]")

    rng_ = np.random.default_rng(0)
    want = 2000
    sel = rng_.choice(len(ents), want, replace=False)
    sel.sort()

    outdir = os.path.join(DATA, "skins")
    os.makedirs(outdir, exist_ok=True)
    ok, fail, bytes_total = 0, 0, 0
    t0 = time.time()
    for i, k in enumerate(sel):
        e = ents[k]
        try:
            d = extract(e)
            with open(os.path.join(outdir, e[0]), 'wb') as f:
                f.write(d)
            ok += 1
            bytes_total += len(d)
        except Exception as ex:
            fail += 1
        if (i + 1) % 200 == 0:
            el = time.time() - t0
            print(f"   {i+1}/{want}  成功 {ok} 失败 {fail}  "
                  f"{bytes_total/1024:.0f} KB  {el:.0f}s  ({bytes_total/1024/max(el,1e-9):.1f} KB/s)",
                  flush=True)
    print(f"\n完成: 成功 {ok} 失败 {fail}  共 {bytes_total/1e6:.2f} MB  "
          f"用时 {time.time()-t0:.0f}s")
    print(f"保存到: {outdir}")


if __name__ == '__main__':
    main()
