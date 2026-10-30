#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
并行下载大量皮肤（多线程 Range 请求）
=======================================
之前串行 0.7 KB/s（每张 2 次请求 x ~1.5s 延迟）。
改为 32 线程并发 -> 预期 20~50 张/秒。
只用中央目录 + Range，流量仍然很小（每张 1~5 KB）。
"""
import urllib.request, struct, os, time, zlib, json, threading
import numpy as np
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
URL = ("https://hf-mirror.com/datasets/MihaiPopa-1/minecraft-skins-1.1m-deduped-64x64"
       "/resolve/main/minecraft_skins_64x64.zip")
CD_OFF, CD_SIZE = 2642232996, 72255405
UA = {'User-Agent': 'Mozilla/5.0'}


def rng_req(a, b, retry=4):
    last = None
    for k in range(retry):
        try:
            req = urllib.request.Request(URL, headers={**UA, 'Range': f'bytes={a}-{b}'})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception as e:
            last = e
            time.sleep(0.5 * (k + 1))
    raise last


def load_index():
    cd = rng_req(CD_OFF, CD_OFF + CD_SIZE - 1)
    ents = []; p = 0
    while p + 46 <= len(cd) and cd[p:p + 4] == b'PK\x01\x02':
        (ver, vn, fl, me, mt, md, crc, csz, usz, nl, el, cl,
         dk, ia, ea, lho) = struct.unpack('<HHHHHHIIIHHHHHII', cd[p + 4:p + 46])
        ents.append((cd[p + 46:p + 46 + nl].decode('utf-8', 'replace'), me, csz, usz, lho, nl, el))
        p += 46 + nl + el + cl
    return ents


def fetch_one(e):
    name, me, csz, usz, lho, nl, el = e
    hdr = rng_req(lho, lho + 30 + nl + el - 1)
    if hdr[:4] != b'PK\x03\x04':
        raise ValueError('bad LFH')
    off = lho + 30 + nl + el
    data = rng_req(off, off + csz - 1)
    return name, (data if me == 0 else zlib.decompress(data, -15))


def main():
    TARGET = int(os.environ.get('SKIN_TARGET', '12000'))
    WORKERS = 32
    print("=" * 88)
    print(f"并行下载皮肤  目标 {TARGET} 张  并发 {WORKERS}")
    print("=" * 88)
    t0 = time.time()
    ents = load_index()
    print(f"索引: {len(ents)} 条  [{time.time()-t0:.0f}s]")

    outdir = os.path.join(DATA, "skins")
    os.makedirs(outdir, exist_ok=True)
    have = set(os.listdir(outdir))
    rng = np.random.default_rng(7)
    cand = rng.choice(len(ents), min(len(ents), TARGET * 3), replace=False)
    todo = [ents[i] for i in cand if ents[i][0] not in have][:TARGET]
    print(f"待下载: {len(todo)}  已有: {len(have)}")

    lock = threading.Lock()
    stat = {'ok': 0, 'fail': 0, 'bytes': 0}
    t0 = time.time()

    def work(e):
        try:
            name, data = fetch_one(e)
            with open(os.path.join(outdir, name), 'wb') as f:
                f.write(data)
            with lock:
                stat['ok'] += 1; stat['bytes'] += len(data)
                if stat['ok'] % 500 == 0:
                    el = time.time() - t0
                    print(f"   {stat['ok']}/{len(todo)}  失败 {stat['fail']}  "
                          f"{stat['bytes']/1e6:.1f} MB  {el:.0f}s  "
                          f"({stat['ok']/max(el,1e-9):.1f} 张/秒)", flush=True)
        except Exception:
            with lock:
                stat['fail'] += 1

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        list(ex.map(work, todo))

    el = time.time() - t0
    print(f"\n完成: 成功 {stat['ok']}  失败 {stat['fail']}  {stat['bytes']/1e6:.1f} MB  "
          f"用时 {el:.0f}s  ({stat['ok']/max(el,1e-9):.1f} 张/秒)")
    total = len([f for f in os.listdir(outdir) if f.endswith('.png')])
    print(f"皮肤总数: {total}")


if __name__ == '__main__':
    main()
