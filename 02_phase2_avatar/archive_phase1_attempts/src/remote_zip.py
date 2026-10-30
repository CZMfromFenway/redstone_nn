#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
远程读取 zip：只下载中央目录，再按需 Range 取少量文件
========================================================
原理: zip 的中央目录在文件末尾（End of Central Directory），
      先 Range 取尾部定位它，再取目录，最后按需取单个文件。
"""
import urllib.request, struct, os, json, time, io, zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
URL = ("https://hf-mirror.com/datasets/MihaiPopa-1/minecraft-skins-1.1m-deduped-64x64"
       "/resolve/main/minecraft_skins_64x64.zip")
UA = {'User-Agent': 'Mozilla/5.0'}


def rng(a, b):
    req = urllib.request.Request(URL, headers={**UA, 'Range': f'bytes={a}-{b}'})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def size():
    req = urllib.request.Request(URL, headers={**UA, 'Range': 'bytes=0-0'})
    with urllib.request.urlopen(req, timeout=60) as r:
        cr = r.headers.get('Content-Range', '')
    return int(cr.split('/')[-1])


print("=" * 84)
t0 = time.time()
N = size()
print(f"zip 总大小: {N} bytes ({N/2**30:.2f} GiB)   [{time.time()-t0:.1f}s]")

# 找 EOCD (End of Central Directory)，签名 PK\x05\x06
tail_len = 200000
tail = rng(max(0, N - tail_len), N - 1)
i = tail.rfind(b'PK\x05\x06')
print(f"EOCD 位置: 尾部偏移 {i}  (即文件偏移 {N-tail_len+i})")
eocd = tail[i:i + 22]
_, disk, cd_disk, n_this, n_total, cd_size, cd_off, cmt = struct.unpack('<IHHHHIIH', eocd)
print(f"条目总数: {n_total}")
print(f"中央目录: offset={cd_off}  size={cd_size} ({cd_size/2**20:.1f} MiB)")

t0 = time.time()
cd = rng(cd_off, cd_off + cd_size - 1)
print(f"下载中央目录: {len(cd)} bytes  [{time.time()-t0:.1f}s]")

# 解析中央目录条目
entries = []
p = 0
while p + 46 <= len(cd) and cd[p:p + 4] == b'PK\x01\x02':
    (ver, vneed, flag, method, mt, md, crc, csz, usz, nlen, elen, clen,
     dsk, iattr, eattr, lho) = struct.unpack('<HHHHHHIIIHHHHHII', cd[p + 4:p + 46])
    name = cd[p + 46:p + 46 + nlen].decode('utf-8', 'replace')
    entries.append({'name': name, 'method': method, 'csz': csz, 'usz': usz,
                    'crc': crc, 'lho': lho, 'nlen': nlen, 'elen': elen})
    p += 46 + nlen + elen + clen
print(f"解析出条目: {len(entries)}")
print("前 5 个:")
for e in entries[:5]:
    print("   ", e['name'], e['csz'], e['method'])
exts = {}
for e in entries:
    x = os.path.splitext(e['name'])[1].lower()
    exts[x] = exts.get(x, 0) + 1
print("扩展名分布:", sorted(exts.items(), key=lambda kv: -kv[1])[:8])

json.dump(entries[:200000], open(os.path.join(DATA, 'zip_index.json'), 'w'),
          ensure_ascii=False)
print(f"\n已保存前 {min(len(entries),200000)} 条索引 -> data/zip_index.json")
