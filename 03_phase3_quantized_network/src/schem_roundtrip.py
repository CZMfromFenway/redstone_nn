# -*- coding: utf-8 -*-
"""用参考文件同一套读法，读回我生成的文件，确认结构一致。"""
import gzip
import importlib.util
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

spec = importlib.util.spec_from_file_location(
    "ls", r"E:\redstone_nn\03_phase3_quantized_network\src\learn_schem.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def load(path):
    raw = gzip.decompress(open(path, "rb").read())
    r = m.R(raw)
    assert r.u1() == 10, "根不是 compound"
    r.s()                      # 根名应为空
    root = r.payload(10)
    return raw, root


base = r"E:\redstone_nn\03_phase3_quantized_network"
for tag, path in (("参考一期", base + r"\data\hand_write_digital.schem"),
                  ("C0", base + r"\results\schem\layer1_ch0_C0_全0.schem"),
                  ("C1", base + r"\results\schem\layer1_ch0_C1_全1.schem"),
                  ("C2", base + r"\results\schem\layer1_ch0_C2_1010.schem"),
                  ("C3", base + r"\results\schem\layer1_ch0_C3_0101.schem")):
    try:
        raw, root = load(path)
        sch = root["Schematic"]
        keys = sorted(sch.keys())
        W, H, L = sch["Width"], sch["Height"], sch["Length"]
        blk = sch["Blocks"]
        bkeys = sorted(blk.keys())
        idx = m.expand_varints(blk["Data"])
        nbe = len(blk.get("BlockEntities", []))
        ok = len(idx) == W * H * L
        print(f"{tag:6s} 根键={list(root.keys())} 尺寸 {W}×{H}×{L} "
              f"格数 {len(idx)}/{W*H*L} {'✓' if ok else '✗'} "
              f"BlockEntities {nbe}  Blocks 键={bkeys}")
    except Exception as e:
        print(f"{tag:6s} ✗ 解析失败: {e}")
