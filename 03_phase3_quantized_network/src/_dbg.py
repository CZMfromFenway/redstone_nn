import gzip, struct, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import importlib.util
spec = importlib.util.spec_from_file_location("ls", r"E:\redstone_nn\03_phase3_quantized_network\src\learn_schem.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
raw = gzip.decompress(open(m.P, "rb").read())
r = m.R(raw); r.u1(); r.s()
root = r.payload(10)
print("root keys:", list(root.keys()))
sch = root.get("Schematic")
print("Schematic keys:", list(sch.keys()) if isinstance(sch, dict) else type(sch))
for k,v in (sch or {}).items():
    print(f"  {k}: {type(v).__name__}", (len(v) if hasattr(v,'__len__') else v))
