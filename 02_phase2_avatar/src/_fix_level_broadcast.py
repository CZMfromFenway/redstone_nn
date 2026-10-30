import io
import re
import sys

# The 16 levels are EVENLY SPACED (linspace(0,255,16), step 17), so the nearest level is
# just round(value/17) clipped to 0..15. The broadcast version materialises an
# (N,8,8,16) float64 temporary -- 7.05 GiB for the 921k-face dataset -- which blows up as
# soon as several training processes run at once. Verified identical on random input.
PATTERN = re.compile(
    r"np\.abs\(\s*(\w+)\[:,\s*:,\s*:,\s*None\]\s*-\s*LEVELS\[None,\s*None,\s*None,\s*:\]\s*\)\.argmin\(-1\)"
)
REPL = r"np.clip(np.rint(\1 / 17.0), 0, 15).astype(np.int64)"

files = sys.argv[1:]
for p in files:
    s = io.open(p, encoding="utf-8").read()
    n = len(PATTERN.findall(s))
    if not n:
        print(f"  {p}: no match")
        continue
    s2 = PATTERN.sub(REPL, s)
    # the original ended with .astype(np.uint8) in a few files; int64 then uint8 is fine
    io.open(p, "w", encoding="utf-8").write(s2)
    import ast
    ast.parse(s2)
    print(f"  {p}: replaced {n} site(s)")
