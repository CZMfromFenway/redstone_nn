import io
import shutil

ROOT = r"E:\redstone_nn\03_phase3_quantized_network"
ARCH = ROOT + r"\viewer\archive\plane_sim_pre_v1.html"
OUT = ROOT + r"\viewer\plane_sim.html"

shutil.copyfile(ARCH, OUT)
s = io.open(OUT, encoding="utf-8").read()

# 在 cA/cB 之后插入"每列对应的输入"
old = """    const cA=CO(i0), cB=CO(i1);                 // 两个 MSB 块的实际列
    const xa=Math.min(cA,cB), xb=Math.max(cA,cB);"""
new = """    const cA=CO(i0), cB=CO(i1);                 // 两个 MSB 块的实际列
    const xa=Math.min(cA,cB), xb=Math.max(cA,cB);
    // 「每一列需要多少信号」：列 xa 属于输入 iAtA，列 xb 属于 iAtB
    const iAtA = (cA<=cB) ? i0 : i1;
    const iAtB = (cA<=cB) ? i1 : i0;"""
assert old in s
s = s.replace(old, new)

# sa/sb 改为按列取值
old2 = """    const sa=S[i0][b-1], sb=S[i1][b-1];"""
new2 = """    const sa=S[iAtA][b-1], sb=S[iAtB][b-1];     // 跟着列，不跟着输入序号"""
assert old2 in s
s = s.replace(old2, new2)

# wa/wb 同上（决定情形）
old3 = """    const wa=Math.abs(WS[i0][b-1]), wb=Math.abs(WS[i1][b-1]);   // i0 对应 cA"""
new3 = """    const wa=Math.abs(WS[iAtA][b-1]), wb=Math.abs(WS[iAtB][b-1]);"""
assert old3 in s
s = s.replace(old3, new3)

# 桶的 ik 用列对应的输入
s = s.replace("      bar(xa,zr+1,sa,WS[i0][b-1],[i0,b-1]);\n      bar(xb,zr+1,sb,WS[i1][b-1],[i1,b-1]);",
              "      bar(xa,zr+1,sa,WS[iAtA][b-1],[iAtA,b-1]);\n      bar(xb,zr+1,sb,WS[iAtB][b-1],[iAtB,b-1]);")
s = s.replace("      bar(xa,zr+1,sLeft ,WS[iLeft ][b-1],[iLeft ,b-1]);\n      bar(xb,zr+1,sRight,WS[iRight][b-1],[iRight,b-1]);",
              "      bar(xa,zr+1,sa,WS[iAtA][b-1],[iAtA,b-1]);\n      bar(xb,zr+1,sb,WS[iAtB][b-1],[iAtB,b-1]);")
s = s.replace("      cmp(xa,zr,'S','compare',sLeft);\n      cmp(xb,zr,'S','compare',sRight);",
              "      cmp(xa,zr,'S','compare',sa);\n      cmp(xb,zr,'S','compare',sb);")

# bigAtA：哪一列是大值（按列上的值）
s = s.replace("    const bigAtA = (sLeft >= sRight);   // 左列是否为大值",
              "    const bigAtA = (sa >= sb);          // 左列（xa）是否为大值")
s = s.replace("    const bigAtA = wa>=wb;", "    const bigAtA = (sa >= sb);          // 左列（xa）是否为大值")

io.open(OUT, "w", encoding="utf-8").write(s)
print("已恢复 pre_v1 并改为「数值跟着列」")
