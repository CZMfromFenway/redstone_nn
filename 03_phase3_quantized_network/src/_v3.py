import io

p = r'E:\redstone_nn\03_phase3_quantized_network\viewer\plane_sim.html'
s = io.open(p, encoding='utf-8').read()

# ---- 在循环前建立"列顺序表"：每列属于哪个输入 ----
old_head = "  const pairInfo = [];\n  for (let p2 = 0; p2 < Math.floor(n / 2); p2++) {"
alt_head = "  const pairInfo=[];\n  for(let p=0;p<Math.floor(n/2);p++){"
if old_head in s:
    head_old = old_head
elif alt_head in s:
    head_old = alt_head
else:
    raise SystemExit("找不到配对循环的开头")

new_head = """  // ---- 按「列」配对：先排出每列属于哪个输入（自动兼容 inRev / parSwap） ----
  const COLS = [];
  for (let i = 0; i < n; i++) COLS.push({ col: CO(XI(i)), i });
  COLS.sort((A, B) => A.col - B.col);

  const pairInfo = [];
  for (let p = 0; p < Math.floor(n / 2); p++) {
    const xa = COLS[2 * p].col, xb = COLS[2 * p + 1].col;
    const iAtA = COLS[2 * p].i, iAtB = COLS[2 * p + 1].i;"""
s = s.replace(head_old, new_head, 1)

# ---- 删掉旧的 i0/i1/cA/cB/ixa 计算，改用列表 ----
old_body = """    const i0=XI(2*p), i1=XI(2*p+1);
    const cA=CO(i0), cB=CO(i1);                 // 两个 MSB 块的实际列
    const xa=Math.min(cA,cB), xb=Math.max(cA,cB);
    // 「每一列需要多少信号」：列 xa 属于输入 iAtA，列 xb 属于 iAtB
    const iAtA = (cA<=cB) ? i0 : i1;
    const iAtB = (cA<=cB) ? i1 : i0;
    const wa=Math.abs(WS[iAtA][b-1]), wb=Math.abs(WS[iAtB][b-1]);
    const sa=S[iAtA][b-1], sb=S[iAtB][b-1];     // 跟着列，不跟着输入序号"""
if old_body not in s:
    # 可能顺序略有差异，做一次更宽松的定位
    import re
    m = re.search(r"\n    const i0=XI\(2\*p\).*?const sa=S\[iAtA\]\[b-1\], sb=S\[iAtB\]\[b-1\];[^\n]*", s, re.S)
    if not m:
        raise SystemExit("找不到旧的主体")
    old_body = m.group(0)[1:]

new_body = """    const i0 = iAtA, i1 = iAtB;                 // 兼容旧变量名
    const sa=S[iAtA][b-1], sb=S[iAtB][b-1];     // 两列各自的信号
    const wa=Math.abs(WS[iAtA][b-1]), wb=Math.abs(WS[iAtB][b-1]);"""
s = s.replace(old_body, new_body, 1)

io.open(p, 'w', encoding='utf-8').write(s)
print("配对改为按列进行（列表 COLS）")
