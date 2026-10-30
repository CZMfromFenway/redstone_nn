import io

p = r'E:\redstone_nn\03_phase3_quantized_network\viewer\plane_sim.html'
s = io.open(p, encoding='utf-8').read()

start = s.index("  // 配对区（按数值定情形）")
end = s.index("  // 奇数 n：最后一个输入单独")

new = r'''  // ================= 配对区 =================
  //  先算「每一列需要多少信号」，再据此选结构：
  //    每列 x 的需求  need[x] = { s, w, i }
  //    梯子统一三级： (x,2)MSB ← (x,3)N ← (x,4) ← (x,5)N ← (x,6)源
  //    情形Ⅲ 两列各一个直通桶放在 (x,4)，省掉 (x,5)
  //    情形Ⅱ 大值列由环角块 (x,6) 供；小值列桶放在 (x,4)
  //    情形Ⅰ 大值列同上；小值列由减法 (x,5) 供，Δ桶在 (cM,5)
  const pairInfo = [];
  for (let p2 = 0; p2 < Math.floor(n / 2); p2++) {
    const iA = XI(2 * p2), iB = XI(2 * p2 + 1);
    const colA = CO(iA), colB = CO(iB);
    const sa = S[iA][b - 1], sb = S[iB][b - 1];
    const wa = Math.abs(WS[iA][b - 1]), wb = Math.abs(WS[iB][b - 1]);

    const cL = Math.min(colA, colB), cR = Math.max(colA, colB), cM = cL + 1;
    // 哪一列的信号大
    const aIsBig = sa >= sb;
    const xBig = aIsBig ? colA : colB;
    const xSml = aIsBig ? colB : colA;
    const iBig = aIsBig ? iA : iB, iSml = aIsBig ? iB : iA;
    const sBig = aIsBig ? sa : sb, sSml = aIsBig ? sb : sa;
    const wBig = aIsBig ? wa : wb, wSml = aIsBig ? wb : wa;
    const cs = (wSml > 882) ? 'I' : (wBig > 882 ? 'II' : 'III');

    pairInfo.push({ cL, cR, cs, sa, sb, xBig, xSml, iA, iB });
    const zr = Zmsb + 1;                     // 3

    if (cs === 'III') {
      // 两列各一个直通桶：(x,4) 是桶，(x,3) 读它并充能 MSB
      bar(colA, zr + 1, sa, WS[iA][b - 1], [iA, b - 1]);
      bar(colB, zr + 1, sb, WS[iB][b - 1], [iB, b - 1]);
      cmp(colA, zr, 'S', 'compare', sa);
      cmp(colB, zr, 'S', 'compare', sb);
      continue;
    }

    // ---- 环：3 列 × 3 行，行 r0..r0+2，四个角块在 cL / cR
    const r0 = zr + 3;                       // 6
    blk(cL, r0, 'wire');     blk(cR, r0, 'wire');
    blk(cL, r0 + 2, 'wire'); blk(cR, r0 + 2, 'wire');
    cmp(cM, r0, 'E', 'compare', sBig, true);        // 顶中点，输出 W → (cL,r0)
    cmp(cR, r0 + 1, 'S', 'compare', sBig, true);    // 右中，输出 N → (cR,r0)
    cmp(cM, r0 + 2, 'W', 'compare', sBig, true);    // 底中点，输出 E → (cR,r0+2)
    cmp(cL, r0 + 1, 'N', 'compare', sBig, true);    // 左中，输出 S → (cL,r0+2)

    // ---- 大值列：两级梯子，底部读环角块 (xBig,6)
    blk(xBig, zr + 1, 'wire');                      // (xBig,4)
    cmp(xBig, zr + 2, 'S', 'compare', sBig);        // (xBig,5) 读 (xBig,6)
    cmp(xBig, zr, 'S', 'compare', sBig);            // (xBig,3) 读 (xBig,4)

    // ---- 小值列
    if (cs === 'II') {
      bar(xSml, zr + 1, sSml, WS[iSml][b - 1], [iSml, b - 1]);  // 桶直接在 (xSml,4)
      cmp(xSml, zr, 'S', 'compare', sSml);
    } else {
      bar(cM, zr + 2, sBig - sSml, null, null);     // Δ桶在 (cM,5)
      blk(xSml, zr + 1, 'wire');                    // (xSml,4)
      cmp(xSml, zr + 2, 'S', 'subtract', sBig);     // 减法：(xSml,6) − Δ
      cmp(xSml, zr, 'S', 'compare', sSml);
    }
  }

'''
s = s[:start] + new + s[end:]
io.open(p, 'w', encoding='utf-8').write(s)
print('配对区：先算每列需求，再选结构')
