import io
p = r'E:\redstone_nn\03_phase3_quantized_network\viewer\plane_viewer2.html'
s = io.open(p, encoding='utf-8').read()
s = s.replace("    cmp(a<=bb?xa:xb,zr+3,'S','compare',true);",
              "    cmp(xa,zr+3,'S','compare',true);   // 固定放在列号小的一侧（inRev 后 a/bb 已不可靠）")
io.open(p,'w',encoding='utf-8').write(s)
print('zr+3 comparator pinned to xa')
