import io
p = r'E:\redstone_nn\03_phase3_quantized_network\viewer\plane_viewer2.html'
s = io.open(p, encoding='utf-8').read()
# 配对区/孤立环的导线块改用 'wire'
s = s.replace("[2,4,6].forEach(dz=>{ blk(xa,zr+dz,'col7',true); blk(xb,zr+dz,'col7',true); });",
              "[2,4,6].forEach(dz=>{ blk(xa,zr+dz,'wire',true); blk(xb,zr+dz,'wire',true); });")
s = s.replace("[[x,zr+4],[x+2,zr+4],[x,zr+2],[x,zr+6],[x+2,zr+6]].forEach(([px,pz])=>blk(px,pz,'col7',true));",
              "[[x,zr+4],[x+2,zr+4],[x,zr+2],[x,zr+6],[x+2,zr+6]].forEach(([px,pz])=>blk(px,pz,'wire',true));")
# 渲染：wire 画成品红
s = s.replace("      else if(c.c==='white') col='#c8c8c8';",
              "      else if(c.c==='white') col='#c8c8c8';\n      else if(c.c==='wire') col='rgb(215,80,200)';")
io.open(p,'w',encoding='utf-8').write(s)
print('wire type separated')
