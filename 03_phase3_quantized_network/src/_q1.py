import io
p = r'E:\redstone_nn\03_phase3_quantized_network\viewer\plane_viewer2.html'
s = io.open(p, encoding='utf-8').read()
s = s.replace('<label><span>位权朝次方向降低</span><input type="checkbox" id="bitRev" checked></label>',
              '<label><span>位权朝次方向升高</span><input type="checkbox" id="bitRev"></label>')
io.open(p,'w',encoding='utf-8').write(s)
print('renamed to 升高, default unchecked')
