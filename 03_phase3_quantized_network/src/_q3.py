import io
p = r'E:\redstone_nn\03_phase3_quantized_network\viewer\plane_viewer2.html'
s = io.open(p, encoding='utf-8').read()
old = """    cmp(xa+1,zr+4,'E','compare',true);
    cmp(xa,zr+5,'N','compare',true); cmp(xb,zr+5,'S','compare',true);
    cmp(xa+1,zr+6,'W','compare',true);"""
new = """    // 这一对的 Sa 环是烧录的 => 4 个比较器是源
    const R2=[];
    cmp(xa+1,zr+4,'E','compare',true); R2.push(m.get(K2(xa+1,zr+4)));
    cmp(xa,zr+5,'N','compare',true); cmp(xb,zr+5,'S','compare',true);
    R2.push(m.get(K2(xa,zr+5)), m.get(K2(xb,zr+5)));
    cmp(xa+1,zr+6,'W','compare',true); R2.push(m.get(K2(xa+1,zr+6)));
    R2.forEach(c=>{ if(c) c.onRing=true; });"""
assert old in s
s = s.replace(old, new)
io.open(p,'w',encoding='utf-8').write(s)
print('pairing rings marked as sources')
