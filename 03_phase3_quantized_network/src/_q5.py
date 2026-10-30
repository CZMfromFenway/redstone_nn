import io
p = r'E:\redstone_nn\03_phase3_quantized_network\viewer\plane_sim.html'
s = io.open(p, encoding='utf-8').read()
s = s.replace("        ctx.fillText(String(c.val), cx2=>0||X+C/2, Y+C/2-5);",
              "        ctx.fillText(String(c.val), X+C/2, Y+C/2-5);")
s = s.replace("function cx2(){ return 0; }\n", "")
io.open(p,'w',encoding='utf-8').write(s)
print('fixed')
