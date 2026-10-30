# D/T Minecraft 原生网络研究档案（截至 2026-09-27）

## 建议阅读顺序

1. **`00_阶段性总报告.md`**：唯一建议从头读的文档。它已经重新解释所有关键名词、公式、实验和结论。[[DT_Minecraft_Network_阶段性总报告_2026-09-27]]
2. `source_notes/01_D_T结构数学研究总结_修正版.md`：最初数学底座，保留作原始口径对照。
3. `source_notes/02_MC原生DT神经网络研究背景.md`：早期工程假设与启动顺序。[[02_MC原生DT神经网络研究背景]]
4. `source_notes/03_用户9.27阶段性复盘提纲.md`：本次重构报告的用户纲要。[[Interim Review and Further Questions 9.27]]
5. `data/`：关键精确枚举与实验汇总数据。
6. `tools/`：函数计算和真值表查找工具。
7. `reports_history/`：v0.2～v0.8 阶段报告，仅用于追踪研究过程；不要把它们当最新口径。

## 术语更新

旧文件中的 `Canonical-T` 在最新总报告中统一改称 **“异模式 T（mixed-mode T）”**：T 的两个被动支路一条为 c、一条为 s。`Full-T` 改称 **“全状态 T”**。

## 快速查一个二值函数

Windows 双击/命令行可使用 `tools/dt_lookup.bat`，或者：

```bash
python tools/dt_lookup.py lookup --bits 0000000011111111 --mode mixed
```

输入的 16 位从左到右依次对应 `x=0,1,...,15`。

## 给定参数直接计算

```bash
python tools/dt_lookup.py eval --chain "D(S,c,5,9);T(C,s,c,12,7)" --threshold gt:7
```

链条按信号流书写。

## 当前工具限制

深度 ≤3 的二值编译数据库可以返回真实参数 witness。深度 4 已知完整二值函数谱和 K_full，但档案中尚未保存 65,100 个函数的完整参数 witness 数据库，因此对部分四层目标只能给“存在性保证”，不能立即返回参数。
