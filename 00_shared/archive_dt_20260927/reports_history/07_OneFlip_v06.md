# 1-Flip Smooth Count：训练算法实验 v0.6

## 结构

每条边限制为仅一次二值跳变，连续训练时使用陡峭 sigmoid：

- 上升边：`sigmoid(beta*(x-t))`
- 下降边：`1-sigmoid(beta*(x-t))`

聚合后 G 限制为两个减法模式的 D：

`D_SS(x;L,R)=max(0, L-x, x-R)`

等价地（令 `c=(L+R)/2, r=(R-L)/2`）：

`ReLU(|x-c|-r)`。

边阈值 `t` 的可编译位置为半整数；D 的物理阈值 `L,R` 为整数。

## 离散正则

- 边阈值半整数吸附：`cos^2(pi*t)`（半整数处为0）
- D 阈值整数吸附：`sin^2(pi*L)+sin^2(pi*R)`
- 方向二值化：`p(1-p)`

固定弱正则会产生严重 soft→hard 投影差。XOR 上出现 `soft=0错, hard=64错`。

将上述正则在训练后半段退火增强后，3个不同随机种子的 XOR 均实现 `soft=0错, hard=0错`，证明用户提出的“格点吸附正则”方向有效。

## 4输入 Pilot

### MLP Teacher N4_MLP_T1

- OneFlip anneal h=8：hard 10055/65536 = 15.34%，soft 10.61%
- OneFlip anneal h=12：hard 22.21%，soft 8.09%
- OneFlip anneal h=15：hard 34.05%，soft 7.83%
- 既有函数级 Count-K4：h=8 8.83%，h=12 8.02%，h=15 7.23%

连续模型的 soft 能力随宽度提升，但 hard 结果反而恶化，表明量化/投影仍是主要训练瓶颈。

### Count Teacher N4_COUNT_T1

- OneFlip anneal h=8：40.41%
- soft-hard consistency h=8：hard 35.85%，soft 31.21%
- OneFlip anneal h=15：40.41%
- 既有函数级 Count-K4：h=8 13.68%，h=12 2.75%，h=15 4.38%

这说明单边仅一次跳变大幅降低了单边表达力；单纯增加宽度到15并未在当前训练方法下追回高频 Count-native Teacher。

## 当前判断

1. **该架构不是失败。** 在低维任务上已验证可微训练+退火离散化能得到完全可编译的精确解。
2. **但目前不能替代函数级 Count 主线。** 4输入 Teacher 上，同宽度明显更弱，尤其对 Count-native 高频函数。
3. **最大问题是 soft→hard mismatch。** MLP Teacher 上 soft h=12/15 已接近函数级 Count 的误差，但硬化后明显退化。
4. **从头使用硬量化 STE 不理想。** Pilot 中 XOR 的 QAT/STE 版本比“先 soft 后 hard”明显更难训练，符合量化网络中常见的训练现象。
5. 下一步应重点测试：soft-then-hard / alpha-blending、逐层量化、量化感知蒸馏、以及网络生长，而不是继续简单扩大 h。

## 物理成本

目前尚未对“一次跳变边”的实际 Minecraft 单元给出可靠尺寸，因此本报告没有擅自给它估体积。若后续确认一条 1-Flip 边的实际比较器/截面/长度成本，就可以与 DT(K<=4) 的 2层矩阵做正式体积—训练—能力 Pareto 对比。
