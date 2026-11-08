# redstone_nn —— 红石神经网络研究仓库

**多期工程仓库** · 目标：用红石电路实现神经网络的计算能力

> **⚠️ 全部已完成的实物都是「强模」网络 —— 生存模式下无法实装。**
> 完全态的强模 LUT 在生存模式无论如何实现不了。
> 这些工作的价值在于「**证明这也能算**」，而不在于「这能用」。

---

## 仓库总览

| 期 | 名称 | 状态 | 一句话 |
|---:|---|---|---|
| **一期** | 强模 LUT → 数字识别 → 头像尝试 | **已完成**（两条线封存） | 唯一实装成果：10 层强模网络，225→10，体积 32×52×47，≤15 秒，名义准确率 87.4%（实机约 50~60%） |
| **二期** | 神经网络生成头像 | **已完成** | 灰度 4-64-32（2,304 乘法）+ 二次打分器（14 乘法），良率 66.5%，矩形夹取后 89.1%；彩色 8-128-128-32×8（50,176 乘法）实测 61.1% |
| **三期** | 量化网络与红石移植 | **进行中**（P3） | 把任意两层 MLP 编译成"整数常数 + 容器 + 投掷器"：编译规则、误差理论已定稿，层 1 已导出 `.schem` |
| **四期** | 彩色头像生成 | **主体完成**（14 轮盲测） | 4 旋钮 → 灰度 4-64-32（2,304）→ 16 色头（86,016）→ 换调色板 → 4 选 1 = **88,380 乘法 / 4 随机源**；好脸率 **65~85%**，与真实数据的 72.5% 统计上并列；**扩散路线同设置下低 30~40 点，判死**。四期不含红石侧（R-05） |

**整理过的新入口**（2026-10 建）：
[`04_phase4_colour_avatar/README.md`](04_phase4_colour_avatar/README.md) 里有两份汇总文档 ——
[沿革](04_phase4_colour_avatar/docs/01_沿革_一期至三期.md)（历史）与
[可用资产](04_phase4_colour_avatar/docs/02_可用资产与规则.md)（今天还能用的规则/工具/数据）。

**知识库**（2026-11 建）：**[`kb/README.md`](kb/README.md)**

> 文档回答"当时我们怎么想的"，知识库回答"**现在什么是真的**"。
> 每页都带状态：`定案` 优先于任何文档、任何总结、任何外部资料；`待验` 引用时必须带状态。
> 从 [`kb/README.md`](kb/README.md) 进去可以逐页点开。

---

## 目录结构

```
E:\redstone_nn\
│
├─ 00_shared\                        跨期通用资料
│   ├─ docs\                          DT 前期研究报告（2026-09/10-01）
│   ├─ data\                          共享数据
│   ├─ archive_dt_20260927\           DT 研究归档（9/27）
│   ├─ archive_dt_20261001\           DT 研究归档（10/1）
│   └─ reference_cohomologo\          辰占鳌头（cohomologo）的资料
│
├─ 01_phase1_digit_recognition\       一期：强模 LUT + 数字识别 + 头像尝试（已完成）
│   ├─ docs\                          18 篇（含 20_工程总结报告）
│   ├─ archive\docs\                  强模 LUT 数学解析的四版修订
│   ├─ src\                           115 个脚本
│   ├─ results\                       权重、热图、图表
│   ├─ build\draw.html                ▶ 画板调试器（双击即用，内嵌权重）
│   ├─ datapack\tiny_cnn\             数据包（898 个桶 + 2340 条放置指令）
│   ├─ data_mnist\                    MNIST 原始数据
│   └─ obsidian\                      Obsidian 库配置
│
├─ 02_phase2_avatar\                  二期：生成头像（已完成）
│   ├─ docs\                          28 篇
│   ├─ PHASE2_TIMELINE.md             ▶ **二期的第一读**（含勘误附录）
│   ├─ SUMMARY.md                     灰度 + 彩色合并总结
│   ├─ RESEARCH_ROUTE.md              路线复盘
│   ├─ src\                           252 个脚本
│   ├─ webapp\                        ▶ 全部模型 + sandbox 单文件交互台
│   ├─ results\                       实验产物（DOC_INDEX.md 在 results\ 下）
│   └─ archive_phase1_attempts\       一期头像九方案的完整记录（6006 文件）
│
├─ 03_phase3_quantized_network\       三期：量化网络与红石移植（进行中）
│   ├─ docs\                          47 篇（00~39 已冻结，40~46 仍在动）
│   ├─ src\                           166 个脚本（含 node 求值器与校验）
│   ├─ viewer\plane_sim.html          ▶ 单层建板 + 求值器（1536/1536 全通）
│   ├─ results\schem\                 .schem 实物（层1 单片/堆叠/层2）
│   └─ data\                          参考 .schem
│
├─ 04_phase4_colour_avatar\           四期：彩色头像生成（主体完成，见 docs/13_四期总结报告）
│   ├─ docs\01_沿革_一期至三期.md
│   ├─ docs\02_可用资产与规则.md
│   ├─ docs\00_文档索引.md
│   └─ _work\                         整理过程的中间摘要
│
├─ kb\                                 **知识库**（结论层，见 kb\README.md）
│   ├─ 00_入口\                        状态与口径、冲突登记、更新记录
│   ├─ 10_游戏机制\                     红石底层（容器、比较器、投掷器、地图画…）
│   ├─ 20_数学\                        从红石到数学（量化、激活、显示、随机源…）
│   ├─ 95_训练记录\                     模型训练的过程留档（地位低于定案页）
│   ├─ 98_核对\                        核对日志（公式 vs 实现的逐项比对）
│   └─ figures\                        配图
│
├─ 99_models\                         跨期模型仓库（一期整数权重 npz/csv）
├─ 99_tools\                          跨期工具（一期画板副本）
├─ 99_mc_texture\                     MC 1.21.11 Fabric jar（取贴图用）
└─ README.md                          本文件
```

---

## 快速入口

### 想了解整个项目做了什么

| 顺序 | 文件 | 内容 |
|---:|---|---|
| 1 | `04_phase4_colour_avatar/docs/01_沿革_一期至三期.md` | **按时间的完整沿革**：四次改道、每期拿到了什么、什么被推翻了 |
| 2 | `04_phase4_colour_avatar/docs/02_可用资产与规则.md` | **今天还能用的**：物理规则、编译规则、误差理论、工具、数据、缺口 |

### 想了解一期

| 顺序 | 文件 | 内容 |
|---:|---|---|
| 1 | `01_phase1_digit_recognition/docs/20_工程总结报告.md` | **工程总结**：数学、权重、数据包、局限性、复现 |
| 2 | `01_phase1_digit_recognition/docs/00_文档索引.md` | 全部文档的索引与阅读顺序 |
| 3 | `01_phase1_digit_recognition/docs/05_强模LUT数学性质_定稿.md` | 强模 LUT 的完整数学（闭式解、步长、反解） |
| 4 | `01_phase1_digit_recognition/docs/07_强模LUT速查约束表.md` | 物理硬约束，设计前必看 |

### 想复现一期

| 步骤 | 做什么 |
|---|---|
| 1 | 打开 `01_phase1_digit_recognition/build/draw.html`，载入"数字 0"预设，应显示 **F=146、识别=0** |
| 2 | 跑 `01_phase1_digit_recognition/src/int_weight_model.py` 重训权重 |
| 3 | 跑 `01_phase1_digit_recognition/src/gen_all_v2.py` 生成数据包 |
| 4 | 在 MC 里 `/function deploy_weights:deploy_all_layers` |

### 想了解二期

1. 先读 `02_phase2_avatar/PHASE2_TIMELINE.md`（**它自带勘误附录 A.1，列出哪条旧结论被谁修正**）
2. 彩色看 `02_phase2_avatar/SUMMARY.md`
3. **不要**照 `02_phase2_avatar/docs/00_已知死路与重启指引.md` 的死路清单办事 —— 其中三条已被二期自己撤销，该文件顶部有指针
4. 交互台：`02_phase2_avatar/webapp/sandbox_standalone.html`（5.3 MB 单文件，双击即用）

### 想接着做三期

1. `03_phase3_quantized_network/docs/41_当前状态与待办.md` —— 交接文档，新会话第一读
2. `03_phase3_quantized_network/docs/40_单层结构定稿与schem导出.md` —— 坐标/结构/导出的定稿
3. `03_phase3_quantized_network/docs/16_通用层编译规则.md` + `30_三类误差与最优b.md` —— 方法论
4. 改结构后必须过 `node src/sweep_options.js`（必须 1536/1536）

---

## 各期核心结论速查

### 一期（强模 LUT + 数字识别）

**数字识别（唯一实装成果）：**

| 项 | 值 |
|---|---|
| 名义准确率 | 87.4%（严格协议 87.9%） |
| 体积 / 延迟 | 32×52×47 / ≤15 秒 |
| **实机真实水平** | **约 50~60%**（粗笔画 1 格即掉到 23.6%） |
| 生存模式 | **不可实装** |

**数学：**
- 容器值：偶数层 `c = 15 − W`，奇数层 `c = 15 + W`
- 支线偏置 `G(i,0) = Σ max(c,15)`；总线偏置 `F(0) = Σ G(i,max)`
- 对齐后 `F = z + 128`，10 层可直接比大小
- argmax 用**二进制余数恢复**，$O(\log n)$、固定 8 轮、通道间不互相引用

**强模 LUT（已封存，但数学仍有效）：**
- 单元固定 16 容器，**一个容器恰好控一个步长**；$\Delta\lambda=15-b_{x+1}$
- 单步长上限 882（$=897-15$）；**$\lambda$ 的上限是 $15\times897=13455$，不是 897**
- 浮动步长：$\lambda(x)=\mathrm{round}(x\rho)$，可**直接反解**，无需搜索

**最重要的教训：**
> MNIST 测试集准确率**不是**泛化性能的度量。评估必须包含输入扰动测试（平移、缩放、笔画粗细）。

### 二期（头像生成）

| | 灰度 | 彩色 |
|---|---|---|
| 结构 | 4 → 64 → 32 | 8 → 128 → 128 → 32×8 |
| 乘法 | **2,304** | **50,176** |
| 每像素 | 4 bit（11 档地图灰） | 3 bit（8 色调色板） |
| 可接受率 | 66.5%（夹取后 **89.1%**） | **61.1%** |

**三条硬约束**（违反必崩）：不用 MSE/L1；不逐像素独立；颜色当离散类别。

**良率结构**：好区是一个连通块（66.5%），球状限制无效，**矩形夹取**才能把良率提到 89.1%。

**空前的教训**：三次"指标骗人"——循环评估报 100%、结构损失预测 95.1%、对抗报"边缘+19%"，实测均远低。

### 三期（量化与移植）

- 架构：**数电输入 → 强模编码 → 强模求和 → 强模转数电输出 → 数电传输**（消掉"强模只能水平传"）
- 编译：七步 + 三开关，`compiler.py` 与手算**11 个量 × 2 种 $w_{\text{ref}}$，最大差全为 0**
- 缩放由容器上限反解，不是自由参数：$\sigma=w_{\text{ref}}/(\max|w|\,\delta\,2^{b-2})$
- 误差：$\epsilon_x\propto2^{-b}$、$\epsilon_w\propto\sqrt b/W_{\text{ref}}$，$\boxed{b^*\approx\log_2W_{\text{ref}}}$
- 情形 A（$w_{\text{ref}}=882$、96 个环）：$b=10$ 时端到端 **3.00%**；三期最终为布局复用统一到 $b=8$，对应 **4.65%**
- **层 2 的权重量化占 85% 误差** —— 要提精度先动层 2

---

## 命名与约定

| 约定 | 说明 |
|---|---|
| 路径 | 期目录与 `src`/`results` 用**纯英文**；文档名可用中文 |
| 期号 | `0X_phaseX_<主题>`，跨期资源用 `99_` |
| 归档 | 每期内部的历史版本放该期的 `archive\` 下 |
| 文档 | 期内的文档用数字前缀排序（`00_`、`01_`…）；**注意三期编号有重号，见 `04/docs/00_文档索引.md §5.2`** |
| 权重 | 跨期模型统一放 `99_models\`，文件名带期号前缀 |

---

## 环境

```
Python:  C:\Users\lenovo\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe
依赖:    numpy, pandas, Pillow            （一期不需要 torch，也不需要 node）
torch:   仅 02_phase2_avatar\.venv 内有（torch 2.14.1+cpu / Python 3.12.14）
         三期没有独立 venv，借用二期的
Node:    v24.21.0，**不在 PATH 上**，必须用绝对路径：
         C:\Users\lenovo\.dsh\...\node\bin\node.exe
         三期的验收全靠它（sweep_options.js / verify_all.js / verify_planes.js），一期完全不需要
```

> **⚠ 路径不是全部可搬移**（旧版本文件写"已核查一期 130 个脚本，零硬编码"，**这条不成立**，2026-10 清点更正）：
>
> - **一期约 50 个脚本**写死了 `D:\czm我的世界\文件\Comparator Network\...`，该目录**已不存在**。
>   但数据在仓库内都有副本，改路径即可修：MNIST → `01_phase1_digit_recognition/data_mnist/data/MNIST/raw`；
>   DT teachers → `00_shared/archive_dt_20261001/DT_baseline_recovery/teachers/`；
>   `iris_teacher.json` → `01_phase1_digit_recognition/results/iris_teacher.json`。
>   一期只有约 1/3 的脚本（`strong_lut_solver.py`、`final_audit.py`、`gen_deploy*.py` 等）用相对路径。
> - **三期 4 个脚本**写死了 `E:/redstone_nn/...`：`sweep_options.js`、`verify_all.js`、`dump_planes_relu.js`、`schem_stack64.py`。
> - 结论：**在当前位置 `E:\redstone_nn` 能跑；整体搬移会坏掉上面这些脚本。**

---

*最后更新：2026-11-08 · 四期主体完成时（新增四期总结报告、修正四期状态）*
