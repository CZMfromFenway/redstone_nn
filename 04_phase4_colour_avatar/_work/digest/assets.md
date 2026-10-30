# 工具与产物清单

> 清点范围：`01_phase1_digit_recognition\`、`02_phase2_avatar\`、`03_phase3_quantized_network\`、`99_models\`、`99_tools\`，以及各期 `README.md`。
> 已跳过 `.venv`、`__pycache__`、`.piptmp`、`node_modules`。
> 所有路径均已用 `Test-Path` / `Get-ChildItem` 逐条核实存在；不确定用途的脚本标"用途不明"。
> 统计口径：一期 848 文件（py 139，其中 `src\` 115）、二期 7252 文件（py 266、js 37）、三期 405 文件（py 153、js 13）。

**运行前提（先说，因为它影响下面每一条"怎么跑"）**

```powershell
$py   = 'C:\Users\lenovo\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe'      # Python 3.12.14
$node = 'C:\Users\lenovo\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\node\bin\node.exe'      # Node v24.21.0
```

- `node` **不在 PATH 上**，必须用上面这个绝对路径。
- 一期的 `python` 只需 numpy/pandas/Pillow（**不需要 torch**）；二期的训练脚本需要 **torch 2.14.1**，只在 `02_phase2_avatar\.venv` 里有；三期沿用二期的 venv。
- 各期脚本普遍假定 `cwd` 是**本期的根目录**（`01_phase1_digit_recognition\`、`02_phase2_avatar\`、`03_phase3_quantized_network\`），而不是 `src\`。

---

## A. 一句话可用的入口（最重要的 10~15 个）

| 路径 | 作用 | 怎么跑 | 输入 | 输出 |
|---|---|---|---|---|
| `01_phase1_digit_recognition\build\draw.html` | 一期画板调试器，单文件网页，双击即用；改 15×15 像素立刻看 10 层 F 值和识别数字 | 双击，或浏览器打开 | 内置"数字 0..9"预设 / 手绘像素 | 屏幕上的 F 值、argmax、识别结果（载入"数字 0"预设应得 F=146、识别=0） |
| `01_phase1_digit_recognition\src\strong_lut_solver.py` | 强模 LUT 单列求解器：给 16 个目标信号 f0..f15，反解出可建造的 16 容器（b_1..b_15、B、截断方式）。**既是库也是入口**（`__main__` 跑 `self_test()` + `coverage_scan()`） | `& $py src\strong_lut_solver.py`（自测+覆盖率扫描）；或 `from strong_lut_solver import solve` | 16 个 0..15 目标值（`solve(f)`） | stdout；`solve()` 返回可行/不可行 + 容器参数列表（不落盘） |
| `01_phase1_digit_recognition\src\example_run.py` | 上面求解器的命令行壳，含 2 个可行例 + 2 个不可行反例 | `& $py src\example_run.py` 或 `& $py src\example_run.py 0 8 15 8 ...`（16 个数） | 16 个整数 | stdout 报告 |
| `01_phase1_digit_recognition\src\gen_all_v2.py` | 生成全部 10 层容器放置 `.mcfunction`（数据包的 `vessel:barrel_*` + `deploy_weights:*`） | `& $py src\gen_all_v2.py` | `results\Wint_model.npz`（+ MNIST raw，见 I-1） | `datapack\tiny_cnn\...`、`results\deploy_all_layers.json`、`results\vessel_coords.json` |
| `01_phase1_digit_recognition\src\make_canvas.py` | 由当前权重重新生成画板网页（权重逐位嵌进 HTML） | `& $py src\make_canvas.py` | `results\Wint_model.npz`、MNIST raw | HTML（**写到 repo 根的 `draw.html`，不是 `build\draw.html`**，见 I-3） |
| `02_phase2_avatar\src\sweep_condition.py` | 二期训练主入口：条件强度扫描（条件给多弱才崩），内部复用 `layout.py` 的规范布局 | `& .venv\Scripts\python.exe src\sweep_condition.py` | `data_v2_canonical\`（L0_split.npz 等） | `results\` 下新建 run 目录（`final.pt` / `metrics.jsonl` / `summary.json`） |
| `02_phase2_avatar\src\export_generic.py` | **任意** 32 输出学生（或 512 教师）ckpt → 网页可用 JSON，并自动在固定 1000 张 hold-out 上重算质量 | `& .venv\Scripts\python.exe src\export_generic.py --pt <ckpt> --name <模型名> [--label <说明>]` | `.pt` 检查点（`--teacher` 默认 `results/coherence/final_lam5_20k/final.pt`） | `webapp\<模型名>` 命名的 `_model.json` |
| `02_phase2_avatar\src\palette_v5.py` | 从原版 jar 里挑 16 个中性灰方块，给出 16 级灰调色板（第 5 版，按用户评审校准） | `& .venv\Scripts\python.exe src\palette_v5.py` | `99_mc_texture\1.21.11-Fabric 0.19.5.jar` | 调色板 JSON/PNG（写到 `results\colour\`） |
| `02_phase2_avatar\src\build_standalone.py` | 把 `sandbox.html` + 全部模型/json 内联成**离线单文件**网页 | `& .venv\Scripts\python.exe src\build_standalone.py` | `webapp\sandbox.html` + `webapp\*_model.json` | `webapp\sandbox_standalone.html`（5.27 MB） |
| `02_phase2_avatar\src\serve_webapp.js` | 本地 http 服务（`file://` 读不了 JSON，必须起服务） | `& $node src\serve_webapp.js webapp 8770` | `webapp\` 目录 | `http://127.0.0.1:8770/`（`/label.html` 打标） |
| `03_phase3_quantized_network\src\spec_extract.py` | **三期起点**：提取与硬件无关的数值规格（权重分布、激活范围、各 bit 位宽量化误差） | `& $py src\spec_extract.py` | 二期的 `pca22_dim4w64L1_model.json`、`colour_k8snap_model.json`、`scorer_quad.json` | `results\spec_grayscale.json`、`spec_colour.json`、`spec_scorer_quad.json` |
| `03_phase3_quantized_network\src\compiler.py` | 把 ReLU/CLIP 型 MLP 编译成整数管线（sigma、位权重 W^(k)、C_min、P_i、c_i/T_i、理论范围） | `& $py src\compiler.py` | 二期模型 JSON（脚本内写死相对路径） | `results\netlist_grayscale.json` |
| `03_phase3_quantized_network\viewer\plane_sim.html` | **三期当前版平面仿真器**：建板 + 求值 + `legalCheck()` 物理合法性闸门（含情形Ⅱ 修正） | 浏览器打开；或用 `node` `eval` 进脚本 | 模型 JSON（页面有 `<input type="file">`）或内置默认 | 页面可视化 + 可下载 JSON（`dump_*` 系列也吃这个文件） |
| `03_phase3_quantized_network\src\sweep_options.js` | 参数扫描器，文档称之为"**唯一可信的尺子**"（b × 通道 × 32 组合） | `& $node src\sweep_options.js`（`cwd` = 三期根） | `viewer\plane_sim.html` + 二期模型 json（路径写死 `E:/redstone_nn/...`） | stdout：`1536/1536` + 合法性 `1536/1536` |
| `03_phase3_quantized_network\src\schem_export.py` | 把 `board_dump.json` 写成 Sponge v3 `.schem`（每个测试用例一个文件） | `& $py src\schem_export.py` | `results\board_dump.json` | `results\schem\layer1_ch0_C{0..3}_*.schem` |

> 三期的完整回归链（文档 `docs\42` §六 给的标准命令序列）：
> `sweep_options.js` → `verify_all.js` → `verify_planes.js` → `dump_planes64.js` → `schem_stack64.py` → `schem_verify_stack.py` → `plane_plan.py`。

---

## B. 一期工具（`01_phase1_digit_recognition\`）

路径前缀省略 `01_phase1_digit_recognition\`。**全部需要 `cwd` = 一期根目录。**
类型：核心 = 结论所依赖；辅助 = 分析/出图；一次性 = 被后续版本取代的历史过程。

### B-1 强模 LUT 求解与数学

| 路径 | 作用 | 类型 |
|---|---|---|
| `src\strong_lut_solver.py` | 强模 LUT 单列求解器（16 容器反解）—— 一期最核心的可复用算法 | 核心 |
| `src\example_run.py` | 上者的命令行示例（2 正例 + 2 反例） | 辅助 |
| `src\neuron_design.py` | 实验：强模 LUT 网络能多接近同结构 MLP | 核心 |
| `src\neuron_design_constrained.py` | 同上的修订版：加入两条物理硬约束 | 核心 |
| `src\multilayer_greedy.py` | 多层强模 LUT 网络（逐层贪心 Fisher vs 联合训练对照） | 核心 |
| `src\fidelity_vs_mlp.py` | 强模 LUT vs MLP 保真度实验（v6，向量化） | 辅助 |
| `src\quant_range.py` | 权重整数化到 ±882 后准确率还剩多少 | 核心 |
| `src\split_b8_final.py` / `split_b8_K.py` / `split_b8_smax.py` / `split_b8_sopt.py` | b=8 拆分结构（4→8 强模→16 弱模→1）的四个定稿过程，`_final` 为定稿 | 核心 / 一次性 |
| `src\split_structure.py`、`src\split_cost.py`、`src\relu_split_cost.py` | 拆分结构与拆分层成本分析 | 辅助 |
| `src\compile_tanh_vs_relu.py` | 编译对照：tanh 拆分 vs ReLU 线性转弱模 | 辅助 |
| `src\teacher_compile_split.py` | 把原 Teacher 直接编译到"强模+拆分"结构 | 辅助 |
| `src\final_compile_4_8_16_1.py` | 定稿实验：Teacher 精确编译到 4→8→16→1 离散强模 LUT | 核心 |
| `src\final_tanh_buildable.py` | 最终验证：可建造阶梯 → 8bit 拆分 → 整数读出 | 核心 |
| `src\final_audit.py` | 定稿公式大规模验证 | 核心 |
| `src\verify_buildable.py` | 最小验证 v3（最小二乘定标）—— **路径失效，见 I-1** | 辅助 |
| `src\relu_active.py` / `relu_clean.py` / `relu_deadzone.py` / `relu_final2.py` / `relu_selfconsistent.py` / `relu_trueQ.py` / `relu_vs_tanh.py` | ReLU 路线的 7 个版本，`relu_final2.py` 与 `relu_trueQ.py` 为定稿口径 | 一次性 / 核心 |
| `archive\src\strong_lut_mechanism*.py`（12 个版本，`v11` 定稿） | 强模 LUT 单元物理机制仿真 + 结构等价性核查 | 一次性（归档） |
| `archive\src\strong_lut_math_{corrected,final}.py`、`strong_lut_step_final.py`、`strong_lut_two_truncations.py`、`strong_lut_span.py`、`strong_lut_confirm.py` | 单元数学解析与步长/截断定稿 | 一次性（归档） |
| `archive\src\formula_audit.py`、`closed_form_check.py`、`truth_probe.py`、`truth_probe2.py`、`czm_test_debug.py` | 公式全量验证与真值探针 | 一次性（归档） |

### B-2 MNIST 训练与输入量化

| 路径 | 作用 | 类型 |
|---|---|---|
| `src\mnist_k1.py` | MNIST 15×15 二值化 + K=1 结构的两条路线 | 核心 |
| `src\mnist_sweep.py` | MNIST 15×15 二值的降体积实验 | 核心 |
| `src\mnist_binary_sweep.py` / `src\mnist_binary_compress.py` | 900→更少二值特征；两档阈值 + 邻域 OR 抽稀 | 辅助 |
| `src\int_weight_model.py` | 整数权重版 225→10 线性分类器：完整分析 + 存表（**产出 `99_models\` 里的权重**） | 核心 |
| `src\analyze_225x10.py` | 225→10 线性分类器的参数与输出分析 | 辅助 |
| `src\train_normalized.py` / `src\train_final.py` / `src\retrain_aug.py` | 输入归一化 / 归一化+增广 / 三类增广重训 | 核心 / 一次性 |
| `src\norm_compare.py` | 对照：只做质心对中（不做缩放） | 辅助 |
| `src\honest_eval.py` | 诚实评估：训练集切验证集，测试集完全不碰 | 核心 |
| `src\compare_protocols.py` | 两种划分下的权重对比 | 辅助 |
| `src\width_scan.py` / `width_scan_final.py` / `width_scan_loss.py` / `width_scan_refine.py` / `width_scan_softquant.py` | 加宽隐层的 5 轮扫描，`_final` 为定稿 | 一次性 / 核心 |
| `src\boldness_test.py`、`src\stroke_stats.py` / `stroke_stats2.py` / `stroke_stats3.py` | 笔画加粗影响、笔画粗细统计（`3` 为合成笔画标定版） | 辅助 |
| `src\digit_code*.py`（`digit_code`、`digit_code2`、`digit_codeB_correct`、`digit_codeB_h`、`digit_codeC`、`digit_code_cmp`、`digit_code_h`） | 225→h 位二进制编码路线的 7 个分支 | 一次性 |
| `src\trace_sample.py`、`src\pick10.py` | 单样本逐层 F 期望值；挑 10 个识别正确的样本并展示 | 辅助 |
| `src\diag_shapes.py`、`src\diag_align.py`、`src\diag_iris*.py`（4 个）、`src\probe_*.py`（4 个） | 形状/对齐/鸢尾花/前向口径诊断 —— **多数路径失效，见 I-1** | 一次性 |
| `data_mnist\optimal_cnn_test.py` ~ `_4.py` | 四份 MNIST "最优 CNN"对照试验（与主结论无关） | 一次性 |

### B-3 比较器网络编译与 MC 容器信号

| 路径 | 作用 | 类型 |
|---|---|---|
| `src\mc_10layers.py` | 10 层完整参数表 | 核心 |
| `src\mc_container_correct.py` | MC 容器值：正确公式（权重直接加减，无倍数） | 核心 |
| `src\mc_container_signals.py` | MC 容器信号分配表：10 层 × 225 容器 | 核心 |
| `src\mc_comparable.py` | 验证 10 层 F 可直接比大小、max(F) = 识别数字 | 核心 |
| `src\mc_F0_align.py` / `mc_F0_align_v2.py` | F_0 重对齐到 z=0 时 F=128，并测截断对 argmax 的影响 | 核心 |
| `src\mc_digit9_v4.py` → `v5` → `v6` → `_precise` → `bias` → `bias_v2` → `check` → `final` → `F0`（9 个） | 数字 9（偶数层）支线偏置的 9 轮推导，`mc_digit9_final.py` / `mc_digit9_check.py` 为定稿 | 一次性 / 核心 |
| `src\mc_g1_digit9.py` / `_v2` / `_v3` / `src\mc_g1_solve.py` | 数字 9 最底层 15 像素的 G_1 模拟与求解 | 一次性 |
| `src\gen_all_layers.py` / `src\gen_all_v2.py` | 全部 10 层容器放置函数（`_v2`：y 步长 = 2、层内 (x,z) 一致） | 核心 |

### B-4 数据包与放置生成

| 路径 | 作用 | 类型 |
|---|---|---|
| `src\gen_deploy.py` / `gen_deploy_v2.py` / `gen_deploy_v3.py` | `deploy_weights` 放置函数三代（`v3` = 最终坐标规则 + 真实容器值） | 一次性 / 核心 |
| `src\gen_layer10.py` / `_bias` / `_full` / `_v2` | 第 10 层（数字 9）225 容器 + 8 个 G_{i,0} + F_0 的整层放置 | 一次性 |
| `datapack\tiny_cnn\pack.mcmeta` | 数据包描述（唯一的数据包） | 核心 |
| `datapack\tiny_cnn\data\deploy_weights\function\deploy_all_layers.mcfunction` | **2340 条**放置指令（已实测：非注释非空行 = 2340），入口 `/function deploy_weights:deploy_all_layers` | 核心 |
| `datapack\tiny_cnn\data\deploy_weights\function\layer10_digit9_full.mcfunction`、`place_layer10_digit9.mcfunction`、`place_top_layer.mcfunction` | 第 10 层与顶层的分步放置 | 辅助 |
| `datapack\tiny_cnn\data\vessel\function\barrel_0.mcfunction` … `barrel_511.mcfunction`（512 个） | 单个木桶的 `setblock`，用 `CustomName` 记 `signal: S, item: N` | 核心 |

### B-5 可视化与调试

| 路径 | 作用 | 类型 |
|---|---|---|
| `src\make_canvas.py` | 画板网页生成器（单文件，双击即用） | 核心 |
| `build\draw.html` | 上面生成的成品画板（28 KB）—— 一期最重要的交互资产 | 核心 |
| `src\weight_heatmap.py` / `src\weight_heatmap2.py` | 10 层权重热图（`2` 为增强对比度 + 偏置可视化） | 辅助 |
| `src\viz_mnist15.py` | 可视化 MNIST 缩到 15×15 并二值化后的样子 | 辅助 |
| `src\controlled_readout.py` | 受控对照：同一第一层 + 三种第二层整数读出 | 辅助 |
| `src\check_observations.py`、`src\restart_probe.py`（后者在二期） | 核对关键观察 | 一次性 |
| `src\migrate_phase2.py` | 一次性迁移脚本（D 盘 → E 盘），**含写死的源/目标绝对路径**，不可重跑 | 一次性 |
| `obsidian\*.json`（5 个） | Obsidian 库配置，无研究内容 | 辅助 |

---

## C. 二期工具（`02_phase2_avatar\src\`，211 py + 37 js）

路径前缀省略 `02_phase2_avatar\`。二期脚本普遍用 `HERE/ROOT` 相对定位，**没有硬编码绝对路径**（已核查），但需要 `02_phase2_avatar\.venv` 的 torch。

### C-1 模型定义与训练

| 路径 | 作用 | 类型 |
|---|---|---|
| `src\layout.py` | **规范布局**：一个定义，被所有 generator 脚本 import（改这里会动全二期） | 核心 |
| `src\cond_gan.py` | 条件 GAN：从左半补右半 | 核心 |
| `src\gan_L0.py` | L0 GAN：生成完美对称的 16 级灰度 8×8 头像 | 核心 |
| `src\dim_sweep.py` | 在 32 输出配方下列 PCA-k 阶梯 | 核心 |
| `src\dim_sweep_parallel.py` | 单进程里并行跑整个维度阶梯 | 辅助 |
| `src\sweep_condition.py` | 条件强度扫描（20 KB，二期最大单文件之一） | 核心 |
| `src\style_conditional.py` / `src\style_conditioned.py` | 按风格专门化训练 / 风格条件化提纯 | 核心 / 辅助 |
| `src\shrink_32out.py` / `src\phase5_shrink.py` | 512→32 的输出压缩与中层瓶颈收缩 | 核心 |
| `src\distill_32.py` | 512→32 蒸馏（"PHASE 2 of 512→32"） | 核心 |
| `src\tv_sweep.py` | 蒸馏 32 输出学生的 TV 权重扫描（单进程） | 辅助 |
| `src\train_contrast_variants.py` / `src\train_transform.py` | 训练期对比度补救 / 输入旋钮的 4→4 变换 | 辅助 |
| `src\colour_stage2_train.py` / `src\colour_yield_train.py` | 彩色 stage-2 实验矩阵 / C1·C2 提高彩色良率 | 核心 |
| `src\style_cluster.py`、`src\kernel_style.py` | 自动恢复视觉风格 / 可学习微小卷积核风格库 | 辅助 |
| `src\continuous_vs_discrete.py`、`src\commit_fix.py` | 不量化输出是否更好学 / 在取整后的输出上训练 | 辅助 |

### C-2 实验调度与时间管理

| 路径 | 作用 | 类型 |
|---|---|---|
| `src\run_experiments.py` | 带时间预算的实验跑批器 | 辅助 |
| `src\run_plan.py`、`src\plan_from_timings.py` | 按实测 timings 执行计划、按实时时钟生成 plan.json | 辅助 |
| `src\measure_then_plan.py`、`src\runtime_util.py` | 先测每步真实耗时再排计划；过夜实验基础设施 | 辅助 |
| `src\overnight_runner.py`、`src\chain_queues.py` | 自主过夜跑批 / 把队列串起来跑到截止时间 | 辅助 |
| `src\run_compress.py` / `src\run_colour_gan.py` / `src\run_colour_parallel.py` / `src\run_adv_sweep.py` | 四个薄壳跑批入口（分别跑宽度扫描、C2/C3 彩色、并行彩色、对抗强度） | 一次性质壳 |
| `src\smoke_torch.py` | torch 冒烟测试（前向+反向跑通即止，不落盘） | 辅助 |
| `src\install_wheels.py` | **不走 pip** 把 torch 及其依赖装进二期 venv | 一次性（已完成） |

### C-3 采样与打分（reward / preference）

| 路径 | 作用 | 类型 |
|---|---|---|
| `src\fit_reward.py` | 从人工标签拟合 reward 模型并导出给网页（结论：MLP 过拟合，用户审美在 22 PC 上**基本线性**） | 核心 |
| `src\fit_preference.py` | 从标签拟合 22 维风格空间上的偏好分布 | 核心 |
| `src\refit_style_reward.py` | 在**风格标签**上重拟 reward，检验旧 reward 能否迁移 | 辅助 |
| `src\quad_filter.py` | 用**非线性** reward 做筛选增益（线性版 AUC 0.543 失败） | 核心 |
| `src\perarm_scorer.py` | 每个压缩模型是否需要自己的筛选打分器 | 辅助 |
| `src\simulate_labeling.py` | 仿真：只有 ~100 条人工标签时，偏好引导 CEM 到底能不能改进采样 | 核心 |
| `src\strategy_compare.py` | 100 标签下哪种打标策略真的有效 | 核心 |
| `src\preference_flow.py` | "600 候选 → 保留前 35%" 能否塌缩成红石可做的事 | 辅助 |
| `src\tilt_prior.py` | 直接倾斜采样分布而不是筛样本 | 辅助 |
| `src\styles_gauss.py` / `src\gauss_envelope.py` | 把手工风格点变成联合高斯 / 在 22 维里拟合高斯包络 | 核心 |
| `src\style_embeddings.py`、`src\style_switches.py`、`src\styles_pc3.py` | 真实脸索引画廊与风格嵌入 / 风格切换 / PC3 观察验证 | 辅助 |
| `src\learnability.py`、`src\radial_probe.py`、`src\style_nonlinear_check.py`、`src\pca4_geometry.py` | "好"到底可不可学、是方向还是半径、是否非线性、几何检验 | 辅助 |
| `src\export_scorer.py` / `export_scorer_anime.py` / `export_scorer_quad.py` | 把 4→16→1 打分器 / 类二次元探针 / 二次打分器导出成网页可用的 **JSON** | 核心 |
| `src\analyse_*.py`（`analyse_labels`、`analyse_round`、`analyse_rounds`、`analyse_round2_post`、`analyse_round3`、`analyse_style5`、`analyse_dimmix`）、`src\round*_stats.py`（8/10/11/12）、`src\round2_clean_test.py`、`src\round10_borderline.py` | 12 轮盲测的统计与结论（Wilson 区间、Fisher、漂移检查） | 辅助 |

### C-4 调色板与色彩

| 路径 | 作用 | 类型 |
|---|---|---|
| `src\palette_from_jar.py` / `palette_from_jar2.py` | 从原版 jar 找 16 个中性灰方块（前两版） | 一次性 |
| `src\palette_v3.py` / `palette_v4.py` / `palette_v5.py` | 16 级灰方块的第 3、4、5 版，**`palette_v5.py` 为定稿** | 一次性 / 核心 |
| `src\mc_blocks.py` / `mc_fit.py` / `mc_snap.py` / `mc_block_cmp.py` | 方块色（不是地图色）/ 调色板能否用 MC 颜色搭出 / 吸附到真实方块色后重训 / 前后对比 | 核心 |
| `src\mc_grey.py`、`src\mc_merge11.py`、`src\mc_shade.py` | 灰度吸附到真实灰色 / 16 级并成 11 档 / 在**已训练**模型上换色（形状不变） | 核心 |
| `src\colour_encoding.py`、`src\colour_sampler.py`、`src\colour_structure.py` | 花训练时间前先做两个便宜测量 / 色彩分量从哪来 / 好坏脸的实际分界是结构不是颜色 | 核心 |
| `src\colour_palette.py` / `colour_palette_cmp.py` / `colour_purify.py` / `colour_purify_all3.py` / `colour_quantise.py` | 分类成调色板索引而不是回归颜色；提纯；量化缺口 | 核心 |
| `src\k8_schemes.py` / `src\k8_figure.py` | 三个 K=8 彩色方案（只差调色板来源）+ 出图 | 核心 |
| `src\hsv_feasibility.py`、`src\luminance_to_colour.py`、`src\colour_feasibility.py`、`src\colour_stage2.py` | HSV 拆分是否划算 / 亮度能否驱动彩色 / 饱和度色相能否从亮度图推断 | 辅助 |
| `src\colour_gan.py` / `colour_hardd.py` / `colour_expand.py` / `colour_expand_summary.py` | 彩色生成器（advisor-critic 配方）/ 直通估计器硬判别 / 扩容与对抗项诊断 | 核心 / 一次性 |
| `src\colour_eval.py` / `colour_metrics.py` / `colour_scorer.py` / `colour_criteria.py` / `colour_index.py` / `colour_three_way.py` / `colour_yield.py` | 彩色评估、指标重算（写**一个** utf-8 结果文件）、按用户标准打分、编号好脸、好/中/差三分、良率归因 | 核心 |
| `src\colour_filter.py` / `colour_filter_fair.py` | 把种子筛选搬到彩色管线；**去掉循环论证**的公平版 | 核心 |
| `src\colour_stretch.py`、`colour_final.py`、`colour_samples.py`、`colour_adv_fig.py`、`colour_c1c2_export.py`、`colour_c1c2_fig.py`、`colour_data.py` | 逐脸拉伸补救、带全局拉伸的成品图、大批可读样张、对抗变体一图全览、C1/C2 导出与出图、数据准备 | 辅助 / 核心 |
| `src\export_colour.py` / `export_palette.py` | 彩色模型 / 调色板分类器 → webapp JSON（与灰度导出同格式） | 核心 |

### C-5 蒸馏与正则（碎片化 / 斑点 / 相干性）

| 路径 | 作用 | 类型 |
|---|---|---|
| `src\distill_decisive.py` / `distill_visual.py` / `full_metrics.py` | 在**同一批 1000 张** hold-out 上做决定性对比 / 看蒸馏学生实际产出 / 32 输出学生是否真的更差 | 核心 |
| `src\coherence_screen.py` / `coherence_lam.py` / `coherence_long.py` | 哪种空间相干惩罚真能去掉斑点；lam 扫描；赢家设置的长跑 | 核心 |
| `src\fragment_fix.py` / `fragment_fix2.py` | 攻击最后的质量缺口"碎片化"；正确惩罚是对**取整后**输出做 TV（直通） | 核心 |
| `src\measure_speckle.py` / `zoom_compare.py` | 量化"麻麻赖赖"并对比真实脸 / 放大看碎片化是否可见 | 辅助 |
| `src\contrast_diagnosis.py` / `contrast_fixes.py` / `contrast_remedies_eval.py` / `contrast_corr.py` / `gradient_cause.py` | 为什么发白/发灰；两套补救的实测；模型知不知道要多对比；导致软的原因指标 | 核心 |
| `src\brightness_study.py` / `brightness_hist.py`、`src\frozen_pixel_check.py` | min-max 拉伸后仍偏暗？直方图；对比度损失模型是否靠冻结像素作弊 | 辅助 |
| `src\mixture_out.py`、`src\unimodal_analysis.py`、`src\why_distribution.py`、`src\ladder_probe.py` | 混合输出能否打败单连续值；16 通道按亮度排序后是否单峰；为什么必须 4×8×16 | 核心 |
| `src\size_ladder.py`、`src\family_sheet.py` | 生成器能小到多小；全宽度×层数一页对照 | 辅助 |

### C-6 可视化与进度页

| 路径 | 作用 | 类型 |
|---|---|---|
| `src\progress_sheet.py` / `progress_sheet_v2.py` / `final_progress.py` / `morning_report.py` | 二期进度页（`v2` 为用户要求重绘版）；`final_progress` 含拒绝退化模型的守卫；过夜晨报 | 核心 |
| `src\visual_shrink.py`、`src\family_sheet.py`、`src\pca4_map.py`、`src\pca48_showcase.py` | 新架构实际产出；4-PCA 四旋钮 × 8 级全脸图（4096 点标签的来源） | 核心 |
| `src\pca_analysis.py`、`src\pca_check.py`、`src\pca4_followup.py`、`src\pca4_labels.py` | PCA-4/8 为什么出奇地好；PCA 是否适合调色板分类；4096 点标签图 | 核心 |
| `src\arch_diagram.py`、`src\input_cost.py`、`src\input_distribution.py`、`src\input_dist_visual.py`、`src\input_visual.py` | 架构图（含乘法数）；22 维输入有多少真有用；输入分布敏感性（决定投掷器电路量） | 核心 |
| `src\render_fixed.py`、`src\forensic_render.py`、`src\render_page_faces.js` | 从 ckpt 按正确空间布局渲染真输出；对 PNG 做取证（不对称是否真存在）；用**页面自己的解码路径**渲染 | 核心 |
| `src\interpret_kernels.py` / `visualize_kernels.py`、`src\name_the_knobs.py` | 学到的核到底找到了什么；命名四个存活的 PCA 旋钮 | 辅助 |
| `src\equal_budget.py`、`src\filter_gain.py`、`src\filter_final.py`、`src\w128_before_after.py`、`src\perceptual_diversity.py` | 同预算公平对比；用用户自己的 hold-out 标签估算筛选真实增益；筛选定稿；"2 个旋钮够了"的感知多样性更正 | 辅助 |
| `src\gray_vs_colour.py`、`src\colour_*_fig.py`、`src\stage2_conditions_fig.py` | 同缩放灰度 vs 彩色并排；各彩色变体渲染；stage-2 图（只用手头实物拼装） | 辅助 |

### C-7 webapp（交互实验台）

| 路径 | 作用 | 类型 |
|---|---|---|
| `webapp\sandbox_standalone.html` | 5.27 MB 单文件，**双击即用、完全离线**，含全部灰度模型 + 两个打分器 + 风格抽样 + 矩形夹取 | 核心 |
| `webapp\sandbox.html` | 同页面的服务器版（配合 `serve_webapp.js`） | 核心 |
| `webapp\index.html` | PCA-22 主页：22 个旋钮实时出脸，五个模型一键切换 | 核心 |
| `webapp\plane.html` | 4-PCA 平面标注工具，产出 4,096 点标签 | 核心 |
| `webapp\label.html` / `webapp\label_colour.html` | 灰度 / 彩色盲测打标页 | 核心 |
| `webapp\compare.html` | 对比页（配 reward 模型做引导采样） | 辅助 |
| `webapp\*.json`（共 46 个：`pca22_*_model.json` **23 个**、`colour_*_model.json` **8 个**、`scorer_*.json` 3 个、`*_manifest.json` 6 个，另加 `preference_reward.json`、`styles_gauss.json`、`style_envelopes.json`、`merge_mc11.json`、`grayscale_family.json`） | 页面消费的模型/清单/reward | 核心 |
| `webapp\README.md` | 五个模型的结构、逐像素准确率、三种 `output_mode` 解码说明 | 辅助 |
| `src\build_standalone.py` | 由 `sandbox.html` + 全部 json 打包出 `sandbox_standalone.html` | 核心 |
| `src\export_web_pca22.py` / `export_best_and_clusters.py` / `export_both_models.py` / `export_dims.py` / `export_compressed.py` / `export_new_arch.py` / `export_stretch_arm.py` / `export_envelopes.py` / `export_styles.py` / `export_generic.py` | 10 个导出器：把 ckpt/风格点/包络变成网页能跑的 JSON | 核心 |
| `src\js_forward_check.js` / `verify_web_model.py` | 页面 JS 前向 vs Python 模型的数值等价校验 | 核心 |
| `src\test_*.js`（**34 个**：`test_all_models`、`test_decode`、`test_sandbox`、`test_sandbox2`、`test_label_page`、`test_label_all_models`、`test_label_w128`、`test_mix_mode`、`test_mix_ui`、`test_plane`、`test_plane2`、`test_round2_setup`、`test_round9`、`test_round10`、`test_round11`、`test_round12`、`test_colour_label`、`test_colour_round8`、`test_style_sandbox`、`test_style_scheme`、`test_webapp_two_models`、`test_webapp_three`、`test_webapp_five`、`test_quad_sandbox`、`test_reward_e2e`、`test_standalone`、`test_post_stretch`、`test_random_button`、`test_speckle_only`、`test_envelope_page`、`test_compare_page`、`test_compare_w128`、`test_models_distinct`、`test_blind3`） | 页面回归测试，跑在 node 里（多数用 headless/jsdom 式自建 DOM 桩） | 核心（回归网） |
| `src\test_pipeline.py` | 用**合成标签**跑通打标管线的端到端测试（注意：这是 `.py`，不是 `test_*.js` 那 34 个之一） | 核心 |

> `test_compare_page.js`、`test_models_distinct.js` 无文件头说明，**用途不明**（从命名推断与 compare 页、模型区分度有关，未验证）。

### C-8 数据管线

| 路径 | 作用 | 类型 |
|---|---|---|
| `data_v2_canonical\L0_manifest.json`、`L0_split.npz`、`L0_full_idx.npy`、`L0_left_idx.npy` | **规范数据集**（v2 canonical），训练脚本的默认输入 | 核心 |
| `data\manifest_v2.json` / `v2a`、`faces8_v2.npy` / `v2a`、`faces8_rgb_v2.npy` / `v2a`、`palette_avatar16_v2.npy` / `v2a`、`alpha_cov_v2.npy` / `v2a`、`source_names_v2.json` / `v2a`、`mc_map_colors.json` | 两版数据集与来源清单（`faces8_rgb_v2.npy` 单文件 177 MB） | 核心 |
| `src\rebuild_faces8.py`、`src\build_L0_data.py` | 干净重建 8×8 脸数据集（带溯源）/ 构建 L0 数据集 | 核心 |
| `src\check_rebuild.py` | 校验重建后的 v2 是否真的去掉了审计发现的污染 | 辅助 |
| `src\colour_data.py` | 彩色数据准备（与灰度管线逐行对齐） | 核心 |
| `audit_tmp\AUDIT_REPORT.md` + `audit_tmp\01..11_*.py`（11 个） | 对 `archive_phase1_attempts\data\` 数据管线的独立审计（含 `tree_compare.json` 11.8 MB 树对比） | 辅助 |
| `archive_phase1_attempts\`（44 py + 5820 皮肤 png + 130 结果） | 一期头像九方案（GAN/自回归/混合近邻/蒸馏/对称分解）的完整归档 | 一次性（归档） |

---

## D. 三期工具（`03_phase3_quantized_network\src\`，153 py + 13 js）

路径前缀省略 `03_phase3_quantized_network\`。**多数 JS 与部分 Python 里写死了 `E:/redstone_nn/...` 绝对路径**（见 I-2），但仓库当前就在这个位置，所以今天能跑。

### D-1 规格提取

| 路径 | 作用 | 类型 |
|---|---|---|
| `src\spec_extract.py` | 提取硬件无关的数值规格（权重分布 / 激活范围 / 多 bit 位宽量化误差 / 乘法与权重总数） | 核心 |
| `docs\01_规格提取.md` | 上者的文字结论（4 bit 相对 RMS 0.041、8 bit 0.0023；位宽是唯一杠杆） | 辅助 |

### D-2 编译器与记号

| 路径 | 作用 | 类型 |
|---|---|---|
| `src\compiler.py` | **核心编译器**：`Layer(w,theta,kind,comparable,out_bits/lam)` → `compile_network(layers,xmin,xmax,in_bits,w_ref)` | 核心 |
| `src\intpipe.py` | 纯整数管线（编译期常数全整数，运行期只加减比较右移） | 核心 |
| `src\layer1_compile.py` | 编译 4→64 层1（b=8, W_ref=882）并落到平面图 | 核心 |
| `src\range_check.py` | 按最终结构 `c_j = ceil((Fmax_j − A_j)/(2^b−1))` 重算 | 核心 |
| `src\step3_check.py` / `step4_check.py` / `step5_numbers.py` / `step5b_check.py` | 编译七步中第 3/4/5 步的重算与对账（`5b`：Z_p = G_p + B_p 通道平移量） | 核心 |
| `src\define_terms.py` / `notation_check.py` / `symbol_audit.py` | 记号钉死与"全局平移是否足够"；x/y 量化不对称来源；**符号清算**（从 X 到第二层输出的每个符号给定义与实测值） | 核心 |
| `src\diag_bits.js` / `diag_compile.js` / `compare_compile.js` | b≠8 分歧来源；层1 的 32 处差异是否为 .5 平局；JS `compile()` vs Python 逐元素对比 | 辅助 |
| `src\export_py_compile.py` / `export_py_forward.py` | 导出 Python 侧编译结果 / 整数前向 N 值，供 JS 对账 | 辅助 |

### D-3 前向仿真与 node 校验

| 路径 | 作用 | 类型 |
|---|---|---|
| `src\cmp_sim.py` | **比较器电路静态模拟器**（核心工具，被多个后处理脚本 import） | 核心 |
| `src\cmp_sim3.py` | 按 schem 原样块表建板，翻转与否做成开关，两种都跑 | 核心 |
| `src\verify_all.js` | 总验收：仿真器 vs Python 编译模型（层1 全 b、层2 全对；头部 vs 整数前向 N = 168/168） | 核心 |
| `src\verify_planes.js` | 层1 全 64 通道 × 32 选项 × 多组 8 位输入逐片验收（原漏的维度） | 核心 |
| `src\verify_towers.js` | 火把塔规则在 `inRev / bitRev / parSwap` 全 32 组合下的适配性 | 核心 |
| `src\sweep_options.js` | 参数扫描器（`b × 通道 × 32 组合`），文档称"唯一可信的尺子"；**不通过合法性闸门就不许导出** | 核心 |
| `src\verify_compiler.py` | 两件事：编译器产物 vs 手算公式逐元素；输出对比 | 核心 |
| `src\predict.js` | 随机输入测试：给定 X=(X0..X3)，仿真 64 通道，输出游戏内可对照的检查点 | 核心 |
| `src\dump_planes64.js` / `dump_planes_L2.js` / `dump_planes_relu.js` / `dump_board.js` / `dump_bitrev.js` | 导出层1 64 片 / 层2 32 通道 / ReLU 平面 / 整块板含数值 / 位权镜像对比 | 核心 |
| `src\relu_chain_check.py` / `relu_probe.py` | 矩阵乘法头部 → ReLU 除法器 → 与编译器整数前向对账；反解 `data\relu.schem` 的 8 位除法器 | 核心 |
| `src\report_planes.py` | 把 `planes_*.json` 里 X=0 的检查点整理成游戏内可对照的表 | 辅助 |
| `src\ring_solve.py` | 用模拟器反解两个环的烧录值 | 辅助 |

### D-4 `.schem` 读写与校验

| 路径 | 作用 | 类型 |
|---|---|---|
| `src\schem_read.py` | **最小 NBT 解析器 + Sponge schematic 读取**（纯 Python，无第三方依赖） | 核心 |
| `src\schem_write.py` | 把模拟器里的 Board 写成 Sponge v3 `.schem` | 核心 |
| `src\schem_export.py` | `board_dump.json` → 每个测试用例一个 `.schem` | 核心 |
| `src\schem_verify.py` | 校验：NBT 完整回读 + 比较器 `powered` 与 `OutputSignal` 是否一致 | 核心 |
| `src\schem_roundtrip.py` | 用参考文件同一套读法读回自己生成的文件，确认结构一致 | 核心 |
| `src\schem_stack64.py` | 层1 的 64 个通道竖直堆叠成一份 `.schem` | 核心 |
| `src\schem_verify_stack.py` | 回读 `layer1_stack64.schem`，与 `planes64.json` 逐格对账（实测 12864 格 0 处不一致、不可达 0） | 核心 |
| `src\schem_full.py` / `schem_verify_full.py` | 64 片平面 + 32 根火把塔拼成整网 / 逐格回读对账 | 核心 |
| `src\schem_plane.py` / `schem_plane2.py` / `plane_gen.py` / `plane_full.py` / `plane_full_v2.py` | 按 `rsnn.schem` 反推规则生成任意 (n,b) 平面图；单平面逐块完整图（`_v2` 按用户指正修正） | 核心 |
| `src\schem_sr.py` / `schem_sr_zoom.py` / `schem_columns.py` / `schem_grid.py` / `schem_view.py` / `schem_annot.py` | 通用 schem 读取俯视图 / 环区放大 / 按列看 y 结构 / 打成俯视表格 / 画俯视图+ASCII / 标注俯视图 | 辅助 |
| `src\schem_values.py` / `schem_render_vals.py` / `learn_schem.py` / `schem_algebra.py` / `schem_algebra2.py` | 编排样机 32 容器信号值与两环常数；学习一期 `hand_write_digital.schem`；`rsnn.schem` 的代数分析 | 辅助 |
| `src\plane_plan.py` | 画层1 一片平面的俯视平面图（火把塔选址用） | 辅助 |

### D-5 投掷器搜索（dropper）

| 路径 | 作用 | 类型 |
|---|---|---|
| `src\dropper_search.py` / `dropper_search2.py` | 用投掷器之和逼近截断正态（`2` 为修正版搜索） | 核心 |
| `src\dropper_shapes.py` / `dropper_spec.py` / `dropper_cap.py` | 按真正投掷器约束重算 / 绿色方案（INDEP）参数表 / 单投掷器 ≤15 的硬约束 | 核心 |
| `src\dropper3.py` / `dropper3b.py` | 三值投掷器 {0,1,2} 逼近截断正态（`3b` 用累积分布距离评判） | 核心 |
| `src\dropper_final.py` / `dropper_final16.py` / `dropper_fixed.py` | 13 路三值方案验证 / m=16 最终常数 / alpha 定点 8→12 位 | 核心 |
| `src\dropper_quiz.py` / `dropper_fig.py`、`src\slots_search.py`、`src\lahc_search.py`、`src\final_lahc.py` | (4,1,4) 形状考题答案；出图；N≤9 格数搜索；LAHC 搜索算法；最终 N=9 形状族 | 辅助 / 核心 |
| `src\direct30.py` / `direct30b.py` / `direct4.py` / `direct_m8.py` / `direct_shape.py` / `shape_final.py` | 直接法（每通道 M 个投掷器，权重全整数，无定点乘数）的 6 条路线，`direct_m8.py` / `shape_final.py` 为定稿 | 核心 |
| `src\style_dropper.py` / `style_batch.py` / `style_main.py` / `style_indep.py` / `style_meanface.py` / `style_refit.py` | 投掷器采样各风格 → 4→4 变换 → 整数模型 → MC11 出图 | 核心 |

### D-6 容器可达性与 barrel 编码

| 路径 | 作用 | 类型 |
|---|---|---|
| `src\barrel_palette.py` | **容器编码公式出处**：0..897 全覆盖的调色板生成器 | 核心 |
| `src\barrel_probe.py` | 反推容器信号强度与物品数量的关系，并核对现有 512 个 barrel 函数 | 核心 |
| `src\container_reach.py` | 木桶 / 漏斗 / 投掷器的 0..897 信号强度可达性与所需物品 | 核心 |
| `src\container_round.py` | 修正：每格可留空（贡献 0），上一版漏了导致 U=0 被判不可达 | 核心 |
| `src\container_gaps.py` | 补算木桶/漏斗缺哪些信号 | 核心 |
| `src\wref_container.py` | 决定性测量：容器上限 897 ⇒ 要让每个位权重塞进一个容器必须 `2·w_ref ≤ 897` ⇒ `w_ref ≤ 448` | 核心 |
| `src\lut_redundancy.py` / `lut_redundancy2.py` | 量化 LUT 的容器冗余（`2` 按正确截断分符号计算） | 辅助 |

### D-7 误差分析

| 路径 | 作用 | 类型 |
|---|---|---|
| `src\error_budget.py` | **三类误差的来源、量级与最优 b 的解析预测** | 核心 |
| `src\deviation_dist.py` / `deviation_final.py` / `diagnose_bias.py` | 同一 4→1 通道实数 Y 与整数 Y 的偏差分布；F_0 标定后的最终版；为什么偏差随 b 系统性正偏 | 核心 |
| `src\perbit.py` / `perbit2.py` / `err_w.py` | 逐位残差贡献（等比数列 vs 每位等权）；隔离权重量化误差 | 核心 |
| `src\lut_err.py` / `lut_err2.py` / `fourx.py` | 把量化误差框架应用到强模 LUT，与位权重方案并排；b=8 时 LUT 误差 ≈ 位权重在 W_ref=4×868 的误差 | 核心 |
| `src\mixed_wref.py` / `scenario_sweep.py` / `z_bits.py` | 按层独立 W_ref 与位宽；三种架构约束 × 不同 b 的精度扫描；每种方案 Z 的量程位宽与 min-max 成本 | 核心 |
| `src\one_channel.py` / `one_channel_scale.py` / `edge_vs_centre.py` | 把层0 一个输出通道当 4→1 实例摊开全部参数；缩放 S 怎么选；格心 vs 格边重建 | 辅助 |
| `src\ablate.py` / `skip_y.py` / `test_D.py` / `test_D2.py` / `post_compare.py` | 末层误差消融 + 线性翻转律；跳过 Y 直接用 F 归一化；D=2^b vs 2^b−1 端到端；Z_p 能否直接做 min-max；后处理整数化 | 核心 |
| `src\e2e_compare.py` / `e2e_int.py` / `trace_one.py` / `clip_stat.py` | 端到端对比（含实验发现的改进）/ 全整数管线 / 一个数字的整数旅程 / 碰壁量化 | 核心 |
| `src\equiv_wref.py` / `build_b9.py` / `build_oddb.py` / `check_spec.py` | 可定义部分等验；b=9 两套方案；奇数 b 的"先造 b+1 再砍"；逐条核对用户提的结构性质 | 辅助 |

### D-8 体积核算

| 路径 | 作用 | 类型 |
|---|---|---|
| `src\volume.py` | 数模结合（位权重）vs 强模 LUT 的体积对比（只比全连接核心） | 核心 |
| `src\volume_final.py` | 定稿版（LUT 宽度用 `4.5·n·(b/4)`） | 核心 |
| `src\volume_report.py` | **实际 schem 体积 vs 文档估计** | 核心 |
| `src\volume_vs_pixels.py` / `volume_vs_pixels2.py` | 3 倍体积能换多少像素一致性（`2` 修正环数：每个 >897 的位权重各需一个环） | 核心 |

### D-9 火把塔（tower）

| 路径 | 作用 | 类型 |
|---|---|---|
| `src\tower_spec.py` | 从 `planes64.json` 反推 32 根塔的占地（不依赖槽表硬编码） | 核心 |
| `src\tower_plan.py` | n×b 个塔的 2×2 占地计算 + 与现有 64 片平面的冲突校验 + 平面图 | 核心 |
| `src\tower_gen.py` | 生成单根塔的独立 `.schem`（2×2 × 全高）—— `python src\tower_gen.py 0 0` → `results\schem\tower_i0_k0.schem` | 核心 |
| `src\tower_side.py` | 一根塔的侧视图（y 竖直；2×2 截面展开成 4 列） | 辅助 |
| `docs\43_火把塔结构规则.md` | 从 `data\torch_tower.schem` 反推的完整规则（朝向、y 映射、零改动验证 10240 格全空气） | 核心 |

### D-10 绘图与出图

| 路径 | 作用 | 类型 |
|---|---|---|
| `src\gallery.py` / `final_fig.py` / `shape_final.py` | 生成效果图；实数 vs 整数管线都映射到 MC 11 档配色；形状自由方案的拟合图与出图 | 辅助 |
| `src\dither.py` / `dither_fig.py` / `n16.py` | 单色显示器 16 级灰的最小单元块 n；关键图；n=16 用真实红石灯材质渲染 | 辅助 |
| `src\draw_annotated.py` / `draw_cmp.py` / `draw_diag.py` / `draw_L1.py` / `draw_oddn.py` / `draw_parity.py` / `draw_scheme.py` / `draw_signalmaps.py` | 8 个出图脚本：标注样图 / inRev 对比 / 奇偶诊断 / 层1 四通道 / 奇数 n 验证 / 奇数 b·n / b=9 方案蓝图 / 四用例信号图 | 辅助 |
| `src\cmp_final.py` / `cmp_variants.py` / `sim_state.py` / `sim_state2.py` / `sim_state3.py` | 四变体 + 断头检测出图；迭代剪枝 + 环位置变体；模拟器状态图（`sim_state3` 带断点标注） | 辅助 |
| `src\merge11_compare.py` / `report_planes.py` / `plane_plan.py` | 16 级 vs MC 11 档差异像素是否下降；检查点对照表；平面平面图 | 辅助 |
| `src\graph_diag.py` / `renn_study.py` / `schem_sr_zoom.py` | 把电路抽象成有向图做诊断；品红配对区两种朝向约定各算一次；环区放大 | 辅助 |

### D-11 一次性/草稿脚本（无文档，勿依赖）

`src\_dbg.py`、`_q1.py`、`_q2.py`、`_q3.py`、`_q4.py`、`_q5.py`、`_u1.py`、`_v1.py`、`_v2.py`、`_v3.py`（10 个下划线开头的小脚本，是开发期粘贴片段；`_q2/_q3/_v2/_v3` 的内容是 JS 片段却用 `.py` 后缀）。**用途不明，不要当工具用，也不要 `python` 去跑。**

### D-12 `viewer\` 版本差异（重点）

| 文件 | 大小 | 能力 | 状态 |
|---|---:|---|---|
| `viewer\plane_sim.html` | 25,643 B | **建板 + 求值 + `legalCheck()` 物理合法性闸门 + 情形Ⅱ 修正**；页面含 `<input type="file">` 可载入模型 json，可下载 json | **✅ 当前版本（"全通"的那一版）**。`sweep_options.js` / `verify_all.js` / `dump_planes64.js` 都 `eval` 这个文件 |
| `viewer\plane_viewer.html` | 19,102 B | 只做**单层平面生成**（n/b 输入 → 画平面），无求值、无导出 | 上一代，仅供对照 |
| `viewer\plane_viewer2.html` | 21,035 B | 同上但改为**「原点基准」**坐标系（v2），无求值、无导出 | 上一代，仅供对照 |
| `viewer\archive\plane_sim_20261024_191414.html` … `_194700.html`（6 个） | 20.1–23.2 KB | 时间戳快照，无 `legalCheck` | 历史 |
| `viewer\archive\plane_sim_pre_v1.html` / `_pre_v3.html` | 23.2 / 23.4 KB | v1 / v3 之前 | 历史 |
| `viewer\archive\plane_sim_before_restore896.html` | 23.2 KB | 恢复 897 上限之前 | 历史 |
| `viewer\archive\plane_sim_inrev_fixed.html` | 23.3 KB | 修 `inRev` 之后 | 历史 |
| `viewer\archive\plane_sim_allpass_1536.html` | 23.3 KB | 文档 `docs\41` §六 称"**是全通那一版**" | 历史（**已被 `viewer\plane_sim.html` 取代**，见 I-4） |

**结论**：现在该用的是 `viewer\plane_sim.html`。它同时具备"1536/1536 全通"与 `legalCheck`（容器 ≤897、比较器 `OutputSignal ≤ 2^31−1`、拓扑全解开），并且已经实现了 `cs === 'II'`（一路 |W| > 882）的独立直通桶结构。`archive\` 里 11 个文件都**没有** `legalCheck`，任何"全通"的旧说法都不能替代当前的合法性判据。

---

## E. 可复用模型与权重

| 路径 | 是什么 | 规模 | 备注 |
|---|---|---|---|
| `99_models\phase1_Wfinal_int.npz` | 一期最终**整数**权重（225→10） | 18,814 B | 一期成品权重；与 `01_...\results\Wfinal_int.npz` 同内容 |
| `99_models\phase1_Wfinal_model.npz` | 一期最终**浮点**权重 | 18,814 B | 训练产物，供重算 |
| `99_models\phase1_Wint_model.npz` | 一期整数权重（draw.html 与 gen_all_v2 的输入） | 9,776 B | **最小、最常被引用的一份** |
| `99_models\phase1_Waug_int.npz` / `_model.npz` | 增广训练版权重 | 18,570 B | 次于 `Wfinal` |
| `99_models\phase1_Wnorm_int.npz` / `_model.npz`、`phase1_Wnorm32_int.npz` | 归一化训练版权重 | 18,814 B | 次于 `Wfinal` |
| `99_models\phase1_Wint_225x10.csv` | 一期权重 CSV（带像素坐标） | 7,618 B | 人可读版 |
| `01_phase1_digit_recognition\results\W225x10.csv`、`Wint_225x10.csv`、`Wint_matrix_15x15.csv`、`weights_10x225.csv` | 各口径的权重表 | 7.6–23.7 KB | 一期 results 里另有整套同名 npz 副本 |
| `02_phase2_avatar\results\coherence\final_lam5_20k\final.pt` | **二期灰度最佳模型**：22→256→256→512，条件相干惩罚 lam=5、20000 步 | 2,188,034 B | 文档称逐像素 0.7005 / 相邻同色 60.7%；网页版是 `webapp\pca22_lam5_model.json`（4.45 MB） |
| `02_phase2_avatar\webapp\pca22_dim4w64L1_model.json` | **二期交付的灰度生成器** 4→64→32 | 100,892 B | 2,304 乘法；三期的 `compiler.py` / `sweep_options.js` 默认吃的就是它 |
| `02_phase2_avatar\webapp\colour_k8snap_model.json` | **二期交付的彩色生成器** 8→128→128→32×8（K=8 吸附调色板） | 1,094,525 B | 50,176 乘法，体积优化重点（层2 占 65%） |
| `02_phase2_avatar\webapp\scorer_quad.json` | **二次打分器**（14 项多项式） | 935 B | 输出坏度，阈值 0.4374 |
| `02_phase2_avatar\webapp\merge_mc11.json` | 16 级 → 11 档地图灰归并表 | 879 B | 查表，0 乘法 |
| `02_phase2_avatar\webapp\styles_gauss.json`、`style_envelopes.json` | 六个风格高斯（均值 + Cholesky 因子）/ 高斯包络 | 4,552 / 32,631 B | 采样用 |
| `02_phase2_avatar\webapp\grayscale_family.json` | 灰度 10 个规模的清单 | 1,772 B | 选型用 |
| `02_phase2_avatar\webapp\preference_reward.json` | 拟合出的 reward 模型（网页用） | 2,631 B | 与 `scorer_accept.json`（2,538 B）、`scorer_anime.json`（2,507 B）并列 |
| `02_phase2_avatar\results\colour\palette\P16.pt` / `P32.pt` / `P8.pt`、`S16/S32/S8.pt` | 调色板分类器检查点（P=主，S=对照） | 206–603 KB | 对应 `P16.json` / `P32.json` / `P8.json` |
| `02_phase2_avatar\results\colour\stage2_E{1,2,3,4,5,6,8}.pt` | 彩色 stage-2 实验矩阵的 7 个检查点 | 123–144 KB | 结果在 `stage2_readout.json` |
| `02_phase2_avatar\results\colour\colour_knobs_12000.npz` | 彩色旋钮（12000 步） | 355,508 B | — |
| `02_phase2_avatar\results\**\final.pt`（**共 207 个 `.pt`**） | 每一条实验线的检查点，命名即实验名（`coherence\`、`gan_v2\`、`sweep\`、`shrink\`、`styled\`、`distill\`…） | 1.2–2.9 MB 各 | **绝大多数是扫描中间态**，只有上表列出的少数是"结论级" |
| `03_phase3_quantized_network\results\netlist_grayscale.json` | **三期灰度网表**（`compiler.py` 的产出：位权重、C_min、P_i、c_i/T_i、理论范围） | 84,989 B | 三期最核心的产物 |
| `03_phase3_quantized_network\results\spec_grayscale.json` / `spec_colour.json` / `spec_scorer_quad.json` | 硬件无关规格（层形状、乘法数、权重统计、多 bit 量化误差、激活范围） | 2,930 / 4,737 / 1,726 B | 三期 P0 交付 |
| `03_phase3_quantized_network\results\planes64.json` | 层1 全 64 片原始平面数据 | 1,149,773 B | `schem_stack64.py` 的输入 |
| `03_phase3_quantized_network\results\planes64_relu.json` | 同上，ReLU 版（白环已减 P−1） | 1,154,357 B | — |
| `03_phase3_quantized_network\results\planes_L2.json` | 层2 全部 32 通道的板 | 6,410,502 B | 层2 最大产物 |
| `03_phase3_quantized_network\results\towers32.json` / `towers_L2.json` | 层1 的 32 根火把塔占地 / 层2 的塔 | 86,310 / 805,439 B | — |
| `03_phase3_quantized_network\results\py_compile_reference.json` | Python 侧编译参考值（`verify_all.js` 的对账基准） | 227,145 B | 金标准 |
| `03_phase3_quantized_network\results\py_forward_cases.json` | Python 整数前向的测试样例 | 5,258 B | 金标准 |
| `03_phase3_quantized_network\results\board_dump.json` | 整块板含数值，供 `schem_export.py` 写 `.schem` | 86,437 B | — |
| `03_phase3_quantized_network\results\predict_X.json` | `predict.js` 的随机输入检查点 | 7,665 B | 游戏内对照用 |
| `03_phase3_quantized_network\results\final_spec.json` | 投掷器形状自由方案的最终参数 | 24,274 B | `final_lahc.py` 产出 |
| `03_phase3_quantized_network\results\direct_spec.json` / `direct_spec_m8.json` / `direct_shape_spec.json` | 直接法三种定稿的参数表 | 6,673 / 4,127 / 26,556 B | — |
| `03_phase3_quantized_network\results\slots_spec.json` | N≤9 格数搜索结果 | 28,169 B | — |
| `03_phase3_quantized_network\results\dropper_spec.json` / `src\results\dropper_shapes.json` | 投掷器方案参数表 / 形状约束 | 6,119 / 1,004 B | — |
| `03_phase3_quantized_network\results\error_budget.json` | 三类误差与最优 b | 818 B | — |
| `03_phase3_quantized_network\src\results\container_reach.json` / `container_gaps.json` | 容器可达性结果（**注意在 `src\results\`，不在 `results\`**） | 489 / 1,150 B | — |
| `03_phase3_quantized_network\src\results\barrel_palette.json` / `barrel_formula.json` | 容器编码调色板与公式 | 15,288 / 3,923 B | — |
| `03_phase3_quantized_network\src\results\volume_final.json` / `volume_compare.json` | 体积核算定稿 | 249 / 85 B | — |

---

## F. 实物与可建造产物

### F-1 `.schem`（Sponge v3 / WorldEdit 格式，共 15 个）

| 路径 | 是什么 | 是否游戏内验证过（出处） |
|---|---|---|
| `01_phase1...` — 无 `.schem` | 一期的实物是**数据包**而不是 schem，见 F-2 | — |
| `03_phase3_quantized_network\results\schem\layer1_ch0_C0_all0.schem` | 层1 通道0 单片，输入全 0 | **是** —— `docs\40` 标题行"平面参数全通（1536/1536），游戏内四次测试通过"；`docs\41` §二"游戏内四次测试 全部通过（手动）" |
| `...\layer1_ch0_C1_all1.schem` | 层1 通道0 单片，输入全 1 | **是**，同上 |
| `...\layer1_ch0_C2_1010.schem` | 层1 通道0 单片，输入 1010 | **是**，同上 |
| `...\layer1_ch0_C3_0101.schem` | 层1 通道0 单片，输入 0101 | **是**，同上 |
| `...\layer1_stack64.schem` | 层1 全部 64 片竖直堆叠（14×128×31；比较器 6730、容器 1948、BlockEntities 8678） | **仅 NBT 回读对账**：`docs\42` §五 —— 12864 格 0 处不一致、不可达 0、`powered` 一致（6725 true / 5 false，合法）；文档**未**声明游戏内验证。注意 6730 是**情形Ⅱ修正后**的数字（修正前 6916，47 个 Δ 桶不可达） |
| `...\layer1_full.schem` | 层1 完整网络：64 片平面 + 32 根火把塔 | **仅回读对账**（`schem_verify_full.py`，见 `docs\43` §复现） |
| `...\layer2_full.schem` | 层2 完整网络（368 KB，最大的一份） | **仅回读对账**（`docs\43`：`schem_full.py results/planes_L2.json results/towers_L2.json ...`） |
| `...\layer1_relu.schem` | 层1 的 ReLU 除法器版 | 未在文档中找到验证声明 → **验证状态不明** |
| `...\tower_i0_k0.schem` | 单根火把塔（2×2 × 130，i0/k0 样例） | 结构规则从 `data\torch_tower.schem` 反推并零改动验证（`docs\43` §"零改动已验证：32 塔 × 64 通道 × 5 块 = 10240 个新增位置全部是空气"）；**塔本身未声明游戏内验证** |
| `03_phase3_quantized_network\data\hand_write_digital.schem` | 一期手写数字网络的参考 schem（50 KB） | 参考输入，`learn_schem.py` 解析它 |
| `...\data\rsnn.schem` | 单层强模 LUT 参考件（`schem_plane.py` 反推规则的来源） | 参考输入 |
| `...\data\renn_sr.schem` | 品红配对区参考件（`renn_study.py` / `schem_sr_zoom.py`） | 参考输入 |
| `...\data\relu.schem` | 8 位除法器（restoring division）参考件（`relu_probe.py` 反解） | 参考输入 |
| `...\data\layer1_b8.schem` | 层1 b=8 小参考件（516 B） | 参考输入 |
| `...\data\torch_tower.schem` | 火把塔样例（2×16×5 = 8 通道 × 3 平面行；亮灭是示意值） | 参考输入（`docs\43` L8） |

### F-2 `.mcfunction` / 数据包

| 路径 | 是什么 | 是否游戏内验证过（出处） |
|---|---|---|
| `01_phase1_digit_recognition\datapack\tiny_cnn\pack.mcmeta` | 唯一数据包的描述文件 | — |
| `...\data\deploy_weights\function\deploy_all_layers.mcfunction` | **2340 条**放置指令（实测：非注释非空行 = 2340，总 2550 行）；入口 `/function deploy_weights:deploy_all_layers`；层由 y 决定、y 步长 2（第10层 y=−1 … 第1层 y=−19） | **是** —— 一期状态"已实装完成"（`docs\00_文档索引.md` L9；根 `README.md` 一期"已完成"、体积 32×52×47、≤15 秒） |
| `...\data\deploy_weights\function\layer10_digit9_full.mcfunction` | 第10层（数字9）完整放置：225 权重容器 + 8 个 G_{i,0} + F_0（273 行） | 同上（属成品的一部分） |
| `...\data\deploy_weights\function\place_layer10_digit9.mcfunction` | 第10层分步放置 | 同上 |
| `...\data\deploy_weights\function\place_top_layer.mcfunction` | 顶层分步放置 | 同上 |
| `...\data\vessel\function\barrel_0.mcfunction` … `barrel_511.mcfunction`（**512 个**） | 单个木桶 `setblock`：`minecraft:barrel{Items:[…],CustomName:'"signal: S, item: N"'}`；编码 `signal = 1 + floor(14·fill/27)`，木桶 27 格 × 每格 ≤64 → `signal ≤ 897` | **是**（同数据包），并被三期 `barrel_probe.py` 反读核对 |
| `01_phase1_digit_recognition\results\layer10_bias_snippet.mcfunction` | 第10层偏置的片段（非独立数据包） | 片段，未单独验证 |

> **三期没有生成 `.mcfunction` 数据包** —— 三期的交付是 `.schem`。三期文档里"游戏内四次测试"指的是 F-1 中 4 个 `layer1_ch0_C*_*.schem`。

---

## G. 关键图片产物（每期 5~10 张）

### 一期（`01_phase1_digit_recognition\results\`）

| 路径 | 一句话 |
|---|---|
| `results\heatmap_all_boost.png` | 10 个数字的 15×15 权重热图（增强对比度版），一图看清"每个数字在看哪些像素"—— 一期最有说服力的图 |
| `results\mnist15_binarized_900.png` | 训练集 900 个数字缩到 15×15 并二值化后的真实长相（数据管线的样子） |
| `results\mnist15_by_digit.png` | 同上按数字 0..9 分组，看类别可分性 |
| `results\pick10_digits.png` | 挑出的 10 个识别正确样本（覆盖 0..9），配每层 F 值 |
| `results\bias_chart.png` | 10 层支线偏置 G_{i,0} 的分布，解释 F 对齐到 `F = z + 128` |
| `results\G_i0_chart.png` | G_{i,0} 的逐通道取值图，数字 9 的偏置推导依据 |
| `results\heatmap_row0_compare.png` | 第 0 行权重的两种口径对比（归一化前后） |
| `results\heatmap_digit9.png` | 数字 9 单独一张（奇数/偶数层极性最反直觉的一个） |

### 二期（`02_phase2_avatar\results\`）

| 路径 | 一句话 |
|---|---|
| `results\PROGRESS\PHASE2_PROGRESS_V2.png` | 二期进度页（用户要求重绘版），图 + 数字并列的收官主图 |
| `results\coherence\final_lam5_20k\COMPARE_real_mode_sample.png` | **最佳模型**（lam=5, 20k）与真实脸在两种解码模式下并排 —— 结论级对比 |
| `results\pca4map\MAP_p3_p0_5.png`（同组 8 张 `MAP_p3_{p,m}{0,1,2,3}_5.png`） | 4-PCA 平面图：每个平面 8×8 = 64 格全脸，1,372 条四维标签的来源 |
| `results\colour\COLOUR_FINAL.png` | 带全局拉伸的彩色成品（硬件实际会显示的样子） |
| `results\colour\GRAY_VS_COLOUR.png` | 同缩放下灰度 vs 彩色并排 —— 回答"彩色值不值 21 倍成本" |
| `results\preference\W128_BEFORE_AFTER.png` | w128 偏好模型上线前后对比（筛选增益的直接证据） |
| `results\shrink\VISUAL\COLOR16_real_mode_sample.png` | 16 色彩 / 32 通道压缩架构的真实产出（参数少 61% 的那一支） |
| `results\distill\VISUAL\DISTILL_COMPARE.png` | 512 教师 vs 32 输出学生在**同一批** hold-out 上的对比（蒸馏是否掉质量） |
| `results\dimmix\DIM_YIELD.png` | 维度阶梯（PCA-k）与良率的关系曲线 —— 三期选 k=4 的依据 |
| `results\kernel_style\kernel_style\KERNELS_what_they_see.png` | 学到的 9 个微小卷积核各自响应什么（风格是否自动涌现） |

### 三期（`03_phase3_quantized_network\results\`）

| 路径 | 一句话 |
|---|---|
| `results\annotated_layout.png` | 标注样图：原点、j、相对/绝对坐标、各结构区域 —— 读图的唯一正确起点（`docs\41` §六 列为关键文件） |
| `results\signal_maps_4cases.png` | 四个测试输入下的比较器全局信号图（`docs\41` §六 列为关键文件） |
| `results\plane_plan.png` | 层1 一片平面的俯视平面图，火把塔选址依据 |
| `results\plane_full_v2.png` | 层1 单平面逐块完整图（按用户指正修正后的 v2） |
| `results\schem_annotated.png` | `rsnn.schem` 的标注俯视图：分区、环、颜色-位对应 |
| `results\tower_side_i0_k0.png` | 火把塔侧视图（y 竖直，2×2 截面展开 4 列）—— 塔结构的唯一可读图 |
| `results\sim_state.png` | 模拟器静态状态：每个比较器标上输出值（调试断点的标准图） |
| `results\final_mc11.png` | 最终对比：整数管线映射到 MC 11 档配色的成品脸 |
| `results\n16_mc_view.png` | n=16 单色头像用**真实红石灯材质**渲染（"到底能不能看"的答案） |
| `results\e2e_compare_b8.png` | b=8 端到端：实数管线 vs 整数管线，误差的最终可视化 |

> 三期 `results\` 共 109 张 PNG，上表之外多为参数扫描/奇偶情形/方案对比的过程图（`oddn_*` 7 张、`sim_*` 12 张、`diag_*` 4 张、`variant_*` 4 张、`batch_*` 7 张等），按目录归组即可，不逐条列。

---

## H. 环境与依赖

| 项 | 值 | 证据 |
|---|---|---|
| Python | **3.12.14**（MSC v.1944 64-bit） | `C:\Users\lenovo\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe`，实测 `sys.version` |
| Python 第三方（主运行时） | numpy **2.3.5**、pandas 3.0.1、Pillow 12.3.0、python-docx、python-pptx、openpyxl、lxml、XlsxWriter | `load_workspace_dependencies` |
| torch | **主运行时没有**（`find_spec('torch')` → False）；**`02_phase2_avatar\.venv` 里有 torch 2.14.1** | 实测 + `.venv\Lib\site-packages\torch-2.14.1.dist-info` |
| `.venv` 位置 | **只有一处**：`02_phase2_avatar\.venv`（Python 3.12.14，`--without-pip --system-site-packages`，`include-system-site-packages = true`） | `pyvenv.cfg` |
| `.venv` 内容 | torch 2.14.1、filelock 4.2.0、fsspec 2026.9.0、functorch、jinja2 3.1.6、markupsafe 3.0.4、mpmath 1.4.1、networkx 3.7、sympy 1.14.0、pip 25.0.1、setuptools 84.0.0、typing_extensions 4.16.0 | `site-packages` 目录列表 |
| 三期环境 | **借用** `02_phase2_avatar\.venv`；三期**没有自己的 venv** | `03_...\README.md` §目录约定："运行环境暂借 `../02_phase2_avatar/.venv`" |
| Node.js | **v24.21.0**，但**不在 PATH 上**；必须用 `C:\Users\lenovo\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\node\bin\node.exe` | 实测 `node --version` 报"无法识别"，绝对路径运行得 v24.21.0 |
| Node 是否需要 | **三期必须**（`sweep_options.js` / `verify_all.js` / `verify_planes.js` / `verify_towers.js` / `dump_planes*.js` / `predict.js` 是验收的唯一途径）；**二期可选**（`serve_webapp.js` 起服务、`test_*.js` 30 个页面回归、`render_page_faces.js` / `js_forward_check.js` 数值校验）；**一期不需要**（`src\` 里 0 个 `.js`） | 各期 `src\` 的 `.js` 计数：一期 0、二期 37、三期 13 |
| node 依赖包 | 见 `C:\Users\lenovo\.dsh\...\dependencies\node\node_modules`（harness 自带） | — |
| 一期额外数据 | `01_phase1_digit_recognition\data_mnist\data\MNIST\raw` 有完整 8 个 idx 文件（train/t10k images+labels，含 `.gz`，train-images 47 MB） | 实测 |
| 二期额外数据 | `data_v2_canonical\`（规范集，35 MB）+ `data\`（两版原始，含 177 MB 的 `faces8_rgb_v2.npy`）；`99_mc_texture\1.21.11-Fabric 0.19.5.jar`（31 MB，`palette_v*.py` / `mc_*.py` 的输入） | 实测 |
| 三期额外数据 | `data\` 6 个 `.schem` 参考件；`results\` 大量 JSON | 实测 |
| 磁盘/时间 | 二期 `results\` 有 **207 个 `.pt`**（合计约 400 MB+）、`results\` 40 个一级子目录；全二期 `results\` 只有 8 个文件在根层 | 实测 |

---

## I. 清点过程中发现的问题

### I-1 【严重】一期约 50 个脚本写死了已不存在的 `D:\czm我的世界\...` 路径

`D:\` 存在，但 `D:\czm我的世界\文件\Comparator Network\` **不存在**（实测 `Test-Path` 全 False）。受影响的脚本包括几乎全部 MNIST 训练/评估脚本与全部鸢尾花脚本：

- `RAW = r'D:\czm我的世界\文件\Comparator Network\tiny_cnn\data\MNIST\raw'`
  —— `int_weight_model.py`、`mnist_k1.py`、`mnist_sweep.py`、`make_canvas.py`、`gen_all_v2.py`、`gen_all_layers.py`、`viz_mnist15.py`、`honest_eval.py`、`stroke_stats2.py`、`stroke_stats3.py`、`mc_10layers.py`、`mc_container_signals.py`、`mc_comparable.py`、`mc_F0_align_v2.py`、`mc_digit9_check.py`、`mc_digit9_final.py`、`digit_code*.py`、`train_normalized.py`、`train_final.py`、`retrain_aug.py`、`pick10.py`、`trace_sample.py`、`quant_range.py`、`relu_split_cost.py`、`split_cost.py`、`verify_buildable.py`、`final_tanh_buildable.py`、`analyze_225x10.py`、`boldness_test.py`、`compare_protocols.py`、`norm_compare.py`、`width_scan*.py`（部分）等
- `D:\...\StrongAnalog_LUT_Network\results\iris_teacher.json` —— `iris_compile.py`、`iris_compile_v2.py`、`iris_compiled.py`、`iris_compiled_v3.py`、`iris_tune4bit*.py`（4 个）、`diag_align.py`、`diag_iris*.py`（3 个）、`iris_verify.py`
- `D:\...\DT_Redstone_Research_Archive_20261001\DT_baseline_recovery\teachers\` —— `probe_step_compare.py`、`probe_teacher_forward.py`

**但这些数据在本仓库里都有副本**，可以直接改路径修好：

| 死路径指向 | 本地真实位置 |
|---|---|
| `tiny_cnn\data\MNIST\raw` | `01_phase1_digit_recognition\data_mnist\data\MNIST\raw`（8 个文件齐全） |
| `DT_baseline_recovery\teachers\N4_MLP_T1.json` / `.npy` | `00_shared\archive_dt_20261001\DT_baseline_recovery\teachers\`（存在，实测 True） |
| `StrongAnalog_LUT_Network\results\iris_teacher.json` | **仓库内未找到等价文件**；不过 `01_...\results\iris_teacher.json`（10,940 B）存在，是同一实验的输出 |

另外 `neuron_design_constrained.py` 的路径推导是 `dirname×3` → `E:\redstone_nn\DT_Redstone_Research_Archive_20261001\...`，也**不存在**（正确应是 `E:\redstone_nn\00_shared\archive_dt_20261001\...`）。

**结论**：根 `README.md` 的"脚本内不硬编码绝对路径（已核查一期 130 个脚本，零硬编码）"这条**不成立**，且"整个仓库可以整体移动位置而不破坏任何脚本"也不成立。一期只有约 1/3 的脚本（`current` 的 `strong_lut_solver`、`example_run`、`final_audit`、`neuron_design*`、`multilayer_greedy`、`mc_container_correct`、`mc_g1_*`、`gen_deploy*`、`gen_layer10*`、`diag_shapes`、`controlled_readout`、`probe_int_forward`、`probe_output_layer`、`probe_diagnose` 等）是相对路径。

### I-2 【中】三期的 JS 与部分 Python 写死了 `E:/redstone_nn/...`

- `src\sweep_options.js`：`process.argv[2] || 'E:/redstone_nn/03_phase3_quantized_network/viewer/plane_sim.html'`，且第 11 行写死 `E:/redstone_nn/02_phase2_avatar/webapp/pca22_dim4w64L1_model.json`
- `src\verify_all.js`（3 处）、`src\dump_planes_relu.js`、`src\schem_stack64.py`（`ROOT = r"E:\redstone_nn\03_phase3_quantized_network"`）

今天能跑（仓库就在这个位置），但**整体搬移必坏**。三期的 Python 里这两种风格混杂：`spec_extract.py` / `compiler.py` 用 `HERE/ROOT`（可搬），`schem_stack64.py` 写死（不可搬）。

### I-3 【中】`make_canvas.py` 的输出路径与仓库里的成品不一致

`make_canvas.py` 里 `OUT = os.path.join(ROOT, 'draw.html')` → 会写到 `01_phase1_digit_recognition\draw.html`（**不存在**），而仓库里的成品在 `01_phase1_digit_recognition\build\draw.html`（**已核实存在**）。重跑这个脚本会在 repo 根多出一个 `draw.html`，而不是刷新 `build\draw.html`。

### I-4 【中】文档里"全通那一版"的指认已过期

`03_phase3_quantized_network\docs\41_当前状态与待办.md` L228 写：

> `viewer/archive/` 有 11 个版本。**`plane_sim_allpass_1536.html` 是全通那一版。**

同文件 §六 L211 又写：

> `viewer/plane_sim.html` ← 建板 + 求值（1536/1536）

实测：`viewer\plane_sim.html`（25,643 B，比 archive 里任何一版都大）**同时**具备 (a) `legalCheck()`（`archive\` 11 个版本全都没有）和 (b) `cs==='II'` 的独立直通桶分支（`docs\42` 描述的情形Ⅱ 修正）。所以按 `docs\42` 之后的实际状态，**当前版本是 `viewer\plane_sim.html`，`archive\plane_sim_allpass_1536.html` 是历史**。文档 §六 的存档说明没有跟着 `docs\42` 更新，容易让新会话去用旧文件。

### I-5 【中】`_q*.py` 等草稿脚本内容是 JavaScript

`03_...\src\` 下 10 个下划线开头的文件（`_dbg.py`、`_q1.._q5`、`_u1`、`_v1.._v3`）没有 docstring。其中 `_q2.py`（868 B）、`_q3.py`（812 B）、`_v2.py`（2,374 B）、`_v3.py`（2,194 B）的内容明显是 **JS 片段**（`cmp(xa+1,zr+4,'E','compare',true);`、`/* ---------- 权重 → 整数 U ---------- */`），却用 `.py` 后缀。`_q4.py` / `_q5.py` 只有 395 / 381 B。**用途不明，建议不要当工具依赖，也不要 `python` 去跑。**

### I-6 【轻】`export_model_js.py` 是空占位

`01_...\src\export_model_js.py` 全文只有一个词 `placeholder`（**11 字节**）。名字暗示"把模型导出成 JS"，但**没有任何实现**。

### I-7 【轻】根 README 的目录树已过期两处

| README 写的 | 实际 |
|---|---|
| `03_phase3_placeholder\   三期预留`（第 50 行） | **不存在**；实际是 `03_phase3_quantized_network\` |
| `01_...\src\  一期全部脚本（130 个）`（第 35、155 行） | `src\` 下实际 **115 个** `.py`；连同 `archive\src\`(20) + `data_mnist\`(4) 共 **139 个** |
| `02_phase2_avatar\  二期：生成头像（待启动）` | 二期已**完成**（二期自己的 `README.md` 写"状态：已完成"，且交付物齐备） |

### I-8 【轻】同名的 `results\` 与 `src\results\`

三期的容器可达性/体积/调色板结果被写在 **`03_...\src\results\`**（8 个 JSON），而其余结果在 `03_...\results\`（**43** 个非 PNG + 109 PNG）。同一概念分裂在两个目录，容易找错：
- `src\results\`：`barrel_formula.json`、`barrel_palette.json`、`container_gaps.json`、`container_reach.json`、`dropper_shapes.json`、`schem_keys.json`、`volume_compare.json`、`volume_final.json`
- `results\`：其余全部

### I-9 【轻】`results\` 里存在大量"结论级"与"扫描中间态"混放的 `.pt`

二期 `results\` 有 **207 个 `final.pt`**，命名即实验名。其中只有少数几个是文档认可的结论模型（`coherence\final_lam5_20k\final.pt` 是灰度最佳），其余是 `pilot_*` / `t_*` 试验。**没有索引文件说明哪个是"当前最佳"** —— `results\DOC_INDEX.md` 索引的是文档，不是检查点。

### I-10 【轻】一期缺 `README.md`

`01_phase1_digit_recognition\` 根目录**没有** `README.md`（只有 `archive\README.md`）。根 `README.md` 与 `docs\00_文档索引.md` 充当了它的角色，但"各期都有 README"这个惯例在一期是断的。二期有 4 份（`README.md`、`PHASE2_TIMELINE.md`、`RESEARCH_ROUTE.md`、`SUMMARY.md`），三期有 1 份。

### I-11 【轻】文档声明的"2340 条放置指令"已核实为真

`deploy_all_layers.mcfunction` 共 2550 行，其中注释 199 行、空行 11 行，**非注释非空行 = 2340**，与 `docs\00_文档索引.md` L41 和根 `README.md` 的"2340 条"**完全一致**。这条不是问题，记录在此以免下一个会话重复核对。

### I-12 【轻】二期 `README.md` 的相对路径写法

`02_phase2_avatar\README.md` 的交付物表里写 `webapp\label.html`、`label_colour.html` 等不带目录前缀，逐条 `Test-Path` 时只有 `label_colour.html` 在上级目录找不到 —— 实际文件都在 `webapp\` 下，**不是缺失**，只是文档里的路径省略了 `webapp\` 前缀，机械校验会误报。

---

*清点完成。本文所有路径均经 `Test-Path` 核实（791 个反引号路径 token 全部落地，裸文件名在各自分组声明的"路径前缀省略"约定下同样已逐一核对）。*

*规模：A 节 **15** 条；B 节 5 组 **56** 行；C 节 8 组 **84** 行；D 节 12 组 **80** 行（含 `viewer\` 版本表 13 行）；E 节 **37** 行；F 节 **23** 行；G 节 **28** 张图（一期 8 / 二期 10 / 三期 10）；H 节 **14** 行；I 节 **12** 条。*
