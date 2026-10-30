# 二期文档摘要（供四期整理使用）

> 范围：`02_phase2_avatar/docs/` 全部 28 篇 + 期根 4 篇（`PHASE2_TIMELINE.md`、`SUMMARY.md`、`RESEARCH_ROUTE.md`、`README.md`），共 32 篇、5,098 行。
> 引用约定：`docs/xx.md:行` 指 `02_phase2_avatar/docs/` 下的文件；不带 `docs/` 前缀的（`PHASE2_TIMELINE.md:227` 等）指期根文件；`results/...json:行` 指原始结果文件。
> 事实核对：凡能从 `results/` 下的 JSON 直接复核的数字，均已在文档出处之外另标 JSON 出处（如 `results/dimmix/round4_stats.json:11`）；未标 JSON 的仅表示"文档如此陈述"。
> 口径纪律：二期的"准确率""良率""可接受率"是**三个不同量**，且同一量在不同轮次/不同配方下不可比。凡本文说"不可比"，均指原文自己声明或经数据核对后口径不同。
> 二期时间跨度：2026-10-16 15:46 – 10-19（`README.md:3`、`PHASE2_TIMELINE.md:27`）。

---

## A. 时间线

二期有两个**互相独立**的"轮次"编号系统，先说明，否则下面每一行都会被误读：

1. **打标轮次 `round`**（1–12）：写在 `results/preference/labels.jsonl` 的 `round` 字段里，共 3,752 条（`RESEARCH_ROUTE.md:20-34`）。经本人复算，逐轮条数与 `RESEARCH_ROUTE.md` 的表完全一致，另有 743 条 `round=None` 的早期标签。
2. **文档内"第 X 轮"**：`ROUND2_RESULT.md` / `ROUND2_POSTPROCESS_RESULT.md` / `ROUND3_RESULT.md` 说的"轮"**不等于**上面的 `round`。其中：
   - `ROUND2_RESULT.md` 的"第二轮"= w128 的 guided/explore 打标，158 条，落在 `round=None` 里（`docs/ROUND2_RESULT.md:3`、`:31`；数据见 `results/preference/rounds.json`）；
   - `ROUND2_POSTPROCESS_RESULT.md` 的"第 2 轮"= dim4x/dim4/dim22 三臂，334 条，就是 `round=2`（`docs/ROUND2_POSTPROCESS_RESULT.md:3`；`round2_stats.json` 三臂 n=110/103/121）；
   - 两者**同叫 ROUND2，却是两个不同实验**（详见 H-1）。

| 日期或阶段 | 事件 | 出处 |
|---|---|---|
| 10-16 15:46 | 二期开工，立《已知死路与重启指引》，文件头写"状态：待启动" | `docs/00_已知死路与重启指引.md:1-3` |
| 10-16 15:46–16:44 | 阶段 1：资产审计、路线规划、环境重建（独立 venv + CPU torch） | `PHASE2_TIMELINE.md:35` |
| 10-16 16:14 | 数据管线独立审计报告落盘（19.8 KB） | `results/DOC_INDEX.md:13` |
| 10-16 16:44 | 《盘点与重启建议》：撤销"GAN 死路"的数据前提，改用实测重分类九条死路 | `docs/01_盘点与重启建议.md:279-299` |
| 10-16 17:19–22:04 | 阶段 2：生成式表述搜索；**八臂条件横向扫描 19:48–20:04**（uncond / half / code3 / code5 / pca4 / pca8 / pca16 / pca22），每臂 1,500 步 | `PHASE2_TIMELINE.md:36`、`:89-98` |
| 10-16 19:37 | 核风格聚类条件（`kernel_style`）与 PCA 扫描**同期并行**，不是"早于 PCA" | `PHASE2_TIMELINE.md:412` |
| 10-16 22:03–10-17 14:47 | 阶段 3：空间一致性治理（杂色量化、相干性 λ 扫描、噪声 ramp）+ 人工优选 | `PHASE2_TIMELINE.md:37` |
| 10-17 11:07 | `webapp/README.md`（交互实验台说明） | `results/DOC_INDEX.md:26` |
| 10-17 11:48 | 输入维数路线启动（早于输出压缩定稿 12:03，两条线并行） | `PHASE2_TIMELINE.md:46` |
| 10-17 12:03–14:00 | 阶段 4/批次"512→32"工作段：12:03 PLAN → 12:44 RESULTS → 13:14–13:22 混合密度补充实验 → 13:22 FINAL+INDEX | `PHASE2_TIMELINE.md:38`、`docs/512_to_32_FINAL.md:3`、`:102` |
| 10-17 14:33 | w128 打标 111 条，留出集 AUC 0.795（`round=None`） | `docs/W128_LABELING_RESULT.md:3`、`:20` |
| 10-17 14:47 | "第二轮"打标：120 guided + 38 explore 交错 | `docs/ROUND2_RESULT.md:3` |
| 10-17 14:56 / 15:08 | 收尾进度图 + 灰阶方块 v1 / v5（v5 取代 v1） | `results/DOC_INDEX.md:61`、`:66` |
| 10-17 15:15 | PCA-4/8 机理分析（平滑 vs 准确、噪声槽位闲置） | `results/DOC_INDEX.md:71` |
| 10-17 15:23 | 维数 × 良率实验设计（盲测混合、`label.html?mix=1`） | `docs/DIM_YIELD_DESIGN.md:40-52` |
| 10-17 16:33 | 数据对齐 bug 记录（量化平局舍入导致行错位） | `docs/ALIGNMENT_BUG.md:1-34` |
| 10-17 16:59 | 三个维数模型训练完成（PCA-4/8/22 同配方），准备开标 | `docs/DIM_LADDER_TRAINED.md:3` |
| 10-17 17:48 | 发灰问题诊断（corr 0.646 / 斜率 0.684 拆解） | `docs/CONTRAST_ANALYSIS.md:38-48` |
| 10-17 18:06 | 拉伸目标（训练）vs 拉伸后处理（显示）实验设计 | `docs/STRETCH_EXPERIMENT.md:1`、`:96-112` |
| 10-17 18:22 | 第 2 轮盲测（`round=2`，334 条）：**后处理 min-max 把 PCA-4 从 20.9% 提到 62.1%** | `docs/ROUND2_POSTPROCESS_RESULT.md:1-13` |
| 10-17 18:56 | 第 3 轮（`round=3`，226 条）：**烘焙拉伸失败**（30.3%），意外发现固定增益 ×1.7 | `docs/ROUND3_RESULT.md:9-21`、`:58-71` |
| 10-17 19:04 | dim4g（固定增益 ×1.7）作为第 4 臂就绪 | `docs/DIM4G_READY.md:41-51` |
| 10-17 19:20 / 19:26 | 四旋钮命名（PC0 明度/PC1 发肤/PC2 眼型/PC3 脸型）+ 274 条标签的纯化 | `docs/FOUR_KNOBS.md:9-16`、`docs/KNOBS_AND_PURIFY.md:32-66` |
| 10-17 19:45 | 风格打标分析 + 彩色可行性（明度推不出颜色） | `docs/STYLE_AND_COLOUR.md:24-31`、`:106-118` |
| 10-17 21:22 / 21:33 / 21:49 | 风格开关（线性探针）→ 风格特化 → 评分器定稿（线性/二次/MLP 对比） | `docs/STYLE_SWITCHES_MLP_COLOUR.md:7-14`、`docs/STYLE_CONDITIONED.md:11-18`、`docs/FILTER_FINAL.md:9-14` |
| 10-17 21:49–10-18 15:00 | 阶段 7：评价与筛选体系校准 + MC 颜色体系（62 地图基色 ×4 档、61 方块均色） | `PHASE2_TIMELINE.md:41`、`:251` |
| 10-17 22:17 → 22:22 → 22:26 | 彩色两阶段**设计 → 结果（E6 胜、两阶段被否）→ 夜间记录（E8、代理良率、彩色筛选）** | `results/DOC_INDEX.md:151`、`:156`、`:161` |
| 10-18 10:00–14:30 | 阶段 6：网络定宽定层 → **4-64-32，2,304 乘法**；三臂盲测测不出差异 | `PHASE2_TIMELINE.md:40`、`:218-239` |
| 10-18 17:53–10-19 | 阶段 8：4,096 点全空间地图（1,372 条标注）、矩形夹取、打分器升级、6 个风格 | `PHASE2_TIMELINE.md:42`、`:297-349` |
| 10-18 17:54 | `SUMMARY.md` 定稿（彩色技术摘要，含 round 8/11/12） | `results/DOC_INDEX.md:167-169` |
| 10-18 22:31 | `RESEARCH_ROUTE.md` 定稿（早期路线复盘） | `results/DOC_INDEX.md:174` |
| 10-19 | `README.md` + `PHASE2_TIMELINE.md` 收官；后者自述为时间线主文档 | `README.md:1-3`、`:11-17`、`PHASE2_TIMELINE.md:27` |
| 12 轮打标全程 | round 1–12 共 3,752 条 + 1,372 条 4 维网格标注 | `RESEARCH_ROUTE.md:20-36` |
| 修复/返工插入点 | ①对齐 bug（16:33）②网页"3 层硬编码"→任意层数（`512_to_32_INDEX.md:50-54`）③`fit_reward.py` 跨架构混标签（`:52-53`）④导出权重转置/0–15 当 0–255（`:54`） | `docs/512_to_32_INDEX.md:48-54` |

**批次顺序（按实验批次而非文档顺序）**

1. 一批：审计与复算（restart_probe，`docs/01_盘点与重启建议.md:233-247`）。
2. 二批：八臂条件扫描（10-16 晚）→ 定 PCA 条件（`PHASE2_TIMELINE.md:104-114`）。
3. 三批：512→32 输出压缩（10-17 午间，蒸馏/TV/收缩/混合密度四小组）（`docs/512_to_32_RESULTS.md:33-81`、`512_to_32_FINAL.md:58-125`）。
4. 四批：维数阶梯 dim4/dim8/dim22（10-17 下午）+ 后处理/烘焙/增益三轮（round 2/3/4）（`docs/DIM_LADDER_TRAINED.md:11-16`、`results/dimmix/round{2,3,4}_stats.json`）。
5. 五批：风格与评分器（round 5、10-17 傍晚）（`docs/FILTER_FINAL.md:1-3`）。
6. 六批：彩色两阶段 E1–E8（10-17 夜间）→ 次日变单阶段（`docs/COLOUR_EXPERIMENT_RESULT.md:23-34`、`docs/NIGHT_COLOUR_LOG.md:39-46`）。
7. 七批：结构定宽定层（round 7/9，10-18 上午）（`PHASE2_TIMELINE.md:234-237`）。
8. 八批：彩色调色板 K 扫描（round 8）→ 三种配色方案（round 11）→ 基线/C1/C2（round 12）（`SUMMARY.md:99-144`）。
9. 九批：MC 灰归并（round 10，405 条）（`RESEARCH_ROUTE.md:88`、`SUMMARY.md:63-83`）。
10. 十批：4,096 点全空间 + 风格 + 夹取（10-18 17:53 起）（`PHASE2_TIMELINE.md:293-355`）。

---

## B. 逐篇职责与状态

| 文件 | 一句话内容 | 结论状态 |
|---|---|---|
| `PHASE2_TIMELINE.md` | 八阶段主时间线：任务/做法/结果/指向 + 勘误附录 A；二期唯一完整的叙事主线 | 定稿有效（但 §5 表格有一处数值错位，见 H-5） |
| `SUMMARY.md` | 灰度 + 彩色双线技术摘要：结构、乘法、可接受率、调色板、五次失败的彩色提升尝试 | 定稿有效 |
| `RESEARCH_ROUTE.md` | 早期路线复盘：12 轮标签总量、筛选三次翻车、PCA4 全空间、六条方法论 | 实验记录(结论仍成立)；被 `PHASE2_TIMELINE.md:404` 标注为"其第 1/8 阶段不如本文完整" |
| `README.md` | 期收官页：阅读顺序、交付物清单、最终配置、留给三期的问题 | 定稿有效 |
| `docs/00_已知死路与重启指引.md` | 一期九/八条死路的原始清单 + 三条硬约束 + "先做电路估算"的启动清单 | 已被推翻（死路清单被 01 与 TIMELINE 有条件撤销；"九条"实为八条） |
| `docs/01_盘点与重启建议.md` | 重启会话的资产盘点与三份审计：GAN 死路的数据/判据/架构三重误判；众数 vs 采样复算 | 实验记录(结论仍成立)；§4.1 原判"未决、值得重试"已被自己的审计收紧为"必须重做" |
| `docs/512_to_32_PLAN.md` | 512→32 计划：32 是信息下界、蒸馏方案、5 个阶段、红石成本核算 | 已封存（被 `512_to_32_FINAL.md` 取代，`PHASE2_TIMELINE.md:397`） |
| `docs/512_to_32_RESULTS.md` | 阶段 2–5 数据汇总：蒸馏未达预期、连续场 TV 基本无效、收缩中间层收益最大 | 已封存（同上；数据仍有效且被 FINAL 引用） |
| `docs/512_to_32_FINAL.md` | 最终结论：成本成功（1,728）但质量失败（只有 26.1% 落在整数级），根因是多峰条件分布；四种修法全灭 | 定稿有效 |
| `docs/512_to_32_INDEX.md` | 该工作段索引 + "试过并失败的修法（不要再重复）"清单 + 修掉的三个 bug | 纯资料（索引性质，但 F 节清单价值高） |
| `docs/ALIGNMENT_BUG.md` | 自曝 bug：`rint(x/17)` 与 `argmin` 在平局处不等价 → 321,219 vs 321,221 行 → 行序错位 → 学成平均脸 | 定稿有效（含可复用的门禁） |
| `docs/DIM_YIELD_DESIGN.md` | 维数 × 良率盲测设计：为什么要换指标、6 臂、Wilson 区间、7 项工程自检 | 已封存（被 `DIM_LADDER_TRAINED.md` 取代，`PHASE2_TIMELINE.md:400`） |
| `docs/DIM_LADDER_TRAINED.md` | 三个维数模型训练完成：acc 0.4041/0.5335/0.8582，线性上界 0.3269/0.4220/0.7999；含对齐 bug 全记录 | 实验记录(结论仍成立) |
| `docs/PCA4_8_ANALYSIS.md` | PCA-4/8 为何"看着不错"：是更平滑不是更准；噪声槽位几乎闲置（PCA-8 只影响 8% 像素）；自纠"去噪声=多训 4000 步" | 实验记录(结论仍成立) |
| `docs/CONTRAST_ANALYSIS.md` | 发灰问题拆解：信息缺失（corr 0.646，不可修）vs 收缩（斜率 0.684，可修）；两种训练期修法实测 | 实验记录(结论仍成立)；修法 A 的副作用被 `STRETCH_EXPERIMENT.md:28-49` 追加否证 |
| `docs/STRETCH_EXPERIMENT.md` | 拉伸目标 vs 后处理拉伸的机理与实验设计；对比度损失造成 6 个纯黑冻结像素（作弊） | 实验记录(结论仍成立) |
| `docs/ROUND2_POSTPROCESS_RESULT.md` | 后处理 min-max 把 PCA-4 良率从 20.9% → 62.1%（+41.2 点，p=1.1e-9），并量化标准漂移 2.1 倍 | 实验记录(结论仍成立) |
| `docs/ROUND3_RESULT.md` | 烘焙拉伸失败（30.3% vs 54.1%，p=4.8e-3）；根因仍是 MSE→条件均值；意外发现固定增益 ×1.6–1.76 更贴近真实统计 | 实验记录(结论仍成立) |
| `docs/DIM4G_READY.md` | dim4x 的 177 条标签复用价值 + dim4g（全局增益 ×1.7）就绪；逐通道仿射实测更差 | 实验记录(结论仍成立)；"dim4g 待测"已被 round 4 数据否定（见 H-4） |
| `docs/FOUR_KNOBS.md` | 四旋钮命名与相关性表（32 万张真实脸）+ 控制幅度；对比度不是任何 PC 的线性函数 | 定稿有效（但 :53 引用了未成文的第 4 轮结论，见 H-4） |
| `docs/KNOBS_AND_PURIFY.md` | 四旋钮命名采纳 + 274 条标签、只拟合 PC0–3：cv AUC 0.720→0.791，58.8%→90.2%；PC1 离散度砍到 0.539 | 定稿有效（纯化数值口径问题见 H-13） |
| `docs/STYLE_AND_COLOUR.md` | 153 条风格标签：可接受 67.3%、二次元 34.6%；**旧奖励模型实为"二次元探测器"**；明度对颜色 R²≈0.07 | 定稿有效（对旧纯化数字的"都要改写"主张未落实，见 H-13） |
| `docs/STYLE_SWITCHES_MLP_COLOUR.md` | 风格可由线性方向分离（0.896/0.890/0.735），"可接受"不能（0.543）；小网络 80 乘加 vs 二次型 14 项；彩色两阶段方案 | 已被推翻（§三"两阶段成立"被 E6 否证）；风格开关与二次型部分仍成立 |
| `docs/STYLE_CONDITIONED.md` | 风格特化只在类二次元上有效（83.0% 可接受 + 77.4% 命中）；"风格可以选，质量不能选" | 已封存（被 `FILTER_FINAL.md` 取代，`PHASE2_TIMELINE.md:401`） |
| `docs/FILTER_FINAL.md` | 评分器横向对比：二次型对"可接受"0.690、对二次元 0.930；等成本 94.7%；红石最终方案（4-256-256-32+拉伸+4 选 1） | 定稿有效（其红石主链已被 2,304 版取代，打分器部分保留） |
| `docs/COLOUR_EXPERIMENT_DESIGN.md` | 彩色两阶段实验设计：预先固定的四项判据、E1–E7 矩阵、4 bit/通道 RGB 编码 | 已封存（被 RESULT 取代）；其两阶段前提被自证否证（判据④ FAIL） |
| `docs/COLOUR_EXPERIMENT_RESULT.md` | 六变体结果：**E6（仅颜色 8 维）最好**，灰度输入多余；生成时两阶段炸成荧光色；改单阶段 8→128→128→96 | 定稿有效 |
| `docs/NIGHT_COLOUR_LOG.md` | 夜间记录：4 bit 够用、共享调色板不值；E8（颜色 16 维）最佳；彩色结构代理良率 64%（灰度 76.7%）；双打分器修正循环论证 | 定稿有效 |
| `docs/PHASE2_WRAPUP.md` | 进度图 + 16 个原版灰阶方块（v1，含 bedrock/blast_furnace）+ 四个诚实问题 | 已封存（被 `_V2.md` 取代，`PHASE2_TIMELINE.md:399`） |
| `docs/PHASE2_WRAPUP_V2.md` | 进度图 v2 + 灰阶方块 v5：`effective_sat` 公式、`face_var ≤ 3`、活塞可推、平均误差 3.52 | 定稿有效（灰阶实物化仍与"11 档地图灰"并存，见 H-14） |

---

## C. 关键数字

> 只收录有出处的。同量多值者并列，并注明口径差异。

| 量 | 值 | 出处(文件:行号) |
|---|---|---|
| 灰度最终结构 / 乘法 | 4-64-32 / **2,304** | `PHASE2_TIMELINE.md:227`；模型元数据 `webapp/pca22_dim4w64L1_model.json`（`meta.multiplies=2304`） |
| 灰度压缩前结构 / 乘法 | 4-256-256-32 / 74,752 | `README.md:27`、`PHASE2_TIMELINE.md:230` |
| 二次打分器乘法 | 14（4 一次 + 4 平方 + 6 交叉） | `PHASE2_TIMELINE.md:333`、`RESEARCH_ROUTE.md:311` |
| 矩形夹取成本 | 0（随机源限幅） | `README.md:60`、`RESEARCH_ROUTE.md:461` |
| 灰度可接受率（三臂盲测 round 7，n=248） | 60.5% / 59.5% / 61.4%，p≈1.0 | `PHASE2_TIMELINE.md:236` |
| 灰度可接受率（round 9，n=263） | 64.8% / 62.5% / 61.8%，p=0.76/0.75/1.00 | `PHASE2_TIMELINE.md:237` |
| 灰度可接受率（README 口径"约 62%"） | ~62%（不筛选） | `README.md:61` |
| 矩形夹取良率 | **89.1%**（覆盖 23.4%，坏率 10.9%） | `PHASE2_TIMELINE.md:325`、`RESEARCH_ROUTE.md:301`；复核 `results/pca4map/scorer_shapes.json`（coverage 0.234375 / bad 0.109375） |
| 全空间坏率 / 好区 | 33.5% / 66.5% | `PHASE2_TIMELINE.md:312`、`:318`；复核 `results/pca4map/labels_analysis.json`（bad_rate 0.33496） |
| 球状采样（半径 3.0） | 覆盖 9.7%，坏率 32.8%（仅改善 0.7 点） | `RESEARCH_ROUTE.md:292`；复核 `labels_analysis.json`（coverage 0.0974 / bad 0.3283） |
| 二次打分器 AUC / 线性 AUC | 0.853 / 0.694 | `PHASE2_TIMELINE.md:332-333`；复核 `scorer_shapes.json`（0.85323 / 0.69405） |
| 最近中心 12 球 AUC | 0.776（48 乘 + 12 比较） | `RESEARCH_ROUTE.md:312` |
| 打分器误判率（角落） | 约 15%（AUC 0.853 ⇒ 15% 判错） | `RESEARCH_ROUTE.md:325` |
| 同色率（真实 / 教师 / 学生） | 77.6% / 77.6% / 68.5%–72.8% | `512_to_32_RESULTS.md:15-20` |
| 输出落在整数级 ±0.1 的比例 | 26.1%（w128）、25.2%（22→32→32）、28.6%（蒸馏） | `512_to_32_FINAL.md:31-33` |
| 多峰位置占比 / 平均条件熵 | 28.14% / 2.610 bit | `512_to_32_FINAL.md:48`、`512_to_32_INDEX.md:15` |
| 教师 / 最省学生 乘法与准确率 | 202,240 / 0.7151；1,728 / 0.6791 | `512_to_32_FINAL.md:12`、`:15` |
| 22→128→128→32 准确率 | 0.7392（23,296 乘法，11.5%） | `512_to_32_FINAL.md:14` |
| 混合密度（64 输出） | 落级 51.2%，准确率 0.4782 | `512_to_32_FINAL.md:109-110` |
| 蒸馏 vs 硬目标 | 0.7348 vs 0.7719 | `512_to_32_RESULTS.md:45-46` |
| PCA-4 良率：无拉伸 / min-max / 烘焙 | 20.9% / 62.1% / 54.1%（烘焙为 dim4x 之误，见 H-5；真实 dim4st=30.3%） | `ROUND2_POSTPROCESS_RESULT.md:11-13`、`ROUND3_RESULT.md:9-11`；复核 `round2_stats.json`、`round3_stats.json` |
| dim4x 与 dim4/dim22 差异（round 2） | +41.2 点（p=1.1e-9）/ +54.7 点（p=5.0e-19） | `ROUND2_POSTPROCESS_RESULT.md:19-21` |
| dim4x vs dim4st（round 3） | +23.8 点，p=4.8e-3 | `ROUND3_RESULT.md:17` |
| 第 4 轮三臂良率（无对应文档） | dim4x 58.8% / **dim4g 30.9%** / dim22 7.7% | `results/dimmix/round4_stats.json:11`、`:25`、`:39`（原始数据，未经文档化） |
| 真实脸对比度 / 发灰率 / 暗端 / 亮端 | 11.20 / 9.7% / 31.2% / 30.3% | `ROUND3_RESULT.md:64` |
| 固定增益 ×1.6 / ×1.76 | 对比度 11.04 / 11.36（真实 11.20）；发灰 7.7% / 7.4% | `ROUND3_RESULT.md:67-68` |
| dim4g 发灰率（同文件内两值） | 8.73%（:50） vs 7.1%（:62） | `DIM4G_READY.md:50`、`:62` |
| 逐通道仿射 vs 全局增益 | 9.85 vs 11.25（逐通道更差） | `DIM4G_READY.md:62-63` |
| dim4 发灰率 / 对比度 | 14.77% / 8.68（dim22 仅 0.40%） | `CONTRAST_ANALYSIS.md:11-16` |
| 对比度损失的发灰率与代价 | 0.33%（45 倍），acc 0.4128→0.2830，且冻结 6/64 像素为纯黑 | `CONTRAST_ANALYSIS.md:69-75`、`STRETCH_EXPERIMENT.md:36-47` |
| 烘焙目标跨度 vs 实际输出跨度 | 目标 14.59 → 输出 11.27；目标 11.20 → 输出 8.52 | `ROUND3_RESULT.md:45-48` |
| 四旋钮命名相关性 | PC0↔brightness +0.996；PC1↔mouth band −0.867；PC2↔hair band −0.610；PC3↔eye band −0.626 | `FOUR_KNOBS.md:22-33` |
| 四旋钮控制幅度 | PC1 22.88 > PC2 12.50 > PC3 9.38 > PC0 7.94 | `KNOBS_AND_PURIFY.md:21-26` |
| 各轴跨度（4096 点） | PC0 23 / PC1 36 / PC2 35 / PC3 11 点 | `RESEARCH_ROUTE.md:272-275`、`PHASE2_TIMELINE.md:304-310` |
| 空间连通性 | 相邻格点同簇 81.7%（随机 13.8%）；输出内在维数 4（92.7% 方差） | `PHASE2_TIMELINE.md:312`；复核 `results/pca4map/geometry.json`（same_cluster_frac 0.928、intrinsic_90 4） |
| 网格标注量 / 忠实度 | 1,372 条；"中" 33.5% vs 盲测不可接受 35.2% | `PHASE2_TIMELINE.md:297`、`:301`；复核 `pca4_plane_marks.json`（marks 长度 1372） |
| 4 维解释灰度方差 | 71.1%；14 维 90.0%；20 维 95.0% | `RESEARCH_ROUTE.md:369-373` |
| 提维代价 | 4→14 维：2,304→2,944 乘法（+27.8%） | `RESEARCH_ROUTE.md:382` |
| 纯化（dim4x，274 条） | 58.8% → 81.8%(keep50) / 88.4%(35) / 89.7%(25) / 90.2%(15) | `KNOBS_AND_PURIFY.md:60-66` |
| 纯化 cv AUC（只拟合 PC0–3） | 0.791（22 维拟合为 0.720） | `KNOBS_AND_PURIFY.md:36-39` |
| 纯化多样性代价 | PC1 离散度 0.539，总体 0.867 | `KNOBS_AND_PURIFY.md:74-80` |
| w128 奖励模型 | 111 标签 AUC 0.795；269 标签 0.826 | `W128_LABELING_RESULT.md:20`、`ROUND2_RESULT.md:79-81` |
| guided vs explore 净效果 | 54.2% vs 21.1%，2.57 倍，Fisher p=0.00036 | `ROUND2_RESULT.md:13`、`:44` |
| 多样性损失（纯化） | 离散度 0.983 vs 1.000（−1.7%） | `ROUND2_RESULT.md:58-62` |
| 风格分布（153 条） | 二次元 34.6% / 游戏 24.2% / 三次元 8.5% / 中 20.3% / 差 12.4%；可接受 67.3% | `STYLE_AND_COLOUR.md:26-31` |
| 旧奖励模型"二次元偏好" | 类二次元 1.801 vs 其他 +0.510，t=11.20；且"中"(1.410) > 类游戏(1.288) | `STYLE_AND_COLOUR.md:43-55` |
| 可接受性的模型上限 | 线性 0.543（≈随机）/ 二次 0.683 / kNN 0.664 / 4-16-1 0.690 | `STYLE_AND_COLOUR.md:66-70` |
| 二次型（153 标签） | 对可接受 0.690、对二次元 0.930；4-16-16-1 为 0.709 / 0.931 | `FILTER_FINAL.md:11-14` |
| 等成本筛选良率 | 指定二次元 + 二次型 → 94.7%（95%CI 82.7–98.5） | `FILTER_FINAL.md:53` |
| 风格特化（保留 35%） | 二次元 77.4% 命中 / 83.0% 可接受；游戏 47.2%/64.2%；三次元 22.6%/58.5% | `STYLE_CONDITIONED.md:15-18` |
| 彩色对齐样本 | 321,219 张（923,975 张中过滤对称后） | `COLOUR_EXPERIMENT_DESIGN.md:12`、`STYLE_AND_COLOUR.md:103` |
| **4 bit/通道量化下限** | **0.2049 级**（每通道，全像素 MAE） | `COLOUR_EXPERIMENT_RESULT.md:13`、`NIGHT_COLOUR_LOG.md:14` |
| 3/5/6 bit 量化误差 | 0.4488 / 0.1004 / 0.0495 | `NIGHT_COLOUR_LOG.md:13-16` |
| 共享调色板重建 MAE | 16 色 0.8333(+307%) / 24 色 0.7003 / 32 色 0.6069 / 48 色 0.5231 / 64 色 0.4700(+129%) | `NIGHT_COLOUR_LOG.md:22-28` |
| 64 色调色板合成误差 | √(1.35²+0.47²)=1.43，只差 6% | `NIGHT_COLOUR_LOG.md:34-35` |
| 数据里出现的不同 RGB 三元组 | 195,360 种 | `NIGHT_COLOUR_LOG.md:30` |
| 颜色镜像对称率 | 98.7%（灰度 80.9%） | `COLOUR_EXPERIMENT_RESULT.md:14`、`COLOUR_EXPERIMENT_DESIGN.md:51` |
| 颜色 PCA 解释方差 | 8 维 75.0–75.1% / 16 维 86.1–86.2% / 22 维 90.8% / 32 维 95.1% | `SUMMARY.md:196-201`、`COLOUR_EXPERIMENT_DESIGN.md:15-16` |
| 颜色 PCA 8 维协方差 | 特征值 0.98–1.01，条件数 1.0，平均非对角相关 0.003 | `COLOUR_EXPERIMENT_RESULT.md:82-86` |
| 明度→颜色可推断性 | 饱和度 R²=0.071–0.073；色相 R²=0.062–0.063 | `STYLE_AND_COLOUR.md:108-114`、`COLOUR_EXPERIMENT_DESIGN.md:13-14` |
| 基本是灰的像素 | 36.6%（chroma<0.05） | `STYLE_AND_COLOUR.md:104` |
| 六变体 RGB MAE / 色相误差 | E1 2.154/0.108；E2 1.685/0.061；E3 1.369/0.044；E4 0.970/0.040；E5 1.339/0.044；E6 1.349/0.043 | `COLOUR_EXPERIMENT_RESULT.md:29-34` |
| E6 vs E3（有无灰度） | MAE 1.349 vs 1.369（无灰度反而更好） | `COLOUR_EXPERIMENT_RESULT.md:51` |
| E8（颜色 16 维） | MAE 0.933，色相 0.039（全变体最佳） | `NIGHT_COLOUR_LOG.md:44` |
| 彩色采样器统计 | 真实 饱和 0.274/明度 7.71；N(0,Σ) 0.367/7.70；N(0,I) 0.483/6.44 | `COLOUR_EXPERIMENT_RESULT.md:92-97` |
| 彩色结构代理良率 | 灰度 76.7% / E6 63.2% / E8 64.0% | `NIGHT_COLOUR_LOG.md:54-58` |
| 彩色 PC1 均值 | 灰度 −0.436 / E6 +0.049 / E8 +0.065 | `NIGHT_COLOUR_LOG.md:62-67` |
| 彩色筛选（双打分器） | 67.1% → 79.1%(50%) → 87.9%(25%) → 92.9%(10%) | `NIGHT_COLOUR_LOG.md:90-99` |
| 彩色最终结构 / 乘法 | 8-128-128-32×8 / **50,176**（3 bit/像素） | `SUMMARY.md:13-14`；复核 `webapp/colour_k8snap_model.json`（`meta.mults=50176`） |
| 调色板 K 扫描（round 8） | K=8 50.0% / K=16 51.8% / K=32 60.7%，两两 p=0.879/0.175/0.282（均不显著）；乘法 50,176/82,944/148,480 | `SUMMARY.md:99-105`、`RESEARCH_ROUTE.md:141-147` |
| 三种配色方案（round 11） | free 50.5% / **snap 57.3%** / mc 33.6%；snap-mc p=0.0008 | `SUMMARY.md:113-121` |
| 彩色最终成绩（round 12） | **61.1%**（331 条） | `SUMMARY.md:15`、`:144` |
| 彩色提升五路全灭 | 训练 4× 仅边缘 +6.5%；加宽 2× 无变化；C1 −20.2 点(p=0.004)；C2 −17.9 点(p=0.008)；C2 硬采样 D 0.86→0.99 | `SUMMARY.md:150-158`、`RESEARCH_ROUTE.md:189-195` |
| 彩色 GAN 判别器准确率 | 软 0.863 / 硬 0.988 / 谱归一化 0.867（永远到不了 0.5） | `SUMMARY.md:166`、`RESEARCH_ROUTE.md:212` |
| 8 维潜变量预测调色板索引 | PCA 70.1% / one-hot 50.6% / 多数类 29.8% | `SUMMARY.md:209-211` |
| 提维的乘法代价（彩色） | 8→12 维 50,688(+1.0%)；8→16 维 51,200(+2.0%) | `SUMMARY.md:217-221` |
| 11 档灰归并 vs 16 级 | 66.0% vs 64.8%，p=0.90；可分辨仅 9 档；近重复率 3–4 倍 | `SUMMARY.md:81-83`、`PHASE2_TIMELINE.md:278-283` |
| round 10 三臂 | 64.8% / 66.0% / 70.7%（n=145/144/116） | `RESEARCH_ROUTE.md:88`；复核 `results/dimmix/round10_stats.json` |
| MC 颜色亮度天花板 | 地图 NORMAL 12.94 级 / 四档全开 15.00 / 方块均色 14.94 | `RESEARCH_ROUTE.md:151-157` |
| 筛种子 vs 筛脸相关性 | r=0.52–0.67 → r²≈0.3；筛选实测增益 +2.6 / +4.7 点 | `SUMMARY.md:56-59`、`RESEARCH_ROUTE.md:118-127` |
| 标签噪声 | 11–12%（像素距离<0.5 的近重复被判成不同级） | `SUMMARY.md:61`、`RESEARCH_ROUTE.md:131` |
| 16 灰阶方块 v5 | 平均误差 3.52、最大 11.5、全部活塞可推、分面一致 | `PHASE2_WRAPUP_V2.md:82` |
| 16 灰阶方块 v1 | 平均误差 3.30、平均饱和度 0.037 | `PHASE2_WRAPUP.md:75` |
| 灰度 16 色量化 RMS | **20.62/255**（旧文档写的 11.9 是 bug） | `01_盘点与重启建议.md:74-77` |
| 灰度数据指标复算 | 平均颜色数 5.3132 / 多样性 57.79 / 对称率 0.8598 | `01_盘点与重启建议.md:66` |
| 一期 GAN 审计 | D 单训 95.5–98.4%；对归档 G 99.0/97.9%；单层线性 98.4/97.6%；联合训练 50.8% | `01_盘点与重启建议.md:178-181` |
| 标签平滑的常数最优 | p*=0.45 | `01_盘点与重启建议.md:166`、`PHASE2_TIMELINE.md:64` |
| 众数 vs 采样（复算） | T=30 采样颜色数 5.44 / T=1 4.95 / 逐位置众数 3.47 / 逐对众数+镜像 3.38（并出现 2/64 重复） | `01_盘点与重启建议.md:240-244`；复核 `results/restart_probe.json` |
| 风格包（6 个） | 中心坐标 + 1σ 坏率全 0.0%，2σ 0–3.3% | `PHASE2_TIMELINE.md:344-349`、`RESEARCH_ROUTE.md:341-346` |
| 黑暗法师边界修正 | 2σ 坏率 11.5% → 0.0%（欧氏球内拟合） | `RESEARCH_ROUTE.md:350-354` |
| 标签总量 | 3,752 条（12 轮）+ 1,372 网格 | `RESEARCH_ROUTE.md:34-36`；复核 `labels.jsonl` 行数 3,752 |

---

## D. 仍然可用的规则、方法与资产

### D1 模型架构与规格

| 做法/公式 | 适用条件 | 出处 |
|---|---|---|
| 灰度生成器 **4 → 64 → 32（单层隐藏层，连续输出）**，乘法 = 4×64 + 64×32 = **2,304** | 4 维 PCA 条件、8×8、镜像到 32 个自由位置 | `PHASE2_TIMELINE.md:227`、`README.md:56`；复核模型 `meta`（in_dim 4, hid 64, layers 1, mults 2304） |
| **同等乘法预算下单层优于两层**：4-64-32（2,304, acc 0.3651）> 4-32-32-32（2,176, 0.3562）；原因是第二层只有 32 宽 | 输出层宽度被压到 32 时的宽度/层数选择 | `PHASE2_TIMELINE.md:226-232`、`SUMMARY.md:27-41` |
| 完整灰度链路：`4 PCA → 4-64-32 → 32 连续值 → min-max 拉伸 → 量化到 16 级或 11 档` | 部署顺序：先抽种子→打分→只生成一次 | `README.md:56-63` |
| 彩色生成器 **8 → 128 → 128 → 32×K**（K=8 时 50,176 乘法），`argmax → 调色板查表 → 行内镜像` | 调色板分类解决"糊"；索引离散 ⇒ 边界天然硬 | `SUMMARY.md:93-95`、`:13-14`；`webapp/colour_k8snap_model.json`（type `palette_index`, 3 bit/像素） |
| 彩色乘法随 K 的增长点 | 大头在最后一层（32×K）；K=8/16/32 → 50,176 / 82,944 / 148,480 | `SUMMARY.md:99-105`、`:223` |
| 输出 32 是信息下界（32 位置 × 4 bit = 128 bit），PCA 降维替代不了（PCA-22 重建 RMS 17/255 vs 量化 4.58/255） | 想再压缩输出维度时 | `512_to_32_PLAN.md:11-18` |

### D2 采样与打分器

| 做法/公式 | 适用条件 | 出处 |
|---|---|---|
| **矩形夹取**：`PC2 ≥ −0.5 且 PC1 ≤ −1.5` → 覆盖 23.4%、坏率 10.9%、良率 **89.1%** | 前提是"坏区在各轴上有序"（PC2 单调、PC1 倒 U）；成本 0（随机源限幅） | `PHASE2_TIMELINE.md:325`、`RESEARCH_ROUTE.md:296-304` |
| 备选夹取：`PC1 ≤ −2.5` → 覆盖 25.0%、坏率 17.2%、良率 82.8% | 需要更大多样性时 | `RESEARCH_ROUTE.md:302` |
| **不要用球状限制**：最优球（半径 3.0）覆盖 9.7%、坏率仍 32.8% | 坏区是切穿空间的片状，球是各向同性必然浪费 | `RESEARCH_ROUTE.md:287-294` |
| **二次型打分器（14 项）**：`y = c + Σaᵢxᵢ + Σbᵢxᵢ² + Σ_{i<j}dᵢⱼxᵢxⱼ`，4 输入 → 4+4+6 = 14 项，AUC 0.853 | 需要表达 PC1 的倒 U；只需乘法器、不需要比较器 | `STYLE_SWITCHES_MLP_COLOUR.md:94-98`、`PHASE2_TIMELINE.md:333` |
| 二次型实际权重（可复用）：`PC0·PC1 +0.129`、`PC2 −0.124`、`PC0·PC2 +0.106`、`PC1² −0.081`、`b=0.335` | 直接照抄系数 | `PHASE2_TIMELINE.md:336`；`results/pca4map/scorer_shapes.json`（full coef） |
| 线性打分器只够做**风格开关**，不够做"可接受" | 风格：0.896 / 0.890 / 0.735；可接受：0.543 | `STYLE_SWITCHES_MLP_COLOUR.md:9-14` |
| 三个风格方向近似一组基：二次元 = PC1 低 + PC2 低；游戏 = PC1 高；三次元 = PC2 高 | 想做"去二次元"就把 PC1/PC2 往上推 | `STYLE_SWITCHES_MLP_COLOUR.md:19-24` |
| **"可接受"没有任何单标量判据**：\|PC3\| 0.609、最大\|PC\| 0.528、PC0-3 半径 0.522、\|PC1\| 0.372 | 别再试径向/阈值型判据 | `STYLE_SWITCHES_MLP_COLOUR.md:30-39` |
| **风格 = 联合高斯**：中心可移、协方差各向异性；边界处改用"欧氏球内拟合"（以种子为中心找最大干净球，用球内格点拟合） | 中心贴立方体边界时（如 PC1=3.5），拒绝采样会系统性内缩（实际均值偏 0.78） | `PHASE2_TIMELINE.md:340`、`RESEARCH_ROUTE.md:348-354` |
| **打分器只能当候选过滤器，不能当唯一裁判** | 角落约 15% 误判（潦草大叔 (3.5,3.5,3.5,−0.5)：二次坏度 0.671 判坏，离散标签判好）；安全区必须用离散全量标签 | `RESEARCH_ROUTE.md:318-327` |
| **只对生成器真正用到的维度拟合奖励**：dim4x 只读 4 维，22 维拟合等于喂 18 个噪声特征（AUC 0.720 → 0.791） | 任何"生成器只吃 k 维"的架构 | `KNOBS_AND_PURIFY.md:34-44` |
| 筛选的**结构性上限**：筛的是种子、判的是脸，r=0.52–0.67 ⇒ 有效性 ≈ r² ≈ 0.3，实测只 +2.6/+4.7 点 | 想用打分器提良率前先量相关性；红石部署顺序决定该筛种子还是筛脸 | `SUMMARY.md:56-59`、`RESEARCH_ROUTE.md:116-127` |

### D3 调色板与灰阶方块

| 做法/公式 | 适用条件 | 出处 |
|---|---|---|
| **16 个原版灰阶方块（v5）**：`effective_sat = sat × (0.25 + 0.75 × (1 − abs(grey−127.5)/127.5))`，即极端明度处只按 25% 惩罚饱和度 | 允许等级 0 选偏蓝的 `black_concrete`；平均误差 3.52 | `PHASE2_WRAPUP_V2.md:31-37`、`:82` |
| **分面一致性硬过滤 `face_var ≤ 3`**："颜色方差"的正确含义是**分面之间的平均色差**，不是贴图内部噪声（高炉 12.66、骨块 11.29 vs 闪长岩 0.00） | 单像素取方块时必须分面一致 | `PHASE2_WRAPUP_V2.md:38-50` |
| **活塞可推性排除**：带方块实体的方块一律不可推（高炉/熔炉/箱子/漏斗/潜影盒），基岩直接不可推；**台阶放行**（两格能拼整方块），门/活板门/栅栏/玻璃板排除 | 需在生存装置里搬运时 | `PHASE2_WRAPUP_V2.md:52-59` |
| 灰阶选块算法：jar 提贴图 → 按方块合并分面 → alpha 加权平均 → 只留完整立方体 → 饱和度<0.25 候选 → **亮度有序的动态规划求保序最优分配** | 需要"等级→方块"映射时 | `PHASE2_WRAPUP.md:51-54` |
| **11 档地图 HIGH 灰归并表**（16→11 映射）：0,1,2→BLACK(1.47)；3,4→GRAY(4.47)；5→TERRACOTTA_CYAN(5.32)；6→DEEPSLATE(5.88)；7→STONE(6.59)；8→LIGHT_GRAY(9.00)；9→METAL(9.82)；10,11→CLAY(9.92)；12,13→WOOL(11.71)；14→QUARTZ(14.83)；15→SNOW(15.00) | 每像素只需 4 bit 索引；归并零代价（66.0% vs 64.8%，p=0.90），但只有 9 档肉眼可分 | `SUMMARY.md:66-83`、`webapp/merge_mc11.json` |
| **MC 颜色亮度天花板**：地图 NORMAL 61 色最高 12.94 级（没有好白）；四档全开 244 色可达 15.00；方块均色 61 色 14.94 | 选颜色来源前必须先看这条 | `RESEARCH_ROUTE.md:151-159` |
| **HIGH 列 = 基色本身 = 不做任何修饰**：既含纯白又是最省的硬件实现 | 彩色最终 8 色全部取自地图 HIGH 列 | `RESEARCH_ROUTE.md:159` |

### D4 色彩空间与量化

| 做法/公式 | 适用条件 | 出处 |
|---|---|---|
| **RGB 每通道 4 bit**：`RGB₄ = round(RGB₈ / 17) ∈ [0,15]³`；选 RGB 不选 HSV（红石直接驱动三色灯/染色方块、损失可用欧氏、与灰度线性可解） | 连续上色路线；HSV 仅作诊断 | `COLOUR_EXPERIMENT_DESIGN.md:32-44` |
| 量化下限必须先量：4 bit/通道 **0.2049 级**，RMS 级别以下的模型误差不可解释 | 评估任何彩色模型前 | `COLOUR_EXPERIMENT_RESULT.md:13`、`:177-179` |
| **不要用共享调色板索引换 12 bit 通道**：16 色 0.8333 比 4bit 差 307% | 除非布线是主矛盾 | `NIGHT_COLOUR_LOG.md:20-32` |
| 折中：**64 色调色板（6 bit）** 合成误差只差 6%（1.43 vs 1.35），可把 12 根颜色线减到 6 根 | 布线紧张时 | `NIGHT_COLOUR_LOG.md:34-35` |
| **训练后吸附 ≫ 事前约束**：自由 k-means K=8 → 训练 → 吸附到最近 MC HIGH 色（零成本），57.3% vs 事前限定 MC 色重训 33.6%（p=0.0008） | 想让输出能被真实方块摆出来 | `SUMMARY.md:107-127`、`RESEARCH_ROUTE.md:161-177` |
| 吸附成败的原因：k-medoids 只优化像素重建误差 ⇒ 选出一堆褐色、无纯白；自由色吸附后反而得到 QUARTZ/CRIMSON_NYLIUM/LAPIS | 设计调色板时不要直接对重建误差优化 | `SUMMARY.md:127` |
| 彩色最终 8 色调色板（含亮度）：QUARTZ 14.83 / CRIMSON_NYLIUM 5.31 / RAW_IRON 10.83 / YELLOW 12.28 / LAPIS 7.43 / BLACK 1.47 / TERRACOTTA_LIGHT_GRAY 6.73 / TERRACOTTA_BROWN 3.30 | 可直接复用 | `SUMMARY.md:133-142` |
| **颜色可以安全镜像**：98.7%（比灰度 80.9% 还高），沿用"32 位置 + 行内镜像"布局 | 8×8 对称头像 | `COLOUR_EXPERIMENT_RESULT.md:14`、`:19` |
| **颜色 PCA 前 8 维协方差 ≈ 单位阵**（特征值 0.98–1.01、条件数 1.0、非对角 0.003）⇒ **8 个独立 N(0,1) 就是正确的联合分布，不需要 8×8 变换** | 单阶段彩色采样的免费替代 | `COLOUR_EXPERIMENT_RESULT.md:80-88` |
| 三种采样器都可用：N(0,Σ) 饱和 0.367 / N(0,I) 0.483（偏饱和）/ 取自真实样本 0.358，真实 0.274 | MSE 训练不会主动压到极端色 | `COLOUR_EXPERIMENT_RESULT.md:90-99` |
| 色相误差必须用**循环距离** `min(abs(h₁−h₂), 1−abs(h₁−h₂))` | 任何色相指标 | `COLOUR_EXPERIMENT_DESIGN.md:121` |
| 布局约定（可与灰度共用）：`row = p mod 8, col = ⌊p/8⌋`，镜像到 `7−col` | 32 自由位置 | `COLOUR_EXPERIMENT_DESIGN.md:48-50` |

### D5 训练技巧

| 做法/公式 | 适用条件 | 出处 |
|---|---|---|
| **量化必须"平局向下"**：用 `np.clip(np.ceil(gray/17.0 − 0.5), 0, 15)`，**不要用** `np.rint(gray/17)`（银行家舍入；实测 594 个元素/127 行不同 → 行序错位） | 任何用灰度复现 `L0_full_idx` 的地方 | `ALIGNMENT_BUG.md:74-84` |
| **训练前门禁：线性读出 < 0.30 直接中止** | 防"训练失败"被误判为"低维塌陷"；实测线性上界 k=4/8/22 = 0.3269/0.4220/0.7999 | `ALIGNMENT_BUG.md:88-104`、`DIM_LADDER_TRAINED.md:82-88` |
| **后处理 min-max 拉伸（不是训练期烘焙）**：`lo=min(v), hi=max(v), v'=clamp((v−lo)·15/(hi−lo),0,15)` → 对比度恒 15、发灰率 0、0 冻结像素 | 只作用于解码后的 32 个连续值，权重不动 | `STRETCH_EXPERIMENT.md:53-81` |
| 烘焙失败是**原理性**的：网络逐位输出，无法拿到"整张脸的 min/max"，而 MSE 又把它拉回条件均值（目标 14.59 → 输出 11.27） | 想省掉输出端 min/max 电路时，先看这条 | `ROUND3_RESULT.md:39-53`、`PHASE2_TIMELINE.md:200` |
| 固定增益 `v' = 7.5 + (v−7.5)×1.7`（一次乘法 + 一次加法）**统计上**最贴近真实（11.04–11.36 vs 11.20，发灰 7.4–7.7% vs 9.7%） | 但**良率实测只有 30.9%**（见 F-14）；只能当"统计量匹配"技巧，不能替代 min-max | `ROUND3_RESULT.md:58-71`、`DIM4G_READY.md:49-50`、`results/dimmix/round4_stats.json:25` |
| **全局增益优于逐通道仿射**：逐通道匹配的是"跨样本时间方差"，不放大"同一张脸内部位置差"（9.85 vs 11.25） | 红石"所有 Bᵢ 相等"不是妥协而是更优解 | `DIM4G_READY.md:53-70` |
| **对比度损失项是陷阱**：`L += w·(std(ŷ)−std(y))²` 会把 6/64 个像素永远钉在 0（纯黑）来"抬高 std"，发灰率降到 0.33% 是假的 | 且斜率反而从 0.684 掉到 0.483（实现"永远鲜艳"而非"对比度正确"） | `CONTRAST_ANALYSIS.md:54-91`、`STRETCH_EXPERIMENT.md:28-49` |
| **TV 惩罚只对概率场有效，对连续场无效**：连续场 λ 0→0.2 只把同色率 73.2%→74.4% 而 acc 掉 3.7 点 | 想做"平滑"时选分类输出 + 概率 TV | `512_to_32_RESULTS.md:52-67` |
| **直通量化（STE）是数学空操作**：`MSE(round_st(y), t)` 的梯度与未量化完全相同，只改前向 | 别再试"量化感知"的 STE 变体当银弹 | `512_to_32_FINAL.md:64-67` |
| **蒸馏不如硬目标**（0.7348 vs 0.7719），尽管教师用期望值解码 0.7333 > argmax 0.7151 | 学生要在"教师自己的输出"上训，且软信息优势不足以抵消误差累积 | `512_to_32_RESULTS.md:38-48` |
| **标签平滑 0.9 会让"判别器 sigmoid 均值"失去意义**：常数预测器最优解 p*=0.45 | 评估 GAN 时永不只看 sigmoid 均值，要看留出集 acc/AUC + 每 checkpoint 的 D loss | `01_盘点与重启建议.md:161-170`、`:227` |
| 判别器诊断：必须每 N 轮冻结 G 快照并单独训 D（同 lr），实测能到 95.5–98.4% | 怀疑"GAN 学不动"时先做这个反证 | `01_盘点与重启建议.md:172-183`、`:228` |
| **交叉验证估良率时要用两个不相交打分器**（一个筛、一个判），且阈值取"匹配实测可接受率"的分位数 | 否则得到 100%/98.5% 的循环论证结果 | `NIGHT_COLOUR_LOG.md:77-99`、`RESEARCH_ROUTE.md:98-115` |
| 风格判断用线性（近免费），可接受性用二次型/小网络；两级串联无额外收益 | 串级（先风格 35% 再可接受 50%）= 80.8% ≤ 单级 83.0% | `STYLE_CONDITIONED.md:26-34`、`FILTER_FINAL.md:63-80` |
| 训练配方（可复用）：`MSE + 5.0×TV + 对抗(1.0)`，Adam 3e-4，12,000 步，batch 256，hid 256 | 灰度 32 输出系列的共同配方 | `DIM_YIELD_DESIGN.md:26-34`、`webapp/pca22_dim4w64L1_model.json`（lam 5.0, adv 1.0, steps 12000） |

### D6 数据管线

| 做法/资产 | 适用条件 | 出处 |
|---|---|---|
| `data/faces8.npy`：(50000, 64) uint8，取值 0–15（16 色索引）；平均每张 **5.33** 种颜色 | 二期唯一数据基础（审计确认可信） | `01_盘点与重启建议.md:31`、`:41`、`:68` |
| `data/faces8_rgb.npy`：(50000, 8, 8, 3) uint8，与索引版同源 | 需要 RGB 时 | `01_盘点与重启建议.md:32` |
| 裁切区 (8,8)–(16,16)，块对齐（平移 1 像素会让"完全不透明"从 94.6% 掉到 5.2%） | 重新切脸时 | `01_盘点与重启建议.md:61`、`:204` |
| 分割文件 `data/L0_split.npz`（`va` 为验证集）、`data/L0_full_idx.npy`、`data/L0_manifest.json`；规范副本在 `data_v2_canonical/` | 训练/验证必须用同一份行序 | `COLOUR_EXPERIMENT_DESIGN.md:107`、`ALIGNMENT_BUG.md:84` |
| 彩色对齐集：923,975 张 → 过滤对称后 **321,219 张**，与明度数据**逐行完全对齐**（R² 检验到行级） | 彩色任何训练 | `STYLE_AND_COLOUR.md:103`、`COLOUR_EXPERIMENT_RESULT.md:12` |
| 数据三处污染（需知道，不一定要修）：52,690 条(4.76%)全透明头被当全黑；重复率 15.78%（单一全黑裁切 51,156 次）；取数是"按 zip 顺序前 5 万个"，是模板聚集数据的头 8.2% | 若要重做更干净的数据集 | `01_盘点与重启建议.md:193-198` |
| 调色板文件 `data/palette_avatar16_v2.npy` / `_v2a.npy`、`data/mc_map_colors.json`、`results/colour/mcfit/blocks.json`（61 方块均色） | 灰度/彩色量化 | `01_盘点与重启建议.md:37-38`、`PHASE2_TIMELINE.md:251` |
| 打标工具链：`webapp/label.html?mix=1`（盲测混合，臂随机、参数纯随机、不显示来源）、`webapp/label_colour.html?round=6`、`webapp/plane.html`（4-PCA 平面标注，单击/拖过/自定义类别） | 所有人工良率测量 | `DIM_YIELD_DESIGN.md:40-47`、`NIGHT_COLOUR_LOG.md:110-122`、`RESEARCH_ROUTE.md:253-257` |
| 打标页的坑：`elementFromPoint` 返回 `<canvas>` 无 `.cell` 类 → 点击被静默丢弃；修法 `pointer-events:none` + 从 `e.target` 向上找 | 复刻标注工具时 | `RESEARCH_ROUTE.md:257` |

---

## E. 已被推翻或已失效的结论

| 原结论 | 为什么失效 | 出处 |
|---|---|---|
| "8×8 规模下无条件 GAN 学不动，判别器不学"是硬结论 | 该次训练集仅约 1,205 张（数据集 10-05 11:48 才生成，GAN 产物 10-04 21:33–22:32，早 13 小时）；同 D 同 lr 单训 95.5–98.4% | `docs/00_已知死路与重启指引.md:38-47` → 推翻于 `01_盘点与重启建议.md:148-183`、`PHASE2_TIMELINE.md:381` |
| sigmoid 均值 0.45 证明"判别器什么都没学到" | 0.45 是标签平滑 0.9 下常数预测器的最优解 `(p−0.9)+p=0`，无法区分失败模式 | `00_已知死路与重启指引.md:42` → `01_盘点与重启建议.md:161-170`、`PHASE2_TIMELINE.md:62-66` |
| 那次 GAN 是"标准 DCGAN 结构" | 实为 `z(32)→Linear(32,128)→ReLU→Linear(128,64)→tanh`，D 是 `64→64→1`：**无卷积、无 BN 的 2 层 MLP** | `01_盘点与重启建议.md:185-189`（并指出 `00:42` 与 `gan_train.py` 自相矛盾，`:288-289`） |
| "任何逐像素独立的生成模型都会退化"（条件独立是数学必然） | 需限定为"**逐位均值**必然失败"；"逐位边缘分布"在足够尖时可以成功（混合近邻成功即反例） | `00_已知死路与重启指引.md:57` → `01_盘点与重启建议.md:306-313`、`PHASE2_TIMELINE.md:384` |
| 条件 GAN 不可行（把调色板索引当标量） | 是工程缺陷，one-hot/embedding 可修 | `00:59-67` → `PHASE2_TIMELINE.md:383` |
| 混合近邻"不可移植红石"，所以整条路否掉 | 真正缺的是**采样能力**：同模型只把采样换成取众数，颜色数 4.95→3.38 并出现完全重复 | `00:111-119` → `01_盘点与重启建议.md:249-262`、`:323` |
| 死路 6 的机理"反复应用均值算子 ⇒ 不断模糊"适用于 MSE 本身 | 只适用于**迭代**精修；前馈一次性预测不迭代就不成立 | `01_盘点与重启建议.md:270-273` |
| "去掉 64 维噪声 ≈ 多训练 4000 步" | 混淆步数变量（pca22_nonoise 6000 步 vs sweep_pca22 1500 步）；固定 1500 步下噪声 0→1.0 反而提高 9.7 点 | `PCA4_8_ANALYSIS.md:75-92` → `PHASE2_TIMELINE.md:387` |
| "低维模型塌陷"（k=4/k=8 acc 0.05，低于随机） | 是自引 bug：`rint` 与 `argmin` 平局不等价 → 多 2 行 → 行序整体错开，条件配 A 脸、目标配 B 脸 | `DIM_LADDER_TRAINED.md:60-80` → `ALIGNMENT_BUG.md:1-34` |
| 生成质量差的主因是"碎片化"（色块数 21.5 vs 13.8） | 放大 3 倍看图后改为"输出落在级间"：整数级命中率仅 26.1%，平均小数 0.230 | `512_to_32_RESULTS.md:28-29` → `512_to_32_FINAL.md:21-35`、`PHASE2_TIMELINE.md:385` |
| 蒸馏（教师期望值）应优于硬目标 | 实测 0.7348 < 0.7719；作者自述"因听起来更优雅而偏向它" | `512_to_32_PLAN.md:38-47` → `512_to_32_RESULTS.md:41-48` |
| TV / 整数吸附 / 直通量化 / 四舍五入后 TV 能治"糊" | 四种全部无效：TV 惩罚差值不惩罚跨级；吸附冗余；STE 梯度不变是空操作 | `512_to_32_FINAL.md:58-68`、`512_to_32_INDEX.md:20-27` |
| 混合密度（每位置 2 分量、64 输出）能治双峰 | 落级率 27.7%→51.2% 但 acc 0.7364→0.4782，选错峰概率约 30%，"选哪个峰"的信息不在输出里 | `512_to_32_FINAL.md:102-125` |
| 把 min-max 拉伸烘焙进训练目标可省掉输出端 min/max 电路 | dim4st 30.3% vs dim4x 54.1%（+23.8 点，p=4.8e-3）；原理上网络拿不到整脸 min/max | `ROUND2_POSTPROCESS_RESULT.md:77-84` 的待验假设 → `ROUND3_RESULT.md:9-21`、`:39-53` |
| 用固定增益 ×1.7 替代 min-max 即可（红石只要一个乘法器） | 第 4 轮实测 **30.9% vs 58.8%**（dim4g vs dim4x，n=97/97） | `DIM4G_READY.md:47-51`、`ROUND3_RESULT.md:86-97`（假设）→ `results/dimmix/round4_stats.json:25` 与 `:11`（否证） |
| "对比度损失把发灰率降 45 倍"是真收益 | 它把 6/64 像素钉成纯黑来抬高 std，指标是作弊；acc 0.4128→0.2830 | `CONTRAST_ANALYSIS.md:69-75` → `STRETCH_EXPERIMENT.md:28-49` |
| 逐通道仿射（每位置各自 gain/offset）应优于全局增益 | 实测 9.85 vs 11.25；逐通道匹配的是跨样本时间方差，与"脸内对比度"无关 | `DIM4G_READY.md:55-70` |
| **彩色两阶段**（先灰度、再拿灰度上色）成立且优于"只用颜色 PCA" | E6（仅颜色 8 维，MAE 1.349）优于 E3（灰阶+颜色 1.369）；且生成时 E2–E5 出荧光粉/绿/蓝/品红，因为"灰度是这张脸、颜色是另一张脸"；判据④灰度一致性 2.21 > 1.5 FAIL | `STYLE_SWITCHES_MLP_COLOUR.md:102-127` 与 `COLOUR_EXPERIMENT_DESIGN.md:68-72` → `COLOUR_EXPERIMENT_RESULT.md:43`、`:49-72`、`NIGHT_COLOUR_LOG.md:132` |
| 明度图可推断饱和度/色相（"零成本上色"） | 饱和度 R²=0.073、色相 R²=0.062；颜色必须有独立条件输入 | `STYLE_AND_COLOUR.md:106-118` |
| 线性打分器足以刻画"可接受"（旧 +0.15 差距被忽略） | 可接受 0.543（≈随机）；只对**风格**成立（0.896/0.890） | `W128_LABELING_RESULT.md:98-106` 与 `ROUND2_RESULT.md:101-112` 的隐含前提 → `TIMELINE:386`、`FILTER_FINAL.md:16` |
| 旧奖励模型测的是"优/良" | 它实为"类二次元探测器"：二次元 1.801 vs 其他 +0.510（t=11.20），且"中"(1.410) > 类游戏(1.288) | `STYLE_AND_COLOUR.md:39-57` |
| 风格可用"球状区域"复现 | 球采样只提升 0.7 点（覆盖 9.7%、坏率 32.8%）；好区是单一连通块、坏区是片状 | `RESEARCH_ROUTE.md:287-294` → `TIMELINE:388` |
| 二次打分器可作最终裁判 | 角落约 15% 误判，安全区必须以 1,372 条离散标签为准 | `RESEARCH_ROUTE.md:318-327` → `TIMELINE:389` |
| 筛选能带来 +19 点良率（按交叉验证外推） | 实测 +2.6 / +4.7 点，因为筛种子≠筛脸（r²≈0.3） | `SUMMARY.md:56-59`、`RESEARCH_ROUTE.md:116-127` |
| C1 结构损失能把预测良率做到 95.1% | 自证循环（探针用的正是 C1 优化的量）；实测 41.0%，所有脸戴同一条深色眼带 | `SUMMARY.md:178-184`、`RESEARCH_ROUTE.md:197-206` |
| 对抗（C2）能提升彩色良率 | 软平均 −17.9 点（43.2%）；直通硬采样 D 升到 0.99、眼睛对比崩到 1.27 | `SUMMARY.md:157-158`、`RESEARCH_ROUTE.md:194-195` |
| 多训练（12k→48k）/加宽（128→256）能提升彩色 | 前者仅边缘 +6.5%，后者无变化 | `SUMMARY.md:154-155` |
| 事前把调色板限定在 MC 颜色更好 | 33.6% vs 自由训练后吸附 57.3%（p=0.0008） | `SUMMARY.md:113-125` |
| K=32 调色板值得（60.7% 最高） | 与 K=8 的 50.0%、K=16 的 51.8% 两两 Fisher 全不显著，却要 3 倍乘法 | `SUMMARY.md:105` |
| 两阶段方案"复用已训好的阶段 1、风险低"因而优先 | 该建议在 5 小时后被同级实验否掉；单阶段反而更便宜（29,696 乘法） | `STYLE_SWITCHES_MLP_COLOUR.md:146` → `COLOUR_EXPERIMENT_RESULT.md:105-114` |

---

## F. 死路清单

> 分两类：**F1** = 一期九方案（原始死路，含已被重新定性的）；**F2** = 二期自己撞死的路。

### F1 一期九/八方案

| 路线 | 失败原因 | 出处 |
|---|---|---|
| 无条件 GAN（MLP 判别器） | 原判"判别器不学"；实际是 1.2k 张图 + 坏判据 + 无卷积无 BN 的 2 层 MLP。**可复现的只剩**：连续 tanh 回归到有序索引、G/D 同 lr、1:1 更新这套规则会把 D 收敛成常数预测器，且削弱 D（lr_D 1e-5、3:1 步数）也修不好，怀疑真假梯度相互抵消 | `01_盘点与重启建议.md:279-299`、`00:38-47` |
| 朴素贝叶斯混合（逐像素独立） | 条件独立破坏空间相关性；**正确定性**：逐位均值必败，逐位边缘分布只要够尖可行 | `00:49-57`、`01:306-313` |
| 条件 GAN（调色板索引当标量） | 离散索引被当连续标量，颜色空间拓扑被破坏；可修 | `00:59-67`、`PHASE2_TIMELINE.md:383` |
| 自回归 | 教师强制与自采样分布不匹配（exposure bias）→ 模式崩塌；可修（scheduled sampling / Gumbel） | `00:69-77`、`01:320` |
| 蒸馏（一期） | 目标逐位置独立 + 教师本身有噪声 → 学到噪声均值，输出像马赛克 | `00:79-85`、`01:321` |
| 卷积 + MSE | 重建损失的最优解是条件均值 → 退化成 2 色；像素级灰度条件下 28.14% 的位置多峰，连续标量原理上只能取均值 | `00:87-97`、`512_to_32_FINAL.md:39-53` |
| 卷积 v2（"未知"与"白色"编码冲突） | 编码设计错误 → 中轴线白色占 25.81%；加未知标志通道后降到 9.77%，acc 0.567→0.677（**这条是干净的真 bug**） | `00:99-109`、`01:322` |
| 混合近邻 + 对称（唯一能出图的） | 依赖近邻查找，不可移植红石；实际缺的是**采样**（取众数即退化：颜色数 3.38、出现重复） | `00:111-119`、`01:243-262` |

### F2 二期撞死的路

| 路线 | 失败原因 | 出处 |
|---|---|---|
| 逐位置/逐对**取众数**（确定性输出） | 同一近邻模型，采样→众数：颜色数 4.95→3.38，最近邻中位 32→19，出现 2/64 完全重复；视觉退化成大色块 | `01_盘点与重启建议.md:243-247` |
| **连续场 TV 惩罚** | 惩罚"差值大小"不惩罚"跨级"（6.49 与 6.51 差 0.02 却落在不同级）；λ 0→0.2 只把同色率 73.2→74.4%，acc 掉 3.7 点 | `512_to_32_FINAL.md:62`、`512_to_32_RESULTS.md:52-67` |
| 整数吸附惩罚 `\|y−round(y)\|` | MSE 已在推向整数目标，纯冗余，实测零效果（22.4→22.5） | `512_to_32_FINAL.md:63` |
| 直通量化（STE）目标 | 梯度与未量化完全相同，只改前向，数学空操作（26.6% vs 27.7%） | `512_to_32_FINAL.md:64-67` |
| 对四舍五入后的场做 TV | 同样受"梯度没变"限制 | `512_to_32_FINAL.md:65` |
| 混合密度输出（2 分量 / 64 输出） | 落级 51.2% 但 acc 0.7364→0.4782；选峰信息不在输出里 | `512_to_32_FINAL.md:102-125` |
| 训练期烘焙 min-max 拉伸（`--stretch-targets`） | 30.3% vs 54.1%（p=4.8e-3）；网络逐位输出拿不到整脸 min/max，MSE 又拉回均值 | `ROUND3_RESULT.md:9-21`、`:39-53` |
| 对比度损失项 | 6/64 像素被钉成纯黑作弊；acc 0.4128→0.2830 | `CONTRAST_ANALYSIS.md:69-75`、`STRETCH_EXPERIMENT.md:28-49` |
| **全局固定增益 ×1.7 替代 min-max** | 第 4 轮实测 30.9% vs 58.8%（n=97/97）；红石省下的极值+除法电路不值得 | `results/dimmix/round4_stats.json:11`、`:25`（文档侧只有 `FOUR_KNOBS.md:53` 一句提及） |
| 逐通道仿射增益 | 9.85 vs 11.25，匹配错对象 | `DIM4G_READY.md:55-70` |
| 球状采样限制 | 半径 3.0 球覆盖 9.7%、坏率 32.8%，只比全体好 0.7 点 | `RESEARCH_ROUTE.md:287-294` |
| 用线性/径向标量判据分"中/差" | 全部接近随机（\|PC1\| 0.372、PC1-2 半径 0.399、PC0-3 半径 0.522、\|PC3\| 0.609） | `STYLE_SWITCHES_MLP_COLOUR.md:30-39` |
| 用 λ 扫描/相干性约束消除杂色 | 能缓解不能消除（治疗方向最终是"换输出编码 + 后处理"） | `PHASE2_TIMELINE.md:132`、`512_to_32_RESULTS.md:63-67` |
| 两阶段彩色（灰度→上色） | 生成时出荧光色；重建灰度一致性 2.21 超阈值；E6 无灰度更好 | `COLOUR_EXPERIMENT_RESULT.md:43`、`:59-72` |
| 明度/灰度图推断颜色 | R²=0.06–0.11（饱和度 0.073、色相 0.062） | `STYLE_AND_COLOUR.md:106-118` |
| 彩色 C1 结构损失（行方差+眼睛对比） | −20.2 点（p=0.004）；模型用"眼睛行挖深色横带"满足损失 | `SUMMARY.md:156`、`RESEARCH_ROUTE.md:197-206` |
| 彩色 C2 对抗（软平均 / 直通硬采样） | −17.9 点（p=0.008）；硬采样 D 0.86→0.99、眼睛对比崩到 1.27 | `SUMMARY.md:157-158` |
| 训练 4×（12k→48k）提升彩色 | 仅边缘 +6.5%，其余不变 | `SUMMARY.md:154` |
| 加宽 2×（128→256）提升彩色 | 无变化 | `SUMMARY.md:155` |
| K=32 调色板 | 60.7% 与 K=8 的 50.0% 不显著（p=0.879），成本 3 倍 | `SUMMARY.md:99-105` |
| 事前把调色板限定为 MC 色再训练 | 33.6% vs 自由训练后吸附 57.3%（p=0.0008）；k-medoids 只看重建误差，选不出纯白/点缀色 | `SUMMARY.md:113-127` |
| 循环评估（同打分器筛+判） | 得 100%/98.5% 的假结果 | `RESEARCH_ROUTE.md:98-102`、`NIGHT_COLOUR_LOG.md:77-88` |
| 用"分数 > 0"当通过阈值 | 得 98.5%，阈值必须按留出集校准到实测可接受率 | `RESEARCH_ROUTE.md:104-108` |
| 展示榜单头部样板 | 头部图全黑/全同；应从幸存者随机抽样 | `RESEARCH_ROUTE.md:110-114` |
| 用交叉验证外推筛选收益（+19 点） | 实测 +2.6 / +4.7 点；r²≈0.3 | `SUMMARY.md:56-59` |
| 靠筛选拿到非二次元风格 | 类游戏 47.2%、类三次元 22.6%（不可超基线 2.7 倍），**必须改模型** | `STYLE_CONDITIONED.md:20-24`、`:106-108` |
| 两层窄输出网络（4-h-h-32） | 第二层只有 32 宽，表达力被卡（2,176 < 2,304） | `PHASE2_TIMELINE.md:232` |

---

## G. 与四期的接口

四期主线是"彩色头像生成（可能用 diffusion）"。二期留下的彩色相关资产**比灰度侧更适合直接用**，因为彩色侧已经完成了"从连续 RGB 退到调色板分类"的转向，并已量化了量化下限、调色板、MC 色表。

### G1 可直接加载的权重（文件路径级）

| 资产 | 路径 | 规格 / 状态 |
|---|---|---|
| **彩色最终生成器（首推起点）** | `02_phase2_avatar/webapp/colour_k8snap_model.json` | 8→128→128→32×8，50,176 乘法，3 bit/像素；`meta.type=palette_index`、`scheme=mc_high_snapped`、8 色 palette 已内嵌（0–15 标度）；round 12 实测 61.1% |
| 彩色其它调色板尺寸 | `webapp/colour_pal8_model.json`、`colour_pal16_model.json`、`colour_pal32_model.json` | K=8/16/32，50,176/82,944/148,480 乘法 |
| 彩色事前限定 MC 色（反例，可作对照） | `webapp/colour_k8mc_model.json` | round 11 的 `k8mc` 臂，33.6% |
| 彩色 C1/C2 实验权重（反例） | `webapp/colour_c1_model.json`、`colour_c2_model.json`、`colour_c1c2_manifest.json` | round 12；C1/C2 均有害 |
| 彩色连续 RGB 生成器（8→128→128→96） | `webapp/colour_e6_model.json`（2.29 MB，含 `stage1`） | E6 单阶段，RGB MAE 1.349；**但"糊和脏"正是它被调色板方案取代的原因**（`SUMMARY.md:91`） |
| 连续 RGB 全变体权重 | `results/colour/stage2_E1.pt` … `stage2_E8.pt` | E1–E8，可直接复训/蒸馏 |
| 灰度生成器（2,304） | `webapp/pca22_dim4w64L1_model.json` | 4-64-32，`meta.multis=2304`、`post=stretch`；灰度定案模型 |
| 灰度全家族（10 个规模） | `webapp/grayscale_family.json` + `pca22_dim4w{32,64,96,128,192,256}{,L1,L2}_model.json`、`pca22_dim4x_model.json` | 用于宽度/层数对照 |
| 二次打分器（14 乘法，AUC 0.853） | `webapp/scorer_quad.json` | `type/in_dim/coef/thresholds/meta` 齐全，可直接复算 |
| 两个 4→16→1 打分器 | `webapp/scorer_accept.json`、`scorer_anime.json` | 可接受 / 类二次元 |
| 六个风格高斯 | `webapp/styles_gauss.json`（均值 + Cholesky）、`webapp/style_envelopes.json`、`results/styles/styles.json` | 1σ 坏率全 0 |
| 11 档灰归并表 | `webapp/merge_mc11.json`（`_lut` + `names`） | 16→11 映射，零成本 |
| 偏好奖励模型（w128 口径） | `webapp/preference_reward.json` | 含 `cv_auc`、`caveat` 字段 |

### G2 可直接继承的数据与调色板

| 资产 | 路径 | 说明 |
|---|---|---|
| 彩色对齐数据缓存 | `results/colour/cache.npz`（58.9 MB）+ `results/colour/cache_meta.json` | 321,219 张对齐样本，四期训练直接可用 |
| RGB 8×8 数据集 | `data/faces8_rgb_v2.npy`、`faces8_rgb_v2a.npy` | 与索引版逐行同源 |
| 8 色调色板（自由拟合 + 吸附） | `results/colour/palette/palette_k8.json`、`palette_k8_optimised.npy` | `k8snap` 用的就是它 |
| 多种调色板对照 | `results/colour/palette/{P8,P16,P32,S8,S16,S32}.json`（P=自由、S=吸附） | 含 `cmp.json` |
| **MC 色表（四档 × 62 基色 + 61 方块）** | `results/colour/mcfit/blocks.json`、`snapped.json`、`fit_all.json`、`fit_normal.json`、`grey_ladder.json`、`merge11.json` | 四期上色/吸附的唯一权威色表 |
| 量化工件 | `results/colour/quantise/quantise.json` + `QUANTISE.png`；`results/colour/encoding/encoding.json` | 位深误差曲线（3–6 bit） |
| 4-PCA 全空间标签 | `results/pca4map/pca4_plane_marks.json`（**1,372 条**）、`labels_analysis.json`、`scorer_shapes.json`、`geometry.json`、`presets.json` | 安全区/夹取/打分器的事实依据 |
| 12 轮打标标签 | `results/preference/labels.jsonl`（3,752 条，含 round/scheme/model 字段）、`rounds.json`、`label_analysis.json`、`filter_gain.json` | 可跨期复用（注意 H-1 的轮次编号重载） |
| 各轮统计 | `results/dimmix/round{2,3,4,7,9,10}_stats.json` + 对应 PNG；`results/colour/round8/*.json`、`k8schemes/round11.json`、`yield/round12.json` | 四期做基线对照时直接引用 |
| 彩色筛选/代理良率 | `results/colour/colour_scorer.json`、`colour_filter.json`、`colour_filter_fair.json`、`proxy_accept.json`、`colour_criteria.json`、`three_way.json` | 双打分器协议 |

### G3 可直接复用的脚本（`02_phase2_avatar/src/`）

- 彩色生成与评估：`colour_palette.py`、`colour_palette_cmp.py`、`colour_gan.py`、`colour_gan_compare.py`、`colour_yield_train.py`、`colour_eval.py`、`colour_metrics.py`、`colour_quantise.py`、`colour_stage2_train.py`、`colour_sampler.py`、`colour_filter_fair.py`、`colour_three_way.py`。
- MC 色表与灰阶：`mc_fit.py`、`mc_shade.py`、`mc_merge11.py`、`mc_block_cmp.py`、`palette_from_jar2.py`、`palette_v5.py`。
- 空间测量与打分器：`pca4_map.py`、`pca4_geometry.py`、`pca4_labels.py`、`export_scorer_quad.py`、`styles_gauss.py`、`k8_schemes.py`、`k8_figure.py`。
- 导出与校验：`export_colour.py`、`export_generic.py`、`export_compressed.py`、`js_forward_check.js`（独立前向校验，最大误差 5.7e-6，`512_to_32_FINAL.md:96`）。
- 环境：`02_phase2_avatar/.venv/Scripts/python.exe`（torch 2.14.1+cpu；`01_盘点与重启建议.md:96-105`）。注意 `pip` 在沙箱不可用，装包用 `src/install_wheels.py`（`01:440-455`）。

### G4 四期应当换方法重做的线

| 该放弃/改法的线 | 理由 | 出处 |
|---|---|---|
| **连续 RGB + MSE 回归**（E1–E8 那条） | 目标上界被"糊和脏"卡死；E6 的 MAE 1.35 已接近上限但视觉不可用；二期用调色板分类才解决 | `SUMMARY.md:91-95`、`COLOUR_EXPERIMENT_RESULT.md:59-72` |
| **靠调色板分类再往上顶良率** | K=8/16/32 无显著差别（50.0/51.8/60.7，p=0.879/0.175/0.282），训练 4×、加宽、C1、C2 全灭，**剩下的杠杆只有输入维度** | `SUMMARY.md:99-105`、`:150-158` |
| **8 维颜色 PCA 作为条件** | 只解释 75.0% 方差，是 GAN 判据上不去的直接原因（支撑集够不到真实分布）；8→12/16 维只加 1.0%/2.0% 乘法 | `SUMMARY.md:160-174`、`:215-225` |
| **自回归/逐像素判别式细化** | 一期 exposure bias 死路；二期未在彩色上重试 | `00:69-77`、`01:320` |
| **两阶段（灰度→上色）** | E6 反证；且生成时会炸成荧光色 | `COLOUR_EXPERIMENT_RESULT.md:49-72` |
| **筛选器驱动良率** | r²≈0.3 的天花板；彩色的 67→79→88→93% 是**交叉验证外推**（双打分器协议），不是盲测实测 | `SUMMARY.md:56-59`、`NIGHT_COLOUR_LOG.md:90-99` |
| **手动种子点的风格自动化** | 6 个风格的中心需用户给点，未自动化 | `PHASE2_TIMELINE.md:369` |
| **打分器当最终裁判** | 角落 15% 误判；必须用离散全量标签 | `RESEARCH_ROUTE.md:318-327` |
| diffusion 的可用起点 | 二期**没有**做过 diffusion；唯一可迁移的结论是：①离散/调色板输出优于连续回归 ②多峰条件分布需能表达双峰的输出形式 ③4 bit/通道的量化下限 0.2049 与 64 色 6 bit 的 6% 折中 | `512_to_32_FINAL.md:71-82`、`NIGHT_COLOUR_LOG.md:34-35` |

---

## H. 文档本身的问题

| # | 问题 | 证据 / 说明 |
|---:|---|---|
| H-1 | **两套"轮次"编号重载，且两篇都叫 ROUND2 却是不同实验** | `ROUND2_RESULT.md` 的"第二轮"= w128 guided/explore 158 条，落在 `labels.jsonl` 的 `round=None`（`results/preference/rounds.json`）；`ROUND2_POSTPROCESS_RESULT.md` 的"第 2 轮"= dim4x/dim4/dim22 334 条，即 `round=2`。同理 `ROUND3_RESULT.md` 的"第 3 轮"是 `round=3`（226 条）。四期若按文件名理解轮次，会把三套实验混成一套 |
| H-2 | **编号冲突：多篇 00/01 且阅读顺序被定义两次** | `docs/00_已知死路与重启指引.md` 与 `docs/01_盘点与重启建议.md` 是 10-16 的一对；另有 `audit_tmp/AUDIT_REPORT.md`（编号外的审计）与 `results/DOC_INDEX.md`（"00 类"总索引）。`README.md:11-17` 与 `results/DOC_INDEX.md:1-3` 各自给出一套阅读顺序，且互不相同（前者把 `RESEARCH_ROUTE` 排第 2，后者按 mtime 排） |
| H-3 | **死路条数自相矛盾：8 还是 9** | `docs/00:36` 称"九种方案，全部失败"，但其汇总表只有 8 行（`00:123-132`）；`01:277` 标题写"九条死路"而表内只有 6 项；`PHASE2_TIMELINE.md:9`、`:21` 一律说"八条" |
| H-4 | **第 4 轮实验有数据、无文档，且被主文档漏报** | `results/dimmix/round4_stats.json`（97/97/78 条）给出 dim4x **58.8%**、dim4g **30.9%**、dim22 7.7%，但 `docs/` 下没有 `ROUND4_RESULT.md`；只有 `docs/FOUR_KNOBS.md:53` 一句"第 4 轮的结论是 min-max 比全局增益好得多"。而主文档 `PHASE2_TIMELINE.md:200` 只说"固定增益更接近真实分布，且成本低得多"，**没报 30.9% 这个决定性负结果**——四期若照主文档选固定增益会直接踩坑 |
| H-5 | **主文档数值错位（把 dim4x 的值贴到 dim4st 上）** | `PHASE2_TIMELINE.md:198`："把拉伸烘焙进训练 \| 54.1%（不显著劣于拉伸）"。但 `docs/ROUND3_RESULT.md:9-10` 与 `results/dimmix/round3_stats.json` 明确：dim4st = **30.3%**、dim4x = 54.1%，两者差 +23.8 点、p=4.8e-3（**显著**）。主文档把"显著更差"写成了"不显著" |
| H-6 | **同一量在三个文档里三个值（dim4g 的对比度/发灰率）** | `DIM4G_READY.md:50`：×1.7 → 对比度 11.36 / 发灰 8.73%；同文件 `:62`：×1.7 → 11.25 / 7.1%；`ROUND3_RESULT.md:68`：×1.76 → 11.36 / 7.4%。三处口径（不同增益、不同采样数）未说明 |
| H-7 | **"准确率"口径漂移，同名不同义** | PCA-4 有 **0.2104**（512 输出+64 维噪声+1500 步，`PCA4_8_ANALYSIS.md:31`）、**0.4041**（32 输出+TV5+12000 步无噪声，`DIM_LADDER_TRAINED.md:13`）、**0.4128**（`CONTRAST_ANALYSIS.md:69`）、**0.3651**（4-64-32 定稿，`PHASE2_TIMELINE.md:227`）。PCA-22 有 0.3820 / 0.8582 / 0.8594 / 0.7719 四个值。同表横比会得出相反结论 |
| H-8 | **"62%"这个数字有三个来源，含义不同** | `README.md:61` 的"约 62%（不筛选）"是 round 7/9 三臂的 60.5–61.4 / 62.5–64.8；`ROUND2_POSTPROCESS_RESULT.md:11` 的 62.1% 是 dim4x（round 2）；dim4x 之后又变 54.1%(round 3)→58.8%(round 4)。报"62%"必须带臂名与轮次 |
| H-9 | **量化误差三套口径混用** | ①灰度 16 色调色板 425.2835 MSE / **RMS 20.62**（旧文档错写 11.9，`01:74-77`）②`palette_avatar16.json` 的 `quant_err=1183.2` 是**三通道求和**，与 425.3 差 3 倍不可直接比（`01:77`）③彩色 4bit/通道 **0.2049 级**（`COLOUR_EXPERIMENT_RESULT.md:13`）与共享调色板 **0.8333**（`NIGHT_COLOUR_LOG.md:24`）是不同量纲 |
| H-10 | **空壳交付物** | `results/colour/stage2_results.json` 只有 `{}`（2 字节），但 `COLOUR_EXPERIMENT_RESULT.md:148` 称其为"全部指标"。真实数值散落在 `results/colour/stage2_readout.json` 与 `stage2_results_E8.json` |
| H-11 | **引用了不存在的路径（3 处，全在已被取代的设计文档）** | `COLOUR_EXPERIMENT_DESIGN.md:163-165` 承诺 `results/colour/HUE_ERR.png`、`results/colour/summary.json`、`webapp/colour_model.json` ——三者均不存在；实际对应 `COLOUR_SAMPLER.png`、`stage2_readout.json`、`colour_e6_model.json`。已核对：其余全部 inline 资产路径（`results/`、`webapp/`、`src/`）与 8 个 `![]()` 图片链接**均可解析** |
| H-12 | **索引与实际数目不符** | `README.md:16` 与 `PHASE2_TIMELINE.md:405` 说 `results/DOC_INDEX.md` 收录"34 份"，而 `DOC_INDEX.md:3` 自称"33 份"；且 DOC_INDEX（10-18 前生成）**未收录**期根 `README.md` 与 `PHASE2_TIMELINE.md` 自身 |
| H-13 | **"旧纯化数字都要改写"这条自我修正从未落实** | `STYLE_AND_COLOUR.md:57` 宣布"之前所有纯化数字（58.8% → 90.2%）都要改写：它们优化的是'更像二次元'，不是'更可接受'"；但同日之后的 `KNOBS_AND_PURIFY.md:94` 与 `DIM4G_READY.md:37` 继续原样引用同一组数字，`PHASE2_TIMELINE.md` 也未收录这条勘误（其 A.1 表 9 条里没有它） |
| H-14 | **两套灰度实物化方案并存、互不引用** | ①16 个原版方块（`PHASE2_WRAPUP_V2.md:63-82`，`palette16_v5.json`，平均误差 3.52、16 级）②11 档地图 HIGH 归并（`SUMMARY.md:66-83`，`merge_mc11.json`，66.0% 良率、仅 9 档可分）。主文档只写 ②（`PHASE2_TIMELINE.md:278-283`），README 只写 ②（`README.md:62-63`），`PHASE2_WRAPUP*` 只写 ①。四期要明确继承哪一个 |
| H-15 | **章节编号错乱** | `docs/512_to_32_FINAL.md`：正文顺序为 一→二→三→四→五→**八**（:102 补充实验）→**七**（:129 建议），且"六"是交付表（:85）。引用时按节号会错位 |
| H-16 | **版本链只在两处声明，被取代的旧文件自身无任何标记** | 取代关系仅见于 `PHASE2_TIMELINE.md:397-402`（PLAN/RESULTS→FINAL、WRAPUP→V2、DIM_YIELD_DESIGN→DIM_LADDER_TRAINED、STYLE_CONDITIONED→FILTER_FINAL、COLOUR_DESIGN→RESULT）与 `README.md:17` 一句"含已被取代的旧版本"。`docs/` 里的旧文件正文完全没有"已作废"提示，容易误引（本轮就有 3 处旧路径） |
| H-17 | **归属错位：一期的死路写在"二期指引"里** | `docs/00_已知死路与重启指引.md:1` 标题是"二期重启指引"，但第 1 节九条死路实为一期结论；`PHASE2_TIMELINE.md:411` 明确"无条件 GAN 属一期（10-04/05），二期在 10-16 只是验证性复核" |
| H-18 | **阶段/日期在文档间不完全一致** | `PHASE2_TIMELINE.md:46` 承认阶段 4（输出压缩 12:03–14:00）与阶段 5（输入压缩 11:48 起）并行；`results/DOC_INDEX.md` 只给时分不给日期，28 篇 docs 全落在同一"2026-10-17"段（12:03–22:26），但其中 `DIM_LADDER_TRAINED.md:3` 自述 10-17、而 `SUMMARY.md`/`RESEARCH_ROUTE.md` 的 mtime 是 10-18。跨文档排时间线时不能只看 DOC_INDEX |
| H-19 | **"已验证"与"推测"在同一文档内混排** | 例：`01_盘点与重启建议.md:226` 把"真假梯度相互抵消"标为"（推测，未证实）"，但 `:298` 又当作解释使用；`PHASE2_TIMELINE.md:188-200` 把"固定增益更接近真实"（统计量已测）与"红石实现简单一个量级"（未测良率）并列陈述 |

---

## I. 我读过的文件清单

> 行数用 `splitlines()` 计（与 `read` 工具显示的总行数一致）。共 32 个文件、5,098 行。

| # | 文件 | 行数 |
|---:|---|---:|
| 1 | `02_phase2_avatar/docs/00_已知死路与重启指引.md` | 237 |
| 2 | `02_phase2_avatar/docs/01_盘点与重启建议.md` | 588 |
| 3 | `02_phase2_avatar/docs/512_to_32_FINAL.md` | 138 |
| 4 | `02_phase2_avatar/docs/512_to_32_INDEX.md` | 62 |
| 5 | `02_phase2_avatar/docs/512_to_32_PLAN.md` | 122 |
| 6 | `02_phase2_avatar/docs/512_to_32_RESULTS.md` | 125 |
| 7 | `02_phase2_avatar/docs/ALIGNMENT_BUG.md` | 126 |
| 8 | `02_phase2_avatar/docs/COLOUR_EXPERIMENT_DESIGN.md` | 179 |
| 9 | `02_phase2_avatar/docs/COLOUR_EXPERIMENT_RESULT.md` | 149 |
| 10 | `02_phase2_avatar/docs/CONTRAST_ANALYSIS.md` | 103 |
| 11 | `02_phase2_avatar/docs/DIM4G_READY.md` | 94 |
| 12 | `02_phase2_avatar/docs/DIM_LADDER_TRAINED.md` | 110 |
| 13 | `02_phase2_avatar/docs/DIM_YIELD_DESIGN.md` | 99 |
| 14 | `02_phase2_avatar/docs/FILTER_FINAL.md` | 101 |
| 15 | `02_phase2_avatar/docs/FOUR_KNOBS.md` | 73 |
| 16 | `02_phase2_avatar/docs/KNOBS_AND_PURIFY.md` | 103 |
| 17 | `02_phase2_avatar/docs/NIGHT_COLOUR_LOG.md` | 143 |
| 18 | `02_phase2_avatar/docs/PCA4_8_ANALYSIS.md` | 111 |
| 19 | `02_phase2_avatar/docs/PHASE2_WRAPUP.md` | 131 |
| 20 | `02_phase2_avatar/docs/PHASE2_WRAPUP_V2.md` | 104 |
| 21 | `02_phase2_avatar/docs/ROUND2_POSTPROCESS_RESULT.md` | 84 |
| 22 | `02_phase2_avatar/docs/ROUND2_RESULT.md` | 146 |
| 23 | `02_phase2_avatar/docs/ROUND3_RESULT.md` | 97 |
| 24 | `02_phase2_avatar/docs/STRETCH_EXPERIMENT.md` | 112 |
| 25 | `02_phase2_avatar/docs/STYLE_AND_COLOUR.md` | 136 |
| 26 | `02_phase2_avatar/docs/STYLE_CONDITIONED.md` | 108 |
| 27 | `02_phase2_avatar/docs/STYLE_SWITCHES_MLP_COLOUR.md` | 146 |
| 28 | `02_phase2_avatar/docs/W128_LABELING_RESULT.md` | 108 |
| 29 | `02_phase2_avatar/PHASE2_TIMELINE.md` | 426 |
| 30 | `02_phase2_avatar/SUMMARY.md` | 280 |
| 31 | `02_phase2_avatar/RESEARCH_ROUTE.md` | 483 |
| 32 | `02_phase2_avatar/README.md` | 74 |
| | **合计** | **5,098** |

**为核对事实额外读过的非交付文件**（未计入上表）：
`results/DOC_INDEX.md`（177 行）、`results/restart_probe.json`、`results/pca4map/labels_analysis.json`、`results/pca4map/scorer_shapes.json`、`results/pca4map/geometry.json`、`results/dimmix/round{2,3,4,7,10}_stats.json`、`results/preference/rounds.json`、`results/preference/labels.jsonl`（3,752 行，程序统计）、`results/pca4map/pca4_plane_marks.json`（程序统计 1,372 条）、`webapp/pca22_dim4w64L1_model.json`、`webapp/colour_k8snap_model.json`、`webapp/merge_mc11.json`。
