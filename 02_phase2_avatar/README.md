# 二期收官

**状态：已完成**（2026-10-16 15:46 – 2026-10-19）

目标是用神经网络生成 8×8 Minecraft 头像，并压缩到红石可承受的成本。任务已达成，灰度侧具备移植条件。

---

## 从哪读起

| 顺序 | 文档 | 内容 |
|---:|---|---|
| 1 | **`PHASE2_TIMELINE.md`** | **时间线主文档**。八阶段，每阶段任务/做法/结果/指向。读这一份即可掌握全貌 |
| 2 | `RESEARCH_ROUTE.md` | 同一段工作的早期复盘，失败方向的技术细节更细 |
| 3 | `SUMMARY.md` | 彩色的技术摘要，与时间线互补 |
| 4 | `results/DOC_INDEX.md` | 全部 34 份过程文档的清单与摘要 |
| 5 | `docs/` | 过程文档原件，含已被取代的旧版本 |

---

## 交付物

### 模型

| 文件 | 内容 | 成本 |
|---|---|---:|
| `webapp/pca22_dim4w64L1_model.json` | **灰度生成器** 4-64-32 | **2,304 乘法** |
| `webapp/colour_k8snap_model.json` | **彩色生成器** 8-128-128-32×8 | 50,176 乘法 |
| `webapp/scorer_quad.json` | **二次打分器** 14 项 | 14 乘法 |
| `webapp/merge_mc11.json` | 11 档地图灰归并表 | 查表 |
| `webapp/styles_gauss.json` | 六个风格高斯（均值 + Cholesky 因子） | 采样用 |
| `webapp/grayscale_family.json` | 灰度 10 个规模的清单 | — |

### 交互工具

| 文件 | 用途 |
|---|---|
| `webapp/sandbox_standalone.html` | 5.27 MB 单文件，双击即用，离线。含全部灰度模型、两个打分器、风格抽样、矩形夹取 |
| `webapp/sandbox.html` | 同一页面的服务器版 |
| `webapp/plane.html` | 4-PCA 平面标注工具，产出本文所述的 4,096 点标签 |
| `webapp/label.html`、`label_colour.html` | 灰度 / 彩色盲测页 |

### 数据

| 文件 | 内容 |
|---|---|
| `results/pca4map/pca4_plane_marks.json` | 1,372 条完整四维标签，覆盖全部 64 个平面 |
| `results/pca4map/labels_analysis.json` | 各轴坏率、球模型、矩形规则 |
| `results/pca4map/scorer_shapes.json` | 线性/二次 AUC、夹取规则表 |
| `results/preference/labels.jsonl` | 12 轮盲测共 3,752 条标签 |

---

## 最终配置

$$\text{灰度：}\ 4 \to 64 \to 32\ (\text{连续值}) \to \text{min-max 拉伸} \to \text{量化到 16 级或 11 档地图灰}$$

| 项 | 值 |
|---|---|
| 乘法 | 2,304（主网络）+ 14（打分器）+ 0（采样夹取）|
| 实测良率 | 约 62%（不筛选）/ 89.1%（矩形夹取）|
| 每像素 | 4 bit 索引（11 档灰）或 4 bit 亮度（16 级）|
| 可实物化 | 11 档灰对应 11 个地图 HIGH 色，全部可摆 |

---

## 留给三期的问题

1. **全部已完成实物都是强模**，生存模式下无法实装。这是三期要正面处理的问题。
2. 灰度模型的 32 个输出都是**连续值**，红石端需要一个确定的量化方案。
3. 二次打分器与矩形夹取需要与主网络**共存**在同一装置里。
4. 彩色模型 50,176 乘法，成本是灰度的 21 倍，是否移植属于三期决策。

三期的任务定义见 `../03_phase3_quantized_network/docs/00_任务与路线.md`。
