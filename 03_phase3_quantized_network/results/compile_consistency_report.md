# 仿真器 vs 已编译量化模型 · 一致性报告

日期：2026-10-24
对象：`viewer/plane_sim.html`
基准：`src/compiler.py`（`compile_network` / `integer_forward`）
模型：`02_phase2_avatar/webapp/pca22_dim4w64L1_model.json`

---

## 一、结论

| 项目 | 结果 |
|---|---|
| 层1 权重编译（b = 8…12） | **逐元素完全一致** ✓ |
| 层2 权重编译（b = 8） | **逐元素完全一致** ✓ |
| 层1 输出步长 δ_out | 最大差 **0** ✓ |
| 头部输出 vs Python 整数前向 N | **168 / 168** ✓（42 用例 × 4 种灰/白朝向） |
| 灰环值 = K | ✓ |
| 白环值 = W₀ | ✓ |
| 默认形态头部（无输入反转、无位权镜像） | **336 / 336** ✓ |
| 输入顺序反转 | 96 / 336 ✗ |
| 位权朝次方向升高（镜像） | 0 / 336 ✗ |

---

## 二、修改的两处编译错误

### ① 舍入方式

`compiler.py` 用 `np.rint`（四舍六入五取偶），仿真器原来用 `Math.round`（五入）。
层1 b=8 有 **32 处**权重正好落在 `.5`：

```
ch1 in0 bit4:  Python 220   JS 221        （真值 220.5，220 是偶数）
```

改成 `rint()` 后：**0 处不一致**。

### ② 层2 的输入步长必须逐通道

`compiler.py` 里每层的 `delta_j` 是该层输入的实数分辨率。层1 是均匀的
$7/256$，但**层2 的 delta 是层1 的输出步长 $c_i/\sigma_i$**，64 个通道不等：

```
min 7.5913e-3   max 5.1983e-2   均值 2.2171e-2   最大/最小 = 6.85 倍
```

仿真器原来对层2 也用统一的 7，导致 **15362 / 16384** 个权重错，最大差 1220。

### ③ `b` 必须同时是"输入位宽"

`compiler.py` 中 `in_bits` 既是权重位宽也是输入端位宽（层1 都是 8）。
把两者拆开会让权重按 $2^{b-8}$ 放大，$b>8$ 时越界：

| b | 非最高位 \|W\| 上界 | 是否合法 |
|---|---|---|
| 8 | 882 | ✓ |
| 9 | 1764 | ✗ 超容器上限 |
| 10 | 3528 | ✗ |

修正后（`inBits` 默认等于 `b`）：**b = 8…12 全部 882 / 1764，全部合法**。

---

## 三、模型的实际容器占用（层1，b=8）

$$
\text{MSB 权重 } |W^{(7)}_i|:\ \max = 1764,\quad 23/64\ \text{个通道} > 882
$$

$$\text{MSB 信号 } S = 15+|W|:\ \text{范围 } [18,\ 1779],\qquad 138\ \text{格} > 897$$

$$\Rightarrow\ \text{必须用环（容器放不下），与设计一致}$$

配对情形统计（32 对）：

| 情形 | 对数 |
|---|---:|
| Ⅰ（两路都 $>882$） | 3 |
| Ⅱ（一路 $>882$） | 17 |
| Ⅲ（两路都 $\le 882$） | 11 |

$$\text{即：绝大多数通道需要环，}\ 3\ \text{对需要 } \Delta\ \text{桶 + 减法}$$

---

## 四、仍然错误的两个自由度

### 输入顺序反转（96/336）

$$\text{只有 }Z=0\text{ 那条链（含 MSB）的减量不同，后三条链完全一致}$$

配对区在 $Z=3$ 输出四个 MSB 值是正确的（实测 `1779 / 1603 / 176`），
但进入链的侧输入与块的列归属在 `inRev` 下有一处错位。

### 位权镜像（0/336）

镜像会重建全部 cell，`src` 标记、链的左右顺序、端口位置都需要跟着翻。
目前只翻了坐标与比较器朝向（$N\leftrightarrow S$），依赖关系没翻。

---

## 五、复现方法

```powershell
$py   = '...\python.exe'
$node = '...\node.exe'

# 1) 导出 Python 基准
& $py   03_phase3_quantized_network\src\export_py_compile.py
& $py   03_phase3_quantized_network\src\export_py_forward.py

# 2) 逐元素比对 + 端到端比对
& $node 03_phase3_quantized_network\src\verify_all.js
```

---

## 六、存档

```
viewer/archive/plane_sim_20261024_191414.html   默认形态 384/384
viewer/archive/plane_sim_20261024_191504.html
viewer/archive/plane_sim_20261024_191733.html
viewer/archive/plane_sim_20261024_191811.html
viewer/archive/plane_sim_20261024_193456.html   ← 编译已对齐 Python（当前）
```
