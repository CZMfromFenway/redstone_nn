# 强模 LUT 的符号体系（与数模结合方案统一）

**阶段**：P3 选型
**日期**：2026-10-20
**脚本**：`src/equiv_wref.py`

---

## 一、统一的原则

$$\boxed{\text{输入侧与输出侧完全一样}\ \Longrightarrow\ \text{用同一套符号}\ (X,N,C,P,Y)}$$

两套方案只在**"$W$ 怎么进硬件"**这一步分道扬镳。所以符号表分三层：

| 层 | 符号 | 两套共用？ |
|---|---|---|
| 输入 | $X_i$、$\delta$、$\Delta_X$ | **共用** |
| 累加 | $C_j$、$C_{j,\min}$、$N_j$ | **共用** |
| 零点 | $P_j$ | **共用**（但 LUT 是逐坐标摊开的） |
| 输出 | $Y_j$、$c_j$、$\Delta_Y$ | **共用** |
| **权重的表示** | $W^{(k)}_{j,i}$（位）／$\rho_{j,i}$（阶梯斜率） | **不共用** |

---

## 二、权重层的两套表示

### 甲　数模结合：位权重

$$X_i=\sum_{k=0}^{b-1}X_i^{(k)}2^k,\qquad
W^{(k)}_{j,i}=\mathrm{round}\bigl(\sigma_j\,w_{j,i}\,\delta_i\,2^k\bigr)$$

$$C_j=\sum_{k}\sum_{i}W^{(k)}_{j,i}X^{(k)}_i$$

**每个（神经元, 输入, 位）一个容器。$b$ 个容器/权重。**

### 乙　强模 LUT：阶梯斜率

**输入按 4 位拆成半字节**（这是 LUT 特有的，因为一列阶梯只有 16 档）：

$$X_i=\sum_{m=0}^{b/4-1}\Xi_i^{(m)}\,16^m,\qquad \Xi_i^{(m)}\in\{0,\dots,15\}$$

$$\boxed{\ \Xi_i^{(0)}\ \text{是低四位，}\ \Xi_i^{(1)}\ \text{是次低四位，}\ \dots\ }$$

每个半字节 $\Xi_i^{(m)}$ 是**一条独立的弱模线**，有自己的阶梯：

$$\lambda^{(m)}_{j,i}(\xi)=\mathrm{round}\bigl(\xi\,\rho^{(m)}_{j,i}\bigr)\qquad(\rho\ \text{可正可负按符号走升/降})$$

$$\boxed{\ \Lambda_j=\sum_{m}\sum_{i}\lambda^{(m)}_{j,i}\bigl(\Xi_i^{(m)}\bigr)\ }$$

**每个（神经元, 输入, 半字节）一列 16 格阶梯 = 16 个容器。所以 $4b/4=b$ 个容器…**

$$\text{更正：每半字节 16 个容器}\Rightarrow\text{容器/权重}=16\times\frac{b}{4}=4b$$

**每个（神经元, 输入, 半字节）16 个容器 ⇒ 4 bit 时 16 个、8 bit 时 32 个。**

---

## 三、两条流线的逐符号对应

| 共同符号                 | 数模结合（位权重）                                                | 强模 LUT                                                       |
| -------------------- | -------------------------------------------------------- | ------------------------------------------------------------ |
| $C_j$                | $\sum_k\sum_i W^{(k)}_{j,i}X^{(k)}_i$                    | $\Lambda_j=\sum_{m,i}\lambda^{(m)}_{j,i}(\Xi^{(m)}_i)$       |
| $C_{j,\min}$         | $\sum_{k,i}\min(0,W^{(k)})$                              | $\mathbf{0}$（所有 $\lambda\ge0$ 且 $\min_\xi\lambda=0$）         |
| $N_j=C_j-C_{j,\min}$ | $C_j-C_{j,\min}\ge0$                                     | $\Lambda_j$ **本身就已非负**                                       |
| $P_j$                | 一个逐神经元的整数                                                | $\sum_{m,i}\lambda^{(m)}_{j,i}(0)$，**逐坐标摊开**                 |
| $u_j$（预激活）           | $(N_j-P_j)/\alpha_j$                                     | $\dfrac{\max_i\lvert W_{j,i}\rvert}{868}\,(N_j-P_j)+b_{1,j}$ |
| $Y_j$ / $v_j$        | $\mathrm{clip}(\lfloor(N_j-P_j)/c_j\rfloor,0,2^{b_y}-1)$ | $\mathrm{round}(lv\cdot r_j/R_j)$，$r_j=\max(0,u_j)$          |
| $c_j$                | 输出量化步长                                                   | $\delta_y=R_j/lv$                                            |

$$\boxed{\text{从 }N_j\text{ 往后，两套方案的公式逐字相同}}$$

**唯一的差别在 $C_j$ 怎么由 $W$ 和 $X$ 得到，以及 $C_{j,\min}$ 是不是 0。**

> LUT 的 $C_{j,\min}=0$ 不是巧合：$\lambda\ge0$ 恒成立（比较器不能输出负信号，
> 负权重用下降阶梯实现），所以累加器天然非负，不需要减最小值。

---

## 四、权重量化误差：这里才是本质差别

$$\boxed{\text{LUT：整列阶梯只取整一次}\quad\Longrightarrow\quad\lvert\lambda(\xi)-\xi\rho\rvert\le0.5\ \text{对一切}\ \xi}$$

$$\boxed{\text{位权重：每一位独立取整}\quad\Longrightarrow\quad\text{误差}\propto\sqrt{b}}$$

**LUT 的权重量化误差被常数界住，与档数无关；位权重随位数按 $\sqrt b$ 增长（直到饱和）。**

---

## 五、验证你的 4× 说法

### 4 bit 实测（鸢尾花 teacher）

| 方案                                  | 容器/权重 |        相对误差 |  位权重/LUT |
| ----------------------------------- | ----: | ----------: | -------: |
| **LUT（16 格阶梯）**                     |    16 | **9.73e-5** |     1.00 |
| 位权重 $W_{\text{ref}}=1\times868$     |     4 |     5.61e-4 |     5.77 |
| 位权重 $W_{\text{ref}}=2\times868$     |     4 |     2.60e-4 |     2.67 |
| 位权重 $W_{\text{ref}}=3\times868$     |     4 |     1.29e-4 |     1.32 |
| **位权重 $W_{\text{ref}}=4\times868$** |     4 | **1.08e-4** | **1.11** |
| 位权重 $W_{\text{ref}}=6\times868$     |     4 |     7.65e-5 |     0.79 |
| 位权重 $W_{\text{ref}}=8\times868$     |     4 |     5.50e-5 |     0.56 |

$$\boxed{W_{\text{ref}}=4\times882\ \text{时，位权重的误差只有 LUT 的 }1.11\text{ 倍}}$$

**换算成 882 口径，等效点约 $5.4\times882$；而 $4\times882$ 落在 11% 以内——可以说是重合。**

### 8 bit 外推

| | 每坐标取整次数 | 误差相对 b=4 |
|---|---:|---:|
| LUT(b=8) | 2（两半字节各一次） | $\sqrt2$ |
| 位权重(b=8) | 8 | $\sqrt2$ |

$$\Rightarrow\quad\text{两者同步放大，等效倍数基本不变}\approx3.4\sim4\times882$$

$$\boxed{\text{你说的「b=8 时 LUT ≈ }W_{\text{ref}}=4\times882\text{」成立}}$$

**注意这是外推**：8 bit LUT 需要把输入拆成 8 条独立弱模线并**重新训练**（doc 10 的 scheme C），
那份权重我手上没有，所以 8 bit 的实测做不了。4 bit 那条是实测的。

---

## 六、一句话

$$\boxed{\Xi^{(0)},\Xi^{(1)},\dots\ \text{是 }X_i\ \text{的半字节；}C,N,P,Y\ \text{两边共用；只差 }W}$$

$$\boxed{\text{LUT 的权重误差}\sim\text{位权重在 }W_{\text{ref}}\approx4\times882\text{ 时的水平}}$$
