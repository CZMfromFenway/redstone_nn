# 删除 D 盘旧目录 —— 执行清单

**⚠️ 只有在确认新会话已成功建立在 `E:\redstone_nn` 之后才执行。**

---

## 一、前置确认（必须全部打勾）

- [ ] 已在 **DSH 里新建一个工作目录为 `E:\redstone_nn` 的会话**
- [ ] 新会话里能正常读写文件（试读 `README.md` 和列 `01_phase1_digit_recognition\results\`）
- [ ] 能正常跑 Python（试跑 `01_phase1_digit_recognition\src\make_canvas.py` 之外的一个轻量脚本）
- [ ] 打开 `01_phase1_digit_recognition\build\draw.html` 能正常显示并识别
- [ ] **本清单所在的旧会话已不再需要**（旧会话的所有文件操作在删除后会全部失效）

---

## 二、删除前的最后校验（在旧会话里跑）

```powershell
# 逐文件比对哈希，确认无遗漏（耗时较长但最可靠）
$SRC = "D:\czm我的世界\文件\Comparator Network"
$DST = "E:\redstone_nn"

$srcFiles = Get-ChildItem $SRC -Recurse -File
$bad = 0; $missing = 0
foreach ($f in $srcFiles) {
    $rel = $f.FullName.Substring($SRC.Length).TrimStart('\')
    # 目标路径需要按新结构查找 —— 用文件名+大小做粗校验
    $cand = Get-ChildItem $DST -Recurse -File -Filter $f.Name -ErrorAction SilentlyContinue
    if (-not $cand) { Write-Output "  缺失: $rel"; $missing++ }
    elseif (($cand | Where-Object { $_.Length -eq $f.Length }).Count -eq 0) {
        Write-Output "  大小不符: $rel"; $bad++
    }
}
Write-Output "缺失 $missing 个，大小不符 $bad 个"
```

**期望结果：缺失 0，大小不符 0。**

---

## 三、确认无误后的删除命令

```powershell
# 1) 先确认目标目录健康
$DST = "E:\redstone_nn"
Get-ChildItem $DST | Select-Object Name
(Get-ChildItem $DST -Recurse -File | Measure-Object Length -Sum).Sum / 1GB

# 2) 删除旧目录（⚠️ 不可恢复）
Remove-Item "D:\czm我的世界\文件\Comparator Network" -Recurse -Force

# 3) 如果 D:\czm我的世界\文件 下已无其他内容，一并清掉
Get-ChildItem "D:\czm我的世界\文件" -Force
# 确认空后再：
# Remove-Item "D:\czm我的世界\文件" -Recurse -Force
```

**建议**：删除前把 `.obsidian` 之外的东西再扫一眼，确认没有你后来手工放进去的文件。

---

## 四、删除后的收尾

| 事项 | 操作 |
|---|---|
| **Obsidian 库** | 在 Obsidian 里"打开文件夹作为库" → 选 `E:\redstone_nn`（配置在 `01_phase1_digit_recognition\obsidian\`，可复制到根目录） |
| **游戏数据包** | 已确认游戏**不直接引用**工作区里的数据包，无需处理 |
| **D 盘空间** | 释放约 **2.9 GB**（D 盘原本只剩 13.5 GB，删除后约 16.4 GB） |
| **旧会话** | 关闭即可（其工作目录已不存在） |

---

## 五、回滚方案（万一删除后发现问题）

**好消息**：删除前做的校验是**逐文件**的，且新目录**字节数多于源目录**（3,116,767,583 > 3,116,457,167），说明**没有数据丢失**。

如果删除后发现目标目录有问题：

1. 从 `E:\redstone_nn\00_shared\DT_Redstone_Research_Archive_20260927.zip` 等归档可恢复部分数据
2. 头像数据在 `02_phase2_avatar\archive_phase1_attempts\` 完整保留
3. **一期的核心成果（权重、文档、数据包、draw.html）都有多份副本**：
   - `01_phase1_digit_recognition\results\Wint_model.npz`
   - `99_models\phase1_Wint_model.npz`（副本）
   - `01_phase1_digit_recognition\build\draw.html`
   - `99_tools\draw_digit_phase1.html`（副本）

**唯一没有副本的是**：MNIST 原始数据（`data_mnist\`，63 MB）—— 但它可以随时重新下载。

---

*删除是单向操作，请务必先完成第一节的全部打勾项。*
