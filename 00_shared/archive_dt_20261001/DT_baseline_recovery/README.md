# DT 原始基线恢复包与实施核查

2026-10-02。本包解决原始 Teacher 缺失，并保留本轮查明的源码证据。没有重造 Teacher，也没有把新训练结果写入旧曲线。

## Aitch老师现在怎样做

1. 解压本包，保留整个目录结构。`teachers` 就是找回的四张原始 `.npy` 和四份参数 JSON；不必再检查电脑才能继续。
2. 在本包目录运行 `python verify_teachers.py`。需要 numpy。四项应均显示 `JSON mismatches=0` 与 `SHA256 OK`。输入第4维最快；禁止自行改排列。
3. 需要复核旧基线时运行 `python replay_v05_baseline.py`，需要 numpy、pandas。预期 MLP T1 / K4 / h12 为5257错；本轮已实跑达到。此命令只复核一个已明确对应原始启动参数的配置。
4. 原始 v03/v04/v05/v07/v19 ZIP 在 `original_archives`，保持原样。原Teacher的出处是 v05 ZIP 的 `dt_duel_v05/teachers/`。后续每次交付都携带原标签和 manifest，勿只留seed。
5. 下一轮v23正式运行示例（本轮未执行全部7 restart）：

```bash
python src/v23_multirestart_trainer.py --teachers teachers --out results_v23 --restarts 7
```

numpy、pandas必须在所用Python中可导入。本包所有输入都来自本包；v23已修复包外DTF004依赖和 `/mnt/data` 固定路径。程序还会核对Teacher标签SHA-256；标签不符时拒绝训练。

如果日后又缺失，不要只搜单个.npy：优先找 `DT_MLP_scaling_v05.zip`，其次找名字含v05、v18、v19、v20、v21的旧包，查看压缩包内部 `teachers/`、`DT_v18_archive/` 和嵌套ZIP。原始数组本轮已全部取回，因此当前不需要全盘扫描或上传电脑其他文件。v19包本身不含这些标签，不能因它叫“完整实验包”就认定依赖齐全。

## 本包包含和未包含的内容

- 四张原Teacher、JSON、原始H表顺序、完整SHA-256记录。
- v05单配置复跑结果 `baseline_replay.json`。
- v23八个原始精确深度2池、完整DTF004依赖、便携训练器。
- v23局部直方图与逐状态评价等价核验，以及真实Teacher前向核验；具体范围在 `v23_portability_check.json`。
- 修订的三份报告、逐条审计回应。

v23运行输出额外保存每次`.npz`模型，包括每条边在函数池中的索引、通道二值读出索引、最终读出索引、seed、passes及函数池hash。二值读出排序按v23代码重建；它和原v03的H首次遇见顺序不同，禁止交换head_id。

**未包含**：原始12313项DDD采样池、`v18_strong_seq.py`、v07产出4999/346的精确启动记录、全部历史最佳模型、完整比较器witness导出或Minecraft实物网络。这些仍应逐项恢复；新建替代物时须标成新实现/新实验。

## 便携修复范围

`src/legacy_primitives.py`从v03抽出不依赖PyTorch的三个原始前向函数；`count_func_cd_v03.py`仅修改该导入依赖，避免原共用模块在PyTorch未安装时类定义报错。v05 `count_general.py`保留训练实现；历史路径不会在本包中产生有效依赖，Python从本包src解析模块。v23改为按自身文件位置定位函数池与manifest，补依赖、标签校验和模型保存，训练目标、坐标扫描与权重更新保持不变。

本包原始档案、便携修复和新的核验日志三类分别标明，不把修复后的源码冒充历史逐字原件。
