# 实验运行与结果说明

## 使用现有环境

已使用本机 Python 3.11.5、PyTorch 2.7.1+cu118、RTX 3050 Laptop GPU 4GB 完成短训练验证，没有安装或升级现有环境的软件包。

本地 CSV 预测入口已将远程数据、M4、UEA 分类和 Reformer 的专属依赖改为按需导入。运行本项目的 DLinear、LSTM、GRU、核心模型与小型 PatchTST，无须先安装完整上游 requirements 中所有大型模型依赖。完整上游任务仍有各自的依赖要求。

## Windows 命令

进入项目根目录：

```powershell
Set-Location -LiteralPath 'D:\develop\deeplearning\electricity_forecasting'
```

四个模型的流程短测（默认 3 个训练 batch、2 个验证/测试 batch、1 个 epoch）：

```powershell
python scripts\run_experiments.py --suite smoke
```

正式基线对比，运行 DLinear、LSTM、GRU、小型 PatchTST、Core，默认 24/96 小时和 2021/2022/2023 三个随机种子，共 30 次训练：

```powershell
python scripts\run_experiments.py --suite baselines --epochs 10
```

正式消融：

```powershell
python scripts\run_experiments.py --suite ablation --epochs 10
```

一次运行全部 10 个变体（含 NoPeriod 等价性对照），默认共 60 次训练：

```powershell
python scripts\run_experiments.py --suite all --epochs 10
```

先只跑核心模型的 96 小时正式训练：

```powershell
python scripts\run_experiments.py --suite baselines --variants Core --horizons 96 --seeds 2021 --epochs 10
```

只查看计划，不开始训练：

```powershell
python scripts\run_experiments.py --suite all --dry-run
```

也可以运行 `scripts\run_coursework.ps1`，参数与 Python 入口相同。CPU 环境追加 `--cpu`。

## 公平设置与资源控制

- 固定完整 321 个用户，训练/验证/测试按时间约 70%/10%/20% 划分。
- 历史窗口 336 小时，预测 24 或 96 小时；默认训练最多 10 epoch，验证集连续 3 epoch 不改善时停止。
- 统一 Adam、MSE、初始学习率 0.001、余弦学习率调整和梯度裁剪阈值 1.0。
- LSTM/GRU：2 层、隐藏维度 64；周期修正和门控隐藏维度均为 32。
- 一次只运行一个训练进程。大部分模型 batch=16，小型 PatchTST 为控制显存最多 batch=2。
- 不同 batch 是资源约束，应该在论文中披露；不要据此声称各模型计算预算完全相同。
- `--seed` 已同时控制 Python、NumPy、PyTorch 和训练数据打乱；跨平台/CUDA 实现不保证位级一致。
- `--suite smoke` 或 `--smoke` 都会启用 batch 限制。结果标记 `smoke_test=true`，不得混入正式论文表格。

## 结果目录

所有输出位于项目根目录的 `outputs/<运行名称>/`，默认名称含北京时间。显式 `--run-name` 不允许覆盖已有目录。

每组运行包含：

- `run_manifest.json`：运行计划、完成状态、相关源码 SHA-256、Python 和软件包版本。
- `summary.csv` / `summary.json`：每个变体、跨度和随机种子的主要指标。
- `aggregate.json`：同变体同跨度的均值与样本标准差；只有一个种子时标准差为 null。
- 每次实验的 `.log`：完整控制台日志。
- 每个实验独立的 checkpoint 和结果目录，短文件名降低 Windows 路径长度问题。

每次训练的结果包含：

- `config.json`、`history.json`、`training.json`：完整参数、训练/验证损失、最优 epoch、训练耗时和 GPU 张量峰值显存。
- `metrics.json`：MAE、MSE、RMSE，以及原 CSV 尺度的 `raw_*` 指标；另有参数量、测试窗口数、数据 SHA-256 和短测标记。
- `metrics.npy`：本课程入口的 `[MAE, MSE, RMSE]`，不是上游五指标格式。
- `per_horizon.csv`：逐预测步的标准化 MAE/MSE。
- `forecast_example.npz`：第一个测试窗口的历史、真实值、预测值，均为训练集标准化后的尺度。
- `gate_weights_mean.csv`：核心模型及周期消融版本在测试窗口上的每用户平均专家权重。

默认逐 batch 累计误差，只保留一个预测样例；不保存所有高度重叠的完整预测数组，以控制内存和磁盘占用。显存统计是 PyTorch 分配的张量峰值，不包含全部系统显示、驱动和 CUDA 上下文开销。

## 直接使用框架

新增模型也兼容原框架的 `--model LSTM`、`--model GRU` 和 `--model PeriodGatedDLinear`。课程入口在调用原 `run.py` 时加上 `--coursework`，切换到本项目的训练与流式评价逻辑；不加这个开关时仍使用上游实验类。

## 检查

```powershell
python tests\test_forecasting_models.py -v
```

检查覆盖 24/96 步输出、192 步周期延拓对齐、未来 decoder 隔离、门控概率与输入依赖、用户顺序等变性、分支梯度、历史窗口约束、训练段标准化，以及流式评价中的最后一个不足整 batch 的样本和原尺度还原。
