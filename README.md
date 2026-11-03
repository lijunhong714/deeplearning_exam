# 融合日周周期与输入相关门控的电力负荷预测

项目地址：[lijunhong714/deeplearning_exam](https://github.com/lijunhong714/deeplearning_exam)

在 [Time-Series-Library](https://github.com/thuml/Time-Series-Library) 基础上实现经典LSTM、GRU及PeriodGatedDLinear（Core）。Core结合DLinear、历史日/周周期模板及输入相关门控。论文比较DLinear、LSTM、GRU和Core，并进行三项模块消融。

## 环境

论文实验使用Python 3.11.5、PyTorch 2.7.1+cu118、RTX 3050 Laptop GPU（4 GiB）。根目录 requirements.txt 固定课程实验入口依赖，上游 requirements 面向完整框架的其他任务。

```powershell
git clone git@github.com:lijunhong714/deeplearning_exam.git
cd deeplearning_exam
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
```

CPU环境可使用CPU版PyTorch，并在实验命令追加 --cpu。其他平台/CUDA实现不保证逐位相同。

## 数据准备

```powershell
python scripts\prepare_electricity.py
```

下载至 Time-Series-Library/dataset/electricity/electricity.csv。需要代理时追加 --proxy http://127.0.0.1:7892，脚本不修改系统代理。文件已存在时只重新校验。

- 发布源：[THUML Electricity](https://huggingface.co/datasets/thuml/Time-Series-Library/blob/main/electricity/electricity.csv)。
- 原始来源：[UCI ElectricityLoadDiagrams20112014](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014)，数据许可CC BY 4.0。
- 26,304个连续小时、321个用户变量，95,581,762字节。
- SHA-256：7e45845d54c5219bad0ae6bc1b5316cf8ff9cead5d33fa998a5a51c2e4a497ad。
- 发布CSV时间索引为2016-07-01 02:00至2019-07-02 01:00，与原始采集年份说明不同。保留原文件，仅使用24/168步相对周期，不匹配真实节假日或天气。详见 data_manifest.json。

## 复现论文

在项目根目录运行。先做流程短测，短测不用于论文：

```powershell
python scripts\run_experiments.py --suite smoke
```

论文24次对比运行：

```powershell
python scripts\run_experiments.py --suite baselines --variants DLinear LSTM GRU Core --history 336 --horizons 24 96 --seeds 2021 2022 2023 --epochs 20 --run-name paper_main
```

18次新增消融，Core复用前一组：

```powershell
python scripts\run_experiments.py --suite ablation --variants Weekly Daily Static --history 336 --horizons 24 96 --seeds 2021 2022 2023 --epochs 20 --run-name paper_ablation_three
```

Weekly保留周分支、去除日分支；Daily保留日分支、去除周分支；Static保留三个分支，使用全局可学习权重。运行名称不可覆盖已有目录；--dry-run 仅打印计划。为保持原实验入口及源码指纹，其预设 suite 保持不变，论文复现以这里显式给出的 --variants 为准。

统一按时间顺序约70%/10%/20%划分，历史336、预测24/96、批大小16、Adam初始学习率0.001、余弦调整、梯度裁剪1，最多20轮、早停耐心值3。标准化仅拟合训练段，验证选择检查点，测试包含最后一个不足整批的批次。

## 已记录结果

experiment_records/ 提供论文所用42次运行的指标、配置、逐轮损失与逐预测步误差。原始运行计划和源码指纹保存在各组 run_manifest.json；原始组可能包含额外任务，公开结果按 selection.json 的论文配置筛选。保留的原始指标文件字节不变。

```powershell
python paper\build_results.py
```

生成 paper/tables/ 与 paper/results_summary.json，优先读取本地 outputs/，没有本地运行时读取 experiment_records/。MAE/RMSE在训练段逐用户标准化后的空间计算，跨三个种子报告均值与样本标准差。时间包含训练、验证和检查点保存；显存为PyTorch张量分配峰值MiB。

在所选基线范围内，Core相对DLinear的MAE降低6.32%（24小时）和5.10%（96小时）。Core训练时间高于三个基线。结论限于本文所列比较范围。

## 检查和论文

```powershell
python tests\test_forecasting_models.py -v
```

检查输出形状、未来信息隔离、周期对齐、门控输入依赖、梯度、训练段标准化及完整流式评价。论文源码位于 paper/，使用XeLaTeX和BibTeX编译，公式和结构图均为原生源码。署名在本地 paper/author.tex 设置，公开版本使用 author.example.tex。详见[论文说明](paper/README.md)。

## 来源与许可

Time-Series-Library/ 为上游提交 4e938a1767106324dd753b2a44832bf870a0252e 的源码快照及本项目扩展，以普通目录提供，避免嵌套Git仓库导致克隆后缺失代码。保留上游MIT许可证，修改说明见 [THIRD_PARTY.md](THIRD_PARTY.md)。本项目从头训练，不使用预训练参数，属于对已有方法的组合扩展。