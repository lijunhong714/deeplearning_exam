# 电力负荷预测课程论文

正文采用提供的 ctexart + spconf 中文双栏模板。公式使用原生 LaTeX，模型结构图使用 TikZ，结果表由正式运行记录生成。原始模板保持不变。

代码与复现说明：https://github.com/lijunhong714/deeplearning_exam

## 编译

Windows、TeX Live，字体使用宋体与 Times New Roman。在本目录运行：

```powershell
xelatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
xelatex -interaction=nonstopmode -halt-on-error main.tex
xelatex -interaction=nonstopmode -halt-on-error main.tex
```

也可运行 build.ps1。本地 author.tex 存在时采用本地署名，否则使用 author.example.tex；公开源码不包含个人署名文件。源码已含结果表，可直接编译。

## 结果口径

- 对比：DLinear、LSTM、GRU、Core，两个预测长度、三个种子，共24次。
- 消融：Weekly（去除日分支）、Daily（去除周分支）、Static（替换为静态可学习门控），共18次。
- 本文纳入42次正式运行，完整模型在消融表中复用对比结果。历史336，预测24/96，种子2021/2022/2023，最多20轮、早停耐心值3。
- build_results.py 优先读取原项目 outputs/，不存在时读取公开仓库 experiment_records/。按论文配置筛选并核对42次运行、完整测试窗口数、无调试限批、逐步指标与总指标一致，再生成三张结果表和 results_summary.json。
- 原项目的其他实验记录保持原样；表格生成程序仅选取本文声明的配置。
- RMSE先在每次运行内计算，再跨种子汇总；标准差使用样本标准差。
- 实验为从头训练，不依赖预训练参数。

## 论证范围

Core在所选三个基线范围内平均误差最低，相比DLinear的MAE降低6.32%/5.10%；结论不扩展到未纳入比较的模型。周分支贡献更明显，日分支收益较小。Core训练时间高于三个基线。

## 文献

references.bib 包含23条正文实际引用的来源，其中20条为学术论文，另有数据与代码来源。BibTeX按首次引用顺序生成参考文献。