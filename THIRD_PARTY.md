# 上游来源与修改

本项目基于THUML的[Time-Series-Library](https://github.com/thuml/Time-Series-Library)，基准提交 `4e938a1767106324dd753b2a44832bf870a0252e`。源码快照保留完整上游目录及原MIT许可证，版权归原作者所有。

本项目新增标准LSTM/GRU（含共享RecurrentForecast层）、PeriodGatedDLinear、课程训练引擎、数据准备、实验入口、检查与论文。修改 `run.py` 以支持随机种子、课程模型参数及短路径输出；修改数据工厂以仅打乱训练集；修改数据加载和SelfAttention模块以按需导入可选依赖。

DLinear来自Ailing Zeng等人的“Are Transformers Effective for Time Series Forecasting?”（AAAI 2023），其实现沿用上游框架。日周模板和门控为本项目组合扩展，不代表DLinear原论文的结构。

Electricity数据独立遵循CC BY 4.0，源文件由下载脚本取得，并保留来源和SHA-256校验；软件MIT许可不替代数据许可。
