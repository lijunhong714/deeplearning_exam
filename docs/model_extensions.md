# 可补充的结构与接入方式

以下基于上游提交 `4e938a1767106324dd753b2a44832bf870a0252e`。本项目现已实现 LSTM、GRU 和 PeriodGatedDLinear；其余结构仍为扩展建议。正式预测性能尚需完整实验验证。

已实现模型公式见 [model_design.md](model_design.md)，运行方式见 [experiment_guide.md](experiment_guide.md)。

## 已有结构

框架已包含 DLinear、PatchTST、iTransformer、TimeMixer、TimesNet、TSMixer、SegRNN 等，可直接作为对照候选。`SegRNN` 内部使用 GRU，但其分段编码结构不同于常规 LSTM/GRU 基线。

`layers/StandardNorm.py` 已提供可逆实例归一化组件，多个现有模型也包含自己的归一化逻辑。复用归一化或已有多尺度模块时，应明确来源，不将其作为原创结构。

## 适合 4GB 显存的候选

| 结构 | 实现思路 | 作用与实验 |
| --- | --- | --- |
| 常规 LSTM / GRU | 1—2 层，隐藏维度 32—64，多步输出头 | 补充经典循环网络对照 |
| 轻量 TCN | 因果扩张卷积、2—3 个残差块、小通道数 | 建模局部波动；可比较纯 TCN 与 DLinear + TCN 残差 |
| 日/周周期分支 | 只从历史中提取最近 24/168 步模板或多周期统计，映射到预测跨度 | 比较无周期、仅日周期、仅周周期、双周期 |
| 输入相关门控 | 从历史序列统计或小型 MLP 产生融合权重 | 自适应融合 DLinear、日周期与周周期；比较固定权重与动态门控 |
| 周期位置编码 | 由相对序列索引生成 24/168 步正余弦编码 | 提供相位信息；本数据索引未核对真实日历，不能直接解释为真实节假日特征 |
| 低秩变量混合 | 对归一化后的变量做 C→r→C 映射，r 可从 8/16 起步 | 尝试跨用户信息共享；通过消融检查弱相关变量是否反而引入噪声 |

当前核心模型已选定“DLinear + 日/周历史分支 + 输入相关门控”。TCN、周期位置编码和低秩变量混合可以留作后续独立研究，避免首次实验堆叠过多模块。

DLinear 本身已有移动平均趋势/季节分解，不能将这部分视为新增贡献。若只叠加多个线性映射并固定加权，整体仍可能等价于一个线性映射；输入相关门控或非线性残差是否有效，需要消融与对比实验验证。

## 周期分支的数据边界

预测 96 步时，某些未来时刻的“前 24 步真实值”也处在预测区间内，不能作为输入。可使用历史同相位均值、历史模板周期延拓，或完全由观测历史生成的预测分支。所有改进只依赖预测起点之前的观测，以及在预测时确实可获得的时间索引。

## 新模型接入

本版本的 `exp/exp_basic.py` 会扫描 `models/` 中的 Python 模块并延迟加载，不需要逐个手动修改模型注册表。

例如创建 `Time-Series-Library/models/PeriodGatedDLinear.py`，需要满足以下接口约定：

```python
class Model(nn.Module):
    def __init__(self, configs):
        ...

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        ...
        return prediction
```

- 构造函数接收 `configs`，可读取 `seq_len`、`pred_len`、`enc_in` 等参数。
- 历史输入 `x_enc` 为 `[batch, seq_len, channels]`。
- 多变量预测 `features=M` 时返回 `[batch, pred_len, channels]`。
- 启动参数使用 `--model PeriodGatedDLinear`。
- 从 `Time-Series-Library` 根目录启动，因为当前模块扫描使用相对 `models` 路径。
- 新增自定义超参数时，同时在 `run.py` 中补充参数解析。
- 24 小时预测、96 小时预测以及各消融版本使用独立结果标识，避免覆盖输出。

如需输出预测区间或修改损失函数，除模型外还需调整 `exp/exp_long_term_forecasting.py` 的训练、评价和结果保存逻辑，工作量高于单纯新增模型模块。
