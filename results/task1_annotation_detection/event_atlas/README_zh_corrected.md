# 修正后的临床事件图集

本图集沿用旧 clinical event atlas 的三层版式：原始波形、概率轨迹、decoded state。
模型阳性窗口按真实物理区间 `[window_start, window_end)`（4 秒）绘制并做区间并集，不再把窗口中心压缩为 1 秒事件。

- 蓝色半透明：人工事件区间。
- 橙色边框：模型预测区间。
- 橙色状态带：interval-aware decoder 的完整窗口支持。
- label_external_prediction 仅表示与当前人工标签零时间重叠，仍需临床复核，不等同于已确认误报。
- 旧目录保持只读；本目录是 corrected v1 的唯一新图集。
