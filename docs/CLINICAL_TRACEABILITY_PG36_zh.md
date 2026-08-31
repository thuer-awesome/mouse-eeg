# PG36 临床波形溯源与对齐说明

## 结论先行

PG36 的正式观察单位是“一个临床事件 × 一个有效独立通道”，不是单个 4 秒 epoch。事件内的 4 秒物理滑窗只提供事件形态的时间支持；事件标签由同一事件的聚合特征和冻结的 pooled taxonomy 继承。因此，单个 4 秒窗不能被写成“该发作的典型临床波形”。

本次新增的 traceability renderer 做了三件事：

1. 把一个事件内的全部物理滑窗按真实时间顺序保留；
2. 把每个事件回溯到连续的 0.5–70 Hz 带通、50 Hz 陷波 EEG，并显示人工事件边界；
3. 把代表窗从“事件首窗”改为离该事件特征中位数最近的真实窗，并记录精确缓存位置。

## 两个层次必须分开

### 分析/分类层

- 一行事件表 = 一个事件 × 一个有效通道。
- CH7、CH8 是两个独立观察，绝不拼接为双通道输入。
- 24 维形态特征（12 个原始特征 + 12 个同动物/通道训练背景 z 值）先在事件内对物理窗取中位数。
- 每只鼠把两个有效通道的事件行放在同一个 pooled GMM 中拟合；PG36 是跨鼠约束合并后的探索性形态标签。
- 同一事件内的所有窗口继承该事件的 PG 标签；窗口本身不是重新拟合的临床亚型。

### 临床显示层

- 短事件：显示事件前后 10 s 的连续波形和事件内所有 4 s 窗。
- 长事件（当前规则为 >120 s）：显示完整事件窗口地图，并把原始波形拆成 onset、sustained、offset 三个临床可读放大区。
- 右侧紫色曲线是“窗口到本事件特征中位数的 robust 距离”，不是 Task-1 的 seizure probability。
- 橙色边框是代表窗，选择规则为 `nearest_event_feature_median_robust_MAD`，并在源表记录 `cache_dir`、`cache_local_index`、起止时间和窗口序号。

## 当前对齐审计

旧 PG36 冻结表和当前严格 Task-1 protocol 不是同一个发布版本：

- PG36：9,144 个事件×通道、194,530 个物理窗；其 train/val/test 是形态学表自己的 6,393/1,375/1,376 划分；
- 严格 Task-1 protocol：6,412 个事件×通道；
- 测试集 PG36：1,376 个事件×通道、32,583 个物理窗；
- 测试集完全相同 `event_id + channel + split` 的键只有 111 个；
- 642 个键在 protocol 中存在但 split 不一致；
- 591 个键在当前 protocol 中完全不存在；
- 测试集中共有 68 个事件超过 120 s（其中 32 个同时缺少严格 protocol 键、30 个同时 split 不一致、6 个仅因超长触发复核）；全量 9,144 行中共有 451 个超过 120 s。

因此，当前 PG36 分类结果只能作为形态探索和标签溯源审计，不能与严格 Task-1 测试指标混写。此前“任务二继承任务一严格 split”的说法与现有表格不一致，已由本次键审计纠正。重新训练 PG36 前，必须从同一份严格 protocol 重新生成事件特征、鼠内 GMM、跨鼠 PG 合并和分类 split；对齐门控未通过的行进入 quarantine/unknown，而不是静默删除或改写标签。

## 图集目录

`07_clinical_traceability_pg36_v1/` 是当前唯一新增的临床溯源目录：

- `pdf_by_mouse/`：测试集按鼠分开的连续事件 PDF；每页一个事件×通道观察。
- `preview/first_event_traceability.*`：用于快速检查版式和坐标的 SVG/PDF/PNG/TIFF。
- `source_data/event_traceability.parquet`：事件级标签、边界、代表窗和 QC 状态。
- `source_data/window_traceability.parquet`：事件内全部物理窗口、窗口序号、PG 标签、缓存位置和窗口 QC 距离。
- `source_data/label_qc_audit.csv`：超长事件、重叠链、缺少物理窗和对齐问题。
- `source_data/protocol_alignment_summary.csv`：PG36 与严格 protocol 按鼠/通道的键交集。
- `all_events_index/`：train/val/test 全部 9,144 个事件和 194,530 个窗口的索引版，不渲染成数千页 PDF。

## 任务一和任务二不可混用

任务一是二分类连续检测：模型输出每个独立通道 4 s 窗口的 `P(seizure)`，再由 interval-aware decoder 做滞回、持续性确认和物理窗口并集，得到连续预测事件。任务二 PG36 是事件形态探索分类，输出 PG 标签和 GMM/原型距离；它不提供任务一的 seizure probability，也不能修正任务一的起止边界。

任何临床图必须在标题/图注中写明使用的是哪一条线，避免把 PG assignment probability 当成 seizure probability。

## 新数据接入门控

新数据进入训练或 PG36 前按以下顺序执行：

1. 统一 250 Hz、0.5–70 Hz 零相位带通、50 Hz 陷波和通道 QC；
2. 将每个有效通道展开为独立流，保留稳定的 `animal/session/channel/event/window` 键；
3. 先完成标签坐标和事件组审计，再按独立事件/背景块划分 70/15/15，并执行跨 split purge；
4. 检查 `event_id + channel + split` 的一对一键一致性；
5. 只有通过门控的事件进入训练和正式指标；其余保留在 quarantine 表供临床复核。

这套顺序优先保证可复现和可追溯，避免通过调阈值、换代表窗或删掉难例来制造虚假的性能提升。
