# 任务二：全鼠全有效通道形态学与类别冻结（正式审计版）

## 结论

本目录保存的是旧批次与新批次合并后的**探索性形态类别**。每个有效 EEG 通道作为独立样本流处理；类别先在每只鼠/每个通道内发现，再用保守的跨鼠原型图合并并冻结。当前冻结结果为 **45 个 G 类别**，不是二分类，也不是把历史 21 类直接当作正式标签。G 类别的急性、非急性或具体发作类型仍需临床复核后命名。

## 数据与质控

| 审计项 | 数值/规则 | 说明 |
|---|---:|---|
| 源形态窗口行数 | 114,219 | 合并特征表中的原始 4 s 窗口行 |
| 源临床事件 | 6,225 | 事件级审计分母，未先按通道翻倍 |
| 严格纳入窗口 | 194,530 | CH7/CH8 展开后，且 `event_overlap_fraction >= 0.99` |
| 来源批次（严格窗口） | old 58,686；combined_protocol 135,844 | 两批均经过同一特征定义后合并 |
| 事件×通道观察 | 9,144 | 一个事件在每个有效通道各有一行形态中位数标签 |
| 无完整 4 s 窗口的事件 | 1,381 | 从正式形态标签/分类中排除，但保留审计，不是模型漏检 |
| 动物 | E07、E50、E76、E77、E79、E81、E82 | 7 只；不是跨动物留一验证 |
| 通道规则 | CH7、CH8 独立展开 | E77/CH7 无效并排除；E07 当前只有 CH8 有效记录 |
| 采样与滤波 | 250 Hz；0.5–70 Hz 零相位带通；50 Hz 陷波 | 绝对 ADC→µV 未确认，幅值只作相对/a.u. |

`run_audit.json`、`category_registry.csv` 和原始 Parquet 是上述数字的依据。CH7/CH8 不做跨通道拼接，也不把动物或通道 ID 作为分类器输入。

合并特征源表为 `sleep_staging/output/epilepsy_program_20260821/02_morphology_taxonomy/all_mice_old_plus_new/tables/combined_morphology_features_relative.parquet`；本目录保存的是该表经过严格事件/通道展开与标签冻结后的可复核快照。

## 划分与泄漏审计

`01_tables/event_splits.parquet` 按**临床事件**随机划分，动物在三个 split 中混合出现；同一事件的所有窗口及其有效通道保持在同一 split。源事件计数为 train/val/test = **4,357/934/934**（约 70/15/15）。完整窗口筛选后，事件×通道计数为 **6,393/1,375/1,376**，窗口计数为 **130,216/31,731/32,583**。

对去重后的 4,844 个有完整窗口的独立事件按 `animal_id + session_id + event_start_ms/end_ms` 检查，跨 split 的事件时间重叠对为 **0**；因此本批次没有需要额外删除的跨 split 原始窗口。分类器的 21-epoch 上下文只从同一事件、同一通道取邻近窗口，不足部分零填充，不会借用另一事件或另一 split 的波形。

## 形态类别冻结

- 局部模型只在 train 事件×通道中拟合；当前实际得到 **55 个局部原型**（以 `02_per_mouse/local_category_prototypes.parquet` 为准）。
- 跨鼠合并使用 RobustScaler 后的跨动物原型距离图，选定距离阈值 **0.90**；连通分量中若出现同一鼠/通道冲突则保留为独立探索类别。
- 冻结后为 **45 个全局类别（G01–G45）**；其中 G04、G12、G26 只有一只鼠的支持，标记为 `mouse_specific_flag=true`，应优先临床复核。
- `category_registry.csv` 的事件×通道支持范围为 **33–1,159**；全 45 类在正式测试窗口中均出现，但窗口支持极不均衡（test 最小 11、最大 9,004）。
- 低置信度仅作 QC 标记，不静默删除：事件×通道层 44 行，窗口层 349 行。

类别特征由相对幅度/稳健 RMS、线长、Hjorth、峰频、谱熵、delta/theta/beta/低高 gamma 相对功率及其基线 z 分数组成；这保证了跨记录幅值标定未知时仍可比较，但也意味着类别是**表示空间中的探索性簇**，不能直接宣称为临床亚型。

## 正式标签文件（不要混用）

1. `04_label_freeze/frozen_channel_window_labels.parquet`：**正式分类输入**，194,530 个独立通道窗口，含 `global_category/global_label`、split、时间和特征。
2. `04_label_freeze/frozen_event_labels.parquet`：9,144 个事件×通道的全局标签摘要。
3. `04_label_freeze/frozen_event_labels_with_representatives.parquet`：同为 9,144 行，同时保留局部簇和代表性字段。
4. `04_label_freeze/frozen_window_labels.parquet`：历史遗留的二类别文件（G01/G02，9,144 行），仅作旧敏感性结果追溯，**不是本任务的 45 类标签**。
5. `04_label_freeze/historical_21_read_only.parquet`：历史 21 类探索结果，只读，不参与正式 45 类训练。

## 图形位置

- `06_figures/legacy_style_morphology/`：沿用项目既有 renderer 的逐鼠、逐通道波形/PSD/STFT/事件时间线图。
- `06_figures/legacy_style_cross_mouse/`：跨鼠类别支持、原型和类别存在性图。
- `06_figures/figure_1_balanced_pca.*`、`figure_2_balanced_umap.*`、`figure_3_category_support.*`：类别冻结前后的总体结构图。
- `06_figures/historical_21_read_only/`：历史 21 类原图，便于追溯，不代表当前标签定义。
- `06_figures/per_mouse_channel_atlas/`：临时审计 atlas，不作为主文图形入口。

快速抽查入口：`legacy_style_morphology/mouse_E50_CH7_morphology_atlas.png`、`mouse_E50_CH8_morphology_atlas.png`；跨鼠入口为 `legacy_style_cross_mouse/figure_1_per_mouse_channel_clusters.png` 和 `figure_2_cross_mouse_category_map.png`。每张图的 PDF/SVG/PNG 三种版本同名配套。

下一步临床工作是给 G01–G45 逐类附上急性/非急性/发作类型解释；在此之前不得把这些探索性名称写成诊断结论。
