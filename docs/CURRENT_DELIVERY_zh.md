# 当前交付索引（任务一 + 任务二）

本文档是本项目的单一交接入口。它只指向当前审计后的两条主线；旧的 21 类、45 类、旧 decoder 和 smoke 目录只保留作取证，不应作为当前结果入口。

## 任务一：二分类连续打标

**当前结果根目录**

`E:\自动睡眠分析\sleep_staging\output\epilepsy_program_20260821\01_annotation_detection\06_decoder_audit_v2\corrected_combined_old_plus_new\`

**输入协议与权重**

- 严格协议：`..\03_combined_old_plus_new\protocol\`；`channel_events.parquet` 共有 6,412 个事件×有效通道观察，按 `train/val/test = 4,495/960/957` 划分。
- 三个冻结权重：`..\04_model_runs\combined_old_plus_new\<model>\model.pt`。
- corrected 根目录的 `models\<model>\decoder_fit.json` 是 validation-only 连续解码参数，不是模型权重。

**临床图和模型图**

- `figures\model_performance\`：三模型窗口级 PR/ROC、训练/验证诊断、与参考样式一致的窗口混淆矩阵，以及事件级审计汇总。数值源在同目录 `source_data\`。
- `figures\clinical_event_atlas\`：按覆盖率、完全漏检、长事件漏检、边界偏移和标签外候选分组的连续波形图。每页显示人工边界、模型概率、decoded state 和物理 4 秒窗口。
- `models\<model>\events_corrected_v1_test\`：每个模型的 `decoded_tracks.parquet`、`predicted_events.parquet`、`event_matches.parquet` 和事件指标。

**当前测试结果（953 个完全可观测事件）**

| 模型 | 事件灵敏度 | 事件精确率 | 事件 F1 | 零覆盖率 | 误报候选/通道小时 |
|---|---:|---:|---:|---:|---:|
| EEGWaveNet-SC | 0.782 | 0.297 | 0.430 | 19.94% | 17.21 |
| Physio-Sequence | 0.894 | 0.215 | 0.347 | 6.40% | 30.27 |
| Physio + statistics | 0.885 | 0.241 | 0.378 | 6.19% | 25.91 |

957 是图集中的全部测试人工事件分母；953 是形式 interval-aware 指标分母，4 条形式审计排除记录仍保留在图集和源表中。事件级区间没有自然 TN，因此不能用普通二分类混淆矩阵替代事件审计。`P(seizure)` 是模型排序分数，未做临床概率校准；连续事件必须经过 validation-only decoder 才能得到起止点。

当前结论是：Physio 两个版本漏检较少，EEGWaveNet-SC 精确率和误报率较好；三者都仍需医生复核，不能宣称无人值守临床部署。零相位滤波和居中 21 窗上下文使当前入口属于离线/回放，不是严格因果实时系统。

## 任务二：PG36 形态学探索分类

**当前结果根目录**

`E:\自动睡眠分析\sleep_staging\output\epilepsy_program_20260821\02_morphology_taxonomy\all_mice_old_plus_new\strict_all_channel_v1\`

**唯一正式子目录**

- `01_tables\`：事件×通道特征和物理窗口表。
- `02_per_mouse\`：每只鼠、每个独立通道的局部聚类结果。
- `03_cross_mouse\`：逐鼠 pooled 原型、跨鼠 PCA/UMAP 和支持矩阵。
- `04_label_freeze_pooled\`：冻结的 `PG01–PG36` 标签、类别注册表和审计。
- `05_classification_pooled\`：五个模型的 checkpoint、全测试窗口预测、指标和逐类表。
- `06_figures\paper_ready_morphology_pooled_v2\`：当前 36 类形态图；包括逐鼠独立通道 PCA、逐鼠 PG 波形、严格 36 点全局质心 PCA、支持矩阵和全部类别图。
- `06_figures\paper_ready_classifier_pooled\`：主模型比较、训练曲线、最佳模型混淆矩阵和 split 审计图。
- `06_figures\strict_classifier_full_pooled\`：五个模型各自完整 PG36×PG36 混淆矩阵、全部 36 类 PR/ROC 和支持-召回图。
- `07_clinical_traceability_pg36_v1\`：事件回溯图集（见下一节）。

**标签和观察单位**

- 7 只鼠，E77/CH7 排除；CH7、CH8 始终是独立输入和独立观察，不拼接。
- 9,144 个事件×通道观察、194,530 个物理窗口、40 个鼠内 pooled 原型，跨鼠约束合并后为 36 个全局 PG 类。
- 每只鼠只显示它实际出现的 PG 子集；36 是所有小鼠的类别并集，不是每只鼠都应有 36 类。全局质心图严格为 36 个点。

**五模型全窗口快照（该快照自身测试 split，n=32,583）**

| 模型 | Accuracy | Balanced accuracy | Macro-F1 | Macro-AUPRC | 事件 Macro-F1 |
|---|---:|---:|---:|---:|---:|
| Physio + statistics | 0.605 | 0.593 | 0.465 | 0.513 | 0.625 |
| Physio-TF | 0.522 | 0.582 | 0.434 | 0.497 | 0.647 |
| EEGConformer | 0.266 | 0.365 | 0.207 | 0.279 | 0.409 |
| EEGNetv4 | 0.229 | 0.328 | 0.188 | 0.246 | 0.369 |
| EEGWaveNet-SC | 0.219 | 0.260 | 0.154 | 0.221 | 0.247 |

**重要对齐门控**：PG36 冻结表自己的事件 split 是 `6,393/1,375/1,376`，而任务一严格 protocol 是 `4,495/960/957`。在 PG36 测试集中只有 111/1,376 条 `event_id + channel + split` 完全一致，642 条 split 不一致，591 条键缺失。因此上表是“形态探索快照”，不是任务一严格测试集上的正式泛化性能。重新训练 PG36 前必须从同一 canonical protocol 重建事件特征、鼠内聚类、跨鼠合并和分类 split。

## 临床波形溯源图集

`07_clinical_traceability_pg36_v1\` 是当前唯一的 PG36 临床回溯目录：

- `pdf_by_mouse\`：全部 1,376 个测试事件×通道页面（E07/41、E50/907、E76/141、E77/25、E79/90、E81/70、E82/102）。
- `review_ready\pdf_by_mouse\`：通过当前键、split、缓存和时长门控的 105 页，供临床快速复核。
- `source_data\event_traceability.parquet`：当前测试图集的 1,376 个事件×通道索引、PG 标签、代表窗和 QC 状态。
- `source_data\window_traceability.parquet`：当前测试图集的 32,583 个物理窗及其准确时间、窗口序号、缓存位置和事件中位数距离。
- `all_events_index\source_data\`：全 train/val/test 的 9,144 个事件和 194,530 个窗口索引版，不分页生成数千页 PDF。

代表窗不再取事件首窗，而是从事件内全部物理窗中选择距离该事件特征中位数最近的真实窗；选择规则和 `cache_dir/cache_local_index/start/end` 都写入源表。单个 4 秒窗不能代表一个长临床事件，长事件图会显示完整窗口地图并提供 onset/sustained/offset 连续波形放大。

## 复现和防发散规则

1. 先读各结果根的 `manifest.json`、`README_zh.md` 和源表，再读图；不要依据文件夹名称猜类别或分母。
2. 任何新数据先统一 250 Hz、0.5–70 Hz 零相位带通、50 Hz 陷波、坏道/QC、事件坐标和独立通道键；对齐门控失败的行进入 quarantine，不静默删除。
3. 任务一训练/解码与任务二 PG36 分类必须使用不同输出根；不使用任务二标签修正任务一的起止边界。
4. 训练结果只使用全窗口测试表作为主分母；“每事件最多 8 个窗口”只能作为敏感性分析。
5. 当前任务二的旧 `paper_ready_morphology`、`paper_ready_classifier`、`publication_*`、`legacy_style_*`、`historical_21_read_only` 不属于正式入口。

完整模型结构、预处理、划分和 CLI 说明见 `docs/PIPELINE_SPEC_TASK1_TASK2_zh.md`、`docs/REPRODUCIBILITY_zh.md`、`docs/TASK1_ANNOTATION_DETECTION_zh.md` 和 `docs/TASK2_MORPHOLOGY_PG36_zh.md`。
