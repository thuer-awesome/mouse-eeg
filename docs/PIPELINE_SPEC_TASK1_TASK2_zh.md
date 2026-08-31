# 任务一与任务二完整流程规格

这份规格是当前合并旧数据与新批次后的唯一解释入口。旧 run 目录仍保留作审计快照，但不作为当前流程的默认输入。

## 1. 数据与共同预处理

原始数据由合并缓存协议管理，协议表包含 `full_index.parquet`、`channel_events.parquet` 和训练背景基线。每条有效动物-通道流独立展开；不把 CH7/CH8 拼接，也不把动物号或通道号作为模型输入。E07/CH7 与 E77/CH7 按已审计坏道规则屏蔽，CH8 保留。幅值 ADC→微伏尚未完成绝对标定，所以所有幅值只解释为相对训练背景或 a.u.。

信号处理为 250 Hz、四阶 0.5–70 Hz 零相位带通和 50 Hz 陷波（Q=30）。协议构建阶段检查有限值、连续性、平线/饱和/硬件轨迹和跨 split 原始窗口重叠。训练背景拟合每条动物-通道流的 median 与 `1.4826×MAD`，验证集和测试集不参与基线拟合。

## 2. 任务一：二分类连续检测与打标

### 标签和划分

- 事件内重叠至少 2 s 的 4 s 窗标为 seizure=1；仅短暂接触边界的窗标为 `-1`，不作为监督中心。
- 窗长 4 s、步长 1 s；所有鼠混合划分，按独立临床事件和背景块目标 70/15/15。
- 同一临床事件及其通道副本不跨 split；21 窗上下文必须留在同一连续片段，并对跨 split 原始 4 s 窗执行 purge。
- 当前严格协议审计：6,155 个独立事件、6,412 个通道事件；train/val/test 通道事件为 4,495/960/957。正式全窗口测试中心和 sampler 中心必须分开报告，sampler 的每事件上限不能替代正式测试分母。

### 三个正式模型

1. **EEGWaveNet-SC**：输入 `[B,1,1000]`；六个 stride-2 深度可分离时域卷积（1000→500→250→125→62→31→15），取五个尺度；每尺度 `Conv 1→32(k=4) + BN + LeakyReLU + Conv 32→32(k=4) + BN + LeakyReLU + temporal mean`，拼接 160 维后接 `Linear 160→64→32→2` 和 log-softmax。该基线只有单窗，不是真正 21 窗上下文。
2. **Physio-Sequence（no auxiliary）**：输入 `[B,21,1,1000]`；每窗 9/31/101 点多尺度时域卷积，经 1×1 投影为 64 维 token；可微 STFT（Hann，n_fft=128，hop=32，0.5–70 Hz）产生频域 token；13 个统计量（RMS、line length、Hjorth、五频带功率、theta/delta、谱熵、主频/谱边缘等）经 MLP；时间↔频率双向 4-head cross-attention、门控残差融合、4 层非因果 TCN、正弦位置编码、2 层 4-head Transformer，取中心 token 接 `64→32→2` 分类头。此版本仍把统计量作为输入门控，只是不加辅助回归头，不能称作“完全不使用统计量”。
3. **Physio + statistics**：结构同上，训练期再加 `64→64→13` 统计量辅助回归头；损失为 CE + 0.15×masked SmoothL1（实际训练实现中的权重以 run manifest 为准）。

训练共同使用 AdamW（lr 3e-4、weight decay 1e-4）、ReduceLROnPlateau、梯度裁剪；最大 12 epoch、验证指标早停 patience=4，按验证 macro-animal AUPRC 保存 checkpoint。达到 epoch 上限不等于已证明收敛。

### 连续解码

模型先输出每个物理窗的 `P(seizure)`，再做 rolling median、滞回阈值、k-of-n 持续性确认、短间隙/短片段处理，并把完整 4 s 窗支持区间做并集。模型输出窗和连续事件不是同一个计数。当前 corrected decoder 的参数和验证冻结记录在各模型 `decoder_fit.json`；测试集不参与阈值选择。

### 当前结果如何读

当前 corrected 测试事件主表位于 `sleep_staging/output/epilepsy_program_20260821/01_annotation_detection/06_decoder_audit_v2/corrected_combined_old_plus_new/`。三模型的测试人工事件分母为 957，形式完整可观测分母为 953；其余 4 行仍保留在图集。当前 pooled 事件 precision 约为 EEGWaveNet-SC 0.297、Physio-Sequence 0.215、Physio+statistics 0.241；这不是“预测很低”，而是候选区间分母中外部候选较多。临床使用必须同时看事件 sensitivity、precision、F1、IoU、零覆盖率、边界误差和每通道小时误报，不能只看 accuracy 或 AUPRC。

## 3. 任务二：PG36 形态学探索分类

### 观察单位与特征

一行正式 PG36 事件表 = 一个临床事件 × 一个有效通道。事件内物理窗的 12 个形态特征和 12 个同流训练背景 z 值先取中位数，形成事件级向量。每只鼠把 CH7、CH8 的事件行放在同一个 pooled GMM 中拟合；稳定性、最小支持和 BIC 规则选择鼠内 pooled prototypes。跨鼠只允许不同鼠原型以 conservative complete-linkage 合并，禁止同鼠两个原型进入同一 PG。PG01–PG36 是探索性形态标签，不是临床发作亚型。

### 当前冻结快照和一个必须纠正的事实

当前 PG36 快照包含 9,144 个事件×通道观察、194,530 个物理窗、40 个鼠内 pooled prototypes，冻结为 36 个 PG 类。它自己的 split 是 6,393/1,375/1,376（train/val/test），并不是当前 Task-1 严格 protocol 的 4,495/960/957。键审计显示测试集只有 111/1,376 同时满足 `event_id+channel+split` 完全一致；642 个键存在但 split 不一致，591 个键完全不在严格 protocol。因此现有 PG36 分类指标只能作为形态探索快照，不能与任务一的严格测试指标合并宣称同一划分。

### 五个分类模型

PG36 当前比较五个模型：EEGWaveNet-SC、Physio-TF、Physio+statistics、Braindecode EEGNetv4 和 EEGConformer。前两类沿用单通道时域/时频分支；EEGNetv4 和 EEGConformer 使用固定 Braindecode 结构、随机初始化、单通道输入并适配 250 Hz/1000 点窗口，没有导入预训练权重。每事件最多 8 个窗口只用于训练平衡敏感性分析；正式全窗口测试必须使用 `all_window_test_predictions.parquet` 和 `all_window_model_comparison.csv`。

### PCA、聚类和典型波形

- 鼠内散点：每个点是独立事件×通道；同鼠类别数只等于该鼠实际观察到的 PG 子集，不应把各鼠类别数相加当成 36。
- 全局 PCA：每个 PG 类只有一个冻结特征质心点，共 36 点；PC1/PC2 是对 24 维 robust-scaled 质心特征的线性投影，不是原始电压轴，也不是临床严重度轴。
- 典型波形：必须从真实事件内全部物理窗中按 robust feature median 选择，并记录确切 cache 位置；单个窗不能替代 10 s 以上临床事件。

## 4. 临床溯源图集（本次修正）

新增脚本 `scripts/render_pg36_event_traceability.py`，输出目录：

`sleep_staging/output/epilepsy_program_20260821/02_morphology_taxonomy/all_mice_old_plus_new/strict_all_channel_v1/07_clinical_traceability_pg36_v1/`

- `pdf_by_mouse/`：测试集全部事件×通道页面；每页连续滤波波形、人工边界、事件内窗条带和代表窗。
- `review_ready/pdf_by_mouse/`：只包含通过当前键、split、缓存和时长门控的页面；源表仍保留完整测试分母。
- `source_data/event_traceability.parquet`：事件级 PG 标签、窗口数、代表窗和 QC。
- `source_data/window_traceability.parquet`：每一个 4 s 物理窗的时间、序号、PG 标签、缓存目录和本事件中位数距离。
- `source_data/protocol_alignment_summary.csv`：PG36 与严格 protocol 的键交集，按鼠/通道拆分。
- `all_events_index/`：全 train/val/test 的 9,144 事件和 194,530 窗索引版。

QC 规则为：超过 120 s、存在重叠链、没有物理窗、缓存缺失、严格 protocol 键缺失或 split 不一致就标为 `review_required`。这些行不被 renderer 静默删除；重新训练前必须由临床复核后进入 quarantine 或重建标签。

## 5. 从原始数据重跑和减少发散

每次新批次只允许使用一个新的 run 目录和一个 manifest。先生成过滤缓存，再生成严格协议和 alignment audit，门控通过后才训练；训练、全窗口评估、事件解码、图形和报告均从该 manifest 的同一输入生成。任何图形都必须带有 source-data 表和 `event_id/session/channel/cache_local_index` 回溯字段。若对齐门控失败，停止指标解释，不用调阈值、删难例或换代表窗制造性能提升。
