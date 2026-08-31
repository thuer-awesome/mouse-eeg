# 任务二 PG36 正式图形口径

## 结论

`PG01–PG36` 是全部合格小鼠、合格通道的 `event × channel` 事件联合后冻结的 36 个全局探索性形态类别。每个通道仍是独立样本；同鼠 CH7 和 CH8 只在同一张鼠内图中并列显示，不拼接为一个模型输入。

正式数据共有 `9,144` 个独立 `event × channel` 点，7 只鼠，合格通道为 CH7/CH8；E77/CH7 排除。36 个 PG 标签与通道组合理论上有 72 个，实际有 61 个有事件支持，11 个在支持矩阵中明确为 `not observed`。

## 划分与对齐限制

PG36 自己的事件划分为 `6,393/1,375/1,376`（训练/验证/测试），不是任务一严格 protocol 的 `4,495/960/957`。键审计显示，PG36 测试集只有 `111/1,376` 条 `event_id + channel + split` 与严格 protocol 完全一致；另有 642 条键存在但 split 不一致，591 条键完全缺失。因此，本报告的分类结果是形态探索快照，不是任务一严格测试集上的跨域泛化估计，不能与任务一指标合并解读。

## 每只鼠的 PG 标签核对

| 小鼠 | CH7 标签 | CH8 标签 | 鼠内并集 | 类别数 |
|---|---|---|---|---:|
| E07 | 无有效样本 | PG04, PG07, PG08 | PG04, PG07, PG08 | 3 |
| E50 | PG11, PG17, PG19, PG20, PG21, PG24, PG30 | PG11, PG17, PG19, PG20, PG21, PG24, PG30 | 同左 | 7 |
| E76 | PG06, PG18, PG19, PG22, PG35 | PG05, PG06, PG18, PG19, PG22, PG35, PG36 | PG05, PG06, PG18, PG19, PG22, PG35, PG36 | 7 |
| E77 | 无效并排除 | PG15, PG29 | PG15, PG29 | 2 |
| E79 | PG16, PG19, PG23, PG26, PG27, PG34 | PG16, PG19, PG23, PG26, PG27, PG34 | 同左 | 6 |
| E81 | PG03, PG10, PG13, PG32, PG33 | PG13, PG14, PG28, PG32, PG33 | PG03, PG10, PG13, PG14, PG28, PG32, PG33 | 7 |
| E82 | PG01, PG02, PG09, PG11, PG12, PG25, PG27, PG31 | PG01, PG02, PG09, PG11, PG12, PG25, PG27, PG31 | 同左 | 8 |

因此，**每只鼠不应各自出现 36 类**。36 是所有小鼠的类别并集；每只鼠只显示其实际观测到的子集。完整核对表在 `source_data/pg36_category_mouse_membership.csv` 和 `source_data/pg36_mouse_channel_support_matrix.csv`。

## 图形含义

- `01_per_mouse_pg_pca/`：每只鼠一张图，所有该鼠的事件点同时显示；圆点为 CH7，三角为 CH8；颜色和标注统一使用 PG 标签。标签只标注该鼠实际出现的 PG 类别。
- `02_per_mouse_pg_waveforms/`：每只鼠一行一个 PG 类别，左右列分别为 CH7/CH8；没有支持的通道格显示 `not observed`。
- `03_global_pg_centroid_pca/`：严格 36 个点，每个点是一个冻结的训练事件 PG 原型；不是几个鼠内簇数，也不是事件点抽样。右侧给出每个 PG 到最近 PG 原型的距离。
- `04_pg_support_presence/`：PG01–PG36 与每个独立鼠/通道流的完整支持矩阵，便于逐项核对类别是否存在。
- `06_classifier_pr_roc_all36/`：五个模型各自对 PG01–PG36 的完整一对多 PR/ROC 小图，每类一个 panel。

旧图中的 `Pxx` 是鼠内 pooled 原型，`Lxx` 是通道局部原型；它们不再用于当前正式 PG 图，也不能与 PG01–PG36 相加。

## PR/ROC 灰线说明

旧的 overview PR/ROC 图中，饱和色曲线是支持数最大的 8 类，浅灰曲线是其余 28 类。浅灰并不表示“不典型”、无效或被排除，只是为了避免 36 条曲线互相遮挡。当前 v2 目录已经改为每个 PG 类别独立 panel，所有类别都显示并标注支持数和 AP/AUROC。
