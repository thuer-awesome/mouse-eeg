# 脚本索引

`train_pg36.py`、`render_task2_pg36_corrected_v2.py` 和 `render_task1_corrected_model_performance.py` 是正式结果对应的脚本入口。其余脚本按来源阶段保留，均要求显式传入数据/输出路径；不应直接把默认路径当作新的实验结果。

优先使用顶层 `epilepsy-eeg` CLI：它统一记录参数、阻止非空输出目录被静默覆盖，并把 YAML 配置和命令行覆盖写入运行 manifest。直接运行脚本适合研究人员调试，仍必须显式指定输入和输出，并在完成后运行 `epilepsy-eeg audit`。

任务一主链：

- `run_combined_epilepsy_models.py`：二分类训练/连续推理入口（只在重新训练时使用）。
- `run_corrected_event_decoder_v1.py`：验证集拟合 corrected interval-aware decoder。
- `render_corrected_event_atlas.py`：沿用三层临床事件图布局生成完整 PDF。
- `render_task1_corrected_model_performance.py`：窗口混淆矩阵、PR/ROC、训练诊断和事件指标图。

任务二主链：

- `run_channel_independent_morphology.py`、`fit_pooled_mouse_morphology.py`、`fit_pooled_global_taxonomy.py`：逐通道特征、逐鼠聚类和 PG36 冻结。
- `render_task2_pg36_corrected_v2.py`：逐鼠 PCA/波形、全局 36 质心、支持矩阵和五模型全类别 PR/ROC。
- `train_pg36.py`、`evaluate_strict_classifiers_all_windows.py`：五个分类器和全部测试窗口主分母评估。
