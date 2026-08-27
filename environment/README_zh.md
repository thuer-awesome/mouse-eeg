# 环境快照

`requirements-core.txt` 用于轻量数据和协议检查；`requirements-deep.txt` 增加 PyTorch；`requirements-pg36.txt` 增加 Braindecode；`environment.yml` 给出完整 conda 参考环境。

PyTorch CUDA wheel 必须和目标驱动匹配，不能由这个仓库替用户决定。安装后先运行：

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -m compileall -q src scripts
python -m pytest -q
```

正式权重和大表使用 Git LFS；原始 EEG 和机构标签始终留在本地数据管理员控制的路径。
