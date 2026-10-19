"""FactorGCL 模块化复现：包名独立于 Python 标准库 code。"""

# Windows Anaconda 的 NumPy/MKL 与 PyTorch 可能载入两个 Intel OpenMP。
# 使用 MKL 支持的顺序线程层，保留 PyTorch 自身线程；Linux 不改变此设置。
import os
if os.name == "nt":
    os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
import torch

