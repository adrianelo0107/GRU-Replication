"""Temporal Residual Contrastive Learning：投影、InfoNCE 与联合目标，式 (11)-(13)。

未来残差由 model.py 按式 (9)-(10) 组装；这里计算投影和训练损失。
"""
import torch
from torch import nn
import torch.nn.functional as F


class ContrastiveProjection(nn.Sequential):
    def __init__(self, input_dim, hidden_dim, output_dim, negative_slope):
        # 正文只明确“两层 MLP + LeakyReLU”；宽度和最后激活来自指南默认。
        super().__init__(
            nn.Linear(input_dim, hidden_dim), nn.LeakyReLU(negative_slope),
            nn.Linear(hidden_dim, output_dim), nn.LeakyReLU(negative_slope),
        )


def masked_mse(prediction, target, mask):
    if not mask.any():
        raise ValueError("该日没有有效收益标签。")
    # 严格采用指南第 5.5 节的 mask 扩展：对有效元素平均。
    # 全标签有效时四期限等权；缺失不均衡时任务实际权重随有效数变化。
    return (prediction[mask] - target[mask]).square().mean()


def temporal_infonce(history, future, temperature=0.1, anchor_chunk=512):
    if len(history) < 2:
        return history.sum() * 0.0
    past = F.normalize(history, dim=-1, eps=1e-8)
    ahead = F.normalize(future, dim=-1, eps=1e-8)
    n = len(past)
    total = past.sum() * 0.0
    # 只拆 anchor 行，每行仍包含全体未来候选；数学上等于原式。
    for start in range(0, n, anchor_chunk):
        logits = past[start:start + anchor_chunk] @ ahead.T / temperature
        labels = torch.arange(start, start + len(logits), device=logits.device)
        total = total + F.cross_entropy(logits, labels, reduction="sum")
    return total / n


def compute_losses(model, historical, batch, training):
    mse = masked_mse(historical["prediction"], batch.target, batch.label_mask)
    cl = mse.new_zeros(())
    if model.cl_enabled and training.contrastive_weight > 0 and batch.cl_mask.sum() >= 2:
        future_alpha = model.forward_future(
            batch.future, batch.beta, historical["beta_h"]
        )
        past = model.projection(historical["e_alpha"][batch.cl_mask])
        future = model.projection(future_alpha[batch.cl_mask])
        cl = temporal_infonce(
            past, future, training.temperature, training.contrastive_anchor_chunk
        )
    return {"mse": mse, "cl": cl, "total": mse + training.contrastive_weight * cl}
