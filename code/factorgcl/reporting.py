"""Training-loss plotting used by the A-share financial GRU seed-0 baseline."""
import pandas as pd
from .config import resolve_path


def plot_training_loss(log_path, output_path=None, best_epoch=None, title=None):
    """读取完整 epoch 日志，绘制三项原始训练损失；不平滑、不补造缺失轮次。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    log_path = resolve_path(log_path)
    output_path = resolve_path(output_path) if output_path else log_path.parent / "loss_curve.png"
    logs = pd.read_csv(log_path)
    required = {"epoch", "loss", "mse", "cl"}
    if not required.issubset(logs.columns) or logs.empty:
        raise ValueError(f"损失日志须包含非空的 epoch/loss/mse/cl 列：{log_path}")
    logs = logs.sort_values("epoch")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    marker = "o" if len(logs) <= 20 else None
    for ax, column, label, color in zip(
        axes, ("loss", "mse", "cl"),
        ("Total training loss", "MSE", "Contrastive loss (raw)"),
        ("tab:blue", "tab:orange", "tab:green"),
    ):
        # 独立纵轴避免不同数量级掩盖各项损失变化；每点为一个 epoch 的日均损失。
        ax.plot(logs.epoch, logs[column], color=color, marker=marker, linewidth=1.6, label=label)
        if best_epoch is not None:
            ax.axvline(best_epoch, color="gray", linestyle="--", linewidth=1,
                       label=f"Best validation epoch: {best_epoch}")
        ax.set_ylabel(label)
        ax.grid(alpha=.25)
        ax.legend(loc="best", fontsize=9)
    axes[-1].set_xlabel("Epoch")
    axes[-1].xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.suptitle(title or f"Training loss: {log_path.parent.name}")
    fig.tight_layout()
    try:
        fig.savefig(output_path, dpi=200)
    finally:
        plt.close(fig)
    return output_path

