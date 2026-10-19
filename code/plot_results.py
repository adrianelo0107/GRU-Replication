"""Plot only the published A-share seed-0 aggregate results."""
import argparse
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default='outputs/figures')
    args = parser.parse_args()
    out = Path(args.output_dir)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    base = ROOT / 'results/ashare_gru_seed0'
    metrics = json.loads((base / 'oos_metrics.json').read_text())
    horizons = [1, 5, 10, 20]
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    for ax, key in zip(axes.flat, ['ic','icir','mse','f1']):
        values = [metrics[str(h)][key] for h in horizons]
        ax.bar(range(4), values, color='#35618f', width=.6)
        ax.set_xticks(range(4), [str(h) for h in horizons])
        ax.set_xlabel('Horizon (trading days)')
        ax.set_ylabel(key.upper())
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle('Unofficial financial GRU replication | A-share baseline, seed 0')
    fig.tight_layout()
    fig.savefig(out / 'main_results.png', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharex='col')
    for col, name in enumerate(['fold_1','fold_2']):
        logs = pd.read_csv(base / name / 'epochs.csv')
        result = json.loads((base / name / 'result.json').read_text())
        for key, label in [('mse','Training MSE')]:
            axes[0,col].plot(logs.epoch, logs[key], label=label, linewidth=1.4)
        axes[0,col].set_title(f'{name}: best {result["best_epoch"]}, stop {int(logs.epoch.max())}')
        axes[0,col].set_ylabel('Training loss')
        axes[0,col].legend(fontsize=8)
        axes[1,col].plot(logs.epoch, logs.validation_ic, color='#35618f', label='Validation mean IC')
        axes[1,col].plot(logs.epoch, logs.best_ic, color='#bd663a', linestyle='--', label='Best so far')
        axes[1,col].set_ylabel('Four-horizon validation IC')
        axes[1,col].set_xlabel('Epoch')
        axes[1,col].legend(fontsize=8)
        for row in range(2):
            axes[row,col].axvline(result['best_epoch'], color='gray', linestyle=':', linewidth=1)
            axes[row,col].grid(alpha=.2)
    fig.suptitle('A-share seed 0 | complete training and validation histories')
    fig.tight_layout()
    fig.savefig(out / 'training_history.png', dpi=180)
    plt.close(fig)
    monthly = pd.read_csv(base / 'monthly_metrics.csv')
    fig, ax = plt.subplots(figsize=(11, 4))
    for horizon, group in monthly.groupby('horizon'):
        ax.plot(pd.to_datetime(group.month), group.ic, marker='.', linewidth=1.1, label=f'{horizon}-day')
    ax.axhline(0, color='gray', linewidth=.8)
    ax.set_ylabel('Mean daily cross-sectional IC')
    ax.set_xlabel('Test month')
    ax.set_title('A-share seed 0 | 42 test months (descriptive evaluation)')
    ax.legend(ncol=4, fontsize=9)
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(out / 'monthly_ic.png', dpi=180)
    plt.close(fig)
    print(f'Wrote three figures to {out}')


if __name__ == '__main__':
    main()
