"""Verify the published archive and recompute all metrics without market data."""
import hashlib
import json
from pathlib import Path
# Initialize the package's Windows MKL policy before importing NumPy/pandas.
from factorgcl import torch
import numpy as np
import pandas as pd
from factorgcl.config import load_config
from factorgcl.evaluation import aggregate_metrics
from factorgcl.models import make_model

ROOT = Path(__file__).resolve().parents[1]


def compare(computed, stored):
    if set(computed) != set(stored):
        raise ValueError('Horizon mismatch.')
    for horizon, metrics in computed.items():
        for key, value in metrics.items():
            other = stored[horizon][key]
            if value is None or other is None:
                if value is not other:
                    raise ValueError((horizon, key, value, other))
            elif not np.isclose(value, other, rtol=1e-10, atol=1e-12):
                raise ValueError((horizon, key, value, other))


def main():
    checksums = json.loads((ROOT / 'SHA256SUMS.json').read_text(encoding='utf-8'))
    for name, expected in checksums.items():
        path = ROOT / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'SHA-256 mismatch: {name}')
    base = ROOT / 'results/ashare_gru_seed0'
    config = load_config('code/configs/ashare_gru_seed0.json')
    daily = pd.read_csv(base / 'oos_daily_metrics.csv')
    if daily.duplicated(['date', 'horizon']).any() or daily.date.nunique() != 846 or len(daily) != 3384:
        raise ValueError('Unexpected OOS coverage or duplicate keys.')
    compare(aggregate_metrics(daily.to_dict('records')), json.loads((base / 'oos_metrics.json').read_text()))
    pieces = []
    for name in ['fold_1', 'fold_2']:
        folder = base / name
        logs = pd.read_csv(folder / 'epochs.csv')
        validations = [json.loads(line) for line in (folder / 'validation_epochs.jsonl').read_text().splitlines() if line.strip()]
        result = json.loads((folder / 'result.json').read_text())
        if logs.epoch.tolist() != list(range(1, len(logs) + 1)) or [v['epoch'] for v in validations] != logs.epoch.tolist():
            raise ValueError(f'{name}: missing training or validation epochs.')
        np.testing.assert_allclose(logs.loss, logs.mse, rtol=0, atol=1e-7)
        np.testing.assert_array_equal(logs.cl, np.zeros(len(logs)))
        scores = [np.mean([x['ic'] for x in v['validation'].values()]) for v in validations]
        np.testing.assert_allclose(scores, logs.validation_ic, rtol=0, atol=1e-12)
        best, bad = -np.inf, 0
        for row in logs.itertuples():
            if row.validation_ic > best:
                best, bad = row.validation_ic, 0
            else:
                bad += 1
            if bad != row.bad_epochs or not np.isclose(best, row.best_ic, rtol=0, atol=1e-12):
                raise ValueError(f'{name}: early stopping record mismatch.')
        if bad != 20 or result['best_epoch'] != int(logs.loc[logs.validation_ic.idxmax(), 'epoch']):
            raise ValueError(f'{name}: unexpected best/stop epoch.')
        model = make_model(config.model, len(config.data.horizons))
        if sum(p.numel() for p in model.parameters()) != 10308:
            raise ValueError('Expected the archived 10,308-parameter FinancialGRU.')
        if any(isinstance(module, torch.nn.modules.batchnorm._BatchNorm) for module in model.modules()):
            raise ValueError('The archived financial GRU has no BatchNorm.')
        model.load_state_dict(torch.load(folder / 'model_state.pt', map_location='cpu', weights_only=True), strict=True)
        fold_daily = pd.read_csv(folder / 'test/daily_metrics.csv')
        compare(aggregate_metrics(fold_daily.to_dict('records')), result['test'])
        pieces.append(fold_daily)
    pd.testing.assert_frame_equal(daily.reset_index(drop=True), pd.concat(pieces).sort_values(['date', 'horizon']).reset_index(drop=True))
    monthly = pd.read_csv(base / 'monthly_metrics.csv')
    if len(monthly) != 42 * 4:
        raise ValueError('Expected 42 months and four horizons.')
    for month, group in daily.groupby(daily.date.str[:7]):
        saved = monthly[monthly.month == month].set_index('horizon')
        compare(aggregate_metrics(group.to_dict('records')), {str(h): r.to_dict() for h, r in saved.iterrows()})
    print(f'PASS: {len(checksums)} SHA-256 entries; 2 folds; 846 days; 42 months; metrics, loss, validation, early stopping and model tensors verified.')


if __name__ == '__main__':
    main()
