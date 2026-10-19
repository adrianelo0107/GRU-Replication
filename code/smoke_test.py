"""Temporary synthetic CPU training and exact checkpoint-resume verification."""
import copy
import json
from pathlib import Path
import tempfile
# Initialize the package's Windows MKL policy before importing NumPy/pandas.
from factorgcl import torch
import numpy as np
import pandas as pd
from factorgcl.config import load_config, FoldConfig
from factorgcl import training as engine
import protocol as launcher
from factorgcl.models.gru import reference_cell


def same(left, right):
    if isinstance(left, torch.Tensor):
        if not torch.equal(left, right):
            raise AssertionError('Tensor mismatch after resume.')
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        if left.keys() != right.keys():
            raise AssertionError('State keys mismatch.')
        for key in left:
            same(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        if len(left) != len(right):
            raise AssertionError('Sequence length mismatch.')
        for a, b in zip(left, right):
            same(a, b)
    elif left != right:
        raise AssertionError(f'State mismatch: {left!r} versus {right!r}')


def run(cfg, resume=False):
    launcher.install_runtime(cfg, 'fold_1', resume)
    try:
        return engine.run_experiment(cfg, resume, 'fold_1')
    finally:
        engine.experiment_signature = launcher.ORIGINAL_SIGNATURE
        engine.evaluate_dataset = launcher.ORIGINAL_EVALUATE


def main():
    torch.manual_seed(0)
    cell = torch.nn.GRUCell(6, 32).double()
    x, h = torch.randn(3, 6, dtype=torch.float64), torch.randn(3, 32, dtype=torch.float64)
    expected = reference_cell(x, h, cell.weight_ih, cell.weight_hh, cell.bias_ih, cell.bias_hh)
    torch.testing.assert_close(cell(x, h), expected, atol=1e-12, rtol=1e-12)
    print('PASS: independent reset-after gate expansion matches PyTorch GRUCell.')
    with tempfile.TemporaryDirectory(prefix='factorgcl_synthetic_') as temp:
        folder = Path(temp)
        cache = folder / 'cache'
        cache.mkdir()
        rng = np.random.default_rng(818)
        days, stocks = 150, 8
        close = 10 * np.exp(np.cumsum(rng.normal(0, .015, (days, stocks)), axis=0))
        features = np.stack([close * 1.02, close * .995, close * .98, close, close * 1.001, rng.uniform(1000, 2000, (days, stocks))], axis=-1)
        dates = pd.bdate_range('2013-10-10', periods=days).strftime('%Y-%m-%d').to_numpy(dtype='U10')
        arrays = {'features':features.astype('float32'), 'vwap':features[:, :, 4], 'dates':dates, 'stocks':np.array([f'SYNTH{i:03}' for i in range(stocks)]), 'industries':np.arange(stocks), 'sample_period':np.ones(days, dtype=bool)}
        for name, values in arrays.items():
            np.save(cache / (name + '.npy'), values, allow_pickle=False)
        (cache / 'manifest.json').write_text(json.dumps({'shape':[days,stocks,6], 'feature_order':['high','open','low','close','vwap','volume'], 'signature':'synthetic-only-not-a-financial-experiment', 'request':{'stock_limit':stocks}}), encoding='utf-8')
        cfg = load_config('code/configs/rq_gru_seed0.json')
        cfg.data.cache_dir = str(cache)
        cfg.training.device = 'cpu'
        cfg.training.cpu_threads = 1
        cfg.training.max_epochs = 2
        cfg.folds = [FoldConfig('fold_1', dates[59], dates[87], dates[88], dates[114], dates[115], dates[119])]
        cfg.training.output_dir = str(folder / 'continuous')
        run(cfg)
        resumed = copy.deepcopy(cfg)
        resumed.training.output_dir = str(folder / 'resumed')
        resumed.training.max_epochs = 1
        run(resumed)
        resumed.training.max_epochs = 2
        run(resumed, resume=True)
        a = folder / 'continuous/fold_1'
        b = folder / 'resumed/fold_1'
        first, second = (engine.load_checkpoint(p / 'last.pt') for p in (a, b))
        for key in ['model','optimizer','scaler','rng','epoch','best_score','bad_epochs']:
            same(first[key], second[key])
        if (a / 'test/predictions.csv').read_bytes() != (b / 'test/predictions.csv').read_bytes():
            raise AssertionError('Predictions differ after resume.')
        if (a / 'validation_epochs.jsonl').read_bytes() != (b / 'validation_epochs.jsonl').read_bytes():
            raise AssertionError('Validation records differ after resume.')
        print('PASS: synthetic CPU forward/backward, two epochs, exact resumed model/Adam/scaler/RNG/validation/predictions. No market data used; temporary files removed on exit.')


if __name__ == '__main__':
    main()
