"""Portable extraction of the archived common-OHLC-factor VWAP method.

Inputs are already cleaned, uniquely keyed, sorted CSVs. This utility does not
download or redistribute any data and does not recreate the raw-provider export.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

KEYS = ['order_book_id', 'date']
PRICE_COLUMNS = ['open', 'high', 'low', 'close']
HALF_UNIT = .00005
ROUND_EPS = 1e-9
TOL = .011


def estimate_factor(new, old):
    """No-intercept least squares constrained to common rounding intervals."""
    lower = np.zeros(len(new))
    upper = np.full(len(new), np.inf)
    numerator = np.zeros(len(new))
    denominator = np.zeros(len(new))
    for col in PRICE_COLUMNS:
        original = old[col].to_numpy(np.float64)
        adjusted = new[col].to_numpy(np.float64)
        if not np.all(np.isfinite(original) & (original > 0)):
            raise ValueError('Original OHLC must be finite and positive.')
        lower = np.maximum(lower, (adjusted - HALF_UNIT) / original)
        upper = np.minimum(upper, (adjusted + HALF_UNIT) / original)
        numerator += original * adjusted
        denominator += original * original
    if np.any(lower > upper + 1e-12):
        raise ValueError('Incompatible same-date OHLC adjustment-factor intervals.')
    upper = np.maximum(upper, lower)
    factor = np.clip(numerator / denominator, lower, upper)
    return factor, lower, upper


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adjusted', required=True, help='Cleaned forward-adjusted A-share daily OHLC/volume/total_turnover CSV.')
    parser.add_argument('--unadjusted', required=True, help='Same-key unadjusted OHLC/volume/total_turnover CSV.')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    temporary = output.with_suffix(output.suffix + '.partial')
    if output.exists() or temporary.exists():
        raise FileExistsError('Refusing to overwrite existing output or a partial transfer.')
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    previous = None
    dtype = {k: str for k in KEYS}
    with pd.read_csv(args.adjusted, dtype=dtype, chunksize=200_000) as adjusted, pd.read_csv(args.unadjusted, dtype=dtype, chunksize=200_000) as original, temporary.open('w', encoding='utf-8-sig', newline='') as stream:
        from itertools import zip_longest
        for new, old in zip_longest(adjusted, original):
            if new is None or old is None or not np.array_equal(new[KEYS].to_numpy(), old[KEYS].to_numpy()):
                raise ValueError('All stock/date keys must match exactly; no fills or dropped keys are allowed.')
            index = pd.MultiIndex.from_frame(new[KEYS])
            if index.has_duplicates or not index.is_monotonic_increasing or (previous is not None and index[0] <= previous):
                raise ValueError('Inputs must be sorted by stock/date with globally unique keys.')
            previous = index[-1]
            if not np.allclose(new.total_turnover, old.total_turnover, rtol=1e-10, atol=1e-5):
                raise ValueError('New and original turnover differ.')
            if not np.all(np.isfinite(old.volume) & (old.volume > 0)):
                raise ValueError('Unadjusted volume must be finite and positive.')
            raw_vwap = new.total_turnover.to_numpy() / old.volume.to_numpy()
            factor, lower, upper = estimate_factor(new, old)
            vwap = raw_vwap * factor
            for column in PRICE_COLUMNS:
                if np.any(np.abs(old[column].to_numpy() * factor - new[column].to_numpy()) > HALF_UNIT + ROUND_EPS):
                    raise ValueError('OHLC reconstruction exceeds source rounding precision.')
            if not np.all(np.isfinite(vwap) & (vwap > 0)):
                raise ValueError('VWAP must be finite and positive.')
            if np.any((vwap < new.low.to_numpy() - TOL) | (vwap > new.high.to_numpy() + TOL)):
                raise ValueError('Material VWAP range violation; values are never clamped or silently removed.')
            new['vwap'] = vwap
            new['vwap_reconstruction_factor'] = factor
            new['vwap_factor_lower'] = lower
            new['vwap_factor_upper'] = upper
            new['unadjusted_volume'] = old.volume.to_numpy()
            new.to_csv(stream, index=False, header=(count == 0), float_format='%.12g')
            count += len(new)
    if count == 0:
        raise ValueError('No input rows.')
    temporary.replace(output)
    print(f'Reconstructed {count:,} rows: {output}')


if __name__ == '__main__':
    main()
