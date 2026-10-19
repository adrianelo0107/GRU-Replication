"""模型数据准备：缓存、窗口/标签、训练裁剪与时间切分集中维护。"""
from dataclasses import dataclass, asdict, fields
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from .config import resolve_path

FEATURES = ["high", "open", "low", "close", "vwap", "volume"]
CACHE_VERSION = 1


# 1. 读取主表并准备固定行业缓存

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def source_metadata(config):
    result = {}
    for key in ("panel_path", "calendar_path", "industry_path", "industry_dictionary_path"):
        path = resolve_path(getattr(config, key))
        stat = path.stat()
        result[key] = {"size": stat.st_size}
        if key == "panel_path":
            result[key]["mtime_ns"] = stat.st_mtime_ns
        else:
            result[key]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def prepare_cache(config, stock_limit=None, rebuild=False, chunksize=200_000):
    directory = resolve_path(config.cache_dir)
    manifest_path = directory / "manifest.json"
    metadata = source_metadata(config)
    request = {"version": CACHE_VERSION, "sources": metadata, "stock_limit": stock_limit}
    if manifest_path.exists() and not rebuild:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing["request"] != request:
            raise ValueError("缓存与数据/stock_limit 不匹配。请指定新的 cache_dir 或显式 --rebuild。")
        return PanelCache(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # 完成标志最后写入；中断后不会把半成品当有效缓存。
    if manifest_path.exists():
        manifest_path.unlink()
    calendar = pd.read_csv(resolve_path(config.calendar_path), dtype={"date": str})
    if calendar.date.duplicated().any() or not calendar.date.is_monotonic_increasing:
        raise ValueError("交易日历必须严格递增且无重复。")
    dates = calendar.date.to_numpy(dtype="U10")
    industry = pd.read_csv(resolve_path(config.industry_path), dtype={"stock_code": str})
    dictionary = pd.read_csv(resolve_path(config.industry_dictionary_path))
    if len(dictionary) != 134 or set(dictionary.sw2021_l2_id) != set(range(134)):
        raise ValueError("SW2021 字典应包含完整 134 个二级行业。")
    if industry.stock_code.duplicated().any():
        raise ValueError("静态股票行业表存在重复股票。")
    industry = industry.sort_values("stock_code").reset_index(drop=True)
    # 主表清洗已经限定沪深 A 股；静态表中只接受沪深股票编码。
    industry = industry[industry.stock_code.str.startswith(("0", "3", "6"))]
    ids = industry.sw2021_l2_id.to_numpy(dtype=np.int64)
    if np.any((ids < 0) | (ids >= 134)):
        raise ValueError("行业 ID 超出固定 134 列字典。")
    industry = industry.assign(order_book_id=[
        code + (".XSHG" if code.startswith("6") else ".XSHE")
        for code in industry.stock_code
    ]).sort_values("order_book_id")
    if stock_limit is not None:
        if stock_limit < 2:
            raise ValueError("烟测股票上限至少为 2。")
        industry = industry.head(stock_limit)
    stocks = industry.order_book_id.to_numpy(dtype="U11")
    ids = industry.sw2021_l2_id.to_numpy(dtype=np.int64)
    n_days, n_stocks = len(dates), len(stocks)
    feature = np.lib.format.open_memmap(
        directory / "features.npy", mode="w+", dtype="float32", shape=(n_days, n_stocks, 6)
    )
    prices = np.lib.format.open_memmap(
        directory / "vwap.npy", mode="w+", dtype="float64", shape=(n_days, n_stocks)
    )
    feature[:] = np.nan
    prices[:] = np.nan
    date_lookup = {d: i for i, d in enumerate(dates)}
    stock_lookup = {s: i for i, s in enumerate(stocks)}
    selected_rows = 0
    usecols = ["date", "order_book_id"] + FEATURES
    with pd.read_csv(
        resolve_path(config.panel_path), usecols=usecols, chunksize=chunksize,
        dtype={"date": str, "order_book_id": str},
    ) as reader:
        for chunk in reader:
            matched = chunk[chunk.order_book_id.isin(stock_lookup)]
            if not matched.empty:
                d = matched.date.map(date_lookup)
                if d.isna().any():
                    raise ValueError("主表日期不在 trading_calendar 内。")
                d = d.to_numpy(dtype=np.int64)
                s = matched.order_book_id.map(stock_lookup).to_numpy(dtype=np.int64)
                keys = d * n_stocks + s
                if len(np.unique(keys)) != len(keys) or np.isfinite(prices[d, s]).any():
                    raise ValueError("主表存在重复 stock/date，请重新清洗。")
                values = matched[FEATURES].to_numpy(dtype=np.float64)
                if not np.all(np.isfinite(values) & (values > 0)):
                    raise ValueError("主表六特征须为有限正值，请先运行 validate_data.py。")
                feature[d, s] = values
                prices[d, s] = values[:, 4]
                selected_rows += len(matched)
            # 上一步清洗的主表已按股票/日期排序；烟测可以在股票上限后提前结束。
            if stock_limit is not None and chunk.order_book_id.iloc[-1] > stocks[-1]:
                break
    if not selected_rows:
        raise ValueError("静态映射没有匹配到任何主表行情。")
    feature.flush()
    prices.flush()
    np.save(directory / "dates.npy", dates, allow_pickle=False)
    np.save(directory / "stocks.npy", stocks, allow_pickle=False)
    np.save(directory / "industries.npy", ids, allow_pickle=False)
    np.save(directory / "sample_period.npy", calendar.is_sample_period.to_numpy(bool), allow_pickle=False)
    manifest = {
        "request": request, "feature_order": FEATURES,
        "shape": [n_days, n_stocks, 6], "selected_rows": selected_rows,
        "industry_policy": "static_SW2021_134_latest_known_classification",
        "date_start": str(dates[0]), "date_end": str(dates[-1]),
        "observed_stocks": int(np.isfinite(prices).any(axis=0).sum()),
        "signature": digest([request, dates.tolist(), stocks.tolist(), ids.tolist()]),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return PanelCache(directory)


class PanelCache:
    def __init__(self, directory: str | Path):
        self.directory = resolve_path(directory)
        path = self.directory / "manifest.json"
        if not path.exists():
            raise FileNotFoundError(f"缺少完成的缓存 {path}，先运行 prepare_data.py。")
        self.manifest = json.loads(path.read_text(encoding="utf-8"))
        self.features = np.load(self.directory / "features.npy", mmap_mode="r", allow_pickle=False)
        self.vwap = np.load(self.directory / "vwap.npy", mmap_mode="r", allow_pickle=False)
        self.dates = np.load(self.directory / "dates.npy", allow_pickle=False)
        self.stocks = np.load(self.directory / "stocks.npy", allow_pickle=False)
        self.industries = np.load(self.directory / "industries.npy", allow_pickle=False)
        self.sample_period = np.load(self.directory / "sample_period.npy", allow_pickle=False)
        if list(self.features.shape) != self.manifest["shape"]:
            raise ValueError("缓存 shape 与 manifest 不一致。")
        if self.manifest["feature_order"] != FEATURES:
            raise ValueError("缓存特征顺序不匹配。")
        if self.vwap.shape != self.features.shape[:2] or len(self.dates) != self.features.shape[0]:
            raise ValueError("缓存价格/日期维度不一致。")
        if len(self.stocks) != self.features.shape[1] or len(self.industries) != len(self.stocks):
            raise ValueError("缓存股票/行业维度不一致。")

    @property
    def signature(self):
        return self.manifest["signature"]




# 2. 市场日历窗口、相对价量与收益标签

def normalize_window(values):
    """输入 [L,N,6]；价格除以窗口末日 close，成交量除以有效日均量。"""
    values = np.array(values, dtype=np.float64, copy=True)
    valid_count = np.isfinite(values[:, :, 5]).sum(axis=0)
    volume_mean = np.divide(
        np.nansum(values[:, :, 5], axis=0), valid_count,
        out=np.full(values.shape[1], np.nan), where=valid_count > 0,
    )
    close = values[-1, :, 3]
    denominator_ok = np.isfinite(close) & (close > 0) & np.isfinite(volume_mean) & (volume_mean > 0)
    missing = (~np.isfinite(values)).mean(axis=(0, 2))
    values[:, :, :5] /= np.where(denominator_ok, close, np.nan)[None, :, None]
    values[:, :, 5] /= np.where(denominator_ok, volume_mean, np.nan)[None, :]
    return values.transpose(1, 0, 2), denominator_ok, missing


def history_window(cache, day, config):
    if day < config.history_length - 1:
        return np.empty((0, config.history_length, 6)), np.array([], dtype=np.int64)
    normalized, ok, missing = normalize_window(
        cache.features[day - config.history_length + 1:day + 1]
    )
    eligible = ok & (missing <= config.missing_ratio_threshold)
    indices = np.flatnonzero(eligible)
    return normalized[indices], indices


def make_targets(cache, day, stocks, horizons):
    """原式买入 t+1、卖出 t+delta+1；不沿股票稀疏行情做 shift。"""
    n, h = len(stocks), len(horizons)
    raw = np.full((n, h), np.nan, dtype=np.float64)
    if day + 1 < len(cache.dates):
        entry = np.asarray(cache.vwap[day + 1, stocks])
        for j, horizon in enumerate(horizons):
            if day + horizon + 1 >= len(cache.dates):
                continue
            exit_price = np.asarray(cache.vwap[day + horizon + 1, stocks])
            mask = np.isfinite(entry) & (entry > 0) & np.isfinite(exit_price) & (exit_price > 0)
            raw[mask, j] = exit_price[mask] / entry[mask] - 1
    target = np.zeros((n, h), dtype=np.float32)
    mask = np.zeros((n, h), dtype=bool)
    for j in range(h):
        valid = np.isfinite(raw[:, j])
        if valid.sum() < 2:
            continue
        std = raw[valid, j].std(ddof=0)
        if not np.isfinite(std) or std <= 1e-12:
            continue
        target[valid, j] = (raw[valid, j] - raw[valid, j].mean()) / std
        mask[valid, j] = True
    return target, raw, mask


@dataclass
class DailyBatch:
    date: str
    stock_indices: np.ndarray
    stock_ids: np.ndarray
    x: torch.Tensor
    beta: torch.Tensor
    target: torch.Tensor
    raw_returns: np.ndarray
    label_mask: torch.Tensor
    future: torch.Tensor | None
    cl_mask: torch.Tensor

    def to(self, device):
        return DailyBatch(**{
            f.name: (getattr(self, f.name).to(device) if isinstance(getattr(self, f.name), torch.Tensor)
                     else getattr(self, f.name))
            for f in fields(self)
        })


class DailyDataset(Dataset):
    def __init__(self, cache, config, dates, scaler, include_future=False, require_labels=True):
        self.cache, self.config, self.scaler = cache, config, scaler
        self.include_future = include_future
        self.dates = []
        # 不足横截面仅在锚点列表中剔除；绝不按未来数据改变图节点。
        for day in dates:
            _, stocks = history_window(cache, int(day), config)
            if len(stocks) < config.min_stocks:
                continue
            if require_labels and not make_targets(cache, int(day), stocks, config.horizons)[2].any():
                continue
            self.dates.append(int(day))
        if not self.dates:
            raise ValueError("该时间段没有可用的逐日横截面，检查窗口、日期与股票上限。")

    def __len__(self):
        return len(self.dates)

    def __getitem__(self, index):
        day = self.dates[index]
        x, stocks = history_window(self.cache, day, self.config)
        target, raw, mask = make_targets(self.cache, day, stocks, self.config.horizons)
        beta = np.zeros((len(stocks), 134), dtype=np.float32)
        beta[np.arange(len(stocks)), self.cache.industries[stocks]] = 1
        future = None
        cl_mask = np.zeros(len(stocks), dtype=bool)
        if self.include_future:
            if day + self.config.future_length >= len(self.cache.dates):
                raise ValueError("训练未来窗口超出缓存，应先执行严格时间 purge。")
            normalized, ok, missing = normalize_window(
                self.cache.features[day + 1:day + self.config.future_length + 1, stocks]
            )
            cl_mask = ok & (missing <= self.config.missing_ratio_threshold)
            future = self.scaler.transform(normalized)
            # 保留历史图全部节点；未来不可用节点置零但不参加 InfoNCE。
            future[~cl_mask] = 0
        return DailyBatch(
            date=str(self.cache.dates[day]), stock_indices=stocks, stock_ids=self.cache.stocks[stocks],
            x=torch.from_numpy(self.scaler.transform(x)), beta=torch.from_numpy(beta),
            target=torch.from_numpy(target), raw_returns=raw, label_mask=torch.from_numpy(mask),
            future=None if future is None else torch.from_numpy(future), cl_mask=torch.from_numpy(cl_mask),
        )




# 3. 只用训练锚点拟合裁剪阈值

@dataclass
class ClipScaler:
    lower: list[float]
    upper: list[float]
    observed: list[int]
    sampled: list[int]
    fit_dates_sha256: str
    method: str = "seeded_uniform_priority_reservoir"

    def transform(self, values):
        result = np.clip(values, np.array(self.lower), np.array(self.upper))
        return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def to_dict(self):
        return asdict(self)


def fit_scaler(cache, config, dates, seed=0):
    rng = np.random.default_rng(seed)
    samples = [np.array([], dtype=np.float64) for _ in range(6)]
    priorities = [np.array([], dtype=np.float64) for _ in range(6)]
    counts = np.zeros(6, dtype=np.int64)
    capacity = config.clip_reservoir_size
    for day in dates:
        x, _ = history_window(cache, int(day), config)
        for j in range(6):
            v = x[:, :, j].reshape(-1)
            v = v[np.isfinite(v)]
            counts[j] += len(v)
            values = np.concatenate([samples[j], v])
            keys = np.concatenate([priorities[j], rng.random(len(v))])
            if len(values) > capacity:
                idx = np.argpartition(keys, capacity - 1)[:capacity]
                values, keys = values[idx], keys[idx]
            samples[j], priorities[j] = values, keys
    if any(len(v) == 0 for v in samples):
        raise ValueError("训练锚点没有足够的有效特征拟合裁剪阈值。")
    lower, upper = zip(*(np.quantile(v, config.clip_quantiles) for v in samples))
    return ClipScaler(
        list(lower), list(upper), counts.tolist(), [len(x) for x in samples],
        hashlib.sha256("\n".join(str(cache.dates[d]) for d in dates).encode()).hexdigest(),
        "exact" if np.all(counts <= capacity) else "seeded_uniform_priority_reservoir",
    )




# 4. 滚动分区和目标端点隔离

def fold_dates(cache, fold, config):
    result = {}
    lookahead = max(max(config.horizons) + 1, config.future_length)
    for split in ("train", "valid", "test"):
        start, end = getattr(fold, split + "_start"), getattr(fold, split + "_end")
        dates = np.flatnonzero(
            (cache.dates >= start) & (cache.dates <= end) & cache.sample_period
        )
        dates = dates[dates >= config.history_length - 1]
        if split != "test":
            # train/valid 的所有监督端点必须留在本分区；不借用下一分区标签。
            boundary = int(np.searchsorted(cache.dates, end, side="right")) - 1
            dates = dates[dates + lookahead <= boundary]
        result[split] = dates
    return result
