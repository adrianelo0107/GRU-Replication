"""预测评估：每日指标、跨日汇总、JSON 保存和历史分支评价集中维护。"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def f1_from_counts(tp, fp, fn):
    denominator = 2 * tp + fp + fn
    return 2 * tp / denominator if denominator else np.nan


def daily_metrics(date, prediction, target, mask, horizons):
    rows = []
    for j, horizon in enumerate(horizons):
        p, y = prediction[mask[:, j], j], target[mask[:, j], j]
        if not np.isfinite(p).all():
            raise FloatingPointError(f"{date}/{horizon} 预测出现 NaN/Inf。")
        n = len(p)
        ic = np.nan
        if n >= 2 and p.std(ddof=0) > 1e-12 and y.std(ddof=0) > 1e-12:
            ic = float(np.corrcoef(p, y)[0, 1])
        tp = int(((p > 0) & (y > 0)).sum())
        fp = int(((p > 0) & (y < 0)).sum())
        fn = int(((p < 0) & (y > 0)).sum())
        squared_error = float(np.square(p.astype(np.float64) - y).sum())
        rows.append({
            "date": date, "horizon": horizon, "n": n, "ic": ic,
            "mse": squared_error / n if n else np.nan,
            "squared_error": squared_error, "tp": tp, "fp": fp, "fn": fn,
            "zero_sign_count": int(((p == 0) | (y == 0)).sum()),
            "f1": f1_from_counts(tp, fp, fn),
        })
    return rows


def aggregate_metrics(rows):
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("没有评价日期。")
    result = {}
    for horizon, group in frame.groupby("horizon", sort=True):
        ic = group.ic.dropna().to_numpy()
        ic_std = float(ic.std(ddof=0)) if len(ic) else np.nan
        n = int(group.n.sum())
        tp, fp, fn = (int(group[c].sum()) for c in ("tp", "fp", "fn"))
        result[str(int(horizon))] = {
            "ic": float(ic.mean()) if len(ic) else np.nan,
            "icir": float(ic.mean() / ic_std) if len(ic) and ic_std > 1e-12 else np.nan,
            "ic_std": ic_std, "valid_ic_days": len(ic), "total_days": len(group),
            "undefined_ic_days": len(group) - len(ic), "valid_labels": n,
            "mse": float(group.mse.mean()),
            "mse_pooled": float(group.squared_error.sum() / n) if n else np.nan,
            "f1": f1_from_counts(tp, fp, fn), "f1_daily_mean": float(group.f1.mean()),
            "tp": tp, "fp": fp, "fn": fn, "zero_sign_count": int(group.zero_sign_count.sum()),
        }
    return json_ready(result)


def validation_score(summary):
    values = [x["ic"] for x in summary.values()]
    # 四个期限全部必须可评价，避免以改变期限集合的方式选择 checkpoint。
    if not values or any(v is None for v in values):
        raise ValueError("验证集至少一个期限无有效 IC；请扩大样本或复核标签。")
    return float(np.mean(values))


# 验证和测试只执行历史分支，不更新 BatchNorm。

def identity_batch(value):
    return value


@torch.no_grad()
def evaluate_dataset(model, dataset, horizons, device, output_dir=None, fold_name=""):
    model.eval()
    daily = []
    prediction_path = None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        prediction_path = output_dir / "predictions.csv"
        # 每次评价重写自身的结果文件，避免 resume 时重复追加旧测试记录。
        if prediction_path.exists():
            prediction_path.unlink()
    for batch in DataLoader(dataset, batch_size=None, shuffle=False, num_workers=0, collate_fn=identity_batch):
        tensors = batch.to(device)
        prediction = model.predict(tensors.x, tensors.beta).cpu().numpy()
        target = batch.target.numpy()
        mask = batch.label_mask.numpy()
        rows = daily_metrics(batch.date, prediction, target, mask, horizons)
        for row in rows:
            row["fold"] = fold_name
        daily.extend(rows)
        if prediction_path is not None:
            count = len(batch.stock_ids)
            frame = pd.DataFrame({
                "fold": fold_name, "date": batch.date,
                "order_book_id": np.repeat(batch.stock_ids, len(horizons)),
                "horizon": np.tile(horizons, count),
                "prediction": prediction.reshape(-1),
                "normalized_label": np.where(mask, target, np.nan).reshape(-1),
                "raw_return": batch.raw_returns.reshape(-1),
                "label_mask": mask.reshape(-1),
            })
            frame.to_csv(prediction_path, mode="a", header=not prediction_path.exists(), index=False)
    summary = aggregate_metrics(daily)
    if output_dir is not None:
        pd.DataFrame(daily).to_csv(output_dir / "daily_metrics.csv", index=False)
        save_json(summary, output_dir / "metrics.json")
    return summary, daily
