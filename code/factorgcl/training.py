"""按日期训练、最佳验证 checkpoint 与可恢复的滚动折实验。"""
import csv
import hashlib
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from .config import resolve_path, validate_config
from .data import PanelCache, DailyDataset, ClipScaler, fit_scaler, fold_dates
from .evaluation import evaluate_dataset, identity_batch, save_json, aggregate_metrics, validation_score
from .models.temporal_contrastive import compute_losses
from .models import make_model
from .reporting import plot_training_loss
from .monitoring import TrainingMonitor


def choose_device(training):
    torch.set_num_threads(training.cpu_threads)
    if training.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(training.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定 CUDA，但当前 PyTorch/设备不可用。可选择 cpu 或 auto。")
    return device


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def rng_state():
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


def experiment_signature(cfg, fold, cache):
    values = cfg.to_dict()
    # 恢复可改总轮数、设备、输出路径；不能静默换模型/学习率/标签口径。
    for key in ("max_epochs", "device", "cpu_threads", "output_dir",
                "monitoring_enabled","monitoring_dates_per_epoch","scaler_reference_dir"):
        values["training"].pop(key)
    values["data"] = {
        k: v for k, v in values["data"].items() if not k.endswith("_path") and k != "cache_dir"
    }
    values["folds"] = [vars(fold)]
    values.pop("notes")
    return hashlib.sha256(
        (str(cache.signature) + repr(sorted(values.items()))).encode()
    ).hexdigest()


def save_checkpoint(path, model, optimizer, epoch, best_score, bad_epochs, cfg, fold, scaler, cache, signature):
    state = {
        "format_version": 1, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "epoch": epoch, "best_score": best_score, "bad_epochs": bad_epochs,
        "config": cfg.to_dict(), "fold": vars(fold), "scaler": scaler.to_dict(),
        "cache_signature": cache.signature, "experiment_signature": signature, "rng": rng_state(),
    }
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def load_checkpoint(path, device="cpu"):
    # 仅载入本项目产生的可信 checkpoint；包含 NumPy RNG，不能 weights_only=True。
    return torch.load(path, map_location=device, weights_only=False)


def run_fold(cfg, fold, cache, resume=False):
    seed_everything(cfg.training.seed)
    device = choose_device(cfg.training)
    output = resolve_path(cfg.training.output_dir) / fold.name
    output.mkdir(parents=True, exist_ok=True)
    last_path, best_path = output / "last.pt", output / "best.pt"
    if last_path.exists() and not resume:
        raise FileExistsError(f"{last_path} 已存在；恢复用 --resume，新实验请更改 output_dir。")
    splits = fold_dates(cache, fold, cfg.data)
    if any(len(v) == 0 for v in splits.values()):
        raise ValueError(f"{fold.name} 在 purge 后至少一个分区为空。")
    signature = experiment_signature(cfg, fold, cache)
    state = load_checkpoint(last_path) if resume and last_path.exists() else None
    if state and state["experiment_signature"] != signature:
        raise ValueError("checkpoint 与当前模型/数据/折/训练设置不一致。")
    if state:
        scaler = ClipScaler(**state["scaler"])
    elif cfg.training.scaler_reference_dir:
        reference=load_checkpoint(resolve_path(cfg.training.scaler_reference_dir)/fold.name/"best.pt")
        data_settings=lambda values:{k:v for k,v in values.items()
            if not k.endswith("_path") and k!="cache_dir"}
        expected_hash=hashlib.sha256("\n".join(str(cache.dates[d]) for d in splits["train"]).encode()).hexdigest()
        if (reference["cache_signature"]!=cache.signature or reference["fold"]!=vars(fold)
            or data_settings(reference["config"]["data"])!=data_settings(cfg.to_dict()["data"])
            or reference["scaler"]["fit_dates_sha256"]!=expected_hash):
            raise ValueError("复用裁剪阈值的缓存、数据设置或训练日期不匹配。")
        scaler=ClipScaler(**reference["scaler"])
        save_json({"checkpoint":str(Path(cfg.training.scaler_reference_dir)/fold.name/"best.pt"),
                   "cache_signature":cache.signature,"fit_dates_sha256":expected_hash,
                   "model_weights_reused":False},output/"scaler_reference.json")
    else:
        print(f"[{fold.name}] 拟合训练段裁剪阈值，锚点 {len(splits['train'])} 天。", flush=True)
        scaler = fit_scaler(cache, cfg.data, splits["train"], cfg.training.seed)
    save_json(scaler.to_dict(), output / "clip_scaler.json")
    model = make_model(cfg.model, len(cfg.data.horizons)).to(device)
    train = DailyDataset(
        cache, cfg.data, splits["train"], scaler,
        include_future=model.cl_enabled and cfg.training.contrastive_weight > 0,
    )
    valid = DailyDataset(cache, cfg.data, splits["valid"], scaler, require_labels=False)
    test = DailyDataset(cache, cfg.data, splits["test"], scaler, require_labels=False)
    cfg.save(output / "config.json")
    save_json({
        "device": str(device), "cache_signature": cache.signature,
        "experiment_signature": signature, "parameters": sum(p.numel() for p in model.parameters()),
        "signals": {k: [str(cache.dates[d]) for d in dataset.dates]
                    for k, dataset in (("train", train), ("valid", valid), ("test", test))},
        "requested_after_purge": {k: len(v) for k, v in splits.items()},
        "industry_columns": 134, "smoke_stock_limit": cache.manifest["request"]["stock_limit"],
    }, output / "run_manifest.json")
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.training.learning_rate, betas=tuple(cfg.training.adam_betas),
        eps=cfg.training.adam_eps, weight_decay=cfg.training.weight_decay,
    )
    start_epoch, best_score, bad_epochs = 0, -float("inf"), 0
    if state:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        # CPU checkpoint 在 GPU 恢复时移动优化器的动量。
        for values in optimizer.state.values():
            for key, value in values.items():
                if isinstance(value, torch.Tensor):
                    values[key] = value.to(device)
        start_epoch, best_score, bad_epochs = state["epoch"], state["best_score"], state["bad_epochs"]
        restore_rng(state["rng"])
    monitor=TrainingMonitor(output,cfg.training.monitoring_enabled,
        [str(cache.dates[d]) for d in train.dates],[str(cache.dates[d]) for d in valid.dates],
        cfg.training.monitoring_dates_per_epoch,cfg.data.horizons)
    monitor.trim_for_resume(start_epoch)
    print(f"[{fold.name}] {device}; train/valid/test={len(train)}/{len(valid)}/{len(test)}；start_epoch={start_epoch}", flush=True)
    log_path = output / "epochs.csv"
    if state and log_path.exists():
        logs = pd.read_csv(log_path)
        # 允许写日志时中断；已提交 checkpoint 后的重跑不会重复计入 epoch。
        logs = logs[logs.epoch <= start_epoch].drop_duplicates("epoch", keep="last")
        logs.to_csv(log_path, index=False)
    for epoch in range(start_epoch, cfg.training.max_epochs):
        if bad_epochs >= cfg.training.patience:
            break
        model.train()
        order = np.arange(len(train))
        if cfg.training.shuffle_dates:
            np.random.default_rng(cfg.training.seed + epoch).shuffle(order)
        loader = DataLoader(Subset(train, order.tolist()), batch_size=None, num_workers=0, collate_fn=identity_batch)
        losses = np.zeros(3, dtype=np.float64)
        started = time.perf_counter()
        for step, batch in enumerate(loader, start=1):
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            history = model.forward_history(batch.x, batch.beta)
            loss = compute_losses(model, history, batch, cfg.training)
            if not torch.isfinite(loss["total"]):
                raise FloatingPointError(f"{fold.name}/{batch.date} 损失出现 NaN/Inf。")
            diagnostic=monitor.training_record(model,history,batch,loss,epoch+1,cfg.training.contrastive_weight)
            loss["total"].backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError(f"{fold.name}/{batch.date} 梯度出现 NaN/Inf。")
            monitor.append(diagnostic,model)
            optimizer.step()
            losses += [loss[k].detach().item() for k in ("total", "mse", "cl")]
            if step % 100 == 0:
                print(f"[{fold.name}] epoch={epoch + 1}, step={step}/{len(train)}, mean_loss={losses[0]/step:.6f}", flush=True)
        metrics, _ = evaluate_dataset(model, valid, cfg.data.horizons, device)
        monitor.validation(model,valid,device,epoch+1)
        score = validation_score(metrics)
        improved = score > best_score + cfg.training.min_delta
        if improved:
            best_score, bad_epochs = score, 0
        else:
            bad_epochs += 1
        record = {
            "epoch": epoch + 1, "loss": losses[0] / len(train), "mse": losses[1] / len(train),
            "cl": losses[2] / len(train), "validation_ic": score, "best_ic": best_score,
            "bad_epochs": bad_epochs, "seconds": time.perf_counter() - started,
        }
        save_json(metrics, output / "validation_last.json")
        if improved:
            save_checkpoint(best_path, model, optimizer, epoch + 1, best_score, bad_epochs,
                            cfg, fold, scaler, cache, signature)
        save_checkpoint(last_path, model, optimizer, epoch + 1, best_score, bad_epochs,
                        cfg, fold, scaler, cache, signature)
        with log_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(record))
            if file.tell() == 0:
                writer.writeheader()
            writer.writerow(record)
        print(f"[{fold.name}] epoch={epoch + 1} loss={record['loss']:.6f} valid_IC={score:.6f} best={best_score:.6f}", flush=True)
    if not best_path.exists():
        raise FileNotFoundError("没有最佳 checkpoint；检查训练/验证日志。")
    best = load_checkpoint(best_path, device)
    # 正常结束、早停或恢复已完成训练时，均从完整日志绘制该折损失曲线。
    curve_path = plot_training_loss(log_path, best_epoch=best["epoch"], title=f"Training loss: {fold.name}")
    print(f"[{fold.name}] 损失曲线已保存：{curve_path}", flush=True)
    model.load_state_dict(best["model"])
    metrics, daily = evaluate_dataset(model, test, cfg.data.horizons, device, output / "test", fold.name)
    save_json({"best_epoch": best["epoch"], "validation_ic": best["best_score"], "test": metrics}, output / "result.json")
    return metrics, daily


def run_experiment(cfg, resume=False, fold_name=None):
    validate_config(cfg)
    if not cfg.folds:
        raise ValueError("至少配置一个滚动折。")
    cache = PanelCache(cfg.data.cache_dir)
    output = resolve_path(cfg.training.output_dir)
    selected = [f for f in cfg.folds if fold_name is None or f.name == fold_name]
    if not selected:
        raise ValueError(f"未找到折 {fold_name}。")
    all_daily, fold_results = [], {}
    for fold in selected:
        metrics, daily = run_fold(cfg, fold, cache, resume)
        all_daily.extend(daily)
        fold_results[fold.name] = metrics
    # 拼接时读已完成但本次未选择的折；独立 fold CLI 也能逐步形成最终 OOS 表。
    for fold in cfg.folds:
        if fold in selected:
            continue
        path = output / fold.name / "test" / "daily_metrics.csv"
        if path.exists():
            import json
            manifest_path = output / fold.name / "run_manifest.json"
            saved = json.loads(manifest_path.read_text(encoding="utf-8"))
            if saved["experiment_signature"] != experiment_signature(cfg, fold, cache):
                raise ValueError(f"{fold.name} 的旧测试结果与当前实验设置不同，不能拼接。请更换 output_dir。")
            all_daily.extend(pd.read_csv(path).to_dict("records"))
    daily_frame = pd.DataFrame(all_daily).sort_values(["date", "horizon"])
    if daily_frame.duplicated(["date", "horizon"]).any():
        raise ValueError("样本外日期/期限重复，不能合并结果。")
    daily_frame.to_csv(output / "oos_daily_metrics.csv", index=False)
    combined = aggregate_metrics(daily_frame.to_dict("records"))
    save_json(combined, output / "oos_metrics.json")
    save_json(fold_results, output / "fold_metrics_latest.json")
    pd.DataFrame([{"horizon": int(k), **v} for k, v in combined.items()]).to_csv(output / "oos_metrics.csv", index=False)
    return combined
