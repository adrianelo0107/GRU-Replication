"""隔离GRU工厂、实验签名和只读验证诊断；复用原训练循环与评价。"""
import hashlib
import json
from factorgcl import torch
from factorgcl.config import load_config, resolve_path
from factorgcl import training as engine
from factorgcl.models import make_model

CONFIG = 'code/configs/ashare_gru_seed0.json'
RUN = 'outputs/ashare_gru_seed0'
SPEC = dict(id='ashare_gru_seed0_20261007', architecture='pytorch_gru_last_state_linear',
            gate='reset_after', layers=2, hidden_dim=32, input_dim=6, outputs=[1, 5, 10, 20],
            batchnorm=False, bias=True, dropout=0., bidirectional=False, initial_hidden='zero_each_window',
            initialization='pytorch_default_uniform', loss='masked_mse_only', seed=0,
            labels='original_unclipped', future_branch=False, graph=False,
            disclosure='baseline_layers_width_shared_head_and_no_BN_are_explicit_completion_assumptions')
ORIGINAL_SIGNATURE = engine.experiment_signature
ORIGINAL_EVALUATE = engine.evaluate_dataset


def check_protocol(cfg):
    if cfg.notes.get('gru_protocol') != SPEC:
        raise ValueError('Use the archived financial GRU seed-0 protocol.')
    if (cfg.model.name != 'gru' or cfg.training.seed != 0
        or cfg.model.gru_layers != 2 or cfg.model.embedding_dim != 32
        or cfg.model.input_dim != 6 or cfg.data.horizons != [1, 5, 10, 20]
        or cfg.model.use_prior or cfg.model.use_hidden or cfg.model.use_cl
        or cfg.training.contrastive_weight != 0
        or cfg.training.scaler_reference_dir is not None):
        raise ValueError('The archive uses a fresh 2-layer 32-unit GRU, seed 0, MSE only.')
    return SPEC


def experiment_signature(cfg, fold, cache):
    spec = json.dumps(cfg.notes['gru_protocol'], sort_keys=True, separators=(',', ':'))
    return hashlib.sha256((ORIGINAL_SIGNATURE(cfg, fold, cache) + '|financial-gru-v1|' + spec).encode()).hexdigest()


def install_runtime(cfg, fold_name=None, resume=False):
    check_protocol(cfg)
    engine.make_model = make_model
    engine.experiment_signature = experiment_signature
    engine.evaluate_dataset = ORIGINAL_EVALUATE
    if fold_name is None:
        return
    directory = resolve_path(cfg.training.output_dir) / fold_name
    directory.mkdir(parents=True, exist_ok=True)
    start = engine.load_checkpoint(directory / 'last.pt')['epoch'] if resume and (directory / 'last.pt').exists() else 0
    sidecar = directory / 'validation_epochs.jsonl'
    if sidecar.exists():
        records = [json.loads(line) for line in sidecar.read_text().splitlines() if line.strip()]
        sidecar.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records if r['epoch'] <= start), encoding='utf-8')
    epoch = [start]

    def evaluate(*args, **kwargs):
        model = args[0] if args else kwargs['model']
        output_dir = args[4] if len(args) > 4 else kwargs.get('output_dir')
        observations = []
        original_predict = model.predict

        def observed_predict(x, beta):
            prediction = original_predict(x, beta)
            values = prediction.detach()
            observations.append(dict(mean=values.mean(0).cpu().tolist(),
                std=values.std(0, correction=0).cpu().tolist(),
                positive=(values > 0).float().mean(0).cpu().tolist()))
            return prediction

        had_override = 'predict' in model.__dict__
        model.predict = observed_predict
        try:
            result = ORIGINAL_EVALUATE(*args, **kwargs)
        finally:
            if had_override: model.predict = original_predict
            else: del model.predict
        if output_dir is None:
            epoch[0] += 1
            row = dict(epoch=epoch[0], validation=result[0], labels='original_unclipped',
                       future_branch=False, batchnorm=False,
                       prediction_stats_scope='all_history_predictable_nodes_before_label_mask')
            row['validation_prediction_stats'] = {
                str(h): {key: sum(v[field][j] for v in observations) / len(observations)
                    for key, field in [('mean_daily_prediction_mean', 'mean'),
                                      ('mean_daily_prediction_std', 'std'),
                                      ('mean_daily_positive_fraction', 'positive')]}
                for j, h in enumerate(cfg.data.horizons)}
            with sidecar.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        return result

    engine.evaluate_dataset = evaluate
    (directory / 'gru_protocol.json').write_text(json.dumps(SPEC, ensure_ascii=False, indent=2), encoding='utf-8')
