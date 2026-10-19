"""Train the archived RQ financial GRU baseline, seed 0, in a standalone repo."""
import argparse
import json
from protocol import CONFIG, check_protocol, install_runtime, engine
from factorgcl.config import load_config, resolve_path
from factorgcl.data import prepare_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=CONFIG)
    parser.add_argument('--fold', choices=['fold_1', 'fold_2'])
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'])
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.device:
        cfg.training.device = args.device
    check_protocol(cfg)
    output = resolve_path(cfg.training.output_dir).resolve()
    archive = resolve_path('results').resolve()
    if output == archive or archive in output.parents:
        raise ValueError('Training cannot overwrite the published results/ archive.')
    cache = prepare_cache(cfg.data)
    if args.prepare_only:
        print(json.dumps(cache.manifest, indent=2))
        return
    for fold in cfg.folds:
        if args.fold and fold.name != args.fold:
            continue
        install_runtime(cfg, fold.name, args.resume)
        print(engine.run_experiment(cfg, args.resume, fold.name), flush=True)


if __name__ == '__main__':
    main()
