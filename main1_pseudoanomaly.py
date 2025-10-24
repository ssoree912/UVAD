import argparse
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml

from codes import cleanse, featurebank


def set_global_seeds(seed: int):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def setup_run_logger(run_name: str, log_dir: Path, verbose: bool = True) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{run_name}.log"

    logger = logging.getLogger(f"cknn.{run_name}")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    if verbose:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)

    logger.propagate = False
    logger.info("Logger initialized. Writing to %s", log_path)
    return logger


def resolve_from_config(arg_value, config_section: Optional[dict], key: str, default):
    if arg_value is not None:
        return arg_value
    if config_section and config_section.get(key) is not None:
        return config_section[key]
    return default


def load_appae_config(config_arg: Optional[str], dataset_name: str):
    config_path = None
    if config_arg:
        config_path = Path(config_arg)
    else:
        candidate = Path("configs") / f"config_{dataset_name}.yaml"
        if candidate.exists():
            config_path = candidate

    config_section = {}
    if config_path and config_path.exists():
        with open(config_path, "r", encoding="utf-8") as handle:
            raw_cfg = yaml.safe_load(handle) or {}
        config_section = raw_cfg.get("appaerecon", {}) or {}
    return config_path, config_section


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='avenue')
    parser.add_argument('--uvadmode', default='merge', choices=['merge', 'partial'])
    parser.add_argument('--mode', default='app', choices=['app', 'mot'])
    parser.add_argument('--gmm_n', default=12, type=int)

    parser.add_argument('--config', type=str, default=None,
                        help='Path to dataset configuration YAML (defaults to configs/config_<dataset>.yaml).')

    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility.')
    parser.add_argument('--run_name', type=str, default=None,
                        help='Unique name for this run. Defaults to seed+timestamp.')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs (overrides config).')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Training batch size (overrides config).')
    parser.add_argument('--lr', type=float, default=None,
                        help='Optimizer learning rate (overrides config).')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='DataLoader worker count (overrides config).')
    parser.add_argument('--save_root', type=str, default=None,
                        help='Base directory for saving checkpoints and metadata.')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='Directory to store training logs.')
    parser.add_argument('--device', type=str, default=None,
                        help='Device override, e.g., cpu, cuda, cuda:0.')

    parser.add_argument('--magnitude_prune', type=float, default=None,
                        help='Fraction (0-1) for magnitude pruning. 0 disables.')
    parser.add_argument('--random_prune', type=float, default=None,
                        help='Fraction (0-1) for random pruning. 0 disables.')
    parser.add_argument('--random_prune_seed', type=int, default=None,
                        help='Seed for random pruning mask (if random pruning enabled).')
    parser.add_argument('--unprune_epoch', type=int, default=None,
                        help='Epoch index (1-based) after which pruning masks are removed.')

    parser.add_argument('--eval_config', type=str, default=None,
                        help='Optional path to config YAML for AUROC evaluation.')
    parser.add_argument('--eval_interval', type=int, default=None,
                        help='Run evaluation every N epochs (0 disables).')
    parser.add_argument('--no_stream_logs', action='store_true',
                        help='Disable streaming logs to stdout (file logging remains).')
    parser.add_argument('--score_output', type=str, default=None,
                        help='Override output path for AppAE scores when mode=app. '
                             'You can use placeholders {dataset}, {mode}, {uvadmode}, {run}, {seed}.')
    return parser


def resolve_device(device_arg: Optional[str]) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


args = build_parser().parse_args()


def main():
    dataset_name = args.dataset_name
    uvadmode = args.uvadmode

    config_path, config_section = load_appae_config(args.config, dataset_name)

    seed = resolve_from_config(args.seed, config_section, 'seed', 111)
    epochs = resolve_from_config(args.epochs, config_section, 'epochs', 10)
    batch_size = resolve_from_config(args.batch_size, config_section, 'batch_size', 64)
    optimizer_lr = resolve_from_config(args.lr, config_section, 'optimizer_lr', 1e-3)
    num_workers = resolve_from_config(args.num_workers, config_section, 'num_workers', 4)
    magnitude_prune = resolve_from_config(args.magnitude_prune, config_section, 'magnitude_prune', 0.0)
    random_prune = resolve_from_config(args.random_prune, config_section, 'random_prune', 0.0)
    random_prune_seed = args.random_prune_seed if args.random_prune_seed is not None else config_section.get('random_prune_seed')
    unprune_epoch = args.unprune_epoch if args.unprune_epoch is not None else config_section.get('unprune_epoch')
    eval_config_value = args.eval_config if args.eval_config is not None else config_section.get('eval_config')
    eval_interval = resolve_from_config(args.eval_interval, config_section, 'eval_interval', 0)

    save_root_value = resolve_from_config(args.save_root, config_section, 'save_root', 'artifacts')
    log_dir_value = resolve_from_config(args.log_dir, config_section, 'log_dir', 'logger')
    device_arg = args.device if args.device is not None else config_section.get('device')
    run_name_prefix = config_section.get('run_name_prefix')

    timestamp = datetime.utcnow().strftime('%Y%m%d-%H%M%S')
    default_run_name = f"{dataset_name}_{uvadmode}_{args.mode}_seed{seed}_{timestamp}"
    if run_name_prefix:
        default_run_name = f"{run_name_prefix}_seed{seed}_{timestamp}"
    run_name = args.run_name or default_run_name

    seed = int(seed)
    epochs = int(epochs)
    batch_size = int(batch_size)
    optimizer_lr = float(optimizer_lr)
    num_workers = int(num_workers)
    magnitude_prune = float(magnitude_prune)
    random_prune = float(random_prune)
    if random_prune_seed is not None:
        random_prune_seed = int(random_prune_seed)
    if unprune_epoch is not None:
        unprune_epoch = int(unprune_epoch)
    eval_interval = int(eval_interval)

    save_root_path = Path(str(save_root_value)).expanduser()
    log_dir_path = Path(str(log_dir_value)).expanduser()

    if eval_config_value and config_path and not os.path.isabs(str(eval_config_value)):
        eval_config_value = str((config_path.parent / str(eval_config_value)).resolve())

    logger = setup_run_logger(run_name, log_dir_path, verbose=not args.no_stream_logs)

    if config_path:
        logger.info("Loaded training config from %s", config_path)
    else:
        logger.info("No training config file supplied; using built-in defaults.")

    set_global_seeds(seed)
    logger.info("Random seed set to %d", seed)

    device = resolve_device(device_arg)
    logger.info("Using device: %s", device)

    logger.info(
        "Training hyperparameters -> epochs=%d, batch_size=%d, lr=%.6f, num_workers=%d",
        epochs, batch_size, optimizer_lr, num_workers
    )
    logger.info(
        "Pruning settings -> magnitude=%.2f%%, random=%.2f%%, unprune_epoch=%s, random_seed=%s",
        magnitude_prune * 100,
        random_prune * 100,
        str(unprune_epoch) if unprune_epoch is not None else "None",
        str(random_prune_seed) if random_prune_seed is not None else "None"
    )
    logger.info("Save root: %s", save_root_path)
    logger.info("Log directory: %s", log_dir_path)
    if eval_config_value:
        logger.info("Evaluation config: %s (interval=%s)", eval_config_value, eval_interval)

    logger.info("Pseudo anomaly generation started (dataset=%s, uvadmode=%s, mode=%s, seed=%d, run=%s)",
                dataset_name, uvadmode, args.mode, seed, run_name)

    dpath = f'features/{dataset_name}/cleansescores'

    if args.mode == 'app':
        logger.info('Launching AppAE reconstruction pipeline.')
        ret = cleanse.AppAErecon(
            dataset_name,
            uvadmode,
            run_name=run_name,
            save_root=str(save_root_path),
            log_dir=str(log_dir_path),
            logger=logger,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            optimizer_lr=optimizer_lr,
            num_workers=num_workers,
            magnitude_prune=magnitude_prune,
            random_prune=random_prune,
            random_prune_seed=random_prune_seed,
            unprune_epoch=unprune_epoch,
            eval_config=eval_config_value,
            eval_interval=eval_interval,
            seed=seed,
            config_path=str(config_path) if config_path else None,
        ).infer()
        if args.score_output:
            fpath = args.score_output.format(
                dataset=dataset_name,
                mode=args.mode,
                uvadmode=uvadmode,
                run=run_name,
                seed=seed
            )
        else:
            suffix = ""
            if run_name:
                suffix = f"_{run_name}"
            elif seed is not None:
                suffix = f"_seed{seed}"
            fpath = f'{dpath}/{uvadmode}_aerecon{suffix}_flat.npy'
    elif args.mode == 'mot':
        logger.info('Launching fGMM pseudo anomaly pipeline.')
        tr_f = featurebank.get(dataset_name, 'mot', 'train', uvadmode=uvadmode).astype(np.float32)

        mot_run_dir = save_root_path / dataset_name / uvadmode / run_name
        mot_run_dir.mkdir(parents=True, exist_ok=True)
        mot_model_path = mot_run_dir / 'fgmm_model.npz'
        logger.info("Saving fGMM model to %s", mot_model_path)

        fgmm = cleanse.fGMM(
            dataset_name,
            uvadmode,
            tr_f,
            args.gmm_n,
            logger=logger,
            model_path=mot_model_path,
            random_state=seed
        )
        ret = fgmm.infer(tr_f)

        if args.score_output:
            fpath = args.score_output.format(
                dataset=dataset_name,
                mode=args.mode,
                uvadmode=uvadmode,
                run=run_name,
                seed=seed
            )
        else:
            suffix = f"_{run_name}" if run_name else ""
            fpath = f'{dpath}/{uvadmode}_velo_fgmm{suffix}_flat.npy'

    else:
        raise ValueError()

    output_path = Path(fpath)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, ret)
    logger.info("Saved outputs to %s", output_path)

    if args.mode == 'app' and not args.score_output:
        canonical_path = Path(f'{dpath}/{uvadmode}_aerecon_flat.npy')
        if canonical_path != output_path:
            canonical_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(output_path, canonical_path)
            logger.info("Saved compatibility copy to %s", canonical_path)


if __name__ == '__main__':
    main()
