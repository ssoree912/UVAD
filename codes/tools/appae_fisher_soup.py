#!/usr/bin/env python3
"""
Fisher-weighted model soup for AppAE checkpoints.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

# Ensure project root is on sys.path
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from codes.tools.appae_fisher_utils import (
    load_appae_training_config,
    build_patch_dataloader,
    load_appae_model,
    compute_fisher_for_appae,
    fisher_weighted_average,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fisher-weighted soup for AppAE checkpoints.")
    parser.add_argument("--folders", nargs="+", required=True,
                        help="Run directories containing AppAE checkpoints.")
    parser.add_argument("--dataset_name", required=True, choices=["shanghaitech", "avenue", "ped2"])
    parser.add_argument("--uvadmode", required=True, choices=["merge", "partial"])
    parser.add_argument("--output", required=True,
                        help="Output path for merged checkpoint.")
    parser.add_argument("--config", default=None,
                        help="Training config YAML (defaults to configs/config_<dataset>.yaml).")
    parser.add_argument("--checkpoint_name", default="appaerecon_best.pkl",
                        help="Checkpoint filename within each folder.")
    parser.add_argument("--fisher_name", default="appaerecon_fisher.pt",
                        help="Filename to store/load Fisher information within each folder.")
    parser.add_argument("--device", default=None,
                        help="Device override for Fisher computation (cpu, cuda, cuda:0).")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size during Fisher computation.")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Override dataloader worker count.")
    parser.add_argument("--max_batches", type=int, default=50,
                        help="Max batches for Fisher (0 means unlimited).")
    parser.add_argument("--subset", type=int, default=None,
                        help="Number of patches to sample for Fisher computation.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for subset sampling.")
    parser.add_argument("--coefficients", type=str, default=None,
                        help="Comma-separated soup coefficients. Defaults to uniform weights.")
    parser.add_argument("--strategy", choices=["uniform"], default="uniform",
                        help="Coefficient strategy (currently only uniform).")
    parser.add_argument("--recompute_fisher", action="store_true",
                        help="Force recomputation of Fisher information.")
    parser.add_argument("--eps", type=float, default=1e-8,
                        help="Stability epsilon when normalising weights.")
    parser.add_argument("--metadata", default=None,
                        help="Optional JSON file to store soup metadata.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    return parser.parse_args()


def setup_logger(verbose: bool) -> logging.Logger:
    logger = logging.getLogger("appae_fisher_soup")
    logger.handlers.clear()
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO if verbose else logging.WARNING)
    logger.propagate = False
    return logger


def parse_coefficients(raw: Optional[str], n: int, strategy: str) -> List[float]:
    if raw:
        parts = [float(x.strip()) for x in raw.split(",")]
        if len(parts) != n:
            raise ValueError("Number of coefficients must match number of folders.")
        total = sum(parts)
        if total == 0:
            raise ValueError("Coefficient sum must be non-zero.")
        return [p / total for p in parts]

    if strategy == "uniform":
        return [1.0 / n] * n

    raise ValueError(f"Unsupported strategy: {strategy}")


def ensure_fisher(folder: Path,
                  fisher_path: Path,
                  cfg,
                  args,
                  logger: logging.Logger,
                  dataloader,
                  device: torch.device) -> dict:
    if fisher_path.exists() and not args.recompute_fisher:
        package = torch.load(str(fisher_path), map_location="cpu")
        return package["fisher"] if isinstance(package, dict) and "fisher" in package else package

    logger.info("Computing Fisher for %s", folder)

    checkpoint_path = folder / args.checkpoint_name
    model = load_appae_model(checkpoint_path, device)
    fisher, processed = compute_fisher_for_appae(
        model,
        dataloader,
        device=device,
        max_batches=None if args.max_batches == 0 else args.max_batches,
        logger=logger if args.verbose else None,
    )

    package = {
        "fisher": fisher,
        "metadata": {
            "folder": str(folder),
            "checkpoint": str(checkpoint_path),
            "batches_processed": processed,
            "max_batches": args.max_batches,
            "subset": args.subset,
        }
    }
    torch.save(package, fisher_path)
    return fisher


def main():
    args = parse_args()
    logger = setup_logger(args.verbose)

    folders = [Path(p) for p in args.folders]
    for folder in folders:
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder}")
        checkpoint_path = folder / args.checkpoint_name
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata_path = Path(args.metadata) if args.metadata else output_path.with_suffix(".json")

    config_path, cfg = load_appae_training_config(args.config, args.dataset_name)
    batch_size = args.batch_size or cfg.batch_size
    num_workers = args.num_workers or cfg.num_workers
    seed = args.seed or cfg.seed

    device_str = args.device or cfg.device
    if device_str:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Dataset: %s | UVAD mode: %s", args.dataset_name, args.uvadmode)
    logger.info("Folders: %s", ", ".join(str(f) for f in folders))
    logger.info("Output checkpoint: %s", output_path)
    logger.info("Using device: %s", device)
    if config_path:
        logger.info("Config file: %s", config_path)

    dataloader, total_samples = build_patch_dataloader(
        args.dataset_name,
        args.uvadmode,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        subset=args.subset,
        seed=seed,
    )
    logger.info("Total samples available for Fisher computation: %d", total_samples)

    state_dicts = []
    fisher_dicts = []

    for folder in folders:
        checkpoint_path = folder / args.checkpoint_name
        state = torch.load(str(checkpoint_path), map_location="cpu")
        state_dicts.append(state)

        fisher_path = folder / args.fisher_name
        fisher = ensure_fisher(
            folder,
            fisher_path,
            cfg,
            args,
            logger,
            dataloader,
            device
        )
        fisher_dicts.append(fisher)

    coefficients = parse_coefficients(args.coefficients, len(folders), args.strategy)
    logger.info("Using coefficients: %s", coefficients)

    merged_state = fisher_weighted_average(state_dicts, fisher_dicts, coefficients, eps=args.eps)
    torch.save(merged_state, output_path)
    logger.info("Saved merged checkpoint to %s", output_path)

    metadata = {
        "dataset_name": args.dataset_name,
        "uvadmode": args.uvadmode,
        "folders": [str(f) for f in folders],
        "checkpoint_name": args.checkpoint_name,
        "fisher_name": args.fisher_name,
        "output": str(output_path),
        "config_path": str(config_path) if config_path else None,
        "coefficients": coefficients,
        "strategy": args.strategy,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "subset": args.subset,
        "max_batches": args.max_batches,
        "seed": seed,
        "device": str(device),
    }

    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    logger.info("Saved soup metadata to %s", metadata_path)


if __name__ == "__main__":
    main()
