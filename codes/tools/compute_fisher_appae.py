#!/usr/bin/env python3
"""
Compute Fisher information for AppAE checkpoints in the CKNN project.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

# Ensure project root is on sys.path
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from codes.tools.appae_fisher_utils import (
    load_appae_training_config,
    build_patch_dataloader,
    load_appae_model,
    compute_fisher_for_appae,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Fisher information for AppAE checkpoints.")
    parser.add_argument("--dataset_name", required=True, choices=["shanghaitech", "avenue", "ped2"])
    parser.add_argument("--uvadmode", required=True, choices=["merge", "partial"])
    parser.add_argument("--run_dir", required=True, help="Directory containing appaerecon checkpoints.")
    parser.add_argument("--checkpoint_name", default="appaerecon_best.pkl",
                        help="Checkpoint filename relative to run_dir.")
    parser.add_argument("--output", default=None,
                        help="Output path for Fisher tensor (defaults to run_dir/appaerecon_fisher.pt).")
    parser.add_argument("--config", default=None,
                        help="Training config YAML (defaults to configs/config_<dataset>.yaml).")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size during Fisher computation.")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Override dataloader worker count.")
    parser.add_argument("--max_batches", type=int, default=50,
                        help="Maximum batches to use for Fisher estimation (0 means unlimited).")
    parser.add_argument("--subset", type=int, default=None,
                        help="Optional number of patches to sample for Fisher estimation.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override random seed used for subset sampling.")
    parser.add_argument("--device", default=None,
                        help="Device override (cpu, cuda, cuda:0).")
    parser.add_argument("--overwrite", action="store_true", help="Recompute even if output exists.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    return parser.parse_args()


def main():
    args = parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    checkpoint_path = run_dir / args.checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    output_path = Path(args.output) if args.output else run_dir / "appaerecon_fisher.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not args.overwrite:
        print(f"[INFO] Fisher file already exists at {output_path}. Use --overwrite to recompute.")
        return

    logger = logging.getLogger("compute_fisher_appae")
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO if args.verbose else logging.WARNING)
    logger.propagate = False

    config_path, cfg = load_appae_training_config(args.config, args.dataset_name)
    batch_size = args.batch_size or cfg.batch_size
    num_workers = args.num_workers or cfg.num_workers
    seed = args.seed or cfg.seed

    logger.info("Dataset: %s | UVAD mode: %s", args.dataset_name, args.uvadmode)
    logger.info("Run directory: %s", run_dir)
    logger.info("Checkpoint: %s", checkpoint_path)
    logger.info("Output path: %s", output_path)
    if config_path:
        logger.info("Config file: %s", config_path)
    logger.info("Batch size: %d | Num workers: %d | Seed: %d | Max batches: %s",
                batch_size, num_workers, seed,
                "unlimited" if args.max_batches == 0 else str(args.max_batches))
    if args.subset:
        logger.info("Patch subset size: %d", args.subset)

    device_str = args.device or cfg.device
    if device_str:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    dataloader, total_samples = build_patch_dataloader(
        args.dataset_name,
        args.uvadmode,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        subset=args.subset,
        seed=seed,
    )
    logger.info("Total samples considered for Fisher: %d", total_samples)

    model = load_appae_model(checkpoint_path, device)
    fisher, batches_processed = compute_fisher_for_appae(
        model,
        dataloader,
        device=device,
        max_batches=None if args.max_batches == 0 else args.max_batches,
        logger=logger if args.verbose else None,
    )

    package = {
        "fisher": fisher,
        "metadata": {
            "dataset_name": args.dataset_name,
            "uvadmode": args.uvadmode,
            "run_dir": str(run_dir),
            "checkpoint": str(checkpoint_path),
            "config_path": str(config_path) if config_path else None,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "subset": args.subset,
            "seed": seed,
            "max_batches": args.max_batches,
            "batches_processed": batches_processed,
            "total_samples": total_samples,
            "device": str(device),
        }
    }
    torch.save(package, output_path)
    logger.info("Saved Fisher information to %s", output_path)

    summary_path = output_path.with_suffix(".json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(package["metadata"], handle, indent=2)
    logger.info("Saved metadata summary to %s", summary_path)


if __name__ == "__main__":
    main()
