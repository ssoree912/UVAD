#!/usr/bin/env python3
"""
Enhanced Fisher-weighted model soup for AppAE checkpoints with VAD_soup improvements.
Incorporates advanced features like proper Fisher weighting, mask handling, and coefficient optimization.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any

# Ensure project root is on sys.path
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from codes.tools.enhanced_appae_fisher_utils import (
    load_appae_training_config,
    build_patch_dataloader,
    load_appae_model,
    compute_fisher_for_appae,
    enhanced_fisher_weighted_average,
    load_masks,
    combine_masks,
    create_pairwise_grid_coeffs,
    create_random_coeffs,
    evaluate_fisher_soup_candidates
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enhanced Fisher-weighted soup for AppAE checkpoints.")
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
    
    # Enhanced Fisher soup parameters (from VAD_soup)
    parser.add_argument("--coefficients", type=str, default=None,
                        help="Comma-separated soup coefficients. If not provided, will search for optimal.")
    parser.add_argument("--coefficient_strategy", choices=["uniform", "grid", "random"], default="grid",
                        help="Strategy for generating coefficient candidates.")
    parser.add_argument("--n_coefficient_candidates", type=int, default=10,
                        help="Number of coefficient combinations to try.")
    parser.add_argument("--fisher_floor", type=float, default=1e-6,
                        help="Minimum Fisher value for numerical stability.")
    parser.add_argument("--favor_target_model", action="store_true",
                        help="Don't apply fisher_floor to the first (target) model.")
    parser.add_argument("--normalize_fishers", action="store_true", default=True,
                        help="Normalize Fisher information for stability.")
    parser.add_argument("--use_masks", action="store_true", default=True,
                        help="Load and apply pruning masks if available.")
    
    parser.add_argument("--recompute_fisher", action="store_true",
                        help="Force recomputation of Fisher information.")
    parser.add_argument("--eps", type=float, default=1e-8,
                        help="Stability epsilon when normalizing weights.")
    parser.add_argument("--metadata", default=None,
                        help="Optional JSON file to store soup metadata.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    return parser.parse_args()


def setup_logger(verbose: bool) -> logging.Logger:
    logger = logging.getLogger("enhanced_appae_fisher_soup")
    logger.handlers.clear()
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO if verbose else logging.WARNING)
    logger.propagate = False
    return logger


def parse_coefficients(raw: Optional[str], n: int) -> Optional[List[float]]:
    if not raw:
        return None
    parts = [float(x.strip()) for x in raw.split(",")]
    if len(parts) != n:
        raise ValueError("Number of coefficients must match number of folders.")
    total = sum(parts)
    if total == 0:
        raise ValueError("Coefficient sum must be non-zero.")
    return [p / total for p in parts]


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


def simple_evaluation_function(state_dict: Dict[str, torch.Tensor]) -> float:
    """
    Simple evaluation function that computes a proxy score.
    In practice, you would replace this with actual model evaluation.
    """
    # Simple proxy: L2 norm of parameters (could be replaced with actual validation)
    total_norm = 0.0
    for param in state_dict.values():
        total_norm += torch.norm(param.float()).item()
    return -total_norm  # Negative because we want smaller norms to be better


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

    # Load state dictionaries
    state_dicts = []
    checkpoint_paths = []
    for folder in folders:
        checkpoint_path = folder / args.checkpoint_name
        state = torch.load(str(checkpoint_path), map_location="cpu")
        state_dicts.append(state)
        checkpoint_paths.append(str(checkpoint_path))

    # Load Fisher information
    fisher_dicts = []
    for folder in folders:
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

    # Load masks if requested
    masks = None
    if args.use_masks:
        masks = load_masks(state_dicts, checkpoint_paths)
        logger.info("Loaded masks for %d models", sum(1 for m in masks if m is not None))

    # Determine coefficients
    manual_coeffs = parse_coefficients(args.coefficients, len(folders))
    if manual_coeffs is not None:
        logger.info("Using manual coefficients: %s", manual_coeffs)
        # Direct merge with specified coefficients
        merged_state = enhanced_fisher_weighted_average(
            state_dicts=state_dicts,
            fisher_dicts=fisher_dicts,
            coefficients=manual_coeffs,
            fisher_floor=args.fisher_floor,
            favor_target_model=args.favor_target_model,
            normalize_fishers=args.normalize_fishers,
            eps=args.eps,
            masks=masks
        )
        best_coeffs = manual_coeffs
        best_score = None
    else:
        # Generate coefficient candidates
        if args.coefficient_strategy == "uniform":
            coefficient_candidates = [[1.0 / len(folders)] * len(folders)]
        elif args.coefficient_strategy == "grid" and len(folders) == 2:
            coefficient_candidates = create_pairwise_grid_coeffs(args.n_coefficient_candidates)
        else:
            coefficient_candidates = create_random_coeffs(
                len(folders), args.n_coefficient_candidates, seed
            )
        
        logger.info("Generated %d coefficient candidates using %s strategy", 
                   len(coefficient_candidates), args.coefficient_strategy)

        # Evaluate candidates and select best
        merged_state, best_coeffs, best_score = evaluate_fisher_soup_candidates(
            state_dicts=state_dicts,
            fisher_dicts=fisher_dicts,
            coefficient_candidates=coefficient_candidates,
            evaluation_fn=simple_evaluation_function,  # Replace with actual evaluation
            fisher_floor=args.fisher_floor,
            favor_target_model=args.favor_target_model,
            normalize_fishers=args.normalize_fishers,
            masks=masks,
            logger=logger
        )

    # Save merged checkpoint
    torch.save(merged_state, output_path)
    logger.info("Saved merged checkpoint to %s", output_path)

    # Prepare metadata
    metadata = {
        "dataset_name": args.dataset_name,
        "uvadmode": args.uvadmode,
        "folders": [str(f) for f in folders],
        "checkpoint_paths": checkpoint_paths,
        "checkpoint_name": args.checkpoint_name,
        "fisher_name": args.fisher_name,
        "output": str(output_path),
        "config_path": str(config_path) if config_path else None,
        "selected_coefficients": best_coeffs,
        "selected_score": best_score,
        "coefficient_strategy": args.coefficient_strategy,
        "n_coefficient_candidates": args.n_coefficient_candidates,
        "fisher_floor": args.fisher_floor,
        "favor_target_model": args.favor_target_model,
        "normalize_fishers": args.normalize_fishers,
        "use_masks": args.use_masks,
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

    logger.info("Enhanced Fisher soup completed successfully!")
    logger.info("Selected coefficients: %s", best_coeffs)
    if best_score is not None:
        logger.info("Best score: %.6f", best_score)


if __name__ == "__main__":
    main()