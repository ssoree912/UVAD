#!/usr/bin/env python3
"""
End-to-end pipeline for AppAE Fisher soup:
1. Merge multiple checkpoints with Fisher-weighted averaging.
2. Place the merged checkpoint in a target run directory.
3. Recompute AppAE scores (mode=app) using the merged weights.
4. (Optional) Recompute MOT scores (mode=mot).
5. (Optional) Run AUROC evaluation via main2_evaluate.py (supports evaluating source runs + soup).
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure project root on sys.path
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from codes.cleanse import AppAE, Cleanse, PatchDataset
from codes.tools.appae_fisher_utils import (
    load_appae_training_config,
    build_patch_dataloader,
    load_appae_model,
    compute_fisher_for_appae,
    fisher_weighted_average,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AppAE Fisher soup and evaluation pipeline.")
    parser.add_argument("--folders", nargs="+", required=True,
                        help="Run directories containing AppAE checkpoints (appaerecon_best.pkl).")
    parser.add_argument("--dataset_name", required=True, choices=["shanghaitech", "avenue", "ped2"])
    parser.add_argument("--uvadmode", required=True, choices=["merge", "partial"])

    parser.add_argument("--config", default=None,
                        help="Training config YAML (defaults to configs/config_<dataset>.yaml).")
    parser.add_argument("--checkpoint_name", default="appaerecon_best.pkl",
                        help="Checkpoint filename present in each folder.")
    parser.add_argument("--fisher_name", default="appaerecon_fisher.pt",
                        help="Filename to cache Fisher tensors inside each folder.")
    parser.add_argument("--target_folder", default=None,
                        help="Folder where merged checkpoint will be stored. "
                             "Defaults to artifacts/<dataset>/<uvadmode>/soup_run.")
    parser.add_argument("--output_checkpoint", default=None,
                        help="Filename for merged checkpoint (default: appaerecon_best.pkl in target folder).")
    parser.add_argument("--overwrite_target", action="store_true",
                        help="Overwrite existing checkpoint in target folder.")

    parser.add_argument("--coefficients", type=str, default=None,
                        help="Comma-separated coefficients for soup (default: uniform).")
    parser.add_argument("--subset", type=int, default=None,
                        help="Optional number of patches sampled for Fisher computation.")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size during Fisher computation.")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Override dataloader workers during Fisher computation.")
    parser.add_argument("--max_batches", type=int, default=50,
                        help="Max batches when computing Fisher (0 = unlimited).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for subset sampling/Fisher.")
    parser.add_argument("--device", default=None,
                        help="Device override for Fisher/inference (cpu, cuda, cuda:0).")
    parser.add_argument("--recompute_fisher", action="store_true",
                        help="Force recomputation of Fisher information for each folder.")
    parser.add_argument("--eps", type=float, default=1e-8,
                        help="Stability epsilon for Fisher-weighted averaging.")

    parser.add_argument("--skip_inference", action="store_true",
                        help="Skip generating AppAE scores (mode=app).")
    parser.add_argument("--score_output_template", default="features/{dataset}/cleansescores/{mode}_aerecon_{label}.npy",
                        help="Template for generated AppAE scores. Available placeholders: "
                             "{dataset}, {mode}, {label}, {uvadmode}.")
    parser.add_argument("--inference_batch_size", type=int, default=None,
                        help="Batch size when recomputing scores (default: use config batch_size).")
    parser.add_argument("--inference_workers", type=int, default=None,
                        help="Num workers for inference (default: use config num_workers).")

    parser.add_argument("--run_mot", action="store_true",
                        help="After soup, run main1_pseudoanomaly.py --mode mot to refresh MOT scores.")
    parser.add_argument("--mot_gmm_n", type=int, default=12,
                        help="Number of GMM components when running MOT stage.")

    parser.add_argument("--run_eval", action="store_true",
                        help="Run main2_evaluate.py to compute AUROC after scores are generated.")
    parser.add_argument("--eval_config", default=None,
                        help="Config YAML for evaluation (defaults to --config).")
    parser.add_argument("--eval_mode", default=None, choices=["merge", "partial"],
                        help="Mode passed to evaluation (default: same as --uvadmode).")
    parser.add_argument("--eval_include_folders", action="store_true",
                        help="When --run_eval, also evaluate each source folder in addition to the soup.")

    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging.")
    return parser.parse_args()


def setup_logger(verbose: bool) -> logging.Logger:
    logger = logging.getLogger("run_appae_soup_pipeline")
    logger.handlers.clear()
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO if verbose else logging.WARNING)
    logger.propagate = False
    return logger


def parse_coefficients(raw: Optional[str], n: int) -> List[float]:
    if raw:
        coeffs = [float(x.strip()) for x in raw.split(",")]
        if len(coeffs) != n:
            raise ValueError("Number of coefficients must match number of folders.")
        total = sum(coeffs)
        if total == 0:
            raise ValueError("Coefficient sum must be non-zero.")
        return [c / total for c in coeffs]
    return [1.0 / n] * n


def ensure_fisher(folder: Path,
                  fisher_path: Path,
                  checkpoint_name: str,
                  cfg,
                  args,
                  logger: logging.Logger,
                  device: torch.device,
                  dataloader,
                  seed: int):
    if fisher_path.exists() and not args.recompute_fisher:
        package = torch.load(str(fisher_path), map_location="cpu")
        return package["fisher"] if isinstance(package, dict) and "fisher" in package else package

    logger.info("Computing Fisher for %s", folder)
    checkpoint_path = folder / checkpoint_name
    model = load_appae_model(checkpoint_path, device)
    fisher, processed = compute_fisher_for_appae(
        model,
        dataloader,
        device=device,
        max_batches=None if args.max_batches == 0 else args.max_batches,
        logger=logger if args.verbose else None,
    )

    torch.save(
        {
            "fisher": fisher,
            "metadata": {
                "folder": str(folder),
                "checkpoint": str(checkpoint_path),
                "batches_processed": processed,
                "max_batches": args.max_batches,
                "subset": args.subset,
                "seed": seed,
            }
        },
        fisher_path
    )
    return fisher


def run_appae_inference(dataset_name: str,
                        uvadmode: str,
                        checkpoint_path: Path,
                        output_path: Path,
                        batch_size: int,
                        num_workers: int,
                        device: torch.device,
                        logger: logging.Logger):
    logger.info("Running AppAE inference for soup checkpoint.")
    helper = Cleanse(dataset_name, uvadmode)
    paths = helper.get_app_fpaths()
    if not paths:
        raise RuntimeError(f"No patch files available for dataset={dataset_name}, uvadmode={uvadmode}.")

    dataset = PatchDataset(paths)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    model = AppAE().to(device)
    state = torch.load(str(checkpoint_path), map_location=device)
    model.load_state_dict(state)
    model.eval()

    scores = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Soup inference"):
            batch = batch.to(device)
            recon = model(batch)
            loss = ((recon - batch) ** 2).mean(dim=(1, 2, 3))
            scores.append(loss.cpu().numpy())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, np.concatenate(scores))
    logger.info("Saved AppAE scores to %s", output_path)


def run_mot_stage(dataset_name: str,
                  uvadmode: str,
                  seed: int,
                  save_root: str,
                  log_dir: str,
                  gmm_n: int,
                  logger: logging.Logger):
    logger.info("Running MOT stage to refresh velo_fgmm scores.")
    cmd = [
        sys.executable,
        "main1_pseudoanomaly.py",
        "--dataset_name", dataset_name,
        "--uvadmode", uvadmode,
        "--mode", "mot",
        "--seed", str(seed),
        "--save_root", save_root,
        "--log_dir", log_dir,
        "--gmm_n", str(gmm_n),
        "--no_stream_logs",
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to run MOT stage ({result.returncode}).")
    logger.info("MOT stage completed.")


def run_evaluation(dataset_name: str,
                   mode: str,
                   eval_config: str,
                   logger: logging.Logger) -> Tuple[str, Optional[float]]:
    logger.info("Running evaluation via main2_evaluate.py.")
    cmd = [
        sys.executable,
        "main2_evaluate.py",
        "--config", eval_config,
        "--dataset_name", dataset_name,
        "--mode", mode,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("Evaluation failed:\n%s", result.stderr)
        raise RuntimeError(f"Evaluation failed ({result.returncode}).")
    stdout = result.stdout.strip()
    if stdout:
        logger.info(stdout)
    pattern = rf"AUROC {dataset_name} \({mode}\):\s*([0-9]+\.?[0-9]*)"
    match = re.search(pattern, stdout)
    auroc = float(match.group(1)) if match else None
    return stdout, auroc


def main():
    args = parse_args()
    logger = setup_logger(args.verbose)

    folders = [Path(p) for p in args.folders]
    for folder in folders:
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder}")
        checkpoint_path = folder / args.checkpoint_name
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint {args.checkpoint_name} not found in {folder}")

    config_path, cfg = load_appae_training_config(args.config, args.dataset_name)
    batch_size = args.batch_size or cfg.batch_size
    num_workers = args.num_workers or cfg.num_workers
    seed = args.seed or cfg.seed

    device_str = args.device or cfg.device
    if device_str:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Dataset: %s | UVAD mode: %s | Device: %s", args.dataset_name, args.uvadmode, device)
    if config_path:
        logger.info("Training config: %s", config_path)
    logger.info("Folders to merge: %s", ", ".join(str(f) for f in folders))

    dataloader, total_samples = build_patch_dataloader(
        args.dataset_name,
        args.uvadmode,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        subset=args.subset,
        seed=seed,
    )
    logger.info("Samples available for Fisher computation: %d", total_samples)

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
            args.checkpoint_name,
            cfg,
            args,
            logger,
            device,
            dataloader,
            seed
        )
        fisher_dicts.append(fisher)

    coefficients = parse_coefficients(args.coefficients, len(folders))
    logger.info("Soup coefficients: %s", coefficients)
    merged_state = fisher_weighted_average(state_dicts, fisher_dicts, coefficients, eps=args.eps)

    target_folder = Path(args.target_folder) if args.target_folder else (
        Path(cfg.save_root) / args.dataset_name / args.uvadmode / "soup_run"
    )
    target_folder.mkdir(parents=True, exist_ok=True)

    output_checkpoint = Path(args.output_checkpoint) if args.output_checkpoint else (
        target_folder / args.checkpoint_name
    )
    if output_checkpoint.exists() and not args.overwrite_target:
        raise FileExistsError(f"Output checkpoint already exists: {output_checkpoint} "
                              "(use --overwrite_target to overwrite).")

    torch.save(merged_state, output_checkpoint)
    logger.info("Saved merged checkpoint to %s", output_checkpoint)

    metadata = {
        "dataset_name": args.dataset_name,
        "uvadmode": args.uvadmode,
        "folders": [str(f) for f in folders],
        "coefficients": coefficients,
        "checkpoint_name": args.checkpoint_name,
        "fisher_name": args.fisher_name,
        "target_folder": str(target_folder),
        "output_checkpoint": str(output_checkpoint),
        "subset": args.subset,
        "max_batches": args.max_batches,
        "seed": seed,
        "device": str(device),
    }
    with open(target_folder / "soup_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    score_template = args.score_output_template

    if not args.skip_inference:
        inference_batch_size = args.inference_batch_size or batch_size
        inference_workers = args.inference_workers or num_workers
        label = Path(target_folder).name
        score_path_str = score_template.format(
            dataset=args.dataset_name,
            mode=args.uvadmode,
            label=label,
            uvadmode=args.uvadmode
        )
        scores_path = Path(score_path_str)
        run_appae_inference(
            args.dataset_name,
            args.uvadmode,
            output_checkpoint,
            scores_path,
            inference_batch_size,
            inference_workers,
            device,
            logger
        )

    evaluation_summary = []

    if args.run_mot:
        run_mot_stage(
            args.dataset_name,
            args.uvadmode,
            seed,
            cfg.save_root,
            cfg.log_dir,
            args.mot_gmm_n,
            logger
        )

    if args.run_eval:
        eval_config = args.eval_config or (str(config_path) if config_path else None)
        if not eval_config:
            raise ValueError("Evaluation requested but no config provided (use --eval_config).")
        eval_mode = args.eval_mode or args.uvadmode

        evaluation_targets: List[Tuple[str, Path]] = []
        if args.eval_include_folders:
            for folder in folders:
                label = Path(folder).name
                evaluation_targets.append((label, folder / args.checkpoint_name))
        evaluation_targets.append((Path(target_folder).name, output_checkpoint))

        inference_batch_size = args.inference_batch_size or batch_size
        inference_workers = args.inference_workers or num_workers

        for label, ckpt_path in evaluation_targets:
            logger.info("Evaluating run '%s' using checkpoint %s", label, ckpt_path)
            score_path_str = score_template.format(
                dataset=args.dataset_name,
                mode=eval_mode,
                label=label,
                uvadmode=args.uvadmode
            )
            scores_path = Path(score_path_str)
            run_appae_inference(
                args.dataset_name,
                eval_mode,
                ckpt_path,
                scores_path,
                inference_batch_size,
                inference_workers,
                device,
                logger
            )
            stdout, auroc = run_evaluation(
                args.dataset_name,
                eval_mode,
                eval_config,
                logger
            )
            evaluation_summary.append({
                "label": label,
                "checkpoint": str(ckpt_path),
                "auroc": auroc,
                "raw_output": stdout,
            })

    if evaluation_summary:
        logger.info("Evaluation summary:")
        for item in evaluation_summary:
            logger.info("  %s -> AUROC: %s", item["label"],
                        f"{item['auroc']:.3f}" if item["auroc"] is not None else "N/A")
        with open(target_folder / "evaluation_summary.json", "w", encoding="utf-8") as handle:
            json.dump(evaluation_summary, handle, indent=2)

    logger.info("AppAE soup pipeline completed successfully.")


if __name__ == "__main__":
    main()
