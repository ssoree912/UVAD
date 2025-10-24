#!/usr/bin/env python3
"""
Enhanced end-to-end pipeline for AppAE Fisher soup with VAD_soup improvements:
1. Merge multiple checkpoints with enhanced Fisher-weighted averaging
2. Support for pruning masks and advanced coefficient optimization
3. Automatic evaluation and best model selection
4. Comprehensive metadata tracking
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure project root on sys.path
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.mixture import GaussianMixture
from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky

from codes.cleanse import AppAE, Cleanse, PatchDataset
from codes import featurebank
from codes.tools.soup_methods.appae_weight_soup.enhanced_appae_fisher_utils import (
    load_appae_training_config,
    build_patch_dataloader,
    load_appae_model,
    compute_fisher_for_appae,
    enhanced_fisher_weighted_average,
    load_masks,
    create_pairwise_grid_coeffs,
    create_random_coeffs,
    evaluate_fisher_soup_candidates
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enhanced AppAE Fisher soup pipeline with evaluation.")
    parser.add_argument("--folders", nargs="+", required=True,
                        help="Run directories containing AppAE checkpoints.")
    parser.add_argument("--dataset_name", required=True, choices=["shanghaitech", "avenue", "ped2"])
    parser.add_argument("--uvadmode", required=True, choices=["merge", "partial"])

    parser.add_argument("--config", default=None,
                        help="Training config YAML.")
    parser.add_argument("--checkpoint_name", default="appaerecon_best.pkl",
                        help="Checkpoint filename.")
    parser.add_argument("--fisher_name", default="appaerecon_fisher.pt",
                        help="Fisher information filename.")
    parser.add_argument("--target_folder", default=None,
                        help="Target folder for merged checkpoint.")
    parser.add_argument("--output_checkpoint", default=None,
                        help="Output checkpoint filename.")
    parser.add_argument("--overwrite_target", action="store_true",
                        help="Overwrite existing files.")

    # Enhanced Fisher soup parameters
    parser.add_argument("--coefficients", type=str, default=None,
                        help="Manual coefficients (comma-separated).")
    parser.add_argument("--coefficient_strategy", choices=["uniform", "grid", "random"], default="grid",
                        help="Coefficient generation strategy.")
    parser.add_argument("--n_coefficient_candidates", type=int, default=15,
                        help="Number of coefficient combinations to try.")
    parser.add_argument("--fisher_floor", type=float, default=1e-6,
                        help="Minimum Fisher value.")
    parser.add_argument("--favor_target_model", action="store_true", default=True,
                        help="Don't apply fisher_floor to first model.")
    parser.add_argument("--normalize_fishers", action="store_true", default=True,
                        help="Normalize Fisher information.")
    parser.add_argument("--use_masks", action="store_true", default=True,
                        help="Use pruning masks if available.")

    parser.add_argument("--subset", type=int, default=None,
                        help="Patch subset for Fisher computation.")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size override.")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Worker count override.")
    parser.add_argument("--max_batches", type=int, default=50,
                        help="Max batches for Fisher.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed.")
    parser.add_argument("--device", default=None,
                        help="Device override.")
    parser.add_argument("--recompute_fisher", action="store_true",
                        help="Force Fisher recomputation.")
    parser.add_argument("--eps", type=float, default=1e-8,
                        help="Numerical stability epsilon.")

    # Evaluation parameters
    parser.add_argument("--skip_inference", action="store_true",
                        help="Skip AppAE score generation.")
    parser.add_argument("--score_output_template", 
                        default="features/{dataset}/cleansescores/{mode}_aerecon_{label}_flat.npy",
                        help="Score output template.")
    parser.add_argument("--inference_batch_size", type=int, default=None,
                        help="Inference batch size.")
    parser.add_argument("--inference_workers", type=int, default=None,
                        help="Inference workers.")

    parser.add_argument("--run_mot", action="store_true",
                        help="Run MOT stage after soup.")
    parser.add_argument("--mot_gmm_n", type=int, default=12,
                        help="GMM components for MOT.")
    parser.add_argument("--mot_model_name", default="fgmm_model.npz",
                        help="Filename for stored fGMM models inside each run directory.")
    parser.add_argument("--mot_score_output_template",
                        default="features/{dataset}/cleansescores/{mode}_velo_fgmm_{label}_flat.npy",
                        help="Template for MOT pseudo score output path.")

    parser.add_argument("--run_eval", action="store_true",
                        help="Run evaluation.")
    parser.add_argument("--eval_config", default=None,
                        help="Evaluation config.")
    parser.add_argument("--eval_mode", default=None,
                        help="Evaluation mode.")
    parser.add_argument("--eval_include_folders", action="store_true",
                        help="Evaluate individual folders too.")

    parser.add_argument("--verbose", action="store_true", help="Verbose logging.")
    return parser.parse_args()


def setup_logger(verbose: bool) -> logging.Logger:
    logger = logging.getLogger("enhanced_run_appae_soup_pipeline")
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
    coeffs = [float(x.strip()) for x in raw.split(",")]
    if len(coeffs) != n:
        raise ValueError("Number of coefficients must match number of folders.")
    total = sum(coeffs)
    if total == 0:
        raise ValueError("Coefficient sum must be non-zero.")
    return [c / total for c in coeffs]


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
                        logger: logging.Logger) -> Path:
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
    return output_path


def load_fgmm_model(path: Path) -> GaussianMixture:
    data = np.load(path, allow_pickle=True)
    cov_type = str(data['covariance_type'])
    gmm = GaussianMixture(n_components=len(data['weights']), covariance_type=cov_type)
    gmm.weights_ = data['weights']
    gmm.means_ = data['means']
    gmm.covariances_ = data['covariances']
    gmm.precisions_cholesky_ = _compute_precision_cholesky(gmm.covariances_, cov_type)
    gmm.converged_ = bool(data.get('converged', True))
    gmm.n_features_in_ = gmm.means_.shape[1]
    return gmm


def save_fgmm_model(gmm: GaussianMixture, path: Path, logger: logging.Logger):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        weights=gmm.weights_,
        means=gmm.means_,
        covariances=gmm.covariances_,
        covariance_type=gmm.covariance_type,
        converged=gmm.converged_
    )
    logger.info("Saved merged fGMM model to %s", path)


def merge_fgmm_models(models: List[GaussianMixture], coeffs: List[float]) -> GaussianMixture:
    if not models or not coeffs:
        raise ValueError("No models or coefficients provided for fGMM merge.")

    if len(models) != len(coeffs):
        raise ValueError("Number of fGMM models must match coefficients.")

    n_components = models[0].weights_.shape[0]
    cov_type = models[0].covariance_type
    dim = models[0].means_.shape[1]

    weights = np.zeros(n_components, dtype=np.float64)
    means = np.zeros((n_components, dim), dtype=np.float64)
    covariances = np.zeros((n_components, dim, dim), dtype=np.float64)

    for coeff, model in zip(coeffs, models):
        weights += coeff * model.weights_
        means += coeff * model.means_
        covariances += coeff * model.covariances_

    weights = np.clip(weights, 1e-12, None)
    weights /= weights.sum()
    covariances = 0.5 * (covariances + np.transpose(covariances, (0, 2, 1)))

    merged = GaussianMixture(n_components=n_components, covariance_type=cov_type)
    merged.weights_ = weights
    merged.means_ = means
    merged.covariances_ = covariances
    merged.precisions_cholesky_ = _compute_precision_cholesky(covariances, cov_type)
    merged.converged_ = all(m.converged_ for m in models)
    merged.n_features_in_ = dim
    return merged


def run_mot_soup(dataset_name: str,
                 uvadmode: str,
                 folders: List[str],
                 coefficients: List[float],
                 target_folder: Path,
                 score_template: str,
                 model_name: str,
                 logger: logging.Logger) -> Optional[Path]:
    if not coefficients:
        logger.warning("Skipping MOT soup: no coefficients available.")
        return None

    mot_models = []
    missing = []
    for folder in folders:
        folder_path = Path(folder)
        candidate_paths = [
            folder_path / model_name,
            folder_path.with_name(folder_path.name + "_mot") / model_name
        ]
        model_path = next((p for p in candidate_paths if p.exists()), None)
        if model_path is None:
            missing.append(str(folder_path))
            continue
        try:
            mot_models.append(load_fgmm_model(model_path))
        except Exception as exc:
            logger.warning("Failed to load fGMM model from %s: %s", model_path, exc)
            missing.append(str(folder_path))

    if missing:
        logger.warning("MOT soup skipped due to missing models: %s", missing)
        return None

    if len(mot_models) != len(coefficients):
        logger.warning("MOT soup skipped: number of models (%d) != coefficients (%d)",
                       len(mot_models), len(coefficients))
        return None

    merged_gmm = merge_fgmm_models(mot_models, coefficients)

    mot_model_path = target_folder / model_name
    save_fgmm_model(merged_gmm, mot_model_path, logger)

    train_features = featurebank.get(dataset_name, 'mot', 'train', uvadmode=uvadmode).astype(np.float32)
    logger.info("Generating MOT pseudo scores on %d samples.", train_features.shape[0])
    mot_scores = -merged_gmm.score_samples(train_features)

    label = target_folder.name
    score_path = Path(score_template.format(
        dataset=dataset_name,
        mode=uvadmode,
        label=label,
        uvadmode=uvadmode
    ))
    score_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(score_path, mot_scores)
    logger.info("Saved MOT pseudo scores to %s", score_path)
    return score_path


def run_evaluation(dataset_name: str,
                   mode: str,
                   eval_config: str,
                   logger: logging.Logger,
                   override: Optional[str] = None) -> Tuple[str, Optional[float]]:
    logger.info("Running evaluation.")
    cmd = [
        sys.executable,
        "main2_evaluate.py",
        "--config", eval_config,
        "--dataset_name", dataset_name,
        "--mode", mode,
    ]
    if override:
        cmd += ["--override", override]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("Evaluation failed:\\n%s", result.stderr)
        raise RuntimeError(f"Evaluation failed ({result.returncode}).")
    stdout = result.stdout.strip()
    if stdout:
        logger.info(stdout)
    pattern = rf"AUROC {dataset_name} \\({mode}\\):\\s*([0-9]+\\.?[0-9]*)"
    match = re.search(pattern, stdout)
    auroc = float(match.group(1)) if match else None
    return stdout, auroc


class ModelEvaluator:
    """Enhanced model evaluator that can run actual AUROC evaluation."""
    
    def __init__(self, dataset_name: str, uvadmode: str, eval_config: str, 
                 target_folder: Path, device: torch.device, 
                 inference_batch_size: int, inference_workers: int,
                 score_template: str, logger: logging.Logger):
        self.dataset_name = dataset_name
        self.uvadmode = uvadmode
        self.eval_config = eval_config
        self.target_folder = target_folder
        self.device = device
        self.inference_batch_size = inference_batch_size
        self.inference_workers = inference_workers
        self.score_template = score_template
        self.logger = logger
        self.evaluation_count = 0
    
    def __call__(self, state_dict: Dict[str, torch.Tensor]) -> float:
        """Evaluate a state dict and return AUROC score."""
        try:
            # Save temporary checkpoint
            temp_checkpoint = self.target_folder / f"_temp_eval_{self.evaluation_count}.pkl"
            torch.save(state_dict, temp_checkpoint)
            
            # Generate scores
            label = f"temp_eval_{self.evaluation_count}"
            score_path_str = self.score_template.format(
                dataset=self.dataset_name,
                mode=self.uvadmode,
                label=label,
                uvadmode=self.uvadmode
            )
            scores_path = Path(score_path_str)
            
            run_appae_inference(
                self.dataset_name,
                self.uvadmode,
                temp_checkpoint,
                scores_path,
                self.inference_batch_size,
                self.inference_workers,
                self.device,
                self.logger
            )
            
            # Run evaluation
            override = json.dumps({
                "signals": {
                    "app": {
                        "cleanse_scorename": f"aerecon_{label}"
                    }
                }
            })
            
            _, auroc = run_evaluation(
                self.dataset_name,
                self.uvadmode,
                self.eval_config,
                self.logger,
                override=override
            )
            
            # Cleanup
            temp_checkpoint.unlink()
            if scores_path.exists():
                scores_path.unlink()
            
            self.evaluation_count += 1
            return auroc if auroc is not None else 0.0
            
        except Exception as e:
            self.logger.warning(f"Evaluation failed: {e}")
            return 0.0


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

    config_path, cfg = load_appae_training_config(args.config, args.dataset_name)
    batch_size = args.batch_size or cfg.batch_size
    num_workers = args.num_workers or cfg.num_workers
    seed = args.seed or cfg.seed

    device_str = args.device or cfg.device
    device = torch.device(device_str) if device_str else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    eval_config_path = args.eval_config or (str(config_path) if config_path else None)
    eval_mode_default = args.eval_mode or args.uvadmode

    logger.info("Dataset: %s | UVAD mode: %s | Device: %s", args.dataset_name, args.uvadmode, device)
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
    logger.info("Samples for Fisher computation: %d", total_samples)

    # Load state dictionaries and Fisher information
    state_dicts = []
    fisher_dicts = []
    checkpoint_paths = []
    
    for folder in folders:
        checkpoint_path = folder / args.checkpoint_name
        state = torch.load(str(checkpoint_path), map_location="cpu")
        state_dicts.append(state)
        checkpoint_paths.append(str(checkpoint_path))

        fisher_path = folder / args.fisher_name
        fisher = ensure_fisher(
            folder, fisher_path, args.checkpoint_name, cfg, args,
            logger, device, dataloader, seed
        )
        fisher_dicts.append(fisher)

    # Load masks
    masks = None
    if args.use_masks:
        masks = load_masks(state_dicts, checkpoint_paths)
        logger.info("Loaded masks for %d models", sum(1 for m in masks if m is not None))

    # Setup target folder
    target_folder = Path(args.target_folder) if args.target_folder else (
        Path(cfg.save_root) / args.dataset_name / args.uvadmode / "enhanced_soup_run"
    )
    target_folder.mkdir(parents=True, exist_ok=True)

    output_checkpoint = Path(args.output_checkpoint) if args.output_checkpoint else (
        target_folder / args.checkpoint_name
    )
    if output_checkpoint.exists() and not args.overwrite_target:
        raise FileExistsError(f"Output exists: {output_checkpoint} (use --overwrite_target)")

    # Determine coefficients and merge strategy
    manual_coeffs = parse_coefficients(args.coefficients, len(folders))
    inference_batch_size = args.inference_batch_size or batch_size
    inference_workers = args.inference_workers or num_workers
    
    if manual_coeffs is not None:
        logger.info("Using manual coefficients: %s", manual_coeffs)
        best_state = enhanced_fisher_weighted_average(
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

        # Setup evaluator
        if args.run_eval and eval_config_path:
            evaluator = ModelEvaluator(
                args.dataset_name, eval_mode_default, eval_config_path,
                target_folder, device, inference_batch_size, inference_workers,
                args.score_output_template, logger
            )
        else:
            # Fallback to simple evaluation
            def simple_evaluator(state_dict):
                total_norm = sum(torch.norm(p.float()).item() for p in state_dict.values())
                return -total_norm
            evaluator = simple_evaluator

        # Find best coefficients
        best_state, best_coeffs, best_score = evaluate_fisher_soup_candidates(
            state_dicts=state_dicts,
            fisher_dicts=fisher_dicts,
            coefficient_candidates=coefficient_candidates,
            evaluation_fn=evaluator,
            fisher_floor=args.fisher_floor,
            favor_target_model=args.favor_target_model,
            normalize_fishers=args.normalize_fishers,
            masks=masks,
            logger=logger
        )

    # Save final merged model
    torch.save(best_state, output_checkpoint)
    logger.info("Saved enhanced soup to %s", output_checkpoint)
    logger.info("Selected coefficients: %s", best_coeffs)
    if best_score is not None:
        logger.info("Best score: %.6f", best_score)

    # Save metadata
    metadata = {
        "dataset_name": args.dataset_name,
        "uvadmode": args.uvadmode,
        "folders": [str(f) for f in folders],
        "checkpoint_paths": checkpoint_paths,
        "selected_coefficients": best_coeffs,
        "selected_score": best_score,
        "fisher_floor": args.fisher_floor,
        "favor_target_model": args.favor_target_model,
        "normalize_fishers": args.normalize_fishers,
        "use_masks": args.use_masks,
        "coefficient_strategy": args.coefficient_strategy,
        "n_coefficient_candidates": args.n_coefficient_candidates,
        "seed": seed,
        "device": str(device),
        "score_output_template": args.score_output_template,
        "mot_score_output_template": args.mot_score_output_template,
        "mot_model_name": args.mot_model_name,
        "mot_score_path": None,
    }
    
    # Run inference and evaluation if requested
    label = Path(target_folder).name
    mot_score_path: Optional[Path] = None

    if not args.skip_inference:
        score_path_str = args.score_output_template.format(
            dataset=args.dataset_name,
            mode=args.uvadmode,
            label=label,
            uvadmode=args.uvadmode
        )
        scores_path = Path(score_path_str)
        run_appae_inference(
            args.dataset_name, args.uvadmode, output_checkpoint,
            scores_path, inference_batch_size, inference_workers,
            device, logger
        )

    if args.run_mot:
        mot_score_path = run_mot_soup(
            args.dataset_name,
            args.uvadmode,
            folders,
            best_coeffs,
            target_folder,
            args.mot_score_output_template,
            args.mot_model_name,
            logger
        )
        if mot_score_path is None:
            logger.warning("MOT soup was requested but could not be completed.")
        else:
            metadata["mot_score_path"] = str(mot_score_path)

    metadata_path = target_folder / "enhanced_soup_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    logger.info("Saved metadata to %s", metadata_path)

    if args.run_eval and eval_config_path:
        evaluation_results = []
        
        # Evaluate individual folders if requested
        if args.eval_include_folders:
            logger.info("Evaluating individual folders...")
            for folder in folders:
                folder_name = Path(folder).name
                individual_checkpoint = Path(folder) / args.checkpoint_name
                
                if individual_checkpoint.exists():
                    # Generate scores for individual model
                    individual_score_path_str = args.score_output_template.format(
                        dataset=args.dataset_name,
                        mode=eval_mode_default,
                        label=folder_name,
                        uvadmode=args.uvadmode
                    )
                    individual_scores_path = Path(individual_score_path_str)
                    
                    run_appae_inference(
                        args.dataset_name, eval_mode_default, individual_checkpoint,
                        individual_scores_path, inference_batch_size, inference_workers,
                        device, logger
                    )

                    mot_cleanse_name = f"velo_fgmm_{folder_name}"
                    mot_score_candidate = Path(args.mot_score_output_template.format(
                        dataset=args.dataset_name,
                        mode=eval_mode_default,
                        label=folder_name,
                        uvadmode=args.uvadmode
                    ))
                    if not mot_score_candidate.exists():
                        mot_cleanse_name = "velo_fgmm"

                    # Evaluate individual model
                    override = json.dumps({
                        "signals": {
                            "app": {
                                "cleanse_scorename": f"aerecon_{folder_name}"
                            },
                            "mot": {
                                "cleanse_scorename": mot_cleanse_name
                            }
                        }
                    })
                    
                    stdout, individual_auroc = run_evaluation(
                        args.dataset_name, eval_mode_default, eval_config_path,
                        logger, override=override
                    )
                    
                    evaluation_results.append({
                        "label": folder_name,
                        "checkpoint": str(individual_checkpoint),
                        "score_path": str(individual_scores_path),
                        "auroc": individual_auroc,
                        "raw_output": stdout.strip() if stdout else None
                    })
                    
                    if individual_auroc is not None:
                        logger.info("Individual %s AUROC: %.6f", folder_name, individual_auroc)
        
        # Evaluate soup model
        label = Path(target_folder).name
        score_path_str = args.score_output_template.format(
            dataset=args.dataset_name,
            mode=eval_mode_default,
            label=label,
            uvadmode=args.uvadmode
        )
        scores_path = Path(score_path_str)
        
        run_appae_inference(
            args.dataset_name, eval_mode_default, output_checkpoint,
            scores_path, inference_batch_size, inference_workers,
            device, logger
        )
        
        override = json.dumps({
            "signals": {
                "app": {
                    "cleanse_scorename": f"aerecon_{label}"
                },
                "mot": {
                    "cleanse_scorename": f"velo_fgmm_{label}" if mot_score_path and mot_score_path.exists() else "velo_fgmm"
                }
            }
        })
        
        stdout, final_auroc = run_evaluation(
            args.dataset_name, eval_mode_default, eval_config_path,
            logger, override=override
        )
        
        evaluation_results.append({
            "label": "soup_model",
            "checkpoint": str(output_checkpoint),
            "score_path": str(scores_path),
            "auroc": final_auroc,
            "raw_output": stdout.strip() if stdout else None
        })
        
        if final_auroc is not None:
            logger.info("Soup model AUROC: %.6f", final_auroc)
        
        # Save evaluation results
        eval_results_path = target_folder / "evaluation_results.json"
        with open(eval_results_path, "w", encoding="utf-8") as handle:
            json.dump(evaluation_results, handle, indent=2)
        logger.info("Saved evaluation results to %s", eval_results_path)

    logger.info("Enhanced AppAE soup pipeline completed successfully!")


if __name__ == "__main__":
    main()
