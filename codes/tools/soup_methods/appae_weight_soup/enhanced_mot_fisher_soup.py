#!/usr/bin/env python3
"""
Enhanced Fisher-weighted soup for motion (fGMM) checkpoints.

Computes Fisher information over motion feature banks, performs coefficient
search, and writes out a merged fGMM model (plus optional pseudo scores).
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Ensure project root is on sys.path
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from codes import featurebank
from codes.tools.soup_methods.appae_weight_soup.enhanced_appae_fisher_utils import (
    create_pairwise_grid_coeffs,
    create_random_coeffs,
)
from codes.tools.soup_methods.appae_weight_soup.enhanced_mot_fisher_utils import (
    FgmmFisherConfig,
    compute_fisher_for_fgmm,
    fgmm_to_state_dict,
    fisher_weighted_average_fgmm,
    load_fgmm_fisher,
    save_fgmm_fisher,
    state_dict_to_fgmm,
)
from sklearn.mixture import GaussianMixture
from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enhanced Fisher-weighted soup for fGMM checkpoints.")
    parser.add_argument("--folders", nargs="+", required=True,
                        help="Run directories containing fGMM checkpoints (fgmm_model.npz).")
    parser.add_argument("--dataset_name", required=True, choices=["shanghaitech", "avenue", "ped2"])
    parser.add_argument("--uvadmode", required=True, choices=["merge", "partial"])
    parser.add_argument("--output", required=True,
                        help="Output path for merged fGMM model (npz).")
    parser.add_argument("--score_output", default=None,
                        help="Optional npy path to save merged pseudo scores.")
    parser.add_argument("--model_name", default="fgmm_model.npz",
                        help="Filename of fGMM model inside each folder.")
    parser.add_argument("--fisher_name", default="fgmm_fisher.pt",
                        help="Filename to store/load Fisher tensors inside each folder.")
    parser.add_argument("--max_samples", type=int, default=1024,
                        help="Maximum motion samples used when computing Fisher. (None uses all)")
    parser.add_argument("--fisher_eps", type=float, default=1e-6,
                        help="Numerical jitter added to covariances during Fisher computation.")
    parser.add_argument("--recompute_fisher", action="store_true",
                        help="Force Fisher recomputation even if cached tensors exist.")
    parser.add_argument("--seed", type=int, default=111,
                        help="Random seed for sampling / coefficient search.")
    parser.add_argument("--coefficients", type=str, default=None,
                        help="Manual coefficients (comma-separated). Overrides search.")
    parser.add_argument("--coefficient_strategy", choices=["uniform", "grid", "random"], default="grid",
                        help="Coefficient generation strategy when not set manually.")
    parser.add_argument("--n_coefficient_candidates", type=int, default=15,
                        help="Number of coefficient candidates to evaluate.")
    parser.add_argument("--fisher_floor", type=float, default=1e-6,
                        help="Minimum Fisher value for stability.")
    parser.add_argument("--favor_target_model", action="store_true",
                        help="Skip Fisher floor on first model (baseline).")
    parser.add_argument("--normalize_fishers", action="store_true", default=True,
                        help="Normalize Fisher tensors before weighting.")
    parser.add_argument("--eval_subset", type=int, default=None,
                        help="Number of motion samples to evaluate likelihood on (None uses all).")
    parser.add_argument("--metadata", default=None,
                        help="Optional JSON path for recording soup metadata.")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging.")
    return parser.parse_args()


def setup_logger(verbose: bool) -> logging.Logger:
    logger = logging.getLogger("enhanced_mot_fisher_soup")
    logger.handlers.clear()
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO if verbose else logging.WARNING)
    logger.propagate = False
    return logger


def resolve_coefficients(raw: Optional[str], n_models: int) -> Optional[List[float]]:
    if raw is None:
        return None
    values = [float(s.strip()) for s in raw.split(",")]
    if len(values) != n_models:
        raise ValueError("Number of coefficients must match the number of folders.")
    total = sum(values)
    if total == 0:
        raise ValueError("Coefficient sum must be non-zero.")
    return [v / total for v in values]


def load_fgmm_model(path: Path) -> GaussianMixture:
    data = np.load(path, allow_pickle=True)
    covariance_type = str(data["covariance_type"])
    gmm = GaussianMixture(n_components=len(data["weights"]), covariance_type=covariance_type)
    gmm.weights_ = data["weights"]
    gmm.means_ = data["means"]
    gmm.covariances_ = data["covariances"]
    gmm.precisions_cholesky_ = _compute_precision_cholesky(gmm.covariances_, covariance_type)
    gmm.converged_ = bool(data.get("converged", True))
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


def evaluate_fgmm_candidates(
    state_dicts: List[Dict[str, torch.Tensor]],
    fisher_dicts: List[Dict[str, torch.Tensor]],
    coefficient_candidates: List[List[float]],
    eval_features: np.ndarray,
    *,
    fisher_floor: float,
    favor_target_model: bool,
    normalize_fishers: bool,
    logger: Optional[logging.Logger] = None,
) -> Tuple[Dict[str, torch.Tensor], List[float], float]:
    best_state = None
    best_coeffs = None
    best_score = float("-inf")

    for idx, coeffs in enumerate(coefficient_candidates):
        merged_state = fisher_weighted_average_fgmm(
            state_dicts,
            fisher_dicts,
            coeffs,
            fisher_floor=fisher_floor,
            favor_target_model=favor_target_model,
            normalize_fishers=normalize_fishers,
        )
        gmm = state_dict_to_fgmm(merged_state)
        mean_ll = float(np.mean(gmm.score_samples(eval_features)))
        if logger:
            logger.info("Candidate %d | coeffs=%s | mean log-lik=%.6f", idx + 1, coeffs, mean_ll)
        if mean_ll > best_score:
            best_score = mean_ll
            best_coeffs = coeffs
            best_state = merged_state

    if logger and best_coeffs is not None:
        logger.info("Selected coefficients=%s (mean log-lik=%.6f)", best_coeffs, best_score)

    return best_state, best_coeffs, best_score


def main():
    args = parse_args()
    logger = setup_logger(args.verbose)

    folders = [Path(p) for p in args.folders]
    for folder in folders:
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata_path = Path(args.metadata) if args.metadata else output_path.with_suffix(".json")

    # Load motion features once (float64 for numerical stability in likelihood)
    motion_features = featurebank.get(args.dataset_name, "mot", "train", uvadmode=args.uvadmode).astype(np.float64)
    if motion_features.size == 0:
        raise RuntimeError("No motion features available for Fisher/evaluation.")

    eval_features = motion_features
    if args.eval_subset is not None and args.eval_subset < motion_features.shape[0]:
        rng = np.random.default_rng(args.seed)
        indices = rng.choice(motion_features.shape[0], size=args.eval_subset, replace=False)
        eval_features = motion_features[indices]
        logger.info("Evaluation subset: %d/%d samples", eval_features.shape[0], motion_features.shape[0])
    else:
        logger.info("Using all %d samples for evaluation.", eval_features.shape[0])

    fisher_config = FgmmFisherConfig(
        max_samples=args.max_samples,
        seed=args.seed,
        eps=args.fisher_eps
    )

    state_dicts: List[Dict[str, torch.Tensor]] = []
    fisher_dicts: List[Dict[str, torch.Tensor]] = []
    model_paths: List[str] = []

    for folder in folders:
        candidate_paths = [
            folder / args.model_name,
            folder.with_name(folder.name + "_mot") / args.model_name,
        ]
        model_path = next((p for p in candidate_paths if p.exists()), None)
        if model_path is None:
            raise FileNotFoundError(f"fGMM model not found for folder={folder}")
        gmm = load_fgmm_model(model_path)
        state_dicts.append(fgmm_to_state_dict(gmm))
        model_paths.append(str(model_path))

        fisher_path = model_path.parent / args.fisher_name
        fisher = None
        if fisher_path.exists() and not args.recompute_fisher:
            fisher = load_fgmm_fisher(fisher_path)

        if fisher is None or args.recompute_fisher:
            logger.info("Computing Fisher for %s", model_path.parent)
            fisher = compute_fisher_for_fgmm(
                gmm,
                motion_features,
                config=fisher_config,
                logger=logger if args.verbose else None,
            )
            save_fgmm_fisher(fisher, fisher_path, metadata={
                "folder": str(folder),
                "model_path": str(model_path),
                "max_samples": fisher_config.max_samples,
                "seed": fisher_config.seed,
                "eps": fisher_config.eps,
                "total_features": int(motion_features.shape[0]),
            })
        fisher_dicts.append(fisher)

    manual_coeffs = resolve_coefficients(args.coefficients, len(folders))

    if manual_coeffs is not None:
        logger.info("Using manual coefficients: %s", manual_coeffs)
        best_state = fisher_weighted_average_fgmm(
            state_dicts,
            fisher_dicts,
            manual_coeffs,
            fisher_floor=args.fisher_floor,
            favor_target_model=args.favor_target_model,
            normalize_fishers=args.normalize_fishers,
        )
        best_coeffs = manual_coeffs
        best_score = float(np.mean(state_dict_to_fgmm(best_state).score_samples(eval_features)))
    else:
        if args.coefficient_strategy == "uniform":
            candidates = [[1.0 / len(folders)] * len(folders)]
        elif args.coefficient_strategy == "grid" and len(folders) == 2:
            candidates = create_pairwise_grid_coeffs(args.n_coefficient_candidates)
        else:
            candidates = create_random_coeffs(len(folders), args.n_coefficient_candidates, args.seed)
        logger.info("Generated %d coefficient candidates.", len(candidates))

        best_state, best_coeffs, best_score = evaluate_fgmm_candidates(
            state_dicts,
            fisher_dicts,
            candidates,
            eval_features,
            fisher_floor=args.fisher_floor,
            favor_target_model=args.favor_target_model,
            normalize_fishers=args.normalize_fishers,
            logger=logger if args.verbose else None,
        )

    merged_gmm = state_dict_to_fgmm(best_state)
    save_fgmm_model(merged_gmm, output_path, logger)
    logger.info("Selected coefficients: %s", best_coeffs)
    logger.info("Mean log-likelihood on eval set: %.6f", best_score)

    if args.score_output:
        score_path = Path(args.score_output)
        score_path.parent.mkdir(parents=True, exist_ok=True)
        mot_scores = -merged_gmm.score_samples(motion_features)
        np.save(score_path, mot_scores.astype(np.float32))
        logger.info("Saved merged pseudo scores to %s", score_path)

    metadata = {
        "dataset_name": args.dataset_name,
        "uvadmode": args.uvadmode,
        "folders": [str(f) for f in folders],
        "model_paths": model_paths,
        "output_model": str(output_path),
        "score_output": args.score_output,
        "selected_coefficients": best_coeffs,
        "selected_score": best_score,
        "fisher_floor": args.fisher_floor,
        "favor_target_model": args.favor_target_model,
        "normalize_fishers": args.normalize_fishers,
        "max_samples": args.max_samples,
        "fisher_eps": args.fisher_eps,
        "recompute_fisher": args.recompute_fisher,
        "seed": args.seed,
        "coefficient_strategy": args.coefficient_strategy,
        "n_coefficient_candidates": args.n_coefficient_candidates,
        "eval_subset": args.eval_subset,
    }

    if metadata_path:
        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
        logger.info("Saved metadata to %s", metadata_path)


if __name__ == "__main__":
    main()
