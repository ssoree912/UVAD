#!/usr/bin/env python3
"""
Enhanced end-to-end pipeline for AppAE + fGMM Fisher soup with VAD_soup improvements:
1. Merge multiple checkpoints with enhanced Fisher-weighted averaging for appearance
2. Support Fisher-guided motion (fGMM) soup with shared coefficient schedules
3. Automatic evaluation, best model selection, and optional MOT refresh
4. Comprehensive metadata tracking for both modalities
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
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import shlex
from datetime import datetime

# Ensure project root on sys.path
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[3]
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
from codes.tools.soup_methods.appae_weight_soup.enhanced_mot_fisher_utils import (
    FgmmFisherConfig,
    compute_fisher_for_fgmm,
    fgmm_to_state_dict,
    fisher_weighted_average_fgmm,
    load_fgmm_fisher,
    save_fgmm_fisher,
    state_dict_to_fgmm
)

# ---------- ECDF / Fisher-combine utilities for bank-compactness ----------
def _ecdf_pvals(scores: np.ndarray) -> np.ndarray:
    ranks = np.argsort(np.argsort(scores))
    cdf = (ranks + 1.0) / (len(scores) + 1.0)
    return 1.0 - cdf


def _fisher_combine_pvals(pvals_list: List[np.ndarray], eps: float = 1e-12) -> np.ndarray:
    Ps = np.clip(np.stack(pvals_list, axis=0), eps, 1.0)
    return -2.0 * np.sum(np.log(Ps), axis=0)


def create_anchor_biased_coeffs_local(n_models: int,
                                      n_candidates: int,
                                      anchor_idx: int,
                                      alpha_anchor: float,
                                      alpha_others: float,
                                      min_anchor: float = 0.5,
                                      seed: Optional[int] = None) -> List[List[float]]:
    """
    Sample Dirichlet coefficients while enforcing a strong bias toward an anchor model.
    """
    if not (0 <= anchor_idx < n_models):
        raise ValueError(f"anchor_idx {anchor_idx} out of range for n_models={n_models}")
    alphas = np.full(n_models, float(alpha_others), dtype=np.float64)
    alphas[anchor_idx] = float(alpha_anchor)
    rng = np.random.default_rng(None if seed is None else int(seed))
    coeffs: List[List[float]] = []
    max_attempts = max(10 * n_candidates, 100)
    attempts = 0
    while len(coeffs) < n_candidates and attempts < max_attempts:
        attempts += 1
        vec = rng.dirichlet(alphas).astype(np.float64)
        if vec[anchor_idx] < min_anchor:
            continue
        coeffs.append((vec / vec.sum()).tolist())
    if len(coeffs) < n_candidates:
        base = np.zeros(n_models, dtype=np.float64)
        base[anchor_idx] = 1.0
        coeffs.append(base.tolist())
    return coeffs


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
    parser.add_argument("--coefficient_strategy", choices=["uniform", "grid", "random", "anchor"], default="grid",
                        help="Coefficient generation strategy.")
    parser.add_argument("--n_coefficient_candidates", type=int, default=15,
                        help="Number of coefficient combinations to try.")
    parser.add_argument("--anchor_index", type=int, default=0,
                        help="Index of anchor model (0-based) when using anchor coefficient strategy.")
    parser.add_argument("--anchor_alpha", type=float, default=5.0,
                        help="Dirichlet alpha for anchor model when coefficient_strategy=anchor.")
    parser.add_argument("--others_alpha", type=float, default=1.0,
                        help="Dirichlet alpha for other models when coefficient_strategy=anchor.")
    parser.add_argument("--anchor_min", type=float, default=0.5,
                        help="Minimum anchor weight enforced during anchor-biased sampling.")
    parser.add_argument("--fisher_floor", type=float, default=1e-6,
                        help="Minimum Fisher value.")
    parser.add_argument("--favor_target_model", action="store_true", default=True,
                        help="Don't apply fisher_floor to first model.")
    parser.add_argument("--normalize_fishers", action="store_true", default=True,
                        help="Normalize Fisher information.")
    parser.add_argument("--use_masks", action="store_true", default=True,
                        help="Use pruning masks if available.")
    parser.add_argument("--no_use_masks", dest="use_masks", action="store_false",
                        help="Disable pruning masks even if available.")

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
    parser.add_argument("--refresh_bn", action="store_true",
                        help="Refresh BatchNorm running stats on merged model before saving.")
    parser.add_argument("--bn_refresh_batches", type=int, default=200,
                        help="Number of batches to use for BN refresh (0 uses full dataloader).")

    parser.add_argument("--run_mot", action="store_true",
                        help="Run MOT stage after soup.")
    parser.add_argument("--mot_gmm_n", type=int, default=12,
                        help="GMM components for MOT.")
    parser.add_argument("--mot_model_name", default="fgmm_model.npz",
                        help="Filename for stored fGMM models inside each run directory.")
    parser.add_argument("--mot_fisher_name", default="fgmm_fisher.pt",
                        help="Filename for stored fGMM Fisher tensors inside each run directory.")
    parser.add_argument("--mot_max_samples", type=int, default=1024,
                        help="Maximum number of motion features for Fisher estimation.")
    parser.add_argument("--mot_fisher_eps", type=float, default=1e-6,
                        help="Numerical jitter for fGMM Fisher computation.")
    parser.add_argument("--mot_recompute_fisher", action="store_true",
                        help="Force recomputation of fGMM Fisher information.")
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
    
    # Coefficient evaluation mode
    parser.add_argument("--coeff_eval", choices=["bank", "auroc", "proxy", "logp_app"], default="bank",
                        help=("How to evaluate coefficient candidates: "
                              "bank = KNN compactness (unsupervised), "
                              "auroc = run evaluation script for AUROC, "
                              "proxy = simple norm proxy, "
                              "logp_app = average GMM log-likelihood over AppAE latent."))
    parser.add_argument("--bank_include_mot", action="store_true",
                        help="When using coeff_eval=bank, include MOT cleansescores via Fisher p-value combine.")
    parser.add_argument("--bank_percentile", type=float, default=None,
                        help="Percentile for cleansing when coeff_eval=bank. Defaults to config signals.app.percentile_cleanse or 75.")
    parser.add_argument("--bank_knn_k", type=int, default=None,
                        help="K for KNN compactness when coeff_eval=bank. Defaults to config signals.app.NN or 8.")
    parser.add_argument("--bank_quantile", type=float, default=0.5,
                        help="Quantile (0-1) of LOO distances to score compactness (lower is better). Default 0.5 (median).")
    parser.add_argument("--bank_max_train_samples", type=int, default=50000,
                        help="Max number of kept App samples used for KNN compactness (subsample if larger). 0 disables sampling.")
    parser.add_argument("--keep_mask_strategy", choices=["soup", "best", "consensus_and", "consensus_or"], default="soup",
                        help="Strategy for determining keep-mask when coeff_eval=bank.")
    parser.add_argument("--bank_anchor_score", type=str, default=None,
                        help="Path to anchor cleanse score (e.g., best single model) used for keep_mask_strategy=best/consensus.")
    parser.add_argument("--bank_whiten", action="store_true",
                        help="Whiten kept features before measuring compactness (recommended).")
    parser.add_argument("--logp_app_gmm_k", type=int, default=12,
                        help="Components for AppAE latent GMM when coeff_eval=logp_app.")
    parser.add_argument("--logp_app_holdout", type=float, default=0.1,
                        help="Holdout fraction for logp_app evaluator.")
    parser.add_argument("--logp_latent", choices=["encoder", "recon"], default="recon",
                        help="Latent representation to use for logp_app (encoder output if available or reconstruction error).")
    parser.add_argument("--logp_whiten", action="store_true",
                        help="Apply whitening before fitting GMM in logp_app evaluator.")
    parser.add_argument("--logp_max_train_samples", type=int, default=50000,
                        help="Max number of latent samples for logp_app evaluator (0 disables sampling).")
    return parser.parse_args()


def setup_logger(verbose: bool) -> logging.Logger:
    logger = logging.getLogger("enhanced_run_appae_mot_soup_pipeline")
    logger.handlers.clear()
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO if verbose else logging.WARNING)
    logger.propagate = False
    return logger


def save_run_configuration(target_folder: Path, args: argparse.Namespace):
    """Persist CLI invocation and parsed arguments for reproducibility."""
    target_folder.mkdir(parents=True, exist_ok=True)
    config_path = target_folder / "run_config.json"
    payload = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "command": " ".join(shlex.quote(part) for part in sys.argv),
        "args": vars(args)
    }
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def refresh_batchnorm_stats(model: nn.Module,
                            loader: DataLoader,
                            device: torch.device,
                            max_batches: int = 200,
                            logger: Optional[logging.Logger] = None):
    """Run a forward-only pass to update BatchNorm running stats."""
    has_bn = any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in model.modules())
    if not has_bn:
        if logger:
            logger.info("BN refresh skipped: no BatchNorm layers detected.")
        return
    prev_mode = model.training
    model.train()
    n_seen = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            _ = model(batch)
            n_seen += 1
            if max_batches and max_batches > 0 and n_seen >= max_batches:
                break
    if not prev_mode:
        model.eval()
    if logger:
        logger.info("BN refresh completed on %d mini-batches.", n_seen)


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


def run_mot_soup(dataset_name: str,
                 uvadmode: str,
                 folders: List[str],
                 coefficients: List[float],
                 target_folder: Path,
                 score_template: str,
                 model_name: str,
                 fisher_name: str,
                 fisher_config: FgmmFisherConfig,
                 recompute_fisher: bool,
                 fisher_floor: float,
                 favor_target_model: bool,
                 normalize_fishers: bool,
                 logger: logging.Logger,
                 verbose: bool = False) -> Optional[Path]:
    if not coefficients:
        logger.warning("Skipping MOT soup: no coefficients available.")
        return None

    motion_features = featurebank.get(dataset_name, 'mot', 'train', uvadmode=uvadmode).astype(np.float32)
    if motion_features.size == 0:
        logger.warning("Skipping MOT soup: no motion features available.")
        return None

    mot_state_dicts = []
    mot_fishers = []
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
            gmm = load_fgmm_model(model_path)
            mot_state_dicts.append(fgmm_to_state_dict(gmm))

            fisher_path = model_path.parent / fisher_name
            fisher = None
            if fisher_path.exists() and not recompute_fisher:
                fisher = load_fgmm_fisher(fisher_path)

            if fisher is None or recompute_fisher:
                if verbose:
                    logger.info("Computing fGMM Fisher information for %s", model_path.parent)
                sample_seed = None if fisher_config.seed is None else fisher_config.seed + len(mot_state_dicts) - 1
                cfg = FgmmFisherConfig(
                    max_samples=fisher_config.max_samples,
                    seed=sample_seed,
                    eps=fisher_config.eps
                )
                fisher = compute_fisher_for_fgmm(
                    gmm,
                    motion_features,
                    config=cfg,
                    logger=logger if verbose else None
                )
                metadata = {
                    "folder": str(model_path.parent),
                    "model_path": str(model_path),
                    "max_samples": cfg.max_samples,
                    "seed": cfg.seed,
                    "eps": cfg.eps,
                    "total_features": int(motion_features.shape[0]),
                }
                save_fgmm_fisher(fisher, fisher_path, metadata=metadata)
            mot_fishers.append(fisher)
        except Exception as exc:
            logger.warning("Failed to load fGMM model from %s: %s", model_path, exc)
            missing.append(str(folder_path))

    if missing:
        logger.warning("MOT soup skipped due to missing models: %s", missing)
        return None

    if len(mot_state_dicts) != len(coefficients):
        logger.warning("MOT soup skipped: number of models (%d) != coefficients (%d)",
                       len(mot_state_dicts), len(coefficients))
        return None

    merged_state = fisher_weighted_average_fgmm(
        mot_state_dicts,
        mot_fishers,
        coefficients,
        fisher_floor=fisher_floor,
        favor_target_model=favor_target_model,
        normalize_fishers=normalize_fishers
    )
    merged_gmm = state_dict_to_fgmm(merged_state)

    mot_model_path = target_folder / model_name
    save_fgmm_model(merged_gmm, mot_model_path, logger)

    logger.info("Generating MOT pseudo scores on %d samples.", motion_features.shape[0])
    mot_scores = -merged_gmm.score_samples(motion_features)

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
            
            # Cleanup (Py3.7 compat)
            try:
                if temp_checkpoint.exists():
                    temp_checkpoint.unlink()
            except Exception:
                pass
            try:
                if scores_path.exists():
                    scores_path.unlink()
            except Exception:
                pass
            
            self.evaluation_count += 1
            return auroc if auroc is not None else 0.0
            
        except Exception as e:
            self.logger.warning(f"Evaluation failed: {e}")
            return 0.0


class BankCompactnessEvaluator:
    """Unsupervised evaluator for coefficient candidates based on bank compactness.

    Steps per candidate:
    1) Save temp AppAE checkpoint and generate cleanse scores for app
    2) Optionally load MOT cleanse scores and Fisher-combine p-values
    3) Build keep mask at given percentile and compute LOO-KNN distances on kept app features
    4) Return negative quantile (lower distance -> higher score)
    """

    def __init__(self,
                 dataset_name: str,
                 uvadmode: str,
                 target_folder: Path,
                 inference_batch_size: int,
                 inference_workers: int,
                 score_template: str,
                 percentile: float,
                 knn_k: int,
                 include_mot: bool,
                 quantile: float,
                 max_train_samples: Optional[int],
                 keep_mask_strategy: str,
                 anchor_score_path: Optional[Path],
                 whiten: bool,
                 device: torch.device,
                 logger: logging.Logger):
        self.dataset_name = dataset_name
        self.uvadmode = uvadmode
        self.target_folder = target_folder
        self.inference_batch_size = inference_batch_size
        self.inference_workers = inference_workers
        self.score_template = score_template
        self.percentile = percentile
        self.knn_k = knn_k
        self.include_mot = include_mot
        self.quantile = quantile
        self.max_train_samples = max_train_samples if (max_train_samples and max_train_samples > 0) else None
        self.keep_mask_strategy = keep_mask_strategy
        self.anchor_score_path = anchor_score_path
        self.whiten = whiten
        self.device = device
        self.logger = logger
        self.evaluation_count = 0
        self._anchor_scores = None
        if self.keep_mask_strategy != "soup" and self.anchor_score_path is None:
            raise ValueError("keep_mask_strategy requires --bank_anchor_score.")

    def __call__(self, state_dict: Dict[str, torch.Tensor]) -> float:
        from codes import featurebank as fb
        from codes.grader import KNNGrader
        try:
            # 1) App cleanse scores for candidate
            tmp_ckpt = self.target_folder / f"_temp_bank_{self.evaluation_count}.pkl"
            torch.save(state_dict, tmp_ckpt)
            label = f"temp_eval_{self.evaluation_count}"
            score_path = Path(self.score_template.format(
                dataset=self.dataset_name, mode=self.uvadmode, label=label, uvadmode=self.uvadmode
            ))
            run_appae_inference(
                self.dataset_name, self.uvadmode, tmp_ckpt, score_path,
                self.inference_batch_size, self.inference_workers, self.device, self.logger
            )
            app_scores = np.load(score_path, allow_pickle=True)

            # 2) p-values and optional MOT
            pvals_list = [_ecdf_pvals(app_scores)]
            if self.include_mot:
                mot_path = Path(f"features/{self.dataset_name}/cleansescores/{self.uvadmode}_velo_fgmm_flat.npy")
                if mot_path.exists():
                    mot_scores = np.load(mot_path, allow_pickle=True)
                    if len(mot_scores) == len(app_scores):
                        pvals_list.append(_ecdf_pvals(mot_scores))
                    else:
                        self.logger.warning("MOT scores length mismatch; skipping MOT in bank evaluator.")
                else:
                    self.logger.warning("MOT cleanse scores not found at %s; skipping MOT.", mot_path)

            if len(pvals_list) == 1:
                stat = _fisher_combine_pvals(pvals_list)  # same as using single source
            else:
                stat = _fisher_combine_pvals(pvals_list)

            keep_mask = self._resolve_keep_mask(stat)

            # 3) App features compactness on kept
            tr_app = fb.get(self.dataset_name, 'app', 'train', uvadmode=self.uvadmode)
            if len(tr_app) != len(keep_mask):
                self.logger.warning("App features length mismatch with scores; falling back to proxy.")
                return -float(np.linalg.norm(np.concatenate([v.flatten() for v in state_dict.values() if v.numel() > 0])[:1024]))

            kept = tr_app[keep_mask]
            if kept.shape[0] < max(self.knn_k + 2, 10):
                self.logger.warning("Too few kept samples (%d); penalizing.", kept.shape[0])
                score = -1e9
            else:
                if self.max_train_samples and kept.shape[0] > self.max_train_samples:
                    idx = np.random.default_rng(42 + self.evaluation_count).choice(
                        kept.shape[0], size=self.max_train_samples, replace=False
                    )
                    kept = kept[idx]
                    stat = stat[idx]
                kept = kept.astype(np.float32)
                if self.whiten:
                    mu = kept.mean(axis=0, keepdims=True)
                    sigma = kept.std(axis=0, keepdims=True) + 1e-6
                    kept = (kept - mu) / sigma
                grader = KNNGrader(kept, K=self.knn_k, key='bank')
                dists = grader.grade_flat(kept)
                compact = np.quantile(dists, self.quantile)
                score = -float(compact)  # smaller is better

            # 4) Cleanup (Py3.7 compat, guard against missing files)
            try:
                if tmp_ckpt.exists():
                    tmp_ckpt.unlink()
            except Exception:
                pass
            try:
                if score_path.exists():
                    score_path.unlink()
            except Exception:
                pass
            self.evaluation_count += 1
            return score
        except Exception as e:
            self.logger.warning("Bank evaluator failed: %s", e)
            return 0.0

    def _resolve_keep_mask(self, soup_stat: np.ndarray) -> np.ndarray:
        """Determine keep mask based on configured strategy."""
        soup_thr = np.percentile(soup_stat, self.percentile)
        soup_mask = soup_stat <= soup_thr
        if self.keep_mask_strategy == "soup":
            return soup_mask

        anchor_scores = self._load_anchor_scores()
        if anchor_scores is None or len(anchor_scores) != len(soup_mask):
            self.logger.warning("Anchor scores unavailable or length mismatch; falling back to soup mask.")
            return soup_mask
        anchor_thr = np.percentile(anchor_scores, self.percentile)
        anchor_mask = anchor_scores <= anchor_thr

        if self.keep_mask_strategy == "best":
            return anchor_mask
        if self.keep_mask_strategy == "consensus_and":
            return anchor_mask & soup_mask
        if self.keep_mask_strategy == "consensus_or":
            return anchor_mask | soup_mask
        return soup_mask

    def _load_anchor_scores(self) -> Optional[np.ndarray]:
        if self.anchor_score_path is None:
            return None
        if self._anchor_scores is None:
            try:
                self._anchor_scores = np.load(self.anchor_score_path, allow_pickle=True)
            except Exception as exc:
                self.logger.warning("Failed to load anchor scores from %s: %s", self.anchor_score_path, exc)
                self._anchor_scores = None
        return self._anchor_scores


def _extract_app_latent(model: AppAE, batch: torch.Tensor, mode: str = "recon") -> np.ndarray:
    """Return latent representation for AppAE batch."""
    with torch.no_grad():
        if mode == "encoder" and hasattr(model, "encoder"):
            latent = model.encoder(batch)
            if isinstance(latent, (list, tuple)):
                latent = latent[-1]
            return latent.flatten(1).detach().cpu().numpy()
        # Fallback to reconstruction-error vector
        recon = model(batch)
        err = ((recon - batch) ** 2).flatten(1)
        return err.detach().cpu().numpy()


class LogProbEvaluatorApp:
    """Evaluate soup candidates using GMM log-likelihood over AppAE latent space."""

    def __init__(self,
                 dataset_name: str,
                 uvadmode: str,
                 target_folder: Path,
                 device: torch.device,
                 inference_batch_size: int,
                 inference_workers: int,
                 score_template: str,
                 percentile: float,
                 keep_mask_strategy: str,
                 anchor_score_path: Optional[Path],
                 gmm_k: int,
                 holdout_frac: float,
                 whiten: bool,
                 max_train_samples: Optional[int],
                 seed: int,
                 logger: logging.Logger,
                 latent_mode: str = "recon"):
        self.dataset_name = dataset_name
        self.uvadmode = uvadmode
        self.target_folder = target_folder
        self.device = device
        self.batch_size = inference_batch_size
        self.num_workers = inference_workers
        self.score_template = score_template
        self.percentile = percentile
        self.keep_mask_strategy = keep_mask_strategy
        self.anchor_score_path = anchor_score_path
        self.gmm_k = gmm_k
        self.holdout_frac = holdout_frac
        self.whiten = whiten
        self.max_train_samples = max_train_samples if (max_train_samples and max_train_samples > 0) else None
        self.seed = seed
        self.logger = logger
        self.evaluation_count = 0
        self.latent_mode = latent_mode
        self._anchor_scores: Optional[np.ndarray] = None

    def __call__(self, state_dict: Dict[str, torch.Tensor]) -> float:
        try:
            tmp_ckpt = self.target_folder / f"_temp_logp_{self.evaluation_count}.pkl"
            torch.save(state_dict, tmp_ckpt)
            label = f"temp_logp_{self.evaluation_count}"
            score_path = Path(self.score_template.format(
                dataset=self.dataset_name,
                mode=self.uvadmode,
                label=label,
                uvadmode=self.uvadmode
            ))
            run_appae_inference(
                self.dataset_name,
                self.uvadmode,
                tmp_ckpt,
                score_path,
                self.batch_size,
                self.num_workers,
                self.device,
                self.logger
            )
            soup_scores = np.load(score_path, allow_pickle=True)
            keep_mask = self._resolve_keep_mask(soup_scores)

            helper = Cleanse(self.dataset_name, self.uvadmode)
            paths = helper.get_app_fpaths()
            dataset = PatchDataset(paths)
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False,
                                num_workers=self.num_workers, pin_memory=True)

            model = AppAE().to(self.device)
            model.load_state_dict(state_dict)
            model.eval()

            latents = []
            start = 0
            with torch.no_grad():
                for batch in loader:
                    bsz = batch.shape[0]
                    mask_chunk = keep_mask[start:start + bsz]
                    if mask_chunk.any():
                        batch = batch.to(self.device)
                        latent = _extract_app_latent(model, batch, mode=self.latent_mode)
                        latents.append(latent[mask_chunk])
                    start += bsz

            if not latents:
                score = -1e9
            else:
                latent_mat = np.concatenate(latents, axis=0)
                if self.max_train_samples and latent_mat.shape[0] > self.max_train_samples:
                    base = 0 if self.seed is None else int(self.seed)
                    rng = np.random.default_rng(base + 123 + self.evaluation_count)
                    idx = rng.choice(latent_mat.shape[0], size=self.max_train_samples, replace=False)
                    latent_mat = latent_mat[idx]
                if self.whiten:
                    mean = latent_mat.mean(axis=0, keepdims=True)
                    std = latent_mat.std(axis=0, keepdims=True) + 1e-6
                    latent_mat = (latent_mat - mean) / std
                n = latent_mat.shape[0]
                holdout = max(32, int(n * self.holdout_frac))
                if n <= self.gmm_k + holdout:
                    score = -1e9
                else:
                    base = 0 if self.seed is None else int(self.seed)
                    rng = np.random.default_rng(base + 999 + self.evaluation_count)
                    perm = rng.permutation(n)
                    Xte = latent_mat[perm[:holdout]]
                    Xtr = latent_mat[perm[holdout:]]
                    gmm = GaussianMixture(n_components=self.gmm_k, covariance_type="full",
                                          random_state=base + 2024 + self.evaluation_count)
                    gmm.fit(Xtr)
                    score = float(gmm.score_samples(Xte).mean())

            # Cleanup
            try:
                if tmp_ckpt.exists():
                    tmp_ckpt.unlink()
            except Exception:
                pass
            try:
                if score_path.exists():
                    score_path.unlink()
            except Exception:
                pass
            self.evaluation_count += 1
            return score
        except Exception as exc:
            self.logger.warning("LogProb evaluator failed: %s", exc)
            self.evaluation_count += 1
            return -1e9

    def _resolve_keep_mask(self, soup_scores: np.ndarray) -> np.ndarray:
        thr = np.percentile(soup_scores, self.percentile)
        soup_mask = soup_scores <= thr
        if self.keep_mask_strategy == "soup":
            return soup_mask
        anchor = self._load_anchor_scores()
        if anchor is None or len(anchor) != len(soup_scores):
            return soup_mask
        anchor_thr = np.percentile(anchor, self.percentile)
        anchor_mask = anchor <= anchor_thr
        if self.keep_mask_strategy == "best":
            return anchor_mask
        if self.keep_mask_strategy == "consensus_and":
            return anchor_mask & soup_mask
        if self.keep_mask_strategy == "consensus_or":
            return anchor_mask | soup_mask
        return soup_mask

    def _load_anchor_scores(self) -> Optional[np.ndarray]:
        if self.anchor_score_path is None:
            return None
        if self._anchor_scores is None:
            try:
                self._anchor_scores = np.load(self.anchor_score_path, allow_pickle=True)
            except Exception as exc:
                self.logger.warning("Failed to load anchor scores from %s: %s", self.anchor_score_path, exc)
                self._anchor_scores = None
        return self._anchor_scores


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

    if device.type == "cuda":
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass
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
    save_run_configuration(target_folder, args)

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
        elif args.coefficient_strategy == "anchor":
            coefficient_candidates = create_anchor_biased_coeffs_local(
                n_models=len(folders),
                n_candidates=args.n_coefficient_candidates,
                anchor_idx=int(args.anchor_index),
                alpha_anchor=float(args.anchor_alpha),
                alpha_others=float(args.others_alpha),
                min_anchor=float(args.anchor_min),
                seed=seed
            )
        else:
            coefficient_candidates = create_random_coeffs(
                len(folders), args.n_coefficient_candidates, seed
            )
        
        logger.info("Generated %d coefficient candidates using %s strategy", 
                   len(coefficient_candidates), args.coefficient_strategy)

        # Setup evaluator
        if args.coeff_eval == 'auroc' and eval_config_path:
            evaluator = ModelEvaluator(
                args.dataset_name, eval_mode_default, eval_config_path,
                target_folder, device, inference_batch_size, inference_workers,
                args.score_output_template, logger
            )
        elif args.coeff_eval == 'bank':
            # Resolve percentile/K from config if available
            bank_percentile = args.bank_percentile
            bank_knn_k = args.bank_knn_k
            if config_path and (bank_percentile is None or bank_knn_k is None):
                try:
                    import yaml
                    with open(config_path, 'r', encoding='utf-8') as fh:
                        cfg_yaml = yaml.safe_load(fh) or {}
                    sig_app = ((cfg_yaml.get('signals') or {}).get('app') or {})
                    bank_percentile = bank_percentile if bank_percentile is not None else float(sig_app.get('percentile_cleanse', 75))
                    bank_knn_k = bank_knn_k if bank_knn_k is not None else int(sig_app.get('NN', 8))
                except Exception as e:
                    logger.warning("Failed to read signals from config: %s", e)
            if bank_percentile is None:
                bank_percentile = 75.0
            if bank_knn_k is None:
                bank_knn_k = 8

            anchor_score_path = Path(args.bank_anchor_score) if args.bank_anchor_score else None

            evaluator = BankCompactnessEvaluator(
                args.dataset_name, args.uvadmode, target_folder,
                inference_batch_size, inference_workers, args.score_output_template,
                bank_percentile, bank_knn_k, args.bank_include_mot, args.bank_quantile,
                args.bank_max_train_samples,
                args.keep_mask_strategy, anchor_score_path, args.bank_whiten,
                device, logger
            )
        elif args.coeff_eval == 'logp_app':
            anchor_score_path = Path(args.bank_anchor_score) if args.bank_anchor_score else None
            logp_percentile = args.bank_percentile or 75.0
            evaluator = LogProbEvaluatorApp(
                args.dataset_name,
                args.uvadmode,
                target_folder,
                device,
                inference_batch_size,
                inference_workers,
                args.score_output_template,
                logp_percentile,
                args.keep_mask_strategy,
                anchor_score_path,
                args.logp_app_gmm_k,
                args.logp_app_holdout,
                args.logp_whiten,
                args.logp_max_train_samples if args.logp_max_train_samples > 0 else None,
                seed,
                logger,
                latent_mode=args.logp_latent
            )
        else:
            # Simple proxy fallback
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

    # Optionally refresh BN running stats before saving
    if args.refresh_bn:
        try:
            bn_model = AppAE().to(device)
            bn_model.load_state_dict(best_state)
            refresh_batchnorm_stats(
                bn_model,
                dataloader,
                device,
                max_batches=args.bn_refresh_batches,
                logger=logger if args.verbose else None
            )
            best_state = bn_model.state_dict()
            del bn_model
            if device.type == "cuda":
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
        except Exception as exc:
            logger.warning("BN refresh failed: %s", exc)

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
        "mot_fisher_name": args.mot_fisher_name,
        "mot_max_samples": args.mot_max_samples,
        "mot_fisher_eps": args.mot_fisher_eps,
        "mot_recompute_fisher": args.mot_recompute_fisher,
        "mot_fisher_seed": seed,
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
        mot_fisher_config = FgmmFisherConfig(
            max_samples=args.mot_max_samples,
            seed=seed,
            eps=args.mot_fisher_eps
        )
        mot_score_path = run_mot_soup(
            args.dataset_name,
            args.uvadmode,
            folders,
            best_coeffs,
            target_folder,
            args.mot_score_output_template,
            args.mot_model_name,
            args.mot_fisher_name,
            mot_fisher_config,
            args.mot_recompute_fisher,
            args.fisher_floor,
            args.favor_target_model,
            args.normalize_fishers,
            logger,
            verbose=args.verbose
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
