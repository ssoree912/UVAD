"""
Utilities for Fisher-weighted model soup of motion pseudo anomaly models (fGMM).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky


@dataclass
class FgmmFisherConfig:
    max_samples: Optional[int] = 1024
    seed: Optional[int] = None
    eps: float = 1e-6


def fgmm_to_state_dict(gmm: GaussianMixture) -> Dict[str, torch.Tensor]:
    """Convert scikit-learn GaussianMixture parameters to a torch-friendly dict."""
    device = torch.device("cpu")
    dtype = torch.float64
    return {
        "weights": torch.as_tensor(gmm.weights_, dtype=dtype, device=device),
        "means": torch.as_tensor(gmm.means_, dtype=dtype, device=device),
        "covariances": torch.as_tensor(gmm.covariances_, dtype=dtype, device=device),
    }


def state_dict_to_fgmm(state_dict: Dict[str, torch.Tensor],
                       covariance_type: str = "full") -> GaussianMixture:
    """Create a GaussianMixture instance from state dict."""
    weights = state_dict["weights"].detach().cpu().numpy()
    means = state_dict["means"].detach().cpu().numpy()
    covariances = state_dict["covariances"].detach().cpu().numpy()

    n_components = weights.shape[0]
    gmm = GaussianMixture(n_components=n_components, covariance_type=covariance_type)
    gmm.weights_ = weights
    gmm.means_ = means
    gmm.covariances_ = covariances
    gmm.precisions_cholesky_ = _compute_precision_cholesky(covariances, covariance_type)
    gmm.converged_ = True
    gmm.n_features_in_ = means.shape[1]
    return gmm


def _mixture_log_prob(samples: torch.Tensor,
                      weights: torch.Tensor,
                      means: torch.Tensor,
                      covariances: torch.Tensor,
                      eps: float = 1e-6) -> torch.Tensor:
    """Compute log probability for samples under a full-covariance Gaussian mixture."""
    samples = samples.unsqueeze(1)  # (B, 1, D)
    weights = torch.clamp(weights, min=eps)

    diff = samples - means.unsqueeze(0)  # (B, K, D)
    chol = torch.linalg.cholesky(
        covariances + torch.eye(covariances.size(-1), device=covariances.device, dtype=covariances.dtype) * eps
    )  # (K, D, D)
    # Solve triangular systems for Mahalanobis distance
    # torch.cholesky_solve expects RHS [..., D, R], so reshape diff accordingly
    diff_expanded = diff.unsqueeze(-1)  # (B, K, D, 1)
    solved = torch.cholesky_solve(diff_expanded, chol)  # (B, K, D, 1)
    mahalanobis = (diff_expanded * solved).sum(dim=2).squeeze(-1)  # (B, K)

    log_det = 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(dim=1)  # (K,)
    dim = samples.size(-1)
    log_coeff = torch.log(weights) - 0.5 * (log_det + dim * torch.log(torch.tensor(2 * np.pi, dtype=chol.dtype, device=chol.device)))
    component_log_probs = log_coeff.unsqueeze(0) - 0.5 * mahalanobis
    return torch.logsumexp(component_log_probs, dim=1)  # (B,)


def compute_fisher_for_fgmm(gmm: GaussianMixture,
                            features: np.ndarray,
                            *,
                            config: Optional[FgmmFisherConfig] = None,
                            logger: Optional[logging.Logger] = None) -> Dict[str, torch.Tensor]:
    """
    Approximate Fisher information for fGMM parameters using per-sample gradients.

    The result is a dictionary keyed by parameter name with tensors that match
    the corresponding parameter shapes. The Fisher entries are averaged over the
    sampled feature subset.
    """
    if config is None:
        config = FgmmFisherConfig()

    if features.ndim != 2:
        raise ValueError(f"Expected 2-D features, got shape {features.shape}")

    total_samples = features.shape[0]
    if config.max_samples is not None and config.max_samples < total_samples:
        rng = np.random.default_rng(config.seed)
        indices = rng.choice(total_samples, size=config.max_samples, replace=False)
        features = features[indices]
        if logger:
            logger.info("FGMM Fisher: using %d/%d samples.", features.shape[0], total_samples)
    else:
        if logger:
            logger.info("FGMM Fisher: using all %d samples.", total_samples)

    torch_features = torch.from_numpy(features).to(torch.float64)
    # torch.as_tensor in older versions doesn't accept requires_grad; use torch.tensor
    weights = torch.tensor(gmm.weights_, dtype=torch.float64, requires_grad=True)
    means = torch.tensor(gmm.means_, dtype=torch.float64, requires_grad=True)
    covariances = torch.tensor(gmm.covariances_, dtype=torch.float64, requires_grad=True)

    params = [weights, means, covariances]
    fisher_accumulators = [
        torch.zeros_like(weights),
        torch.zeros_like(means),
        torch.zeros_like(covariances),
    ]

    for sample in torch_features:
        log_prob = _mixture_log_prob(sample.unsqueeze(0), weights, means, covariances, eps=config.eps)
        grads = torch.autograd.grad(log_prob, params, retain_graph=False, allow_unused=True)
        for acc, grad in zip(fisher_accumulators, grads):
            if grad is None:
                continue
            acc += grad.detach() ** 2

    num_samples = float(torch_features.shape[0])
    fisher_dict = {
        "weights": (fisher_accumulators[0] / num_samples).detach(),
        "means": (fisher_accumulators[1] / num_samples).detach(),
        "covariances": (fisher_accumulators[2] / num_samples).detach(),
    }
    return fisher_dict


def fisher_weighted_average_fgmm(state_dicts: Sequence[Dict[str, torch.Tensor]],
                                 fisher_dicts: Sequence[Dict[str, torch.Tensor]],
                                 coefficients: Sequence[float],
                                 *,
                                 fisher_floor: float = 1e-6,
                                 favor_target_model: bool = True,
                                 normalize_fishers: bool = True,
                                 eps: float = 1e-12) -> Dict[str, torch.Tensor]:
    """Perform Fisher-weighted averaging for fGMM parameters."""
    if len(state_dicts) != len(fisher_dicts):
        raise ValueError("State dict and Fisher dict counts must match.")
    if len(state_dicts) != len(coefficients):
        raise ValueError("Number of coefficients must match number of models.")

    coeffs = [float(c) for c in coefficients]

    merged: Dict[str, torch.Tensor] = {}
    for key in ("weights", "means", "covariances"):
        params = [sd[key].to(torch.float64) for sd in state_dicts]

        base_fishers: List[torch.Tensor] = []
        for idx, fisher_dict in enumerate(fisher_dicts):
            if fisher_dict is None or key not in fisher_dict:
                fisher_tensor = torch.ones_like(params[idx])
            else:
                fisher_tensor = fisher_dict[key].to(torch.float64)
            if fisher_floor > 0 and (not favor_target_model or idx != 0):
                fisher_tensor = torch.clamp(fisher_tensor, min=fisher_floor)
            base_fishers.append(fisher_tensor)

        if normalize_fishers:
            stacked = torch.stack(base_fishers, dim=0)
            denom = stacked.sum(dim=0, keepdim=True).clamp_min(eps)
            normed = [tensor / denom.squeeze(0) for tensor in base_fishers]
        else:
            normed = base_fishers

        fisher_weights = [
            normed_tensor * coeff
            for normed_tensor, coeff in zip(normed, coeffs)
        ]

        numerator = torch.zeros_like(params[0], dtype=torch.float64)
        denominator = torch.zeros_like(params[0], dtype=torch.float64)

        for param_tensor, fisher_weight in zip(params, fisher_weights):
            numerator += fisher_weight * param_tensor
            denominator += fisher_weight

        merged_param = numerator / denominator.clamp_min(eps)

        if key == "weights":
            merged_param = torch.clamp(merged_param, min=1e-12)
            merged_param = merged_param / merged_param.sum()
        elif key == "covariances":
            merged_param = 0.5 * (merged_param + merged_param.transpose(-1, -2))

        merged[key] = merged_param.to(torch.float64)

    return merged


def save_fgmm_fisher(fisher: Dict[str, torch.Tensor],
                     path: Path,
                     *,
                     metadata: Optional[Dict[str, object]] = None):
    payload = {"fisher": fisher}
    if metadata is not None:
        payload["metadata"] = metadata
    torch.save(payload, path)


def load_fgmm_fisher(path: Path) -> Optional[Dict[str, torch.Tensor]]:
    if not path.exists():
        return None
    package = torch.load(str(path), map_location="cpu")
    if isinstance(package, dict) and "fisher" in package:
        return package["fisher"]
    return package
