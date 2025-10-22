"""
Enhanced Fisher-weighted model soup utilities with VAD_soup improvements.
Incorporates the best features from VAD_soup while maintaining stability.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, List, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import yaml

from codes.cleanse import Cleanse, PatchDataset, AppAE


APP_CONFIG_SECTION = "appaerecon"


@dataclass
class AppAEConfig:
    seed: int = 111
    epochs: int = 10
    batch_size: int = 64
    optimizer_lr: float = 1e-3
    num_workers: int = 4
    save_root: str = "artifacts"
    log_dir: str = "logger"
    device: Optional[str] = None
    magnitude_prune: float = 0.0
    random_prune: float = 0.0
    random_prune_seed: Optional[int] = None
    unprune_epoch: Optional[int] = None
    eval_config: Optional[str] = None
    eval_interval: int = 0
    run_name_prefix: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict) -> "AppAEConfig":
        if data is None:
            data = {}
        return cls(
            seed=int(data.get("seed", cls.seed)),
            epochs=int(data.get("epochs", cls.epochs)),
            batch_size=int(data.get("batch_size", cls.batch_size)),
            optimizer_lr=float(data.get("optimizer_lr", cls.optimizer_lr)),
            num_workers=int(data.get("num_workers", cls.num_workers)),
            save_root=str(data.get("save_root", cls.save_root)),
            log_dir=str(data.get("log_dir", cls.log_dir)),
            device=data.get("device", cls.device),
            magnitude_prune=float(data.get("magnitude_prune", cls.magnitude_prune)),
            random_prune=float(data.get("random_prune", cls.random_prune)),
            random_prune_seed=(
                None if data.get("random_prune_seed") is None
                else int(data.get("random_prune_seed"))
            ),
            unprune_epoch=(
                None if data.get("unprune_epoch") is None
                else int(data.get("unprune_epoch"))
            ),
            eval_config=data.get("eval_config"),
            eval_interval=int(data.get("eval_interval", cls.eval_interval)),
            run_name_prefix=data.get("run_name_prefix"),
        )

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2)


def resolve_config_path(config_arg: Optional[str], dataset_name: str) -> Optional[Path]:
    if config_arg:
        path = Path(config_arg)
        if path.exists():
            return path
        raise FileNotFoundError(f"Config file not found: {config_arg}")
    candidate = Path("configs") / f"config_{dataset_name}.yaml"
    return candidate if candidate.exists() else None


def load_appae_training_config(config_arg: Optional[str],
                               dataset_name: str) -> Tuple[Optional[Path], AppAEConfig]:
    cfg_path = resolve_config_path(config_arg, dataset_name)
    if cfg_path is None:
        return None, AppAEConfig()

    with open(cfg_path, "r", encoding="utf-8") as handle:
        raw_cfg = yaml.safe_load(handle) or {}
    app_cfg = AppAEConfig.from_dict(raw_cfg.get(APP_CONFIG_SECTION, {}))
    return cfg_path, app_cfg


def collect_patch_paths(dataset_name: str, uvadmode: str) -> List[str]:
    helper = Cleanse(dataset_name, uvadmode)
    return helper.get_app_fpaths()


def build_patch_dataloader(dataset_name: str,
                           uvadmode: str,
                           batch_size: int,
                           num_workers: int,
                           *,
                           shuffle: bool = True,
                           subset: Optional[int] = None,
                           seed: int = 111) -> Tuple[DataLoader, int]:
    paths = collect_patch_paths(dataset_name, uvadmode)
    if not paths:
        raise RuntimeError(
            f"No patch files found for dataset={dataset_name}, uvadmode={uvadmode}."
        )
    dataset = PatchDataset(paths)
    total_samples = len(dataset)

    if subset is not None and subset < total_samples:
        rng = np.random.default_rng(seed)
        indices = rng.choice(total_samples, size=subset, replace=False)
        dataset = Subset(dataset, indices.tolist())
        total_samples = subset

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader, total_samples


def load_appae_model(checkpoint_path: Path, device: torch.device) -> AppAE:
    state_dict = torch.load(str(checkpoint_path), map_location=device)
    model = AppAE().to(device)
    model.load_state_dict(state_dict)
    return model


def compute_fisher_for_appae(model: AppAE,
                             dataloader: DataLoader,
                             device: torch.device,
                             *,
                             max_batches: Optional[int] = None,
                             logger: Optional[logging.Logger] = None) -> Tuple[Dict[str, torch.Tensor], int]:
    if logger:
        logger.info("Starting Fisher computation (max_batches=%s)...", str(max_batches))

    model.train()
    fisher_accumulators: Dict[str, torch.Tensor] = {
        name: torch.zeros_like(param, device=device)
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    num_batches = 0
    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        inputs = batch.to(device)
        model.zero_grad(set_to_none=True)

        recon = model(inputs)
        loss = ((recon - inputs) ** 2).mean()
        loss.backward()

        with torch.no_grad():
            for name, param in model.named_parameters():
                if not param.requires_grad or param.grad is None:
                    continue
                fisher_accumulators[name] += param.grad.detach() ** 2

        num_batches += 1
        if logger and (batch_idx + 1) % 10 == 0:
            logger.info("Processed %d batches for Fisher computation.", batch_idx + 1)

    if num_batches == 0:
        raise RuntimeError("No batches processed while computing Fisher information.")

    for name in fisher_accumulators:
        fisher_accumulators[name] = (
            fisher_accumulators[name] / float(num_batches)
        ).detach().cpu()

    if logger:
        logger.info("Fisher computation complete. Processed %d batches.", num_batches)

    return fisher_accumulators, num_batches


def load_masks(state_dicts: List[Dict[str, torch.Tensor]], 
               checkpoint_paths: List[str]) -> List[Optional[Dict[str, torch.Tensor]]]:
    """Load pruning masks for each checkpoint if they exist."""
    masks = []
    for i, ckpt_path in enumerate(checkpoint_paths):
        mask_path = ckpt_path + ".mask"
        if Path(mask_path).exists():
            try:
                raw_mask_dict = torch.load(mask_path, map_location='cpu')
                
                # Handle different mask formats
                if isinstance(raw_mask_dict, dict):
                    # Convert to boolean mask - handle both VAD_soup and codes/cleanse formats
                    bool_mask = {}
                    for key, tensor in raw_mask_dict.items():
                        if isinstance(tensor, torch.Tensor):
                            # Convert any non-zero values to True (active weights)
                            bool_mask[key] = (tensor != 0).to(torch.bool)
                        else:
                            logging.warning(f"Skipping non-tensor mask entry: {key}")
                    masks.append(bool_mask if bool_mask else None)
                else:
                    logging.warning(f"Unexpected mask format in {mask_path}: {type(raw_mask_dict)}")
                    masks.append(None)
            except Exception as e:
                logging.warning(f"Failed to load mask for {ckpt_path}: {e}")
                masks.append(None)
        else:
            masks.append(None)
    return masks


def combine_masks(masks: List[Optional[Dict[str, torch.Tensor]]]) -> Optional[Dict[str, torch.Tensor]]:
    """Combine multiple pruning masks using OR logic (union of non-pruned weights)."""
    combined: Dict[str, torch.Tensor] = {}
    has_mask = False
    
    for mask in masks:
        if mask is None:
            continue
        has_mask = True
        for key, tensor in mask.items():
            bool_tensor = tensor.to(torch.bool)
            if key not in combined:
                combined[key] = bool_tensor.clone()
            else:
                combined[key] = torch.logical_or(combined[key], bool_tensor)
    
    return combined if has_mask else None


def apply_mask_to_state_dict(state_dict: Dict[str, torch.Tensor], 
                           mask: Optional[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Apply pruning mask to state dictionary."""
    if mask is None:
        return state_dict
    
    masked_state = {}
    for name, param in state_dict.items():
        if name in mask:
            mask_tensor = mask[name].to(param.device, dtype=param.dtype)
            masked_state[name] = param * mask_tensor
        else:
            masked_state[name] = param
    
    return masked_state


def enhanced_fisher_weighted_average(state_dicts: List[Dict[str, torch.Tensor]],
                                   fisher_dicts: List[Optional[Dict[str, torch.Tensor]]],
                                   coefficients: Iterable[float],
                                   *,
                                   fisher_floor: float = 1e-6,
                                   favor_target_model: bool = True,
                                   normalize_fishers: bool = True,
                                   eps: float = 1e-8,
                                   masks: Optional[List[Optional[Dict[str, torch.Tensor]]]] = None) -> Dict[str, torch.Tensor]:
    """
    Enhanced Fisher-weighted averaging with VAD_soup improvements.
    
    Args:
        state_dicts: List of model state dictionaries
        fisher_dicts: List of Fisher information dictionaries
        coefficients: Mixing coefficients
        fisher_floor: Minimum Fisher value for numerical stability
        favor_target_model: Whether to not apply fisher_floor to first model
        normalize_fishers: Whether to normalize Fisher information
        eps: Small epsilon for numerical stability
        masks: Optional pruning masks
        
    Returns:
        Merged state dictionary
    """
    coeffs = list(coefficients)
    if len(coeffs) != len(state_dicts):
        raise ValueError("Number of coefficients must match number of models.")

    # Handle masks
    if masks is not None:
        combined_mask = combine_masks(masks)
        state_dicts = [apply_mask_to_state_dict(sd, combined_mask) for sd in state_dicts]
    
    # Normalize Fisher information if requested
    norm_constants = None
    if normalize_fishers and fisher_dicts is not None and not all(f is None for f in fisher_dicts):
        norm_constants = []
        for fisher_dict in fisher_dicts:
            if fisher_dict is None:
                norm_constants.append(1.0)
            else:
                norm_const = torch.sqrt(sum(torch.sum(f ** 2) for f in fisher_dict.values()))
                norm_constants.append(norm_const.item())
        
        # Apply normalization to coefficients
        coeffs = [c / n for c, n in zip(coeffs, norm_constants)]

    merged: Dict[str, torch.Tensor] = {}
    param_names = state_dicts[0].keys()

    for name in param_names:
        params = [state[name].detach().cpu().float() for state in state_dicts]
        
        # Prepare Fisher weights for each model
        fisher_weights = []
        for i, (coeff, fisher_dict) in enumerate(zip(coeffs, fisher_dicts)):
            if fisher_dict is None or name not in fisher_dict:
                # Use uniform weight if no Fisher info
                fisher_weights.append(torch.full_like(params[0], float(coeff)))
            else:
                fisher_tensor = fisher_dict[name].float()
                
                # Apply fisher floor (except for target model if favor_target_model is True)
                if not favor_target_model or i != 0:
                    fisher_tensor = torch.clamp(fisher_tensor, min=fisher_floor)
                
                fisher_weights.append(fisher_tensor * float(coeff))

        # Enhanced Fisher-weighted averaging using the proper equation:
        # merged_param = Σ(coeff * fisher * param) / Σ(coeff * fisher)
        numerator = torch.zeros_like(params[0], dtype=torch.float32)
        denominator = torch.zeros_like(params[0], dtype=torch.float32)
        
        for param_tensor, fisher_weight in zip(params, fisher_weights):
            numerator += fisher_weight * param_tensor
            denominator += fisher_weight

        # Apply epsilon clamping for numerical stability
        denominator = torch.clamp(denominator, min=eps)
        merged[name] = (numerator / denominator).to(params[0].dtype)

    return merged


def create_pairwise_grid_coeffs(n_weightings: int) -> List[Tuple[float, float]]:
    """Create pairwise grid coefficients for two models (from VAD_soup)."""
    n_weightings -= 2
    denom = n_weightings + 1
    weightings = [((i + 1) / denom, 1 - (i + 1) / denom) for i in range(n_weightings)]
    weightings = [(0.0, 1.0)] + weightings + [(1.0, 0.0)]
    weightings.reverse()
    return weightings


def create_random_coeffs(n_models: int, n_weightings: int, seed: Optional[int] = None) -> List[List[float]]:
    """Create random coefficients using Dirichlet distribution (from VAD_soup)."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # Sample from Dirichlet distribution
    alpha = np.ones(n_models)
    coefficients = []
    for _ in range(n_weightings):
        coeff = np.random.dirichlet(alpha)
        coefficients.append(coeff.tolist())
    
    return coefficients


def evaluate_fisher_soup_candidates(state_dicts: List[Dict[str, torch.Tensor]],
                                  fisher_dicts: List[Optional[Dict[str, torch.Tensor]]],
                                  coefficient_candidates: List[List[float]],
                                  evaluation_fn,
                                  *,
                                  fisher_floor: float = 1e-6,
                                  favor_target_model: bool = True,
                                  normalize_fishers: bool = True,
                                  masks: Optional[List[Optional[Dict[str, torch.Tensor]]]] = None,
                                  logger: Optional[logging.Logger] = None) -> Tuple[Dict[str, torch.Tensor], List[float], float]:
    """
    Evaluate multiple coefficient candidates and return the best performing soup.
    
    Args:
        state_dicts: List of model state dictionaries
        fisher_dicts: List of Fisher information dictionaries  
        coefficient_candidates: List of coefficient combinations to try
        evaluation_fn: Function that takes a state dict and returns a score
        fisher_floor: Minimum Fisher value for numerical stability
        favor_target_model: Whether to not apply fisher_floor to first model
        normalize_fishers: Whether to normalize Fisher information
        masks: Optional pruning masks
        logger: Optional logger
        
    Returns:
        Tuple of (best_state_dict, best_coefficients, best_score)
    """
    if logger:
        logger.info(f"Evaluating {len(coefficient_candidates)} coefficient candidates...")
    
    best_state = None
    best_coeffs = None
    best_score = float('-inf')
    
    for i, coeffs in enumerate(coefficient_candidates):
        # Create merged model with current coefficients
        merged_state = enhanced_fisher_weighted_average(
            state_dicts=state_dicts,
            fisher_dicts=fisher_dicts,
            coefficients=coeffs,
            fisher_floor=fisher_floor,
            favor_target_model=favor_target_model,
            normalize_fishers=normalize_fishers,
            masks=masks
        )
        
        # Evaluate this candidate
        score = evaluation_fn(merged_state)
        
        if logger:
            logger.info(f"Candidate {i+1}: coeffs={coeffs}, score={score:.4f}")
        
        # Update best if this is better
        if score > best_score:
            best_score = score
            best_coeffs = coeffs
            best_state = merged_state
    
    if logger and best_coeffs is not None:
        logger.info(f"Best candidate: coeffs={best_coeffs}, score={best_score:.4f}")
    
    return best_state, best_coeffs, best_score