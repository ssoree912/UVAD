import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, List

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


def fisher_weighted_average(state_dicts: List[Dict[str, torch.Tensor]],
                            fisher_dicts: List[Optional[Dict[str, torch.Tensor]]],
                            coefficients: Iterable[float],
                            *,
                            eps: float = 1e-8) -> Dict[str, torch.Tensor]:
    coeffs = list(coefficients)
    if len(coeffs) != len(state_dicts):
        raise ValueError("Number of coefficients must match number of models.")

    merged: Dict[str, torch.Tensor] = {}
    param_names = state_dicts[0].keys()

    for name in param_names:
        params = [state[name].detach().cpu() for state in state_dicts]
        weights = []

        for coeff, fisher in zip(coeffs, fisher_dicts):
            if fisher is None:
                weights.append(torch.full_like(params[0], float(coeff)))
            else:
                tensor = fisher.get(name)
                if tensor is None:
                    tensor = torch.ones_like(params[0])
                weights.append(tensor.float() * float(coeff))

        numerator = torch.zeros_like(params[0], dtype=torch.float32)
        denominator = torch.zeros_like(params[0], dtype=torch.float32)
        for param_tensor, weight_tensor in zip(params, weights):
            numerator += weight_tensor * param_tensor.float()
            denominator += weight_tensor

        denominator = torch.clamp(denominator, min=eps)
        merged[name] = (numerator / denominator).to(params[0].dtype)

    return merged
