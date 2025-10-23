#!/usr/bin/env python3
"""
CKNN Score Ensemble Pipeline: End-to-end ensemble pipeline for CKNN
"""



import os  # Do NOT force CUDA_VISIBLE_DEVICES here; honor external env or --gpu later in main()
import argparse
import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import faiss

# Add project root to path for imports
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from codes import featurebank, grader
from codes.tools.soup_methods.cknn_score_soup.knn_score_ensemble import KNNScoreEnsemble, create_multi_cleansing_pipeline
try:
    from codes.utils import load_config, task
except ImportError:
    # Utils might not exist, that's ok
    pass

# GPU memory management
import gc
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Import for AUROC calculation
try:
    from sklearn.metrics import roc_auc_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def setup_logger(verbose: bool = True) -> logging.Logger:
    """Setup logger for the pipeline"""
    logger = logging.getLogger("cknn_soup_pipeline")
    logger.handlers.clear()
    
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO if verbose else logging.WARNING)
    logger.propagate = False
    
    return logger


def load_cleansed_features(dataset_name: str, uvadmode: str, 
                          cleanse_scorename: str, percentile: float,
                          feature_type: str = 'app') -> Tuple[np.ndarray, np.ndarray]:
    """
    Load features cleansed by AppAE scores with validation
    
    Args:
        dataset_name: Dataset name
        uvadmode: UVAD mode  
        cleanse_scorename: Name of cleanse score file
        percentile: Percentile threshold for cleansing
        feature_type: Type of features ('app' or 'mot')
        
    Returns:
        (cleansed_train_features, test_features)
    """
    # Load original features
    tr_f = featurebank.get(dataset_name, feature_type, 'train', uvadmode)
    te_f = featurebank.get(dataset_name, feature_type, 'test', uvadmode)
    
    # Load cleanse scores and validate
    cleanse_scorepath = f'features/{dataset_name}/cleansescores/{uvadmode}_{cleanse_scorename}_flat.npy'
    if not Path(cleanse_scorepath).exists():
        raise FileNotFoundError(f"Cleanse scores not found: {cleanse_scorepath}")
    
    cleanse_score = np.load(cleanse_scorepath, allow_pickle=True)
    
    # Critical validation: lengths must match
    assert len(cleanse_score) == len(tr_f), \
        f"Cleanse score mismatch: {len(cleanse_score)} vs train {len(tr_f)} for {cleanse_scorename}"
    
    # Apply threshold (assuming lower scores = more normal, so we keep them)
    thres = np.percentile(cleanse_score, percentile)
    mask = cleanse_score <= thres  # Keep more normal samples (lower scores)
    tr_f_cleansed = tr_f[mask]
    
    return tr_f_cleansed, te_f


def apply_coreset_sampling(features: np.ndarray, max_samples: int, 
                          method: str = "random", seed: int = 42) -> np.ndarray:
    """
    Apply coreset sampling to reduce training features
    
    Args:
        features: Training features array
        max_samples: Maximum number of samples to keep
        method: Sampling method ("random", "kmeans", "farthest")
        seed: Random seed
        
    Returns:
        Sampled features
    """
    if len(features) <= max_samples:
        return features
    
    logger = logging.getLogger(__name__)
    logger.info(f"Applying {method} coreset: {len(features):,} -> {max_samples:,} samples")
    
    if method == "random":
        # Simple random sampling
        np.random.seed(seed)
        indices = np.random.choice(len(features), size=max_samples, replace=False)
        return features[indices]
    
    elif method == "kmeans":
        # K-means clustering and take centroids
        try:
            from sklearn.cluster import MiniBatchKMeans
            kmeans = MiniBatchKMeans(n_clusters=max_samples, random_state=seed, batch_size=10000)
            kmeans.fit(features)
            return kmeans.cluster_centers_.astype(features.dtype)
        except ImportError:
            logger.warning("scikit-learn not available, falling back to random sampling")
            return apply_coreset_sampling(features, max_samples, "random", seed)
    
    elif method == "farthest":
        # Farthest point sampling (greedy)
        np.random.seed(seed)
        n_samples = len(features)
        
        # Start with random point
        selected_indices = [np.random.randint(n_samples)]
        
        for _ in range(max_samples - 1):
            distances = []
            for i in range(n_samples):
                if i in selected_indices:
                    distances.append(0)
                else:
                    # Distance to nearest selected point
                    min_dist = min(np.linalg.norm(features[i] - features[j]) 
                                 for j in selected_indices)
                    distances.append(min_dist)
            
            # Select point farthest from selected points
            farthest_idx = np.argmax(distances)
            selected_indices.append(farthest_idx)
        
        return features[selected_indices]
    
    else:
        raise ValueError(f"Unknown coreset method: {method}")


def check_gpu_memory(use_nvml=True):
    """Check available GPU memory using NVML (accurate) or PyTorch (Faiss-blind)"""
    if use_nvml:
        try:
            import pynvml
            pynvml.nvmlInit()
            device = 0  # Assume GPU 0
            handle = pynvml.nvmlDeviceGetHandleByIndex(device)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            used_gb = info.used / 1024**3
            free_gb = info.free / 1024**3  
            total_gb = info.total / 1024**3
            return f"GPU {device} (NVML): {used_gb:.2f}GB used, {free_gb:.2f}GB free / {total_gb:.2f}GB total"
        except ImportError:
            pass  # Fall back to PyTorch
        except Exception as e:
            return f"NVML error: {e}"
    
    # Fallback: PyTorch (only shows PyTorch allocations, misses Faiss)
    if not HAS_TORCH:
        return "N/A (torch not available)"
    
    try:
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            total = torch.cuda.get_device_properties(device).total_memory / 1024**3  # GB
            allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
            cached = torch.cuda.memory_reserved(device) / 1024**3  # GB
            free = total - allocated
            return f"GPU {device} (PyTorch): {free:.1f}GB free / {total:.1f}GB total (allocated: {allocated:.1f}GB, cached: {cached:.1f}GB)"
        else:
            return "CUDA not available"
    except Exception as e:
        return f"Error checking GPU memory: {e}"


def cleanup_gpu_memory():
    """Clean up GPU memory including Faiss resources"""
    if HAS_TORCH and torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Force garbage collection
    gc.collect()
    
    # Try to clean up any lingering Faiss GPU resources
    try:
        import faiss
        # This forces cleanup of any abandoned GPU resources
        faiss.get_mem_usage_kb()
    except:
        pass


def estimate_gpu_memory_needed(n_samples: int, dim: int) -> float:
    """Estimate GPU memory needed for Faiss index in GB"""
    # Each float32 takes 4 bytes
    # Faiss typically needs extra memory for temporary buffers (2-3x the data size)
    data_size_gb = (n_samples * dim * 4) / (1024**3)
    estimated_total_gb = data_size_gb * 3  # Conservative estimate
    return estimated_total_gb


def load_test_labels(dataset_name: str, uvadmode: str) -> np.ndarray:
    """Load test labels for AUROC calculation"""
    try:
        # Try to load labels from standard locations
        label_paths = [
            f'features/{dataset_name}/labels/{uvadmode}_test_labels.npy',
            f'features/{dataset_name}/labels/test_labels.npy',
            f'features/{dataset_name}/test_labels.npy'
        ]
        
        for label_path in label_paths:
            if Path(label_path).exists():
                labels = np.load(label_path, allow_pickle=True)
                return labels
        
        # If no labels found, return None
        return None
        
    except Exception as e:
        logging.getLogger(__name__).warning(f"Could not load test labels: {e}")
        return None


def compute_auroc(scores, labels) -> Optional[float]:
    """Compute AUROC score if possible"""
    if not HAS_SKLEARN or labels is None or scores is None:
        return None
    
    try:
        # Handle video-wise format
        if isinstance(scores, (list, np.ndarray)) and len(scores) > 0:
            if isinstance(scores[0], np.ndarray):
                # Video-wise format - flatten
                scores_flat = np.concatenate([s for s in scores if len(s) > 0])
            else:
                scores_flat = np.array(scores).flatten()
        else:
            scores_flat = np.array(scores).flatten()
        
        # Handle labels similarly
        if isinstance(labels, (list, np.ndarray)) and len(labels) > 0:
            if isinstance(labels[0], np.ndarray):
                labels_flat = np.concatenate([l for l in labels if len(l) > 0])
            else:
                labels_flat = np.array(labels).flatten()
        else:
            labels_flat = np.array(labels).flatten()
        
        # Check lengths match
        if len(scores_flat) != len(labels_flat):
            logging.getLogger(__name__).warning(
                f"Score/label length mismatch: {len(scores_flat)} vs {len(labels_flat)}"
            )
            return None
        
        # Compute AUROC
        auroc = roc_auc_score(labels_flat, scores_flat)
        return float(auroc)
        
    except Exception as e:
        logging.getLogger(__name__).warning(f"AUROC computation failed: {e}")
        return None


def create_grader_variations(dataset_name: str, uvadmode: str,
                           appae_variations: List[str],
                           cleanse_thresholds: List[float],
                           k_values: List[int],
                           feature_type: str = 'app',
                           max_train_samples: int = 500000,
                           coreset_method: str = "random",
                           gpu_ids: Optional[List[int]] = None,
                           logger: Optional[logging.Logger] = None) -> List[Dict]:
    """
    Create multiple KNN grader variations
    
    Args:
        dataset_name: Dataset name
        uvadmode: UVAD mode
        appae_variations: List of AppAE variation names (e.g., ['seed111', 'seed222'])
        cleanse_thresholds: List of cleansing percentiles
        k_values: List of k values for k-NN
        feature_type: Feature type ('app' or 'mot')
        max_train_samples: Maximum training samples per grader
        coreset_method: Method for coreset sampling
        gpu_ids: GPUs to use for KNN (None for CPU)
        logger: Optional logger
        
    Returns:
        List of grader variation dictionaries
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    variations = []
    
    for appae_var in appae_variations:
        for threshold in cleanse_thresholds:
            for k in k_values:
                try:
                    # Load cleansed features
                    cleanse_scorename = f"aerecon_{appae_var}"
                    tr_f_cleansed, te_f = load_cleansed_features(
                        dataset_name, uvadmode, cleanse_scorename, threshold, feature_type
                    )
                    
                    # Apply coreset sampling if needed
                    original_size = len(tr_f_cleansed)
                    if len(tr_f_cleansed) > max_train_samples:
                        tr_f_cleansed = apply_coreset_sampling(
                            tr_f_cleansed, max_train_samples, coreset_method
                        )
                        logger.info(f"Coreset applied: {original_size:,} -> {len(tr_f_cleansed):,}")
                    
                    # Check GPU memory before creating grader
                    memory_info = check_gpu_memory()
                    logger.info(f"GPU memory before grader creation: {memory_info}")
                    
                    # Estimate memory needed
                    estimated_gb = estimate_gpu_memory_needed(len(tr_f_cleansed), tr_f_cleansed.shape[1])
                    logger.info(f"Estimated GPU memory needed: {estimated_gb:.2f} GB")
                    
                    # Clean up GPU memory before creating grader
                    cleanup_gpu_memory()
                    
                    # Create grader
                    grader_key = f'{feature_type}_{appae_var}_th{threshold}_k{k}'
                    gr = grader.KNNGrader(
                        tr_f_cleansed,
                        K=k,
                        key=grader_key,
                        gpu_ids=gpu_ids
                    )
                    
                    # Check GPU memory after creating grader
                    memory_info_after = check_gpu_memory()
                    logger.info(f"GPU memory after grader creation: {memory_info_after}")
                    
                    # Clean up after each grader creation to prevent accumulation
                    cleanup_gpu_memory()
                    
                    # Compute quality metrics based on original cleansed size
                    total_train = len(featurebank.get(dataset_name, feature_type, 'train', uvadmode))
                    keep_ratio = original_size / total_train  # Use original size before coreset
                    
                    variation = {
                        'grader': gr,
                        'appae_var': appae_var,
                        'threshold': threshold,
                        'k': k,
                        'feature_type': feature_type,
                        'train_features': tr_f_cleansed,
                        'test_features': te_f,
                        'keep_ratio': keep_ratio,
                        'n_train': len(tr_f_cleansed),
                        'variation_name': f"{appae_var}_thresh{threshold}_k{k}"
                    }
                    
                    variations.append(variation)
                    logger.info(f"Created variation: {variation['variation_name']}, "
                               f"keep_ratio={keep_ratio:.3f}, n_train={len(tr_f_cleansed)}")
                    
                except Exception as e:
                    logger.warning(f"[{appae_var}|th{threshold}|k{k}] failed: {type(e).__name__}: {e}", exc_info=True)
                    continue
    
    logger.info(f"Created {len(variations)} grader variations")
    return variations


def run_cknn_soup_evaluation(dataset_name: str, uvadmode: str,
                            appae_variations: List[str],
                            cleanse_thresholds: List[float] = [75, 80, 85],
                            k_values: List[int] = [10, 20, 50],
                            feature_types: List[str] = ['app'],
                            weight_method: str = "keep_ratio",
                            output_dir: Optional[Path] = None,
                            max_train_samples: int = 500000,
                            coreset_method: str = "random",
                            gpu_ids: Optional[List[int]] = None,
                            logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """
    Run complete CKNN score ensemble evaluation
    
    Args:
        dataset_name: Dataset name
        uvadmode: UVAD mode
        appae_variations: List of AppAE variations
        cleanse_thresholds: Cleansing percentile thresholds
        k_values: k-NN k values to try
        feature_types: Types of features to use
        weight_method: Method for computing quality weights
        output_dir: Output directory for results
        gpu_ids: List of GPU IDs to use (None for CPU)
        logger: Optional logger
        
    Returns:
        Results dictionary
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load test labels once for AUROC calculation
    test_labels = load_test_labels(dataset_name, uvadmode)
    if test_labels is not None:
        logger.info(f"Loaded test labels for AUROC evaluation")
    else:
        logger.warning("No test labels found - AUROC evaluation disabled")
    
    results = {
        'dataset_name': dataset_name,
        'uvadmode': uvadmode,
        'appae_variations': appae_variations,
        'cleanse_thresholds': cleanse_thresholds,
        'k_values': k_values,
        'feature_types': feature_types,
        'weight_method': weight_method,
        'gpu_ids': gpu_ids,
        'grader_results': [],
        'soup_results': {},
        'roc_results': []  # For JSON format compatible with your example
    }
    
    for feature_type in feature_types:
        logger.info(f"Processing feature type: {feature_type}")
        
        # Create grader variations
        variations = create_grader_variations(
            dataset_name, uvadmode, appae_variations, cleanse_thresholds,
            k_values, feature_type, max_train_samples, coreset_method,
            gpu_ids, logger
        )
        
        if not variations:
            logger.warning(f"No valid variations for feature type {feature_type}")
            continue
        
        # Get original train size for proper weight calculation
        n_total_train = len(featurebank.get(dataset_name, feature_type, 'train', uvadmode))
        
        # Test individual graders (EPHEMERAL: compute scores and immediately delete grader)
        saved_scores = {}  # Store computed scores for soup ensemble
        for i, var in enumerate(variations):
            try:
                logger.info(f"Computing scores for variation {i+1}/{len(variations)}: {var['variation_name']}")
                
                # Check memory before grading
                memory_info = check_gpu_memory()
                logger.info(f"GPU memory before grading: {memory_info}")
                
                scores = var['grader'].grade(var['test_features'])
                
                # Keep video structure for proper evaluation
                if isinstance(scores, np.ndarray) and scores.dtype == object:
                    video_scores = list(scores)
                elif isinstance(scores, list):
                    video_scores = scores
                else:
                    # Fallback: treat as single video
                    video_scores = [scores.flatten() if hasattr(scores, 'flatten') else np.array(scores)]
                
                # CRITICAL: Delete grader immediately after scoring to free GPU memory
                del var['grader']
                cleanup_gpu_memory()
                
                # Check memory after deletion
                memory_info_after = check_gpu_memory()
                logger.info(f"GPU memory after grader deletion: {memory_info_after}")
                
                # Store scores for ensemble later
                saved_scores[var['variation_name']] = video_scores
                
                # Compute stats
                scores_flat = np.concatenate([s for s in video_scores if len(s) > 0])
                
                # Compute AUROC if labels available
                auroc = compute_auroc(video_scores, test_labels)
                
                var_result = {
                    'variation_name': var['variation_name'],
                    'feature_type': feature_type,
                    'scores_stats': {
                        'mean': float(scores_flat.mean()),
                        'std': float(scores_flat.std()),
                        'min': float(scores_flat.min()),
                        'max': float(scores_flat.max()),
                        'n_scores': len(scores_flat),
                        'n_videos': len(video_scores)
                    },
                    'keep_ratio': var['keep_ratio'],
                    'n_train': var['n_train'],
                    'auroc': auroc
                }
                
                results['grader_results'].append(var_result)
                
                # Add to ROC results in your requested format
                roc_entry = {
                    "label": var['variation_name'],
                    "checkpoint": f"grader_variation_{var['variation_name']}",  # Placeholder
                    "score_path": f"computed_scores_{var['variation_name']}",  # Placeholder
                    "auroc": auroc,
                    "raw_output": f"AUROC {dataset_name} ({uvadmode}): {auroc:.6f}" if auroc else "AUROC computation failed"
                }
                results['roc_results'].append(roc_entry)
                
                # Save individual scores (video-wise format for evaluation)
                if output_dir:
                    score_file = output_dir / f"scores_{var['variation_name']}.npy"
                    np.save(score_file, np.array(video_scores, dtype=object), allow_pickle=True)
                    logger.debug(f"Saved individual scores to {score_file}")
                    
            except Exception as e:
                logger.error(f"Failed to evaluate variation {var['variation_name']}: {e}")
                continue
        
        # Create and run soup ensemble FROM PRE-COMPUTED SCORES (no graders!)
        logger.info(f"Creating soup ensemble for {feature_type} from saved scores")
        
        # Collect pre-computed scores and compute weights
        per_model_scores_video = []
        train_features_list = []
        weights = []
        
        for var in variations:
            # Get saved scores
            if var['variation_name'] in saved_scores:
                per_model_scores_video.append(saved_scores[var['variation_name']])
            else:
                logger.warning(f"No saved scores for {var['variation_name']}, skipping from ensemble")
                continue
            
            # Compute quality weight
            if weight_method == "keep_ratio":
                weight = np.exp(-abs(var['keep_ratio'] - 0.8) / 0.1)
            else:
                weight = 1.0
            weights.append(weight)
            
            train_features_list.append(var['train_features'])
        
        # Use static method to fuse from pre-computed scores (NO RE-GRADING!)
        try:
            logger.info(f"Fusing ensemble from {len(per_model_scores_video)} pre-computed score sets")
            soup_scores = KNNScoreEnsemble.fuse_from_scores(
                per_model_scores_video, 
                weights=np.array(weights), 
                fusion_method="fisher",
                logger=logger
            )
            
            # Compute stats from video-wise scores
            soup_scores_flat = np.concatenate([s for s in soup_scores if len(s) > 0])
            
            # Compute AUROC for ensemble
            soup_auroc = compute_auroc(soup_scores, test_labels)
            
            soup_result = {
                'feature_type': feature_type,
                'n_graders': len(variations),
                'scores_stats': {
                    'mean': float(soup_scores_flat.mean()),
                    'std': float(soup_scores_flat.std()),
                    'min': float(soup_scores_flat.min()),
                    'max': float(soup_scores_flat.max()),
                    'n_scores': len(soup_scores_flat),
                    'n_videos': len(soup_scores)
                },
                'weight_method': weight_method,
                'auroc': soup_auroc
            }
            
            results['soup_results'][feature_type] = soup_result
            
            # Add ensemble result to ROC results
            ensemble_roc_entry = {
                "label": f"ensemble_{feature_type}_{weight_method}",
                "checkpoint": f"ensemble_model_{feature_type}",
                "score_path": f"ensemble_scores_{feature_type}.npy",
                "auroc": soup_auroc,
                "raw_output": f"AUROC {dataset_name} ({uvadmode}): {soup_auroc:.6f}" if soup_auroc else "AUROC computation failed"
            }
            results['roc_results'].append(ensemble_roc_entry)
            
            # Save soup scores (video-wise format for evaluation compatibility)
            if output_dir:
                soup_score_file = output_dir / f"soup_scores_{feature_type}.npy"
                np.save(soup_score_file, np.array(soup_scores, dtype=object), allow_pickle=True)
                logger.info(f"Saved soup scores to {soup_score_file}")
                
                # Save metadata
                soup.save_metadata(output_dir / f"soup_metadata_{feature_type}.json")
                
        except Exception as e:
            logger.error(f"Failed to run soup for {feature_type}: {e}")
            continue
    
    # Save complete results
    if output_dir:
        results_file = output_dir / "cknn_soup_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved complete results to {results_file}")
    
    return results


def main():
    """Main function for CKNN score ensemble pipeline"""
    parser = argparse.ArgumentParser(description="CKNN Score Ensemble Pipeline")
    parser.add_argument("--dataset_name", required=True, 
                       choices=["shanghaitech", "avenue", "ped2"])
    parser.add_argument("--uvadmode", required=True, choices=["merge", "partial"])
    parser.add_argument("--appae_variations", nargs="+", required=True,
                       help="AppAE variation names (e.g., seed111 seed222)")
    parser.add_argument("--cleanse_thresholds", nargs="+", type=float, 
                       default=[75, 80, 85], help="Cleansing percentile thresholds")
    parser.add_argument("--k_values", nargs="+", type=int, default=[10, 20, 50],
                       help="k-NN k values")
    parser.add_argument("--feature_types", nargs="+", default=["app"],
                       choices=["app", "mot"], help="Feature types to process")
    parser.add_argument("--weight_method", default="keep_ratio",
                       choices=["keep_ratio", "density", "uniform"])
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for results")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("--gpu", type=int, default=None, 
                       help="Single GPU device ID (deprecated when using --gpu_list).")
    parser.add_argument("--gpu_list", nargs="+", type=int, default=None,
                       help="Explicit list of GPU IDs for multi-GPU KNN (e.g., 0 1 2).")
    parser.add_argument("--max_train_samples", type=int, default=50000,
                       help="Max training samples per KNN grader (reduces GPU memory usage)")
    parser.add_argument("--coreset_method", choices=["random", "kmeans", "farthest"], default="random",
                       help="Method for selecting coreset when reducing training samples")
    parser.add_argument("--multi_gpu", action="store_true",
                       help="Use all available GPUs for KNN graders if --gpu_list is not provided.")
    
    args = parser.parse_args()

    # Determine GPU usage
    gpu_ids: Optional[List[int]] = None
    if args.gpu_list:
        gpu_ids = [int(g) for g in args.gpu_list]
    elif args.gpu is not None:
        gpu_ids = [int(args.gpu)]
    elif args.multi_gpu:
        try:
            gpu_count = faiss.get_num_gpus()
            if gpu_count > 0:
                gpu_ids = list(range(gpu_count))
        except Exception:
            gpu_ids = None
    else:
        try:
            gpu_count = faiss.get_num_gpus()
            if gpu_count > 0:
                gpu_ids = [0]
        except Exception:
            gpu_ids = None

    logger = setup_logger(args.verbose)

    logger.info("Starting CKNN Score Ensemble Pipeline")
    logger.info(f"Dataset: {args.dataset_name}, Mode: {args.uvadmode}")
    logger.info(f"AppAE variations: {args.appae_variations}")
    logger.info(f"Cleanse thresholds: {args.cleanse_thresholds}")
    logger.info(f"k values: {args.k_values}")
    if gpu_ids:
        logger.info(f"Using GPUs: {gpu_ids}")
    else:
        logger.info("No GPUs specified or detected; falling back to CPU indices.")
    
    try:
        results = run_cknn_soup_evaluation(
            dataset_name=args.dataset_name,
            uvadmode=args.uvadmode,
            appae_variations=args.appae_variations,
            cleanse_thresholds=args.cleanse_thresholds,
            k_values=args.k_values,
            feature_types=args.feature_types,
            weight_method=args.weight_method,
            output_dir=Path(args.output_dir) if args.output_dir else None,
            max_train_samples=args.max_train_samples,
            coreset_method=args.coreset_method,
            gpu_ids=gpu_ids,
            logger=logger
        )
        
        logger.info("CKNN Score Ensemble Pipeline completed successfully")
        
        # Print summary
        logger.info("=== CKNN Score Ensemble Results ===")
        
        # Print individual grader results
        for grader_result in results['grader_results']:
            auroc_str = f"{grader_result['auroc']:.6f}" if grader_result['auroc'] else "N/A"
            logger.info(f"Individual: {grader_result['variation_name']} -> AUROC: {auroc_str}")
        
        # Print ensemble results
        for feature_type, soup_result in results['soup_results'].items():
            auroc_str = f"{soup_result['auroc']:.6f}" if soup_result['auroc'] else "N/A"
            logger.info(f"Ensemble ({feature_type}): {soup_result['n_graders']} graders -> AUROC: {auroc_str}")
        
        # Print ROC results in your requested format
        logger.info("=== ROC Results (JSON format) ===")
        roc_json = json.dumps(results['roc_results'], indent=2)
        print(roc_json)  # Print to stdout for easy extraction
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
