#!/usr/bin/env python3
"""
CKNN Score Ensemble: Multiple KNNGrader ensemble via statistical fusion methods
"""

import numpy as np
from typing import List, Optional, Dict, Any
import logging
from pathlib import Path


def rank_normalize(x: np.ndarray) -> np.ndarray:
    """Rank-based normalization to [0,1]"""
    if len(x) <= 1:
        return np.array([0.5] * len(x))
    ranks = np.argsort(np.argsort(x))
    return (ranks + 1) / (len(x) + 1.0)


def empirical_p_upper(x: np.ndarray) -> np.ndarray:
    """Upper tail p-values using empirical CDF"""
    if len(x) <= 1:
        return np.array([0.5] * len(x))
    
    order = np.sort(x)
    
    def pval(t):
        k = np.searchsorted(order, t, side='right')
        F = k / len(order)
        return max(1e-9, 1.0 - F)
    
    return np.vectorize(pval)(x)


def fisher_fusion_pvals(pvals_list: List[np.ndarray], 
                       weights: Optional[np.ndarray] = None) -> np.ndarray:
    """Fisher's method for combining p-values with weights"""
    M = len(pvals_list)
    if M == 0:
        raise ValueError("Empty p-values list")
    
    if weights is None:
        weights = np.ones(M)
    weights = np.array(weights) / (np.sum(weights) + 1e-9)
    
    # Weighted Fisher: T = -2 * sum_i w_i * log(p_i)
    T = np.zeros_like(pvals_list[0])
    for p, w in zip(pvals_list, weights):
        p_clipped = np.clip(p, 1e-9, 1.0)
        T += -2.0 * w * np.log(p_clipped)
    
    return T  # Higher is more anomalous


def stouffer_fusion_pvals(pvals_list: List[np.ndarray], 
                         weights: Optional[np.ndarray] = None) -> np.ndarray:
    """Stouffer's method for combining p-values with weights"""
    from scipy.stats import norm
    
    M = len(pvals_list)
    if M == 0:
        raise ValueError("Empty p-values list")
    
    if weights is None:
        weights = np.ones(M)
    weights = np.array(weights) / (np.sum(weights) + 1e-9)
    
    # p -> Z-score, weighted sum
    Z = np.zeros_like(pvals_list[0], dtype=float)
    for p, w in zip(pvals_list, weights):
        p_clipped = np.clip(p, 1e-12, 1-1e-12)
        Zi = -norm.ppf(p_clipped)  # Convert p-value to Z-score
        Z += w * Zi
    
    return Z  # Higher is more anomalous


def logit_fusion_pvals(pvals_list: List[np.ndarray], 
                      weights: Optional[np.ndarray] = None) -> np.ndarray:
    """Logit-based fusion for combining p-values with weights"""
    M = len(pvals_list)
    if M == 0:
        raise ValueError("Empty p-values list")
    
    if weights is None:
        weights = np.ones(M)
    weights = np.array(weights) / (np.sum(weights) + 1e-9)
    
    # p -> logit, weighted sum
    logit_sum = np.zeros_like(pvals_list[0], dtype=float)
    for p, w in zip(pvals_list, weights):
        p_clipped = np.clip(p, 1e-9, 1-1e-9)
        logit = np.log(p_clipped / (1 - p_clipped))
        logit_sum += w * logit
    
    # Convert back (inverted for higher = more anomalous)
    return -logit_sum


def check_and_correct_monotonicity(scores: np.ndarray, 
                                  expected_direction: str = "higher_anomalous") -> np.ndarray:
    """
    Check score monotonicity and correct if needed
    
    Args:
        scores: Score array
        expected_direction: "higher_anomalous" or "lower_anomalous"
    
    Returns:
        Corrected scores (higher = more anomalous)
    """
    if len(scores) <= 1:
        return scores
    
    # Simple heuristic: check if top quantile has higher variance (more anomalous)
    q75 = np.percentile(scores, 75)
    q25 = np.percentile(scores, 25)
    
    high_scores = scores[scores >= q75]
    low_scores = scores[scores <= q25]
    
    if len(high_scores) > 0 and len(low_scores) > 0:
        high_var = np.var(high_scores)
        low_var = np.var(low_scores)
        
        # If higher scores have lower variance, they might be "more normal"
        # This is a rough heuristic - domain knowledge is better
        if expected_direction == "higher_anomalous":
            if high_var < low_var * 0.5:  # Significantly lower variance
                return -scores  # Flip scores
        elif expected_direction == "lower_anomalous":
            return -scores  # Always flip to higher_anomalous convention
    
    return scores


def soup_scores_videowise(per_model_scores_list: List[List[np.ndarray]], 
                         weights: Optional[np.ndarray] = None,
                         fusion_method: str = "fisher",
                         logger: Optional[logging.Logger] = None) -> List[np.ndarray]:
    """
    Video-wise score fusion to preserve video boundaries
    
    Args:
        per_model_scores_list: List of [video_scores, ...] from different KNN graders
                              Shape: M models × V videos, each video has (T_v,) scores
        weights: Optional weights for each model 
        fusion_method: "fisher", "stouffer", or "logit"
        logger: Optional logger
        
    Returns:
        List of fused scores per video (preserves video structure)
    """
    if len(per_model_scores_list) == 0:
        raise ValueError("Empty scores list")
    
    M = len(per_model_scores_list)
    V = len(per_model_scores_list[0])
    
    # Video consistency validation
    video_counts = [len(m) for m in per_model_scores_list]
    if not all(len(m) == V for m in per_model_scores_list):
        raise ValueError(f"Video count mismatch: expected {V}, got {video_counts}")
    
    # Frame-level consistency check for each video
    for v in range(V):
        frame_counts = [len(per_model_scores_list[m][v]) for m in range(M)]
        if len(set(frame_counts)) > 1:
            if logger:
                logger.warning(f"Video {v} frame count mismatch: {frame_counts}")
    
    if logger:
        logger.info(f"Video consistency validated: {M} models × {V} videos")
    
    if weights is None:
        weights = np.ones(M) / M
    else:
        weights = np.array(weights)
        weights = weights / (weights.sum() + 1e-9)
    
    # Select fusion function
    fusion_funcs = {
        "fisher": fisher_fusion_pvals,
        "stouffer": stouffer_fusion_pvals, 
        "logit": logit_fusion_pvals
    }
    if fusion_method not in fusion_funcs:
        raise ValueError(f"Unknown fusion method: {fusion_method}. Use: {list(fusion_funcs.keys())}")
    
    fusion_func = fusion_funcs[fusion_method]
    
    if logger:
        logger.info(f"Video-wise fusion: {M} models × {V} videos, method={fusion_method}")
    
    fused_per_video = []
    
    for v in range(V):
        pvals_v = []
        video_lengths = []
        
        for m in range(M):
            try:
                s = per_model_scores_list[m][v].astype(float)
                video_lengths.append(len(s))
                
                # Video-wise rank normalization -> p-value (direct calculation)
                if len(s) <= 1:
                    p = np.array([0.5] * len(s))
                else:
                    ranks = np.argsort(np.argsort(s))
                    s_rank = (ranks + 1) / (len(s) + 1.0)
                    p = 1.0 - s_rank  # Upper-tail p-value
                
                pvals_v.append(p)
                
            except Exception as e:
                if logger:
                    logger.warning(f"Video {v}, Model {m}: {e}")
                # Use neutral p-values as fallback
                p = np.array([0.5] * len(per_model_scores_list[0][v]))
                pvals_v.append(p)
        
        # Apply selected fusion method for this video
        T = fusion_func(pvals_v, weights)
        
        # Monotonicity check and correction
        T_corrected = check_and_correct_monotonicity(T, "higher_anomalous")
        if not np.array_equal(T, T_corrected) and logger:
            logger.debug(f"Video {v}: Applied monotonicity correction")
        
        fused_per_video.append(T_corrected)
        
        if logger and v < 3:  # Log first few videos
            logger.debug(f"Video {v}: lengths={video_lengths}, "
                        f"fused_range=[{T.min():.3f}, {T.max():.3f}]")
    
    if logger:
        total_scores = sum(len(v) for v in fused_per_video)
        logger.info(f"Fused {total_scores} total scores across {V} videos")
    
    return fused_per_video


def soup_scores(per_model_scores: List[np.ndarray], 
               weights: Optional[np.ndarray] = None,
               use_rank: bool = True,
               logger: Optional[logging.Logger] = None) -> np.ndarray:
    """
    Legacy function for backward compatibility - use soup_scores_videowise instead
    """
    if logger:
        logger.warning("Using legacy soup_scores - consider soup_scores_videowise for better performance")
    
    # Convert to video-wise format and back
    per_model_scores_list = [[scores] for scores in per_model_scores]
    fused_videos = soup_scores_videowise(per_model_scores_list, weights, logger)
    return fused_videos[0] if fused_videos else np.array([])


class KNNScoreEnsemble:
    """Multiple KNN grader ensemble manager (Score-level late fusion)"""
    
    def __init__(self, fusion_method: str = "fisher", logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.graders = []
        self.weights = []
        self.metadata = []
        self.fusion_method = fusion_method
        self._last_used_weights = None
        self._quality_ratios = []
    
    def add_grader(self, grader, weight: float = 1.0, metadata: Dict[str, Any] = None):
        """Add a KNN grader to the ensemble"""
        self.graders.append(grader)
        self.weights.append(weight)
        self.metadata.append(metadata or {})
        self.logger.info(f"Added grader {len(self.graders)} with weight {weight:.3f}")
    
    def compute_quality_weights(self, train_features_list: List[np.ndarray],
                               method: str = "keep_ratio") -> np.ndarray:
        """Compute quality-based weights for graders"""
        if method == "keep_ratio":
            return self._keep_ratio_weights(train_features_list)
        elif method == "density":
            return self._density_weights(train_features_list) 
        else:
            return np.ones(len(self.graders))
    
    def _keep_ratio_weights(self, train_features_list: List[np.ndarray],
                           n_total_train: Optional[int] = None,
                           target_ratio: float = 0.8) -> np.ndarray:
        """Weight based on cleansing keep ratio with common denominator"""
        if len(train_features_list) == 0:
            return np.ones(len(self.graders))
        
        # Use provided total or estimate from maximum
        if n_total_train is None:
            n_total_train = max(len(f) for f in train_features_list)
        
        weights = []
        ratios = []
        
        for i, features in enumerate(train_features_list):
            ratio = len(features) / max(1, n_total_train)
            ratios.append(ratio)
            weight = np.exp(-abs(ratio - target_ratio) / 0.1)
            weights.append(weight)
            
            # Log individual grader details
            grader_meta = self.metadata[i] if i < len(self.metadata) else {}
            self.logger.info(f"Grader {i} ({grader_meta.get('variation_name', 'unknown')}): "
                           f"keep_ratio={ratio:.3f}, weight={weight:.3f}, "
                           f"k={grader_meta.get('k', 'N/A')}")
        
        weights = np.array(weights)
        self._quality_ratios = ratios  # Store for metadata
        return weights / (weights.sum() + 1e-9)
    
    def _density_weights(self, train_features_list: List[np.ndarray]) -> np.ndarray:
        """Weight based on feature density consistency (fixed seed, exclude diagonal)"""
        weights = []
        rng = np.random.default_rng(0)  # Fixed seed for reproducibility
        
        for i, features in enumerate(train_features_list):
            if len(features) < 10:  # Too few samples
                weights.append(0.1)
                continue
                
            # Sample subset for efficiency with fixed seed
            subset_size = min(1000, len(features))
            indices = rng.choice(len(features), size=subset_size, replace=False)
            subset = features[indices][:100]  # Take first 100 for distance computation
            
            # Compute pairwise distances
            from sklearn.metrics.pairwise import euclidean_distances
            D = euclidean_distances(subset, subset)  # 100x100
            
            # Extract upper triangular (exclude diagonal zeros)
            upper_tri_mask = np.triu_indices_from(D, k=1)
            upper_tri_dists = D[upper_tri_mask]
            
            if len(upper_tri_dists) == 0:
                weights.append(0.1)
                continue
            
            # Weight = 1 / (mean * std) for stability
            mu = np.mean(upper_tri_dists)
            sig = np.std(upper_tri_dists) + 1e-9
            weight = 1.0 / (mu * sig)
            weights.append(weight)
            
            self.logger.debug(f"Grader {i}: density mu={mu:.3f}, sig={sig:.3f}, weight={weight:.3f}")
        
        weights = np.array(weights)
        self.logger.info(f"Density weights: {weights}")
        return weights / (weights.sum() + 1e-9)
    
    def predict(self, test_features_list: List[np.ndarray], 
               use_quality_weights: bool = True,
               train_features_list: Optional[List[np.ndarray]] = None,
               n_total_train: Optional[int] = None,
               check_monotonicity: bool = True) -> List[np.ndarray]:
        """Predict anomaly scores using video-wise score ensemble"""
        
        if len(self.graders) != len(test_features_list):
            raise ValueError(f"Grader count ({len(self.graders)}) != test feature count ({len(test_features_list)})")
        
        # Compute scores from each grader (preserve video structure)
        per_model_scores_video = []
        for i, (grader, test_features) in enumerate(zip(self.graders, test_features_list)):
            scores = grader.grade(test_features)
            
            # Ensure video-wise format
            if isinstance(scores, np.ndarray) and scores.dtype == object:
                video_scores = list(scores)
            elif isinstance(scores, list):
                video_scores = scores
            else:
                # Emergency fallback - treat as single video
                self.logger.warning(f"Grader {i}: Converting non-video format to single video")
                video_scores = [scores.flatten() if hasattr(scores, 'flatten') else np.array(scores)]
            
            # Check and correct monotonicity if requested
            if check_monotonicity:
                corrected_video_scores = []
                for v_scores in video_scores:
                    corrected = check_and_correct_monotonicity(v_scores, "higher_anomalous")
                    corrected_video_scores.append(corrected)
                video_scores = corrected_video_scores
            
            per_model_scores_video.append(video_scores)
            
            total_scores = sum(len(v) for v in video_scores)
            self.logger.debug(f"Grader {i}: {len(video_scores)} videos, {total_scores} total scores")
        
        # Determine weights and store quality ratios
        if use_quality_weights and train_features_list:
            weights = self.compute_quality_weights(train_features_list, n_total_train=n_total_train)
            if n_total_train:
                self._quality_ratios = [len(f) / n_total_train for f in train_features_list]
                
            # Enhanced logging of weights and metadata
            self.logger.info(f"Quality-based weights computed using {len(train_features_list)} training sets")
            for i, (w, meta) in enumerate(zip(weights, self.metadata)):
                ratio = self._quality_ratios[i] if i < len(self._quality_ratios) else "N/A"
                self.logger.info(f"  Grader {i}: {meta.get('variation_name', 'unknown')} -> "
                               f"weight={w:.4f}, keep_ratio={ratio}, k={meta.get('k', 'N/A')}")
        else:
            weights = np.array(self.weights)
            weights = weights / (weights.sum() + 1e-9)
            self.logger.info("Using uniform weights (no quality weighting)")
        
        # Store actual used weights for metadata
        self._last_used_weights = weights.tolist()
        
        # Video-wise fuse scores
        fused_videos = soup_scores_videowise(
            per_model_scores_video, weights, self.fusion_method, logger=self.logger
        )
        
        return fused_videos
    
    def save_metadata(self, path: Path):
        """Save ensemble metadata including actual used weights and quality ratios"""
        metadata = {
            "ensemble_type": "KNN Score Ensemble (Late Fusion)",
            "fusion_method": self.fusion_method,
            "num_graders": len(self.graders),
            "initial_weights": self.weights,
            "used_weights": getattr(self, "_last_used_weights", self.weights),
            "quality_ratios": getattr(self, "_quality_ratios", []),
            "grader_metadata": self.metadata,
            "grader_summary": [
                {
                    "id": i,
                    "variation_name": meta.get("variation_name", f"grader_{i}"),
                    "keep_ratio": meta.get("keep_ratio", None),
                    "k": meta.get("k", None),
                    "threshold": meta.get("threshold", None),
                    "n_train": meta.get("n_train", None)
                }
                for i, meta in enumerate(self.metadata)
            ]
        }
        
        import json
        with open(path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Saved ensemble metadata to {path}")
        
        # Log summary with enhanced details
        if self._last_used_weights:
            total_weight = sum(self._last_used_weights)
            self.logger.info(f"Final ensemble summary (total_weight={total_weight:.4f}):")
            for i, (w, meta) in enumerate(zip(self._last_used_weights, self.metadata)):
                ratio = self._quality_ratios[i] if i < len(self._quality_ratios) else meta.get('keep_ratio', 'N/A')
                self.logger.info(f"  Grader {i}: {meta.get('variation_name', 'unknown')} "
                                f"weight={w:.4f}, keep_ratio={ratio}, k={meta.get('k', 'N/A')}")


# Backward compatibility alias
KNNScoreSoup = KNNScoreEnsemble


def create_multi_cleansing_pipeline(dataset_name: str, uvadmode: str,
                                   cleanse_thresholds: List[float] = [75, 80, 85],
                                   appae_seeds: List[int] = [111, 222, 333],
                                   logger: Optional[logging.Logger] = None):
    """
    Create multiple cleansing variations for ensemble
    
    Args:
        dataset_name: Dataset name
        uvadmode: UVAD mode
        cleanse_thresholds: List of percentile thresholds for cleansing
        appae_seeds: List of AppAE seeds to use
        logger: Optional logger
        
    Returns:
        List of (cleansed_features, metadata) tuples
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    variations = []
    
    for seed in appae_seeds:
        for threshold in cleanse_thresholds:
            # This would be implemented to create cleansed features
            # using different AppAE models and thresholds
            variation_name = f"seed{seed}_thresh{threshold}"
            
            # Placeholder - actual implementation would:
            # 1. Load AppAE model for this seed
            # 2. Generate cleanse scores
            # 3. Apply threshold to get cleansed features
            # 4. Return features and metadata
            
            metadata = {
                "seed": seed,
                "threshold": threshold,
                "variation_name": variation_name
            }
            
            variations.append((None, metadata))  # Placeholder
            logger.info(f"Created variation: {variation_name}")
    
    return variations