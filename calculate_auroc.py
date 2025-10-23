import numpy as np
from sklearn.metrics import roc_auc_score
import glob
import os
from pathlib import Path

# Configuration
dataset = "shanghaitech"
results_dir = "results/cknn_ensemble/base_line"

# Load labels
frame_labels = np.load(f"meta/frame_labels_{dataset}.npy", allow_pickle=True)
test_lengths = np.load(f"meta/test_lengths_{dataset}.npy", allow_pickle=True)

# Prepare per-frame labels
splits = np.insert(np.cumsum(test_lengths), 0, 0)
labels = [frame_labels[s:e] for s, e in zip(splits[:-1], splits[1:])]
labels_flat = np.concatenate([l for l in labels if len(l) > 0])

print(f"Total frames: {len(labels_flat)}")
print(f"Anomaly frames: {labels_flat.sum()}")
print(f"Normal frames: {(1-labels_flat).sum()}\n")
print("="*80)

# Find all score files
score_files = sorted(glob.glob(os.path.join(results_dir, "*.npy")))

results = []

for score_file in score_files:
    filename = os.path.basename(score_file)

    try:
        # Load scores
        scores = np.load(score_file, allow_pickle=True)

        # Flatten scores (handle both list of arrays and single array)
        if isinstance(scores, np.ndarray) and scores.dtype == object:
            scores_flat = np.concatenate([s for s in scores if len(s) > 0])
        else:
            scores_flat = scores.flatten()

        # Check length match
        if len(scores_flat) != len(labels_flat):
            print(f"⚠️  {filename}")
            print(f"   Length mismatch: scores={len(scores_flat)}, labels={len(labels_flat)}")
            print()
            continue

        # Calculate AUROC
        auroc = roc_auc_score(labels_flat, scores_flat)
        results.append((filename, auroc))

        print(f"✓ {filename}")
        print(f"  AUROC: {auroc:.4f}")
        print()

    except Exception as e:
        print(f"✗ {filename}")
        print(f"  Error: {str(e)}")
        print()

print("="*80)
print("\nSummary (sorted by AUROC):")
print("-"*80)

# Sort by AUROC
results.sort(key=lambda x: x[1], reverse=True)

for filename, auroc in results:
    print(f"{auroc:.4f}  {filename}")

# Calculate statistics
if results:
    aurocs = [r[1] for r in results]
    print("\n" + "="*80)
    print(f"Best AUROC:  {max(aurocs):.4f}")
    print(f"Worst AUROC: {min(aurocs):.4f}")
    print(f"Mean AUROC:  {np.mean(aurocs):.4f}")
    print(f"Std AUROC:   {np.std(aurocs):.4f}")
