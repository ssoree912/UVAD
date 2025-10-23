#!/usr/bin/env python3
"""
Calculate AUROC for all score files in results directory using bbox-to-frame mapping.
"""
import numpy as np
from sklearn.metrics import roc_auc_score
import glob
import os
from codes import bbox, vad

# Configuration
dataset = "shanghaitech"
results_dir = "results/cknn_ensemble/base_line"

# Load metadata
print("Loading metadata...")
frame_labels = np.load(f"meta/frame_labels_{dataset}.npy", allow_pickle=True)
test_lengths = np.load(f"meta/test_lengths_{dataset}.npy", allow_pickle=True)
vnames = vad.get_vnames(dataset, mode='test')

# Prepare per-frame labels
splits = np.insert(np.cumsum(test_lengths), 0, 0)
labels_dict = {}
for vname, i_s, i_e in zip(vnames, splits[:-1], splits[1:]):
    labels_dict[vname] = frame_labels[i_s:i_e]

print(f"Dataset: {dataset}")
print(f"Total videos: {len(vnames)}")
print(f"Total frames: {sum(test_lengths)}")
print(f"Anomaly frames: {frame_labels.sum()}")
print()

# Load bbox mapping
print("Loading bbox-to-frame mapping...")
obj = bbox.VideosFrameBBs.load(dataset, mode='test')
print("Bbox mapping loaded successfully!")
print("=" * 80)
print()

# Find all score files
score_files = sorted(glob.glob(os.path.join(results_dir, "*.npy")))
results = []

for score_file in score_files:
    filename = os.path.basename(score_file)

    try:
        # Load bbox-level scores (video별 배열의 리스트)
        bbox_scores = np.load(score_file, allow_pickle=True)

        # Add scores to bbox object
        obj_copy = bbox.VideosFrameBBs.load(dataset, mode='test')
        obj_copy.add_signal('score', bbox_scores)

        # Convert bbox scores to frame scores using max pooling
        frame_scores_dict = obj_copy.get_framesignal_maximum(keys=['score'])['score']

        # Flatten to calculate AUROC
        all_frame_scores = []
        all_frame_labels = []

        for vname in vnames:
            if vname in frame_scores_dict and vname in labels_dict:
                scores_video = frame_scores_dict[vname]
                labels_video = labels_dict[vname]

                # Ensure same length
                min_len = min(len(scores_video), len(labels_video))
                all_frame_scores.extend(scores_video[:min_len])
                all_frame_labels.extend(labels_video[:min_len])

        all_frame_scores = np.array(all_frame_scores)
        all_frame_labels = np.array(all_frame_labels)

        # Calculate AUROC
        auroc = roc_auc_score(all_frame_labels, all_frame_scores)
        results.append((filename, auroc))

        print(f"✓ {filename}")
        print(f"  AUROC: {auroc:.4f}")
        print()

    except Exception as e:
        print(f"✗ {filename}")
        print(f"  Error: {str(e)}")
        import traceback
        traceback.print_exc()
        print()

print("=" * 80)
print("\nSummary (sorted by AUROC):")
print("-" * 80)

# Sort by AUROC
results.sort(key=lambda x: x[1], reverse=True)

for filename, auroc in results:
    print(f"{auroc:.4f}  {filename}")

# Calculate statistics
if results:
    aurocs = [r[1] for r in results]
    print("\n" + "=" * 80)
    print(f"Best AUROC:  {max(aurocs):.4f}")
    print(f"Worst AUROC: {min(aurocs):.4f}")
    print(f"Mean AUROC:  {np.mean(aurocs):.4f}")
    print(f"Std AUROC:   {np.std(aurocs):.4f}")
    print()

    # Find best model
    best_idx = aurocs.index(max(aurocs))
    print(f"Best model: {results[best_idx][0]}")
