import numpy as np
import os

dataset = "shanghaitech"

# Check meta directory for bbox-related files
meta_dir = f"meta"
if os.path.exists(meta_dir):
    print("Files in meta directory:")
    for f in sorted(os.listdir(meta_dir)):
        print(f"  {f}")
    print()

# Check feature directory structure
feature_dir = f"features/{dataset}/test"
if os.path.exists(feature_dir):
    print("Files in feature test directory:")
    for f in sorted(os.listdir(feature_dir)):
        fpath = os.path.join(feature_dir, f)
        if f.endswith('.npy'):
            data = np.load(fpath, allow_pickle=True)
            print(f"  {f}: shape={data.shape}, dtype={data.dtype}")
            if data.dtype == object and len(data) > 0:
                print(f"    First video: {len(data[0])} bboxes")
                print(f"    Total videos: {len(data)}")
                total_bboxes = sum(len(v) for v in data if len(v) > 0)
                print(f"    Total bboxes: {total_bboxes}")
    print()

# Load test feature to understand structure
print("="*80)
print("Analyzing feature structure:")
print("="*80)

try:
    app_features = np.load(f"{feature_dir}/app.npy", allow_pickle=True)
    print(f"app.npy: {len(app_features)} videos")

    test_lengths = np.load(f"meta/test_lengths_{dataset}.npy", allow_pickle=True)
    print(f"test_lengths: {len(test_lengths)} videos, total frames: {sum(test_lengths)}")
    print()

    # Compare structure
    print("Video-by-video comparison (first 10 videos):")
    print(f"{'Video':<8} {'Frames':<10} {'Bboxes':<10} {'Bbox/Frame':<12}")
    print("-" * 45)

    for i in range(min(10, len(app_features))):
        n_frames = test_lengths[i]
        n_bboxes = len(app_features[i]) if len(app_features[i].shape) > 0 else 0
        ratio = n_bboxes / n_frames if n_frames > 0 else 0
        print(f"{i:<8} {n_frames:<10} {n_bboxes:<10} {ratio:<12.2f}")

    print()
    print("If bbox/frame ratio ≈ 1, then it's likely 1 bbox per frame (max pooled)")
    print("If bbox/frame ratio > 1, then we need bbox-to-frame mapping info")

except Exception as e:
    print(f"Error: {e}")
