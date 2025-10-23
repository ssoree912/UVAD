import numpy as np

# Load one score file to check structure
scores = np.load("results/cknn_ensemble/base_line/soup_scores_app.npy", allow_pickle=True)

print("Type:", type(scores))
print("Dtype:", scores.dtype)
print("Shape:", scores.shape)
print("Total elements:", len(scores))
print()

# Check if it's a list of arrays (one per video)
if scores.dtype == object:
    print("It's an object array (likely list of video scores)")
    print(f"Number of videos: {len(scores)}")
    print()

    total_bboxes = 0
    for i, video_scores in enumerate(scores[:5]):  # First 5 videos
        if hasattr(video_scores, '__len__'):
            print(f"Video {i}: {len(video_scores)} bboxes, shape: {video_scores.shape if hasattr(video_scores, 'shape') else 'N/A'}")
            total_bboxes += len(video_scores)
        else:
            print(f"Video {i}: scalar value {video_scores}")

    print(f"\nTotal bboxes in first 5 videos: {total_bboxes}")
    print(f"Total bboxes in all videos: {sum(len(s) for s in scores if hasattr(s, '__len__'))}")
else:
    print("It's a regular array")
    print("First 10 values:", scores[:10])

# Load test_lengths to see video structure
print("\n" + "="*80)
test_lengths = np.load("meta/test_lengths_shanghaitech.npy", allow_pickle=True)
print(f"Number of test videos: {len(test_lengths)}")
print(f"Test lengths (frames per video): {test_lengths[:10]}")
print(f"Total test frames: {sum(test_lengths)}")
