#!/usr/bin/env python3
"""
CKNN evaluation helper.
- 이미 계산된 score(.npy) 파일에서 CKNN 논문과 동일한 방식으로 AUROC를 계산합니다.
- bbox → frame max, temporal smoothing(sigma=3) → 영상 단위 sentinel padding → roc_auc_score.
- meta/frame_labels_<dataset>.npy, meta/test_lengths_<dataset>.npy를 사용합니다.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.ndimage import gaussian_filter1d

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def load_labels_and_frame_counts(dataset: str) -> Tuple[List[np.ndarray], np.ndarray]:
    """영상별 프레임 라벨과 프레임 수를 meta/*.npy에서 읽어온다."""
    labels_path = Path(f"meta/frame_labels_{dataset}.npy")
    lengths_path = Path(f"meta/test_lengths_{dataset}.npy")

    if not labels_path.exists() or not lengths_path.exists():
        raise FileNotFoundError(
            f"Meta files not found: {labels_path} or {lengths_path}"
        )

    labels_raw = np.load(labels_path, allow_pickle=True)
    lengths_raw = np.load(lengths_path, allow_pickle=True)

    # 영상별 프레임 수 계산
    if lengths_raw.dtype == object:
        frame_counts = np.array([len(seq) for seq in lengths_raw], dtype=int)
    else:
        frame_counts = lengths_raw.astype(int)

    # 영상별 라벨 구성
    if labels_raw.dtype == object and len(labels_raw) == len(frame_counts):
        video_labels = [np.asarray(v, dtype=np.float32) for v in labels_raw]
    else:
        # flatten된 경우 test_lengths로 분할
        splits = np.insert(np.cumsum(frame_counts), 0, 0)
        if splits[-1] > len(labels_raw):
            raise ValueError("frame_labels 배열 길이가 예상보다 짧습니다.")
        video_labels = [
            labels_raw[s:e] for s, e in zip(splits[:-1], splits[1:])
        ]

    return video_labels, frame_counts


def aggregate_frame_score(item) -> float:
    """단일 프레임에서 여러 bbox score가 있을 때 max 사용."""
    arr = np.asarray(item, dtype=np.float32)
    if arr.ndim == 0:
        return float(arr)
    return float(arr.max()) if arr.size else 0.0


def reshape_scores_to_frames(
    raw_scores, frame_counts: np.ndarray
) -> List[np.ndarray]:
    """
    score가 (1) 영상별 object array거나 (2) flatten된 프레임 배열인 경우
    프레임 단위로 재구성한다.
    """
    raw_scores = np.array(raw_scores, dtype=object)
    num_videos = len(frame_counts)
    total_frames = int(frame_counts.sum())

    # 영상별 object array인 경우
    if len(raw_scores) == num_videos:
        per_video = []
        for video_scores in raw_scores:
            if isinstance(video_scores, (list, tuple, np.ndarray)):
                per_frame = [aggregate_frame_score(v) for v in video_scores]
                per_video.append(np.asarray(per_frame, dtype=np.float32))
            else:
                per_video.append(
                    np.asarray([aggregate_frame_score(video_scores)], dtype=np.float32)
                )
        return per_video

    # flatten 형태
    if len(raw_scores) != total_frames:
        raise ValueError(
            f"Score length {len(raw_scores)} != total frames {total_frames}"
        )

    per_video = []
    idx = 0
    for frame_count in frame_counts:
        frames = [
            aggregate_frame_score(raw_scores[idx + j]) for j in range(int(frame_count))
        ]
        idx += int(frame_count)
        per_video.append(np.asarray(frames, dtype=np.float32))
    return per_video


def apply_temporal_smoothing(
    per_video_scores: List[np.ndarray], sigma: float
) -> List[np.ndarray]:
    """CKNN 논문처럼 가우시안 시간축 스무딩 적용."""
    if sigma <= 0:
        return [np.asarray(v, dtype=np.float32) for v in per_video_scores]

    smoothed = []
    for scores in per_video_scores:
        if len(scores) <= 1:
            smoothed.append(np.asarray(scores, dtype=np.float32))
        else:
            smoothed.append(
                gaussian_filter1d(scores.astype(np.float32), sigma=sigma).astype(np.float32)
            )
    return smoothed


def compute_auroc_like_main2(
    per_video_scores: List[np.ndarray],
    video_labels: List[np.ndarray],
    dataset_name: str,
) -> Optional[float]:
    """
    main2_evaluate.py의 calc_AUROC_d와 동일한 방식으로 AUROC 계산.
    (영상별 padding [0], [1000000], ROC → 평균)
    """
    from codes import vad  # CKNN util 모듈
    vnames = vad.get_vnames(dataset_name, mode="test")

    aurocs = []
    for idx, (scores, labels) in enumerate(zip(per_video_scores, video_labels)):
        if len(scores) == 0 or len(labels) == 0:
            continue

        min_len = min(len(scores), len(labels))
        if len(scores) != len(labels):
            logger.debug(
                "Frame length mismatch (video %s): scores=%d, labels=%d (truncated to %d)",
                vnames[idx],
                len(scores),
                len(labels),
                min_len,
            )

        y_true = np.concatenate([[0], labels[:min_len], [1]])
        y_pred = np.concatenate([[0], scores[:min_len], [1000000]])

        try:
            auroc = roc_auc_score(y_true, y_pred)
            aurocs.append(auroc * 100.0)  # 퍼센트 단위
        except ValueError:
            continue

    return float(np.mean(aurocs)) if aurocs else None


# --------------------------------------------------------------------------- #
# Main entry
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Compute CKNN AUROC from pre-computed scores."
    )
    parser.add_argument(
        "--dataset_name",
        default="shanghaitech",
        choices=["shanghaitech", "avenue", "ped2"],
    )
    parser.add_argument(
        "--scores_dir",
        type=str,
        default="results/cknn_ensemble/base_line",
        help="scores_*.npy / soup_scores_*.npy가 있는 디렉터리",
    )
    parser.add_argument("--pattern", type=str, default="*.npy", help="파일 글롭 패턴")
    parser.add_argument(
        "--sigma",
        type=float,
        default=3.0,
        help="Temporal smoothing sigma (CKNN 기본값은 3)",
    )
    parser.add_argument("--output", type=str, default=None, help="JSON 결괏값 저장 경로")
    parser.add_argument("--verbose", action="store_true", help="Verbose 로그")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s",
    )

    # 1. 라벨과 프레임 수 로드
    video_labels, frame_counts = load_labels_and_frame_counts(args.dataset_name)

    # 2. 점수 파일 나열
    score_dir = Path(args.scores_dir)
    files = sorted(score_dir.glob(args.pattern))
    if not files:
        print(f"No files matching {args.pattern} in {score_dir}")
        return

    print(f"[AUROC] dataset={args.dataset_name}, sigma={args.sigma}")
    results = []

    # 3. 각 점수 파일 평가
    for score_path in files:
        try:
            raw_scores = np.load(score_path, allow_pickle=True)
            per_video_scores = reshape_scores_to_frames(raw_scores, frame_counts)
            smoothed_scores = apply_temporal_smoothing(per_video_scores, args.sigma)
            auroc = compute_auroc_like_main2(
                smoothed_scores, video_labels, args.dataset_name
            )
            if auroc is None:
                raise ValueError("AUROC could not be computed (no valid videos)")
            print(f"{score_path.name:<40} AUROC = {auroc:.3f}%")
            results.append({"file": score_path.name, "auroc_percent": auroc})
        except Exception as exc:
            print(f"{score_path.name:<40} 오류 → {exc}")
            results.append({"file": score_path.name, "error": str(exc)})

    # 4. JSON 저장 옵션
    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2))
        print(f"\nSaved AUROC summary to {args.output}")


if __name__ == "__main__":
    main()
