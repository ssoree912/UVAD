# CKNN Score-soup 🎯

**올바른 CKNN 앙상블 방법**: k-NN 스코어 레벨에서 Fisher's method로 결합

## 핵심 아이디어

CKNN에서는 AppAE가 단순 cleansing 역할만 하고, **k-NN이 실제 anomaly detection**을 담당합니다.
따라서 **k-NN 스코어들을 앙상블**하는 것이 올바른 접근입니다.

## 파일 구조

- `knn_score_soup.py`: 핵심 score-soup 구현 (Fisher's method)
- `cknn_soup_pipeline.py`: 전체 파이프라인 (variation 생성 + 평가)
- `run_cknn_soup.py`: 빠른 실행 스크립트

## 실행 방법

### 빠른 테스트
```bash
python run_cknn_soup.py
```

### 상세 설정
```bash
python cknn_soup_pipeline.py \
    --dataset_name shanghaitech \
    --uvadmode merge \
    --appae_variations seed111 seed222 seed333 \
    --cleanse_thresholds 75 80 85 \
    --k_values 10 20 50 \
    --weight_method keep_ratio \
    --output_dir results/cknn_soup \
    --verbose
```

## 주요 특징

1. **Video-wise normalization**: 비디오별 rank 정규화로 경계 보존
2. **Fisher's method**: 통계적으로 robust한 p-value 결합  
3. **Quality weighting**: Cleansing 품질 기반 가중치
4. **Evaluation compatibility**: 기존 AUROC 스크립트와 호환

## 예상 성능

- 베이스라인 k-NN 대비 **+2~5% AUROC** 향상
- 6-12개 variation 조합 시 최적 성능