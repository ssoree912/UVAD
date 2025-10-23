# CKNN Soup Methods 🍜

각각 다른 앙상블 방법들이 정리되어 있습니다.

## 📁 폴더 구조

### 🎯 **cknn_score_soup/** (추천)
**올바른 CKNN 앙상블 방법**: k-NN 스코어 레벨에서 Fisher's method로 결합
- `knn_score_soup.py`: 핵심 score-soup 구현
- `cknn_soup_pipeline.py`: 전체 파이프라인
- `run_cknn_soup.py`: 빠른 실행 스크립트

### ⚠️ **appae_weight_soup/** (기존 방법)
**AppAE 가중치 soup**: 모델 파라미터 레벨에서 Fisher 가중 평균 (효과 제한적)
- `enhanced_appae_fisher_*.py`: VAD_soup 개선사항 적용
- `appae_fisher_*.py`: 기본 Fisher 가중 평균
- `run_appae_soup_pipeline.py`: 실행 파이프라인
- `compute_fisher_appae.py`: Fisher 정보 계산

### 📦 **model_soup_basic/**
**단순 가중 평균**: Fisher 정보 없이 균등/수동 가중치로 결합
- `model_soup.py`: 기본 모델 soup

### 🔧 **utils_and_tests/**
**유틸리티 및 테스트**: 공통 도구들
- `launch_appae_runs.py`: 다중 AppAE 훈련 런처
- `test_mask_extraction.py`: 마스크 추출 테스트
- `compute_fisher_vad.py`: VAD Fisher 계산

## 🚀 권장 사용법

### 1. CKNN Score-soup (최우선)
```bash
cd cknn_score_soup
python run_cknn_soup.py
```

### 2. AppAE Weight Soup (비교용)
```bash
cd appae_weight_soup
python enhanced_run_appae_soup_pipeline.py --folders ... --dataset_name ...
```

### 3. 기본 모델 Soup (베이스라인)
```bash
cd model_soup_basic  
python model_soup.py --checkpoints ... --output ...
```

## ⚡ 성능 비교

| 방법 | 적용 대상 | 예상 성능 | 복잡도 |
|------|-----------|-----------|--------|
| **CKNN Score-soup** | k-NN scores | **+2~5% AUROC** | 중간 |
| AppAE Weight Soup | AppAE weights | +0~2% AUROC | 높음 |
| Model Soup Basic | Any weights | +0~1% AUROC | 낮음 |

## 📋 선택 가이드

- **성능 중심**: `cknn_score_soup` 사용
- **기존 방법 개선**: `appae_weight_soup` 사용  
- **빠른 프로토타입**: `model_soup_basic` 사용