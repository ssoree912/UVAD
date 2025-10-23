# AppAE Weight Soup ⚠️

**AppAE 가중치 soup**: 모델 파라미터 레벨에서 Fisher 가중 평균

## ⚠️ 주의사항

이 방법은 **효과가 제한적**입니다. CKNN에서 AppAE는 cleansing 역할만 하고, 실제 detection은 k-NN이 담당하기 때문입니다.

**권장**: `../cknn_score_soup/` 사용

## 파일 구조

### Enhanced 버전 (VAD_soup 개선 적용)
- `enhanced_appae_fisher_utils.py`: VAD_soup 개선사항 적용
- `enhanced_appae_fisher_soup.py`: 단독 실행 스크립트
- `enhanced_run_appae_soup_pipeline.py`: 전체 파이프라인

### Basic 버전
- `appae_fisher_utils.py`: 기본 Fisher 가중 평균
- `appae_fisher_soup.py`: 단독 실행 스크립트  
- `run_appae_soup_pipeline.py`: 기본 파이프라인

### 유틸리티
- `compute_fisher_appae.py`: Fisher 정보 계산

## 실행 방법

### Enhanced 버전 (권장)
```bash
python enhanced_run_appae_soup_pipeline.py \
    --folders artifacts/shanghaitech/merge/seed111 seed222 seed333 \
    --dataset_name shanghaitech \
    --uvadmode merge \
    --config configs/config_shanghaitech.yaml \
    --target_folder artifacts/soup_result \
    --run_eval --eval_include_folders \
    --verbose
```

### Basic 버전
```bash
python run_appae_soup_pipeline.py \
    --folders artifacts/shanghaitech/merge/seed111 seed222 seed333 \
    --dataset_name shanghaitech \
    --uvadmode merge \
    --output soup_model.pkl
```

## 한계점

1. **간접적 영향**: AppAE soup → cleansing 개선 → k-NN 성능에 간접 영향
2. **제한적 성능**: k-NN이 최종 성능 결정, AppAE soup 효과 제한적  
3. **복잡성**: Fisher 계산 오버헤드 대비 성능 향상 미미

## 대안

**CKNN Score-soup** 사용 권장: `../cknn_score_soup/`