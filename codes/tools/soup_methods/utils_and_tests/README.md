# Utils and Tests 🔧

**유틸리티 및 테스트**: 공통 도구들

## 파일 구조

### 훈련 도구
- `launch_appae_runs.py`: 다중 AppAE 훈련 런처
  - 여러 seed, pruning rate 조합으로 AppAE 모델들 일괄 훈련
  - GPU 분산 처리 지원

### 테스트 도구  
- `test_mask_extraction.py`: 마스크 추출/저장 기능 테스트
  - Pruning mask 추출 검증
  - 저장/로딩 호환성 테스트

### 외부 도구
- `compute_fisher_vad.py`: VAD 모델용 Fisher 정보 계산
  - VAD_soup 프로젝트와의 호환성

## 실행 방법

### 다중 AppAE 훈련
```bash
python launch_appae_runs.py \
    --dataset_name shanghaitech \
    --uvadmode merge \
    --base_seeds 111 222 333 \
    --gpus 0 1 2 \
    --config configs/config_shanghaitech.yaml
```

### 마스크 추출 테스트
```bash
python test_mask_extraction.py
```

## 의존성

- 다른 soup methods와 독립적
- 기본 codes/ 모듈들과 연동