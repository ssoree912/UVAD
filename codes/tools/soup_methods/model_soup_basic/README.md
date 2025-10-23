# Model Soup Basic 📦

**단순 가중 평균**: Fisher 정보 없이 균등/수동 가중치로 모델 결합

## 개념

가장 기본적인 model soup 방법입니다. 복잡한 Fisher 계산 없이 단순 가중 평균으로 모델들을 결합합니다.

## 파일 구조

- `model_soup.py`: 기본 모델 soup 구현

## 실행 방법

```bash
python model_soup.py \
    --checkpoints model1.pkl model2.pkl model3.pkl \
    --weights 0.4 0.3 0.3 \
    --output soup_model.pkl
```

## 특징

- **단순함**: Fisher 계산 불필요
- **빠름**: 오버헤드 최소
- **제한적**: 정교한 가중치 없음

## 사용 사례

- **빠른 프로토타입**: 개념 검증용
- **베이스라인**: 다른 방법과 비교 기준
- **단순 앙상블**: 복잡성 불필요한 경우

## 한계점

- 모델 간 중요도 차이 반영 불가
- 파라미터별 세밀한 조정 불가
- 성능 향상 제한적