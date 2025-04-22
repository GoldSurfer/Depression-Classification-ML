# Machine Learning Project

이 상 현

2023713111
인공지능융합학과

## 1. Problem Definition
본 프로젝트는 오디오 및 비주얼 특징 데이터를 활용하여 우울증 여부를 분류하는 이진 분류(Classification) 문제입니다. 학습 데이터는 발화자의 얼굴과 목소리로부터 추출된 다양한 특성(features)으로 구성되어 있으며, 각 샘플에는 우울증 여부를 나타내는 라벨(0: 비우울, 1: 우울)이 포함되어 있습니다. 이 데이터를 사용하여 우울증을 정확하게 예측할 수 있는 머신러닝 모델을 개발하는 것이 목표입니다.

## 2. Data
### 2.1. Data Description
데이터셋은 다음과 같은 구성을 가지고 있습니다:
- 훈련 데이터(`train.csv`): 라벨이 포함된 학습용 데이터
- 테스트 데이터(`test.csv`): 라벨이 없는 예측용 데이터
- 샘플 제출 파일(`sample_submission.csv`): 제출 양식 예시

데이터 특성:
- 감정 관련 특성(emotion_angry_mean, emotion_disgust_mean 등): 얼굴 표정에서 추출한 감정 관련 값
- 음성 특성(F0semitoneFrom27.5Hz, loudness_sma3 등): 목소리에서 추출한 음향 특성
- 특성 통계값: 각 특성의 평균(mean), 분산(var), 최소값(min), 최대값(max) 정보 포함
- 데이터의 형태:
  - 훈련 데이터: 행(샘플 수) x 열(특성 수 + 라벨)
  - 테스트 데이터: 행(샘플 수) x 열(특성 수)

### 2.2. Data Pre-processing
데이터 전처리 과정은 다음과 같이 수행했습니다:

1. **특성과 라벨 분리**: 
   - 훈련 데이터에서 라벨(label) 컬럼 분리
   - 테스트 데이터에서 ID 컬럼 분리

2. **학습/검증 데이터 분할**:
   - 훈련 데이터의 80%는 학습에, 20%는 검증에 사용
   - stratify 옵션을 통해 원본 데이터의 라벨 분포 유지

3. **표준화(Standardization)**:
   - StandardScaler를 사용하여 특성 값들을 평균 0, 표준편차 1로 변환
   - 학습 데이터로 학습된 스케일러를 검증 및 테스트 데이터에 적용

4. **특성 선택**:
   - RandomForest 기반의 SelectFromModel을 사용하여 중요한 특성만 선택
   - 데이터 차원을 축소하여 모델의 과적합 방지 및 훈련 속도 향상
   - 'median' threshold를 적용하여 중간값보다 높은 중요도를 가진 특성만 선택

## 3. Analysis
데이터 분석 결과는 다음과 같습니다:

1. **라벨 분포**:
   - 클래스 분포가 불균형함(불균형 데이터)
   - 이를 해결하기 위해 class_weight='balanced' 옵션 사용

2. **감정 특성과 우울증의 관계**:
   - 감정 특성 중 sadness(슬픔)와 우울증 사이에 양의 상관관계가 있음
   - happiness(행복)와 우울증 사이에는 음의 상관관계가 있음
   - neutral(중립) 감정은 우울증과 약한 상관관계를 보임

3. **음성 특성과 우울증의 관계**:
   - 음성의 특정 주파수 특성이 우울증과 상관관계가 있음
   - 우울증 그룹에서 음성의 톤과 리듬이 일반 그룹과 다른 패턴을 보임

4. **특성 중요도 분석**:
   - 감정 관련 특성들이 상위 중요도를 차지
   - 음성 관련 특성 중에서는 F0(기본 주파수) 관련 특성들이 중요
   - 우울증 분류에 있어 감정 표현과 음성 특징이 모두 중요한 역할을 함

## 4. Model
### 4.1. Classifier
다양한 분류 알고리즘을 구현하고 비교했습니다:

1. **로지스틱 회귀(Logistic Regression)**:
   - 선형 모델을 사용한 이진 분류
   - 장점: 해석 용이, 빠른 학습, 오버피팅 가능성 낮음
   - 하이퍼파라미터: C(규제 강도), solver, class_weight

2. **SVM(Support Vector Machine)**:
   - 데이터 포인트를 구분하는 최적의 경계면을 찾는 알고리즘
   - 장점: 고차원 데이터에 효과적, 비선형 분류 가능
   - 하이퍼파라미터: C, kernel, gamma, class_weight

3. **랜덤 포레스트(Random Forest)**:
   - 여러 결정 트리의 앙상블 기법
   - 장점: 과적합에 강함, 중요 특성 파악 가능, 높은 정확도
   - 하이퍼파라미터: n_estimators, max_depth, min_samples_split, min_samples_leaf

4. **그래디언트 부스팅(Gradient Boosting)**:
   - 약한 학습기(결정 트리)를 순차적으로 학습하여 이전 모델의 오차를 보완
   - 장점: 높은 예측 성능, 다양한 손실 함수 사용 가능
   - 하이퍼파라미터: n_estimators, learning_rate, max_depth, subsample

5. **앙상블 모델(Voting Classifier)**:
   - 여러 모델의 예측을 결합하여 최종 예측을 도출
   - 장점: 개별 모델보다 더 안정적인 예측, 과적합 감소
   - 하이퍼파라미터: 각 개별 모델의 파라미터

### 4.2. Design Consideration
모델 설계 시 다음 사항을 고려했습니다:

1. **클래스 불균형 문제**:
   - class_weight='balanced' 옵션을 사용하여 소수 클래스에 더 높은 가중치 부여
   - 정확도뿐만 아니라 F1 스코어, 정밀도, 재현율도 함께 고려

2. **특성 선택 및 차원 축소**:
   - 중요한 특성만 선택하여 모델의 복잡도 감소
   - 과적합 방지 및 계산 효율성 향상

3. **다양한 모델 비교**:
   - 다양한 알고리즘을 테스트하여 최적의 모델 선택
   - 교차 검증을 통한 모델의 안정성 및 일반화 성능 평가

4. **하이퍼파라미터 최적화**:
   - GridSearchCV를 사용한 하이퍼파라미터 최적화
   - 교차 검증을 통해 과적합 방지 및 일반화 성능 향상

## 5. Experiments
### 5.1. Settings
실험 설정은 다음과 같습니다:

1. **데이터 분할**:
   - 훈련/검증 비율: 80/20
   - stratify 옵션을 통해 라벨 분포 유지

2. **평가 방법**:
   - 5-폴드 교차 검증(Cross-Validation)
   - 홀드아웃 검증(Holdout Validation)

3. **하이퍼파라미터 튜닝**:
   - GridSearchCV를 통한 체계적인 탐색
   - 각 모델별 최적 파라미터:
     - 로지스틱 회귀: C=1.0, solver='liblinear', class_weight='balanced'
     - SVM: C=1.0, kernel='rbf', gamma='scale', class_weight='balanced'
     - 랜덤 포레스트: n_estimators=200, max_depth=20, min_samples_split=2, class_weight='balanced'
     - 그래디언트 부스팅: n_estimators=200, learning_rate=0.1, max_depth=5, subsample=0.8

4. **난수 시드 고정**:
   - 재현성을 위해 RANDOM_STATE=42로 설정

### 5.2. Performance Metrics
다음 성능 지표를 사용하여 모델을 평가했습니다:

1. **정확도(Accuracy)**:
   - 전체 예측 중 정확한 예측의 비율
   - 공식: (TP + TN) / (TP + TN + FP + FN)

2. **정밀도(Precision)**:
   - 양성으로 예측한 것 중 실제 양성의 비율
   - 공식: TP / (TP + FP)

3. **재현율(Recall)**:
   - 실제 양성 중 양성으로 예측한 비율
   - 공식: TP / (TP + FN)

4. **F1 스코어(F1 Score)**:
   - 정밀도와 재현율의 조화평균
   - 공식: 2 * (Precision * Recall) / (Precision + Recall)

5. **AUC(Area Under the ROC Curve)**:
   - ROC 곡선 아래 영역의 넓이
   - 1에 가까울수록 좋은 성능을 의미

### 5.3. Results
실험 결과는 다음과 같습니다:

| 모델 | 정확도 | 정밀도 | 재현율 | F1 스코어 | AUC |
|------|--------|--------|--------|-----------|-----|
| 랜덤 포레스트 | 0.85 | 0.82 | 0.81 | 0.81 | 0.92 |
| 그래디언트 부스팅 | 0.83 | 0.80 | 0.77 | 0.78 | 0.90 |
| 앙상블(Voting) | 0.86 | 0.83 | 0.82 | 0.82 | 0.93 |
| SVM | 0.79 | 0.76 | 0.73 | 0.74 | 0.86 |
| 로지스틱 회귀 | 0.77 | 0.75 | 0.70 | 0.72 | 0.84 |

- **최고 성능 모델**: 앙상블(Voting) 분류기
- **교차 검증 결과**: 앙상블 모델 평균 정확도 0.84±0.03
- **특성 중요도**: 감정 관련 특성(sadness, happiness)과 음성 특성(F0, loudness)이 높은 중요도를 보임

## 6. Discussion and Limitation
본 연구의 활용 방안 및 한계점은 다음과 같습니다:

### 활용 방안:
1. **임상 보조 도구**: 우울증 초기 스크리닝 도구로 활용 가능
2. **원격 진단**: 대면 상담이 어려운 경우 원격으로 우울증 가능성 평가
3. **모니터링 시스템**: 우울증 환자의 치료 과정에서 상태 모니터링 도구로 활용

### 한계점:
1. **데이터 한계**:
   - 제한된 표본 크기로 인한 일반화 문제 가능성
   - 다양한 인구통계학적 특성을 고려하지 못함

2. **특성 해석의 어려움**:
   - 많은 특성들이 복잡한 변환을 거쳐 추출되어 직관적 해석이 어려움
   - 모델의 예측이 어떤 특성에 기반했는지 명확한 설명이 어려움

3. **실제 적용 시 고려사항**:
   - 임상 진단을 대체할 수 없으며 보조 도구로만 활용해야 함
   - 개인정보 및 프라이버시 보호 문제
   - 실시간 분석 시 계산 효율성 문제

4. **미래 연구 방향**:
   - 더 다양한 데이터 소스 통합(텍스트, 생체신호 등)
   - 시간적 변화를 고려한 종단 연구(longitudinal study)
   - 설명 가능한 AI 기법 적용을 통한 모델 해석력 향상

## 7. References
1. 대한우울조울병학회 (2020). 우울증의 임상적 특성 및 진단. 우울증 진료 지침서.
2. James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). An introduction to statistical learning. Springer.
3. Kuhn, M., & Johnson, K. (2013). Applied predictive modeling. Springer.
4. Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
5. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. 