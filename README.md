# 우울증 분류 머신러닝 프로젝트

## 📌 프로젝트 개요
오디오 및 비주얼 특성 데이터를 기반으로 우울증 유무를 분류하는 이진 분류 문제입니다. 주어진 학습 데이터를 바탕으로 머신러닝 모델을 학습시켜, 테스트 데이터에 대한 예측을 수행하고 정확도를 기준으로 리더보드에서 경쟁합니다.


---

## 🗂 데이터 구성

- `train.csv`: 학습용 데이터셋 (label 포함)
- `test.csv`: 테스트 데이터셋 (label 없음)
- `sample_submission.csv`: 제출 파일 형식 예시

---

## 🧠 문제 정의
- **입력**: 오디오 및 비주얼 피처 값들 (다수의 수치형 변수)
- **출력**: 우울증 여부 (`label`: 0 or 1)
- **목표**: 테스트 데이터의 우울증 여부를 정확하게 예측하는 분류기 개발

---

## 💻 프로젝트 구조
```
.
├── data/                  # 데이터 디렉토리
│   ├── train.csv          # 학습 데이터
│   ├── test.csv           # 테스트 데이터
│   └── sample_submission.csv  # 제출 양식
├── plots/                 # 시각화 결과 저장 디렉토리
│   ├── eda/               # 탐색적 데이터 분석 시각화
│   └── evaluation/        # 모델 평가 시각화
├── models/                # 학습된 모델 저장 디렉토리
├── train_and_submit.py    # 메인 훈련 및 제출 스크립트
├── quick_model.py         # 간소화된 모델 스크립트
├── exploratory_analysis.py # 데이터 탐색 및 시각화 스크립트
├── model_evaluation.py    # 모델 성능 평가 스크립트
├── report.md              # 프로젝트 보고서
├── requirements.txt       # 필요 패키지 목록
└── README.md              # 프로젝트 설명
```

---

## 🛠️ 설치 방법

1. 저장소 복제
```bash
git clone <repository_url>
cd <repository_directory>
```

2. 가상 환경 생성 및 활성화 (선택 사항)
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는
venv\Scripts\activate  # Windows
```

3. 의존성 설치
```bash
pip install -r requirements.txt
```

---

## ⚙️ 실행 방법

### 1. 탐색적 데이터 분석
```bash
python exploratory_analysis.py
```
데이터의 분포, 상관관계, 중요 특성 등을 시각화하여 `plots/eda/` 디렉토리에 저장합니다.

### 2. 간소화된 모델 실행 (빠른 실행)
```bash
python quick_model.py
```
GridSearch 없이 기본 Random Forest 모델을 빠르게 학습하고 예측합니다.

### 3. 메인 모델 학습 및 제출 파일 생성
```bash
python train_and_submit.py
```
여러 모델을 학습하고 하이퍼파라미터 튜닝을 통해 최적의 모델을 찾아 제출 파일을 생성합니다.

### 4. 모델 성능 평가
```bash
python model_evaluation.py
```
다양한 모델의 성능을 비교하고 평가하여 결과를 `plots/evaluation/` 디렉토리에 저장합니다.

---

## 📊 평가 기준

- Accuracy 기반 리더보드 점수
- Public Rank: 테스트 데이터의 80%
- Private Rank: 테스트 데이터의 20%
- 최종 점수는 Private Rank 포함 평가됨

---

## 📈 구현 모델
본 프로젝트에서 구현한 주요 모델은 다음과 같습니다:

1. **로지스틱 회귀(Logistic Regression)**
2. **서포트 벡터 머신(SVM)**
3. **랜덤 포레스트(Random Forest)**
4. **그래디언트 부스팅(Gradient Boosting)**
5. **앙상블 모델(Voting Classifier)**

각 모델의 성능 및 하이퍼파라미터 튜닝 결과는 `report.md` 파일에서 확인할 수 있습니다.

---

## ⚠️ 규칙 요약

- 딥러닝 금지, 사이킷런 기반 ML 모델 사용
- 외부 데이터 사용 금지
- 테스트 데이터를 학습에 활용 불가
- 하루 최대 5회 제출 (Private 1회)

---

## 🔧 주요 패키지
- pandas (1.5.3)
- numpy (1.23.5)
- scikit-learn (1.2.2)
- matplotlib (3.7.1)
- seaborn (0.12.2)

---

## 📝 보고서
자세한 프로젝트 분석 및 결과는 `report.md` 파일에서 확인할 수 있습니다.

---

