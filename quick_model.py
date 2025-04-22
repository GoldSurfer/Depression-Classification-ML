"""
우울증 분류 머신러닝 프로젝트 - 간소화된 모델

Grid Search 없이 간단한 오디오 및 비주얼 특징 기반 우울증 분류 모델
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
import os

# 경고 메시지 무시
warnings.filterwarnings('ignore')

# 시드 고정
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def create_directory(path):
    """디렉토리가 존재하지 않으면 생성"""
    if not os.path.exists(path):
        os.makedirs(path)

# 결과 저장을 위한 디렉토리 생성
create_directory("plots")

def load_data():
    """데이터 로드 및 기본 정보 출력"""
    print("데이터 로드 중...")
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    
    print(f"훈련 데이터 크기: {train_df.shape}")
    print(f"테스트 데이터 크기: {test_df.shape}")
    
    return train_df, test_df

def analyze_data(train_df):
    """데이터 분석 및 시각화"""
    print("\n데이터 분석 중...")
    
    # 라벨 분포 확인
    label_counts = train_df['label'].value_counts()
    print("라벨 분포:")
    print(label_counts)
    
    # 라벨 분포 시각화
    plt.figure(figsize=(8, 6))
    sns.countplot(x='label', data=train_df)
    plt.title('우울증 라벨 분포')
    plt.xlabel('우울증 여부 (0: 비우울, 1: 우울)')
    plt.ylabel('샘플 수')
    plt.savefig('plots/label_distribution_quick.png')

def preprocess_data(train_df, test_df):
    """데이터 전처리"""
    print("\n데이터 전처리 중...")
    
    # 학습 데이터에서 X, y 분리
    X = train_df.drop('label', axis=1)
    y = train_df['label']
    
    # 테스트 데이터에서 ID 분리
    test_ids = test_df['id']
    X_test = test_df.drop('id', axis=1)
    
    # 학습/검증 데이터 분할
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"학습 데이터 크기: {X_train.shape}")
    print(f"검증 데이터 크기: {X_val.shape}")
    print(f"테스트 데이터 크기: {X_test.shape}")
    
    # 표준화
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, test_ids

def train_random_forest(X_train, y_train, X_val, y_val):
    """랜덤 포레스트 모델 훈련"""
    print("\n랜덤 포레스트 모델 훈련 중...")
    
    # 기본 파라미터 사용
    model = RandomForestClassifier(
        n_estimators=200, 
        max_depth=20,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # 모델 평가
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    
    print(f"\n랜덤 포레스트 모델 검증 정확도: {accuracy:.4f}")
    
    # 분류 보고서
    print("\n분류 보고서:")
    print(classification_report(y_val, y_pred))
    
    # 혼동 행렬
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['비우울', '우울'], yticklabels=['비우울', '우울'])
    plt.xlabel('예측')
    plt.ylabel('실제')
    plt.title('랜덤 포레스트 혼동 행렬')
    plt.savefig('plots/random_forest_confusion_matrix_quick.png')
    
    # 특성 중요도 시각화
    feature_importances = model.feature_importances_
    indices = np.argsort(feature_importances)[-20:]  # 상위 20개 특성만 시각화
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(indices)), feature_importances[indices], align='center')
    plt.yticks(range(len(indices)), [f'Feature {i}' for i in indices])
    plt.xlabel('특성 중요도')
    plt.title('랜덤 포레스트 - 상위 20개 특성 중요도')
    plt.tight_layout()
    plt.savefig('plots/random_forest_feature_importance_quick.png')
    
    return model, accuracy

def generate_submission(model, X_test, test_ids, accuracy):
    """제출 파일 생성"""
    print("\n제출 파일 생성 중...")
    
    # 테스트 데이터에 대한 예측
    predictions = model.predict(X_test)
    
    # 제출 파일 생성
    submission_df = pd.DataFrame({
        'id': test_ids,
        'label': predictions
    })
    
    submission_path = 'submission_quick.csv'
    submission_df.to_csv(submission_path, index=False)
    
    print(f"제출 파일이 생성되었습니다: {submission_path}")
    print(f"모델 검증 정확도: {accuracy:.4f}")

def main():
    """메인 실행 함수"""
    print("=" * 50)
    print("우울증 분류 머신러닝 프로젝트 - 간소화된 모델")
    print("=" * 50)
    
    # 데이터 로드
    train_df, test_df = load_data()
    
    # 데이터 분석
    analyze_data(train_df)
    
    # 데이터 전처리
    X_train, X_val, X_test, y_train, y_val, test_ids = preprocess_data(train_df, test_df)
    
    # 랜덤 포레스트 모델 훈련
    model, accuracy = train_random_forest(X_train, y_train, X_val, y_val)
    
    # 제출 파일 생성
    generate_submission(model, X_test, test_ids, accuracy)

if __name__ == "__main__":
    main() 
