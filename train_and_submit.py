"""
우울증 분류 머신러닝 프로젝트

오디오 및 비주얼 특징 데이터를 활용한 우울증 분류 모델 구현
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
import warnings
import os
import joblib

# 경고 메시지 무시
warnings.filterwarnings('ignore')

# 그래프 설정
plt.style.use('seaborn-v0_8-whitegrid')
sns.set(font_scale=1.2)

# 시드 고정
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def create_directory(path):
    """디렉토리가 존재하지 않으면 생성"""
    if not os.path.exists(path):
        os.makedirs(path)

# 결과 저장을 위한 디렉토리 생성
create_directory("plots")
create_directory("models")

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
    plt.savefig('plots/label_distribution.png')
    
    # 상관관계 분석 (일부 특성만 사용)
    emotion_cols = [col for col in train_df.columns if 'emotion_' in col and '_var' not in col 
                   and '_min' not in col and '_max' not in col]
    
    plt.figure(figsize=(12, 10))
    emotion_corr = train_df[emotion_cols + ['label']].corr()
    mask = np.triu(np.ones_like(emotion_corr, dtype=bool))
    sns.heatmap(emotion_corr, mask=mask, cmap='coolwarm', vmin=-1, vmax=1, center=0,
                square=True, linewidths=.5, annot=False)
    plt.title('감정 특성 상관관계')
    plt.tight_layout()
    plt.savefig('plots/emotion_correlation.png')
    
    return emotion_cols

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
    
    # 스케일러 저장
    joblib.dump(scaler, 'models/scaler.pkl')
    
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, test_ids

def select_features(X_train, X_val, X_test, y_train):
    """특성 선택"""
    print("\n특성 선택 중...")
    
    # 랜덤 포레스트 기반 특성 중요도를 사용한 특성 선택
    selector = SelectFromModel(
        RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        threshold='median'
    )
    
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_val_selected = selector.transform(X_val)
    X_test_selected = selector.transform(X_test)
    
    print(f"원본 특성 수: {X_train.shape[1]}")
    print(f"선택된 특성 수: {X_train_selected.shape[1]}")
    
    # 특성 선택 모델 저장
    joblib.dump(selector, 'models/feature_selector.pkl')
    
    return X_train_selected, X_val_selected, X_test_selected

def evaluate_model(model, X_val, y_val, model_name):
    """모델 평가"""
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    
    print(f"\n{model_name} 모델 검증 정확도: {accuracy:.4f}")
    
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
    plt.title(f'{model_name} 혼동 행렬')
    plt.savefig(f'plots/{model_name}_confusion_matrix.png')
    
    # ROC 곡선
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_val)[:, 1]
        fpr, tpr, _ = roc_curve(y_val, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC 곡선 (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{model_name} ROC 곡선')
        plt.legend(loc='lower right')
        plt.savefig(f'plots/{model_name}_roc_curve.png')
    
    return accuracy

def train_logistic_regression(X_train, y_train, X_val, y_val):
    """로지스틱 회귀 모델 훈련"""
    print("\n로지스틱 회귀 모델 훈련 중...")
    
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'lbfgs'],
        'class_weight': [None, 'balanced']
    }
    
    grid_search = GridSearchCV(
        LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
        param_grid=param_grid,
        cv=StratifiedKFold(n_splits=5),
        scoring='accuracy',
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"최적 파라미터: {grid_search.best_params_}")
    model = grid_search.best_estimator_
    
    # 모델 저장
    joblib.dump(model, 'models/logistic_regression.pkl')
    
    # 모델 평가
    accuracy = evaluate_model(model, X_val, y_val, "LogisticRegression")
    
    return model, accuracy

def train_svm(X_train, y_train, X_val, y_val):
    """SVM 모델 훈련"""
    print("\nSVM 모델 훈련 중...")
    
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto', 0.1],
        'class_weight': [None, 'balanced']
    }
    
    grid_search = GridSearchCV(
        SVC(random_state=RANDOM_STATE, probability=True),
        param_grid=param_grid,
        cv=StratifiedKFold(n_splits=5),
        scoring='accuracy',
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"최적 파라미터: {grid_search.best_params_}")
    model = grid_search.best_estimator_
    
    # 모델 저장
    joblib.dump(model, 'models/svm.pkl')
    
    # 모델 평가
    accuracy = evaluate_model(model, X_val, y_val, "SVM")
    
    return model, accuracy

def train_random_forest(X_train, y_train, X_val, y_val):
    """랜덤 포레스트 모델 훈련"""
    print("\n랜덤 포레스트 모델 훈련 중...")
    
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': [None, 'balanced']
    }
    
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE),
        param_grid=param_grid,
        cv=StratifiedKFold(n_splits=5),
        scoring='accuracy',
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"최적 파라미터: {grid_search.best_params_}")
    model = grid_search.best_estimator_
    
    # 모델 저장
    joblib.dump(model, 'models/random_forest.pkl')
    
    # 특성 중요도 시각화
    feature_importances = model.feature_importances_
    indices = np.argsort(feature_importances)[-20:]  # 상위 20개 특성만 시각화
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(indices)), feature_importances[indices], align='center')
    plt.yticks(range(len(indices)), [f'Feature {i}' for i in indices])
    plt.xlabel('특성 중요도')
    plt.title('랜덤 포레스트 - 상위 20개 특성 중요도')
    plt.tight_layout()
    plt.savefig('plots/random_forest_feature_importance.png')
    
    # 모델 평가
    accuracy = evaluate_model(model, X_val, y_val, "RandomForest")
    
    return model, accuracy

def train_gradient_boosting(X_train, y_train, X_val, y_val):
    """그래디언트 부스팅 모델 훈련"""
    print("\n그래디언트 부스팅 모델 훈련 중...")
    
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'subsample': [0.8, 1.0]
    }
    
    grid_search = GridSearchCV(
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        param_grid=param_grid,
        cv=StratifiedKFold(n_splits=5),
        scoring='accuracy',
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"최적 파라미터: {grid_search.best_params_}")
    model = grid_search.best_estimator_
    
    # 모델 저장
    joblib.dump(model, 'models/gradient_boosting.pkl')
    
    # 모델 평가
    accuracy = evaluate_model(model, X_val, y_val, "GradientBoosting")
    
    return model, accuracy

def generate_submission(best_model, X_test, test_ids, accuracy):
    """제출 파일 생성"""
    print("\n제출 파일 생성 중...")
    
    # 테스트 데이터에 대한 예측
    predictions = best_model.predict(X_test)
    
    # 제출 파일 생성
    submission_df = pd.DataFrame({
        'id': test_ids,
        'label': predictions
    })
    
    submission_path = 'submission.csv'
    submission_df.to_csv(submission_path, index=False)
    
    print(f"제출 파일이 생성되었습니다: {submission_path}")
    print(f"최종 모델 검증 정확도: {accuracy:.4f}")

def main():
    """메인 실행 함수"""
    print("=" * 50)
    print("우울증 분류 머신러닝 프로젝트")
    print("=" * 50)
    
    # 데이터 로드
    train_df, test_df = load_data()
    
    # 데이터 분석
    analyze_data(train_df)
    
    # 데이터 전처리
    X_train, X_val, X_test, y_train, y_val, test_ids = preprocess_data(train_df, test_df)
    
    # 특성 선택
    X_train_selected, X_val_selected, X_test_selected = select_features(X_train, X_val, X_test, y_train)
    
    # 모델 훈련
    models = {}
    accuracies = {}
    
    # 로지스틱 회귀 모델
    models['LogisticRegression'], accuracies['LogisticRegression'] = train_logistic_regression(
        X_train_selected, y_train, X_val_selected, y_val
    )
    
    # SVM 모델
    models['SVM'], accuracies['SVM'] = train_svm(
        X_train_selected, y_train, X_val_selected, y_val
    )
    
    # 랜덤 포레스트 모델
    models['RandomForest'], accuracies['RandomForest'] = train_random_forest(
        X_train_selected, y_train, X_val_selected, y_val
    )
    
    # 그래디언트 부스팅 모델
    models['GradientBoosting'], accuracies['GradientBoosting'] = train_gradient_boosting(
        X_train_selected, y_train, X_val_selected, y_val
    )
    
    # 최고 성능 모델 선택
    best_model_name = max(accuracies, key=accuracies.get)
    best_model = models[best_model_name]
    best_accuracy = accuracies[best_model_name]
    
    print("\n모델 비교:")
    for model_name, accuracy in accuracies.items():
        print(f"{model_name}: {accuracy:.4f}")
    
    print(f"\n최고 성능 모델: {best_model_name} (정확도: {best_accuracy:.4f})")
    
    # 제출 파일 생성
    generate_submission(best_model, X_test_selected, test_ids, best_accuracy)

if __name__ == "__main__":
    main() 
