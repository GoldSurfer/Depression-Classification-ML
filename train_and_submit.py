"""
우울증 분류 머신러닝 프로젝트

오디오 및 비주얼 특징 데이터를 활용한 우울증 분류 모델 구현
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.feature_selection import SelectFromModel, RFECV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import warnings
import os
import joblib
from tqdm import tqdm
from scipy.stats import randint, uniform

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
    audio_cols = [col for col in train_df.columns if 'audio_' in col and '_var' not in col 
                 and '_min' not in col and '_max' not in col][:10]  # 시각화를 위해 일부만 선택
    
    # 감정 특성 상관관계
    plt.figure(figsize=(12, 10))
    emotion_corr = train_df[emotion_cols + ['label']].corr()
    mask = np.triu(np.ones_like(emotion_corr, dtype=bool))
    sns.heatmap(emotion_corr, mask=mask, cmap='coolwarm', vmin=-1, vmax=1, center=0,
                square=True, linewidths=.5, annot=False)
    plt.title('감정 특성 상관관계')
    plt.tight_layout()
    plt.savefig('plots/emotion_correlation.png')
    
    # 오디오 특성 상관관계
    plt.figure(figsize=(12, 10))
    audio_corr = train_df[audio_cols + ['label']].corr()
    mask = np.triu(np.ones_like(audio_corr, dtype=bool))
    sns.heatmap(audio_corr, mask=mask, cmap='coolwarm', vmin=-1, vmax=1, center=0,
                square=True, linewidths=.5, annot=False)
    plt.title('오디오 특성 상관관계')
    plt.tight_layout()
    plt.savefig('plots/audio_correlation.png')
    
    return emotion_cols, audio_cols

def feature_engineering(X_train, X_val, X_test):
    """특성 공학: 새로운 특징 생성"""
    print("\n특성 공학 수행 중...")
    
    # 원본 데이터 복사
    X_train_fe = X_train.copy()
    X_val_fe = X_val.copy()
    X_test_fe = X_test.copy()
    
    # 특성 그룹화
    emotion_cols = [col for col in X_train.columns if 'emotion_' in col]
    audio_cols = [col for col in X_train.columns if 'audio_' in col]
    visual_cols = [col for col in X_train.columns if 'visual_' in col]
    freq_cols = [col for col in X_train.columns if 'freq' in col or 'F0' in col]
    energy_cols = [col for col in X_train.columns if 'energy' in col]
    
    # 그룹별 통계량 생성 (평균, 표준편차, 최소값, 최대값, 범위)
    for name, cols in [('emotion', emotion_cols), ('audio', audio_cols), 
                       ('visual', visual_cols), ('freq', freq_cols), ('energy', energy_cols)]:
        if cols:
            # 각 데이터셋에 대해 통계량 계산
            for df_name, df in [('train', X_train_fe), ('val', X_val_fe), ('test', X_test_fe)]:
                df[f'{name}_mean'] = df[cols].mean(axis=1)
                df[f'{name}_std'] = df[cols].std(axis=1)
                df[f'{name}_min'] = df[cols].min(axis=1)
                df[f'{name}_max'] = df[cols].max(axis=1)
                df[f'{name}_range'] = df[f'{name}_max'] - df[f'{name}_min']
                df[f'{name}_median'] = df[cols].median(axis=1)
                df[f'{name}_q25'] = df[cols].quantile(0.25, axis=1)
                df[f'{name}_q75'] = df[cols].quantile(0.75, axis=1)
                df[f'{name}_iqr'] = df[f'{name}_q75'] - df[f'{name}_q25']
                df[f'{name}_skew'] = df[cols].skew(axis=1)
                df[f'{name}_kurt'] = df[cols].kurtosis(axis=1)
    
    # 그룹 간 상호작용 특성 생성
    group_pairs = [
        ('emotion', 'audio'), 
        ('emotion', 'visual'), 
        ('audio', 'visual'),
        ('freq', 'energy')
    ]
    
    for g1, g2 in group_pairs:
        key1 = f'{g1}_mean'
        key2 = f'{g2}_mean'
        
        if key1 in X_train_fe.columns and key2 in X_train_fe.columns:
            for df in [X_train_fe, X_val_fe, X_test_fe]:
                # 그룹 간 곱, 비율, 차이, 합
                df[f'{g1}_{g2}_product'] = df[key1] * df[key2]
                df[f'{g1}_{g2}_ratio'] = df[key1] / (df[key2] + 1e-10)  # 0으로 나누기 방지
                df[f'{g1}_{g2}_diff'] = df[key1] - df[key2]
                df[f'{g1}_{g2}_sum'] = df[key1] + df[key2]
                
                # 변동성 관련 상호작용
                if f'{g1}_std' in df.columns and f'{g2}_std' in df.columns:
                    df[f'{g1}_{g2}_std_ratio'] = df[f'{g1}_std'] / (df[f'{g2}_std'] + 1e-10)
                    df[f'{g1}_{g2}_variability'] = df[f'{g1}_std'] * df[f'{g2}_std']
    
    # 감정 특성들 간의 불일치도 (우울증은 감정 표현의 불일치로 감지될 수 있음)
    if len(emotion_cols) > 1:
        for df in [X_train_fe, X_val_fe, X_test_fe]:
            # 각 감정 특성과 평균 간의 거리
            for col in emotion_cols:
                df[f'{col}_deviation'] = abs(df[col] - df['emotion_mean'])
            
            # 감정 불일치도 (감정 특성들의 표준편차 평균)
            df['emotion_inconsistency'] = df[[f'{col}_deviation' for col in emotion_cols]].mean(axis=1)
    
    # 음성 특성을 활용한 톤 변화 지표 (우울증은 음성 톤의 변화가 적을 수 있음)
    if freq_cols:
        for df in [X_train_fe, X_val_fe, X_test_fe]:
            if 'freq_std' in df.columns:
                df['voice_tone_variability'] = df['freq_std'] / (df['freq_mean'] + 1e-10)
    
    # 생성된 특성에서 무한값이나 NaN 처리
    for df in [X_train_fe, X_val_fe, X_test_fe]:
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)
    
    print(f"원본 특성 수: {X_train.shape[1]}")
    print(f"특성 공학 후 특성 수: {X_train_fe.shape[1]}")
    
    return X_train_fe, X_val_fe, X_test_fe

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
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    
    # 스케일러 저장
    joblib.dump(scaler, 'models/scaler.pkl')
    
    # 특성 공학
    X_train_fe, X_val_fe, X_test_fe = feature_engineering(
        X_train_scaled, X_val_scaled, X_test_scaled
    )
    
    return X_train_fe, X_val_fe, X_test_fe, y_train, y_val, test_ids

def optimize_feature_selection(X_train, X_val, X_test, y_train):
    """교차 검증을 통한 특성 선택 최적화"""
    print("\n교차 검증을 통한 특성 선택 최적화 중...")
    
    # 테스트할 다양한 임계값들 (더 많은 임계값 추가)
    thresholds = ['mean', 'median', '1.25*mean', '1.25*median', '1.5*median', '0.75*mean', '0.75*median',
                 '0.5*mean', '0.5*median', '2*mean', '2*median', '0.3*mean', '0.3*median']
    
    best_threshold = None
    best_score = 0
    best_n_features = 0
    
    # tqdm을 사용한 진행 상황 표시
    with tqdm(total=len(thresholds), desc="특성 선택 임계값 최적화") as pbar:
        for threshold in thresholds:
            # 특성 선택기 생성
            selector = SelectFromModel(
                RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),  # estimators 수 증가
                threshold=threshold
            )
            
            # 특성 선택 적용
            X_train_selected = selector.fit_transform(X_train, y_train)
            
            # 선택된 특성 수
            n_features = X_train_selected.shape[1]
            
            # 10-겹 교차 검증으로 성능 평가 (k-겹 증가)
            cv_scores = cross_val_score(
                RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),  # estimators 수 증가
                X_train_selected, y_train,
                cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE),  # k-겹 증가
                scoring='accuracy',
                n_jobs=-1
            )
            
            # 평균 점수 계산
            mean_score = cv_scores.mean()
            
            print(f"임계값: {threshold}, 특성 수: {n_features}, 교차 검증 점수: {mean_score:.4f} ± {cv_scores.std():.4f}")
            
            # 최고 점수 업데이트
            if mean_score > best_score:
                best_score = mean_score
                best_threshold = threshold
                best_n_features = n_features
            
            pbar.update(1)
    
    print(f"\n최적 임계값: {best_threshold}, 특성 수: {best_n_features}, 교차 검증 점수: {best_score:.4f}")
    
    # 최적 임계값으로 특성 선택
    best_selector = SelectFromModel(
        RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),  # estimators 수 증가
        threshold=best_threshold
    )
    
    X_train_selected = best_selector.fit_transform(X_train, y_train)
    X_val_selected = best_selector.transform(X_val)
    X_test_selected = best_selector.transform(X_test)
    
    # 특성 선택 모델 저장
    joblib.dump(best_selector, 'models/optimized_feature_selector.pkl')
    
    return X_train_selected, X_val_selected, X_test_selected

def dimension_reduction(X_train, X_val, X_test, n_components=0.95):
    """PCA를 사용한 차원 축소"""
    print("\nPCA 차원 축소 수행 중...")
    
    # PCA 모델 훈련
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)
    X_test_pca = pca.transform(X_test)
    
    print(f"PCA 이전 특성 수: {X_train.shape[1]}")
    print(f"PCA 이후 특성 수: {X_train_pca.shape[1]}")
    print(f"설명된 분산 비율: {sum(pca.explained_variance_ratio_):.4f}")
    
    # PCA 모델 저장
    joblib.dump(pca, 'models/pca.pkl')
    
    return X_train_pca, X_val_pca, X_test_pca

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
    
    # 확장된 하이퍼파라미터 탐색 공간
    param_distributions = {
        'C': uniform(0.0001, 1000),  # 범위 확장
        'solver': ['liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga'],  # solver 추가
        'penalty': ['l1', 'l2', 'elasticnet', None],
        'class_weight': [None, 'balanced', {0: 0.8, 1: 1.2}, {0: 0.7, 1: 1.3}, {0: 0.9, 1: 1.1}],  # 가중치 옵션 추가
        'l1_ratio': uniform(0, 1),  # elasticnet 패널티를 위한 파라미터
        'max_iter': [1000, 2000, 3000]  # 최대 반복 횟수 증가
    }
    
    # RandomizedSearchCV로 더 넓은 공간을 효율적으로 탐색
    random_search = RandomizedSearchCV(
        LogisticRegression(random_state=RANDOM_STATE),
        param_distributions=param_distributions,
        n_iter=50,  # 시도할 파라미터 조합 수 증가
        cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE),  # k-겹 증가
        scoring='accuracy',
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1
    )
    
    # tqdm으로 진행 상태 표시
    with tqdm(total=100, desc="로지스틱 회귀 훈련") as pbar:
        random_search.fit(X_train, y_train)
        pbar.update(100)
    
    print(f"최적 파라미터: {random_search.best_params_}")
    model = random_search.best_estimator_
    
    # 모델 저장
    joblib.dump(model, 'models/logistic_regression.pkl')
    
    # 모델 평가
    accuracy = evaluate_model(model, X_val, y_val, "LogisticRegression")
    
    return model, accuracy

def train_svm(X_train, y_train, X_val, y_val):
    """SVM 모델 훈련"""
    print("\nSVM 모델 훈련 중...")
    
    # 확장된 하이퍼파라미터 탐색 공간
    param_distributions = {
        'C': uniform(0.01, 1000),  # 범위 확장
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        'gamma': ['scale', 'auto'] + list(uniform(0.0001, 10).rvs(10)),  # 더 다양한 gamma값 시도
        'degree': randint(2, 6),  # poly 커널을 위한 파라미터 범위 확장
        'class_weight': [None, 'balanced', {0: 0.8, 1: 1.2}, {0: 0.7, 1: 1.3}],  # 가중치 옵션 추가
        'coef0': uniform(0, 10).rvs(5),  # poly 및 sigmoid 커널의 독립항
        'shrinking': [True, False],  # 수축 휴리스틱 사용 여부
        'tol': [1e-3, 1e-4, 1e-5]  # 종료 허용 오차
    }
    
    # RandomizedSearchCV로 더 넓은 공간을 효율적으로 탐색
    random_search = RandomizedSearchCV(
        SVC(random_state=RANDOM_STATE, probability=True),
        param_distributions=param_distributions,
        n_iter=40,  # 시도할 파라미터 조합 수 증가
        cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE),  # k-겹 증가
        scoring='accuracy',
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1
    )
    
    # tqdm으로 진행 상태 표시
    with tqdm(total=100, desc="SVM 훈련") as pbar:
        random_search.fit(X_train, y_train)
        pbar.update(100)
    
    print(f"최적 파라미터: {random_search.best_params_}")
    model = random_search.best_estimator_
    
    # 모델 저장
    joblib.dump(model, 'models/svm.pkl')
    
    # 모델 평가
    accuracy = evaluate_model(model, X_val, y_val, "SVM")
    
    return model, accuracy

def train_random_forest(X_train, y_train, X_val, y_val):
    """랜덤 포레스트 모델 훈련"""
    print("\n랜덤 포레스트 모델 훈련 중...")
    
    # 확장된 하이퍼파라미터 탐색 공간
    param_distributions = {
        'n_estimators': randint(100, 1000),  # 트리 수 증가
        'max_depth': [None] + list(randint(10, 100).rvs(8)),  # 깊이 범위 확장
        'min_samples_split': randint(2, 30),  # 범위 확장
        'min_samples_leaf': randint(1, 20),  # 범위 확장
        'max_features': ['sqrt', 'log2', None] + [0.3, 0.5, 0.7, 0.9],  # 옵션 추가
        'bootstrap': [True, False],
        'class_weight': [None, 'balanced', 'balanced_subsample', 
                        {0: 0.7, 1: 1.3}, {0: 0.8, 1: 1.2}],  # 가중치 옵션 추가
        'criterion': ['gini', 'entropy', 'log_loss'],  # 분할 기준 추가
        'min_impurity_decrease': [0.0, 0.01, 0.05, 0.1],  # 불순도 감소 최소값
        'max_leaf_nodes': [None, 30, 50, 100, 200]  # 최대 리프 노드 수
    }
    
    # RandomizedSearchCV로 더 넓은 공간을 효율적으로 탐색
    random_search = RandomizedSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE),
        param_distributions=param_distributions,
        n_iter=60,  # 시도할 파라미터 조합 수 증가
        cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE),  # k-겹 증가
        scoring='accuracy',
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1
    )
    
    # tqdm으로 진행 상태 표시
    with tqdm(total=100, desc="랜덤 포레스트 훈련") as pbar:
        random_search.fit(X_train, y_train)
        pbar.update(100)
    
    print(f"최적 파라미터: {random_search.best_params_}")
    model = random_search.best_estimator_
    
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
    
    # 확장된 하이퍼파라미터 탐색 공간
    param_distributions = {
        'n_estimators': randint(100, 1000),  # 트리 수 증가
        'learning_rate': uniform(0.001, 0.5),  # 학습률 범위 확장
        'max_depth': randint(3, 20),  # 트리 깊이 범위 확장
        'min_samples_split': randint(2, 30),  # 범위 확장
        'min_samples_leaf': randint(1, 20),  # 범위 확장
        'subsample': uniform(0.5, 0.5),  # 범위 확장
        'max_features': ['sqrt', 'log2', None] + [0.3, 0.5, 0.7, 0.9],  # 옵션 추가
        'loss': ['log_loss', 'exponential'],  # 손실 함수 추가
        'min_impurity_decrease': [0.0, 0.01, 0.05, 0.1],  # 불순도 감소 최소값
        'validation_fraction': uniform(0.1, 0.3).rvs(5),  # 조기 종료를 위한 검증 데이터 비율
        'n_iter_no_change': [5, 10, 20, None],  # 조기 종료 기준
        'tol': [1e-5, 1e-4, 1e-3]  # 수렴 허용 오차
    }
    
    # RandomizedSearchCV로 더 넓은 공간을 효율적으로 탐색
    random_search = RandomizedSearchCV(
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        param_distributions=param_distributions,
        n_iter=60,  # 시도할 파라미터 조합 수 증가
        cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE),  # k-겹 증가
        scoring='accuracy',
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1
    )
    
    # tqdm으로 진행 상태 표시
    with tqdm(total=100, desc="그래디언트 부스팅 훈련") as pbar:
        random_search.fit(X_train, y_train)
        pbar.update(100)
    
    print(f"최적 파라미터: {random_search.best_params_}")
    model = random_search.best_estimator_
    
    # 모델 저장
    joblib.dump(model, 'models/gradient_boosting.pkl')
    
    # 모델 평가
    accuracy = evaluate_model(model, X_val, y_val, "GradientBoosting")
    
    return model, accuracy

def train_voting_ensemble(models, X_train, y_train, X_val, y_val):
    """보팅 앙상블 모델 훈련"""
    print("\n보팅 앙상블 모델 훈련 중...")
    
    # 모델들을 앙상블로 구성
    estimators = []
    for name, model in models.items():
        estimators.append((name, model))
    
    # 앙상블 모델 생성 (소프트 보팅)
    ensemble = VotingClassifier(estimators=estimators, voting='soft')
    
    # tqdm으로 진행 상태 표시
    with tqdm(total=100, desc="앙상블 모델 훈련") as pbar:
        ensemble.fit(X_train, y_train)
        pbar.update(100)
    
    # 모델 저장
    joblib.dump(ensemble, 'models/voting_ensemble.pkl')
    
    # 모델 평가
    accuracy = evaluate_model(ensemble, X_val, y_val, "VotingEnsemble")
    
    return ensemble, accuracy

def train_stacking_ensemble(models, X_train, y_train, X_val, y_val):
    """스태킹 앙상블 모델 훈련"""
    print("\n스태킹 앙상블 모델 훈련 중...")
    
    # 베이스 모델 정의
    base_estimators = [
        ('LogisticRegression', models['LogisticRegression']),
        ('SVM', models['SVM']),
        ('RandomForest', models['RandomForest']),
        ('GradientBoosting', models['GradientBoosting'])
    ]
    
    # 다양한 메타 모델 시도
    meta_models = {
        'LogisticRegression': LogisticRegression(random_state=RANDOM_STATE),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE)
    }
    
    best_meta_model = None
    best_meta_model_name = None
    best_cv_score = 0
    
    # 각 메타 모델에 대해 교차 검증 점수 계산
    print("메타 모델 선택 중...")
    for name, meta_model in meta_models.items():
        # 스태킹 앙상블 생성
        stack = StackingClassifier(
            estimators=base_estimators,
            final_estimator=meta_model,
            cv=5,
            n_jobs=-1,
            passthrough=True  # 원본 특성도 메타 모델에 전달
        )
        
        # 교차 검증 점수 계산
        cv_scores = cross_val_score(
            stack, X_train, y_train, 
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
            scoring='accuracy',
            n_jobs=-1
        )
        
        mean_cv_score = cv_scores.mean()
        print(f"  메타 모델 {name} - 교차 검증 점수: {mean_cv_score:.4f} ± {cv_scores.std():.4f}")
        
        if mean_cv_score > best_cv_score:
            best_cv_score = mean_cv_score
            best_meta_model = meta_model
            best_meta_model_name = name
    
    print(f"최적 메타 모델: {best_meta_model_name} (교차 검증 점수: {best_cv_score:.4f})")
    
    # 하이퍼파라미터 튜닝을 위한 파라미터 설정
    if best_meta_model_name == 'LogisticRegression':
        meta_param_grid = {
            'final_estimator__C': uniform(0.1, 10),
            'final_estimator__solver': ['liblinear', 'lbfgs'],
            'final_estimator__class_weight': [None, 'balanced']
        }
    elif best_meta_model_name == 'RandomForest':
        meta_param_grid = {
            'final_estimator__n_estimators': randint(50, 200),
            'final_estimator__max_depth': [None, 10, 20],
            'final_estimator__min_samples_split': randint(2, 10)
        }
    else:  # GradientBoosting
        meta_param_grid = {
            'final_estimator__n_estimators': randint(50, 200),
            'final_estimator__learning_rate': uniform(0.01, 0.2),
            'final_estimator__max_depth': randint(3, 10)
        }
    
    # 최종 스태킹 앙상블 모델 생성
    final_stack = StackingClassifier(
        estimators=base_estimators,
        final_estimator=best_meta_model,
        cv=10,  # 증가된 k-겹 교차 검증
        n_jobs=-1,
        passthrough=True,
        verbose=1
    )
    
    # 하이퍼파라미터 튜닝
    print("스태킹 앙상블 하이퍼파라미터 튜닝 중...")
    random_search = RandomizedSearchCV(
        final_stack,
        param_distributions=meta_param_grid,
        n_iter=20,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        scoring='accuracy',
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1
    )
    
    # tqdm으로 진행 상태 표시
    with tqdm(total=100, desc="최종 스태킹 앙상블 훈련") as pbar:
        random_search.fit(X_train, y_train)
        pbar.update(100)
    
    print(f"최적 하이퍼파라미터: {random_search.best_params_}")
    
    # 최종 모델
    stack_model = random_search.best_estimator_
    
    # 모델 저장
    joblib.dump(stack_model, 'models/stacking_ensemble.pkl')
    
    # 모델 평가
    accuracy = evaluate_model(stack_model, X_val, y_val, "StackingEnsemble")
    
    return stack_model, accuracy

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
    start_time = pd.Timestamp.now()
    print("=" * 50)
    print("우울증 분류 머신러닝 프로젝트")
    print("=" * 50)
    
    # 데이터 로드
    train_df, test_df = load_data()
    
    # 데이터 분석
    emotion_cols, audio_cols = analyze_data(train_df)
    
    # 데이터 전처리 (특성 공학 포함)
    X_train, X_val, X_test, y_train, y_val, test_ids = preprocess_data(train_df, test_df)
    
    # 교차 검증을 통한 특성 선택 최적화 (2순위 개선 방법)
    X_train_selected, X_val_selected, X_test_selected = optimize_feature_selection(X_train, X_val, X_test, y_train)
    
    # 차원 축소 (선택 사항)
    use_pca = False
    if use_pca:
        X_train_processed, X_val_processed, X_test_processed = dimension_reduction(
            X_train_selected, X_val_selected, X_test_selected
        )
    else:
        X_train_processed, X_val_processed, X_test_processed = X_train_selected, X_val_selected, X_test_selected
    
    # 모델 훈련
    models = {}
    accuracies = {}
    
    # 로지스틱 회귀 모델
    models['LogisticRegression'], accuracies['LogisticRegression'] = train_logistic_regression(
        X_train_processed, y_train, X_val_processed, y_val
    )
    
    # SVM 모델
    models['SVM'], accuracies['SVM'] = train_svm(
        X_train_processed, y_train, X_val_processed, y_val
    )
    
    # 랜덤 포레스트 모델
    models['RandomForest'], accuracies['RandomForest'] = train_random_forest(
        X_train_processed, y_train, X_val_processed, y_val
    )
    
    # 그래디언트 부스팅 모델
    models['GradientBoosting'], accuracies['GradientBoosting'] = train_gradient_boosting(
        X_train_processed, y_train, X_val_processed, y_val
    )
    
    # 앙상블 모델 (모델들의 성능이 유사할 때 효과적)
    models['VotingEnsemble'], accuracies['VotingEnsemble'] = train_voting_ensemble(
        models, X_train_processed, y_train, X_val_processed, y_val
    )
    
    # 스태킹 앙상블 모델 추가 (1순위 개선 방법)
    models['StackingEnsemble'], accuracies['StackingEnsemble'] = train_stacking_ensemble(
        models, X_train_processed, y_train, X_val_processed, y_val
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
    generate_submission(best_model, X_test_processed, test_ids, best_accuracy)
    
    # 총 실행 시간 출력
    end_time = pd.Timestamp.now()
    print(f"\n총 실행 시간: {end_time - start_time}")

if __name__ == "__main__":
    main() 
