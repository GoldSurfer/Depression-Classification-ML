"""
우울증 분류 머신러닝 프로젝트 - 모델 성능 평가

모델 성능 비교 및 분석
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                            roc_auc_score, roc_curve, auc, confusion_matrix, 
                            classification_report, precision_recall_curve)
from sklearn.feature_selection import SelectFromModel
import warnings
import os
import joblib
import time

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
create_directory("plots/evaluation")

def load_data():
    """데이터 로드 및 전처리"""
    print("데이터 로드 중...")
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    
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
    
    # 표준화
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # 특성 선택
    selector = SelectFromModel(
        RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        threshold='median'
    )
    
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_val_selected = selector.transform(X_val_scaled)
    X_test_selected = selector.transform(X_test_scaled)
    
    print(f"학습 데이터 크기: {X_train_selected.shape}")
    print(f"검증 데이터 크기: {X_val_selected.shape}")
    print(f"테스트 데이터 크기: {X_test_selected.shape}")
    
    return X_train_selected, X_val_selected, X_test_selected, y_train, y_val, test_ids

def train_models(X_train, y_train):
    """다양한 모델 훈련"""
    print("\n모델 훈련 중...")
    
    models = {}
    
    # 로지스틱 회귀
    print("로지스틱 회귀 모델 훈련 중...")
    start_time = time.time()
    models['Logistic Regression'] = LogisticRegression(
        C=1.0, solver='liblinear', class_weight='balanced', random_state=RANDOM_STATE, max_iter=1000)
    models['Logistic Regression'].fit(X_train, y_train)
    lr_time = time.time() - start_time
    
    # SVM
    print("SVM 모델 훈련 중...")
    start_time = time.time()
    models['SVM'] = SVC(
        C=1.0, kernel='rbf', gamma='scale', class_weight='balanced', 
        random_state=RANDOM_STATE, probability=True)
    models['SVM'].fit(X_train, y_train)
    svm_time = time.time() - start_time
    
    # 랜덤 포레스트
    print("랜덤 포레스트 모델 훈련 중...")
    start_time = time.time()
    models['Random Forest'] = RandomForestClassifier(
        n_estimators=200, max_depth=20, min_samples_split=2, min_samples_leaf=1,
        class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1)
    models['Random Forest'].fit(X_train, y_train)
    rf_time = time.time() - start_time
    
    # 그래디언트 부스팅
    print("그래디언트 부스팅 모델 훈련 중...")
    start_time = time.time()
    models['Gradient Boosting'] = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.1, max_depth=5, min_samples_split=2,
        min_samples_leaf=1, subsample=0.8, random_state=RANDOM_STATE)
    models['Gradient Boosting'].fit(X_train, y_train)
    gb_time = time.time() - start_time
    
    # 앙상블 (투표 분류기)
    print("앙상블 모델 (투표 분류기) 훈련 중...")
    start_time = time.time()
    models['Voting Classifier'] = VotingClassifier(
        estimators=[
            ('lr', models['Logistic Regression']),
            ('svm', models['SVM']),
            ('rf', models['Random Forest']),
            ('gb', models['Gradient Boosting'])
        ],
        voting='soft'
    )
    models['Voting Classifier'].fit(X_train, y_train)
    vc_time = time.time() - start_time
    
    # 훈련 시간 기록
    training_times = {
        'Logistic Regression': lr_time,
        'SVM': svm_time,
        'Random Forest': rf_time,
        'Gradient Boosting': gb_time,
        'Voting Classifier': vc_time
    }
    
    return models, training_times

def evaluate_models(models, X_val, y_val, training_times):
    """모델 성능 평가"""
    print("\n모델 성능 평가 중...")
    
    results = {
        'Model': [],
        'Accuracy': [],
        'Precision': [],
        'Recall': [],
        'F1 Score': [],
        'AUC': [],
        'Training Time (s)': []
    }
    
    # 각 모델별로 예측 및 성능 평가
    for name, model in models.items():
        print(f"{name} 모델 평가 중...")
        
        # 예측
        y_pred = model.predict(X_val)
        
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_val)[:, 1]
            roc_auc = roc_auc_score(y_val, y_prob)
        else:
            roc_auc = 0.0
        
        # 성능 지표 계산
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        
        # 결과 저장
        results['Model'].append(name)
        results['Accuracy'].append(accuracy)
        results['Precision'].append(precision)
        results['Recall'].append(recall)
        results['F1 Score'].append(f1)
        results['AUC'].append(roc_auc)
        results['Training Time (s)'].append(training_times[name])
        
        # 혼동 행렬
        cm = confusion_matrix(y_val, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['비우울', '우울'], yticklabels=['비우울', '우울'])
        plt.xlabel('예측')
        plt.ylabel('실제')
        plt.title(f'{name} 혼동 행렬')
        plt.savefig(f'plots/evaluation/{name.replace(" ", "_")}_confusion_matrix.png')
        plt.close()
        
        # ROC 곡선
        if hasattr(model, 'predict_proba'):
            fpr, tpr, _ = roc_curve(y_val, y_prob)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC 곡선 (AUC = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'{name} ROC 곡선')
            plt.legend(loc='lower right')
            plt.savefig(f'plots/evaluation/{name.replace(" ", "_")}_roc_curve.png')
            plt.close()
            
            # PR 곡선
            precision_curve, recall_curve, _ = precision_recall_curve(y_val, y_prob)
            
            plt.figure(figsize=(8, 6))
            plt.plot(recall_curve, precision_curve, color='blue', lw=2)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'{name} Precision-Recall 곡선')
            plt.savefig(f'plots/evaluation/{name.replace(" ", "_")}_pr_curve.png')
            plt.close()
    
    return pd.DataFrame(results)

def compare_models(results_df):
    """모델 성능 비교"""
    print("\n모델 성능 비교 중...")
    
    # 정확도 비교
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x='Model', y='Accuracy', data=results_df)
    plt.title('모델별 정확도 비교')
    plt.xticks(rotation=45)
    plt.ylim(0.5, 1.0)  # 정확도 범위 설정
    
    # 막대 위에 값 표시
    for i, v in enumerate(results_df['Accuracy']):
        ax.text(i, v+0.01, f'{v:.4f}', ha='center')
    
    plt.tight_layout()
    plt.savefig('plots/evaluation/model_accuracy_comparison.png')
    plt.close()
    
    # F1 스코어 비교
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x='Model', y='F1 Score', data=results_df)
    plt.title('모델별 F1 Score 비교')
    plt.xticks(rotation=45)
    plt.ylim(0.5, 1.0)
    
    # 막대 위에 값 표시
    for i, v in enumerate(results_df['F1 Score']):
        ax.text(i, v+0.01, f'{v:.4f}', ha='center')
    
    plt.tight_layout()
    plt.savefig('plots/evaluation/model_f1_comparison.png')
    plt.close()
    
    # ROC AUC 비교
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x='Model', y='AUC', data=results_df)
    plt.title('모델별 ROC AUC 비교')
    plt.xticks(rotation=45)
    plt.ylim(0.5, 1.0)
    
    # 막대 위에 값 표시
    for i, v in enumerate(results_df['AUC']):
        ax.text(i, v+0.01, f'{v:.4f}', ha='center')
    
    plt.tight_layout()
    plt.savefig('plots/evaluation/model_auc_comparison.png')
    plt.close()
    
    # 다양한 성능 지표 비교 (레이더 차트)
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']
    models = results_df['Model'].tolist()
    
    # 레이더 차트
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, polar=True)
    
    # 각도 설정
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))  # 그래프 폐합을 위해 처음 값 추가
    
    # 각 모델별로 레이더 차트 그리기
    for i, model in enumerate(models):
        values = results_df.loc[i, metrics].values.flatten().tolist()
        values = np.concatenate((values, [values[0]]))  # 그래프 폐합을 위해 처음 값 추가
        
        ax.plot(angles, values, 'o-', linewidth=2, label=model)
        ax.fill(angles, values, alpha=0.1)
    
    # 그래프 설정
    ax.set_thetagrids(angles[:-1] * 180/np.pi, metrics)
    plt.title('모델별 성능 지표 비교 (레이더 차트)')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('plots/evaluation/model_radar_comparison.png')
    plt.close()
    
    # 훈련 시간 비교
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x='Model', y='Training Time (s)', data=results_df)
    plt.title('모델별 훈련 시간 비교')
    plt.xticks(rotation=45)
    
    # 막대 위에 값 표시
    for i, v in enumerate(results_df['Training Time (s)']):
        ax.text(i, v+0.5, f'{v:.2f}s', ha='center')
    
    plt.tight_layout()
    plt.savefig('plots/evaluation/model_training_time_comparison.png')
    plt.close()
    
    return results_df.sort_values('Accuracy', ascending=False).reset_index(drop=True)

def cross_validation_evaluation(X_train, y_train):
    """교차 검증을 통한 모델 평가"""
    print("\n교차 검증 평가 중...")
    
    # 모델 정의
    models = {
        'Logistic Regression': LogisticRegression(
            C=1.0, solver='liblinear', class_weight='balanced', random_state=RANDOM_STATE, max_iter=1000),
        'SVM': SVC(
            C=1.0, kernel='rbf', gamma='scale', class_weight='balanced', 
            random_state=RANDOM_STATE, probability=True),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=20, min_samples_split=2, min_samples_leaf=1,
            class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=5, min_samples_split=2,
            min_samples_leaf=1, subsample=0.8, random_state=RANDOM_STATE)
    }
    
    # 교차 검증 결과 저장
    cv_results = {
        'Model': [],
        'Mean Accuracy': [],
        'Std Accuracy': []
    }
    
    # 5-폴드 교차 검증
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    # 각 모델 평가
    for name, model in models.items():
        print(f"{name} 교차 검증 중...")
        
        # 교차 검증 실행
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
        
        # 결과 저장
        cv_results['Model'].append(name)
        cv_results['Mean Accuracy'].append(cv_scores.mean())
        cv_results['Std Accuracy'].append(cv_scores.std())
    
    cv_df = pd.DataFrame(cv_results)
    
    # 교차 검증 결과 시각화
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x='Model', y='Mean Accuracy', data=cv_df, yerr=cv_df['Std Accuracy'],
                    capsize=0.2, palette='muted')
    plt.title('5-폴드 교차 검증 정확도')
    plt.xticks(rotation=45)
    plt.ylim(0.5, 1.0)
    
    # 막대 위에 값 표시
    for i, v in enumerate(cv_df['Mean Accuracy']):
        ax.text(i, v+0.01, f'{v:.4f}±{cv_df["Std Accuracy"][i]:.4f}', ha='center')
    
    plt.tight_layout()
    plt.savefig('plots/evaluation/cross_validation_results.png')
    plt.close()
    
    return cv_df

def generate_best_model_submission(models, results_df, X_test, test_ids):
    """최적 모델로 제출 파일 생성"""
    print("\n최적 모델로 제출 파일 생성 중...")
    
    # 정확도 기준으로 최적 모델 선택
    best_model_name = results_df.iloc[0]['Model']
    best_model = models[best_model_name]
    
    print(f"최적 모델: {best_model_name} (정확도: {results_df.iloc[0]['Accuracy']:.4f})")
    
    # 테스트 데이터 예측
    predictions = best_model.predict(X_test)
    
    # 제출 파일 생성
    submission_df = pd.DataFrame({
        'id': test_ids,
        'label': predictions
    })
    
    submission_path = 'submission_best_model.csv'
    submission_df.to_csv(submission_path, index=False)
    
    print(f"제출 파일이 생성되었습니다: {submission_path}")
    
    # 모델 저장
    joblib.dump(best_model, f'models/best_model_{best_model_name.replace(" ", "_").lower()}.pkl')
    
    # 앙상블 모델 제출 파일도 생성
    if 'Voting Classifier' in models:
        ensemble_predictions = models['Voting Classifier'].predict(X_test)
        
        ensemble_submission_df = pd.DataFrame({
            'id': test_ids,
            'label': ensemble_predictions
        })
        
        ensemble_submission_path = 'submission_ensemble.csv'
        ensemble_submission_df.to_csv(ensemble_submission_path, index=False)
        
        print(f"앙상블 모델 제출 파일이 생성되었습니다: {ensemble_submission_path}")
        
        # 앙상블 모델 저장
        joblib.dump(models['Voting Classifier'], 'models/ensemble_voting_classifier.pkl')

def main():
    """메인 실행 함수"""
    print("=" * 50)
    print("우울증 분류 머신러닝 프로젝트 - 모델 성능 평가")
    print("=" * 50)
    
    # 데이터 로드 및 전처리
    X_train, X_val, X_test, y_train, y_val, test_ids = load_data()
    
    # 모델 훈련
    models, training_times = train_models(X_train, y_train)
    
    # 모델 평가
    results_df = evaluate_models(models, X_val, y_val, training_times)
    
    # 모델 비교
    sorted_results = compare_models(results_df)
    
    # 교차 검증 평가
    cv_df = cross_validation_evaluation(X_train, y_train)
    
    # 최적 모델로 제출 파일 생성
    generate_best_model_submission(models, sorted_results, X_test, test_ids)
    
    # 결과 출력
    print("\n모델 성능 순위:")
    print(sorted_results)
    
    print("\n교차 검증 결과:")
    print(cv_df)
    
    print("\n모델 평가가 완료되었습니다.")
    print(f"결과 시각화는 'plots/evaluation/' 디렉토리에 저장되었습니다.")

if __name__ == "__main__":
    main() 
