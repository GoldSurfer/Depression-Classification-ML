"""
우울증 분류 머신러닝 프로젝트 - 데이터 탐색 및 시각화
- 이름: 이상현
- 학번: 2023713111
- 학과: 인공지능융합학과

오디오 및 비주얼 특징 데이터의 탐색적 분석
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
import os
import warnings

# 경고 메시지 무시
warnings.filterwarnings('ignore')

# 그래프 설정
plt.style.use('seaborn-v0_8-whitegrid')
sns.set(font_scale=1.2)

# 결과 저장을 위한 디렉토리 생성
def create_directory(path):
    """디렉토리가 존재하지 않으면 생성"""
    if not os.path.exists(path):
        os.makedirs(path)

create_directory("plots/eda")

def load_data():
    """데이터 로드 및 기본 정보 출력"""
    print("데이터 로드 중...")
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    
    print(f"훈련 데이터 크기: {train_df.shape}")
    print(f"테스트 데이터 크기: {test_df.shape}")
    
    # 데이터 기본 정보 출력
    print("\n훈련 데이터 기본 정보:")
    print(f"컬럼 개수: {len(train_df.columns)}")
    print(f"결측치 개수: {train_df.isnull().sum().sum()}")
    
    # 라벨 분포 확인
    label_counts = train_df['label'].value_counts()
    print("\n라벨 분포:")
    print(label_counts)
    print(f"라벨 비율 (1/0): {label_counts[1]/label_counts[0]:.4f}")
    
    return train_df, test_df

def analyze_data_distribution(train_df):
    """데이터 분포 분석"""
    print("\n데이터 분포 분석 중...")
    
    # 라벨 분포 시각화
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(x='label', data=train_df)
    plt.title('우울증 라벨 분포')
    plt.xlabel('우울증 여부 (0: 비우울, 1: 우울)')
    plt.ylabel('샘플 수')
    
    # 막대 위에 숫자 표시
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha = 'center', va = 'bottom',
                    xytext = (0, 5), textcoords = 'offset points')
    
    plt.savefig('plots/eda/label_distribution.png')
    plt.close()
    
    # 감정 관련 특성 추출
    emotion_cols = [col for col in train_df.columns if 'emotion_' in col and '_var' not in col 
                   and '_min' not in col and '_max' not in col]
    
    # 감정 관련 특성의 분포 시각화
    plt.figure(figsize=(15, 10))
    train_df[emotion_cols].boxplot()
    plt.title('감정 특성 분포')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('plots/eda/emotion_distribution.png')
    plt.close()
    
    # 각 감정 특성별 라벨에 따른 분포 비교
    plt.figure(figsize=(18, 12))
    for i, col in enumerate(emotion_cols):
        plt.subplot(3, 3, i+1)
        sns.boxplot(x='label', y=col, data=train_df)
        plt.title(col)
    plt.tight_layout()
    plt.savefig('plots/eda/emotion_by_label.png')
    plt.close()
    
    return emotion_cols

def analyze_correlations(train_df, emotion_cols):
    """특성 간 상관관계 분석"""
    print("\n상관관계 분석 중...")
    
    # 감정 특성 간 상관관계
    plt.figure(figsize=(12, 10))
    emotion_corr = train_df[emotion_cols].corr()
    mask = np.triu(np.ones_like(emotion_corr, dtype=bool))
    sns.heatmap(emotion_corr, mask=mask, cmap='coolwarm', vmin=-1, vmax=1, center=0,
                square=True, linewidths=.5, annot=True, fmt='.2f')
    plt.title('감정 특성 간 상관관계')
    plt.tight_layout()
    plt.savefig('plots/eda/emotion_correlation.png')
    plt.close()
    
    # 감정 특성과 라벨 간의 상관관계
    plt.figure(figsize=(12, 1))
    label_corr = pd.DataFrame(train_df[emotion_cols].corrwith(train_df['label'])).T
    sns.heatmap(label_corr, cmap='coolwarm', vmin=-1, vmax=1, center=0, 
                annot=True, fmt='.2f')
    plt.title('감정 특성과 우울증 라벨 간의 상관관계')
    plt.tight_layout()
    plt.savefig('plots/eda/emotion_label_correlation.png')
    plt.close()
    
    # 음성 특성 중 일부 선택
    voice_cols = [col for col in train_df.columns if 'F0' in col and '_mean' in col and '_var' not in col 
                 and '_min' not in col and '_max' not in col][:10]  # 처음 10개만 선택
    
    # 음성 특성과 라벨 간의 상관관계
    plt.figure(figsize=(12, 1))
    voice_label_corr = pd.DataFrame(train_df[voice_cols].corrwith(train_df['label'])).T
    sns.heatmap(voice_label_corr, cmap='coolwarm', vmin=-1, vmax=1, center=0, 
                annot=True, fmt='.2f')
    plt.title('음성 특성과 우울증 라벨 간의 상관관계')
    plt.tight_layout()
    plt.savefig('plots/eda/voice_label_correlation.png')
    plt.close()
    
    return voice_cols

def feature_importance(train_df, emotion_cols, voice_cols):
    """특성 중요도 분석"""
    print("\n특성 중요도 분석 중...")
    
    # 학습 데이터에서 X, y 분리
    X = train_df.drop('label', axis=1)
    y = train_df['label']
    
    # 상호 정보량(Mutual Information) 계산
    selected_cols = emotion_cols + voice_cols
    X_selected = train_df[selected_cols]
    
    # 상호 정보량 계산
    mi_scores = mutual_info_classif(X_selected, y, random_state=42)
    mi_scores = pd.Series(mi_scores, index=selected_cols)
    mi_scores = mi_scores.sort_values(ascending=False)
    
    # 상위 20개 특성 시각화
    plt.figure(figsize=(12, 8))
    mi_scores[:20].plot(kind='barh')
    plt.title('상위 20개 특성의 상호 정보량 (Mutual Information)')
    plt.xlabel('상호 정보량')
    plt.tight_layout()
    plt.savefig('plots/eda/feature_importance_mi.png')
    plt.close()
    
    return mi_scores

def pca_analysis(train_df):
    """PCA를 통한 차원 축소 및 시각화"""
    print("\nPCA 분석 중...")
    
    # 학습 데이터에서 X, y 분리
    X = train_df.drop('label', axis=1)
    y = train_df['label']
    
    # 표준화
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA 적용 (2차원으로 축소)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # 분산 설명률 출력
    explained_variance = pca.explained_variance_ratio_
    print(f"PCA 분산 설명률: {explained_variance}")
    print(f"첫 두 개 주성분의 누적 분산 설명률: {sum(explained_variance):.4f}")
    
    # PCA 결과 시각화
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.7)
    plt.legend(handles=scatter.legend_elements()[0], labels=['비우울', '우울'])
    plt.title('PCA를 통한 우울증 데이터 2차원 시각화')
    plt.xlabel(f'주성분 1 (분산 설명률: {explained_variance[0]:.4f})')
    plt.ylabel(f'주성분 2 (분산 설명률: {explained_variance[1]:.4f})')
    plt.tight_layout()
    plt.savefig('plots/eda/pca_visualization.png')
    plt.close()
    
    # 누적 분산 설명률 그래프
    pca_full = PCA()
    pca_full.fit(X_scaled)
    
    plt.figure(figsize=(10, 6))
    cumsum = np.cumsum(pca_full.explained_variance_ratio_)
    plt.plot(cumsum)
    plt.axhline(y=0.95, color='r', linestyle='-')
    plt.text(len(cumsum)/2, 0.96, '95% 설명률', color='r')
    plt.xlabel('주성분 개수')
    plt.ylabel('누적 분산 설명률')
    plt.title('PCA 누적 분산 설명률')
    plt.grid(True)
    plt.savefig('plots/eda/pca_cumulative_variance.png')
    plt.close()

def analyze_by_group(train_df, emotion_cols):
    """우울증 유무에 따른 그룹별 분석"""
    print("\n그룹별 분석 중...")
    
    # 우울증 유무에 따른 그룹 생성
    depressed = train_df[train_df['label'] == 1]
    not_depressed = train_df[train_df['label'] == 0]
    
    # 감정 특성의 평균 비교
    emotion_means = pd.DataFrame({
        '우울증': depressed[emotion_cols].mean(),
        '비우울증': not_depressed[emotion_cols].mean()
    })
    
    # 막대 그래프로 시각화
    plt.figure(figsize=(12, 8))
    emotion_means.plot(kind='bar')
    plt.title('우울증 유무에 따른 감정 특성 평균 비교')
    plt.xlabel('감정 특성')
    plt.ylabel('평균값')
    plt.legend(title='그룹')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('plots/eda/emotion_means_by_group.png')
    plt.close()
    
    # 방사형 그래프(레이더 차트)로 시각화
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, polar=True)
    
    # 각도 설정
    angles = np.linspace(0, 2*np.pi, len(emotion_cols), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))  # 그래프 폐합을 위해 처음 값 추가
    
    # 데이터 준비
    depressed_vals = depressed[emotion_cols].mean().values
    depressed_vals = np.concatenate((depressed_vals, [depressed_vals[0]]))
    
    not_depressed_vals = not_depressed[emotion_cols].mean().values
    not_depressed_vals = np.concatenate((not_depressed_vals, [not_depressed_vals[0]]))
    
    # 그래프 그리기
    ax.plot(angles, depressed_vals, 'o-', linewidth=2, label='우울증')
    ax.plot(angles, not_depressed_vals, 'o-', linewidth=2, label='비우울증')
    ax.fill(angles, depressed_vals, alpha=0.25)
    ax.fill(angles, not_depressed_vals, alpha=0.25)
    
    # 그래프 설정
    ax.set_thetagrids(angles[:-1] * 180/np.pi, emotion_cols)
    plt.title('우울증 유무에 따른 감정 특성 비교 (레이더 차트)')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('plots/eda/emotion_radar_by_group.png')
    plt.close()

def main():
    """메인 실행 함수"""
    print("=" * 50)
    print("우울증 분류 머신러닝 프로젝트 - 데이터 탐색 및 시각화")
    print("=" * 50)
    
    # 데이터 로드
    train_df, test_df = load_data()
    
    # 데이터 분포 분석
    emotion_cols = analyze_data_distribution(train_df)
    
    # 상관관계 분석
    voice_cols = analyze_correlations(train_df, emotion_cols)
    
    # 특성 중요도 분석
    feature_importance(train_df, emotion_cols, voice_cols)
    
    # PCA 분석
    pca_analysis(train_df)
    
    # 그룹별 분석
    analyze_by_group(train_df, emotion_cols)
    
    print("\n데이터 탐색 및 시각화가 완료되었습니다.")
    print(f"결과는 'plots/eda/' 디렉토리에 저장되었습니다.")

if __name__ == "__main__":
    main()