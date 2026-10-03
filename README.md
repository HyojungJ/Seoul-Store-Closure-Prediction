# 서울시 자치구별 매장 폐업률 예측

> 📌 이 저장소는 SKN 19기 2차 팀 프로젝트 결과물을 개인적으로 **정리한 버전**입니다.
> 원본 레포지토리: [SKNetworks-AI19-250818/SKN19_2nd_1team](https://github.com/SKNetworks-AI19-250818/SKN19_2nd_1team)

서울시 상권분석 데이터를 활용해 자치구·업종별 **폐업 위험도를 예측**하는 머신러닝/딥러닝 프로젝트입니다.

## 개요

- 서울시 공공데이터(점포, 매출, 상권변화, 유동/상주/직장인구, 소득소비, 임대료) 8종을 병합
- 폐업률 상위 구간을 '위험'으로 보는 이진 분류 문제로 정의
- RandomForest · XGBoost · LightGBM · CatBoost · TabNet 비교, **CatBoost 최종 채택**

## 디렉토리 구조

```
.
├─ data
│  ├─ raw          # 원본 CSV (자치구 단위 8종)
│  ├─ eda          # EDA 후 정제 데이터 + 병합본(merged_data.csv)
│  └─ processed    # 학습/검증/평가 분할 데이터
├─ notebooks
│  ├─ 01_EDA.ipynb                  # 탐색적 데이터 분석
│  ├─ 02_preprocessing.ipynb        # 전처리 및 병합
│  ├─ 03_regression.ipynb           # 회귀 시도 (RF/CatBoost/XGBoost/LightGBM)
│  ├─ 04_classification.ipynb       # 기본 분류
│  ├─ 05_classification_tuning.ipynb # Optuna + SMOTE 튜닝
│  └─ 06_tabnet.ipynb               # TabNet 딥러닝
├─ model           # 학습 산출물 (gitignore)
└─ streamlit       # 대시보드 앱
```

## 분석 흐름

1. **EDA / 전처리** (`01`, `02`): 8종 데이터 탐색 후 자치구·업종·분기 기준 병합
2. **회귀 시도** (`03`): 폐업률을 연속값으로 예측 → R²가 낮아 분류로 전환
3. **분류** (`04`, `05`): 폐업률 상위 25%를 '위험'으로 라벨링, SMOTE로 불균형 보정, Optuna 튜닝
4. **딥러닝** (`06`): TabNet(기본/Optuna, 누수 제거 버전) 비교

## 결과

폐업률 상위 25% 기준 이진 분류에서 **CatBoost가 가장 우수**(튜닝 후 Accuracy 약 88%)하여 최종 모델로 선정했습니다. 최종 모델은 Streamlit 대시보드에서 사용자 입력(자치구·업종 등)에 따른 폐업 위험도 예측에 활용됩니다.

## 기술 스택

Python · pandas · numpy · scikit-learn · XGBoost · LightGBM · CatBoost · TabNet · Optuna · imbalanced-learn · Streamlit

## 데이터 출처

서울시 상권분석 서비스 / 서울 열린데이터광장 — 점포·상권변화·추정매출·상주인구·길단위인구·직장인구·소득소비(자치구), 지역별 임대시세
