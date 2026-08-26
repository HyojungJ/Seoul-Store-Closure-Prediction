# TabNet 폐업률 예측 모델 - Data Leakage 완전 제거 버전

## 개요

본 문서는 `tabnet_fixed_no_leakage.ipynb`에 구현된 TabNet 기반 폐업률 예측 모델에 대한 설명입니다.
기존 모델에서 발견된 Data Leakage 문제를 완전히 제거하고, 과거 시점 데이터만을 사용하여 현실적인 예측 시나리오를 구현했습니다.

---

## 사용 데이터

### 데이터 출처
- 파일: `eda/data/merged_data.csv`
- 전체 데이터: 39,975개 레코드
- 데이터 기간: 2019년 1분기 ~ 2025년 2분기
- 총 컬럼 수: 137개

### 주요 데이터 필드
- 기준 정보: 기준_년분기_코드, 자치구_코드_명, 서비스_업종_코드_명
- 매출 정보: 당월_매출_금액, 당월_매출_건수, 요일별/시간대별/연령대별 매출
- 점포 정보: 점포_수, 유사_업종_점포_수, 프랜차이즈_점포_수, 개업_률, 개업_점포_수
- 임대료 정보: 전체임대료
- 인구 정보: 총_유동인구_수, 총_상주인구_수, 총_직장인구_수
- 상권 정보: 상권_변화_지표
- 타겟 관련: 폐업_률, 폐업_점포_수

---

## 데이터 전처리

### 1. 타겟 변수 생성
- 전체 데이터의 폐업률 중앙값(2.30%)을 임계값으로 설정
- 폐업률이 임계값 이상이면 1(High Risk), 미만이면 0(Low Risk)로 이진 분류
- 클래스 분포: Low Risk 19,655개, High Risk 20,320개

### 2. 범주형 변수 인코딩
- 자치구_코드_명: 25개 클래스
- 서비스_업종_코드_명: 63개 클래스
- 상권_변화_지표: 4개 클래스

### 3. Lag 특성 생성 (Data Leakage 방지 핵심)
**중요: 현재 시점(t)의 원본 데이터는 절대 사용하지 않음**

20개 주요 변수에 대해 t-1, t-2 시점의 lag 특성 생성:
- 당월_매출_금액_lag1, 당월_매출_금액_lag2
- 당월_매출_건수_lag1, 당월_매출_건수_lag2
- 점포_수_lag1, 점포_수_lag2
- 유사_업종_점포_수_lag1, 유사_업종_점포_수_lag2
- 프랜차이즈_점포_수_lag1, 프랜차이즈_점포_수_lag2
- 개업_률_lag1, 개업_률_lag2
- 개업_점포_수_lag1, 개업_점포_수_lag2
- 전체임대료_lag1, 전체임대료_lag2
- 총_유동인구_수_lag1, 총_유동인구_수_lag2
- 총_상주인구_수_lag1, 총_상주인구_수_lag2
- 총_직장인구_수_lag1, 총_직장인구_수_lag2
- 토요일_매출_금액_lag1, 일요일_매출_금액_lag1
- 시간대_21_24_매출_금액_lag1
- 연령대별_매출_금액_lag1 (10대, 20대, 30대, 40대, 50대, 60대 이상)

총 40개의 lag 특성 생성

### 4. 파생 특성 생성
모든 파생 특성은 t-1, t-2 시점 데이터만 사용:

**변화율 특성:**
- 매출_변화율: (t-1 매출 - t-2 매출) / t-2 매출
- 매출건수_변화율
- 점포수_변화율
- 개업률_변화

**추세 특성:**
- 매출_감소: 매출 변화율이 음수인지 여부
- 연속_매출_감소: 연속으로 매출이 감소한 분기 수
- 매출_3분기_평균: 과거 3분기 매출 평균
- 매출_추세_대비: 현재 매출 / 3분기 평균

**수익성 지표:**
- 점포당_매출: 총 매출 / 점포 수
- 건당_매출: 총 매출 / 매출 건수
- 임대료_부담률: 임대료 / 매출
- 유동인구_전환율: 매출 건수 / 유동인구

**고객 구조:**
- 최대_연령대_비중: 최대 연령대 매출 / 전체 매출
- 주말_매출_비율: (토요일 + 일요일 매출) / 전체 매출
- 야간_매출_비율: 21-24시 매출 / 전체 매출

**경쟁 환경:**
- 프랜차이즈_비율: 프랜차이즈 점포 수 / 전체 점포 수
- 경쟁_밀도: 유사 업종 점포 수 / 전체 점포 수
- 점포_포화도: 점포 수 / (상주인구 + 직장인구) × 10,000

### 5. 그룹 통계량 (Leakage 방지)
**Train 데이터에서만 계산하고 Val/Test에 적용:**
- 상권_매출_평균: 자치구별 평균 매출
- 업종_매출_평균: 업종별 평균 매출
- 상권_대비_매출: 현재 매출 / 상권 평균
- 업종_대비_매출: 현재 매출 / 업종 평균
- 위험_점수: 복합 위험 지표 (매출 감소, 임대료 부담, 상권 대비 매출 등)

### 6. Feature 선택
- 최종 Feature 수: 66개
  - 범주형 Feature: 3개 (자치구, 업종, 상권변화)
  - 연속형 Feature: 63개
- 제외된 Feature: 141개
  - 타겟 변수: 폐업_률, 폐업_점포_수, closure_risk
  - 현재 시점(t) 모든 원본 데이터
  - 시간 정보: year, quarter, year_quarter

---

## 데이터 분할

### 시계열 기준 분할
- 각 그룹(자치구-업종)의 초기 2분기 데이터 제거 (lag2 생성을 위해)
- 전체 데이터: 36,875개 (초기 분기 제거 후)

**분할 비율:**
- Train: 25,812개 (70.0%)
  - Class 0: 12,938개, Class 1: 12,874개
- Validation: 3,687개 (10.0%)
  - Class 0: 1,705개, Class 1: 1,982개
- Test: 7,376개 (20.0%)
  - Class 0: 3,568개, Class 1: 3,808개

### 스케일링
- 연속형 변수: StandardScaler 적용 (Train 기준으로 fit)
- 범주형 변수: 스케일링 미적용

---

## 모델 학습

### 모델 구조: TabNet Classifier
- n_d, n_a: 8 (Decision/Attention dimension)
- n_steps: 3 (Sequential attention steps)
- gamma: 1.4 (Feature reuse coefficient)
- lambda_sparse: 0.000009 (Sparsity regularization)
- cat_emb_dim: 3 (Categorical embedding dimension)
- mask_type: 'entmax' (Sparse attention mechanism)

### 학습 설정
- Optimizer: Adam (learning rate: 0.01)
- Scheduler: StepLR (step_size: 50, gamma: 0.9)
- Batch size: 1,024
- Virtual batch size: 128
- Max epochs: 150
- Early stopping patience: 25
- Device: Mac GPU (MPS)

### 학습 결과
- Best Epoch: 48
- Best Validation Loss: 0.5254
- 학습 시간: 약 2분

**학습 곡선:**
- Epoch 0: val_accuracy 0.608, val_auc 0.664
- Epoch 10: val_accuracy 0.697, val_auc 0.782
- Epoch 24: val_accuracy 0.710, val_auc 0.800
- Epoch 48 (Best): val_accuracy 0.730, val_auc 0.814

---

## 모델 성능

### 최적 임계값
- Validation 데이터에서 ROC 곡선의 Youden Index를 사용하여 결정
- 최적 임계값: 0.4401

### Test 세트 최종 성능
- Accuracy: 0.7061 (70.61%)
- Precision: 0.7221 (72.21%)
- Recall: 0.7001 (70.01%)
- F1 Score: 0.7109
- AUC: 0.7807

### Confusion Matrix
```
              Predicted
              Low Risk  High Risk
Actual Low     2,542     1,026
      High     1,142     2,666
```

### 클래스별 성능
- Low Risk (Class 0):
  - Precision: 0.69, Recall: 0.71, F1-score: 0.70
- High Risk (Class 1):
  - Precision: 0.72, Recall: 0.70, F1-score: 0.71

---

## Feature Importance (Top 20)

1. 야간_매출_비율: 0.2188
2. 개업_률_lag1: 0.0749
3. 개업_점포_수_lag1: 0.0611
4. 점포_수_lag2: 0.0593
5. 프랜차이즈_점포_수_lag1: 0.0539
6. 주말_매출_비율: 0.0439
7. 건당_매출: 0.0389
8. 업종_encoded: 0.0377
9. 개업_률_lag2: 0.0338
10. 프랜차이즈_점포_수_lag2: 0.0335
11. 총_상주인구_수_lag2: 0.0333
12. 당월_매출_건수_lag2: 0.0305
13. 당월_매출_건수_lag1: 0.0252
14. 총_직장인구_수_lag2: 0.0239
15. 매출건수_변화율: 0.0183
16. 연령대_50_매출_금액_lag2: 0.0175
17. 당월_매출_금액_lag1: 0.0153
18. 임대료_부담률: 0.0151
19. 위험_점수: 0.0132
20. 연령대_40_매출_금액_lag2: 0.0130

---

## Data Leakage 방지 전략

### 1. 타겟 변수 정의
- 현재 분기(t)의 폐업률을 기준으로 위험도 분류
- 단, 예측에는 과거 시점(t-1, t-2) 데이터만 사용

### 2. 폐업률 관련 특성 완전 제거
- 폐업_률, 폐업_점포_수 변수는 절대 사용하지 않음
- 이들의 lag 특성도 생성하지 않음

### 3. 현재 시점(t) 데이터 사용 금지
- 모든 원본 변수의 현재 값은 제외
- 오직 lag1, lag2 특성만 사용

### 4. 그룹 통계량 계산 시 Train만 사용
- 상권별, 업종별 평균 등은 Train 데이터에서만 계산
- Validation/Test에는 Train의 통계량을 적용

### 5. 시계열 순서 보존
- 시간 순서대로 Train/Val/Test 분할
- 미래 정보가 과거로 유입되지 않도록 보장

---

## 결론

본 모델은 Data Leakage를 완전히 제거하고 현실적인 예측 시나리오를 구현했습니다.
과거 시점(t-1, t-2)의 데이터만을 사용하여 현재 시점(t)의 폐업 위험을 예측하며,
Test 세트에서 AUC 0.78, Accuracy 70.6%의 성능을 달성했습니다.

주요 예측 변수는 야간 매출 비율, 개업률, 점포 수, 프랜차이즈 비율 등이며,
이는 실무적으로도 합리적인 폐업 위험 지표로 해석됩니다.

---

## 파일 정보
- 노트북: `ml/dl/tabnet_fixed_no_leakage.ipynb`
- 저장 데이터:
  - `ml/dl/preprocessed/train_features.csv`
  - `ml/dl/preprocessed/train_target.csv`
  - `ml/dl/preprocessed/val_features.csv`
  - `ml/dl/preprocessed/val_target.csv`
  - `ml/dl/preprocessed/test_features.csv`
  - `ml/dl/preprocessed/test_target.csv`
