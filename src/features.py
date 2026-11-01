"""
폐업 위험 예측 피처 생성 모듈.

notebooks/02_preprocessing.ipynb 의 전처리·피처 엔지니어링을 그대로 옮긴 것으로,
학습(노트북)과 서비스(Streamlit 대시보드)가 같은 피처 정의를 쓰도록 하기 위한 모듈입니다.
notebooks/05_catboost_final.ipynb 에서 이 모듈로 다시 만든 피처가
data/processed/{train,val,test}_features.csv 와 일치하는지 검증합니다.

피처 구성 (66개): 범주형 3 + lag 40 (t-1, t-2) + 파생 18 + 그룹 통계 5 (train-only)
예측 대상: 분기 t의 폐업 위험 (폐업률 >= 중앙값) — 입력은 t-1, t-2 시점 값만 사용
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

GROUP_KEYS = ["자치구_encoded", "업종_encoded"]

CATEGORICAL_FEATURES = ["자치구_encoded", "업종_encoded", "상권변화_encoded"]

LAG_COLS = [
    "당월_매출_금액", "당월_매출_건수", "점포_수", "유사_업종_점포_수",
    "프랜차이즈_점포_수", "개업_률", "개업_점포_수",
    "전체임대료", "총_유동인구_수", "총_상주인구_수", "총_직장인구_수",
    "토요일_매출_금액", "일요일_매출_금액", "시간대_21_24_매출_금액",
    "연령대_10_매출_금액", "연령대_20_매출_금액", "연령대_30_매출_금액",
    "연령대_40_매출_금액", "연령대_50_매출_금액", "연령대_60_이상_매출_금액",
]

LAG_FEATURES = [f"{col}_lag{k}" for col in LAG_COLS for k in (1, 2)]

DERIVED_FEATURES = [
    "매출_변화율", "매출건수_변화율", "점포수_변화율", "개업률_변화",
    "매출_감소", "연속_매출_감소", "매출_3분기_평균", "매출_추세_대비",
    "점포당_매출", "건당_매출", "임대료_부담률", "유동인구_전환율",
    "최대_연령대_비중", "주말_매출_비율", "야간_매출_비율",
    "프랜차이즈_비율", "경쟁_밀도", "점포_포화도",
]

GROUP_STAT_FEATURES = ["상권_매출_평균", "업종_매출_평균", "상권_대비_매출", "업종_대비_매출", "위험_점수"]

FEATURE_COLS = CATEGORICAL_FEATURES + LAG_FEATURES + DERIVED_FEATURES + GROUP_STAT_FEATURES


# ---------------------------------------------------------------------------
# 1. 시간 변수 + 범주형 인코딩 (02 - 1단계)
# ---------------------------------------------------------------------------
def fit_encoders(merged: pd.DataFrame) -> dict[str, LabelEncoder]:
    """02와 동일하게 병합 데이터 전체에 LabelEncoder를 학습합니다."""
    return {
        "district": LabelEncoder().fit(merged["자치구_코드_명"]),
        "industry": LabelEncoder().fit(merged["서비스_업종_코드_명"]),
        "change": LabelEncoder().fit(merged["상권_변화_지표"]),
    }


def add_time_and_encoding(df: pd.DataFrame, encoders: dict[str, LabelEncoder]) -> pd.DataFrame:
    df = df.copy()
    code = df["기준_년분기_코드"].astype(int).astype(str)
    df["year"] = code.str[:4].astype(int)
    df["quarter"] = code.str[4:].astype(int)
    df["year_quarter"] = df["year"] * 10 + df["quarter"]
    df["자치구_encoded"] = encoders["district"].transform(df["자치구_코드_명"])
    df["업종_encoded"] = encoders["industry"].transform(df["서비스_업종_코드_명"])
    df["상권변화_encoded"] = encoders["change"].transform(df["상권_변화_지표"])
    return df.sort_values(GROUP_KEYS + ["year_quarter"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Lag + 파생 변수 (02 - 3, 4단계)
# ---------------------------------------------------------------------------
def add_lag_and_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    group = df.groupby(GROUP_KEYS)
    for col in LAG_COLS:
        df[f"{col}_lag1"] = group[col].shift(1)
        df[f"{col}_lag2"] = group[col].shift(2)

    g = df.groupby(GROUP_KEYS)
    df["매출_변화율"] = (df["당월_매출_금액_lag1"] - df["당월_매출_금액_lag2"]) / (df["당월_매출_금액_lag2"] + 1)
    df["매출건수_변화율"] = (df["당월_매출_건수_lag1"] - df["당월_매출_건수_lag2"]) / (df["당월_매출_건수_lag2"] + 1)
    df["점포수_변화율"] = (df["점포_수_lag1"] - df["점포_수_lag2"]) / (df["점포_수_lag2"] + 1)
    df["개업률_변화"] = df["개업_률_lag1"] - df["개업_률_lag2"]

    df["매출_감소"] = (df["매출_변화율"] < 0).astype(int)
    df["연속_매출_감소"] = g["매출_감소"].shift(1).rolling(2, min_periods=1).sum()

    df["매출_3분기_평균"] = g["당월_매출_금액_lag1"].shift(1).rolling(3, min_periods=1).mean()
    df["매출_추세_대비"] = df["당월_매출_금액_lag1"] / (df["매출_3분기_평균"] + 1)

    df["점포당_매출"] = df["당월_매출_금액_lag1"] / (df["점포_수_lag1"] + 1)
    df["건당_매출"] = df["당월_매출_금액_lag1"] / (df["당월_매출_건수_lag1"] + 1)
    df["임대료_부담률"] = df["전체임대료_lag1"] / (df["당월_매출_금액_lag1"] + 1)
    df["유동인구_전환율"] = df["당월_매출_건수_lag1"] / (df["총_유동인구_수_lag1"] + 1)

    age_cols_lag1 = [f"연령대_{age}_매출_금액_lag1" for age in ["10", "20", "30", "40", "50", "60_이상"]]
    df["최대_연령대_비중"] = df[age_cols_lag1].max(axis=1) / (df["당월_매출_금액_lag1"] + 1)
    df["주말_매출_비율"] = (df["토요일_매출_금액_lag1"] + df["일요일_매출_금액_lag1"]) / (df["당월_매출_금액_lag1"] + 1)
    df["야간_매출_비율"] = df["시간대_21_24_매출_금액_lag1"] / (df["당월_매출_금액_lag1"] + 1)

    df["프랜차이즈_비율"] = df["프랜차이즈_점포_수_lag1"] / (df["점포_수_lag1"] + 1)
    df["경쟁_밀도"] = df["유사_업종_점포_수_lag1"] / (df["점포_수_lag1"] + 1)
    df["점포_포화도"] = df["점포_수_lag1"] / ((df["총_상주인구_수_lag1"] + df["총_직장인구_수_lag1"]) + 1) * 10000

    return df.replace([np.inf, -np.inf], np.nan).fillna(0)


# ---------------------------------------------------------------------------
# 3. 그룹 통계 (02 - 6단계, 평균은 train에서만 계산)
# ---------------------------------------------------------------------------
def compute_group_means(train_features: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """train 데이터에서만 자치구/업종별 직전 분기 매출 평균을 계산합니다."""
    district_mean = train_features.groupby("자치구_encoded")["당월_매출_금액_lag1"].mean()
    industry_mean = train_features.groupby("업종_encoded")["당월_매출_금액_lag1"].mean()
    return district_mean, industry_mean


def _risk_score(df: pd.DataFrame) -> pd.Series:
    return (
        (df["매출_변화율"] < -0.1).astype(int) * 2
        + (df["임대료_부담률"] > 0.3).astype(int) * 2
        + (df["상권_대비_매출"] < 0.7).astype(int)
        + (df["연속_매출_감소"] >= 2).astype(int) * 3
    )


def add_group_stats(df: pd.DataFrame, district_mean: pd.Series, industry_mean: pd.Series) -> pd.DataFrame:
    df = df.copy()
    df["상권_매출_평균"] = df["자치구_encoded"].map(district_mean)
    df["업종_매출_평균"] = df["업종_encoded"].map(industry_mean)
    df["상권_대비_매출"] = df["당월_매출_금액_lag1"] / (df["상권_매출_평균"] + 1)
    df["업종_대비_매출"] = df["당월_매출_금액_lag1"] / (df["업종_매출_평균"] + 1)
    df["위험_점수"] = _risk_score(df)
    return df.replace([np.inf, -np.inf], np.nan).fillna(0)


# ---------------------------------------------------------------------------
# 4. 학습 데이터셋 재현 (02 - 2, 5단계) — 노트북 05에서 일치 검증용
# ---------------------------------------------------------------------------
def build_train_val_test(merged: pd.DataFrame, encoders: dict[str, LabelEncoder]):
    """02와 같은 순서로 피처·타겟을 만들고 시간 순서 70/10/20 분할합니다."""
    df = add_time_and_encoding(merged, encoders)
    threshold = df["폐업_률"].quantile(0.5)
    df["closure_risk"] = (df["폐업_률"] >= threshold).astype(int)
    df = add_lag_and_derived(df)

    df = df.sort_values(GROUP_KEYS + ["year_quarter"]).reset_index(drop=True)
    grp = df.groupby(GROUP_KEYS)
    idx, size = grp.cumcount(), grp["year_quarter"].transform("size")
    df = df[(idx >= 2) | (size <= 2)].reset_index(drop=True)

    df = df.sort_values(["year_quarter"] + GROUP_KEYS).reset_index(drop=True)
    n = len(df)
    n_train, n_val = int(n * 0.7), int(n * 0.1)
    splits = df.iloc[:n_train].copy(), df.iloc[n_train:n_train + n_val].copy(), df.iloc[n_train + n_val:].copy()

    district_mean, industry_mean = compute_group_means(splits[0])
    return [add_group_stats(s, district_mean, industry_mean) for s in splits], threshold


# ---------------------------------------------------------------------------
# 5. 서비스용: 다음 분기 예측 피처
# ---------------------------------------------------------------------------
def _next_quarter(yq: int) -> int:
    year, quarter = divmod(int(yq), 10)
    return (year + 1) * 10 + 1 if quarter == 4 else year * 10 + quarter + 1


def build_next_quarter_features(
    merged: pd.DataFrame,
    encoders: dict[str, LabelEncoder],
    district_mean: pd.Series,
    industry_mean: pd.Series,
) -> pd.DataFrame:
    """
    (자치구 x 업종)마다 마지막 관측 분기의 '다음 분기' 예측용 피처 행을 만듭니다.

    - 학습 때와 같은 방식으로 lag1 = 마지막 관측 분기, lag2 = 그 직전 분기 값이 됩니다.
    - 상권_변화_지표는 아직 관측되지 않았으므로 마지막 관측값을 그대로 씁니다.
    - 반환: 자치구_코드_명, 서비스_업종_코드_명, 기준_분기(lag1 시점), 예측_분기 + FEATURE_COLS
    """
    last = merged.sort_values("기준_년분기_코드").groupby(["자치구_코드_명", "서비스_업종_코드_명"]).tail(1)
    future = last[["자치구_코드_명", "서비스_업종_코드_명", "상권_변화_지표", "기준_년분기_코드"]].copy()
    future["기준_년분기_코드"] = future["기준_년분기_코드"].map(_next_quarter)
    future["_is_future"] = True

    df = pd.concat([merged.assign(_is_future=False), future], ignore_index=True)
    df = add_lag_and_derived(add_time_and_encoding(df, encoders))
    out = df[df["_is_future"]].copy()
    out = add_group_stats(out, district_mean, industry_mean)
    out["예측_분기"] = out["year_quarter"]
    out["기준_분기"] = out["year_quarter"].map(lambda yq: yq - 1 if yq % 10 != 1 else (yq // 10 - 1) * 10 + 4)
    keep = ["자치구_코드_명", "서비스_업종_코드_명", "기준_분기", "예측_분기"] + FEATURE_COLS
    return out[keep].reset_index(drop=True)


def apply_rent(row: pd.Series, rent_per_area: float) -> pd.Series:
    """
    사용자가 입력한 현재 임대료(3.3㎡ 기준)를 최근 관측 임대료(전체임대료_lag1)로 반영하고,
    이 값에 의존하는 파생 피처(임대료_부담률, 위험_점수)를 다시 계산합니다.
    """
    row = row.copy()
    row["전체임대료_lag1"] = float(rent_per_area)
    row["임대료_부담률"] = row["전체임대료_lag1"] / (row["당월_매출_금액_lag1"] + 1)
    row["위험_점수"] = (
        int(row["매출_변화율"] < -0.1) * 2
        + int(row["임대료_부담률"] > 0.3) * 2
        + int(row["상권_대비_매출"] < 0.7)
        + int(row["연속_매출_감소"] >= 2) * 3
    )
    return row
