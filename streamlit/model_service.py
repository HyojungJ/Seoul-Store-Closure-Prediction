"""
대시보드 공용 예측 서비스.

- 모델: notebooks/05_catboost_final.ipynb 가 저장한 model/catboost/closure_model.pkl
  (04 비교와 동일 설정 CatBoost, 테스트 Accuracy 0.713 · F1 0.694 · ROC-AUC 0.796)
- 피처: src/features.py (02 전처리와 동일, 05에서 일치 검증)
- 예측: (자치구 x 업종)의 마지막 관측 분기를 t-1로 보고 '다음 분기' 폐업 위험 확률을 계산
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import features as F  # noqa: E402

DATA_PATH = ROOT / "data" / "processed" / "merged_data.csv"  # notebooks/00_data_merge 산출물
MODEL_PATH = ROOT / "model" / "catboost" / "closure_model.pkl"  # notebooks/05_catboost_final 산출물


@st.cache_resource(show_spinner=False)
def load_prediction_service():
    """모델 번들과 (자치구 x 업종)별 다음 분기 피처 테이블을 한 번만 만들어 캐시합니다."""
    try:
        bundle = joblib.load(MODEL_PATH)
        merged = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
        table = F.build_next_quarter_features(
            merged, bundle["encoders"], bundle["district_mean"], bundle["industry_mean"]
        )
        return {"bundle": bundle, "table": table.set_index(["자치구_코드_명", "서비스_업종_코드_명"])}
    except Exception as e:  # 모델/데이터가 없으면 페이지에서 안내
        st.error(f"모델 로드 실패: {e} (notebooks 00 → 02 → 05 순서로 실행해 산출물을 만들어 주세요)")
        return None


def predict_closure_risk(service, district: str, industry: str, rent_per_area: float) -> tuple[float, int]:
    """
    다음 분기 폐업 위험 확률(0~1)과 예측 대상 분기(YYYYQ)를 반환합니다.
    사용자가 입력한 임대료(3.3㎡ 기준)는 최근 임대료(전체임대료_lag1)로 반영합니다.
    """
    key = (district, industry)
    if key not in service["table"].index:
        raise KeyError(f"{district} - {industry} 예측용 데이터가 없습니다.")

    row = service["table"].loc[key]
    if rent_per_area:
        row = F.apply_rent(row, rent_per_area)

    bundle = service["bundle"]
    X = pd.DataFrame([row[bundle["feature_names"]]]).astype(float)
    cat_cols = [bundle["feature_names"][i] for i in bundle["cat_features"]]
    X[cat_cols] = X[cat_cols].astype(int)
    proba = float(bundle["model"].predict_proba(X)[:, 1][0])
    return proba, int(row["예측_분기"])
