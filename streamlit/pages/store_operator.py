# store_operator.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from pathlib import Path
from typing import Optional

# 프로젝트 루트 기준 경로 (실행 위치와 무관하게 동작)
ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / 'data' / 'processed' / 'merged_data.csv'  # notebooks/00_data_merge 산출물
MODEL_PATH = ROOT / 'model' / 'catboost' / 'model_and_data_encoded.pkl'

# ==================== 페이지 설정/스타일 ====================
st.set_page_config(
    page_title="매장 운영자를 위한 폐업 예측",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 커스텀 CSS (예비 창업자 페이지 스타일과 정렬)
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }

    .header-container {
        background: #1e40af;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    /* 모든 버튼 공통 스타일 */
    .stButton>button {
        width: 100%;
        background: #1e40af !important;
        color: white !important;
        border: 2px solid transparent !important;
        padding: 0.75rem;
        border-radius: 8px;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        background-color: #ffffff !important;
        color: #1e40af !important;
        border: 2px solid #1e40af !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4) !important;
    }

    .stat-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        text-align: center;
        transition: transform 0.3s ease;
    }

    .stat-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }

    .stat-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #667eea;
        margin: 0.5rem 0;
    }

    .stat-label {
        font-size: 1rem;
        color: #6c757d;
        margin-bottom: 0.5rem;
        position: relative;
        display: inline-block;
    }

    .tooltip-icon {
        display: inline-block;
        margin-left: 5px;
        color: #667eea;
        cursor: help;
        font-size: 0.9rem;
    }

    .tooltip-icon:hover::after {
        content: attr(data-tooltip);
        position: absolute;
        left: 50%;
        top: -40px;
        transform: translateX(-50%);
        background-color: #333;
        color: white;
        padding: 8px 12px;
        border-radius: 6px;
        white-space: nowrap;
        font-size: 0.85rem;
        z-index: 1000;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }

    .tooltip-icon:hover::before {
        content: '';
        position: absolute;
        left: 50%;
        top: -8px;
        transform: translateX(-50%);
        border: 6px solid transparent;
        border-top-color: #333;
        z-index: 1000;
    }

    .info-box {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
    }

    .warning-box {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }

    .success-box {
        background: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }

    .danger-box {
        background: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }

    .industry-card-top {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-left: 4px solid #2196F3;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 0.8rem;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    .industry-card-top:hover {
        transform: translateX(5px);
        box-shadow: 0 4px 8px rgba(33, 150, 243, 0.3);
    }

    .industry-card-safe {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        border-left: 4px solid #4CAF50;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 0.8rem;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    .industry-card-safe:hover {
        transform: translateX(5px);
        box-shadow: 0 4px 8px rgba(76, 175, 80, 0.3);
    }

    .industry-card-risky {
        background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
        border-left: 4px solid #f44336;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 0.8rem;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    .industry-card-risky:hover {
        transform: translateX(5px);
        box-shadow: 0 4px 8px rgba(244, 67, 54, 0.3);
    }

    .rank-badge {
        display: inline-block;
        font-size: 1.2rem;
        margin-right: 0.5rem;
    }

    .industry-name {
        font-size: 1.1rem;
        font-weight: bold;
        color: #212121;
        margin-bottom: 0.3rem;
    }

    .industry-stats {
        font-size: 0.9rem;
        color: #424242;
    }
</style>
""", unsafe_allow_html=True)

if st.button("Home"):
    st.switch_page("app.py")

# 헤더
st.markdown("""
<div class="header-container">
    <h1 style='margin:0; font-size: 2.5rem;'>매장 운영자를 위한 폐업 위험 예측</h1>
    <p style='margin-top: 1rem; font-size: 1.2rem; opacity: 0.9;'>
        AI 기반 데이터 분석으로 현재 매장의 폐업 위험도를 미리 확인하세요
    </p>
</div>
""", unsafe_allow_html=True)



# ==================== 데이터/모델 로드 (예비 창업자 파일 구조 준용) ====================
@st.cache_data
def load_merged_data():
    try:
        df = pd.read_csv(DATA_PATH)
        return df
    except Exception as e:
        st.error(f"데이터 로드 실패: {str(e)}")
        return None

@st.cache_resource
def load_model_and_encoders():
    try:
        data = joblib.load(MODEL_PATH)
        return (
            data['model'],
            data['district_encoder'],
            data['industry_encoder'],
            data['sanggwon_encoder'],
            data.get('feature_names', [])
        )
    except Exception as e:
        st.error(f"모델 로드 실패: {str(e)}")
        return None, None, None, None, []

def get_district_industry_data(df, district, industry):
    filtered = df[(df['자치구_코드_명'] == district) & (df['서비스_업종_코드_명'] == industry)]
    if len(filtered) == 0:
        return None
    latest = filtered.sort_values('기준_년분기_코드', ascending=False).iloc[0]
    return latest


def get_day_recommendation(user_days: list, avg_day_profile: dict) -> str | None:
    """사용자 선택 요일과 지역 평균 매출 요일을 비교하여 조언 생성"""
    if not user_days:
        return None

    # 지역 평균 매출 상위 2개 요일 찾기
    top_avg_days = sorted(avg_day_profile, key=avg_day_profile.get, reverse=True)[:2]

    # 사용자가 영업하지 않지만, 매출이 높은 요일 찾기
    missed_opportunities = [day for day in top_avg_days if day not in user_days]

    if missed_opportunities:
        return f"🗓️ **영업 요일 조정 제안:** 이 상권은 특히 **{', '.join(missed_opportunities)}**에 매출이 높습니다. 해당 요일에 영업 확장을 고려해보세요."
    
    return f"🗓️ **영업 요일 분석:** 현재 영업 요일이 상권의 주요 매출 발생 요일과 잘 맞습니다. 현재 전략을 유지하며 마케팅을 강화하는 것을 추천합니다."

def build_avg_time_sales_profile_from_row(row: pd.Series) -> dict:
    """row에서 시간대별 매출 금액을 퍼센트(0~100)로 정규화"""
    time_sales_keys = [
        ("00-06시", "시간대_00_06_매출_금액"),
        ("06-11시", "시간대_06_11_매출_금액"),
        ("11-14시", "시간대_11_14_매출_금액"),
        ("14-17시", "시간대_14_17_매출_금액"),
        ("17-21시", "시간대_17_21_매출_금액"),
        ("21-24시", "시간대_21_24_매출_금액"),
    ]
    vals = [float(row.get(col, 0)) for _, col in time_sales_keys]
    s = sum(vals)
    labels = [lab for lab, _ in time_sales_keys]
    if s <= 0: return {lab: 0.0 for lab in labels}
    return {lab: (v / s * 100.0) for lab, v in zip(labels, vals)}

def get_time_recommendation(user_times: list, avg_time_sales_profile: dict) -> str | None:
    """사용자 선택 시간과 지역 평균 매출 시간을 비교하여 조언 생성"""
    if not user_times:
        return None

    # 사용자 선택 시간대를 라벨로 변환 ('00_06' -> '00-06시')
    label_map = { "00_06": "00-06시", "06_11": "06-11시", "11_14": "11-14시", "14_17": "14-17시", "17_21": "17-21시", "21_24": "21-24시" }
    user_time_labels = {label_map.get(t) for t in user_times}

    # 지역 평균 매출 상위 2개 시간대 찾기
    top_avg_times = sorted(avg_time_sales_profile, key=avg_time_sales_profile.get, reverse=True)[:2]

    # 사용자가 영업하지 않지만, 매출이 높은 시간대 찾기
    missed_opportunities = [time for time in top_avg_times if time not in user_time_labels]

    if missed_opportunities:
        return f"⏰ **영업 시간 조정 제안:** 이 상권은 **{', '.join(missed_opportunities)}**에 매출이 높습니다. 해당 시간대에 영업 확장을 검토해보세요."

    return f"⏰ **영업 시간 분석:** 현재 영업 시간이 상권의 피크 타임과 잘 일치합니다. 해당 시간대에 인력과 자원을 집중하는 전략이 유효합니다."

def get_main_customer_segment(row):
    """매출이 가장 높은 연령대와 성별을 분석합니다."""
    # 연령대별 매출 분석
    age_sales = {
        '10대': row.get('연령대_10_매출_금액', 0), '20대': row.get('연령대_20_매출_금액', 0),
        '30대': row.get('연령대_30_매출_금액', 0), '40대': row.get('연령대_40_매출_금액', 0),
        '50대': row.get('연령대_50_매출_금액', 0), '60대 이상': row.get('연령대_60_이상_매출_금액', 0)
    }
    main_age_group = max(age_sales, key=age_sales.get)

    # 성별 매출 분석
    gender_sales = {
        '남성': row.get('남성_매출_금액', 0),
        '여성': row.get('여성_매출_금액', 0)
    }
    main_gender = max(gender_sales, key=gender_sales.get)

    return main_age_group, main_gender

def calculate_statistics(row):
    if row is None:
        return None
    stats = {}
    total = float(row['당월_매출_금액'])
    if total > 0:
        stats['평균_매출'] = total
        stats['평균_매출건수'] = float(row.get('당월_매출_건수', 0))
        stats['주말_매출_비율'] = ((float(row.get('토요일_매출_금액', 0)) + float(row.get('일요일_매출_금액', 0))) / total * 100)
        stats['남성_매출_비율'] = float(row.get('남성_매출_금액', 0)) / total * 100
        stats['여성_매출_비율'] = float(row.get('여성_매출_금액', 0)) / total * 100
    else:
        stats['평균_매출'] = 0
        stats['평균_매출건수'] = 0
        stats['주말_매출_비율'] = 0
        stats['남성_매출_비율'] = 0
        stats['여성_매출_비율'] = 0
    return stats

# === 예비 창업자 분석 함수 이식 ===
def get_industry_comparison(df, industry, district):
    industry_data = df[df['서비스_업종_코드_명'] == industry]
    seoul_avg = industry_data.groupby('기준_년분기_코드').agg({
        '당월_매출_금액': 'mean',
        '폐업_률': 'mean',
        '점포_수': 'mean'
    }).iloc[-1]

    district_data = industry_data[industry_data['자치구_코드_명'] == district]
    district_avg = district_data.iloc[-1] if len(district_data) > 0 else None

    latest_quarter = industry_data['기준_년분기_코드'].max()
    latest_data = industry_data[industry_data['기준_년분기_코드'] == latest_quarter]

    sales_ranking = latest_data.sort_values('당월_매출_금액', ascending=False).reset_index(drop=True)
    sales_rank_df = sales_ranking[sales_ranking['자치구_코드_명'] == district]
    sales_rank = sales_rank_df.index[0] + 1 if len(sales_rank_df) > 0 else None

    closure_ranking = latest_data.sort_values('폐업_률', ascending=True).reset_index(drop=True)
    closure_rank_df = closure_ranking[closure_ranking['자치구_코드_명'] == district]
    closure_rank = closure_rank_df.index[0] + 1 if len(closure_rank_df) > 0 else None

    return {
        'seoul_avg_sales': seoul_avg['당월_매출_금액'],
        'seoul_avg_closure': seoul_avg['폐업_률'],
        'district_sales': district_avg['당월_매출_금액'] if district_avg is not None else 0,
        'district_closure': district_avg['폐업_률'] if district_avg is not None else 0,
        'sales_rank': sales_rank,
        'closure_rank': closure_rank,
        'total_districts': len(latest_data)
    }

def get_lower_rent_districts(df, industry, current_district):
    """
    선택한 업종에 대해 평균보다 임대료가 낮은 자치구 목록을 반환합니다.
    """
    # 최신 분기 데이터 필터링
    latest_quarter = df['기준_년분기_코드'].max()
    latest_df = df[df['기준_년분기_코드'] == latest_quarter]

    # 해당 업종 데이터 필터링
    industry_df = latest_df[latest_df['서비스_업종_코드_명'] == industry]

    if industry_df.empty:
        return []

    # 업종 평균 임대료 계산
    avg_rent = industry_df['전체임대료'].mean()

    # 평균보다 낮은 임대료를 가진 자치구 찾기 (현재 자치구 제외)
    lower_rent_df = industry_df[(industry_df['전체임대료'] < avg_rent) & (industry_df['자치구_코드_명'] != current_district)]

    # 임대료가 낮은 순으로 정렬하여 상위 3개 자치구 이름 반환
    return lower_rent_df.sort_values('전체임대료')['자치구_코드_명'].head(3).tolist()

def get_district_comparison(df, district):
    district_data = df[df['자치구_코드_명'] == district]
    latest_quarter = district_data['기준_년분기_코드'].max()
    latest_data = district_data[district_data['기준_년분기_코드'] == latest_quarter]
    top_sales = latest_data.nlargest(3, '당월_매출_금액')[['서비스_업종_코드_명', '당월_매출_금액', '폐업_률']]
    safe_industries = latest_data.nsmallest(3, '폐업_률')[['서비스_업종_코드_명', '당월_매출_금액', '폐업_률']]
    risky_industries = latest_data.nlargest(3, '폐업_률')[['서비스_업종_코드_명', '당월_매출_금액', '폐업_률']]
    return {'top_sales': top_sales, 'safe_industries': safe_industries, 'risky_industries': risky_industries}

def get_time_series_data(df, district, industry, year):
    filtered = df[(df['자치구_코드_명'] == district) & (df['서비스_업종_코드_명'] == industry)].sort_values('기준_년분기_코드')
    year_start = year * 10 + 1
    year_end = year * 10 + 4
    data_year = filtered[(filtered['기준_년분기_코드'] >= year_start) & (filtered['기준_년분기_코드'] <= year_end)]
    def format_quarter(code):
        code_str = str(int(code)); y = code_str[:4]; q = code_str[4]; return f"{y}-Q{q}"
    quarters_formatted = [format_quarter(q) for q in data_year['기준_년분기_코드'].tolist()]
    return {'quarters': quarters_formatted,
            'sales': data_year['당월_매출_금액'].tolist(),
            'closure_rate': data_year['폐업_률'].tolist(),
            'store_count': data_year['점포_수'].tolist()}

def get_available_years(df):
    quarters = df['기준_년분기_코드'].unique()
    years = sorted(set([int(str(int(q))[:4]) for q in quarters]))
    return years

def get_population_stats(row):
    total_flow = row['총_유동인구_수']
    age_distribution = {
        '10대': row['연령대_10_유동인구_수']/total_flow*100 if total_flow>0 else 0,
        '20대': row['연령대_20_유동인구_수']/total_flow*100 if total_flow>0 else 0,
        '30대': row['연령대_30_유동인구_수']/total_flow*100 if total_flow>0 else 0,
        '40대': row['연령대_40_유동인구_수']/total_flow*100 if total_flow>0 else 0,
        '50대': row['연령대_50_유동인구_수']/total_flow*100 if total_flow>0 else 0,
        '60대+': row['연령대_60_이상_유동인구_수']/total_flow*100 if total_flow>0 else 0
    }
    time_distribution = {
        '00-06시': row['시간대_00_06_유동인구_수'],
        '06-11시': row['시간대_06_11_유동인구_수'],
        '11-14시': row['시간대_11_14_유동인구_수'],
        '14-17시': row['시간대_14_17_유동인구_수'],
        '17-21시': row['시간대_17_21_유동인구_수'],
        '21-24시': row['시간대_21_24_유동인구_수']
    }
    population_ratio = {'유동인구': row['총_유동인구_수'], '상주인구': row['총_상주인구_수'], '직장인구': row['총_직장인구_수']}
    return {'age_distribution': age_distribution, 'time_distribution': time_distribution, 'population_ratio': population_ratio}

def get_seoul_population_avg(df):
    """서울시 전체 인구 평균 계산 (최신 분기 기준)"""
    latest_quarter = df['기준_년분기_코드'].max()
    latest_data = df[df['기준_년분기_코드'] == latest_quarter]

    return {
        'avg_flow': latest_data['총_유동인구_수'].mean(),
        'avg_resident': latest_data['총_상주인구_수'].mean(),
        'avg_work': latest_data['총_직장인구_수'].mean()
    }

def get_income_consumption_stats(df, district, row):
    district_q2_2025 = df[(df['자치구_코드_명'] == district) & (df['기준_년분기_코드'] == 20252)]
    if len(district_q2_2025) > 0:
        q2_row = district_q2_2025.iloc[0]
        total_spending = q2_row['지출_총_금액'] / 1000
    else:
        total_spending = row['지출_총_금액'] / 1000
        q2_row = row
    spending_breakdown = {
        '식료품': (q2_row['식료품_지출_총금액']/1000)/total_spending*100 if total_spending>0 else 0,
        '음식': (q2_row['음식_지출_총금액']/1000)/total_spending*100 if total_spending>0 else 0,
        '의류/신발': (q2_row['의류_신발_지출_총금액']/1000)/total_spending*100 if total_spending>0 else 0,
        '생활용품': (q2_row['생활용품_지출_총금액']/1000)/total_spending*100 if total_spending>0 else 0,
        '의료비': (q2_row['의료비_지출_총금액']/1000)/total_spending*100 if total_spending>0 else 0,
        '교통': (q2_row['교통_지출_총금액']/1000)/total_spending*100 if total_spending>0 else 0,
        '교육': (q2_row['교육_지출_총금액']/1000)/total_spending*100 if total_spending>0 else 0,
        '유흥': (q2_row['유흥_지출_총금액']/1000)/total_spending*100 if total_spending>0 else 0,
        '여가/문화': (q2_row['여가_문화_지출_총금액']/1000)/total_spending*100 if total_spending>0 else 0,
    }
    return {'avg_income': row['월_평균_소득_금액'], 'total_spending': total_spending, 'spending_breakdown': spending_breakdown}

# === [ADD] 요일/시간 프로파일 계산 유틸 ===
DAY_KR = ["월요일","화요일","수요일","목요일","금요일","토요일","일요일"]
TIME_KEYS = [
    ("00-06시", "시간대_00_06_유동인구_수"),
    ("06-11시", "시간대_06_11_유동인구_수"),
    ("11-14시", "시간대_11_14_유동인구_수"),
    ("14-17시", "시간대_14_17_유동인구_수"),
    ("17-21시", "시간대_17_21_유동인구_수"),
    ("21-24시", "시간대_21_24_유동인구_수"),
]

def build_avg_day_profile_from_row(row: pd.Series) -> dict:
    """row에서 요일별 매출 금액을 퍼센트(0~100)로 정규화"""
    vals = [float(row.get(f"{d}_매출_금액", 0)) for d in DAY_KR]
    s = sum(vals)
    if s <= 0:
        return {d: 0.0 for d in DAY_KR}
    return {d: (v / s * 100.0) for d, v in zip(DAY_KR, vals)}

def build_avg_time_profile_from_row(row: pd.Series) -> dict:
    """row에서 시간대별 유동인구 값을 퍼센트(0~100)로 정규화"""
    vals = [float(row.get(col, 0)) for _, col in TIME_KEYS]
    s = sum(vals)
    labels = [lab for lab, _ in TIME_KEYS]
    if s <= 0:
        return {lab: 0.0 for lab in labels}
    return {lab: (v / s * 100.0) for lab, v in zip(labels, vals)}

def build_user_day_profile(selected_days: list[str]) -> dict:
    """사용자 선택 요일을 균등 가중으로 100% 분배 (미선택이면 전부 0)"""
    n = len(selected_days)
    base = {d: 0.0 for d in DAY_KR}
    if n == 0:
        return base
    share = 100.0 / n
    for d in selected_days:
        if d in base:
            base[d] = share
    return base

def build_user_time_profile(selected_times: list[str]) -> dict:
    """
    사용자 선택 시간대를 균등 가중으로 100% 분배 (미선택이면 전부 0)
    selected_times는 ['00_06','06_11',...] 형태이므로 라벨로 변환해 비교
    """
    label_map = {
        "00_06": "00-06시", "06_11": "06-11시", "11_14": "11-14시",
        "14_17": "14-17시", "17_21": "17-21시", "21_24": "21-24시"
    }
    labels = [label_map.get(t, t) for t in selected_times]
    all_labels = [lab for lab, _ in TIME_KEYS]
    base = {lab: 0.0 for lab in all_labels}
    n = len(labels)
    if n == 0:
        return base
    share = 100.0 / n
    for lab in labels:
        if lab in base:
            base[lab] = share
    return base

# ==================== 입력 폼 ====================
st.markdown("### 매장 정보를 입력해주세요")
st.markdown("---")

merged_df = load_merged_data()
if merged_df is None:
    st.error("데이터를 불러올 수 없습니다. 관리자에게 문의하세요.")
    st.stop()

districts = sorted(merged_df['자치구_코드_명'].unique())
industries = sorted(merged_df['서비스_업종_코드_명'].unique())

col1, col2 = st.columns(2)
with col1:
    st.markdown("<div class='info-box'><h4>기본 정보</h4></div>", unsafe_allow_html=True)
    selected_district = st.selectbox("자치구 선택", options=districts)
    selected_industry = st.selectbox("서비스 업종 선택", options=industries)

with col2:
    st.markdown("<div class='info-box'><h4>재무 정보</h4></div>", unsafe_allow_html=True)
    input_sales = st.number_input("당월 매출 금액 (원)", min_value=0, max_value=1_000_000_000, value=0, step=10_000)
    rent = rent_per_area = st.number_input(
        "월 임대료 (원/3.3m²)",
        min_value=0,
        max_value=1000000,
        value=150000,
        step=10000,
        help="3.3m² 기준 월 임대료를 입력하세요"
    )

st.markdown("<br>", unsafe_allow_html=True)

# 영업 요일/시간 (입력만, 결과 미반영)
st.markdown("### 영업 요일/시간 (입력 전용)")
cl, cr = st.columns(2)
DAYS = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]
TIME_BUCKETS = ["00_06", "06_11", "11_14", "14_17", "17_21", "21_24"]
with cl:
    st.markdown("<div class='info-box'><h4>영업 요일 (다중 선택)</h4></div>", unsafe_allow_html=True)
    selected_days = st.multiselect("영업하는 요일을 선택하세요", DAYS)
with cr:
    st.markdown("<div class='info-box'><h4>영업 시간대 (다중 선택)</h4></div>", unsafe_allow_html=True)
    selected_times = st.multiselect("영업하는 시간대를 선택하세요", TIME_BUCKETS)

st.markdown("---")

# ==================== Session State ====================
if 'prediction_done' not in st.session_state:
    st.session_state.prediction_done = False

# risk_score는 예측 마지막에 생성되므로, 이게 없으면 비정상 상태로 간주하고 리셋
if 'risk_score' not in st.session_state:
    st.session_state.prediction_done = False

# ==================== 예측 버튼 ====================
if st.button("폐업 위험도 예측하기"):
    with st.spinner("AI가 데이터를 분석하고 있습니다..."):
        model, district_encoder, industry_encoder, sanggwon_encoder, feature_names = load_model_and_encoders()
        if model is None:
            st.error("모델을 불러올 수 없습니다. 관리자에게 문의하세요.")
            st.stop()

        row_data = get_district_industry_data(merged_df, selected_district, selected_industry)
        if row_data is None:
            st.error(f"{selected_district} - {selected_industry} 데이터가 없습니다. 다른 조합을 선택해주세요.")
            st.stop()

        stats = calculate_statistics(row_data)

        # 예측용 입력: 임대료만 반영 (매출/요일/시간 미반영 유지)
        input_data = row_data.copy()
        if '전체임대료' in input_data:
            input_data['전체임대료'] = rent
        elif '임대료' in input_data:
            input_data['임대료'] = rent

        # 인코딩
        try:
            input_data['자치구_코드_명'] = district_encoder.transform([selected_district])[0]
            input_data['서비스_업종_코드_명'] = industry_encoder.transform([selected_industry])[0]
            input_data['상권_변화_지표'] = sanggwon_encoder.transform([input_data['상권_변화_지표']])[0]
        except Exception as e:
            st.error(f"인코딩 오류: {str(e)}")
            st.stop()

        # 불필요 컬럼 제거(학습 제외 컬럼)
        drop_cols = ['기준_년분기_코드', '폐업_점포_수', '폐업_영업_개월_평균', '서울시_폐업_영업_개월_평균', '폐업_률']
        input_df = pd.DataFrame([input_data])
        X_input = input_df.drop(columns=[c for c in drop_cols if c in input_df.columns], errors='ignore')

        # 예측
        try:
            prediction_proba = model.predict_proba(X_input)[:, 1][0]
            risk_score = float(prediction_proba) * 100
        except Exception as e:
            st.error(f"예측 오류: {str(e)}")
            st.error(f"입력 피처 개수: {len(X_input.columns)}")
            st.stop()

        # SessionState 저장 (+ 이식된 비교/통계 전부)
        st.session_state.prediction_done = True
        st.session_state.selected_district = selected_district
        st.session_state.selected_industry = selected_industry
        st.session_state.rent = rent
        st.session_state.input_sales = input_sales   
        st.session_state.selected_days = selected_days  
        st.session_state.selected_times = selected_times 
        st.session_state.row_data = row_data
        st.session_state.stats = stats
        st.session_state.risk_score = risk_score
        st.session_state.industry_comp = get_industry_comparison(merged_df, selected_industry, selected_district)
        st.session_state.district_comp = get_district_comparison(merged_df, selected_district)

# ==================== 결과 표시 (예비 창업자 섹션 전부 이식) ====================
if st.session_state.prediction_done:
    selected_district = st.session_state.selected_district
    selected_industry = st.session_state.selected_industry
    rent = st.session_state.rent
    input_sales = st.session_state.input_sales
    selected_days = st.session_state.selected_days
    selected_times = st.session_state.selected_times
    row_data = st.session_state.row_data
    stats = st.session_state.stats
    risk_score = st.session_state.risk_score
    industry_comp = st.session_state.industry_comp
    district_comp = st.session_state.district_comp

    st.markdown("### 분석 결과")
    st.markdown("---")

    # 위험도 레벨 결정
    if risk_score >= 70:
        risk_level = "높음"
        risk_color = "#dc3545"
        message_class = "danger-box"
        message = "현재 입력하신 조건은 폐업 위험이 높은 편입니다. 신중한 검토가 필요합니다."
    elif risk_score >= 40:
        risk_level = "보통"
        risk_color = "#ffc107"
        message_class = "warning-box"
        message = "현재 입력하신 조건은 보통 수준의 위험도를 보이고 있습니다."
    else:
        risk_level = "낮음"
        risk_color = "#28a745"
        message_class = "success-box"
        message = "현재 입력하신 조건은 비교적 안정적인 편입니다."

    # 게이지 차트
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"폐업 위험도", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': risk_color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 40], 'color': '#d4edda'},
                {'range': [40, 70], 'color': '#fff3cd'},
                {'range': [70, 100], 'color': '#f8d7da'}
            ]
        }
    ))

    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': "darkblue", 'family': "Arial"}
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"""
    <div class="{message_class}">
        <h3>위험도: {risk_level} ({risk_score:.1f}점)</h3>
        <p style='margin:0; font-size: 1.05rem;'>{message}</p>
    </div>
    """, unsafe_allow_html=True)


    # 상세 카드
    st.markdown("### 상세 분석")
    
    # 임대료 부담률 (점포당 평균 매출 기준)
    # 지역 전체 매출을 점포 수로 나누어 점포당 평균 매출 계산
    total_sales = stats['평균_매출']  # 업종 전체 매출
    total_stores = row_data['점포_수']  # 해당 지역의 업종 점포 수
    sales_per_store = (total_sales / total_stores) if total_stores > 0 else 0
    rent_burden = (rent / sales_per_store * 100) if sales_per_store > 0 else 0
    

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">
                임대료 부담률
                <span class="tooltip-icon" data-tooltip="점포당 평균 매출 대비 임대료 비율 (적정: 10% 이하)">ℹ️</span>
            </div>
            <div class="stat-value">{rent_burden:.1f}%</div>
            <div style="color: {'#dc3545' if rent_burden > 15 else '#ffc107' if rent_burden > 10 else '#28a745'};">
                {'높음' if rent_burden > 15 else '주의' if rent_burden > 10 else '적정'}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">지역 평균 매출(단위: 원)</div>
            <div class="stat-value">{stats['평균_매출']/100000000:,.1f}억</div>
            <div style="color: #6c757d;">월 기준</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">지역 평균 건수(단위: 천)</div>
            <div class="stat-value">{stats['평균_매출건수']/1000:,.0f}건</div>
            <div style="color: #6c757d;">월 기준</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">폐업률</div>
            <div class="stat-value">{row_data['폐업_률']:.1f}%</div>
            <div style="color: #6c757d;">해당 지역</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 지역 통계
    st.markdown(f"### {selected_district} - {selected_industry} 통계")
    col_left, col_right = st.columns(2)
    with col_left:
        # 주중/주말 매출 비율
        fig_sales = go.Figure(data=[
            go.Bar(
                x=['주말 매출', '주중 매출'],
                y=[stats['주말_매출_비율'], 100 - stats['주말_매출_비율']],
                marker_color=['#F0067F', "#0976DD"],
                text=[f"{stats['주말_매출_비율']:.1f}%", f"{100-stats['주말_매출_비율']:.1f}%"],
                textposition='auto',
                textfont=dict(
                size=14,        # 폰트 크기
                color='white',  # 폰트 색상
                family='Arial'  # 폰트 종류
                )
            )
        ])
        fig_sales.update_layout(
            title="주중/주말 매출 비율",
            yaxis_title="비율 (%)",
            height=350,
            showlegend=False
        )
        st.plotly_chart(fig_sales, use_container_width=True)
    
    with col_right:
        # 성별 매출 비율
        fig_gender = go.Figure(data=[
            go.Pie(
                labels=['남성', '여성'],
                values=[stats['남성_매출_비율'], stats['여성_매출_비율']],
                marker_colors=['#00A0F3', '#12DAC2'],
                textinfo='label+percent',
                textfont=dict(
                size=12,        # 폰트 크기
                color='white',  # 폰트 색상
                family='Arial'  # 폰트 종류
                )
            )
        ])
        fig_gender.update_layout(
            title="성별 매출 비율",
            height=400
        )
        st.plotly_chart(fig_gender, use_container_width=True)

    st.markdown("---")

    # ==================== 업종별 비교 분석 ====================
    st.markdown(f"### 업종 비교 분석: {selected_industry}")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">서울시 평균 매출</div>
            <div class="stat-value">{industry_comp['seoul_avg_sales']/100000000:,.1f}억</div>
            <div style="color: #6c757d;">월 기준</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        sales_vs_seoul = (industry_comp['district_sales'] / industry_comp['seoul_avg_sales'] * 100) if industry_comp['seoul_avg_sales'] > 0 else 0
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">
                서울 평균 대비
                <span class="tooltip-icon" data-tooltip="선택 지역 매출이 서울시 평균의 몇 %인지 표시">ℹ️</span>
            </div>
            <div class="stat-value">{sales_vs_seoul:,.0f}%</div>
            <div style="color: {'#28a745' if sales_vs_seoul >= 100 else '#dc3545'};">
                {'평균 이상' if sales_vs_seoul >= 100 else '평균 이하'}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">
                매출 순위
                <span class="tooltip-icon" data-tooltip="해당 업종에서 25개 자치구 중 매출 순위">ℹ️</span>
            </div>
            <div class="stat-value">{industry_comp['sales_rank']}</div>
            <div style="color: #6c757d;">/ {industry_comp['total_districts']}개 자치구</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">
                안전도 순위
                <span class="tooltip-icon" data-tooltip="폐업률이 낮을수록 순위가 높음 (1위=가장 안전)">ℹ️</span>
            </div>
            <div class="stat-value">{industry_comp['closure_rank']}</div>
            <div style="color: #6c757d;">낮은 폐업률 기준</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ==================== 자치구별 비교 분석 ====================
    st.markdown(f"### 자치구 업종 분석: {selected_district}",
                help="2025년 2분기 기준")

    col1, col2, col3 = st.columns(3)

    rank_emojis = ['🥇', '🥈', '🥉']

    with col1:
        st.markdown("#### 매출 상위 3개 업종")
        for rank, (idx, row) in enumerate(district_comp['top_sales'].iterrows(), 1):
            st.markdown(f"""
            <div class="industry-card-top">
                <span class="rank-badge">{rank_emojis[rank-1]}</span>
                <div class="industry-name">{row['서비스_업종_코드_명']}</div>
                <div class="industry-stats">
                    매출: {row['당월_매출_금액']/100000000:,.1f}억원 | 폐업률: {row['폐업_률']:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("#### 안전한 업종 3개(폐업률 기준)")
        for rank, (idx, row) in enumerate(district_comp['safe_industries'].iterrows(), 1):
            st.markdown(f"""
            <div class="industry-card-safe">
                <span class="rank-badge">{rank_emojis[rank-1]}</span>
                <div class="industry-name">{row['서비스_업종_코드_명']}</div>
                <div class="industry-stats">
                    폐업률: {row['폐업_률']:.1f}% | 매출: {row['당월_매출_금액']/100000000:,.1f}억원
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col3:
        st.markdown("#### 위험한 업종 3개(폐업률 기준)")
        for rank, (idx, row) in enumerate(district_comp['risky_industries'].iterrows(), 1):
            st.markdown(f"""
            <div class="industry-card-risky">
                <span class="rank-badge">{rank_emojis[rank-1]}</span>
                <div class="industry-name">{row['서비스_업종_코드_명']}</div>
                <div class="industry-stats">
                    폐업률: {row['폐업_률']:.1f}% | 매출: {row['당월_매출_금액']/100000000:,.1f}억원
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ==================== 시계열 트렌드 ====================
    st.markdown("### 분기별 트렌드")

    # 연도 선택
    available_years = get_available_years(merged_df)
    col_year, col_spacer = st.columns([1, 3])
    with col_year:
        selected_year = st.selectbox(
            "연도 선택",
            options=available_years,
            index=len(available_years)-2 if len(available_years) >= 2 else 0,
            key="trend_year"
        )

    time_series = get_time_series_data(merged_df, selected_district, selected_industry, selected_year)

    if len(time_series['quarters']) > 0:
        col1, col2 = st.columns(2)

        with col1:
            # 매출 추이
            fig_sales_trend = go.Figure()
            fig_sales_trend.add_trace(go.Scatter(
                x=time_series['quarters'],
                y=[s/100000000 for s in time_series['sales']],
                mode='lines+markers',
                name='매출',
                line=dict(color='#667eea', width=3),
                marker=dict(size=10)
            ))
            fig_sales_trend.update_layout(
                title=f"{selected_year}년 분기별 매출 추이",
                xaxis_title="분기",
                yaxis_title="매출 (억원)",
                height=300
            )
            st.plotly_chart(fig_sales_trend, use_container_width=True)

        with col2:
            # 폐업률 추이
            fig_closure_trend = go.Figure()
            fig_closure_trend.add_trace(go.Scatter(
                x=time_series['quarters'],
                y=time_series['closure_rate'],
                mode='lines+markers',
                name='폐업률',
                line=dict(color='#dc3545', width=3),
                marker=dict(size=10)
            ))
            fig_closure_trend.update_layout(
                title=f"{selected_year}년 분기별 폐업률 변화",
                xaxis_title="분기",
                yaxis_title="폐업률 (%)",
                height=300
            )
            st.plotly_chart(fig_closure_trend, use_container_width=True)

        # 점포 수 추이
        fig_store_trend = go.Figure()
        fig_store_trend.add_trace(go.Bar(
            x=time_series['quarters'],
            y=time_series['store_count'],
            marker_color='#0976DD',
            text=[f"{int(v):,}" for v in time_series['store_count']],
            textposition='inside',
            textfont=dict(
            size=14,        # 폰트 크기
            color='white',  # 폰트 색상
            family='family=Arial Black, sans-serif'  # 폰트 종류
            )
        ))
        fig_store_trend.update_layout(
            title=f"{selected_year}년 분기별 점포 수 변화",
            xaxis_title="분기",
            yaxis_title="점포 수",
            height=400
        )
        st.plotly_chart(fig_store_trend, use_container_width=True)
    else:
        st.warning("시계열 데이터가 부족합니다.")

    st.markdown("---")
    

    # =================== 매출 비교 (내 입력 vs 자치구 점포당 평균 | 천만원 단위 고정) ===================
    st.markdown("### 매출 비교 (내 입력 vs 자치구 점포당 평균)")

    # 자치구+업종 시계열 (최신 → 과거)
    slice_di = merged_df[
        (merged_df["자치구_코드_명"] == selected_district) &
        (merged_df["서비스_업종_코드_명"] == selected_industry)
    ].sort_values("기준_년분기_코드", ascending=False)

    if len(slice_di) == 0:
        st.warning("선택한 자치구·업종에 대한 데이터가 없습니다.")
    else:
        # ▷ 최근 4개 분기만 사용 (평균)
        use_df = slice_di.head(4)

        # 자치구 내 점포당 평균 매출(원) = 분기별 합산 매출 / 합산 점포수
        total_sales = float(use_df["당월_매출_금액"].sum())
        total_stores = float(use_df["점포_수"].sum())
        avg_sales = total_sales / max(total_stores, 1.0)

        # 내 입력 매출(원)
        user_sales = float(input_sales)

        # ▶ 단위: 천만원 고정
        SCALE = 10_000_000
        def to_cheonman(x): return x / SCALE

        y_labels = ["자치구 점포당 평균", "내 당월 매출(입력)"]
        x_values = [to_cheonman(avg_sales), to_cheonman(user_sales)]

        fig_sales_compare = go.Figure()
        fig_sales_compare.add_trace(go.Bar(
            y=y_labels,
            x=x_values,
            orientation="h",
            marker_color=["#a5b4fc", "#667eea"],  # 평균(연파랑) / 내 매출(진파랑)
            text=[f"{to_cheonman(avg_sales):,.1f}천만", f"{to_cheonman(user_sales):,.1f}천만"],
            textposition="auto",
            hovertemplate="%{y}<br>매출: %{x:.1f}천만<extra></extra>",
            showlegend=False
        ))

        subtitle = f"{selected_district} · {selected_industry} / 최근 4개 분기 평균 기준"
        fig_sales_compare.update_layout(
            title={"text": subtitle, "x": 0.01, "xanchor": "left"},
            xaxis_title="매출 (천만원)",
            yaxis_title="",
            height=260,
            margin=dict(l=10, r=10, t=50, b=20)
        )
        st.plotly_chart(fig_sales_compare, use_container_width=True)

        if user_sales == 0:
            st.caption("※ 현재 입력한 당월 매출이 0원입니다. 값을 입력하면 비교가 더 명확해집니다.")

    st.markdown("---")

    # ==================== 인구통계 분석 ====================
    st.markdown("### 인구 통계")

    pop_stats = get_population_stats(row_data)

    col1, col2 = st.columns(2)

    with col1:
        # 연령대별 유동인구
        fig_age = go.Figure(data=[
            go.Bar(
                x=list(pop_stats['age_distribution'].keys()),
                y=list(pop_stats['age_distribution'].values()),
                marker_color='#1e40af',
                text=[f"{v:.1f}%" for v in pop_stats['age_distribution'].values()],
                textposition='auto'
            )
        ])
        fig_age.update_layout(
            title="연령대별 유동인구 비율",
            xaxis_title="연령대",
            yaxis_title="비율 (%)",
            height=300
        )
        st.plotly_chart(fig_age, use_container_width=True)

    with col2:
        # 시간대별 유동인구
        fig_time = go.Figure(data=[
            go.Bar(
                x=list(pop_stats['time_distribution'].keys()),
                y=list(pop_stats['time_distribution'].values()),
                marker_color="#1d2f81",
                text=[f"{v:,.0f}" for v in pop_stats['time_distribution'].values()],
                textposition='auto'
            )
        ])
        fig_time.update_layout(
            title="시간대별 유동인구",
            xaxis_title="시간대",
            yaxis_title="유동인구 수",
            height=300
        )
        st.plotly_chart(fig_time, use_container_width=True)

    # 인구 구성 - 서울시 평균 대비
    st.markdown("### 인구 구성")

    # 서울시 평균 계산
    seoul_avg = get_seoul_population_avg(merged_df)

    # 현재 지역 인구
    flow_pop = pop_stats['population_ratio']['유동인구']
    resident_pop = pop_stats['population_ratio']['상주인구']
    work_pop = pop_stats['population_ratio']['직장인구']

    # 서울시 평균 대비 비율 계산
    flow_vs_seoul = (flow_pop / seoul_avg['avg_flow'] * 100) if seoul_avg['avg_flow'] > 0 else 0
    resident_vs_seoul = (resident_pop / seoul_avg['avg_resident'] * 100) if seoul_avg['avg_resident'] > 0 else 0
    work_vs_seoul = (work_pop / seoul_avg['avg_work'] * 100) if seoul_avg['avg_work'] > 0 else 0

    # 총 유동인구 큰 카드
    st.markdown(f"""
    <div class="stat-card" style="background: #667eea; color: white; margin-bottom: 1.5rem;">
        <div style="font-size: 1.2rem; margin-bottom: 0.5rem;">총 유동인구</div>
        <div style="font-size: 3rem; font-weight: bold; margin: 1rem 0;">{flow_pop:,.0f}명</div>
        <div style="font-size: 1.1rem;">
            서울시 평균 대비: <strong>{flow_vs_seoul:.0f}%</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 상주인구 / 직장인구 카드
    col_pop1, col_pop2 = st.columns(2)

    with col_pop1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">상주인구</div>
            <div class="stat-value">{resident_pop:,.0f}명</div>
            <div style="font-size: 0.95rem; color: {'#28a745' if resident_vs_seoul >= 100 else '#dc3545'}; margin-top: 0.5rem;">
                서울시 평균 대비 {resident_vs_seoul:.0f}%
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_pop2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">직장인구</div>
            <div class="stat-value">{work_pop:,.0f}명</div>
            <div style="font-size: 0.95rem; color: {'#28a745' if work_vs_seoul >= 100 else '#dc3545'}; margin-top: 0.5rem;">
                서울시 평균 대비 {work_vs_seoul:.0f}%
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ==================== 소득/소비 분석 ====================
    st.markdown("### 소득 및 소비 분석")

    income_stats = get_income_consumption_stats(merged_df, selected_district, row_data)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">
                평균 소득
                <span class="tooltip-icon" data-tooltip="개인 월 평균 소득 금액">ℹ️</span>
            </div>
            <div class="stat-value">{income_stats['avg_income']/10000:,.0f}만</div>
            <div style="color: #6c757d;">개인 월 기준</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">
                지역 총 지출
                <span class="tooltip-icon" data-tooltip="2025년 2분기 기준 지역 전체 총 지출 금액">ℹ️</span>
            </div>
            <div class="stat-value">{income_stats['total_spending']/100000000:,.0f}억</div>
            <div style="color: #6c757d;">2025년 2분기 기준</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # 지출 항목별 비율
        fig_spending = go.Figure(data=[
            go.Bar(
                y=list(income_stats['spending_breakdown'].keys()),
                x=list(income_stats['spending_breakdown'].values()),
                orientation='h',
                marker_color='#667eea',
                text=[f"{v:.1f}%" for v in income_stats['spending_breakdown'].values()],
                textposition='auto'
            )
        ])
        fig_spending.update_layout(
            title="지출 항목별 비율",
            xaxis_title="비율 (%)",
            yaxis_title="항목",
            height=400
        )
        st.plotly_chart(fig_spending, use_container_width=True)

    st.markdown("---")

    # =================== 영업 요일/시간 비교 ===================
    st.markdown("### 영업 요일/시간 비교")

    # 지역 평균(=선택한 자치구+업종의 대표 행) 프로파일
    avg_day_profile  = build_avg_day_profile_from_row(row_data)      # 요일: 매출 비중 기반(%)
    avg_time_profile = build_avg_time_profile_from_row(row_data)     # 시간: 유동 비중 기반(%)

    # 선택 항목 집합
    days = DAY_KR  # ["월요일",..., "일요일"]
    selected_day_set = set(selected_days or [])

    label_map = {
        "00_06": "00-06시", "06_11": "06-11시", "11_14": "11-14시",
        "14_17": "14-17시", "17_21": "17-21시", "21_24": "21-24시"
    }
    time_labels = [lab for lab, _ in TIME_KEYS]
    selected_time_set = set(label_map.get(t, t) for t in (selected_times or []))

    # 색상(선택=진한색, 미선택=연한색)
    COLOR_SELECTED = "#3b82f6"   # 진한 파랑
    COLOR_DEFAULT  = "#93c5fd"   # 연한 파랑

    # ── 요일 비교: 하나의 trace에 막대 색만 다르게 ───────────────────────────────
    y_day = [avg_day_profile[d] for d in days]
    colors_day = [COLOR_SELECTED if d in selected_day_set else COLOR_DEFAULT for d in days]

    fig_days = go.Figure()
    fig_days.add_trace(go.Bar(
        x=days, y=y_day,
        marker_color=colors_day,
        text=[f"{v:.1f}%" for v in y_day],
        textposition="auto",
        hovertemplate="요일: %{x}<br>지역 평균: %{y:.1f}%<extra></extra>",
        name="지역 평균(매출 비중)",
        showlegend=False
    ))
    # 범례용 더미(선택/기본 색 설명)
    fig_days.add_trace(go.Bar(x=[None], y=[None], marker_color=COLOR_SELECTED, name="내가 선택한 요일"))
    fig_days.add_trace(go.Bar(x=[None], y=[None], marker_color=COLOR_DEFAULT,  name="선택하지 않은 요일"))

    fig_days.update_layout(
        title="요일별 비교",
        yaxis_title="비율 (%)",
        barmode="overlay",
        height=360,
        showlegend=True
    )
    st.plotly_chart(fig_days, use_container_width=True)

    # ── 시간대 비교: 하나의 trace에 막대 색만 다르게 ────────────────────────────
    y_time = [avg_time_profile[lab] for lab in time_labels]
    colors_time = [COLOR_SELECTED if lab in selected_time_set else COLOR_DEFAULT for lab in time_labels]

    fig_timecmp = go.Figure()
    fig_timecmp.add_trace(go.Bar(
        x=time_labels, y=y_time,
        marker_color=colors_time,
        text=[f"{v:.1f}%" for v in y_time],
        textposition="auto",
        hovertemplate="시간대: %{x}<br>지역 평균: %{y:.1f}%<extra></extra>",
        name="지역 평균(유동 비중)",
        showlegend=False
    ))
    # 범례용 더미
    fig_timecmp.add_trace(go.Bar(x=[None], y=[None], marker_color=COLOR_SELECTED, name="내가 선택한 시간대"))
    fig_timecmp.add_trace(go.Bar(x=[None], y=[None], marker_color=COLOR_DEFAULT,  name="선택하지 않은 시간대"))

    fig_timecmp.update_layout(
        title="시간대별 비교",
        yaxis_title="비율 (%)",
        barmode="overlay",
        height=360,
        showlegend=True
    )
    st.plotly_chart(fig_timecmp, use_container_width=True)

    st.markdown("---")

    # Session State를 이용한 솔루션 표시/숨기기
    if 'show_solution' not in st.session_state:
        st.session_state.show_solution = False

    if st.button("솔루션 받기", type="primary"):
        st.session_state.show_solution = True

    if st.session_state.show_solution:
        with st.container():
            st.markdown("### 💡 AI 기반 맞춤 솔루션")
            st.markdown("---")

            recommendations = []

            if rent_burden > 10:
                if rent_burden > 15:
                    recommendations.append("🔴 **임대료 부담률이 매우 높습니다.** 매출 증대 방안을 적극적으로 모색하거나, 임대료 재협상을 고려해보세요.")
                else:
                    recommendations.append("🟡 **임대료 부담률이 다소 높은 편입니다.** 매출 증대 또는 비용 절감 방안을 준비하는 것이 좋습니다.")
                
                alt_districts = get_lower_rent_districts(merged_df, selected_industry, selected_district)
                if alt_districts:
                    recommendations.append(f"💡 **대안 지역 추천:** 동일 업종의 평균 임대료가 더 낮은 **{', '.join(alt_districts)}** 지역으로의 이전을 장기적으로 고려해볼 수 있습니다.")
            else:
                recommendations.append("🟢 **임대료 부담률이 적정 수준입니다.**")

            if risk_score >= 70:
                recommendations.append("🔴 **폐업 위험도가 높습니다.** 사업 전략의 근본적인 재검토가 필요합니다.")
                safe_industries_in_district = district_comp.get('safe_industries', pd.DataFrame())
                if not safe_industries_in_district.empty:
                    alt_industries = safe_industries_in_district['서비스_업종_코드_명'].head(3).tolist()
                    if alt_industries:
                        recommendations.append(f"💡 **업종 전환 고려:** 현재 지역에서는 **{', '.join(alt_industries)}** 업종이 비교적 안정적입니다. 업종 변경 또는 아이템 추가를 고려해보세요.")
            elif risk_score >= 40:
                recommendations.append("🟡 **폐업 위험도가 보통입니다.** 차별화된 경쟁력 확보가 시급합니다.")
            else:
                recommendations.append("🟢 **폐업 위험도가 안정적입니다.** 현재 전략을 유지하며 고객 만족도를 높이는 데 집중하세요.")

            # 사용자 입력 영업일/시간 기반 추천
            if selected_days or selected_times:
                avg_day_profile = build_avg_day_profile_from_row(row_data)
                day_rec = get_day_recommendation(selected_days, avg_day_profile)
                if day_rec:
                    recommendations.append(day_rec)

                avg_time_sales_profile = build_avg_time_sales_profile_from_row(row_data)
                time_rec = get_time_recommendation(selected_times, avg_time_sales_profile)
                if time_rec:
                    recommendations.append(time_rec)

            if row_data.get('폐업_률', 0) > 5:
                recommendations.append("⚠️ **높은 경쟁 환경:** 이 지역은 경쟁이 치열하고 폐업률이 높습니다. 단골 고객 확보가 생존의 열쇠입니다.")
                main_age, main_gender = get_main_customer_segment(row_data)
                recommendations.append(f"🎯 **타겟 고객 집중:** 주 고객층인 **{main_age} {main_gender}**을 대상으로 한 멤버십, 맞춤형 이벤트 등을 통해 충성도를 높이세요.")

            recommendations.append("🤝 정기적인 전문가 컨설팅을 통해 사업 방향을 점검하고 새로운 기회를 모색하는 것을 권장합니다.")

            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"{i}. {rec}")

            if st.button("닫기"):
                st.session_state.show_solution = False
                st.rerun()

# 하단 안내/푸터
st.markdown("---")
st.markdown("""
<div class="info-box">
    <h4>ℹ️ 안내사항</h4>
    <ul style='line-height: 2;'>
        <li>본 서비스는 AI 기반 예측 결과로, 참고용으로만 활용하시기 바랍니다.</li>
        <li>실제 운영/확장 결정 시에는 전문가 상담과 충분한 시장조사가 필요합니다.</li>
        <li>예측 결과는 과거 데이터를 기반으로 계산되며, 실제 결과와 다를 수 있습니다.</li>
        <li>시장 상황, 경쟁 환경 등 다양한 외부 요인도 함께 고려하셔야 합니다.</li>
    </ul>
</div>
""", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; padding: 2rem; color: #6c757d; font-size: 0.9rem;'>
    <p>© 2025 서울시 자치구별 매장 폐업 예측 프로젝트. All rights reserved.</p>
    <p>매장 운영자를 위한 AI 기반 의사결정 지원 서비스</p>
</div>
""", unsafe_allow_html=True)
