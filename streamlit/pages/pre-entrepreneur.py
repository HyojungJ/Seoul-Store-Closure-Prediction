import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import sys
from pathlib import Path

# 대시보드 공용 예측 서비스 (streamlit/model_service.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_service import DATA_PATH, load_prediction_service, predict_closure_risk  # noqa: E402

# 페이지 설정
st.set_page_config(
    page_title="예비 창업자를 위한 폐업 예측",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 커스텀 CSS
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
    <h1 style='margin:0; font-size: 2.5rem;'>예비 창업자를 위한 창업 가이드</h1>
    <p style='margin-top: 1rem; font-size: 1.2rem; opacity: 0.9;'>
        AI 기반 데이터 분석으로 창업 전 폐업 위험도를 미리 확인하세요
    </p>
</div>
""", unsafe_allow_html=True)


# ==================== 데이터 로드 함수 ====================

@st.cache_data
def load_merged_data():
    """merged_data.csv 로드"""
    try:
        df = pd.read_csv(DATA_PATH)
        return df
    except Exception as e:
        st.error(f"데이터 로드 실패: {str(e)}")
        return None

def get_district_industry_data(df, district, industry):
    """특정 자치구+업종의 최신 데이터 가져오기"""
    filtered = df[
        (df['자치구_코드_명'] == district) &
        (df['서비스_업종_코드_명'] == industry)
    ]

    if len(filtered) == 0:
        return None

    # 최신 데이터 반환
    latest = filtered.sort_values('기준_년분기_코드', ascending=False).iloc[0]
    return latest

def calculate_statistics(row):
    """통계 계산"""
    if row is None:
        return None

    stats = {}

    # 매출 관련
    if row['당월_매출_금액'] > 0:
        stats['평균_매출'] = row['당월_매출_금액']
        stats['평균_매출건수'] = row['당월_매출_건수']
        stats['주말_매출_비율'] = ((row['토요일_매출_금액'] + row['일요일_매출_금액']) / row['당월_매출_금액'] * 100)
        stats['남성_매출_비율'] = (row['남성_매출_금액'] / row['당월_매출_금액'] * 100)
        stats['여성_매출_비율'] = (row['여성_매출_금액'] / row['당월_매출_금액'] * 100)
    else:
        stats['평균_매출'] = 0
        stats['평균_매출건수'] = 0
        stats['주말_매출_비율'] = 0
        stats['남성_매출_비율'] = 0
        stats['여성_매출_비율'] = 0

    return stats

# ==================== 추가 통계 분석 함수 ====================

def get_industry_comparison(df, industry, district):
    """업종별 비교 통계"""
    industry_data = df[df['서비스_업종_코드_명'] == industry]

    # 서울시 전체 평균
    seoul_avg = industry_data.groupby('기준_년분기_코드').agg({
        '당월_매출_금액': 'mean',
        '폐업_률': 'mean',
        '점포_수': 'mean'
    }).iloc[-1]

    # 선택 자치구 데이터
    district_data = industry_data[industry_data['자치구_코드_명'] == district]
    if len(district_data) > 0:
        district_avg = district_data.iloc[-1]
    else:
        district_avg = None

    # 자치구별 순위
    latest_quarter = industry_data['기준_년분기_코드'].max()
    latest_data = industry_data[industry_data['기준_년분기_코드'] == latest_quarter]

    # 매출 순위 계산 (높은 매출 = 1등)
    sales_ranking = latest_data.sort_values('당월_매출_금액', ascending=False).reset_index(drop=True)
    sales_rank_df = sales_ranking[sales_ranking['자치구_코드_명'] == district]
    sales_rank = sales_rank_df.index[0] + 1 if len(sales_rank_df) > 0 else None

    # 안전도 순위 계산 (낮은 폐업률 = 1등)
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


def get_district_comparison(df, district):
    """자치구별 비교 통계 - 해당 자치구에서 잘되는/위험한 업종"""
    district_data = df[df['자치구_코드_명'] == district]
    latest_quarter = district_data['기준_년분기_코드'].max()
    latest_data = district_data[district_data['기준_년분기_코드'] == latest_quarter]

    # 매출 기준 상위 3개 업종
    top_sales = latest_data.nlargest(3, '당월_매출_금액')[['서비스_업종_코드_명', '당월_매출_금액', '폐업_률']]

    # 폐업률 기준 하위 3개 업종 (안전한 업종)
    safe_industries = latest_data.nsmallest(3, '폐업_률')[['서비스_업종_코드_명', '당월_매출_금액', '폐업_률']]

    # 폐업률 기준 상위 3개 업종 (위험한 업종)
    risky_industries = latest_data.nlargest(3, '폐업_률')[['서비스_업종_코드_명', '당월_매출_금액', '폐업_률']]

    return {
        'top_sales': top_sales,
        'safe_industries': safe_industries,
        'risky_industries': risky_industries
    }

def get_time_series_data(df, district, industry, year):
    """시계열 트렌드 데이터 - 선택한 연도의 1~4분기"""
    filtered = df[
        (df['자치구_코드_명'] == district) &
        (df['서비스_업종_코드_명'] == industry)
    ].sort_values('기준_년분기_코드')

    # 선택한 연도 데이터 필터링
    year_start = year * 10 + 1  # 예: 2024 -> 20241
    year_end = year * 10 + 4    # 예: 2024 -> 20244

    data_year = filtered[
        (filtered['기준_년분기_코드'] >= year_start) &
        (filtered['기준_년분기_코드'] <= year_end)
    ]

    # 분기 코드를 읽기 쉬운 형태로 변환 (예: 20241 -> 2024-Q1)
    def format_quarter(code):
        code_str = str(int(code))
        year = code_str[:4]
        quarter = code_str[4]
        return f"{year}/{quarter}"

    quarters_formatted = [format_quarter(q) for q in data_year['기준_년분기_코드'].tolist()]

    return {
        'quarters': quarters_formatted,
        'sales': data_year['당월_매출_금액'].tolist(),
        'closure_rate': data_year['폐업_률'].tolist(),
        'store_count': data_year['점포_수'].tolist()
    }

def get_available_years(df):
    """데이터에서 사용 가능한 연도 목록 추출"""
    quarters = df['기준_년분기_코드'].unique()
    years = sorted(set([int(str(int(q))[:4]) for q in quarters]))
    return years

def get_population_stats(row):
    """인구통계 상세 분석"""
    total_flow = row['총_유동인구_수']

    # 연령대별 유동인구 비율
    age_distribution = {
        '10대': row['연령대_10_유동인구_수'] / total_flow * 100 if total_flow > 0 else 0,
        '20대': row['연령대_20_유동인구_수'] / total_flow * 100 if total_flow > 0 else 0,
        '30대': row['연령대_30_유동인구_수'] / total_flow * 100 if total_flow > 0 else 0,
        '40대': row['연령대_40_유동인구_수'] / total_flow * 100 if total_flow > 0 else 0,
        '50대': row['연령대_50_유동인구_수'] / total_flow * 100 if total_flow > 0 else 0,
        '60대+': row['연령대_60_이상_유동인구_수'] / total_flow * 100 if total_flow > 0 else 0
    }

    # 시간대별 유동인구
    time_distribution = {
        '00-06시': row['시간대_00_06_유동인구_수'],
        '06-11시': row['시간대_06_11_유동인구_수'],
        '11-14시': row['시간대_11_14_유동인구_수'],
        '14-17시': row['시간대_14_17_유동인구_수'],
        '17-21시': row['시간대_17_21_유동인구_수'],
        '21-24시': row['시간대_21_24_유동인구_수']
    }

    # 인구 구성
    population_ratio = {
        '유동인구': row['총_유동인구_수'],
        '상주인구': row['총_상주인구_수'],
        '직장인구': row['총_직장인구_수']
    }

    return {
        'age_distribution': age_distribution,
        'time_distribution': time_distribution,
        'population_ratio': population_ratio
    }

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
    """소득/소비 분석 - 2025년 2분기 데이터"""
    # 2025년 2분기 기준 시각화 - 가장 최신 데이터
    # 해당 자치구의 2025년 2분기(20252) 데이터 가져오기
    district_q2_2025 = df[
        (df['자치구_코드_명'] == district) &
        (df['기준_년분기_코드'] == 20252)
    ]

    if len(district_q2_2025) > 0:
        q2_row = district_q2_2025.iloc[0]
        # 지출 데이터는 원본에 비해 0이 3개 추가로 붙어있어서 1000으로 나눔
        total_spending = q2_row['지출_총_금액'] / 1000
    else:
        # 2025년 2분기 데이터가 없으면 현재 row 사용
        total_spending = row['지출_총_금액'] / 1000
        q2_row = row

    spending_breakdown = {
        '식료품': (q2_row['식료품_지출_총금액'] / 1000) / total_spending * 100 if total_spending > 0 else 0,
        '음식': (q2_row['음식_지출_총금액'] / 1000) / total_spending * 100 if total_spending > 0 else 0,
        '의류/신발': (q2_row['의류_신발_지출_총금액'] / 1000) / total_spending * 100 if total_spending > 0 else 0,
        '생활용품': (q2_row['생활용품_지출_총금액'] / 1000) / total_spending * 100 if total_spending > 0 else 0,
        '의료비': (q2_row['의료비_지출_총금액'] / 1000) / total_spending * 100 if total_spending > 0 else 0,
        '교통': (q2_row['교통_지출_총금액'] / 1000) / total_spending * 100 if total_spending > 0 else 0,
        '교육': (q2_row['교육_지출_총금액'] / 1000) / total_spending * 100 if total_spending > 0 else 0,
        '유흥': (q2_row['유흥_지출_총금액'] / 1000) / total_spending * 100 if total_spending > 0 else 0,
        '여가/문화': (q2_row['여가_문화_지출_총금액'] / 1000) / total_spending * 100 if total_spending > 0 else 0,
    }

    return {
        'avg_income': row['월_평균_소득_금액'],
        'total_spending': total_spending,
        'spending_breakdown': spending_breakdown
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

# ==================== 메인 화면 ====================

st.markdown("### 창업 정보를 입력해주세요")
st.markdown("---")

# 데이터 로드
merged_df = load_merged_data()

if merged_df is None:
    st.error("데이터를 불러올 수 없습니다. 관리자에게 문의하세요.")
    st.stop()

# 자치구/업종 리스트 추출
districts = sorted(merged_df['자치구_코드_명'].unique())
industries = sorted(merged_df['서비스_업종_코드_명'].unique())

# 입력 폼
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="info-box">
        <h4>기본 정보</h4>
    </div>
    """, unsafe_allow_html=True)

    selected_district = st.selectbox(
        "자치구 선택",
        options=districts,
        help="창업을 계획하고 계신 자치구를 선택하세요"
    )

    selected_industry = st.selectbox(
        "서비스 업종 선택",
        options=industries,
        help="창업을 계획하고 계신 업종을 선택하세요"
    )

with col2:
    st.markdown("""
    <div class="info-box">
        <h4>재무 정보</h4>
    </div>
    """, unsafe_allow_html=True)

    rent_per_area = st.number_input(
        "월 임대료 (원/3.3m²)",
        min_value=0,
        max_value=1000000,
        value=150000,
        step=10000,
        help="3.3m² 기준 월 임대료를 입력하세요"
    )

    store_area = st.number_input(
        "매장 면적 (평)",
        min_value=1,
        max_value=500,
        value=10,
        step=1,
        help="예상하는 매장 면적(평)을 입력하세요"
    )

    # 총 임대료 계산 (1평 = 3.3m²)
    rent = rent_per_area * store_area

st.markdown("---")

# Session State 초기화
if 'prediction_done' not in st.session_state:
    st.session_state.prediction_done = False

# risk_score는 예측 마지막에 생성되므로, 이게 없으면 비정상 상태로 간주하고 리셋
if 'risk_score' not in st.session_state:
    st.session_state.prediction_done = False

# 예측 버튼
if st.button("폐업 위험도 예측하기"):
    with st.spinner("AI가 데이터를 분석하고 있습니다..."):

        # 모델 및 예측용 피처 로드
        service = load_prediction_service()

        if service is None:
            st.error("모델을 불러올 수 없습니다. 관리자에게 문의하세요.")
            st.stop()

        # 해당 자치구+업종 데이터 가져오기
        row_data = get_district_industry_data(merged_df, selected_district, selected_industry)

        if row_data is None:
            st.error(f"{selected_district} - {selected_industry}에 대한 데이터가 없습니다. 다른 조합을 선택해주세요.")
            st.stop()

        # 통계 계산
        stats = calculate_statistics(row_data)

        # 예측: 마지막 관측 분기(t-1)·직전 분기(t-2) 데이터로 다음 분기 폐업 위험 예측
        #       입력 임대료(3.3m² 기준)는 최근 임대료로 반영
        try:
            prediction_proba, predicted_quarter = predict_closure_risk(
                service, selected_district, selected_industry, rent_per_area
            )
            risk_score = prediction_proba * 100
        except Exception as e:
            st.error(f"예측 오류: {str(e)}")
            st.stop()

        # Session State에 결과 저장
        st.session_state.prediction_done = True
        st.session_state.selected_district = selected_district
        st.session_state.selected_industry = selected_industry
        st.session_state.rent = rent
        st.session_state.row_data = row_data
        st.session_state.stats = stats
        st.session_state.risk_score = risk_score
        st.session_state.industry_comp = get_industry_comparison(merged_df, selected_industry, selected_district)
        st.session_state.district_comp = get_district_comparison(merged_df, selected_district)

# ==================== 결과 표시 ====================
if st.session_state.prediction_done:
    # Session State에서 값 가져오기
    selected_district = st.session_state.selected_district
    selected_industry = st.session_state.selected_industry
    rent = st.session_state.rent
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

    # 위험도 메시지
    st.markdown(f"""
    <div class="{message_class}">
        <h3>위험도: {risk_level} ({risk_score:.1f}점)</h3>
        <p style='margin:0; font-size: 1.1rem;'>{message}</p>
    </div>
    """, unsafe_allow_html=True)

    # 상세 분석
    st.markdown("### 상세 분석")

    col1, col2, col3, col4 = st.columns(4)

    # 임대료 부담률 (점포당 평균 매출 기준)
    # 지역 전체 매출을 점포 수로 나누어 점포당 평균 매출 계산
    total_sales = stats['평균_매출']  # 업종 전체 매출
    total_stores = row_data['점포_수']  # 해당 지역의 업종 점포 수
    sales_per_store = (total_sales / total_stores) if total_stores > 0 else 0
    rent_burden = (rent / sales_per_store * 100) if sales_per_store > 0 else 0

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
            <div class="stat-label">지역 평균 건수(단위: 천 건)</div>
            <div class="stat-value">{stats['평균_매출건수']/1000:,.0f}</div>
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
                    recommendations.append("🔴 **임대료 부담률이 매우 높습니다.** 매출 증대 방안을 적극적으로 모색하거나, 더 저렴한 임대료의 매장을 고려해보세요.")
                else:
                    recommendations.append("🟡 **임대료 부담률이 다소 높은 편입니다.** 매출 증대 또는 비용 절감 방안을 준비하는 것이 좋습니다.")
                
                alt_districts = get_lower_rent_districts(merged_df, selected_industry, selected_district)
                if alt_districts:
                    recommendations.append(f"💡 **대안 지역 추천:** 동일 업종의 평균 임대료가 더 낮은 **{', '.join(alt_districts)}** 지역을 고려해보는 것은 어떠신가요?")
            else:
                recommendations.append("🟢 **임대료 부담률이 적정 수준입니다.**")

            if risk_score >= 70:
                recommendations.append("🔴 **폐업 위험도가 높습니다.** 창업 계획을 재검토하거나 다른 대안을 고려해보세요.")
                safe_industries_in_district = district_comp.get('safe_industries', pd.DataFrame())
                if not safe_industries_in_district.empty:
                    alt_industries = safe_industries_in_district['서비스_업종_코드_명'].head(3).tolist()
                    if alt_industries:
                        recommendations.append(f"💡 **대안 업종 추천:** 현재 지역({selected_district})에서는 **{', '.join(alt_industries)}** 업종이 비교적 안정적입니다.")
            elif risk_score >= 40:
                recommendations.append("🟡 **폐업 위험도가 보통입니다.** 차별화된 전략이 필요합니다.")
            else:
                recommendations.append("🟢 **폐업 위험도가 안정적입니다.** 성공적인 창업을 위해 사업 계획을 구체화하세요.")

            if row_data.get('폐업_률', 0) > 5:
                recommendations.append("⚠️ **높은 경쟁 환경:** 해당 지역의 폐업률이 높은 편입니다. 경쟁에서 살아남기 위한 차별화 전략이 중요합니다.")
                main_age, main_gender = get_main_customer_segment(row_data)
                recommendations.append(f"🎯 **타겟 고객 집중:** 이 상권의 주 고객층은 **{main_age} {main_gender}**입니다. 이들을 타겟으로 한 메뉴 개발이나 마케팅 전략을 수립하여 충성 고객을 확보하세요.")

            recommendations.append("📍 해당 지역의 상권 특성과 유동인구를 추가로 분석해보시기 바랍니다.")
            recommendations.append("🤝 전문가 상담을 통해 더 정확한 사업계획을 수립하시는 것을 권장합니다.")

            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"{i}. {rec}")

            if st.button("닫기"):
                st.session_state.show_solution = False
                st.rerun()

# 하단 정보
st.markdown("---")
st.markdown("""
<div class="info-box">
    <h4>ℹ️ 안내사항</h4>
    <ul style='line-height: 2;'>
        <li>본 서비스는 AI 기반 예측 결과로, 참고용으로만 활용하시기 바랍니다.</li>
        <li>실제 창업 결정 시에는 전문가 상담과 충분한 시장조사가 필요합니다.</li>
        <li>예측 결과는 과거 데이터를 기반으로 계산되며, 실제 결과와 다를 수 있습니다.</li>
        <li>시장 상황, 경쟁 환경 등 다양한 외부 요인도 함께 고려하셔야 합니다.</li>
    </ul>
</div>
""", unsafe_allow_html=True)



# 푸터
st.markdown("""
<div style='text-align: center; padding: 2rem; color: #6c757d; font-size: 0.9rem;'>
    <p>© 2025 서울시 자치구별 매장 폐업 예측 프로젝트. All rights reserved.</p>
    <p>예비 창업자를 위한 AI 기반 의사결정 지원 서비스</p>
</div>
""", unsafe_allow_html=True)
