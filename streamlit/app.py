import streamlit as st
import pandas as pd
from datetime import datetime

df = pd.read_csv('../model/catboost/data/merged_data.csv')

region_num = df['자치구_코드_명'].nunique()
service_num = df['서비스_업종_코드_명'].nunique()
q75 = df['폐업_률'].quantile(0.75)

# 2025년 2분기 기준
df_20252 = df[df['기준_년분기_코드'] == 20252]
mean_20252 = df_20252['폐업_률'].mean().round(2)
store_20252 = df_20252[df_20252['폐업_률'] > q75].shape[0]

# 2025년 1분기 기준
df_20251 = df[df['기준_년분기_코드'] == 20242]
mean_20251 = df_20251['폐업_률'].mean().round(2)
store_20251 = df_20251[df_20251['폐업_률'] > q75].shape[0]

# 차이
diff_mean = mean_20252 - mean_20251
diff_store = store_20252 - store_20251


# 페이지 설정
st.set_page_config(
    page_title="서울시 자치구별 매장 폐업 예측",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS
st.markdown("""
<style>
    /* 전체 배경 */
    .main {
        background-color: #f8f9fa;
    }
    
    /* 헤더 스타일 */
    .header-container {
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: black;
        text-align: center;
    }
    
    /* 통계 카드 */
    .stat-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        text-align: center;
        transition: transform 0.3s ease;
        height: 100%;
    }
    
    .stat-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .stat-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1e40af;
        margin: 0.5rem 0;
    }
    
    .stat-label {
        font-size: 1rem;
        color: #6c757d;
        margin-bottom: 0.5rem;
    }
    
    .stat-change {
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    
    .positive {
        color: #28a745;
    }
    
    .negative {
        color: #dc3545;
    }
    
    /* 정보 박스 */
    .info-box {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
        height: 300px;
        overflow-y: auto;
        overflow-x: hidden;
        word-wrap: break-word;
        word-break: keep-all;
    }

    .info-box h3 {
        margin-top: 0;
        margin-bottom: 1rem;
        font-size: 1.5rem;
    }

    .info-box p {
        margin-bottom: 0.8rem;
    }

    .info-box ul {
        margin: 0;
        padding-left: 1.5rem;
    }

    .info-box li {
        margin-bottom: 0.5rem;
    }
            
    /* 버튼 스타일 */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem;
        border-radius: 8px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
                
    .selection-card, .selection-card2 {
        background: white;
        padding: 30px; 
        border-radius: 10px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.25);
        cursor: pointer; 
        text-align: center;
        transition: all 0.3s ease;
        height: 400px; 
        border: none;
        text-decoration: none !important;
        display: block;
    }

    .selection-card h2, .selection-card2 h2 {
        font-size: 2.5rem; 
        color: #212529;
        margin-top: 15px;
        margin-bottom: 10px;
    }

    .selection-card p, .selection-card2 p {
        font-size: 1.5rem;
        color: #6c757d;
    }

    .selection-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
        background: #fdd03b;
    }

    .selection-card:hover h2,
    .selection-card:hover p {
        color: black;
    }

    .selection-card2:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
        background: #1e40af;
    }

    .selection-card2:hover h2,
    .selection-card2:hover p {
        color: white;
    }

</style>
""", unsafe_allow_html=True)


# 메인 헤더
st.markdown("""
<div class="header-container">
    <h1 style='margin:0; font-size: 2.5rem;'>서울시 자치구별 매장 폐업률 예측 및 솔루션 제공 서비스</h1>
    <p style='margin-top: 1rem; font-size: 1.5rem; opacity: 0.9; font-weight: bold; color: #1e40af;'>
        AI 기반 데이터 분석으로 예측하는 서울시 자치구별 상권 현황
    </p>
</div>
""", unsafe_allow_html=True)

# 페이지 이동 버튼
col_card1, col_card2 = st.columns(2)

with col_card1:    
    st.markdown(
        f"""
        <a href="./store_operator" target="_self" class="selection-card">
            <div class="selection-image">
                <p style='font-size: 80px;'>🏢</p>
            </div>
            <h2>매장 운영자</h2>
            <p>폐업률 예측 및 가게 주변의 상권 분석</p>
        </a>
        """,
        unsafe_allow_html=True
    )

with col_card2:
    st.markdown(
        f"""
        <a href="./pre-entrepreneur" target="_self" class="selection-card2">
            <div class="selection-image">
                <p style='font-size: 80px;'>🔭</p>
            </div>
            <h2>예비 창업자</h2>
            <p>예비 사장님들을 위한 폐업률 예측 및 전략적이고 스마트한 창업 분석</p>
        </a>
        """,
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# 현재 날짜 표시
current_date = datetime.now()
st.markdown(f"### {current_date.year}년 {current_date.month}월 기준")
st.markdown("---")

# 주요 통계
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-label">분석 업종 수</div>
        <div class="stat-value">{service_num}</div>
        <div class="stat-change">서울시 전체</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-label">분석 자치구 수</div>
        <div class="stat-value">{region_num}</div>
        <div class="stat-change">서울시 전체</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-label">폐업 위험 매장</div>
        <div class="stat-value">{store_20252}</div>
        <div class="stat-change positive">▼ 전분기 대비 {diff_store}</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-label">평균 폐업 위험도</div>
        <div class="stat-value">{mean_20252}%</div>
        <div class="stat-change positive">▼ 전분기 대비 {diff_mean}%</div>
    </div>
    """, unsafe_allow_html=True)


st.markdown("<br>", unsafe_allow_html=True)

# 프로젝트 소개
col_left, col_right = st.columns([3, 2])

with col_left:
    st.markdown("""
    <div class="info-box">
        <h3>프로젝트 소개</h3>
        <p style='font-size: 1.1rem; line-height: 1.8; color: #495057;'>
           본 프로젝트는 서울시의 실제 상권 데이터를 기반으로 자치구별 매장 폐업률을 예측하고, 예비 창업자 및 기존 매장 운영자에게 데이터 기반의 인사이트와 솔루션을 제공합니다.
        </p>
        <p style='font-size: 1.1rem; line-height: 1.8; color: #495057;'>
            최근 경기 침체 및 상권 변화로 인해 소상공인의 폐업 위험이 증가하고 있는 가운데, 본 프로젝트는 머신러닝 및 딥러닝 기법을 활용하여 폐업 위험을 사전에 예측하고, Streamlit 기반 대시보드를 통해 누구나 쉽게 접근할 수 있는 의사결정 지원 도구를 구축하는 것을 목표로 합니다.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col_right:
    st.markdown("""
    <div class="info-box">
        <h3>주요 내용</h3>
        <ul style='font-size: 1rem; line-height: 2; color: #495057;'>
            <li>서울시 자치구 및 업종별 폐업률 데이터 수집 및 분석</li>
            <li>머신러닝과 딥러닝을 활용한 예측 모델 개발</li>
            <li>사용자 맞춤형 폐업률 예측 및 시각화 제공</li>
            <li>Streamlit을 통한 웹 기반 대시보드 구현</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# 하단 정보
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div style='text-align: center; padding: 1rem;'>
        <h4>깃허브</h4>
        <a href='https://github.com/SKNetworks-AI19-250818/SKN19_2nd_1team' target='_blank' style='color: #667eea; text-decoration: none;'>
            SKN19_2nd_1team
        </a>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style='text-align: center; padding: 1rem;'>
        <h4>데이터 출처</h4>
        <a href='https://data.seoul.go.kr/' target='_blank' style='color: #667eea; text-decoration: none;'>
            서울시 열린데이터광장
        </a>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style='text-align: center; padding: 1rem;'>
        <h4>업데이트</h4>
        <p style='color: #667eea;'>분기별 업데이트</p>
    </div>
    """, unsafe_allow_html=True)