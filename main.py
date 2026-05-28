import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import numpy as np

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="글로벌 주식 비교 분석",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
  }

  /* Dark premium background */
  .stApp {
    background: #0d0f14;
    color: #e8eaf0;
  }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: #13161e !important;
    border-right: 1px solid #1f2435;
  }

  /* Header */
  .main-header {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    background: linear-gradient(135deg, #f0c040 0%, #f5a623 50%, #e8875a 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -1px;
    line-height: 1.1;
  }

  .sub-header {
    color: #6b7394;
    font-size: 0.95rem;
    font-weight: 300;
    letter-spacing: 0.05em;
    text-transform: uppercase;
  }

  /* Metric cards */
  .metric-card {
    background: #13161e;
    border: 1px solid #1f2435;
    border-radius: 16px;
    padding: 20px 24px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
  }
  .metric-card:hover { border-color: #f0c040; }
  .metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #f0c040, #f5a623);
  }
  .metric-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #6b7394;
    margin-bottom: 6px;
  }
  .metric-value {
    font-size: 1.9rem;
    font-weight: 700;
    color: #e8eaf0;
    line-height: 1;
  }
  .metric-delta-pos { color: #34d399; font-size: 0.85rem; }
  .metric-delta-neg { color: #f87171; font-size: 0.85rem; }

  /* Section divider */
  .section-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.35rem;
    color: #e8eaf0;
    border-left: 3px solid #f0c040;
    padding-left: 12px;
    margin: 24px 0 16px;
  }

  /* Tabs */
  button[data-baseweb="tab"] {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.05em;
    text-transform: uppercase;
  }

  /* Plotly chart background override */
  .js-plotly-plot .plotly .bg { fill: #13161e !important; }

  /* Streamlit metric override */
  [data-testid="stMetric"] {
    background: #13161e;
    border: 1px solid #1f2435;
    border-radius: 12px;
    padding: 16px;
  }
  [data-testid="stMetricValue"] { color: #e8eaf0 !important; }
  [data-testid="stMetricDelta"] svg { display: none; }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 5px; height: 5px; }
  ::-webkit-scrollbar-track { background: #0d0f14; }
  ::-webkit-scrollbar-thumb { background: #2a2f45; border-radius: 99px; }
</style>
""", unsafe_allow_html=True)

# ── Stock Universes ───────────────────────────────────────────────────────────
KR_STOCKS = {
    "삼성전자": "005930.KS",
    "SK하이닉스": "000660.KS",
    "LG에너지솔루션": "373220.KS",
    "삼성바이오로직스": "207940.KS",
    "현대차": "005380.KS",
    "POSCO홀딩스": "005490.KS",
    "카카오": "035720.KS",
    "NAVER": "035420.KS",
    "삼성SDI": "006400.KS",
    "셀트리온": "068270.KS",
    "KB금융": "105560.KS",
    "신한지주": "055550.KS",
    "LG화학": "051910.KS",
    "기아": "000270.KS",
    "하이브": "352820.KS",
}

US_STOCKS = {
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "NVIDIA": "NVDA",
    "Amazon": "AMZN",
    "Alphabet (Google)": "GOOGL",
    "Meta": "META",
    "Tesla": "TSLA",
    "Berkshire Hathaway": "BRK-B",
    "JPMorgan Chase": "JPM",
    "Visa": "V",
    "UnitedHealth": "UNH",
    "Walmart": "WMT",
    "Eli Lilly": "LLY",
    "Broadcom": "AVGO",
    "ExxonMobil": "XOM",
}

PERIOD_OPTIONS = {
    "1개월": "1mo",
    "3개월": "3mo",
    "6개월": "6mo",
    "1년": "1y",
    "2년": "2y",
    "5년": "5y",
}

CHART_BG = "#13161e"
GRID_COLOR = "#1f2435"
TEXT_COLOR = "#a0a8c0"
GOLD = "#f0c040"
GREEN = "#34d399"
RED = "#f87171"
BLUE = "#60a5fa"

# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def fetch_data(tickers: list[str], period: str) -> dict[str, pd.DataFrame]:
    data = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
            if not df.empty:
                data[ticker] = df
        except Exception:
            pass
    return data


def calc_return(df: pd.DataFrame) -> float:
    """Total return (%) over the period."""
    closes = df["Close"].dropna()
    if len(closes) < 2:
        return 0.0
    return float((closes.iloc[-1] / closes.iloc[0] - 1) * 100)


def calc_volatility(df: pd.DataFrame) -> float:
    """Annualised volatility (%)."""
    closes = df["Close"].dropna()
    if len(closes) < 2:
        return 0.0
    daily_ret = closes.pct_change().dropna()
    return float(daily_ret.std() * np.sqrt(252) * 100)


def calc_sharpe(df: pd.DataFrame, rf: float = 0.04) -> float:
    closes = df["Close"].dropna()
    if len(closes) < 2:
        return 0.0
    daily_ret = closes.pct_change().dropna()
    excess = daily_ret.mean() * 252 - rf
    vol = daily_ret.std() * np.sqrt(252)
    return float(excess / vol) if vol != 0 else 0.0


def calc_max_drawdown(df: pd.DataFrame) -> float:
    closes = df["Close"].dropna()
    if len(closes) < 2:
        return 0.0
    rolling_max = closes.cummax()
    drawdown = (closes - rolling_max) / rolling_max
    return float(drawdown.min() * 100)


def normalise(df: pd.DataFrame) -> pd.Series:
    closes = df["Close"].dropna()
    return closes / closes.iloc[0] * 100


def safe_value(df: pd.DataFrame) -> float:
    try:
        return float(df["Close"].dropna().iloc[-1])
    except Exception:
        return 0.0


def fmt_price(val: float, is_kr: bool) -> str:
    if is_kr:
        return f"₩{val:,.0f}"
    return f"${val:,.2f}"


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ 분석 설정")
    st.markdown("---")

    period_label = st.selectbox("📅 기간", list(PERIOD_OPTIONS.keys()), index=3)
    period = PERIOD_OPTIONS[period_label]

    st.markdown("#### 🇰🇷 한국 주식 선택")
    selected_kr_names = st.multiselect(
        "한국 종목", list(KR_STOCKS.keys()),
        default=["삼성전자", "SK하이닉스", "현대차", "NAVER"],
    )

    st.markdown("#### 🇺🇸 미국 주식 선택")
    selected_us_names = st.multiselect(
        "미국 종목", list(US_STOCKS.keys()),
        default=["Apple", "NVIDIA", "Tesla", "Amazon"],
    )

    st.markdown("---")
    chart_type = st.radio(
        "📊 캔들/라인 차트",
        ["라인", "캔들스틱"],
        horizontal=True,
    )

    show_volume = st.toggle("거래량 표시", value=True)
    show_ma = st.toggle("이동평균선 (20/60일)", value=True)

    st.markdown("---")
    st.markdown(
        "<span style='color:#6b7394;font-size:0.75rem;'>데이터: Yahoo Finance (yfinance)<br>5분마다 자동 갱신</span>",
        unsafe_allow_html=True,
    )

# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown(
    "<div class='main-header'>글로벌 주식 비교 분석</div>"
    "<div class='sub-header'>한국 · 미국 핵심 종목 수익률 & 차트 한눈에 보기</div>",
    unsafe_allow_html=True,
)
st.markdown("<br>", unsafe_allow_html=True)

kr_tickers = [KR_STOCKS[n] for n in selected_kr_names]
us_tickers = [US_STOCKS[n] for n in selected_us_names]
all_tickers = kr_tickers + us_tickers
all_names = selected_kr_names + selected_us_names
name_map = {**{KR_STOCKS[n]: n for n in selected_kr_names}, **{US_STOCKS[n]: n for n in selected_us_names}}

if not all_tickers:
    st.info("👈 사이드바에서 최소 1개 이상의 종목을 선택해주세요.")
    st.stop()

# Fetch
with st.spinner("📡 시세 데이터 불러오는 중…"):
    data = fetch_data(all_tickers, period)

if not data:
    st.error("데이터를 불러올 수 없습니다. 잠시 후 다시 시도해주세요.")
    st.stop()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📈 수익률 비교", "🕯️ 개별 차트", "📊 성과 지표", "🔥 상관관계"])

# ════════════════════════════════════════════════════════════════════
# TAB 1 — 수익률 비교
# ════════════════════════════════════════════════════════════════════
with tab1:
    # Summary metrics row
    returns = {}
    for ticker in all_tickers:
        if ticker in data:
            returns[ticker] = calc_return(data[ticker])

    sorted_ret = sorted(returns.items(), key=lambda x: x[1], reverse=True)

    if sorted_ret:
        best_ticker, best_ret = sorted_ret[0]
        worst_ticker, worst_ret = sorted_ret[-1]

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
            <div class='metric-card'>
              <div class='metric-label'>🏆 최고 수익 종목</div>
              <div class='metric-value'>{name_map.get(best_ticker, best_ticker)}</div>
              <div class='metric-delta-pos'>+{best_ret:.2f}%</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            avg_ret = np.mean(list(returns.values()))
            color = "metric-delta-pos" if avg_ret >= 0 else "metric-delta-neg"
            sign = "+" if avg_ret >= 0 else ""
            st.markdown(f"""
            <div class='metric-card'>
              <div class='metric-label'>📊 평균 수익률</div>
              <div class='metric-value'>{sign}{avg_ret:.2f}%</div>
              <div class='{color}'>{len(returns)}개 종목 평균</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class='metric-card'>
              <div class='metric-label'>📉 최저 수익 종목</div>
              <div class='metric-value'>{name_map.get(worst_ticker, worst_ticker)}</div>
              <div class='metric-delta-neg'>{worst_ret:.2f}%</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Normalised line chart
    st.markdown("<div class='section-title'>📈 누적 수익률 추이 (기준: 100)</div>", unsafe_allow_html=True)

    fig_line = go.Figure()
    palette = px.colors.qualitative.Set2 + px.colors.qualitative.Pastel
    for i, ticker in enumerate(all_tickers):
        if ticker not in data:
            continue
        norm = normalise(data[ticker])
        is_kr = ticker in kr_tickers
        color = palette[i % len(palette)]
        fig_line.add_trace(go.Scatter(
            x=norm.index,
            y=norm.values,
            name=name_map.get(ticker, ticker),
            line=dict(width=2, color=color),
            mode="lines",
            hovertemplate=f"<b>{name_map.get(ticker, ticker)}</b><br>%{{x|%Y-%m-%d}}<br>지수: %{{y:.1f}}<extra></extra>",
        ))

    fig_line.add_hline(y=100, line_dash="dot", line_color="#2a2f45", line_width=1)
    fig_line.update_layout(
        plot_bgcolor=CHART_BG, paper_bgcolor=CHART_BG,
        font=dict(color=TEXT_COLOR, family="DM Sans"),
        legend=dict(
            bgcolor="#0d0f14", bordercolor="#1f2435", borderwidth=1,
            font=dict(size=11), orientation="v",
        ),
        xaxis=dict(gridcolor=GRID_COLOR, showgrid=True, zeroline=False),
        yaxis=dict(gridcolor=GRID_COLOR, showgrid=True, zeroline=False, title="지수 (시작=100)"),
        margin=dict(l=0, r=0, t=10, b=0),
        height=420,
        hovermode="x unified",
    )
    st.plotly_chart(fig_line, use_container_width=True)

    # Bar chart — total return
    st.markdown("<div class='section-title'>📊 종목별 총 수익률 (%)</div>", unsafe_allow_html=True)

    bar_names = [name_map.get(t, t) for t in all_tickers if t in returns]
    bar_vals = [returns[t] for t in all_tickers if t in returns]
    bar_colors = [GREEN if v >= 0 else RED for v in bar_vals]
    bar_regions = ["🇰🇷 KR" if t in kr_tickers else "🇺🇸 US" for t in all_tickers if t in returns]

    fig_bar = go.Figure(go.Bar(
        x=bar_names, y=bar_vals, marker_color=bar_colors,
        text=[f"{v:+.1f}%" for v in bar_vals],
        textposition="outside",
        textfont=dict(size=11),
        hovertemplate="<b>%{x}</b><br>수익률: %{y:.2f}%<extra></extra>",
        customdata=bar_regions,
    ))
    fig_bar.add_hline(y=0, line_color="#2a2f45", line_width=1)
    fig_bar.update_layout(
        plot_bgcolor=CHART_BG, paper_bgcolor=CHART_BG,
        font=dict(color=TEXT_COLOR, family="DM Sans"),
        xaxis=dict(tickangle=-30, gridcolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR, title="수익률 (%)"),
        margin=dict(l=0, r=0, t=30, b=0),
        height=360,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# ════════════════════════════════════════════════════════════════════
# TAB 2 — 개별 차트
# ════════════════════════════════════════════════════════════════════
with tab2:
    selected_name = st.selectbox(
        "종목 선택",
        [n for n in all_names if (KR_STOCKS.get(n) or US_STOCKS.get(n)) in data],
    )
    if not selected_name:
        st.info("종목을 선택해주세요.")
        st.stop()

    sel_ticker = KR_STOCKS.get(selected_name) or US_STOCKS.get(selected_name)
    sel_is_kr = sel_ticker in kr_tickers
    df_sel = data[sel_ticker]
    close_sel = df_sel["Close"].dropna()

    # Mini KPIs
    ret_sel = calc_return(df_sel)
    vol_sel = calc_volatility(df_sel)
    dd_sel = calc_max_drawdown(df_sel)
    sharpe_sel = calc_sharpe(df_sel)
    last_price = safe_value(df_sel)

    k1, k2, k3, k4, k5 = st.columns(5)
    kpis = [
        ("현재가", fmt_price(last_price, sel_is_kr), None),
        ("기간 수익률", f"{ret_sel:+.2f}%", ret_sel >= 0),
        ("연환산 변동성", f"{vol_sel:.1f}%", None),
        ("최대낙폭", f"{dd_sel:.1f}%", False),
        ("샤프 지수", f"{sharpe_sel:.2f}", sharpe_sel >= 1),
    ]
    for col, (label, val, pos) in zip([k1, k2, k3, k4, k5], kpis):
        if pos is True:
            delta_cls = "metric-delta-pos"
        elif pos is False:
            delta_cls = "metric-delta-neg"
        else:
            delta_cls = ""
        with col:
            st.markdown(f"""
            <div class='metric-card' style='padding:14px 16px;'>
              <div class='metric-label'>{label}</div>
              <div class='metric-value' style='font-size:1.3rem;'>{val}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Build chart
    rows = 2 if show_volume else 1
    row_heights = [0.72, 0.28] if show_volume else [1]
    fig_candle = make_subplots(
        rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.03,
        row_heights=row_heights,
    )

    if chart_type == "캔들스틱":
        fig_candle.add_trace(go.Candlestick(
            x=df_sel.index,
            open=df_sel["Open"].squeeze(),
            high=df_sel["High"].squeeze(),
            close=df_sel["Close"].squeeze(),
            low=df_sel["Low"].squeeze(),
            name=selected_name,
            increasing=dict(line=dict(color=GREEN), fillcolor=GREEN),
            decreasing=dict(line=dict(color=RED), fillcolor=RED),
        ), row=1, col=1)
    else:
        fig_candle.add_trace(go.Scatter(
            x=df_sel.index, y=close_sel,
            name=selected_name,
            line=dict(color=GOLD, width=2),
            fill="tozeroy", fillcolor="rgba(240,192,64,0.05)",
        ), row=1, col=1)

    if show_ma:
        for window, color in [(20, "#60a5fa"), (60, "#c084fc")]:
            ma = close_sel.rolling(window).mean()
            fig_candle.add_trace(go.Scatter(
                x=ma.index, y=ma, name=f"MA{window}",
                line=dict(color=color, width=1.2, dash="dot"),
                opacity=0.85,
            ), row=1, col=1)

    if show_volume and "Volume" in df_sel.columns:
        vol_series = df_sel["Volume"].dropna().squeeze()
        close_series = df_sel["Close"].dropna().squeeze()
        vol_colors = [
            GREEN if float(close_series.iloc[i]) >= float(close_series.iloc[i - 1]) else RED
            for i in range(len(close_series))
        ]
        fig_candle.add_trace(go.Bar(
            x=df_sel.index, y=vol_series,
            name="거래량", marker_color=vol_colors, opacity=0.6,
        ), row=2, col=1)

    fig_candle.update_layout(
        plot_bgcolor=CHART_BG, paper_bgcolor=CHART_BG,
        font=dict(color=TEXT_COLOR, family="DM Sans"),
        xaxis_rangeslider_visible=False,
        legend=dict(bgcolor="#0d0f14", bordercolor="#1f2435", borderwidth=1),
        margin=dict(l=0, r=0, t=10, b=0),
        height=520,
    )
    for i in range(1, rows + 1):
        fig_candle.update_xaxes(gridcolor=GRID_COLOR, row=i, col=1)
        fig_candle.update_yaxes(gridcolor=GRID_COLOR, row=i, col=1)

    st.plotly_chart(fig_candle, use_container_width=True)

# ════════════════════════════════════════════════════════════════════
# TAB 3 — 성과 지표 테이블
# ════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-title'>📋 종합 성과 지표</div>", unsafe_allow_html=True)

    rows_data = []
    for ticker in all_tickers:
        if ticker not in data:
            continue
        df_t = data[ticker]
        rows_data.append({
            "종목": name_map.get(ticker, ticker),
            "시장": "🇰🇷 KR" if ticker in kr_tickers else "🇺🇸 US",
            "현재가": fmt_price(safe_value(df_t), ticker in kr_tickers),
            "수익률 (%)": round(calc_return(df_t), 2),
            "연환산 변동성 (%)": round(calc_volatility(df_t), 2),
            "최대낙폭 (%)": round(calc_max_drawdown(df_t), 2),
            "샤프 지수": round(calc_sharpe(df_t), 2),
        })

    perf_df = pd.DataFrame(rows_data)

    def color_ret(val):
        if isinstance(val, (int, float)):
            color = "#34d399" if val >= 0 else "#f87171"
            return f"color: {color}; font-weight: 600"
        return ""

    def color_dd(val):
        if isinstance(val, (int, float)):
            return "color: #f87171"
        return ""

    def color_sharpe(val):
        if isinstance(val, (int, float)) and val >= 1:
            return "color: #34d399"
        return ""

    styled = (
        perf_df.style
        .applymap(color_ret, subset=["수익률 (%)"])
        .applymap(color_dd, subset=["최대낙폭 (%)"])
        .applymap(color_sharpe, subset=["샤프 지수"])
        .set_properties(**{
            "background-color": "#13161e",
            "color": "#e8eaf0",
            "border-color": "#1f2435",
        })
        .set_table_styles([
            {"selector": "th", "props": [
                ("background-color", "#0d0f14"),
                ("color", "#f0c040"),
                ("font-size", "0.78rem"),
                ("text-transform", "uppercase"),
                ("letter-spacing", "0.08em"),
                ("border-color", "#1f2435"),
            ]},
        ])
    )
    st.dataframe(styled, use_container_width=True, height=min(60 + len(rows_data) * 40, 500))

    # Radar / spider chart
    if len(rows_data) >= 2:
        st.markdown("<div class='section-title'>🕸️ 레이더 비교 (상위 8개 종목)</div>", unsafe_allow_html=True)

        top8 = perf_df.head(8)
        categories = ["수익률 (%)", "연환산 변동성 (%)", "샤프 지수"]

        def norm_col(series):
            mn, mx = series.min(), series.max()
            if mx == mn:
                return pd.Series([0.5] * len(series), index=series.index)
            return (series - mn) / (mx - mn)

        fig_radar = go.Figure()
        theta = ["수익률", "낮은 변동성", "샤프 지수", "수익률"]  # close the loop
        for _, row in top8.iterrows():
            # invert volatility (lower = better)
            inv_vol = 1 - norm_col(top8["연환산 변동성 (%)"])[row.name]
            r_vals = [
                norm_col(top8["수익률 (%)"])[row.name],
                inv_vol,
                norm_col(top8["샤프 지수"])[row.name],
                norm_col(top8["수익률 (%)"])[row.name],
            ]
            fig_radar.add_trace(go.Scatterpolar(
                r=r_vals, theta=theta, fill="toself",
                name=row["종목"], opacity=0.55,
            ))

        fig_radar.update_layout(
            polar=dict(
                bgcolor="#13161e",
                radialaxis=dict(visible=True, range=[0, 1], gridcolor=GRID_COLOR, color=TEXT_COLOR),
                angularaxis=dict(gridcolor=GRID_COLOR, color=TEXT_COLOR),
            ),
            paper_bgcolor=CHART_BG,
            font=dict(color=TEXT_COLOR, family="DM Sans"),
            legend=dict(bgcolor="#0d0f14", bordercolor="#1f2435", borderwidth=1),
            margin=dict(l=40, r=40, t=20, b=20),
            height=420,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

# ════════════════════════════════════════════════════════════════════
# TAB 4 — 상관관계 히트맵
# ════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("<div class='section-title'>🔥 수익률 상관관계 히트맵</div>", unsafe_allow_html=True)

    if len(all_tickers) < 2:
        st.info("상관관계 분석을 위해 최소 2개 이상의 종목을 선택해주세요.")
    else:
        ret_df = pd.DataFrame()
        for ticker in all_tickers:
            if ticker in data:
                s = data[ticker]["Close"].dropna().squeeze()
                s.name = name_map.get(ticker, ticker)
                ret_df = pd.concat([ret_df, s.pct_change()], axis=1)

        if ret_df.shape[1] >= 2:
            corr = ret_df.corr()

            fig_heat = go.Figure(go.Heatmap(
                z=corr.values,
                x=corr.columns.tolist(),
                y=corr.index.tolist(),
                colorscale=[
                    [0.0, "#f87171"],
                    [0.5, "#13161e"],
                    [1.0, "#34d399"],
                ],
                zmin=-1, zmax=1,
                text=[[f"{v:.2f}" for v in row] for row in corr.values],
                texttemplate="%{text}",
                textfont=dict(size=11),
                hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>상관계수: %{z:.3f}<extra></extra>",
                colorbar=dict(
                    title=dict(text="상관계수", side="right"),
                    tickfont=dict(color=TEXT_COLOR),
                    bgcolor=CHART_BG,
                    bordercolor="#1f2435",
                ),
            ))
            fig_heat.update_layout(
                plot_bgcolor=CHART_BG, paper_bgcolor=CHART_BG,
                font=dict(color=TEXT_COLOR, family="DM Sans"),
                xaxis=dict(tickangle=-35, side="bottom"),
                margin=dict(l=10, r=10, t=20, b=10),
                height=max(350, 60 * len(corr)),
            )
            st.plotly_chart(fig_heat, use_container_width=True)

            # Scatter matrix for top 4
            top4_names = [name_map.get(t, t) for t in all_tickers[:4] if t in data]
            if len(top4_names) >= 2:
                st.markdown("<div class='section-title'>🔵 산점도 매트릭스 (최대 4개 종목)</div>", unsafe_allow_html=True)
                sub_ret = ret_df[top4_names].dropna()
                fig_scatter = px.scatter_matrix(
                    sub_ret,
                    dimensions=top4_names,
                    color_discrete_sequence=[GOLD],
                )
                fig_scatter.update_traces(diagonal_visible=False, marker=dict(size=2.5, opacity=0.4))
                fig_scatter.update_layout(
                    plot_bgcolor=CHART_BG, paper_bgcolor=CHART_BG,
                    font=dict(color=TEXT_COLOR, family="DM Sans"),
                    margin=dict(l=0, r=0, t=20, b=0),
                    height=420,
                )
                for ax in fig_scatter.select_xaxes():
                    ax.update(gridcolor=GRID_COLOR)
                for ay in fig_scatter.select_yaxes():
                    ay.update(gridcolor=GRID_COLOR)
                st.plotly_chart(fig_scatter, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#3a405a;font-size:0.75rem;'>본 서비스는 투자 권유가 아닙니다. 데이터 지연이 발생할 수 있으며 투자 결정은 본인 책임하에 하십시오.</div>",
    unsafe_allow_html=True,
)
