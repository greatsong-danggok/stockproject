"""
📈 국내 · 글로벌 주식 분석 대시보드
yfinance + Streamlit + Plotly
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# ── 페이지 설정 ─────────────────────────────────────
st.set_page_config(
    page_title="주식 분석 대시보드",
    page_icon="📈",
    layout="wide",
)

st.title("📈 국내 · 글로벌 주식 분석 대시보드")
st.caption("yfinance 기반 · 실시간 데이터")

# ── 종목 정의 ────────────────────────────────────────
KOREAN_STOCKS = {
    "삼성전자":    "005930.KS",
    "SK하이닉스":  "000660.KS",
    "LG에너지솔루션": "373220.KS",
    "현대차":      "005380.KS",
    "POSCO홀딩스": "005490.KS",
    "카카오":      "035720.KS",
    "NAVER":       "035420.KS",
    "셀트리온":    "068270.KS",
    "KB금융":      "105560.KS",
    "기아":        "000270.KS",
}

GLOBAL_STOCKS = {
    "Apple":      "AAPL",
    "Microsoft":  "MSFT",
    "NVIDIA":     "NVDA",
    "Tesla":      "TSLA",
    "Amazon":     "AMZN",
    "Alphabet":   "GOOGL",
    "Meta":       "META",
    "TSMC":       "TSM",
    "ASML":       "ASML",
    "Berkshire":  "BRK-B",
}

ALL_STOCKS = {**KOREAN_STOCKS, **GLOBAL_STOCKS}

# ── 사이드바 설정 ────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 설정")

    market = st.radio("시장 선택", ["국내", "글로벌", "전체 비교"])

    period_map = {
        "1개월": "1mo", "3개월": "3mo",
        "6개월": "6mo", "1년": "1y", "2년": "2y",
    }
    period_label = st.select_slider("기간", list(period_map.keys()), value="1년")
    period = period_map[period_label]

    if market == "국내":
        stock_pool = KOREAN_STOCKS
    elif market == "글로벌":
        stock_pool = GLOBAL_STOCKS
    else:
        stock_pool = ALL_STOCKS

    selected_names = st.multiselect(
        "종목 선택",
        list(stock_pool.keys()),
        default=list(stock_pool.keys())[:5],
    )

    analysis = st.selectbox(
        "분석 유형",
        [
            "📊 수익률 비교 (정규화)",
            "🕯️ 캔들차트 + 이동평균",
            "🌡️ 상관관계 히트맵",
            "⚡ 변동성 분석",
            "📉 RSI 지표",
            "🗺️ 버블차트 (시총·PER·배당)",
            "📦 수익률 분포 (박스플롯)",
        ],
    )

    st.divider()
    st.caption("데이터: Yahoo Finance")

if not selected_names:
    st.warning("왼쪽에서 종목을 하나 이상 선택해 주세요.")
    st.stop()

selected_tickers = {n: stock_pool[n] for n in selected_names}

# ── 데이터 로드 ──────────────────────────────────────
@st.cache_data(ttl=600)
def load_prices(tickers: dict, period: str) -> pd.DataFrame:
    symbols = list(tickers.values())
    raw = yf.download(symbols, period=period, auto_adjust=True, progress=False)
    close = raw["Close"] if len(symbols) > 1 else raw["Close"].to_frame(symbols[0])
    close.columns = [k for k, v in tickers.items() if v in close.columns]
    return close.dropna(how="all")


@st.cache_data(ttl=600)
def load_info(tickers: dict) -> dict:
    info = {}
    for name, sym in tickers.items():
        try:
            t = yf.Ticker(sym)
            info[name] = t.info
        except Exception:
            info[name] = {}
    return info


with st.spinner("📡 데이터 불러오는 중..."):
    prices = load_prices(selected_tickers, period)
    available = [n for n in selected_names if n in prices.columns]
    prices = prices[available]

if prices.empty:
    st.error("데이터를 가져올 수 없습니다. 종목 또는 기간을 확인해 주세요.")
    st.stop()

# ── 공통 유틸 ────────────────────────────────────────
COLORS = px.colors.qualitative.Bold


def returns(df: pd.DataFrame) -> pd.DataFrame:
    return df.pct_change().dropna()


# ════════════════════════════════════════════════════
# 분석 1 · 수익률 비교 (정규화)
# ════════════════════════════════════════════════════
if analysis == "📊 수익률 비교 (정규화)":
    st.subheader("📊 누적 수익률 비교 (시작일 = 100)")

    norm = prices / prices.iloc[0] * 100
    fig = px.line(norm, labels={"value": "지수", "variable": "종목"},
                  color_discrete_sequence=COLORS)
    fig.update_layout(hovermode="x unified", height=480,
                      legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, use_container_width=True)

    # 요약 테이블
    total_ret = (prices.iloc[-1] / prices.iloc[0] - 1) * 100
    daily_vol = returns(prices).std() * np.sqrt(252) * 100
    sharpe = (returns(prices).mean() / returns(prices).std() * np.sqrt(252)).round(2)

    summary = pd.DataFrame({
        "현재가": prices.iloc[-1].round(2),
        "누적수익률(%)": total_ret.round(2),
        "연간변동성(%)": daily_vol.round(2),
        "샤프지수": sharpe,
    })
    summary = summary.sort_values("누적수익률(%)", ascending=False)
    st.dataframe(
        summary.style.background_gradient(subset=["누적수익률(%)"], cmap="RdYlGn")
                     .format({"누적수익률(%)": "{:+.2f}", "연간변동성(%)": "{:.2f}",
                              "샤프지수": "{:.2f}"}),
        use_container_width=True,
    )


# ════════════════════════════════════════════════════
# 분석 2 · 캔들차트 + 이동평균
# ════════════════════════════════════════════════════
elif analysis == "🕯️ 캔들차트 + 이동평균":
    st.subheader("🕯️ 캔들차트 + 이동평균선")

    target = st.selectbox("종목", available)
    sym = selected_tickers[target]

    @st.cache_data(ttl=600)
    def load_ohlcv(sym, period):
        df = yf.download(sym, period=period, auto_adjust=True, progress=False)
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        return df

    ohlcv = load_ohlcv(sym, period)

    ma_opts = st.multiselect("이동평균", [5, 20, 60, 120], default=[20, 60])
    for ma in ma_opts:
        ohlcv[f"MA{ma}"] = ohlcv["Close"].rolling(ma).mean()

    fig = make_subplots(rows=2, cols=1, row_heights=[0.75, 0.25],
                        shared_xaxes=True, vertical_spacing=0.04)

    fig.add_trace(go.Candlestick(
        x=ohlcv.index,
        open=ohlcv["Open"], high=ohlcv["High"],
        low=ohlcv["Low"],  close=ohlcv["Close"],
        name=target, increasing_line_color="#e63946",
        decreasing_line_color="#457b9d",
    ), row=1, col=1)

    for i, ma in enumerate(ma_opts):
        col = COLORS[i % len(COLORS)]
        fig.add_trace(go.Scatter(
            x=ohlcv.index, y=ohlcv[f"MA{ma}"],
            name=f"MA{ma}", line=dict(color=col, width=1.5),
        ), row=1, col=1)

    colors_vol = ["#e63946" if c >= o else "#457b9d"
                  for c, o in zip(ohlcv["Close"], ohlcv["Open"])]
    fig.add_trace(go.Bar(
        x=ohlcv.index, y=ohlcv["Volume"],
        name="거래량", marker_color=colors_vol, opacity=0.7,
    ), row=2, col=1)

    fig.update_layout(height=580, xaxis_rangeslider_visible=False,
                      legend=dict(orientation="h", y=-0.08))
    st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════
# 분석 3 · 상관관계 히트맵
# ════════════════════════════════════════════════════
elif analysis == "🌡️ 상관관계 히트맵":
    st.subheader("🌡️ 일별 수익률 상관관계")

    corr = returns(prices).corr().round(2)
    fig = px.imshow(
        corr, text_auto=True, color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1, aspect="auto",
    )
    fig.update_layout(height=500, coloraxis_colorbar_title="상관계수")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("해석 가이드"):
        st.markdown("""
- **+1.0 (진한 빨강)** — 거의 동시에 오르고 내림 → 분산 효과 없음  
- **0.0 (흰색)** — 서로 독립적 → 포트폴리오 분산 효과 좋음  
- **-1.0 (진한 파랑)** — 한쪽이 오르면 한쪽이 내림 → 헤지 역할  
""")


# ════════════════════════════════════════════════════
# 분석 4 · 변동성 분석 (롤링)
# ════════════════════════════════════════════════════
elif analysis == "⚡ 변동성 분석":
    st.subheader("⚡ 30일 롤링 연간 변동성")

    ret = returns(prices)
    roll_vol = ret.rolling(30).std() * np.sqrt(252) * 100

    fig = px.line(roll_vol, labels={"value": "변동성(%)", "variable": "종목"},
                  color_discrete_sequence=COLORS)
    fig.update_layout(hovermode="x unified", height=450,
                      legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.caption("평균 변동성 TOP 5 (위험 높음)")
        top = roll_vol.mean().nlargest(5).to_frame("평균변동성(%)")
        st.dataframe(top.style.format("{:.1f}").background_gradient(cmap="Reds"),
                     use_container_width=True)
    with col2:
        st.caption("평균 변동성 BOT 5 (안정적)")
        bot = roll_vol.mean().nsmallest(5).to_frame("평균변동성(%)")
        st.dataframe(bot.style.format("{:.1f}").background_gradient(cmap="Blues_r"),
                     use_container_width=True)


# ════════════════════════════════════════════════════
# 분석 5 · RSI 지표
# ════════════════════════════════════════════════════
elif analysis == "📉 RSI 지표":
    st.subheader("📉 RSI (Relative Strength Index, 14일)")

    def rsi(series: pd.Series, window: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(window).mean()
        loss = (-delta.clip(upper=0)).rolling(window).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - 100 / (1 + rs)

    target = st.selectbox("종목", available)
    rsi_series = rsi(prices[target])

    fig = make_subplots(rows=2, cols=1, row_heights=[0.6, 0.4],
                        shared_xaxes=True, vertical_spacing=0.05)

    fig.add_trace(go.Scatter(
        x=prices.index, y=prices[target], name="종가",
        line=dict(color=COLORS[0]),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=rsi_series.index, y=rsi_series, name="RSI",
        line=dict(color=COLORS[2]),
    ), row=2, col=1)

    for lvl, color, label in [(70, "red", "과매수(70)"), (30, "blue", "과매도(30)")]:
        fig.add_hline(y=lvl, line_dash="dash", line_color=color,
                      annotation_text=label, annotation_position="right",
                      row=2, col=1)

    fig.update_layout(height=500, hovermode="x unified",
                      legend=dict(orientation="h", y=-0.08))
    fig.update_yaxes(range=[0, 100], row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

    latest_rsi = rsi_series.dropna().iloc[-1]
    if latest_rsi >= 70:
        st.warning(f"⚠️ 현재 RSI **{latest_rsi:.1f}** — 과매수 구간. 단기 조정 가능성.")
    elif latest_rsi <= 30:
        st.success(f"✅ 현재 RSI **{latest_rsi:.1f}** — 과매도 구간. 반등 가능성.")
    else:
        st.info(f"ℹ️ 현재 RSI **{latest_rsi:.1f}** — 중립 구간.")


# ════════════════════════════════════════════════════
# 분석 6 · 버블차트 (시총·PER·배당)
# ════════════════════════════════════════════════════
elif analysis == "🗺️ 버블차트 (시총·PER·배당)":
    st.subheader("🗺️ 시가총액 · PER · 배당수익률 버블차트")

    with st.spinner("재무 정보 수집 중..."):
        info_data = load_info(selected_tickers)

    rows = []
    for name in available:
        d = info_data.get(name, {})
        mkt = d.get("marketCap")
        per = d.get("trailingPE")
        div = d.get("dividendYield", 0) or 0
        sect = d.get("sector", "기타")
        cur = d.get("currency", "")
        rows.append({
            "종목": name, "시가총액(억)": mkt / 1e8 if mkt else None,
            "PER": per, "배당수익률(%)": div * 100,
            "섹터": sect, "통화": cur,
        })

    bubble_df = pd.DataFrame(rows).dropna(subset=["시가총액(억)", "PER"])
    bubble_df["PER"] = bubble_df["PER"].clip(upper=200)  # 이상값 제한

    if bubble_df.empty:
        st.warning("재무 정보를 가져올 수 없는 종목이 많습니다. 글로벌 종목 위주로 선택해 보세요.")
    else:
        fig = px.scatter(
            bubble_df, x="PER", y="배당수익률(%)",
            size="시가총액(억)", color="섹터",
            hover_name="종목", text="종목",
            size_max=70, color_discrete_sequence=COLORS,
            labels={"PER": "PER (낮을수록 저평가)", "배당수익률(%)": "배당수익률(%)"},
        )
        fig.update_traces(textposition="top center")
        fig.update_layout(height=520)
        st.plotly_chart(fig, use_container_width=True)

        st.caption("버블 크기 = 시가총액 / X축 = PER / Y축 = 배당수익률 | 좌상단이 '저평가 고배당'")


# ════════════════════════════════════════════════════
# 분석 7 · 수익률 분포 (박스플롯)
# ════════════════════════════════════════════════════
elif analysis == "📦 수익률 분포 (박스플롯)":
    st.subheader("📦 일별 수익률 분포")

    ret = returns(prices) * 100
    fig = go.Figure()
    for i, col in enumerate(ret.columns):
        fig.add_trace(go.Box(
            y=ret[col], name=col,
            marker_color=COLORS[i % len(COLORS)],
            boxmean="sd",  # 평균선 + 표준편차
        ))
    fig.update_layout(
        yaxis_title="일별 수익률(%)", height=500,
        showlegend=False, hovermode="x",
    )
    st.plotly_chart(fig, use_container_width=True)

    # 왜도·첨도 테이블
    skew = ret.skew().round(3)
    kurt = ret.kurt().round(3)
    stats = pd.DataFrame({
        "평균(%)": ret.mean().round(4),
        "표준편차(%)": ret.std().round(4),
        "왜도": skew,
        "첨도": kurt,
    }).sort_values("왜도", ascending=False)
    st.dataframe(stats.style.background_gradient(subset=["왜도"], cmap="RdYlGn"),
                 use_container_width=True)

    with st.expander("왜도·첨도 해석"):
        st.markdown("""
- **왜도 > 0** — 큰 이익이 가끔 발생 (오른쪽 꼬리 긺)  
- **왜도 < 0** — 큰 손실이 가끔 발생 (왼쪽 꼬리 긺)  
- **첨도 > 0** — 정규분포보다 극단값이 자주 등장 (블랙스완 위험)  
""")

# ── 푸터 ─────────────────────────────────────────────
st.divider()
st.caption(
    f"마지막 업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M')} | "
    "데이터 출처: Yahoo Finance (yfinance) | 투자 참고용, 투자 권유 아님"
)
