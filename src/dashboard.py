import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import copy
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from engine import RiskEngine
except ImportError:
    from src.engine import RiskEngine

# ==============================================================================
#  App Configuration
# ==============================================================================
st.set_page_config(
    page_title="Ayush Pattanaik — IIIT BBSR",
    page_icon="⭐️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Premium dark-mode header styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background: #0e1117; }
    .stButton>button {
        background: linear-gradient(135deg, #6c63ff 0%, #3ecfcf 100%);
        color: white; border: none; border-radius: 8px;
        padding: 0.6rem 1.4rem; font-weight: 600;
        transition: all 0.2s ease;
        box-shadow: 0 4px 14px rgba(108, 99, 255, 0.25);
    }
    .stButton>button:hover {
        opacity: 0.9;
        transform: translateY(-1px);
        box-shadow: 0 6px 18px rgba(62, 207, 207, 0.35);
    }
    .metric-card {
        background: #1a1d27; border-radius: 12px;
        padding: 1rem 1.4rem; border-left: 4px solid #6c63ff;
    }
    .badge-bullish    { color: #00e676; font-weight: 700; }
    .badge-bearish    { color: #ff1744; font-weight: 700; }
    .badge-neutral    { color: #90a4ae; font-weight: 700; }
    .badge-s-bullish  { color: #69f0ae; font-weight: 700; }
    .badge-s-bearish  { color: #ff6d00; font-weight: 700; }
    
    /* Scrollable container refinement */
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        background: rgba(18, 22, 31, 0.5) !important;
    }
    [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stVerticalBlock"]::-webkit-scrollbar {
        width: 6px;
    }
    [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stVerticalBlock"]::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.02);
        border-radius: 4px;
    }
    [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stVerticalBlock"]::-webkit-scrollbar-thumb {
        background: rgba(108, 99, 255, 0.45);
        border-radius: 4px;
    }
    [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stVerticalBlock"]::-webkit-scrollbar-thumb:hover {
        background: #6c63ff;
    }
    div[data-testid="stExpander"] {
        border: 1px solid rgba(255, 255, 255, 0.07) !important;
        border-radius: 8px !important;
        margin-bottom: 6px !important;
        background: #141722 !important;
    }
    div[data-testid="stExpander"]:hover {
        border-color: rgba(108, 99, 255, 0.35) !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("# 📊 AI/NLP Risk Engine : Tactical Index Rebalancer")
st.markdown("*Real-time sentiment ingestion → NLP analysis → dynamic portfolio rebalancing*")
st.divider()

# ==============================================================================
#  Load Engine (cached so models are not reloaded on every click)
# ==============================================================================
@st.cache_resource
def load_engine():
    return RiskEngine(news_api_key="46946b89199c424aa61b0ca70196d0f8")

engine = load_engine()

# ==============================================================================
#  Session State Initialisation
# ==============================================================================
if "portfolio" not in st.session_state:
    # Mirror the engine's 15-stock universe
    st.session_state.portfolio = {
        ticker: {"name": info["name"], "weight": info["weight"]}
        for ticker, info in RiskEngine.PORTFOLIO.items()
    }

if "news_log" not in st.session_state:
    st.session_state.news_log = []
else:
    st.session_state.news_log = [s for s in st.session_state.news_log if "INDEX" not in s.get("affected_tickers", [])]

if "cycle" not in st.session_state:
    st.session_state.cycle = 0

if "ingestion_page" not in st.session_state:
    st.session_state.ingestion_page = 1

if "weight_history" not in st.session_state or len(st.session_state.weight_history) == 0:
    # Each entry: {"cycle": int, "AAPL": float, "MSFT": float, ...}
    init_snapshot = {"cycle": 0}
    init_snapshot.update({t: round(d["weight"] * 100, 2) for t, d in st.session_state.portfolio.items()})
    st.session_state.weight_history = [init_snapshot]

# ==============================================================================
#  Rebalancing Logic
# ==============================================================================
TRAJECTORY_ARROW = {
    "Bullish":         "↑ Bullish",
    "Slightly Bullish":"↗ Slightly Bullish",
    "Neutral":         "→ Neutral",
    "Slightly Bearish":"↘ Slightly Bearish",
    "Bearish":         "↓ Bearish",
}

TRAJECTORY_CSS = {
    "Bullish":          "badge-bullish",
    "Slightly Bullish": "badge-s-bullish",
    "Neutral":          "badge-neutral",
    "Slightly Bearish": "badge-s-bearish",
    "Bearish":          "badge-bearish",
}

def apply_rebalance(signals):
    for signal in signals:
        sentiment = signal["sentiment_score"]
        impact    = signal["impact_score"]
        for ticker in signal["affected_tickers"]:
            if ticker in st.session_state.portfolio:
                current = st.session_state.portfolio[ticker]["weight"]
                # Asymmetric risk penalty: fear > greed
                if sentiment < 0:
                    adj = (sentiment * 0.15) * (impact / 10)
                else:
                    adj = (sentiment * 0.05) * (impact / 10)
                st.session_state.portfolio[ticker]["weight"] = max(0.02, current + adj)

    # Renormalise to 100 %
    total = sum(d["weight"] for d in st.session_state.portfolio.values())
    for t in st.session_state.portfolio:
        st.session_state.portfolio[t]["weight"] /= total

    # Record snapshot for history chart
    st.session_state.cycle += 1
    snapshot = {"cycle": st.session_state.cycle}
    snapshot.update({t: round(d["weight"] * 100, 2) for t, d in st.session_state.portfolio.items()})
    st.session_state.weight_history.append(snapshot)

# ==============================================================================
#  Top Metrics Row
# ==============================================================================
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
m1_ph = col_m1.empty()
m2_ph = col_m2.empty()
m3_ph = col_m3.empty()
m4_ph = col_m4.empty()

def update_top_metrics():
    m1_ph.metric("Stocks in Index", len(st.session_state.portfolio))
    m2_ph.metric("Signals Processed", len(st.session_state.news_log))
    m3_ph.metric("Rebalance Cycles", st.session_state.cycle)
    bullish_count = sum(1 for s in st.session_state.news_log if s.get("trajectory") in ["Bullish", "Slightly Bullish"])
    m4_ph.metric("Bullish Signals", bullish_count)

update_top_metrics()

st.divider()

# ==============================================================================
#  Main Layout: Left panel | Right charts
# ==============================================================================
left, right = st.columns([1, 2], gap="large")

# ------------------------------------------------------------------------------
#  LEFT: Ingestion Controls + Signal Log
# ------------------------------------------------------------------------------
with left:
    st.subheader("📡 Data Ingestion")

    # --- Live Feed Button ---
    if st.button("⚡ Fetch & Process Live Feed", type="primary", use_container_width=True):
        current_page = st.session_state.ingestion_page
        with st.spinner(f"Ingesting portfolio signals (5 news + 5 social) — Batch #{current_page}..."):
            signals = engine.process_live_feed(page_num=current_page, news_limit=5, social_limit=5)
            st.session_state.ingestion_page = current_page + 1
            if signals:
                apply_rebalance(signals)
                st.session_state.news_log = signals + st.session_state.news_log
                update_top_metrics()
                st.success(f"✅ Ingested and processed {len(signals)} signals affecting portfolio stocks ({sum(1 for s in signals if s['source']=='News')} news, {sum(1 for s in signals if s['source']=='Social')} social) [Batch #{current_page}]")
            else:
                st.info(f"Ingested batch #{current_page}: no portfolio-relevant entities detected in this cycle.")

    st.markdown("---")

    # --- Manual Text Input ---
    st.subheader("✍️ Manual Text Analysis")
    with st.form("manual_form"):
        user_text = st.text_area(
            "Enter any news headline or tweet:",
            placeholder="e.g. Apple announces record iPhone sales, stock expected to surge...",
            height=90,
        )
        submitted = st.form_submit_button("🔍 Analyse Text", use_container_width=True)
    if submitted and user_text.strip():
        with st.spinner("Analysing text..."):
            result = engine.process_custom_text(user_text)
        if result:
            apply_rebalance([result])
            st.session_state.news_log.insert(0, result)
            update_top_metrics()
            traj_css = TRAJECTORY_CSS.get(result["trajectory"], "badge-neutral")
            st.success(f"Tickers found: **{', '.join(result['affected_tickers'])}**  |  Score: `{result['sentiment_score']}`")
            st.markdown(f"Trajectory: <span class='{traj_css}'>{TRAJECTORY_ARROW.get(result['trajectory'], result['trajectory'])}</span>", unsafe_allow_html=True)
        else:
            st.warning(
                "No portfolio connections found. The engine understands:\n"
                "- **Direct mentions**: Apple, Tesla, NVIDIA, $AAPL, Berkshire Hathaway, etc.\n"
                "- **Macro themes**: rate hike, AI bubble, recession, trade war, tariffs, fuel prices, chip shortage\n"
                "- **Sector events**: oil embargo, banking crisis, FDA approval, streaming wars, EV demand\n"
                "- **Geopolitics**: Russia deal, China ban, Taiwan tension\n\n"
                "Try: *'Trump\'s Russia deal highlights pressure to curb fuel prices'* or *'Fed signals rate hike, growth stocks will suffer'*"
            )

    st.markdown("---")

    # --- Signal Log ---
    st.subheader("📄 Recent NLP Risk Signals")

    f_col1, f_col2 = st.columns([3, 2])
    with f_col1:
        source_filter = st.selectbox(
            "Filter by source:",
            ["All", "News", "Social", "Manual Input"],
            label_visibility="collapsed",
        )
    
    filtered_log = [
        s for s in st.session_state.news_log
        if source_filter == "All" or s["source"] == source_filter
    ]

    with f_col2:
        st.markdown(
            f"<div style='text-align:right;padding-top:7px;color:#8b949e;font-size:0.83rem;font-weight:600;'>"
            f"Showing {len(filtered_log)} signal{'s' if len(filtered_log) != 1 else ''}</div>",
            unsafe_allow_html=True,
        )

    if not filtered_log:
        st.info("No signals yet. Click 'Fetch & Process Live Feed' to begin.")
    else:
        with st.container(height=520, border=True):
            for log in filtered_log:
                traj_css   = TRAJECTORY_CSS.get(log["trajectory"], "badge-neutral")
                traj_label = TRAJECTORY_ARROW.get(log["trajectory"], log["trajectory"])
                
                src_icon = "📰" if log["source"] == "News" else ("💬" if log["source"] == "Social" else "✍️")
                traj_dot = {
                    "Bullish": "🟢 ↑",
                    "Slightly Bullish": "🟢 ↗",
                    "Neutral": "⚪ →",
                    "Slightly Bearish": "🟠 ↘",
                    "Bearish": "🔴 ↓",
                }.get(log["trajectory"], "•")

                tickers_str = ", ".join(log["affected_tickers"][:3])
                if len(log["affected_tickers"]) > 3:
                    tickers_str += f" +{len(log['affected_tickers'])-3}"

                header = f"{traj_dot} {src_icon} {tickers_str} · {log['sentiment_score']:+.2f} | {log['event_classification']}"

                with st.expander(header):
                    col_e1, col_e2 = st.columns([1, 1])
                    with col_e1:
                        st.markdown(f"**Event:** `{log['event_classification']}`")
                        st.markdown(f"**Confidence:** `{log['event_confidence']:.0%}`")
                    with col_e2:
                        st.markdown(f"**Impact:** `{log['impact_score']}/10`")
                        st.markdown(f"**Trajectory:** <span class='{traj_css}'>{traj_label}</span>", unsafe_allow_html=True)

                    st.markdown(
                        f"<div style='background:#121620;border-left:3px solid #6c63ff;padding:8px 12px;"
                        f"border-radius:6px;margin:8px 0;font-size:0.85rem;color:#c9d1d9;line-height:1.45;'>"
                        f"\"{log['text_snippet']}\"</div>",
                        unsafe_allow_html=True,
                    )

                    ticker_sents = log.get("ticker_sentiments", {})
                    if ticker_sents and len(ticker_sents) > 1:
                        st.markdown("**Per-ticker impact:**")
                        breakdown_cols = st.columns(min(len(ticker_sents), 5))
                        for i, (t, score) in enumerate(ticker_sents.items()):
                            col = breakdown_cols[i % len(breakdown_cols)]
                            if score >= 0.15:
                                arrow, color = "↑", "#00e676"
                            elif score <= -0.15:
                                arrow, color = "↓", "#ff1744"
                            else:
                                arrow, color = "→", "#90a4ae"
                            col.markdown(
                                f"<div style='text-align:center;background:#1a1d27;border-radius:8px;padding:4px 8px;border:1px solid rgba(255,255,255,0.06);'>"
                                f"<b>{t}</b><br><span style='color:{color};font-size:1.05em'>{arrow} {score:+.2f}</span></div>",
                                unsafe_allow_html=True
                            )

# ------------------------------------------------------------------------------
#  RIGHT: Portfolio Charts
# ------------------------------------------------------------------------------
with right:
    st.subheader("📊 Portfolio Analytics")

    tab1, tab2, tab3 = st.tabs(["🥧 Current Weights", "🕯 OHLC Chart", "📈 Weight History"])

    # --- Tab 1: Current allocation pie & bar charts ---
    with tab1:
        labels  = list(st.session_state.portfolio.keys())
        weights = [d["weight"] for d in st.session_state.portfolio.values()]
        names   = [d["name"]   for d in st.session_state.portfolio.values()]

        fig_pie = go.Figure(data=[go.Pie(
            labels=labels,
            values=weights,
            hole=0.42,
            textinfo="label+percent",
            textposition="auto",
            automargin=True,
            insidetextorientation="horizontal",
            outsidetextfont=dict(size=11, color="#e0e0e0"),
            insidetextfont=dict(size=12, color="#ffffff"),
            marker=dict(line=dict(color="#0e1117", width=1.5)),
            rotation=90,
            hovertemplate="<b>%{label}</b> (%{customdata})<br>Weight: %{percent}<extra></extra>",
            customdata=names,
        )])
        fig_pie.update_layout(
            title_text="Real-Time Index Allocation",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e0e0e0", family="Inter, sans-serif"),
            margin=dict(t=50, b=80, l=40, r=40),
            height=500,
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.02,
                font=dict(size=11, color="#e0e0e0"),
                bgcolor="rgba(0,0,0,0)",
            ),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        # Bar chart of weights
        fig_bar = go.Figure(go.Bar(
            x=labels,
            y=[w * 100 for w in weights],
            marker_color=["#6c63ff" if w >= (1/15) else "#3ecfcf" for w in weights],
            text=[f"{w*100:.1f}%" for w in weights],
            textposition="outside",
        ))
        fig_bar.update_layout(
            title="Weight Distribution (%)",
            yaxis_title="Weight (%)",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e0e0e0", family="Inter, sans-serif"),
            margin=dict(t=40, b=20, l=10, r=10),
            height=340,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # --- Tab 2: OHLC candlestick ---
    with tab2:
        selected = st.selectbox(
            "Select asset for OHLC view:",
            list(st.session_state.portfolio.keys()),
            format_func=lambda t: f"{t} — {st.session_state.portfolio[t]['name']}",
        )
        period = st.radio("Period:", ["1mo", "3mo", "6mo"], horizontal=True, index=1)
        with st.spinner(f"Fetching {selected} data from Yahoo Finance..."):
            df = yf.Ticker(selected).history(period=period)

        if df.empty:
            st.warning(f"No price data found for {selected}.")
        else:
            fig_c = go.Figure(data=[go.Candlestick(
                x=df.index,
                open=df["Open"], high=df["High"],
                low=df["Low"],  close=df["Close"],
                increasing_line_color="#00e676",
                decreasing_line_color="#ff1744",
            )])
            # Overlay 20-day SMA
            df["SMA20"] = df["Close"].rolling(20).mean()
            fig_c.add_trace(go.Scatter(
                x=df.index, y=df["SMA20"],
                mode="lines", name="SMA 20",
                line=dict(color="#6c63ff", width=1.5, dash="dot"),
            ))
            current_weight = st.session_state.portfolio[selected]["weight"] * 100
            fig_c.update_layout(
                title=f"{selected} OHLC — Current index weight: {current_weight:.1f}%",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(14,17,23,1)",
                font=dict(color="#e0e0e0", family="Inter, sans-serif"),
                xaxis_rangeslider_visible=False,
                margin=dict(t=50, b=10, l=10, r=10),
                height=480,
            )
            st.plotly_chart(fig_c, use_container_width=True)

            # Show trajectory predictions for selected ticker
            ticker_signals = [s for s in st.session_state.news_log if selected in s["affected_tickers"]]
            if ticker_signals:
                st.markdown(f"**Recent trajectory signals for {selected}:**")
                for sig in ticker_signals[:4]:
                    traj_css   = TRAJECTORY_CSS.get(sig["trajectory"], "badge-neutral")
                    traj_arrow = TRAJECTORY_ARROW.get(sig["trajectory"], sig["trajectory"])
                    st.markdown(
                        f"- `{sig['sentiment_score']:+.3f}` — "
                        f"<span class='{traj_css}'>{traj_arrow}</span> "
                        f"({sig['event_classification']}, impact {sig['impact_score']}/10) — "
                        f"*{sig['text_snippet'][:60]}...*",
                        unsafe_allow_html=True,
                    )

    # --- Tab 3: Weight history table & cycle insights ---
    with tab3:
        if not st.session_state.weight_history:
            st.info("No rebalance cycles logged yet. Click 'Fetch & Process Live Feed' to rebalance.")
        else:
            hist_df = pd.DataFrame(st.session_state.weight_history).set_index("cycle")
            num_cycles = len(hist_df)
            latest_cycle_num = hist_df.index[-1]
            
            latest_series = hist_df.iloc[-1]
            prev_series   = hist_df.iloc[-2] if num_cycles > 1 else latest_series
            init_series   = hist_df.iloc[0]
            equal_benchmark = round(100 / len(st.session_state.portfolio), 2)
            
            # KPI Summary Row
            total_deltas = (latest_series - init_series).round(2)
            top_winner = total_deltas.idxmax()
            top_loser  = total_deltas.idxmin()
            net_turnover = (total_deltas.abs().sum() / 2).round(2)

            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric("Active Cycles", f"Cycle {latest_cycle_num}", f"{num_cycles} snapshot{'s' if num_cycles!=1 else ''}")
            kpi2.metric("Top Inflow", f"{top_winner}", f"{total_deltas[top_winner]:+.2f}% total")
            kpi3.metric("Top Outflow", f"{top_loser}", f"{total_deltas[top_loser]:+.2f}% total")
            kpi4.metric("Turnover Shift", f"{net_turnover:.1f}%", "One-way reallocated")

            st.markdown("#### 📋 Weight Evolution & Attribution Table")
            st.caption("Tracking how every stock's index weighting has evolved across rebalancing cycles vs benchmark.")

            rows = []
            for t, d in st.session_state.portfolio.items():
                curr_w = float(latest_series.get(t, d["weight"] * 100))
                prev_w = float(prev_series.get(t, curr_w))
                init_w = float(init_series.get(t, curr_w))

                d_cycle = round(curr_w - prev_w, 2)
                d_total = round(curr_w - init_w, 2)
                d_vs_eq = round(curr_w - equal_benchmark, 2)

                # Format directional arrows
                shift_cycle_str = f"+{d_cycle:.2f}% ↑" if d_cycle > 0.05 else (f"{d_cycle:.2f}% ↓" if d_cycle < -0.05 else f"0.00% →")
                shift_total_str = f"+{d_total:.2f}% ↑" if d_total > 0.05 else (f"{d_total:.2f}% ↓" if d_total < -0.05 else f"0.00% →")

                if d_total > 0.5:
                    tilt = "🟢 Overweight"
                elif d_total < -0.5:
                    tilt = "🔴 Underweight"
                else:
                    tilt = "⚪ Neutral"

                # Trend sparkline list
                trend = [float(val) for val in hist_df[t].tolist()] if t in hist_df.columns else [curr_w]

                rows.append({
                    "Ticker": t,
                    "Asset Name": d["name"],
                    "Initial %": init_w,
                    "Previous %": prev_w,
                    "Current %": curr_w,
                    "1-Cycle Shift": shift_cycle_str,
                    "Cumulative Shift": shift_total_str,
                    "vs Equal Weight": d_vs_eq,
                    "Weight Trend": trend,
                    "Tilt Status": tilt,
                })

            summary_df = pd.DataFrame(rows)
            # Sort by Current % descending
            summary_df = summary_df.sort_values(by="Current %", ascending=False)

            st.dataframe(
                summary_df,
                column_config={
                    "Ticker": st.column_config.TextColumn("Ticker", width="small"),
                    "Asset Name": st.column_config.TextColumn("Asset Name", width="medium"),
                    "Initial %": st.column_config.NumberColumn("Initial %", format="%.2f%%"),
                    "Previous %": st.column_config.NumberColumn("Previous %", format="%.2f%%"),
                    "Current %": st.column_config.NumberColumn("Current %", format="%.2f%%"),
                    "1-Cycle Shift": st.column_config.TextColumn("1-Cycle Shift (Δ)"),
                    "Cumulative Shift": st.column_config.TextColumn("Total Shift (Δ)"),
                    "vs Equal Weight": st.column_config.NumberColumn("vs Equal Weight", format="%+.2f%%"),
                    "Weight Trend": st.column_config.LineChartColumn("Cycle Evolution", y_min=0, y_max=35),
                    "Tilt Status": st.column_config.TextColumn("Portfolio Bias", width="medium"),
                },
                use_container_width=True,
                hide_index=True,
            )

            # Detailed cycle breakdown expander
            if num_cycles > 1:
                with st.expander("🔍 View Complete Historical Cycle Matrix (Cycle 0 → Latest)"):
                    matrix_df = hist_df.T.copy()
                    matrix_df.columns = [f"Cycle {c} (%)" for c in matrix_df.columns]
                    matrix_df.index.name = "Ticker"
                    st.dataframe(matrix_df.style.format("{:.2f}%"), use_container_width=True)
