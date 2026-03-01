"""
SCRIPT NAME: app.py
====================================
Execution Location: market-hawk-mvp/dashboard/
Purpose: Streamlit Dashboard — Live Market Hawk Trading Intelligence
Hardware Optimization: Intel i7-9700F, NVIDIA GTX 1070 8GB VRAM, 64GB DDR4
Creation Date: 2026-03-01

Usage:
    cd K:\\_DEV_MVP_2026\\Market_Hawk_3
    streamlit run dashboard/app.py
"""

import sys
import json
import asyncio
from pathlib import Path
from datetime import datetime

# Project root
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import numpy as np

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="🦅 Market Hawk",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
    /* Dark trading theme */
    .stApp {
        background-color: #0e1117;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1f2e 0%, #141824 100%);
        border: 1px solid #2d3548;
        border-radius: 12px;
        padding: 20px;
        margin: 8px 0;
    }
    .signal-buy { color: #00d26a; font-weight: bold; font-size: 1.3em; }
    .signal-sell { color: #ff4757; font-weight: bold; font-size: 1.3em; }
    .signal-hold { color: #ffa502; font-weight: bold; font-size: 1.3em; }
    .agent-vote {
        background: #1a1f2e;
        border-left: 3px solid #4a6cf7;
        padding: 10px 15px;
        margin: 5px 0;
        border-radius: 0 8px 8px 0;
    }
    .anomaly-alert {
        background: #2d1a1a;
        border-left: 3px solid #ff4757;
        padding: 10px 15px;
        margin: 5px 0;
        border-radius: 0 8px 8px 0;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# CACHED INITIALIZATION
# ============================================================

@st.cache_resource
def init_system():
    """Initialize all agents (cached — only runs once)."""
    import logging
    logging.basicConfig(level=logging.WARNING)

    from brain.orchestrator import Brain
    from agents.knowledge_advisor.rag_engine import KnowledgeAdvisor
    from agents.ml_signal_engine.catboost_predictor import MLSignalEngine
    from agents.news_analyzer.news_sentiment import NewsAnalyzer
    from agents.security_guard.anomaly_detector import SecurityGuard
    from agents.risk_manager.kelly_criterion import RiskManager
    from data.market_data_fetcher import MarketDataFetcher

    brain = Brain()

    advisor = KnowledgeAdvisor()
    if advisor.initialize():
        brain.register_agent("knowledge_advisor", advisor)

    ml_engine = MLSignalEngine()
    for m in ["catboost_v2", "catboost_clean_75"]:
        ml_engine.load_model(m)
    if ml_engine._models:
        brain.register_agent("ml_signal_engine", ml_engine)

    brain.register_agent("news_analyzer", NewsAnalyzer(use_llm=True))
    brain.register_agent("security_guard", SecurityGuard())
    brain.register_agent("risk_manager", RiskManager())

    fetcher = MarketDataFetcher()

    return brain, fetcher


def run_async(coro):
    """Run async function in Streamlit."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.image("https://img.icons8.com/emoji/96/eagle-emoji.png", width=80)
st.sidebar.title("🦅 Market Hawk")
st.sidebar.markdown("**AI Multi-Agent Trading System**")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigation", [
    "📊 Live Scanner",
    "📈 Live Chart",
    "⚡ Quick Trade",
    "💰 Portfolio",
    "📜 Decision Log",
    "🤖 Agent Status",
])

WATCHLIST = st.sidebar.multiselect(
    "Watchlist",
    ["AAPL", "NVDA", "MSFT", "GOOGL", "AMZN", "META", "TSLA",
     "BTCUSDT", "ETHUSDT", "GOLD", "SILVER", "SPY", "QQQ",
     "AMD", "NFLX", "CRM"],
    default=["AAPL", "NVDA", "BTCUSDT", "GOLD", "MSFT"],
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M')}")
st.sidebar.markdown("🔧 Threshold: **0.60**")
st.sidebar.markdown("🧠 Agents: **5**")


# ============================================================
# PAGE: LIVE SCANNER
# ============================================================

def page_live_scanner():
    st.title("📊 Live Market Scanner")

    brain, fetcher = init_system()

    col_status, col_btn = st.columns([3, 1])
    with col_status:
        st.markdown(f"**Watchlist:** {len(WATCHLIST)} symbols | "
                    f"**Agents:** {len(brain.agents)} | "
                    f"**Threshold:** 0.60")
    with col_btn:
        scan = st.button("🔍 Run Scan", type="primary", use_container_width=True)

    if not scan and "scan_results" not in st.session_state:
        st.info("👆 Click **Run Scan** to analyze your watchlist with all 5 agents.")
        return

    if scan:
        results = []
        progress = st.progress(0, text="Initializing scan...")

        for i, symbol in enumerate(WATCHLIST):
            progress.progress((i + 1) / len(WATCHLIST),
                             text=f"Analyzing {symbol}... ({i+1}/{len(WATCHLIST)})")

            try:
                features_df = fetcher.fetch_and_engineer(symbol)
                if features_df is None or features_df.empty:
                    continue

                latest = fetcher.get_latest_features(symbol)
                if latest is None:
                    continue

                price = float(features_df["Close"].iloc[-1])

                decision = run_async(brain.decide(symbol, {
                    "timeframe": "1h",
                    "features": latest.tolist(),
                    "features_df": features_df,
                }))

                results.append({
                    "symbol": symbol,
                    "price": price,
                    "action": decision.action,
                    "consensus": decision.consensus_score,
                    "approved": decision.approved,
                    "votes": decision.agent_votes or [],
                    "position_size": decision.position_size,
                    "stop_loss": decision.stop_loss,
                    "take_profit": decision.take_profit,
                })
            except Exception as e:
                st.warning(f"Error scanning {symbol}: {e}")

        progress.empty()
        st.session_state.scan_results = results
        st.session_state.scan_time = datetime.now().strftime("%H:%M:%S")

    # Display results
    results = st.session_state.get("scan_results", [])
    scan_time = st.session_state.get("scan_time", "")

    if not results:
        return

    st.markdown(f"*Last scan: {scan_time}*")

    # Summary metrics
    buys = sum(1 for r in results if r["consensus"] > 0.3)
    sells = sum(1 for r in results if r["consensus"] < -0.3)
    neutral = len(results) - buys - sells

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Symbols Scanned", len(results))
    col2.metric("Bullish Bias", buys, delta=f"{buys} symbols")
    col3.metric("Bearish Bias", sells, delta=f"-{sells} symbols" if sells else "0")
    col4.metric("Neutral", neutral)

    st.markdown("---")

    # Results table
    st.subheader("📋 Scan Results")

    table_data = []
    for r in sorted(results, key=lambda x: abs(x["consensus"]), reverse=True):
        emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⚪"}.get(r["action"], "❓")
        table_data.append({
            "Symbol": r["symbol"],
            "Price": f"${r['price']:,.2f}",
            "Signal": f"{emoji} {r['action']}",
            "Consensus": f"{r['consensus']:+.4f}",
            "Strength": "█" * max(1, int(abs(r["consensus"]) * 10)),
        })

    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Detailed view per symbol
    st.markdown("---")
    st.subheader("🔬 Detailed Agent Votes")

    for r in results:
        emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⚪"}.get(r["action"], "❓")
        with st.expander(f"{emoji} **{r['symbol']}** — ${r['price']:,.2f} | "
                         f"Consensus: {r['consensus']:+.4f}"):

            # Consensus bar
            bar_val = (r["consensus"] + 1) / 2  # Normalize -1..+1 to 0..1
            st.progress(max(0.0, min(1.0, bar_val)),
                       text=f"Consensus: {r['consensus']:+.4f}")

            if r["votes"]:
                for vote in r["votes"]:
                    v_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⚪"}.get(
                        vote.get("recommendation", ""), "❓")
                    agent = vote.get("agent_name", "unknown")
                    conf = vote.get("confidence", 0)
                    reason = vote.get("reasoning", "")[:100]

                    st.markdown(
                        f'<div class="agent-vote">'
                        f'{v_emoji} <b>{agent}</b> — '
                        f'{vote.get("recommendation", "?")} '
                        f'(conf: {conf:.2f})<br>'
                        f'<small>{reason}</small></div>',
                        unsafe_allow_html=True,
                    )

            if r.get("position_size"):
                c1, c2, c3 = st.columns(3)
                c1.metric("Position Size", f"{r['position_size']:.4%}")
                c2.metric("Stop Loss", f"{r['stop_loss']:.2%}")
                c3.metric("Take Profit", f"{r['take_profit']:.2%}")


# ============================================================
# PAGE: PORTFOLIO
# ============================================================

def page_portfolio():
    st.title("💰 Paper Trading Portfolio")

    portfolio_file = ROOT / "logs" / "paper_portfolio.json"
    trades_file = ROOT / "logs" / "paper_trades.jsonl"

    if portfolio_file.exists():
        with open(portfolio_file) as f:
            state = json.load(f)

        col1, col2, col3, col4 = st.columns(4)
        equity = state.get("cash", 100000)
        initial = state.get("initial_capital", 100000)
        pnl_pct = (equity - initial) / initial

        col1.metric("💵 Equity", f"${equity:,.2f}", f"{pnl_pct:+.2%}")
        col2.metric("💰 Cash", f"${state.get('cash', 0):,.2f}")
        col3.metric("📊 Scans", state.get("total_scans", 0))
        col4.metric("📡 Signals", state.get("total_signals", 0))

        st.markdown("---")

        # Open positions
        positions = state.get("positions", {})
        if positions:
            st.subheader(f"📈 Open Positions ({len(positions)})")
            pos_data = []
            for sym, pos in positions.items():
                pnl = pos.get("unrealized_pnl", 0)
                pos_data.append({
                    "Symbol": sym,
                    "Side": pos["side"],
                    "Entry": f"${pos['entry_price']:,.2f}",
                    "Current": f"${pos['current_price']:,.2f}",
                    "P&L": f"${pnl:+,.2f}",
                    "P&L%": f"{pos.get('unrealized_pnl_pct', 0):+.2%}",
                })
            st.dataframe(pd.DataFrame(pos_data), use_container_width=True, hide_index=True)
        else:
            st.info("No open positions. The system is waiting for strong consensus signals (>0.60).")

        # Stats
        st.markdown("---")
        st.subheader("📊 Performance Stats")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Realized P&L", f"${state.get('realized_pnl', 0):,.2f}")
        c2.metric("Win Rate", f"{state.get('win_rate', 0):.1%}")
        c3.metric("Closed Trades", state.get("closed_trades_count", 0))
        c4.metric("Peak Equity", f"${state.get('peak_equity', 100000):,.2f}")

    else:
        st.info("No portfolio data yet. Run a paper trading scan first:\n\n"
                "`python trading/paper_trader.py --once`")

    # Trade history
    st.markdown("---")
    st.subheader("📜 Trade History")

    if trades_file.exists():
        trades = []
        with open(trades_file) as f:
            for line in f:
                if line.strip():
                    trades.append(json.loads(line))

        if trades:
            df = pd.DataFrame(trades)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            st.dataframe(
                df[["timestamp", "event", "symbol", "side", "price", "pnl", "reason", "equity"]]
                .sort_values("timestamp", ascending=False),
                use_container_width=True,
                hide_index=True,
            )

            # Equity curve
            equity_data = df[df["equity"] > 0][["timestamp", "equity"]].copy()
            if len(equity_data) > 1:
                st.subheader("📈 Equity Curve")
                st.line_chart(equity_data.set_index("timestamp")["equity"])
        else:
            st.info("No trades recorded yet.")
    else:
        st.info("No trade history file found.")


# ============================================================
# PAGE: DECISION LOG
# ============================================================

def page_decision_log():
    st.title("📜 Brain Decision Log")

    log_file = ROOT / "logs" / "decisions.jsonl"

    if not log_file.exists():
        st.info("No decisions logged yet.")
        return

    decisions = []
    with open(log_file) as f:
        for line in f:
            if line.strip():
                try:
                    decisions.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not decisions:
        st.info("No decisions logged yet.")
        return

    st.metric("Total Decisions", len(decisions))

    # Filters
    col1, col2 = st.columns(2)
    symbols = list(set(d.get("symbol", "") for d in decisions))
    actions = list(set(d.get("action", "") for d in decisions))

    with col1:
        filter_sym = st.multiselect("Filter by Symbol", symbols, default=symbols)
    with col2:
        filter_act = st.multiselect("Filter by Action", actions, default=actions)

    filtered = [d for d in decisions
                if d.get("symbol") in filter_sym and d.get("action") in filter_act]

    # Table
    table = []
    for d in filtered[-50:]:  # Last 50
        emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⚪"}.get(d.get("action", ""), "❓")
        table.append({
            "Time": d.get("timestamp", "")[:19],
            "Symbol": d.get("symbol", ""),
            "Signal": f"{emoji} {d.get('action', '')}",
            "Consensus": f"{d.get('consensus_score', 0):+.4f}",
            "Approved": "✅" if d.get("approved") else "❌",
        })

    df = pd.DataFrame(table)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Consensus distribution
    st.subheader("📊 Consensus Distribution")
    consensus_values = [d.get("consensus_score", 0) for d in filtered]
    if consensus_values:
        hist_df = pd.DataFrame({"consensus": consensus_values})
        st.bar_chart(hist_df["consensus"].value_counts(bins=20).sort_index())


# ============================================================
# PAGE: AGENT STATUS
# ============================================================

def page_agent_status():
    st.title("🤖 Agent Status")

    brain, _ = init_system()

    agents_info = [
        {
            "name": "Knowledge Advisor",
            "icon": "📚",
            "role": "VOTER",
            "weight": "25%",
            "tech": "ChromaDB + RAG + Ollama (qwen3:8b)",
            "details": "140,512 chunks from 320+ trading books",
        },
        {
            "name": "ML Signal Engine",
            "icon": "🤖",
            "role": "VOTER",
            "weight": "35%",
            "tech": "CatBoost Ensemble (2 models)",
            "details": "60 technical features, binary classification",
        },
        {
            "name": "News Analyzer",
            "icon": "📰",
            "role": "VOTER",
            "weight": "15%",
            "tech": "RSS feeds + LLM Sentiment (qwen3:8b)",
            "details": "Yahoo Finance, Google News RSS",
        },
        {
            "name": "Security Guard",
            "icon": "🛡️",
            "role": "VOTER",
            "weight": "5%",
            "tech": "Statistical Anomaly Detection",
            "details": "Volume spikes, price gaps, volatility explosions",
        },
        {
            "name": "Risk Manager",
            "icon": "⚖️",
            "role": "GATEKEEPER",
            "weight": "20%",
            "tech": "Quarter-Kelly Criterion",
            "details": "Position sizing, SL/TP, max drawdown limits",
        },
    ]

    for agent in agents_info:
        role_color = "🟢" if agent["role"] == "VOTER" else "🔶"
        with st.expander(f"{agent['icon']} **{agent['name']}** — {role_color} {agent['role']} ({agent['weight']})"):
            st.markdown(f"**Technology:** {agent['tech']}")
            st.markdown(f"**Details:** {agent['details']}")
            st.markdown(f"**Weight:** {agent['weight']}")
            st.markdown(f"**Role:** {agent['role']}")

    st.markdown("---")
    st.subheader("⚙️ System Configuration")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Brain Settings**")
        st.code("""
Consensus Threshold: 0.60
Non-Voting Agents: risk_manager
Decision Log: logs/decisions.jsonl
        """)

    with col2:
        st.markdown("**Risk Limits**")
        st.code("""
Max Position: 2% of portfolio
Max Portfolio Risk: 6%
Max Drawdown: 15%
Kelly Fraction: 0.25 (Quarter-Kelly)
Stop Loss: 2% default
Take Profit: 4% default
        """)


# ============================================================
# PAGE: LIVE CHART
# ============================================================

def page_live_chart():
    st.title("📈 Live Chart & Technical Analysis")

    from dashboard.chart_engine import build_chart

    brain, fetcher = init_system()

    # Symbol selection
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        symbol = st.selectbox("Symbol", [
            "AAPL", "NVDA", "MSFT", "GOOGL", "AMZN", "META", "TSLA",
            "AMD", "NFLX", "CRM", "BTCUSDT", "ETHUSDT",
            "GOLD", "SILVER", "SPY", "QQQ",
        ], index=0)
    with col2:
        period = st.selectbox("Period", ["30d", "60d", "90d", "6mo", "1y"], index=1)
    with col3:
        interval = st.selectbox("Interval", ["15m", "30m", "1h", "4h", "1d"], index=2)

    # TA Indicators toolbar
    st.markdown("**🛠️ Technical Analysis Tools:**")
    indicators = st.multiselect(
        "Select indicators to overlay",
        ["Moving Averages", "Bollinger Bands", "Ichimoku", "MACD", "RSI",
         "Fibonacci", "Gann", "Volume"],
        default=["Moving Averages", "Volume"],
        label_visibility="collapsed",
    )

    chart_type = st.radio("Chart Type", ["candlestick", "line"],
                          horizontal=True, label_visibility="collapsed")

    # Fetch data
    with st.spinner(f"Loading {symbol}..."):
        features_df = fetcher.fetch_and_engineer(symbol, period=period, interval=interval)

    if features_df is None or features_df.empty:
        st.error(f"No data for {symbol}")
        return

    # Price info bar
    last = features_df.iloc[-1]
    prev = features_df.iloc[-2] if len(features_df) > 1 else last
    price = float(last["Close"])
    change = price - float(prev["Close"])
    change_pct = change / float(prev["Close"]) if float(prev["Close"]) > 0 else 0

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Price", f"${price:,.2f}", f"{change:+,.2f} ({change_pct:+.2%})")
    col2.metric("High", f"${float(last['High']):,.2f}")
    col3.metric("Low", f"${float(last['Low']):,.2f}")
    col4.metric("Volume", f"{int(float(last['Volume'])):,}")
    rsi_val = float(last.get('RSI', last.get('rsi', 0)))
    col5.metric("RSI", f"{rsi_val:.1f}",
                delta="Overbought" if rsi_val > 70 else "Oversold" if rsi_val < 30 else "Neutral")

    # Build and display chart
    fig = build_chart(features_df, symbol, indicators, chart_type)
    st.plotly_chart(fig, use_container_width=True)

    # Quick Brain analysis
    st.markdown("---")
    if st.button("🧠 Run Brain Analysis", type="primary"):
        with st.spinner("All 5 agents analyzing..."):
            latest = fetcher.get_latest_features(symbol)
            if latest is not None:
                decision = run_async(brain.decide(symbol, {
                    "timeframe": interval,
                    "features": latest.tolist(),
                    "features_df": features_df,
                }))

                emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⚪"}.get(decision.action, "❓")
                st.markdown(f"### {emoji} {decision.action} — Consensus: {decision.consensus_score:+.4f}")

                if decision.agent_votes:
                    for vote in decision.agent_votes:
                        v_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⚪"}.get(
                            vote.get("recommendation", ""), "❓")
                        st.markdown(
                            f'{v_emoji} **{vote.get("agent_name", "")}**: '
                            f'{vote.get("recommendation", "")} '
                            f'(conf: {vote.get("confidence", 0):.2f})'
                        )


# ============================================================
# PAGE: QUICK TRADE
# ============================================================

def page_quick_trade():
    st.title("⚡ Quick Trade Panel")

    from dashboard.chart_engine import build_chart

    brain, fetcher = init_system()

    # Symbol selection
    col1, col2 = st.columns([3, 1])
    with col1:
        symbol = st.selectbox("Select Asset", [
            "AAPL", "NVDA", "MSFT", "GOOGL", "AMZN", "META", "TSLA",
            "BTCUSDT", "ETHUSDT", "GOLD", "SPY", "QQQ",
        ], key="qt_symbol")
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        refresh = st.button("🔄 Refresh", use_container_width=True)

    # Fetch live data
    features_df = fetcher.fetch_and_engineer(symbol, period="30d", interval="1h")
    if features_df is None or features_df.empty:
        st.error(f"No data for {symbol}")
        return

    price = float(features_df["Close"].iloc[-1])

    # Mini chart
    fig = build_chart(features_df.tail(100), symbol, ["Moving Averages", "Volume"], "candlestick")
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Brain signal
    latest = fetcher.get_latest_features(symbol)
    brain_signal = "HOLD"
    consensus = 0.0
    if latest is not None:
        decision = run_async(brain.decide(symbol, {
            "timeframe": "1h",
            "features": latest.tolist(),
            "features_df": features_df,
        }))
        brain_signal = decision.action
        consensus = decision.consensus_score

    signal_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⚪"}.get(brain_signal, "❓")

    st.markdown(f"### {signal_emoji} Brain Signal: **{brain_signal}** "
                f"(consensus: {consensus:+.4f})")

    st.markdown("---")

    # Trade controls
    st.subheader(f"💱 Trade {symbol} — ${price:,.2f}")

    col1, col2, col3 = st.columns(3)
    with col1:
        trade_type = st.radio("Side", ["BUY (Long)", "SELL (Short)"], horizontal=True)
    with col2:
        amount_usd = st.number_input("Amount ($)", min_value=100, max_value=50000,
                                      value=2000, step=100)
    with col3:
        quantity = amount_usd / price if price > 0 else 0
        st.metric("Quantity", f"{quantity:.4f}")

    # Risk parameters
    col1, col2, col3 = st.columns(3)
    with col1:
        sl_pct = st.slider("Stop Loss %", 0.5, 10.0, 2.0, 0.5)
    with col2:
        tp_pct = st.slider("Take Profit %", 0.5, 20.0, 4.0, 0.5)
    with col3:
        sl_price = price * (1 - sl_pct/100) if "BUY" in trade_type else price * (1 + sl_pct/100)
        tp_price = price * (1 + tp_pct/100) if "BUY" in trade_type else price * (1 - tp_pct/100)
        st.metric("SL Price", f"${sl_price:,.2f}")
        st.metric("TP Price", f"${tp_price:,.2f}")

    # Risk/Reward display
    risk_amount = amount_usd * sl_pct / 100
    reward_amount = amount_usd * tp_pct / 100
    rr_ratio = tp_pct / sl_pct if sl_pct > 0 else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("⚠️ Risk", f"${risk_amount:,.2f}")
    col2.metric("🎯 Reward", f"${reward_amount:,.2f}")
    col3.metric("📊 R:R Ratio", f"1:{rr_ratio:.1f}")

    st.markdown("---")

    # Execute buttons
    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        pass
    with col2:
        side = "LONG" if "BUY" in trade_type else "SHORT"
        action = "BUY" if "BUY" in trade_type else "SELL"

        if st.button(f"🚀 Execute {action} — ${amount_usd:,.0f}",
                     type="primary", use_container_width=True):

            # Execute paper trade
            portfolio_file = ROOT / "logs" / "paper_portfolio.json"
            trade_log = ROOT / "logs" / "paper_trades.jsonl"

            import json
            from datetime import datetime as dt

            # Load portfolio
            cash = 100_000.0
            if portfolio_file.exists():
                with open(portfolio_file) as f:
                    state = json.load(f)
                cash = state.get("cash", 100_000.0)

            if amount_usd > cash:
                st.error(f"Insufficient cash! Available: ${cash:,.2f}")
            else:
                # Log the trade
                trade_log.parent.mkdir(parents=True, exist_ok=True)
                trade_entry = {
                    "timestamp": dt.utcnow().isoformat(),
                    "event": "OPEN",
                    "symbol": symbol,
                    "side": side,
                    "price": price,
                    "quantity": quantity,
                    "consensus": consensus,
                    "pnl": 0,
                    "reason": "quick_trade",
                    "equity": cash - amount_usd if side == "LONG" else cash,
                    "stop_loss": sl_pct / 100,
                    "take_profit": tp_pct / 100,
                    "amount_usd": amount_usd,
                }
                with open(trade_log, "a") as f:
                    f.write(json.dumps(trade_entry) + "\n")

                # Update portfolio
                state = {}
                if portfolio_file.exists():
                    with open(portfolio_file) as f:
                        state = json.load(f)

                new_cash = cash - amount_usd if side == "LONG" else cash
                positions = state.get("positions", {})
                positions[symbol] = {
                    "symbol": symbol,
                    "side": side,
                    "entry_price": price,
                    "quantity": quantity,
                    "entry_time": dt.utcnow().isoformat(),
                    "stop_loss": sl_pct / 100,
                    "take_profit": tp_pct / 100,
                    "consensus_at_entry": consensus,
                    "current_price": price,
                    "unrealized_pnl": 0,
                    "unrealized_pnl_pct": 0,
                }

                state.update({
                    "saved_at": dt.utcnow().isoformat(),
                    "cash": new_cash,
                    "positions": positions,
                    "total_signals": state.get("total_signals", 0) + 1,
                })

                with open(portfolio_file, "w") as f:
                    json.dump(state, f, indent=2)

                st.success(f"✅ {action} {symbol}: {quantity:.4f} units @ ${price:,.2f} "
                           f"(${amount_usd:,.0f}) | SL: ${sl_price:,.2f} | TP: ${tp_price:,.2f}")
                st.balloons()

    with col3:
        pass

    # Disclaimer
    st.markdown("---")
    st.caption("⚠️ **PAPER TRADING ONLY** — No real money is at risk. "
               "All trades are simulated for testing and development purposes.")


# ============================================================
# ROUTER
# ============================================================

if page == "📊 Live Scanner":
    page_live_scanner()
elif page == "📈 Live Chart":
    page_live_chart()
elif page == "⚡ Quick Trade":
    page_quick_trade()
elif page == "💰 Portfolio":
    page_portfolio()
elif page == "📜 Decision Log":
    page_decision_log()
elif page == "🤖 Agent Status":
    page_agent_status()
