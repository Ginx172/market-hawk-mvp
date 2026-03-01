"""
MARKET HAWK MVP — FULL END-TO-END: 5 Agents + Live Data + Brain Decision
Run from: K:\\_DEV_MVP_2026\\Market_Hawk_3\\

Usage:
    python scripts/test_end_to_end.py
"""

import sys
import asyncio
import logging
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)-30s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)
for noisy in ["httpx", "httpcore", "chromadb", "urllib3", "yfinance", "peewee"]:
    logging.getLogger(noisy).setLevel(logging.WARNING)


async def main():
    print("\n" + "=" * 70)
    print("  🦅 MARKET HAWK MVP — FULL 5-AGENT LIVE DECISION")
    print("=" * 70)

    # ================================================================
    # INITIALIZE ALL 5 AGENTS + BRAIN
    # ================================================================
    print("\n[1/6] Initializing Brain...")
    from brain.orchestrator import Brain
    brain = Brain()

    print("\n[2/6] Knowledge Advisor (140K chunks RAG)...")
    from agents.knowledge_advisor.rag_engine import KnowledgeAdvisor
    advisor = KnowledgeAdvisor()
    if advisor.initialize():
        brain.register_agent("knowledge_advisor", advisor)
    else:
        print("  ⚠️  Advisor failed")

    print("\n[3/6] ML Signal Engine (CatBoost ensemble)...")
    from agents.ml_signal_engine.catboost_predictor import MLSignalEngine
    ml_engine = MLSignalEngine()
    models_loaded = 0
    for model_name in ["catboost_v2", "catboost_clean_75"]:
        if ml_engine.load_model(model_name):
            models_loaded += 1
    if models_loaded > 0:
        brain.register_agent("ml_signal_engine", ml_engine)

    print("\n[4/6] News Analyzer (sentiment)...")
    from agents.news_analyzer.news_sentiment import NewsAnalyzer
    news = NewsAnalyzer(use_llm=True)  # LLM sentiment via qwen3:8b
    brain.register_agent("news_analyzer", news)

    print("\n[5/6] Security Guard (anomaly detection)...")
    from agents.security_guard.anomaly_detector import SecurityGuard
    guard = SecurityGuard()
    brain.register_agent("security_guard", guard)

    print("\n[6/6] Data Pipeline...")
    from data.market_data_fetcher import MarketDataFetcher, get_symbol_category
    fetcher = MarketDataFetcher()

    # ================================================================
    # LIVE DECISIONS — ALL AGENTS VOTING
    # ================================================================
    test_symbols = ["AAPL", "BTCUSDT", "NVDA", "GOLD", "MSFT"]

    print("\n" + "=" * 70)
    print("  📊 LIVE TRADING DECISIONS — 5 AGENTS")
    print("=" * 70)

    decisions_summary = []

    for symbol in test_symbols:
        print(f"\n{'━' * 70}")
        print(f"  🎯 {symbol} ({get_symbol_category(symbol)})")
        print(f"{'━' * 70}")

        # Fetch live data + features
        features_df = fetcher.fetch_and_engineer(symbol)
        if features_df is None or features_df.empty:
            print(f"  ❌ No data for {symbol}")
            continue

        latest_features = fetcher.get_latest_features(symbol)
        if latest_features is None:
            print(f"  ❌ No features for {symbol}")
            continue

        last_close = features_df["Close"].iloc[-1]
        print(f"  Price: ${last_close:,.2f} | Data: {len(features_df)} candles | Features: {len(latest_features)}")

        # Brain decision with ALL context
        decision = await brain.decide(symbol, {
            "timeframe": "1h",
            "features": latest_features.tolist(),
            "features_df": features_df,  # For Security Guard
        })

        # Display
        action_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⚪"}.get(decision.action, "❓")
        print(f"\n  {action_emoji} DECISION: {decision.action}")
        print(f"  Consensus:  {decision.consensus_score:+.4f} (threshold: 0.60)")
        print(f"  Approved:   {decision.approved}")

        if decision.position_size:
            print(f"  Position:   {decision.position_size:.4%}")
            print(f"  Stop Loss:  {decision.stop_loss:.2%}")

        if decision.agent_votes:
            print(f"\n  Agent Votes:")
            for vote in decision.agent_votes:
                v_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⚪"}.get(
                    vote['recommendation'], "❓")
                print(f"    {v_emoji} {vote['agent_name']:25s} | "
                      f"{vote['recommendation']:4s} | "
                      f"conf={vote['confidence']:.2f} | "
                      f"{vote['reasoning'][:70]}")

        decisions_summary.append({
            "symbol": symbol,
            "price": last_close,
            "action": decision.action,
            "consensus": decision.consensus_score,
        })

    # ================================================================
    # SUMMARY TABLE
    # ================================================================
    print(f"\n{'=' * 70}")
    print(f"  📋 DECISION SUMMARY")
    print(f"{'=' * 70}")
    print(f"  {'Symbol':<10} {'Price':>12} {'Action':<8} {'Consensus':>10}")
    print(f"  {'─'*10} {'─'*12} {'─'*8} {'─'*10}")
    for d in decisions_summary:
        emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⚪"}.get(d["action"], "❓")
        print(f"  {d['symbol']:<10} ${d['price']:>10,.2f} {emoji}{d['action']:<7} {d['consensus']:>+10.4f}")

    print(f"\n  Agents: {len(brain.agents)} | ML Models: {models_loaded}")
    print(f"  Decision log: {brain.decision_log_path}")
    print(f"{'=' * 70}")

    # Cleanup
    advisor.cleanup()
    ml_engine.cleanup()
    news.cleanup()
    guard.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
