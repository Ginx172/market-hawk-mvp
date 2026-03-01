"""
MARKET HAWK MVP — END-TO-END TEST: Live Data -> Features -> ML Prediction -> Brain Decision
Run from: K:\\_DEV_MVP_2026\\Market_Hawk_3\\

This is the FIRST real decision with live market data!

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
    print("  🦅 MARKET HAWK MVP — FIRST END-TO-END LIVE DECISION")
    print("=" * 70)

    # ================================================================
    # 1. INITIALIZE ALL COMPONENTS
    # ================================================================
    print("\n[1/5] Initializing Brain...")
    from brain.orchestrator import Brain
    brain = Brain()

    print("\n[2/5] Initializing Knowledge Advisor...")
    from agents.knowledge_advisor.rag_engine import KnowledgeAdvisor
    advisor = KnowledgeAdvisor()
    if advisor.initialize():
        brain.register_agent("knowledge_advisor", advisor)
    else:
        print("  ⚠️  Advisor failed — continuing without it")

    print("\n[3/5] Initializing ML Signal Engine...")
    from agents.ml_signal_engine.catboost_predictor import MLSignalEngine
    ml_engine = MLSignalEngine()
    # Load multiple models for ensemble
    models_loaded = 0
    for model_name in ["catboost_v2", "catboost_clean_75"]:
        if ml_engine.load_model(model_name):
            models_loaded += 1
    if models_loaded > 0:
        brain.register_agent("ml_signal_engine", ml_engine)
    else:
        print("  ⚠️  No ML models loaded")

    print("\n[4/5] Initializing Risk Manager...")
    from agents.risk_manager.kelly_criterion import RiskManager
    brain.register_agent("risk_manager", RiskManager())

    print("\n[5/5] Initializing Data Pipeline...")
    from data.market_data_fetcher import MarketDataFetcher, get_symbol_category
    fetcher = MarketDataFetcher()

    # ================================================================
    # LIVE DECISIONS
    # ================================================================
    test_symbols = ["AAPL", "BTCUSDT", "NVDA", "GOLD"]

    print("\n" + "=" * 70)
    print("  📊 LIVE TRADING DECISIONS")
    print("=" * 70)

    for symbol in test_symbols:
        print(f"\n{'━' * 70}")
        print(f"  🎯 {symbol} ({get_symbol_category(symbol)})")
        print(f"{'━' * 70}")

        # Fetch live data + engineer features
        features_df = fetcher.fetch_and_engineer(symbol)
        if features_df is None or features_df.empty:
            print(f"  ❌ No data available for {symbol}")
            continue

        latest_features = fetcher.get_latest_features(symbol)
        if latest_features is None:
            print(f"  ❌ Could not compute features for {symbol}")
            continue

        last_close = features_df["Close"].iloc[-1]
        print(f"  Last Close: ${last_close:,.2f}")
        print(f"  Data points: {len(features_df)}")
        print(f"  Features: {len(latest_features)} values")

        # Brain decision with REAL features
        decision = await brain.decide(symbol, {
            "timeframe": "1h",
            "features": latest_features.tolist(),
        })

        # Display decision
        action_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⚪"}.get(decision.action, "❓")
        print(f"\n  {action_emoji} DECISION: {decision.action}")
        print(f"  Consensus:  {decision.consensus_score:+.4f}")
        print(f"  Approved:   {decision.approved}")

        if decision.position_size:
            print(f"  Position:   {decision.position_size:.4%} of portfolio")
            print(f"  Stop Loss:  {decision.stop_loss:.2%}")
            print(f"  Take Profit: {decision.take_profit:.2%}")

        if decision.agent_votes:
            print(f"\n  Agent Votes:")
            for vote in decision.agent_votes:
                v_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⚪"}.get(
                    vote['recommendation'], "❓")
                print(f"    {v_emoji} {vote['agent_name']:25s} | "
                      f"{vote['recommendation']:4s} | "
                      f"conf={vote['confidence']:.2f}")

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'=' * 70}")
    print(f"  📋 SESSION SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Agents: {len(brain.agents)}")
    print(f"  Symbols analyzed: {len(test_symbols)}")
    print(f"  ML Models loaded: {models_loaded}")
    print(f"  Knowledge base: {advisor.get_stats().get('total_chunks', 'N/A'):,} chunks")
    print(f"  Decision log: {brain.decision_log_path}")
    print(f"{'=' * 70}")

    advisor.cleanup()
    ml_engine.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
