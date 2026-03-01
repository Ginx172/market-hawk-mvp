"""
MARKET HAWK MVP — Integration Test: Brain + Knowledge Advisor + ML Signal Engine
Run from: K:\\_DEV_MVP_2026\\Market_Hawk_3\\

Usage:
    python scripts/test_brain_2agents.py
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
# Reduce noise
for noisy in ["httpx", "httpcore", "chromadb", "urllib3"]:
    logging.getLogger(noisy).setLevel(logging.WARNING)


async def main():
    print("\n" + "=" * 70)
    print("  MARKET HAWK MVP — Brain + 2 Agents Integration Test")
    print("=" * 70)

    # ================================================================
    # 1. CREATE BRAIN
    # ================================================================
    print("\n[1/5] Creating Brain...")
    from brain.orchestrator import Brain
    brain = Brain()
    print("  ✅ Brain created (consensus threshold: 0.60)")

    # ================================================================
    # 2. REGISTER KNOWLEDGE ADVISOR
    # ================================================================
    print("\n[2/5] Registering Knowledge Advisor...")
    from agents.knowledge_advisor.rag_engine import KnowledgeAdvisor
    advisor = KnowledgeAdvisor()

    if advisor.initialize():
        brain.register_agent("knowledge_advisor", advisor)
        stats = advisor.get_stats()
        print(f"  ✅ Knowledge Advisor registered — {stats['total_chunks']:,} chunks")
    else:
        print("  ⚠️  Advisor init failed (is Ollama running?)")

    # ================================================================
    # 3. REGISTER ML SIGNAL ENGINE
    # ================================================================
    print("\n[3/5] Registering ML Signal Engine...")
    from agents.ml_signal_engine.catboost_predictor import MLSignalEngine
    ml_engine = MLSignalEngine()

    # Load primary model
    if ml_engine.load_model("catboost_v2"):
        brain.register_agent("ml_signal_engine", ml_engine)
        print("  ✅ ML Signal Engine registered — catboost_v2 loaded")
    else:
        print("  ⚠️  ML Signal Engine failed to load")

    # ================================================================
    # 4. REGISTER RISK MANAGER
    # ================================================================
    print("\n[4/5] Registering Risk Manager...")
    from agents.risk_manager.kelly_criterion import RiskManager
    risk_mgr = RiskManager()
    brain.register_agent("risk_manager", risk_mgr)
    print("  ✅ Risk Manager registered (Quarter-Kelly)")

    # ================================================================
    # 5. BRAIN DECISION TEST
    # ================================================================
    print("\n[5/5] Testing Brain decision pipeline...")
    print("-" * 70)

    # Test with multiple symbols
    test_symbols = ["BTCUSDT", "AAPL", "EURUSD"]

    for symbol in test_symbols:
        print(f"\n  🧠 BRAIN DECISION: {symbol}")
        print(f"  {'─' * 50}")

        decision = await brain.decide(symbol, {
            "timeframe": "4h",
            # Note: no features provided, so ML engine will return HOLD
            # In production, features would come from data pipeline
        })

        print(f"  Action:    {decision.action}")
        print(f"  Consensus: {decision.consensus_score:.4f}")
        print(f"  Approved:  {decision.approved}")

        if decision.position_size:
            print(f"  Position:  {decision.position_size:.4%}")
            print(f"  Stop Loss: {decision.stop_loss:.2%}")
            print(f"  Take Profit: {decision.take_profit:.2%}")

        if decision.agent_votes:
            print(f"  Agent Votes:")
            for vote in decision.agent_votes:
                print(f"    → {vote['agent_name']:25s} | "
                      f"{vote['recommendation']:4s} | "
                      f"conf={vote['confidence']:.2f} | "
                      f"{vote['reasoning'][:80]}")

    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "=" * 70)
    print("  INTEGRATION TEST SUMMARY")
    print("=" * 70)
    print(f"  Agents registered: {len(brain.agents)}")
    for agent_id in brain.agents:
        print(f"    ✅ {agent_id}")
    print(f"  Symbols tested: {len(test_symbols)}")
    print(f"  Decision log: {brain.decision_log_path}")
    print("=" * 70)

    # Cleanup
    advisor.cleanup()
    ml_engine.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
