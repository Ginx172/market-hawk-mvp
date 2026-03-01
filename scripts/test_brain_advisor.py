"""
Quick test for Brain + Knowledge Advisor integration.
Run from: K:\\_DEV_MVP_2026\\Market_Hawk_3\\

Usage:
    python scripts/test_brain_advisor.py
"""

import sys
import asyncio
import logging
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)

async def main():
    print("\n" + "=" * 60)
    print("MARKET HAWK MVP — Brain + Knowledge Advisor Integration Test")
    print("=" * 60)

    # 1. Create Brain
    print("\n[1/3] Creating Brain...")
    from brain.orchestrator import Brain
    brain = Brain()
    print("  ✅ Brain created")

    # 2. Register Knowledge Advisor
    print("\n[2/3] Registering Knowledge Advisor...")
    from agents.knowledge_advisor.rag_engine import KnowledgeAdvisor
    advisor = KnowledgeAdvisor()

    if advisor.initialize():
        brain.register_agent("knowledge_advisor", advisor)
        print("  ✅ Knowledge Advisor registered with Brain")
    else:
        print("  ⚠️  Advisor init failed — Brain will work without it")

    # 3. Test Brain decision (with only Knowledge Advisor)
    print("\n[3/3] Testing Brain decision for AAPL...")
    decision = await brain.decide("AAPL", {"timeframe": "4h"})
    print(f"\n  Decision: {decision.action}")
    print(f"  Consensus: {decision.consensus_score:.4f}")
    print(f"  Approved: {decision.approved}")
    print(f"  Reasoning: {decision.reasoning[:300]}")

    if decision.agent_votes:
        print(f"\n  Agent Votes:")
        for vote in decision.agent_votes:
            print(f"    - {vote['agent_name']}: {vote['recommendation']} "
                  f"(conf={vote['confidence']:.2f})")

    print("\n" + "=" * 60)
    print("INTEGRATION TEST COMPLETE")
    print("=" * 60)

    advisor.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
