"""Tests for Brain orchestrator weighted consensus and stub exclusion."""

import asyncio
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any

import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from brain.orchestrator import (
    Brain, AgentResponse, BrainDecision,
    AgentCategory, AGENT_CATEGORY_MAP, CATEGORY_WEIGHT_MULTIPLIER,
)
from config.settings import AgentConfig


# ============================================================
# MOCK AGENTS
# ============================================================

class MockMLAgent:
    """Simulates ml_signal_engine — real ML agent."""
    def __init__(self, recommendation: str = "BUY", confidence: float = 0.8):
        self._rec = recommendation
        self._conf = confidence

    def analyze(self, symbol: str, context: Dict = None) -> Dict:
        return {
            "recommendation": self._rec,
            "confidence": self._conf,
            "reasoning": f"ML model predicts {self._rec} for {symbol}",
            "metadata": {"model": "catboost_v2"},
        }


class MockHeuristicAgent:
    """Simulates a heuristic agent (news_analyzer, etc.)."""
    def __init__(self, recommendation: str = "HOLD", confidence: float = 0.5):
        self._rec = recommendation
        self._conf = confidence

    def analyze(self, symbol: str, context: Dict = None) -> Dict:
        return {
            "recommendation": self._rec,
            "confidence": self._conf,
            "reasoning": f"Heuristic says {self._rec}",
        }


class MockStubAgent:
    """Simulates a stub agent that returns NOT_IMPLEMENTED."""
    def analyze(self, symbol: str, context: Dict = None) -> Dict:
        return {
            "status": "NOT_IMPLEMENTED",
            "agent": "mock_stub",
            "recommendation": "HOLD",
            "confidence": 0.0,
            "reasoning": "This agent is a stub",
        }


class MockBrokenAgent:
    """Agent that raises an exception."""
    def analyze(self, symbol: str, context: Dict = None) -> Dict:
        raise RuntimeError("Agent exploded")


class MockNoMethodAgent:
    """Agent with no analyze/predict/query method."""
    pass


class MockRiskManager:
    """Simulates risk_manager — always approves."""
    def evaluate(self, signal: Dict) -> Dict:
        return {
            "approved": True,
            "position_size": 0.02,
            "stop_loss": 0.05,
            "take_profit": 0.10,
        }


# ============================================================
# HELPERS
# ============================================================

def _make_brain(agents: Dict[str, Any], configs: Dict[str, AgentConfig] = None) -> Brain:
    """Create a Brain with mock agents and minimal config."""
    default_configs = {
        "ml_signal_engine": AgentConfig(name="ML Signal Engine", weight=0.35),
        "knowledge_advisor": AgentConfig(name="Knowledge Advisor", weight=0.25),
        "news_analyzer": AgentConfig(name="News Analyzer", weight=0.15),
        "security_guard": AgentConfig(name="Security Guard", weight=0.05),
        "continuous_learner": AgentConfig(name="Continuous Learner", weight=0.10),
        "risk_manager": AgentConfig(name="Risk Manager", weight=0.20),
    }
    brain = Brain(agent_configs=configs or default_configs, consensus_threshold=0.6)
    for agent_id, agent_instance in agents.items():
        brain.register_agent(agent_id, agent_instance)
    return brain


def _run(coro):
    """Run async coroutine synchronously."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ============================================================
# TESTS
# ============================================================

class TestAgentCategory:
    def test_ml_agent_has_correct_category(self):
        assert AGENT_CATEGORY_MAP["ml_signal_engine"] == AgentCategory.ML_MODEL

    def test_heuristic_agents_have_correct_category(self):
        assert AGENT_CATEGORY_MAP["knowledge_advisor"] == AgentCategory.HEURISTIC
        assert AGENT_CATEGORY_MAP["news_analyzer"] == AgentCategory.HEURISTIC
        assert AGENT_CATEGORY_MAP["security_guard"] == AgentCategory.HEURISTIC

    def test_stub_agent_has_correct_category(self):
        assert AGENT_CATEGORY_MAP["continuous_learner"] == AgentCategory.STUB

    def test_weight_multipliers(self):
        assert CATEGORY_WEIGHT_MULTIPLIER[AgentCategory.ML_MODEL] == 2.0
        assert CATEGORY_WEIGHT_MULTIPLIER[AgentCategory.HEURISTIC] == 1.0
        assert CATEGORY_WEIGHT_MULTIPLIER[AgentCategory.STUB] == 0.0


class TestStubExclusion:
    def test_stub_agent_excluded_from_consensus(self):
        brain = _make_brain({
            "ml_signal_engine": MockMLAgent("BUY", 0.9),
            "continuous_learner": MockStubAgent(),
        })
        responses, active, excluded = _run(brain.query_agents("AAPL"))

        assert "ml_signal_engine" in active
        assert "continuous_learner" not in active
        assert any(e["agent"] == "continuous_learner" for e in excluded)
        assert len(responses) == 1
        assert responses[0].agent_name == "ml_signal_engine"

    def test_stub_exclusion_reason_contains_not_implemented(self):
        brain = _make_brain({"continuous_learner": MockStubAgent()})
        _, _, excluded = _run(brain.query_agents("AAPL"))

        exc = next(e for e in excluded if e["agent"] == "continuous_learner")
        assert "NOT_IMPLEMENTED" in exc["reason"]


class TestBrokenAgentExclusion:
    def test_broken_agent_excluded_with_error_reason(self):
        brain = _make_brain({
            "ml_signal_engine": MockMLAgent("BUY", 0.9),
            "news_analyzer": MockBrokenAgent(),
        })
        responses, active, excluded = _run(brain.query_agents("AAPL"))

        assert "ml_signal_engine" in active
        assert "news_analyzer" not in active
        assert any(e["agent"] == "news_analyzer" for e in excluded)
        assert len(responses) == 1

    def test_no_method_agent_excluded(self):
        brain = _make_brain({
            "ml_signal_engine": MockMLAgent("BUY", 0.9),
            "news_analyzer": MockNoMethodAgent(),
        })
        responses, active, excluded = _run(brain.query_agents("AAPL"))

        assert "news_analyzer" not in active
        assert any(e["agent"] == "news_analyzer" for e in excluded)


class TestWeightedConsensus:
    def test_ml_agent_has_more_influence(self):
        """ML agent (weight=0.35, x2=0.70) should outweigh
        heuristic (weight=0.15, x1=0.15) when they disagree."""
        brain = _make_brain({
            "ml_signal_engine": MockMLAgent("BUY", 0.9),
            "news_analyzer": MockHeuristicAgent("SELL", 0.9),
        })
        responses, _, _ = _run(brain.query_agents("AAPL"))
        consensus = brain.calculate_consensus(responses)

        # ML BUY should dominate: 0.70*0.9*1 vs 0.15*0.9*(-1)
        assert consensus > 0, f"ML BUY should dominate, got consensus={consensus}"

    def test_consensus_zero_with_no_agents(self):
        brain = _make_brain({})
        consensus = brain.calculate_consensus([])
        assert consensus == 0.0

    def test_all_stubs_yield_zero_consensus(self):
        brain = _make_brain({
            "continuous_learner": MockStubAgent(),
        })
        responses, active, excluded = _run(brain.query_agents("AAPL"))
        consensus = brain.calculate_consensus(responses)

        assert consensus == 0.0
        assert len(active) == 0
        assert len(excluded) == 1


class TestBrainDecision:
    def test_decision_contains_active_and_excluded(self):
        brain = _make_brain({
            "ml_signal_engine": MockMLAgent("BUY", 0.9),
            "continuous_learner": MockStubAgent(),
            "risk_manager": MockRiskManager(),
        })
        decision = _run(brain.decide("AAPL"))

        assert isinstance(decision, BrainDecision)
        assert "ml_signal_engine" in decision.active_agents
        assert "continuous_learner" not in decision.active_agents
        assert any(e["agent"] == "continuous_learner"
                   for e in decision.excluded_agents)

    def test_decision_reasoning_includes_categories(self):
        brain = _make_brain({
            "ml_signal_engine": MockMLAgent("BUY", 0.9),
            "news_analyzer": MockHeuristicAgent("BUY", 0.7),
        })
        decision = _run(brain.decide("AAPL"))

        assert "ml_model" in decision.reasoning
        assert "heuristic" in decision.reasoning

    def test_risk_manager_not_in_active_or_excluded(self):
        """Risk manager is a gatekeeper, not a voter."""
        brain = _make_brain({
            "ml_signal_engine": MockMLAgent("BUY", 0.9),
            "risk_manager": MockRiskManager(),
        })
        decision = _run(brain.decide("AAPL"))

        assert "risk_manager" not in decision.active_agents
        assert not any(e["agent"] == "risk_manager"
                       for e in decision.excluded_agents)

    def test_all_agents_stub_results_in_hold(self):
        brain = _make_brain({
            "continuous_learner": MockStubAgent(),
        })
        decision = _run(brain.decide("AAPL"))

        assert decision.action == "HOLD"
        assert decision.consensus_score == 0.0
        assert len(decision.active_agents) == 0


class TestContinuousLearnerStub:
    def test_continuous_learner_returns_not_implemented(self):
        from agents.continuous_learner import ContinuousLearner
        learner = ContinuousLearner()
        result = learner.analyze("AAPL")

        assert result["status"] == "NOT_IMPLEMENTED"
        assert result["agent"] == "continuous_learner"
        assert result["recommendation"] == "HOLD"
        assert result["confidence"] == 0.0
