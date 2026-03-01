"""
SCRIPT NAME: orchestrator.py
====================================
Execution Location: K:\_DEV_MVP_2026\Market_Hawk_3\brain\
Purpose: THE BRAIN — Central Orchestrator for Multi-Agent Trading System
Hardware Optimization: Intel i7-9700F, NVIDIA GTX 1070 8GB VRAM, 64GB DDR4
Creation Date: 2026-03-01

The Brain coordinates all agents, collects their recommendations,
applies weighted consensus, and makes final trading decisions.
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path

logger = logging.getLogger("market_hawk.brain")


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class AgentResponse:
    """Response from an individual agent."""
    agent_name: str
    recommendation: str          # BUY, SELL, HOLD
    confidence: float            # 0.0 to 1.0
    reasoning: str
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class BrainDecision:
    """Final decision made by the Brain."""
    timestamp: str
    symbol: str
    action: str                  # BUY, SELL, HOLD
    consensus_score: float
    approved: bool
    position_size: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    agent_votes: List[Dict] = None
    reasoning: str = ""
    risk_check: Dict = None

    def __post_init__(self):
        if self.agent_votes is None:
            self.agent_votes = []
        if self.risk_check is None:
            self.risk_check = {}


# ============================================================
# THE BRAIN
# ============================================================

class Brain:
    """
    Central Orchestrator — THE BRAIN of Market Hawk.

    Coordinates specialized agents, collects their recommendations,
    applies weighted consensus voting, and makes final trading decisions
    with full audit trail.

    Usage:
        brain = Brain()
        brain.register_agent("knowledge_advisor", advisor_instance)
        brain.register_agent("ml_signal_engine", ml_engine_instance)
        decision = await brain.decide("AAPL")
    """

    def __init__(self, agent_configs: Dict = None, consensus_threshold: float = None):
        from config.settings import AGENT_CONFIGS, CONSENSUS_THRESHOLD, DECISION_LOG

        self.agent_configs = agent_configs or AGENT_CONFIGS
        self.consensus_threshold = consensus_threshold or CONSENSUS_THRESHOLD
        self.decision_log_path = DECISION_LOG
        self.agents = {}

        logger.info("Brain initialized | threshold=%.2f", self.consensus_threshold)

    def register_agent(self, agent_id: str, agent_instance: Any) -> None:
        """Register an agent with the Brain."""
        self.agents[agent_id] = agent_instance
        logger.info("Agent registered: %s (total: %d)", agent_id, len(self.agents))

    def unregister_agent(self, agent_id: str) -> None:
        """Remove an agent from the Brain."""
        if agent_id in self.agents:
            del self.agents[agent_id]
            logger.info("Agent unregistered: %s", agent_id)

    async def query_agents(self, symbol: str, context: Dict = None) -> List[AgentResponse]:
        """Query all registered agents for their recommendations."""
        responses = []
        context = context or {}

        for agent_id, agent in self.agents.items():
            try:
                # Try different interfaces (agents may use different method names)
                if hasattr(agent, 'analyze'):
                    response = await agent.analyze(symbol, context) if asyncio_compat(agent.analyze) else agent.analyze(symbol, context)
                elif hasattr(agent, 'predict'):
                    response = agent.predict(symbol, context)
                elif hasattr(agent, 'query'):
                    response = agent.query(f"Trading analysis for {symbol}", context)
                elif hasattr(agent, 'evaluate'):
                    response = agent.evaluate({"symbol": symbol, **context})
                else:
                    logger.warning("Agent %s has no callable analysis method", agent_id)
                    continue

                if isinstance(response, AgentResponse):
                    responses.append(response)
                elif isinstance(response, dict):
                    responses.append(AgentResponse(
                        agent_name=agent_id,
                        recommendation=response.get("recommendation", "HOLD"),
                        confidence=response.get("confidence", 0.0),
                        reasoning=response.get("reasoning", ""),
                        metadata=response.get("metadata", {})
                    ))

            except Exception as e:
                logger.error("Agent %s failed: %s", agent_id, str(e))
                responses.append(AgentResponse(
                    agent_name=agent_id,
                    recommendation="HOLD",
                    confidence=0.0,
                    reasoning=f"Agent error: {str(e)}"
                ))

        return responses

    def calculate_consensus(self, responses: List[AgentResponse]) -> float:
        """
        Calculate weighted consensus score.
        Direction: BUY=+1, SELL=-1, HOLD=0
        Returns score between -1.0 and 1.0
        """
        direction_map = {"BUY": 1.0, "SELL": -1.0, "HOLD": 0.0}
        total_weight = 0.0
        weighted_sum = 0.0

        for response in responses:
            config = self.agent_configs.get(response.agent_name)
            weight = config.weight if config else 0.1
            direction = direction_map.get(response.recommendation.upper(), 0.0)

            weighted_sum += weight * response.confidence * direction
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    async def decide(self, symbol: str, context: Dict = None) -> BrainDecision:
        """
        Full decision pipeline: Query agents → Consensus → Risk check → Log.
        """
        logger.info("=== BRAIN DECISION START: %s ===", symbol)

        # 1. Query all agents
        responses = await self.query_agents(symbol, context)

        # 2. Calculate consensus
        consensus = self.calculate_consensus(responses)

        # 3. Determine action
        if abs(consensus) >= self.consensus_threshold:
            action = "BUY" if consensus > 0 else "SELL"
        else:
            action = "HOLD"

        # 4. Risk check
        risk_check = {}
        approved = True
        position_size = stop_loss = take_profit = None

        if "risk_manager" in self.agents:
            try:
                risk_result = self.agents["risk_manager"].evaluate({
                    "symbol": symbol, "action": action, "consensus": consensus,
                })
                risk_check = risk_result if isinstance(risk_result, dict) else {}
                approved = risk_check.get("approved", True)
                position_size = risk_check.get("position_size")
                stop_loss = risk_check.get("stop_loss")
                take_profit = risk_check.get("take_profit")
            except Exception as e:
                logger.error("Risk Manager failed: %s", str(e))

        # 5. Build decision
        decision = BrainDecision(
            timestamp=datetime.utcnow().isoformat(),
            symbol=symbol,
            action=action if approved else "HOLD",
            consensus_score=consensus,
            approved=approved,
            position_size=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            agent_votes=[asdict(r) for r in responses],
            reasoning=self._build_reasoning(responses, consensus, approved),
            risk_check=risk_check,
        )

        # 6. Log
        self._log_decision(decision)

        logger.info("=== DECISION: %s %s (consensus=%.4f, approved=%s) ===",
                     decision.action, symbol, consensus, approved)

        return decision

    def _build_reasoning(self, responses, consensus, approved):
        parts = [f"Consensus: {consensus:.4f}"]
        for r in responses:
            parts.append(f"  {r.agent_name}: {r.recommendation} "
                         f"(conf={r.confidence:.2f}) — {r.reasoning[:100]}")
        if not approved:
            parts.append("  ⚠️ BLOCKED by Risk Manager")
        return "\n".join(parts)

    def _log_decision(self, decision: BrainDecision) -> None:
        try:
            self.decision_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.decision_log_path, "a") as f:
                f.write(json.dumps(asdict(decision)) + "\n")
        except Exception as e:
            logger.error("Failed to log decision: %s", str(e))


def asyncio_compat(func):
    """Check if a function is a coroutine."""
    import asyncio
    return asyncio.iscoroutinefunction(func)
