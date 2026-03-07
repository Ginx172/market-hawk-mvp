"""
SCRIPT NAME: orchestrator.py
====================================
Execution Location: market-hawk-mvp/brain/
Purpose: THE BRAIN - Central Orchestrator for Multi-Agent Trading System
Hardware Optimization: Intel i7-9700F, NVIDIA GTX 1070 8GB VRAM, 64GB DDR4
Creation Date: 2026-03-01
Last Modified: 2026-03-07
"""

import json
import asyncio
import logging
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field
from pathlib import Path

logger = logging.getLogger("market_hawk.brain")


# ============================================================
# AGENT CATEGORIES — determine weight multiplier in consensus
# ============================================================

class AgentCategory(str, Enum):
    """Agent reliability category for weighted consensus.

    ML_MODEL: Agent uses trained ML model(s) — highest reliability.
    HEURISTIC: Agent uses rule-based logic, data analysis, or LLM — medium.
    STUB: Agent is not implemented — excluded from consensus.
    """
    ML_MODEL = "ml_model"       # weight multiplier = 2
    HEURISTIC = "heuristic"     # weight multiplier = 1
    STUB = "stub"               # weight multiplier = 0 (excluded)


# Maps agent_id -> category. Agents not listed default to HEURISTIC.
AGENT_CATEGORY_MAP: Dict[str, AgentCategory] = {
    "ml_signal_engine": AgentCategory.ML_MODEL,
    "knowledge_advisor": AgentCategory.HEURISTIC,
    "news_analyzer": AgentCategory.HEURISTIC,
    "security_guard": AgentCategory.HEURISTIC,
    "continuous_learner": AgentCategory.STUB,
}

# Weight multiplier per category
CATEGORY_WEIGHT_MULTIPLIER: Dict[AgentCategory, float] = {
    AgentCategory.ML_MODEL: 2.0,
    AgentCategory.HEURISTIC: 1.0,
    AgentCategory.STUB: 0.0,
}


@dataclass
class AgentResponse:
    """Response from an individual agent."""
    agent_name: str
    recommendation: str
    confidence: float
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
    action: str
    consensus_score: float
    approved: bool
    active_agents: List[str] = field(default_factory=list)
    excluded_agents: List[Dict[str, str]] = field(default_factory=list)
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


class Brain:
    """
    Central Orchestrator - THE BRAIN of Market Hawk.

    Coordinates specialized agents, applies weighted consensus voting,
    and makes final trading decisions with full audit trail.

    Consensus weighting:
        effective_weight = config.weight * category_multiplier
        - ML_MODEL agents get 2x multiplier (trained models)
        - HEURISTIC agents get 1x multiplier (rules/LLM)
        - STUB agents get 0x multiplier (auto-excluded)

    Risk Manager is treated specially - it gates decisions but doesn't vote.
    Stub agents are auto-detected via NOT_IMPLEMENTED status and excluded.
    """

    # Agents that DON'T participate in consensus voting
    NON_VOTING_AGENTS = {"risk_manager"}

    def __init__(self, agent_configs: Dict = None, consensus_threshold: float = None):
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from config.settings import AGENT_CONFIGS, CONSENSUS_THRESHOLD, DECISION_LOG

        self.agent_configs = agent_configs or AGENT_CONFIGS
        self.consensus_threshold = consensus_threshold or CONSENSUS_THRESHOLD
        self.decision_log_path = DECISION_LOG
        self.agents = {}

        logger.info("Brain initialized | threshold=%.2f", self.consensus_threshold)

    def register_agent(self, agent_id: str, agent_instance: Any) -> None:
        """Register an agent with the Brain."""
        self.agents[agent_id] = agent_instance
        category = AGENT_CATEGORY_MAP.get(agent_id, AgentCategory.HEURISTIC)
        if agent_id in self.NON_VOTING_AGENTS:
            role = "GATEKEEPER"
        elif category == AgentCategory.STUB:
            role = "STUB (excluded from consensus)"
        else:
            multiplier = CATEGORY_WEIGHT_MULTIPLIER[category]
            role = f"VOTER [{category.value}, x{multiplier:.0f}]"
        logger.info("Agent registered: %s [%s] (total: %d)", agent_id, role, len(self.agents))

    def _get_agent_category(self, agent_id: str) -> AgentCategory:
        """Get the category for an agent, checking the static map first."""
        return AGENT_CATEGORY_MAP.get(agent_id, AgentCategory.HEURISTIC)

    def _is_stub_response(self, response: dict) -> bool:
        """Check if an agent response indicates it is a stub (NOT_IMPLEMENTED)."""
        return response.get("status") == "NOT_IMPLEMENTED"

    async def query_agents(self, symbol: str, context: Dict = None) -> tuple:
        """Query all VOTING agents for their recommendations.

        Returns:
            (responses, active_agents, excluded_agents) tuple.
            - responses: list of AgentResponse from active agents only
            - active_agents: list of agent_id strings that voted
            - excluded_agents: list of dicts {"agent": id, "reason": str}
        """
        responses: List[AgentResponse] = []
        active_agents: List[str] = []
        excluded_agents: List[Dict[str, str]] = []
        context = context or {}

        for agent_id, agent in self.agents.items():
            # Skip non-voting agents (risk_manager is called separately)
            if agent_id in self.NON_VOTING_AGENTS:
                continue

            # Check static category — skip known stubs before calling
            category = self._get_agent_category(agent_id)
            if category == AgentCategory.STUB:
                # Still call analyze() to confirm stub status dynamically
                pass

            try:
                response = None

                # Try analyze() first, then predict(), then query()
                for method_name in ["analyze", "predict", "query"]:
                    method = getattr(agent, method_name, None)
                    if method is None:
                        continue

                    if asyncio.iscoroutinefunction(method):
                        response = await method(symbol, context)
                    else:
                        response = method(symbol, context)
                    break

                if response is None:
                    logger.warning("Agent %s has no callable method", agent_id)
                    excluded_agents.append({
                        "agent": agent_id,
                        "reason": "no callable method (analyze/predict/query)",
                    })
                    continue

                # Convert to dict for stub check
                if isinstance(response, AgentResponse):
                    resp_dict = asdict(response)
                elif isinstance(response, dict):
                    resp_dict = response
                else:
                    logger.warning("Agent %s returned unexpected type: %s",
                                   agent_id, type(response).__name__)
                    excluded_agents.append({
                        "agent": agent_id,
                        "reason": f"unexpected return type: {type(response).__name__}",
                    })
                    continue

                # Dynamic stub detection
                if self._is_stub_response(resp_dict):
                    logger.info("Agent %s excluded: NOT_IMPLEMENTED", agent_id)
                    excluded_agents.append({
                        "agent": agent_id,
                        "reason": f"NOT_IMPLEMENTED — {resp_dict.get('reasoning', 'stub')}",
                    })
                    continue

                # Agent is active — build AgentResponse
                if isinstance(response, AgentResponse):
                    responses.append(response)
                else:
                    responses.append(AgentResponse(
                        agent_name=agent_id,
                        recommendation=resp_dict.get("recommendation", "HOLD"),
                        confidence=resp_dict.get("confidence", 0.0),
                        reasoning=resp_dict.get("reasoning", ""),
                        metadata=resp_dict.get("metadata", {})
                    ))
                active_agents.append(agent_id)

            except Exception as e:
                logger.error("Agent %s failed: %s", agent_id, str(e))
                excluded_agents.append({
                    "agent": agent_id,
                    "reason": f"runtime error: {str(e)}",
                })

        return responses, active_agents, excluded_agents

    def calculate_consensus(self, responses: List[AgentResponse]) -> float:
        """Weighted consensus with category-based multipliers.

        effective_weight = config.weight * category_multiplier
        ML_MODEL agents (x2) have more influence than HEURISTIC (x1).

        BUY=+1, SELL=-1, HOLD=0.
        Returns score between -1.0 and 1.0.
        """
        direction_map = {"BUY": 1.0, "SELL": -1.0, "HOLD": 0.0}
        total_weight = 0.0
        weighted_sum = 0.0

        for response in responses:
            config = self.agent_configs.get(response.agent_name)
            base_weight = config.weight if config else 0.1
            category = self._get_agent_category(response.agent_name)
            multiplier = CATEGORY_WEIGHT_MULTIPLIER.get(category, 1.0)
            effective_weight = base_weight * multiplier

            direction = direction_map.get(response.recommendation.upper(), 0.0)

            weighted_sum += effective_weight * response.confidence * direction
            total_weight += effective_weight

            logger.debug("  Vote: %s -> %s (conf=%.2f, base_w=%.2f, "
                         "cat=%s, mult=%.1f, eff_w=%.3f)",
                         response.agent_name, response.recommendation,
                         response.confidence, base_weight,
                         category.value, multiplier, effective_weight)

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    async def decide(self, symbol: str, context: Dict = None) -> BrainDecision:
        """Full pipeline: Query agents -> Consensus -> Risk check -> Log."""
        logger.info("=== BRAIN DECISION START: %s ===", symbol)

        # 1. Query voting agents (now returns active/excluded lists)
        responses, active_agents, excluded_agents = await self.query_agents(
            symbol, context
        )
        logger.info("Received %d agent votes | active=%s | excluded=%d",
                     len(responses),
                     active_agents,
                     len(excluded_agents))

        for exc in excluded_agents:
            logger.info("  Excluded: %s — %s", exc["agent"], exc["reason"])

        # 2. Consensus
        consensus = self.calculate_consensus(responses)

        # 3. Action
        if abs(consensus) >= self.consensus_threshold:
            action = "BUY" if consensus > 0 else "SELL"
        else:
            action = "HOLD"

        # 4. Risk Manager gate
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
            active_agents=active_agents,
            excluded_agents=excluded_agents,
            position_size=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            agent_votes=[asdict(r) for r in responses],
            reasoning=self._build_reasoning(
                responses, consensus, approved, active_agents, excluded_agents
            ),
            risk_check=risk_check,
        )

        # 6. Log
        self._log_decision(decision)

        logger.info("=== DECISION: %s %s (consensus=%.4f, approved=%s, "
                     "active=%d, excluded=%d) ===",
                     decision.action, symbol, consensus, approved,
                     len(active_agents), len(excluded_agents))
        return decision

    def _build_reasoning(self, responses, consensus, approved,
                         active_agents, excluded_agents):
        parts = [f"Consensus: {consensus:.4f}"]
        parts.append(f"Active agents ({len(active_agents)}): "
                     f"{', '.join(active_agents) if active_agents else 'none'}")

        for r in responses:
            category = self._get_agent_category(r.agent_name)
            multiplier = CATEGORY_WEIGHT_MULTIPLIER.get(category, 1.0)
            parts.append(f"  {r.agent_name} [{category.value}, x{multiplier:.0f}]: "
                         f"{r.recommendation} (conf={r.confidence:.2f})")

        if excluded_agents:
            parts.append(f"Excluded agents ({len(excluded_agents)}):")
            for exc in excluded_agents:
                parts.append(f"  {exc['agent']}: {exc['reason']}")

        if not approved:
            parts.append("  >>> BLOCKED by Risk Manager")
        return "\n".join(parts)

    def _log_decision(self, decision: BrainDecision) -> None:
        try:
            self.decision_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.decision_log_path, "a") as f:
                f.write(json.dumps(asdict(decision), default=str) + "\n")
        except Exception as e:
            logger.error("Failed to log decision: %s", str(e))
