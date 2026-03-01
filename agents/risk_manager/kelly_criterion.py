"""
SCRIPT NAME: kelly_criterion.py
====================================
Execution Location: market-hawk-mvp/agents/risk_manager/
Purpose: Risk Manager Agent — Kelly Criterion + Position Sizing
Creation Date: 2026-03-01
"""

import logging
from typing import Dict, Any

logger = logging.getLogger("market_hawk.risk_manager")


class RiskManager:
    """Risk Manager Agent — Gates all trading decisions."""

    def __init__(self, config=None):
        from config.settings import RISK_CONFIG
        self.config = config or RISK_CONFIG

    def evaluate(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a trading signal for risk compliance."""
        consensus = abs(signal.get("consensus", 0.0))
        action = signal.get("action", "HOLD")

        if action == "HOLD":
            return {"approved": True, "position_size": 0, "reason": "HOLD — no position"}

        # Kelly Criterion
        win_rate = 0.7647
        avg_win = 0.04
        avg_loss = 0.02
        kelly_pct = self._kelly_criterion(win_rate, avg_win, avg_loss)
        position_size = kelly_pct * consensus
        position_size = min(position_size, self.config.max_position_pct)

        # Adaptive SL/TP
        stop_loss = self.config.default_stop_loss_pct * (1 + (1 - consensus))
        take_profit = self.config.default_take_profit_pct * (1 + consensus)

        approved = position_size > 0.001

        result = {
            "approved": approved,
            "position_size": round(position_size, 6),
            "stop_loss": round(stop_loss, 4),
            "take_profit": round(take_profit, 4),
            "kelly_raw": round(kelly_pct, 6),
            "reason": "Risk check passed" if approved else "Position too small",
        }

        logger.info("Risk: %s | pos=%.4f%% SL=%.2f%% TP=%.2f%%",
                     "APPROVED" if approved else "REJECTED",
                     position_size * 100, stop_loss * 100, take_profit * 100)
        return result

    def _kelly_criterion(self, win_rate, avg_win, avg_loss):
        if avg_loss == 0:
            return 0.0
        b = avg_win / avg_loss
        kelly_full = (b * win_rate - (1 - win_rate)) / b
        return max(0, kelly_full * self.config.kelly_fraction)
