"""
SCRIPT NAME: kelly_criterion.py
====================================
Execution Location: market-hawk-mvp/agents/risk_manager/
Purpose: Risk Manager Agent — Kelly Criterion + Position Sizing
Creation Date: 2026-03-01

Kelly parameters are calculated dynamically from closed trade history.
If fewer than MIN_TRADES_FOR_KELLY trades exist, quarter-Kelly with
conservative defaults is used. With zero trades, returns minimum
position size (1% of equity).
"""

import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, Any, List, Optional

logger = logging.getLogger("market_hawk.risk_manager")

# Minimum trades needed for statistically meaningful Kelly calculation
MIN_TRADES_FOR_KELLY = 30

# Conservative fallback when trade history is insufficient
_FALLBACK_POSITION_SIZE = 0.01  # 1% of equity


@dataclass(frozen=True)
class ClosedTradeRecord:
    """Lightweight trade record for Kelly calculation.

    Can be constructed from paper_trader.ClosedTrade or backtest results.
    Only needs pnl_pct (return as ratio, e.g., 0.04 = +4%).
    """
    pnl_pct: float


class RiskManager:
    """Risk Manager Agent — Gates all trading decisions.

    Kelly parameters are derived from actual trade history, not hardcoded.
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        from config.settings import RISK_CONFIG
        self.config = config or RISK_CONFIG

    def evaluate(self, signal: Dict[str, Any],
                 trades: Optional[List[ClosedTradeRecord]] = None) -> Dict[str, Any]:
        """Evaluate a trading signal for risk compliance.

        Args:
            signal: Dict with 'consensus' (float) and 'action' (str).
            trades: List of closed trades for Kelly calculation.
                    If None or empty, uses conservative fallback.

        Returns:
            Dict with 'approved', 'position_size', 'stop_loss',
            'take_profit', 'kelly_raw', 'reason'.
        """
        consensus = abs(signal.get("consensus", 0.0))
        action = signal.get("action", "HOLD")

        if action == "HOLD":
            return {"approved": True, "position_size": 0, "reason": "HOLD — no position"}

        # Calculate Kelly from trade history
        kelly_pct = self._kelly_from_trades(trades)
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

    def _kelly_from_trades(self,
                           trades: Optional[List[ClosedTradeRecord]] = None) -> float:
        """Calculate Kelly fraction from closed trade history.

        Args:
            trades: List of ClosedTradeRecord with pnl_pct.

        Returns:
            Kelly-adjusted position fraction (already multiplied by kelly_fraction).
            Falls back to _FALLBACK_POSITION_SIZE if insufficient data.
        """
        if not trades:
            logger.debug("No trade history — using fallback position size %.2f%%",
                         _FALLBACK_POSITION_SIZE * 100)
            return _FALLBACK_POSITION_SIZE

        wins = [t.pnl_pct for t in trades if t.pnl_pct > 0]
        losses = [t.pnl_pct for t in trades if t.pnl_pct <= 0]

        total = len(trades)
        win_count = len(wins)

        if total < MIN_TRADES_FOR_KELLY:
            logger.debug("Only %d trades (need %d) — using quarter-Kelly with "
                         "conservative estimates", total, MIN_TRADES_FOR_KELLY)
            # Use actual stats but apply extra conservative multiplier
            if not wins or not losses:
                return _FALLBACK_POSITION_SIZE
            win_rate = win_count / total
            avg_win = sum(wins) / len(wins)
            avg_loss = abs(sum(losses) / len(losses))
            kelly = self._kelly_criterion(win_rate, avg_win, avg_loss)
            # Extra-conservative: half of quarter-Kelly for small samples
            return kelly * 0.5

        if not wins or not losses:
            logger.debug("All wins or all losses — using fallback")
            return _FALLBACK_POSITION_SIZE

        win_rate = win_count / total
        avg_win = sum(wins) / len(wins)
        avg_loss = abs(sum(losses) / len(losses))

        kelly = self._kelly_criterion(win_rate, avg_win, avg_loss)

        logger.info("Kelly from %d trades: win_rate=%.2f%%, avg_win=%.2f%%, "
                     "avg_loss=%.2f%%, kelly=%.4f",
                     total, win_rate * 100, avg_win * 100, avg_loss * 100, kelly)
        return kelly

    def _kelly_criterion(self, win_rate: float, avg_win: float,
                         avg_loss: float) -> float:
        """Core Kelly formula: f* = (b*p - q) / b, scaled by kelly_fraction.

        Args:
            win_rate: Probability of winning (0.0 to 1.0).
            avg_win: Average win size as ratio (e.g., 0.04 = 4%).
            avg_loss: Average loss size as positive ratio (e.g., 0.02 = 2%).

        Returns:
            Kelly-adjusted fraction (non-negative).
        """
        if avg_loss == 0:
            return 0.0
        b = avg_win / avg_loss
        kelly_full = (b * win_rate - (1 - win_rate)) / b
        return max(0.0, kelly_full * self.config.kelly_fraction)
