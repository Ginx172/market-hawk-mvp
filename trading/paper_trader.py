"""
SCRIPT NAME: paper_trader.py
====================================
Execution Location: market-hawk-mvp/trading/
Purpose: Paper Trading Loop — Automated virtual trading with P&L tracking
Hardware Optimization: Intel i7-9700F, NVIDIA GTX 1070 8GB VRAM, 64GB DDR4
Creation Date: 2026-03-01

Runs Market Hawk's 5-agent system on a schedule:
    - Scans watchlist every cycle (default: hourly)
    - Executes virtual trades based on Brain consensus
    - Tracks positions, P&L, win rate, drawdown
    - Logs everything to JSONL for analysis

Portfolio starts with $100,000 virtual capital.
All money/price calculations use decimal.Decimal for precision.

Usage:
    # Single scan (test mode)
    python trading/paper_trader.py --once

    # Continuous loop (every 60 minutes)
    python trading/paper_trader.py --interval 60

    # Show portfolio status
    python trading/paper_trader.py --status
"""

import json
import asyncio
import logging
import argparse
import time
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass, field, asdict

from config.settings import RISK_CONFIG

logger = logging.getLogger("market_hawk.paper_trader")

# Constante Decimal reutilizabile
D = Decimal
ZERO = D("0")
ONE = D("1")
D100 = D("100")
INITIAL_CAPITAL = D("100000")
MAX_TRADE_PCT = D("0.1")   # Max 10% per trade


def _to_decimal(value: Any) -> Decimal:
    """Convert a numeric value to Decimal safely via string representation."""
    if isinstance(value, Decimal):
        return value
    if value is None:
        return ZERO
    return D(str(value))


def _validated_decimal(value: Any, name: str,
                       min_val: Decimal = ZERO,
                       max_val: Decimal = D("1000000000")) -> Decimal:
    """Convert to Decimal with bounds validation. Raises ValueError if out of range."""
    d = _to_decimal(value)
    if d < min_val or d > max_val:
        raise ValueError(f"{name}={d} out of bounds [{min_val}, {max_val}]")
    return d


def _d_to_json(value: Decimal) -> str:
    """Serialize Decimal to string for JSON (preserves precision)."""
    return str(value)


# ============================================================
# PORTFOLIO DATA STRUCTURES
# ============================================================

@dataclass
class Position:
    """An open paper trading position."""
    symbol: str
    side: str               # LONG or SHORT
    entry_price: Decimal
    quantity: Decimal
    entry_time: str
    stop_loss: float        # Percentage (not money)
    take_profit: float      # Percentage (not money)
    consensus_at_entry: float
    current_price: Decimal = ZERO
    unrealized_pnl: Decimal = ZERO
    unrealized_pnl_pct: float = 0.0

    def update_price(self, price: Decimal) -> None:
        """Update current price and recalculate unrealized P&L."""
        self.current_price = price
        if self.entry_price == ZERO:
            return
        if self.side == "LONG":
            self.unrealized_pnl = (price - self.entry_price) * self.quantity
            self.unrealized_pnl_pct = float((price - self.entry_price) / self.entry_price)
        else:
            self.unrealized_pnl = (self.entry_price - price) * self.quantity
            self.unrealized_pnl_pct = float((self.entry_price - price) / self.entry_price)

    def should_stop_loss(self, price: Decimal) -> bool:
        """Check if price has hit stop-loss level."""
        sl = _to_decimal(self.stop_loss)
        if self.side == "LONG":
            return price <= self.entry_price * (ONE - sl)
        return price >= self.entry_price * (ONE + sl)

    def should_take_profit(self, price: Decimal) -> bool:
        """Check if price has hit take-profit level."""
        tp = _to_decimal(self.take_profit)
        if self.side == "LONG":
            return price >= self.entry_price * (ONE + tp)
        return price <= self.entry_price * (ONE - tp)


@dataclass
class ClosedTrade:
    """A completed paper trade."""
    symbol: str
    side: str
    entry_price: Decimal
    exit_price: Decimal
    quantity: Decimal
    entry_time: str
    exit_time: str
    pnl: Decimal
    pnl_pct: float          # Ratio, not money
    exit_reason: str        # stop_loss, take_profit, signal_reversal, manual


@dataclass
class Portfolio:
    """Paper trading portfolio state."""
    initial_capital: Decimal = INITIAL_CAPITAL
    cash: Decimal = INITIAL_CAPITAL
    positions: Dict[str, Position] = field(default_factory=dict)
    closed_trades: List[ClosedTrade] = field(default_factory=list)
    peak_equity: Decimal = INITIAL_CAPITAL
    total_scans: int = 0
    total_signals: int = 0

    @property
    def equity(self) -> Decimal:
        """Total portfolio value: cash + unrealized P&L."""
        positions_value = sum(
            (p.unrealized_pnl for p in self.positions.values()), ZERO
        )
        return self.cash + positions_value

    @property
    def total_pnl(self) -> Decimal:
        """Total P&L (realized + unrealized)."""
        return self.equity - self.initial_capital

    @property
    def total_pnl_pct(self) -> float:
        """Total P&L as a percentage."""
        if self.initial_capital == ZERO:
            return 0.0
        return float(self.total_pnl / self.initial_capital)

    @property
    def realized_pnl(self) -> Decimal:
        """Sum of realized P&L from closed trades."""
        return sum((t.pnl for t in self.closed_trades), ZERO)

    @property
    def win_rate(self) -> float:
        """Win rate from closed trades."""
        if not self.closed_trades:
            return 0.0
        wins = sum(1 for t in self.closed_trades if t.pnl > ZERO)
        return wins / len(self.closed_trades)

    @property
    def max_drawdown(self) -> float:
        """Current drawdown from peak equity."""
        if self.peak_equity == ZERO:
            return 0.0
        return float((self.peak_equity - self.equity) / self.peak_equity)

    @property
    def sharpe_proxy(self) -> float:
        """Simple Sharpe proxy from closed trades."""
        if len(self.closed_trades) < 2:
            return 0.0
        import numpy as np
        returns = [t.pnl_pct for t in self.closed_trades]
        mean_r = np.mean(returns)
        std_r = np.std(returns)
        if std_r == 0:
            return 0.0
        return float(mean_r / std_r * (252 ** 0.5))  # Annualized


# ============================================================
# PAPER TRADER ENGINE
# ============================================================

class PaperTrader:
    """
    Paper Trading Engine — executes virtual trades based on Brain decisions.

    Flow per scan:
        1. Update open positions (check SL/TP)
        2. Scan watchlist through Brain (5 agents)
        3. Open new positions if consensus > threshold
        4. Log everything
        5. Display portfolio status
    """

    SAVE_FILE = Path("logs/paper_portfolio.json")
    TRADE_LOG = Path("logs/paper_trades.jsonl")

    def __init__(self, watchlist: List[str] = None):
        self.watchlist = watchlist or [
            "AAPL", "NVDA", "MSFT", "GOOGL", "AMZN", "META", "TSLA",
            "BTCUSDT", "ETHUSDT",
            "GOLD", "SILVER",
            "SPY", "QQQ",
        ]
        self.portfolio = Portfolio()
        self.brain = None
        self.fetcher = None
        self._initialized = False

        # Load saved state if exists
        self._load_state()

    async def initialize(self) -> None:
        """Initialize all agents and Brain."""
        if self._initialized:
            return

        logger.info("Initializing Paper Trader...")

        # Brain
        from brain.orchestrator import Brain
        self.brain = Brain()

        # Knowledge Advisor
        from agents.knowledge_advisor.rag_engine import KnowledgeAdvisor
        advisor = KnowledgeAdvisor()
        if advisor.initialize():
            self.brain.register_agent("knowledge_advisor", advisor)

        # ML Signal Engine
        from agents.ml_signal_engine.catboost_predictor import MLSignalEngine
        ml_engine = MLSignalEngine()
        for model in ["catboost_v2", "catboost_clean_75"]:
            ml_engine.load_model(model)
        if ml_engine._models:
            self.brain.register_agent("ml_signal_engine", ml_engine)

        # News Analyzer (LLM sentiment via qwen3:8b)
        from agents.news_analyzer.news_sentiment import NewsAnalyzer
        self.brain.register_agent("news_analyzer", NewsAnalyzer(use_llm=True))

        # Security Guard
        from agents.security_guard.anomaly_detector import SecurityGuard
        self.brain.register_agent("security_guard", SecurityGuard())

        # Risk Manager
        from agents.risk_manager.kelly_criterion import RiskManager
        self.brain.register_agent("risk_manager", RiskManager())

        # Data Pipeline
        from data.market_data_fetcher import MarketDataFetcher
        self.fetcher = MarketDataFetcher()

        self._initialized = True
        logger.info("Paper Trader ready — %d agents, %d symbols in watchlist",
                     len(self.brain.agents), len(self.watchlist))

    async def scan_and_trade(self) -> None:
        """
        One full scan cycle:
        1. Update open positions
        2. Check SL/TP exits
        3. Scan watchlist for new signals
        4. Open positions on strong consensus
        """
        if not self._initialized:
            await self.initialize()

        self.portfolio.total_scans += 1
        scan_time = datetime.utcnow().isoformat()
        logger.info("=" * 60)
        logger.info("SCAN #%d — %s", self.portfolio.total_scans, scan_time)
        logger.info("=" * 60)

        # 1. Update open positions & check exits
        await self._update_positions()

        # 2. Scan watchlist
        for symbol in self.watchlist:
            # Skip if already have a position
            if symbol in self.portfolio.positions:
                continue

            try:
                # Fetch data + features
                features_df = self.fetcher.fetch_and_engineer(symbol)
                if features_df is None or features_df.empty:
                    continue

                latest_features = self.fetcher.get_latest_features(symbol)
                if latest_features is None:
                    continue

                last_price = _to_decimal(features_df["Close"].iloc[-1])

                # Brain decision
                decision = await self.brain.decide(symbol, {
                    "timeframe": "1h",
                    "features": latest_features.tolist(),
                    "features_df": features_df,
                })

                # Log signal
                if decision.action != "HOLD":
                    self.portfolio.total_signals += 1
                    logger.info("SIGNAL: %s %s @ $%s (consensus: %+.4f)",
                                 decision.action, symbol, last_price,
                                 decision.consensus_score)

                # Open position if approved and strong consensus
                if decision.action in ("BUY", "SELL") and decision.approved:
                    self._open_position(
                        symbol=symbol,
                        action=decision.action,
                        price=last_price,
                        consensus=decision.consensus_score,
                        position_size=decision.position_size or 0.02,
                        stop_loss=decision.stop_loss or 0.02,
                        take_profit=decision.take_profit or 0.04,
                    )

            except Exception:
                logger.exception("Error scanning %s", symbol)

        # 3. Update peak equity
        if self.portfolio.equity > self.portfolio.peak_equity:
            self.portfolio.peak_equity = self.portfolio.equity

        # 4. Save state
        self._save_state()

        # 5. Display status
        self._print_status()

    async def _update_positions(self) -> None:
        """Update prices and check SL/TP for open positions."""
        to_close: List[tuple] = []

        for symbol, pos in self.portfolio.positions.items():
            try:
                features_df = self.fetcher.fetch_and_engineer(symbol, period="5d")
                if features_df is None or features_df.empty:
                    continue

                current_price = _to_decimal(features_df["Close"].iloc[-1])
                pos.update_price(current_price)

                # Check stop loss
                if pos.should_stop_loss(current_price):
                    to_close.append((symbol, current_price, "stop_loss"))
                    logger.info("STOP LOSS: %s @ $%s (entry: $%s)",
                                 symbol, current_price, pos.entry_price)

                # Check take profit
                elif pos.should_take_profit(current_price):
                    to_close.append((symbol, current_price, "take_profit"))
                    logger.info("TAKE PROFIT: %s @ $%s (entry: $%s)",
                                 symbol, current_price, pos.entry_price)

            except Exception:
                logger.exception("Error updating %s", symbol)

        # Close positions
        for symbol, price, reason in to_close:
            self._close_position(symbol, price, reason)

    def _open_position(self, symbol: str, action: str, price: Decimal,
                       consensus: float, position_size: float,
                       stop_loss: float, take_profit: float) -> None:
        """Open a new paper position."""
        # Circuit breaker: refuse new trades if max drawdown exceeded
        if self.portfolio.max_drawdown > RISK_CONFIG.max_drawdown_pct:
            logger.warning(
                "CIRCUIT BREAKER: refusing %s %s — max drawdown exceeded (%.2f%% > %.2f%%)",
                action, symbol,
                self.portfolio.max_drawdown * 100,
                RISK_CONFIG.max_drawdown_pct * 100)
            return

        pos_size_d = _to_decimal(position_size)
        trade_value = self.portfolio.cash * pos_size_d

        max_value = self.portfolio.cash * MAX_TRADE_PCT
        if trade_value > max_value:
            trade_value = max_value

        quantity = trade_value / price if price > ZERO else ZERO

        if quantity <= ZERO:
            return

        side = "LONG" if action == "BUY" else "SHORT"
        pos = Position(
            symbol=symbol,
            side=side,
            entry_price=price,
            quantity=quantity,
            entry_time=datetime.utcnow().isoformat(),
            stop_loss=stop_loss,
            take_profit=take_profit,
            consensus_at_entry=consensus,
            current_price=price,
        )

        self.portfolio.positions[symbol] = pos

        # Reserve cash (for LONG positions)
        if side == "LONG":
            self.portfolio.cash -= trade_value

        logger.info("OPENED %s %s: %s units @ $%s ($%s) | SL=%.2f%% TP=%.2f%%",
                     side, symbol, quantity.quantize(D("0.0001")),
                     price.quantize(D("0.01")), trade_value.quantize(D("0.01")),
                     stop_loss * 100, take_profit * 100)

        self._log_trade_event("OPEN", symbol, side, price, quantity, consensus)

    def _close_position(self, symbol: str, exit_price: Decimal, reason: str) -> None:
        """Close a paper position and record the trade."""
        if symbol not in self.portfolio.positions:
            return

        pos = self.portfolio.positions[symbol]
        pos.update_price(exit_price)

        # Calculate P&L
        if pos.side == "LONG":
            pnl = (exit_price - pos.entry_price) * pos.quantity
            self.portfolio.cash += exit_price * pos.quantity
        else:
            pnl = (pos.entry_price - exit_price) * pos.quantity
            self.portfolio.cash += pnl

        entry_value = pos.entry_price * pos.quantity
        pnl_pct = float(pnl / entry_value) if entry_value > ZERO else 0.0

        trade = ClosedTrade(
            symbol=symbol,
            side=pos.side,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            quantity=pos.quantity,
            entry_time=pos.entry_time,
            exit_time=datetime.utcnow().isoformat(),
            pnl=pnl,
            pnl_pct=pnl_pct,
            exit_reason=reason,
        )
        self.portfolio.closed_trades.append(trade)
        del self.portfolio.positions[symbol]

        tag = "WIN" if pnl > ZERO else "LOSS"
        logger.info("%s CLOSED %s %s @ $%s -> $%s | P&L: $%s (%.2f%%) | %s",
                     tag, pos.side, symbol,
                     pos.entry_price.quantize(D("0.01")),
                     exit_price.quantize(D("0.01")),
                     pnl.quantize(D("0.01")),
                     pnl_pct * 100, reason)

        self._log_trade_event("CLOSE", symbol, pos.side, exit_price,
                               pos.quantity, 0, pnl=pnl, reason=reason)

    def _print_status(self) -> None:
        """Display current portfolio status."""
        p = self.portfolio
        equity = p.equity
        print(f"\n{'---' * 20}")
        print(f"  PORTFOLIO STATUS")
        print(f"{'---' * 20}")
        print(f"  Equity:      ${equity:>14,.2f}  ({p.total_pnl_pct:+.2%})")
        print(f"  Cash:        ${p.cash:>14,.2f}")
        print(f"  Realized:    ${p.realized_pnl:>14,.2f}")
        print(f"  Open:        {len(p.positions)} positions")
        print(f"  Trades:      {len(p.closed_trades)} closed")
        print(f"  Win Rate:    {p.win_rate:.1%}")
        print(f"  Max DD:      {p.max_drawdown:.2%}")
        print(f"  Scans:       {p.total_scans}")
        print(f"  Signals:     {p.total_signals}")

        if p.positions:
            print(f"\n  Open Positions:")
            for sym, pos in p.positions.items():
                tag = "+" if pos.unrealized_pnl >= ZERO else "-"
                print(f"    [{tag}] {pos.side:5s} {sym:<10s} "
                      f"entry=${pos.entry_price:>10,.2f} "
                      f"now=${pos.current_price:>10,.2f} "
                      f"P&L=${pos.unrealized_pnl:>+10,.2f} "
                      f"({pos.unrealized_pnl_pct:+.2%})")

        if p.closed_trades:
            recent = p.closed_trades[-5:]
            print(f"\n  Recent Trades (last {len(recent)}):")
            for t in recent:
                tag = "WIN " if t.pnl > ZERO else "LOSS"
                print(f"    [{tag}] {t.side:5s} {t.symbol:<10s} "
                      f"${t.entry_price:,.2f}->${t.exit_price:,.2f} "
                      f"P&L=${t.pnl:>+,.2f} ({t.exit_reason})")

        print(f"{'---' * 20}")

    def _log_trade_event(self, event: str, symbol: str, side: str,
                          price: Decimal, qty: Decimal, consensus: float = 0,
                          pnl: Decimal = ZERO, reason: str = "") -> None:
        """Log trade event to JSONL (encrypted if MH_ENCRYPTION_KEY is set)."""
        try:
            from trading.crypto_log import write_log_line

            entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "event": event,
                "symbol": symbol,
                "side": side,
                "price": _d_to_json(price),
                "quantity": _d_to_json(qty),
                "consensus": consensus,
                "pnl": _d_to_json(pnl),
                "reason": reason,
                "equity": _d_to_json(self.portfolio.equity),
            }
            write_log_line(self.TRADE_LOG, json.dumps(entry))
        except Exception:
            logger.exception("Failed to log trade")

    def _save_state(self) -> None:
        """Save portfolio state to JSON."""
        try:
            self.SAVE_FILE.parent.mkdir(parents=True, exist_ok=True)

            # Serialize positions — convert Decimal fields to string
            positions_data = {}
            for sym, pos in self.portfolio.positions.items():
                positions_data[sym] = {
                    "symbol": pos.symbol,
                    "side": pos.side,
                    "entry_price": _d_to_json(pos.entry_price),
                    "quantity": _d_to_json(pos.quantity),
                    "entry_time": pos.entry_time,
                    "stop_loss": pos.stop_loss,
                    "take_profit": pos.take_profit,
                    "consensus_at_entry": pos.consensus_at_entry,
                    "current_price": _d_to_json(pos.current_price),
                    "unrealized_pnl": _d_to_json(pos.unrealized_pnl),
                    "unrealized_pnl_pct": pos.unrealized_pnl_pct,
                }

            state = {
                "saved_at": datetime.utcnow().isoformat(),
                "cash": _d_to_json(self.portfolio.cash),
                "initial_capital": _d_to_json(self.portfolio.initial_capital),
                "peak_equity": _d_to_json(self.portfolio.peak_equity),
                "total_scans": self.portfolio.total_scans,
                "total_signals": self.portfolio.total_signals,
                "positions": positions_data,
                "closed_trades_count": len(self.portfolio.closed_trades),
                "realized_pnl": _d_to_json(self.portfolio.realized_pnl),
                "win_rate": self.portfolio.win_rate,
            }
            with open(self.SAVE_FILE, "w") as f:
                json.dump(state, f, indent=2)
        except Exception:
            logger.exception("Failed to save state")

    _VALID_SIDES = {"LONG", "SHORT"}

    def _load_state(self) -> None:
        """Load portfolio state from JSON with validation."""
        if not self.SAVE_FILE.exists():
            return
        try:
            with open(self.SAVE_FILE) as f:
                state = json.load(f)

            if not isinstance(state, dict):
                raise ValueError("State file root must be a JSON object")

            self.portfolio.cash = _validated_decimal(
                state.get("cash", "100000"), "cash", min_val=ZERO)
            self.portfolio.peak_equity = _validated_decimal(
                state.get("peak_equity", "100000"), "peak_equity", min_val=ZERO)
            self.portfolio.total_scans = int(state.get("total_scans", 0))
            self.portfolio.total_signals = int(state.get("total_signals", 0))

            # Restore open positions with validation
            for sym, pos_data in state.get("positions", {}).items():
                if not isinstance(pos_data, dict):
                    logger.warning("Skipping invalid position data for %s", sym)
                    continue

                side = pos_data.get("side", "")
                if side not in self._VALID_SIDES:
                    logger.warning("Skipping position %s: invalid side %r", sym, side)
                    continue

                self.portfolio.positions[sym] = Position(
                    symbol=str(pos_data.get("symbol", sym))[:20],
                    side=side,
                    entry_price=_validated_decimal(
                        pos_data.get("entry_price", 0), "entry_price", min_val=ZERO),
                    quantity=_validated_decimal(
                        pos_data.get("quantity", 0), "quantity", min_val=ZERO),
                    entry_time=str(pos_data.get("entry_time", "")),
                    stop_loss=float(pos_data.get("stop_loss", 0.02)),
                    take_profit=float(pos_data.get("take_profit", 0.04)),
                    consensus_at_entry=float(pos_data.get("consensus_at_entry", 0.0)),
                    current_price=_validated_decimal(
                        pos_data.get("current_price", 0), "current_price", min_val=ZERO),
                    unrealized_pnl=_to_decimal(pos_data.get("unrealized_pnl", 0)),
                    unrealized_pnl_pct=float(pos_data.get("unrealized_pnl_pct", 0.0)),
                )

            logger.info("Loaded portfolio state from %s (scans: %d, positions: %d)",
                         self.SAVE_FILE, self.portfolio.total_scans,
                         len(self.portfolio.positions))
        except (ValueError, TypeError, KeyError) as e:
            logger.warning("Invalid state data in %s: %s", self.SAVE_FILE, str(e))
        except Exception:
            logger.exception("Could not load state")


# ============================================================
# MAIN
# ============================================================

async def run_loop(interval_minutes: int) -> None:
    """Continuous paper trading loop."""
    trader = PaperTrader()
    await trader.initialize()

    print(f"\nPaper Trading Loop — scanning every {interval_minutes} minutes")
    print(f"   Watchlist: {', '.join(trader.watchlist)}")
    print(f"   Press Ctrl+C to stop\n")

    while True:
        try:
            await trader.scan_and_trade()
            logger.info("Next scan in %d minutes...", interval_minutes)
            await asyncio.sleep(interval_minutes * 60)
        except KeyboardInterrupt:
            print("\n\nPaper Trading stopped by user")
            trader._save_state()
            trader._print_status()
            break
        except Exception:
            logger.exception("Scan error — retrying in 60s")
            await asyncio.sleep(60)


async def run_once() -> None:
    """Single scan (test mode)."""
    trader = PaperTrader()
    await trader.initialize()
    await trader.scan_and_trade()


async def show_status() -> None:
    """Show portfolio status without scanning."""
    trader = PaperTrader()
    trader._print_status()


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(name)-30s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    for noisy in ["httpx", "httpcore", "chromadb", "urllib3", "yfinance", "peewee"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description="Market Hawk Paper Trader")
    parser.add_argument("--once", action="store_true", help="Single scan")
    parser.add_argument("--status", action="store_true", help="Show portfolio")
    parser.add_argument("--interval", type=int, default=60, help="Scan interval (minutes)")
    args = parser.parse_args()

    if args.status:
        asyncio.run(show_status())
    elif args.once:
        asyncio.run(run_once())
    else:
        asyncio.run(run_loop(args.interval))
