#!/usr/bin/env python3
"""
SCRIPT NAME: alerts.py
====================================
Execution Location: K:\\_DEV_MVP_2026\\Market_Hawk_3\\backtesting\\
Required Directory Structure:
    Market_Hawk_3/
    └── backtesting/
        ├── __init__.py
        ├── alerts.py           ← THIS FILE
        ├── data_loader.py
        ├── engine.py
        ├── strategies.py
        └── report.py

Author: Professional AI Development System
Level: Doctoral — AI & Machine Learning Specialization
Hardware Optimization: Intel i7-9700F, NVIDIA GTX 1070 8GB VRAM, 64GB DDR4
Creation Date: 2026-03-01
Purpose: Real-time alert system that monitors consensus scores and strategy
         signals, triggering multi-channel notifications (desktop toast,
         sound, log file, webhook) when thresholds are exceeded.
"""

import json
import logging
import os
import time
import threading
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Callable, Any
from enum import Enum

logger = logging.getLogger("market_hawk.backtest.alerts")


# ============================================================
# ALERT DEFINITIONS
# ============================================================

class AlertLevel(Enum):
    """Severity tiers — controls which channels fire."""
    INFO = "INFO"           # Low-priority, log only
    WARNING = "WARNING"     # Moderate, log + desktop toast
    CRITICAL = "CRITICAL"   # High-priority, all channels


class AlertChannel(Enum):
    """Delivery channels for notifications."""
    LOG = "log"             # Always-on: write to log file
    DESKTOP = "desktop"     # Windows 10/11 toast notification
    SOUND = "sound"         # Audible beep / wav file
    WEBHOOK = "webhook"     # HTTP POST (Slack, Discord, Telegram)
    CONSOLE = "console"     # Coloured terminal output
    CALLBACK = "callback"   # Custom Python callable


@dataclass
class Alert:
    """Single alert event."""
    timestamp: str
    level: str
    symbol: str
    strategy: str
    signal: str                 # BUY / SELL / HOLD
    score: float                # Consensus score [-1.0, +1.0]
    confidence: float           # Signal confidence [0.0, 1.0]
    reason: str
    price: float = 0.0
    votes: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)

    def summary_line(self) -> str:
        """One-line summary for console/log."""
        arrow = "🟢 BUY" if self.signal == "BUY" else "🔴 SELL" if self.signal == "SELL" else "⚪ HOLD"
        return (
            f"[{self.level}] {arrow} {self.symbol} | "
            f"score={self.score:+.3f} conf={self.confidence:.0%} | "
            f"price=${self.price:,.2f} | {self.reason}"
        )

    def detail_block(self) -> str:
        """Multi-line detail for desktop toast / webhook."""
        lines = [
            f"{'🟢 BUY' if self.signal == 'BUY' else '🔴 SELL'} — {self.symbol}",
            f"Strategy: {self.strategy}",
            f"Score: {self.score:+.3f}  |  Confidence: {self.confidence:.0%}",
            f"Price: ${self.price:,.2f}",
        ]
        if self.votes:
            vote_str = " | ".join(f"{k}:{v:+.2f}" for k, v in self.votes.items())
            lines.append(f"Votes: {vote_str}")
        lines.append(f"Reason: {self.reason}")
        return "\n".join(lines)


# ============================================================
# ALERT RULES
# ============================================================

@dataclass
class AlertRule:
    """
    Defines when an alert should fire.

    score_threshold: Fires when |consensus_score| exceeds this value.
    confidence_min:  Only fire if confidence >= this.
    signal_type:     Filter by "BUY", "SELL", or None (both).
    cooldown_sec:    Minimum seconds between alerts for same symbol.
    level:           AlertLevel for this rule.
    channels:        Which channels to deliver on.
    symbols:         Restrict to these symbols, or None for all.
    """
    name: str
    score_threshold: float = 0.50
    confidence_min: float = 0.0
    signal_type: Optional[str] = None       # "BUY", "SELL", or None
    cooldown_sec: int = 300                  # 5 min default
    level: AlertLevel = AlertLevel.WARNING
    channels: List[AlertChannel] = field(default_factory=lambda: [
        AlertChannel.LOG, AlertChannel.CONSOLE, AlertChannel.DESKTOP
    ])
    symbols: Optional[List[str]] = None     # None = all symbols

    def matches(self, signal: str, score: float, confidence: float,
                symbol: str) -> bool:
        """Check if a signal matches this rule's conditions."""
        if abs(score) < self.score_threshold:
            return False
        if confidence < self.confidence_min:
            return False
        if self.signal_type and signal != self.signal_type:
            return False
        if self.symbols and symbol not in self.symbols:
            return False
        return True


# ============================================================
# DEFAULT RULE PRESETS
# ============================================================

def default_rules() -> List[AlertRule]:
    """Sensible default rules for Market Hawk."""
    return [
        AlertRule(
            name="strong_signal",
            score_threshold=0.50,
            confidence_min=0.40,
            cooldown_sec=300,
            level=AlertLevel.WARNING,
            channels=[AlertChannel.LOG, AlertChannel.CONSOLE,
                      AlertChannel.DESKTOP, AlertChannel.SOUND],
        ),
        AlertRule(
            name="extreme_signal",
            score_threshold=0.75,
            confidence_min=0.60,
            cooldown_sec=120,
            level=AlertLevel.CRITICAL,
            channels=[AlertChannel.LOG, AlertChannel.CONSOLE,
                      AlertChannel.DESKTOP, AlertChannel.SOUND,
                      AlertChannel.WEBHOOK],
        ),
        AlertRule(
            name="any_trade",
            score_threshold=0.30,
            cooldown_sec=60,
            level=AlertLevel.INFO,
            channels=[AlertChannel.LOG],
        ),
    ]


# ============================================================
# ALERT MANAGER
# ============================================================

class AlertManager:
    """
    Central alert dispatcher for Market Hawk 3.

    Usage:
        mgr = AlertManager()
        mgr.add_rule(AlertRule(name="my_rule", score_threshold=0.5))
        mgr.check_signal(symbol, strategy_name, trade_signal, price, score)

    Or run standalone watcher:
        mgr.start_watcher(data_feed, strategy, interval_sec=60)
    """

    def __init__(self,
                 rules: Optional[List[AlertRule]] = None,
                 log_dir: Optional[str] = None,
                 webhook_url: Optional[str] = None,
                 sound_file: Optional[str] = None,
                 max_history: int = 1000):
        """
        Args:
            rules:        Alert rules (uses defaults if None).
            log_dir:      Directory for alert log files.
            webhook_url:  URL for Slack/Discord/Telegram webhook.
            sound_file:   Path to .wav for sound alerts (None = system beep).
            max_history:  Max alerts kept in memory.
        """
        self.rules = rules or default_rules()
        self.webhook_url = webhook_url
        self.sound_file = sound_file
        self.max_history = max_history
        self._callbacks: List[Callable[[Alert], None]] = []

        # Alert history + cooldown tracking
        self._history: List[Alert] = []
        self._last_fired: Dict[str, float] = {}   # "rule_name:symbol" -> timestamp

        # Log directory
        if log_dir:
            self._log_dir = Path(log_dir)
        else:
            self._log_dir = Path(__file__).resolve().parent.parent / "logs" / "alerts"
        self._log_dir.mkdir(parents=True, exist_ok=True)

        # Alert log file (daily rotation)
        self._log_file = self._log_dir / f"alerts_{datetime.now():%Y%m%d}.jsonl"

        # Watcher thread control
        self._watcher_running = False
        self._watcher_thread: Optional[threading.Thread] = None

        logger.info("AlertManager initialized: %d rules, log=%s",
                     len(self.rules), self._log_dir)

    # ============================================================
    # RULE MANAGEMENT
    # ============================================================

    def add_rule(self, rule: AlertRule) -> None:
        """Add a new alert rule."""
        self.rules.append(rule)
        logger.info("Alert rule added: %s (threshold=%.2f, level=%s)",
                     rule.name, rule.score_threshold, rule.level.value)

    def remove_rule(self, name: str) -> bool:
        """Remove a rule by name."""
        before = len(self.rules)
        self.rules = [r for r in self.rules if r.name != name]
        removed = len(self.rules) < before
        if removed:
            logger.info("Alert rule removed: %s", name)
        return removed

    def add_callback(self, fn: Callable[[Alert], None]) -> None:
        """Register a custom callback for CALLBACK channel alerts."""
        self._callbacks.append(fn)

    # ============================================================
    # CORE: CHECK SIGNAL AGAINST RULES
    # ============================================================

    def check_signal(self, symbol: str, strategy: str,
                     signal_type: str, score: float,
                     confidence: float, price: float,
                     reason: str = "",
                     votes: Optional[Dict[str, float]] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> List[Alert]:
        """
        Evaluate a trading signal against all rules.
        Fires alerts on matching channels if cooldown allows.

        Args:
            symbol:      e.g. "AAPL", "BTCUSDT"
            strategy:    Strategy name
            signal_type: "BUY", "SELL", or "HOLD"
            score:       Consensus score [-1.0, +1.0]
            confidence:  Signal confidence [0.0, 1.0]
            price:       Current price
            reason:      Human-readable reason
            votes:       Individual voter scores
            metadata:    Extra metadata

        Returns:
            List of Alert objects that were fired.
        """
        fired = []

        for rule in self.rules:
            if not rule.matches(signal_type, score, confidence, symbol):
                continue

            # Cooldown check
            cooldown_key = f"{rule.name}:{symbol}"
            now = time.time()
            last = self._last_fired.get(cooldown_key, 0)
            if (now - last) < rule.cooldown_sec:
                logger.debug("Alert suppressed (cooldown): %s on %s",
                             rule.name, symbol)
                continue

            # Create alert
            alert = Alert(
                timestamp=datetime.now().isoformat(timespec="seconds"),
                level=rule.level.value,
                symbol=symbol,
                strategy=strategy,
                signal=signal_type,
                score=score,
                confidence=confidence,
                reason=reason,
                price=price,
                votes=votes or {},
                metadata=metadata or {},
            )

            # Dispatch to channels
            self._dispatch(alert, rule.channels)

            # Record
            self._last_fired[cooldown_key] = now
            self._history.append(alert)
            if len(self._history) > self.max_history:
                self._history = self._history[-self.max_history:]

            fired.append(alert)
            logger.info("ALERT FIRED [%s]: %s", rule.name, alert.summary_line())

        return fired

    def check_trade_signal(self, symbol: str, strategy_name: str,
                           trade_signal, price: float,
                           consensus_score: float = 0.0) -> List[Alert]:
        """
        Convenience: check a TradeSignal dataclass from strategies.py.

        Args:
            trade_signal: TradeSignal object from strategy
            consensus_score: Raw consensus score (if available)
        """
        return self.check_signal(
            symbol=symbol,
            strategy=strategy_name,
            signal_type=trade_signal.signal.value,
            score=consensus_score if consensus_score != 0 else trade_signal.confidence,
            confidence=trade_signal.confidence,
            price=price,
            reason=trade_signal.reason,
            votes=trade_signal.metadata.get("votes", {}),
            metadata=trade_signal.metadata,
        )

    # ============================================================
    # DISPATCH TO CHANNELS
    # ============================================================

    def _dispatch(self, alert: Alert, channels: List[AlertChannel]) -> None:
        """Send alert to all specified channels."""
        for ch in channels:
            try:
                if ch == AlertChannel.LOG:
                    self._send_log(alert)
                elif ch == AlertChannel.CONSOLE:
                    self._send_console(alert)
                elif ch == AlertChannel.DESKTOP:
                    self._send_desktop(alert)
                elif ch == AlertChannel.SOUND:
                    self._send_sound(alert)
                elif ch == AlertChannel.WEBHOOK:
                    self._send_webhook(alert)
                elif ch == AlertChannel.CALLBACK:
                    self._send_callbacks(alert)
            except Exception as e:
                logger.warning("Alert channel %s failed: %s", ch.value, e)

    # ---------- LOG ----------
    def _send_log(self, alert: Alert) -> None:
        """Append alert as JSON line to daily log file."""
        # Rotate log file if date changed
        today_file = self._log_dir / f"alerts_{datetime.now():%Y%m%d}.jsonl"
        if today_file != self._log_file:
            self._log_file = today_file

        with open(self._log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(alert.to_dict()) + "\n")

    # ---------- CONSOLE ----------
    def _send_console(self, alert: Alert) -> None:
        """Print coloured alert to terminal."""
        # ANSI colour codes
        colours = {
            "CRITICAL": "\033[91m",  # Red
            "WARNING":  "\033[93m",  # Yellow
            "INFO":     "\033[96m",  # Cyan
        }
        reset = "\033[0m"
        colour = colours.get(alert.level, "")
        print(f"\n{colour}{'='*70}")
        print(f"  🔔 MARKET HAWK ALERT — {alert.level}")
        print(f"{'='*70}{reset}")
        print(f"  {alert.detail_block()}")
        print(f"{colour}{'='*70}{reset}\n")

    # ---------- DESKTOP (Windows Toast) ----------
    @staticmethod
    def _sanitize_ps_string(text: str) -> str:
        """Strip characters that could escape a PowerShell single-quoted string.

        Single-quoted strings in PowerShell have no escape sequences except ''
        for a literal quote. We replace ' with '' and strip control chars.
        """
        import re
        # Elimina caractere de control (tab/newline pastrate ca spatiu)
        cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
        cleaned = cleaned.replace('\n', ' ').replace('\r', '')
        # Escape single quotes pentru PowerShell single-quoted strings
        cleaned = cleaned.replace("'", "''")
        return cleaned

    def _send_desktop(self, alert: Alert) -> None:
        """Send Windows 10/11 toast notification via PowerShell."""
        if os.name != "nt":
            logger.debug("Desktop notifications only supported on Windows")
            return

        title = self._sanitize_ps_string(
            f"Market Hawk -- {alert.signal} {alert.symbol}"
        )
        body = self._sanitize_ps_string(
            f"Score: {alert.score:+.3f} | Conf: {alert.confidence:.0%} "
            f"Price: ${alert.price:,.2f} "
            f"{alert.reason[:120]}"
        )

        # Folosim single-quoted strings in PowerShell (fara interpolare)
        ps_script = (
            "$ErrorActionPreference = 'SilentlyContinue'\n"
            "$title = '" + title + "'\n"
            "$body = '" + body + "'\n"
            "if (Get-Module -ListAvailable -Name BurntToast) {\n"
            "    Import-Module BurntToast\n"
            "    New-BurntToastNotification -Text $title, $body -AppLogo $null\n"
            "} else {\n"
            "    [void] [System.Reflection.Assembly]::LoadWithPartialName('System.Windows.Forms')\n"
            "    $balloon = New-Object System.Windows.Forms.NotifyIcon\n"
            "    $balloon.Icon = [System.Drawing.SystemIcons]::Information\n"
            "    $balloon.BalloonTipTitle = $title\n"
            "    $balloon.BalloonTipText = $body\n"
            "    $balloon.Visible = $true\n"
            "    $balloon.ShowBalloonTip(5000)\n"
            "    Start-Sleep -Seconds 6\n"
            "    $balloon.Dispose()\n"
            "}\n"
        )

        try:
            subprocess.Popen(
                ["powershell", "-NoProfile", "-WindowStyle", "Hidden",
                 "-Command", ps_script],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
            )
        except Exception as e:
            logger.debug("Desktop toast failed: %s", e)

    # ---------- SOUND ----------
    def _send_sound(self, alert: Alert) -> None:
        """Play alert sound (wav file or system beep)."""
        if self.sound_file and Path(self.sound_file).exists():
            try:
                if os.name == "nt":
                    import winsound
                    winsound.PlaySound(self.sound_file,
                                       winsound.SND_FILENAME | winsound.SND_ASYNC)
                else:
                    subprocess.Popen(["aplay", self.sound_file],
                                     stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL)
            except Exception:
                self._system_beep(alert)
        else:
            self._system_beep(alert)

    @staticmethod
    def _system_beep(alert: Alert) -> None:
        """Fallback: system beep based on severity."""
        try:
            if os.name == "nt":
                import winsound
                freq = 1200 if alert.level == "CRITICAL" else 800
                duration = 500 if alert.level == "CRITICAL" else 300
                beeps = 3 if alert.level == "CRITICAL" else 1
                for i in range(beeps):
                    winsound.Beep(freq, duration)
                    if i < beeps - 1:
                        time.sleep(0.15)
            else:
                print("\a", end="", flush=True)
        except Exception:
            pass

    # ---------- WEBHOOK ----------
    _WEBHOOK_DOMAIN_WHITELIST = {
        "discord.com",
        "discordapp.com",
        "hooks.slack.com",
        "api.telegram.org",
    }

    @classmethod
    def _validate_webhook_url(cls, url: str) -> bool:
        """Validate webhook URL against domain whitelist."""
        from urllib.parse import urlparse
        try:
            parsed = urlparse(url)
            if parsed.scheme not in ("https",):
                logger.warning("Webhook rejected: non-HTTPS scheme %r", parsed.scheme)
                return False
            hostname = (parsed.hostname or "").lower()
            return any(
                hostname == domain or hostname.endswith("." + domain)
                for domain in cls._WEBHOOK_DOMAIN_WHITELIST
            )
        except Exception:
            return False

    def _send_webhook(self, alert: Alert) -> None:
        """Send alert via HTTP webhook (Slack/Discord/Telegram)."""
        if not self.webhook_url:
            logger.debug("Webhook URL not configured, skipping")
            return

        if not self._validate_webhook_url(self.webhook_url):
            logger.warning("Webhook URL rejected — domain not in whitelist: %s",
                           self.webhook_url[:80])
            return

        try:
            import urllib.request

            # Auto-detect service from URL
            url_lower = self.webhook_url.lower()

            if "discord" in url_lower:
                payload = {
                    "content": f"**{alert.level}** {alert.summary_line()}",
                    "embeds": [{
                        "title": f"{'🟢' if alert.signal == 'BUY' else '🔴'} {alert.signal} — {alert.symbol}",
                        "description": alert.detail_block(),
                        "color": 3066993 if alert.signal == "BUY" else 15158332,
                        "timestamp": alert.timestamp,
                    }]
                }
            elif "slack" in url_lower or "hooks.slack" in url_lower:
                payload = {
                    "text": alert.summary_line(),
                    "blocks": [{
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"*{alert.level}*\n```{alert.detail_block()}```"
                        }
                    }]
                }
            else:
                # Generic webhook — send full JSON
                payload = alert.to_dict()

            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self.webhook_url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            # Non-blocking send
            def _do_post():
                try:
                    urllib.request.urlopen(req, timeout=10)
                except Exception as e:
                    logger.warning("Webhook POST failed: %s", e)

            threading.Thread(target=_do_post, daemon=True).start()

        except Exception as e:
            logger.warning("Webhook setup failed: %s", e)

    # ---------- CALLBACK ----------
    def _send_callbacks(self, alert: Alert) -> None:
        """Invoke registered Python callbacks."""
        for fn in self._callbacks:
            try:
                fn(alert)
            except Exception as e:
                logger.warning("Alert callback failed: %s", e)

    # ============================================================
    # HISTORY & REPORTING
    # ============================================================

    @property
    def history(self) -> List[Alert]:
        """All fired alerts in memory."""
        return self._history

    def recent(self, n: int = 20) -> List[Alert]:
        """Last N alerts."""
        return self._history[-n:]

    def count_by_level(self) -> Dict[str, int]:
        """Count alerts by severity level."""
        counts = {}
        for a in self._history:
            counts[a.level] = counts.get(a.level, 0) + 1
        return counts

    def alerts_for_symbol(self, symbol: str) -> List[Alert]:
        """Filter alerts for a specific symbol."""
        return [a for a in self._history if a.symbol == symbol]

    def export_history_json(self, path: Optional[str] = None) -> str:
        """Export full alert history as JSON."""
        if path is None:
            path = str(self._log_dir / f"alert_history_{datetime.now():%Y%m%d_%H%M%S}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump([a.to_dict() for a in self._history], f, indent=2)
        logger.info("Alert history exported: %s (%d alerts)", path, len(self._history))
        return path

    def print_summary(self) -> None:
        """Print alert summary to console."""
        if not self._history:
            print("\n  No alerts fired.")
            return

        counts = self.count_by_level()
        print(f"\n{'='*60}")
        print(f"  🔔 ALERT SUMMARY — {len(self._history)} total alerts")
        print(f"{'='*60}")
        for level in ["CRITICAL", "WARNING", "INFO"]:
            if level in counts:
                print(f"  {level:<12s}: {counts[level]}")

        # Last 5 alerts
        print(f"\n  Recent alerts:")
        for a in self._history[-5:]:
            print(f"    {a.summary_line()}")
        print(f"{'='*60}\n")

    # ============================================================
    # LIVE WATCHER (background thread)
    # ============================================================

    def start_watcher(self, data_feed_fn: Callable,
                      strategy,
                      symbol: str = "UNKNOWN",
                      interval_sec: float = 60.0,
                      max_iterations: Optional[int] = None) -> None:
        """
        Start a background thread that periodically fetches new data,
        runs the strategy, and checks for alerts.

        Args:
            data_feed_fn:    Callable that returns a DataFrame with latest OHLCV.
            strategy:        A StrategyBase instance.
            symbol:          Symbol being watched.
            interval_sec:    Seconds between checks.
            max_iterations:  Stop after N checks (None = run forever).
        """
        if self._watcher_running:
            logger.warning("Watcher already running — stop it first")
            return

        self._watcher_running = True

        def _watch_loop():
            iteration = 0
            logger.info("🔔 Alert watcher started: %s every %.0fs",
                         symbol, interval_sec)
            while self._watcher_running:
                try:
                    df = data_feed_fn()
                    if df is not None and len(df) > 0:
                        # Run strategy on_init if first time
                        if iteration == 0:
                            df = strategy.on_init(df)

                        idx = len(df) - 1
                        sig = strategy.generate_signal(df, idx)
                        price = df["Close"].iloc[idx]

                        # Extract consensus score from metadata
                        consensus_score = 0.0
                        if sig.metadata and "votes" in sig.metadata:
                            votes = sig.metadata["votes"]
                            if votes:
                                consensus_score = sum(votes.values()) / len(votes)

                        if sig.signal.value != "HOLD":
                            self.check_signal(
                                symbol=symbol,
                                strategy=strategy.name,
                                signal_type=sig.signal.value,
                                score=consensus_score,
                                confidence=sig.confidence,
                                price=price,
                                reason=sig.reason,
                                votes=sig.metadata.get("votes", {}),
                            )

                    iteration += 1
                    if max_iterations and iteration >= max_iterations:
                        logger.info("Watcher completed %d iterations", iteration)
                        break

                except Exception as e:
                    logger.error("Watcher error: %s", e)

                # Sleep in small increments for responsive shutdown
                waited = 0.0
                while waited < interval_sec and self._watcher_running:
                    time.sleep(min(1.0, interval_sec - waited))
                    waited += 1.0

            self._watcher_running = False
            logger.info("🔔 Alert watcher stopped")

        self._watcher_thread = threading.Thread(
            target=_watch_loop, daemon=True, name="alert-watcher"
        )
        self._watcher_thread.start()

    def stop_watcher(self) -> None:
        """Stop the background watcher thread."""
        self._watcher_running = False
        if self._watcher_thread:
            self._watcher_thread.join(timeout=10)
            self._watcher_thread = None
            logger.info("Alert watcher stopped")


# ============================================================
# ENGINE INTEGRATION HOOK
# ============================================================

def attach_alerts_to_engine(engine, alert_manager: AlertManager) -> None:
    """
    Monkey-patch the BacktestEngine to fire alerts on every trade signal.
    Call this before engine.run() to get alerts during backtesting.

    Usage:
        engine = BacktestEngine()
        mgr = AlertManager()
        attach_alerts_to_engine(engine, mgr)
        result = engine.run(df, strategy, symbol, timeframe)
        mgr.print_summary()
    """
    original_run = engine.run

    def patched_run(df, strategy, symbol="UNKNOWN", timeframe="1h", **kwargs):
        # Wrap generate_signal to also fire alerts
        original_gen = strategy.generate_signal

        def wrapped_generate_signal(df_inner, idx):
            sig = original_gen(df_inner, idx)
            if sig.signal.value != "HOLD":
                price = df_inner["Close"].iloc[idx]
                consensus_score = 0.0
                if sig.metadata and "votes" in sig.metadata:
                    votes = sig.metadata["votes"]
                    if votes:
                        consensus_score = sum(votes.values()) / len(votes)

                alert_manager.check_signal(
                    symbol=symbol,
                    strategy=strategy.name,
                    signal_type=sig.signal.value,
                    score=consensus_score if consensus_score else sig.confidence,
                    confidence=sig.confidence,
                    price=price,
                    reason=sig.reason,
                    votes=sig.metadata.get("votes", {}),
                )
            return sig

        strategy.generate_signal = wrapped_generate_signal
        result = original_run(df, strategy, symbol, timeframe, **kwargs)
        strategy.generate_signal = original_gen  # Restore
        return result

    engine.run = patched_run


# ============================================================
# CLI: STANDALONE ALERT MONITOR
# ============================================================

def run_alert_monitor():
    """
    Standalone alert monitor that watches a symbol using live data feed.

    Usage:
        python -m backtesting.alerts --symbol AAPL --interval 60
    """
    import argparse
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    parser = argparse.ArgumentParser(
        description="Market Hawk 3 — Alert Monitor")
    parser.add_argument("--symbol", type=str, default="AAPL",
                        help="Symbol to watch")
    parser.add_argument("--interval", type=int, default=60,
                        help="Check interval in seconds")
    parser.add_argument("--threshold", type=float, default=0.50,
                        help="Score threshold for alerts")
    parser.add_argument("--webhook", type=str, default=None,
                        help="Webhook URL (Slack/Discord)")
    parser.add_argument("--no-sound", action="store_true",
                        help="Disable sound alerts")
    parser.add_argument("--no-desktop", action="store_true",
                        help="Disable desktop notifications")
    parser.add_argument("--test", action="store_true",
                        help="Fire a test alert and exit")
    args = parser.parse_args()

    # Logging setup
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    # Build channels list
    channels = [AlertChannel.LOG, AlertChannel.CONSOLE]
    if not args.no_desktop:
        channels.append(AlertChannel.DESKTOP)
    if not args.no_sound:
        channels.append(AlertChannel.SOUND)
    if args.webhook:
        channels.append(AlertChannel.WEBHOOK)

    # Create manager
    rules = [
        AlertRule(
            name="custom_threshold",
            score_threshold=args.threshold,
            cooldown_sec=max(args.interval, 60),
            level=AlertLevel.WARNING,
            channels=channels,
        ),
        AlertRule(
            name="extreme",
            score_threshold=0.75,
            cooldown_sec=60,
            level=AlertLevel.CRITICAL,
            channels=channels + ([AlertChannel.WEBHOOK] if args.webhook else []),
        ),
    ]

    mgr = AlertManager(rules=rules, webhook_url=args.webhook)

    # Test mode
    if args.test:
        print("\n🔔 Firing test alert...\n")
        mgr.check_signal(
            symbol=args.symbol,
            strategy="TestStrategy",
            signal_type="BUY",
            score=0.85,
            confidence=0.75,
            price=185.50,
            reason="Test alert — all channels check",
            votes={"RSI": 0.6, "MACD": 0.9, "MA": 0.7, "BB": 0.5, "Vol": 0.3},
        )
        mgr.print_summary()
        return

    # Live monitor with yfinance data feed
    print(f"\n🔔 Starting alert monitor for {args.symbol}")
    print(f"   Interval: {args.interval}s | Threshold: {args.threshold}")
    print(f"   Channels: {', '.join(c.value for c in channels)}")
    print(f"   Press Ctrl+C to stop\n")

    from backtesting.strategies import AgentConsensusStrategy
    strategy = AgentConsensusStrategy(consensus_threshold=args.threshold)

    _init_done = False

    def fetch_latest():
        """Fetch latest data via yfinance."""
        nonlocal _init_done
        try:
            import yfinance as yf
            import pandas as pd
            df = yf.download(args.symbol, period="60d", interval="1h",
                             progress=False, auto_adjust=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            if not _init_done and df is not None and len(df) > 0:
                from data.market_data_fetcher import MarketDataFetcher
                fetcher = MarketDataFetcher()
                df = fetcher.engineer_features(df)
                _init_done = True
            return df
        except Exception as e:
            logger.error("Data fetch error: %s", e)
            return None

    try:
        mgr.start_watcher(
            data_feed_fn=fetch_latest,
            strategy=strategy,
            symbol=args.symbol,
            interval_sec=args.interval,
        )

        # Keep main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n⏹ Stopping alert monitor...")
        mgr.stop_watcher()
        mgr.print_summary()


# ============================================================
# STANDALONE ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_alert_monitor()
