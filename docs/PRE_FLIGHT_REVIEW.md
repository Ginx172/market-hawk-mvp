# Market Hawk 3 — Pre-Flight Review

**Date:** 2026-03-07
**Scope:** Full codebase audit — security + quantitative trading
**Status:** READ-ONLY audit (no changes applied)

---

## Table of Contents

1. [Security Audit (White Hat / Penetration Tester)](#1-security-audit)
2. [Trading Audit (Algo Trading Specialist / Quant)](#2-trading-audit)
3. [Priority Matrix](#3-priority-matrix)
4. [Conclusions](#4-conclusions)

---

## 1. Security Audit

**Perspective:** White Hat Hacker / Penetration Tester

### Executive Summary

Good security hygiene in credential management (env vars, RBAC, TLS enforcement),
but several CRITICAL and HIGH issues remain: command injection in PowerShell alerts,
pickle deserialization in ML model loading, XSS via `unsafe_allow_html` in Streamlit,
and weak dashboard authentication without rate limiting or session tokens.

### Findings

| # | Finding | Severity | File / Line | Recommended Fix | Effort |
|---|---------|----------|-------------|-----------------|--------|
| S1 | **Command injection in PowerShell alerts** — `subprocess.Popen(["powershell", "-Command", ps_script])` where `ps_script` includes unsanitized `title`/`body` from alert data. Backtick or `$()` in reason field can execute arbitrary commands. | **CRITICAL** | `backtesting/alerts.py:439-466` | Use `-EncodedCommand` with base64-encoded script, or replace PowerShell with HTML/JSON notification. Never concatenate untrusted strings into shell commands. | 2-3h |
| S2 | **Insecure deserialization (pickle)** — `joblib.load()` used for ML models without hash verification or source validation. Attacker replacing `.pkl` file on G: drive achieves RCE on model load. | **CRITICAL** | `agents/ml_signal_engine/catboost_predictor.py:29` | Use CatBoost native `.cbm` format only (no pickle). Add SHA-256 hash verification before loading any model file. For sklearn/xgboost models, use ONNX. | 4-6h |
| S3 | **XSS via `unsafe_allow_html=True`** — Agent votes, price displays, and markdown rendered with `unsafe_allow_html=True`. If ML model outputs HTML/JS in reason field, it executes in analyst's browser. | **HIGH** | `dashboard/app.py:100,315,640,797` | Remove `unsafe_allow_html=True` on all user-controlled data. Sanitize with `bleach.clean(text, tags=[], strip=True)` or use `st.write()` which auto-escapes. | 2-3h |
| S4 | **Weak dashboard authentication** — Password-only check (`pwd == _DASHBOARD_PASSWORD`), no rate limiting on login attempts, no session timeout, no JWT tokens. SessionManager exists in `config/session.py` but is not used. | **HIGH** | `dashboard/app.py:45-60` | Integrate existing `config/session.py` SessionManager. Add rate limiting (max 5 attempts/15 min), JWT-based sessions, 30-min idle timeout, login attempt logging. | 1-2h |
| S5 | **Path traversal in model loading** — `_is_trusted_path()` validates against `ML_CONFIG.models_base_dir`, but that dir is env-var-controlled. Attacker with env access can redirect to malicious path. No symlink checking. | **HIGH** | `agents/ml_signal_engine/catboost_predictor.py:56-63` | Hardcode trusted model dirs (no env override for security-critical paths). Add symlink rejection: `if path.resolve().is_symlink(): reject`. Verify file ownership. | 2h |
| S6 | **Insufficient security event logging** — RBAC denials raise `PermissionError` but don't log which user/role attempted what. No audit trail for failed logins, failed orders, role changes. Dashboard errors go to `st.warning()`, not persistent log. | **HIGH** | `config/rbac.py:138-140`, `dashboard/app.py:241`, multiple | Create dedicated audit logger. Log all auth attempts (success + failure), permission checks, order submissions with timestamps. Use structured JSON logging. | 3-4h |
| S7 | **Symbol validation too permissive** — Regex `r'^[A-Za-z0-9=^.\-]{1,20}$'` allows characters that could be problematic in different broker APIs. No per-broker validation. | **MEDIUM** | `data/market_data_fetcher.py:70-78` | Add per-broker symbol validation. Tighten regex or use whitelist. Limit to 10 chars for standard equities. | 1h |
| S8 | **Silent exception handling** — Broad `except Exception as e:` blocks catch all errors including `KeyboardInterrupt`/`SystemExit`. Only `str(e)` logged, no stack traces. Bugs silently swallowed. | **MEDIUM** | `trading/paper_trader.py:344`, `dashboard/app.py:241`, `agents/knowledge_advisor/rag_engine.py:215` | Catch specific exceptions. Use `logger.exception()` for stack traces. Re-raise unrecoverable errors. | 2-3h |
| S9 | **Rate limiter bypass (per-instance)** — `_RateLimiter` in AlpacaAdapter is per-instance. Multiple adapter instances bypass the 200 req/min limit (5 instances = 1000 req/min). | **MEDIUM** | `executor/broker_adapters/alpaca_adapter.py:58-85` | Make `_RateLimiter` a class-level singleton or module-level global. Enforce across all instances. | 1-2h |
| S10 | **TLS context created but not enforced in adapters** — `broker_auth.get_tls_context()` creates TLS 1.2+ context, but Alpaca adapter uses SDK defaults (may not use custom context). IB adapter uses ib_insync built-in SSL. Actual TLS version not verified. | **MEDIUM** | `executor/broker_auth.py:184-204`, `executor/broker_adapters/alpaca_adapter.py`, `executor/broker_adapters/ib_adapter.py` | Verify SDK TLS behavior. Pass custom SSL context where possible. Add certificate pinning for known broker API domains. | 2-3h |
| S11 | **Hardcoded fallback paths** — `config/settings.py` has hardcoded `J:\`, `C:\_AI\` paths as fallbacks when env vars not set. Misleading on other machines. | **LOW** | `config/settings.py:36-65` | Require env vars for all paths (no fallback). Validate path existence at startup with clear error messages. | 0.5h |
| S12 | **Fixed timeouts may be insufficient** — 30-second timeouts for yfinance and IB on slow networks can cause spurious failures. No adaptive timeout. | **LOW** | `data/market_data_fetcher.py:134`, `executor/broker_adapters/ib_adapter.py:92,128` | Make timeouts configurable via config. Add adaptive timeout based on observed latency. | 1-2h |
| S13 | **Encryption silently disabled** — `crypto_log.py` falls back to plaintext if `MH_ENCRYPTION_KEY` is invalid. Only logs a warning, no strong alert. | **LOW** | `trading/crypto_log.py:34-54` | Raise error on invalid encryption key (don't silently degrade). Validate key format at startup. | 0.5h |
| S14 | **Webhook domain whitelist incomplete** — Only Discord, Slack, Telegram whitelisted. Missing Teams, custom endpoints. Blocks legitimate use. | **LOW** | `backtesting/alerts.py:507-529` | Allow configurable whitelist via env var or config file. Require HTTPS for all webhook domains. | 0.5h |

### Dependencies — Known Vulnerabilities

| Package | Version | Status | Action |
|---------|---------|--------|--------|
| `cryptography` | ~=42.0.0 | OK | None |
| `PyJWT` | ~=2.8.0 | OK | Verify JWT_SECRET strength |
| `streamlit` | ~=1.28.0 | OK (auth bypass fixed in 1.18+) | None |
| `langchain` | ~=0.1.0 | **OLD** — security patches in 0.2.x | Upgrade to 0.2.x |
| `chromadb` | ~=0.4.0 | **OLD** — multiple fixes in 0.4.26+ | Upgrade to 0.4.26+ or 0.5.x |
| `joblib` | ~=1.3.0 | OK (but pickle-based = inherently risky) | Minimize use |
| `alpaca-trade-api` | ~=3.0.0 | OK | None |
| `ib_insync` | ~=0.9.0 | OK | None |

### Missing Security Controls

| Control | Impact | Recommendation |
|---------|--------|----------------|
| No API response schema validation | Response poisoning risk | Add Pydantic models for broker API responses |
| No secrets rotation automation | Stale keys risk | Implement monthly rotation + 7-day expiry alerts |
| No dashboard rate limiting | DoS risk | Add Nginx reverse proxy or Streamlit middleware |
| No HTTPS enforcement in dev | MITM risk | Use `--server.sslCertFile` / `--server.sslKeyFile` in Streamlit |
| No intrusion detection | Insider threat blindness | Add anomaly detection for trade patterns |

---

## 2. Trading Audit

**Perspective:** Algo Trading Specialist / Quant

### Executive Summary

Well-structured backtesting engine with proper Decimal handling and clean strategy
interface (StrategyBase). However, CRITICAL gaps in ML model validation (100% accuracy
claims = overfitting), hardcoded Kelly criterion parameters, no walk-forward optimization,
unrealistic fill assumptions, and non-standard indicator calculations. The system is
suitable for MVP demos and paper trading exploration but NOT for live capital deployment.

### Findings

| # | Finding | Severity | File / Line | Recommended Fix | Effort |
|---|---------|----------|-------------|-----------------|--------|
| T1 | **ML models claim 100% accuracy (overfitting)** — `catboost_80plus` reports `accuracy: 1.0`, `xgboost_80plus` reports `0.9996`. No real financial model achieves this. Indicates train/test leakage, look-ahead bias in features, or data snooping across 10+ model variants. | **CRITICAL** | `agents/ml_signal_engine/catboost_predictor.py:118-139` | Retrain with proper walk-forward validation on held-out data (e.g., train on 2024, test on 2025). Expect realistic accuracy of 52-60% for binary classification. Disable "80plus" suite until validated. | 20h+ |
| T2 | **Kelly criterion hardcoded** — `win_rate=0.7647`, `avg_win=0.04`, `avg_loss=0.02` are magic numbers, not calculated from actual trade history. Applied identically to all strategies regardless of their actual performance. | **CRITICAL** | `agents/risk_manager/kelly_criterion.py:31-35` | Calculate win_rate, avg_win, avg_loss dynamically from closed trades in backtester. Use quarter-Kelly (conservative) when trade history < 30 trades. Update after each completed trade. | 4h |
| T3 | **Multiple agent stubs in consensus system** — Continuous Learner, Security Guard, News Analyzer may be stubs. 5-agent consensus assumes all agents are functional; stubs produce meaningless votes that corrupt consensus scores. | **HIGH** | `agents/continuous_learner/`, `agents/security_guard/`, `agents/news_analyzer/` | Audit all agents. Implement or remove stubs. Adjust consensus weighting to exclude non-functional agents. | 10h+ |
| T4 | **No walk-forward optimization** — Backtester runs over full data period with fixed parameters. No rolling/anchored window train-validate splits. Strategy parameters are optimized on in-sample data only. | **HIGH** | `backtesting/engine.py` (not implemented) | Add `run_walk_forward(df, strategy, train_pct=0.6, step_pct=0.1)`. Divide data into rolling windows: train on [0:N], validate on [N:N+step], advance, repeat. | 6h |
| T5 | **No cross-validation methodology for ML** — Code loads pre-trained models but provides no time-series cross-validation. No out-of-sample testing framework. | **HIGH** | `agents/ml_signal_engine/catboost_predictor.py` (loads models only) | Implement time-series-aware walk-forward CV: train on expanding/rolling window, test on next period. Never shuffle time series data. | 8h |
| T6 | **No dividend/split adjustment** — Engine uses unadjusted prices (`auto_adjust=False` in yfinance). `data_loader.py` sets `Dividends: 0.0`, `Stock Splits: 0.0`. A 2:1 split appears as a 50% price drop. | **HIGH** | `backtesting/engine.py` (not addressed), `backtesting/data_loader.py:134,188-189,546` | Use adjusted close for backtesting (`auto_adjust=True`), or explicitly handle corporate actions. Document which price type is used. | 4h |
| T7 | **Max drawdown not enforced** — Config defines `max_drawdown_pct: 0.15` but it is never checked during backtest or paper trading. Strategy continues trading through 50%+ drawdowns. | **HIGH** | `config/settings.py:189`, `backtesting/engine.py` (no check) | Add drawdown check in equity calculation loop. Halt trading (or log warning) when drawdown exceeds configured limit. | 2h |
| T8 | **Look-ahead bias risk in feature engineering** — Early NaN values for SMA/EMA/RSI (bars 0-19 for period=20) are forward-filled with `result.ffill().bfill().fillna(0)`. `bfill()` uses future values to fill past NaN = look-ahead. | **HIGH** | `data/market_data_fetcher.py:288` | Drop early NaN rows instead of backward-filling: `result = result.iloc[warmup_period:]`. Or use `ffill()` only (no `bfill()`). | 1h |
| T9 | **Fill assumptions unrealistic** — All orders fill at current bar's close price + static slippage. Real execution fills at next bar's open (or worse). No fill delay modeling. | **MEDIUM** | `backtesting/engine.py:330-378` | Add `fill_delay_bars` parameter (default=1): fill at next bar's open, not current bar's close. Model market impact: `price *= (1 + impact * sqrt(size / volume))`. | 3h |
| T10 | **Static slippage model** — Fixed 0.05% slippage regardless of order size, volume, volatility, or time of day. Large orders on illiquid stocks will slip far more. | **MEDIUM** | `backtesting/engine.py:150-172` | Make slippage volume/volatility-aware: `slippage = base_slippage * (order_size / avg_volume) * volatility_factor`. Add separate bid-ask spread model. | 6h |
| T11 | **Sharpe ratio ignores risk-free rate** — Formula: `sharpe = (mean_r / std_r) * sqrt(annual_factor)`. Should be `(mean_r - rf_rate) / std_r * sqrt(annual_factor)`. With rf ~4.5% annualized, Sharpe is overstated by ~0.5-1.0. | **MEDIUM** | `backtesting/engine.py:504-530` | Add `risk_free_rate` parameter to config (default 0.045). Subtract from mean return before dividing by std. Document annualization assumptions for crypto vs equities. | 2h |
| T12 | **RSI uses SMA instead of EMA** — Industry standard RSI (Wilder) uses exponential smoothing for gain/loss averages. Code uses `rolling(period).mean()` (simple MA). Values differ ~10-20% from TA-Lib/TradingView. | **MEDIUM** | `data/market_data_fetcher.py:207-208`, `backtesting/strategies.py:194-200` | Use `ewm(span=period, adjust=False).mean()` instead of `rolling(period).mean()`. Or document explicitly that non-standard SMA-RSI is used. | 1h |
| T13 | **Timezone ambiguity** — Code assumes UTC, but yfinance returns market-local time (NYSE = EST). Session detection (`is_london_session`, `is_ny_session`) uses raw hours without timezone conversion. | **MEDIUM** | `data/market_data_fetcher.py:256-268`, `backtesting/data_loader.py:592-668` | Explicitly convert all timestamps to UTC after yfinance download. Document timezone assumption. Validate session detection hours against UTC. | 3h |
| T14 | **Data gap detection missing** — No warning if data jumps multiple days (e.g., 3-day gap without holiday). Equity curve and drawdown calculations silently skip gaps. | **MEDIUM** | `backtesting/data_loader.py:336-346` | Detect consecutive timestamp gaps > 1.5x expected frequency. Log warning with gap details. Optionally forward-fill or interpolate. | 3h |
| T15 | **Position sizing not volatility-adjusted** — Fixed fraction sizing (e.g., 2% of equity) regardless of market volatility. 2% position on VIX=10 vs VIX=40 has very different risk exposure. | **MEDIUM** | `backtesting/engine.py:336-343` | Scale position size by inverse volatility: `adjusted_size = base_size * (target_vol / realized_vol)`. Track portfolio heat across all positions. | 5h |
| T16 | **No partial fill handling** — Code assumes positions open fully or not at all. Real brokers can partially fill, especially on illiquid symbols or large orders. | **MEDIUM** | `backtesting/engine.py` (not addressed) | Add partial fill support with fill probability model based on order size vs average volume. | 4h |
| T17 | **No order latency modeling** — Paper trader fetches data, decides, and fills in same cycle. Real execution has ~5-15s latency (data fetch + ML inference + order routing). | **MEDIUM** | `trading/paper_trader.py:280-342` | Add `execution_latency_bars` parameter. Fill order N bars after signal, not on signal bar. | 2h |
| T18 | **Order lifecycle incomplete** — Broker adapter defines order states but no retry on rejection, no timeout on pending, no cancelled order cleanup. | **MEDIUM** | `executor/broker_adapter.py` | Implement order queue with auto-retry on rejection, 5-min timeout on pending, partial fill merging. | 5h |
| T19 | **No market regime detection** — Strategy parameters are static. No detection of bull/bear, high/low volatility, trending/ranging regimes. Strategy that works in bull may fail in bear. | **MEDIUM** | Project-wide (not implemented) | Add regime detector using VIX level, ADX trend strength, return autocorrelation, volatility quantile. Adjust strategy parameters per regime. | 6h |
| T20 | **Risk parameters defined but not enforced** — `RiskConfig` defines `max_drawdown_pct=0.15`, `max_portfolio_risk=0.06`, `max_correlated_positions=3` but none are checked during execution. | **MEDIUM** | `config/settings.py:184-193` | Wire risk checks into backtester and paper trader main loops. Log violations. Optionally halt trading on breach. | 3h |
| T21 | **Minimal data validation** — No checks for negative prices, `High < Low` inversions, zero-volume bars, or extreme outliers. Single bad bar can corrupt equity curve. | **MEDIUM** | `backtesting/data_loader.py:670-699` | Add assertions: `High >= Low`, `Close > 0`, `Volume >= 0`. Flag bars with > 20% gap from previous close as potential data errors. | 2h |
| T22 | **Backtest test coverage ~40%** — Tests use synthetic 10-100 bar data. No tests for multi-month backtests, extreme conditions, drawdown calculations, or edge cases. | **MEDIUM** | `tests/test_backtest_engine.py` | Add tests for 1-year real data, market gaps, drawdown accuracy, profit factor edge cases. Target 80% coverage per project standard. | 6h |
| T23 | **Sortino ratio ignores target return** — Formula uses `mean_r / downside_std` instead of `(mean_r - target_return) / downside_std`. Minor overstatement. | **LOW** | `backtesting/engine.py:532-535` | Add `target_return` parameter (default 0% or risk-free rate). | 1h |
| T24 | **Monthly returns doesn't flag partial months** — If backtest starts mid-month, first month's return is partial but reported alongside full months. | **LOW** | `backtesting/engine.py:548-556` | Flag or exclude partial months in monthly return series. | 1h |

---

## 3. Priority Matrix

### Immediate (fix before any demo or paper trading)

| # | Issue | Category | Effort |
|---|-------|----------|--------|
| S1 | Command injection in PowerShell alerts | Security | 2-3h |
| S2 | Pickle deserialization RCE | Security | 4-6h |
| T1 | ML models 100% accuracy (overfitting) | Trading | 20h+ |
| T2 | Kelly criterion hardcoded | Trading | 4h |

### Before live trading (paper trading OK with caveats)

| # | Issue | Category | Effort |
|---|-------|----------|--------|
| S3 | XSS in dashboard | Security | 2-3h |
| S4 | Weak dashboard auth | Security | 1-2h |
| S5 | Path traversal in model loading | Security | 2h |
| S6 | Insufficient security logging | Security | 3-4h |
| T3 | Agent stubs in consensus | Trading | 10h+ |
| T4 | No walk-forward optimization | Trading | 6h |
| T5 | No ML cross-validation | Trading | 8h |
| T6 | No dividend/split adjustment | Trading | 4h |
| T7 | Max drawdown not enforced | Trading | 2h |
| T8 | Look-ahead bias in bfill() | Trading | 1h |

### Before production deployment

All MEDIUM findings from both tables (~50-60h combined).

### Total Estimated Remediation

| Priority | Security | Trading | Combined |
|----------|----------|---------|----------|
| CRITICAL | 6-9h | 24h+ | 30-33h |
| HIGH | 8-11h | 31h+ | 39-42h |
| MEDIUM | 6-8h | 45h | 51-53h |
| LOW | 2.5h | 2h | 4.5h |
| **Total** | **~27-35h** | **~105h+** | **~130h+** |

---

## 4. Conclusions

### Strengths

- **Credential management**: All secrets via env vars, never hardcoded
- **RBAC framework**: Well-designed Permission/UserRole/requires_permission system
- **Decimal precision**: Money/price calculations use `decimal.Decimal` throughout
- **Strategy interface**: Clean `StrategyBase` ABC with proper signal generation
- **Feature engineering**: 60+ features calculated (RSI, MACD, Bollinger, session, volume)
- **Multi-agent architecture**: Consensus-based decision making is sound in principle
- **Rate limiting**: Alpaca adapter has per-instance rate limiter
- **TLS enforcement**: BrokerAuthManager enforces TLS 1.2+

### Verdict

| Use Case | Ready? |
|----------|--------|
| Educational demos | Yes (with caveats about accuracy claims) |
| MVP investor presentation | Yes (disable "80plus" accuracy display) |
| Paper trading exploration | Yes (after fixing S1, S2, T2, T8) |
| Strategy prototyping | Yes (after fixing T8, T11, T12) |
| Live money trading | **NO** (fix all CRITICAL + HIGH first) |
| Regulatory submission | **NO** (insufficient audit trail, missing controls) |
| Performance comparison vs other systems | **NO** (metrics are non-standard) |

### OWASP / CWE Compliance

- **CWE-78** (OS Command Injection): Present in S1
- **CWE-502** (Insecure Deserialization): Present in S2
- **CWE-79** (XSS): Present in S3
- **CWE-22** (Path Traversal): Present in S5
- **CWE-307** (Brute Force): No login rate limiting (S4)

---

*Report generated: 2026-03-07*
*Auditors: Security Specialist (simulated), Quant Specialist (simulated)*
*Mode: READ-ONLY (no code changes applied)*
