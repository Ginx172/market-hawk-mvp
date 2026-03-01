"""
SCRIPT NAME: chart_engine.py
====================================
Execution Location: market-hawk-mvp/dashboard/
Purpose: Interactive charting with technical analysis overlays
Creation Date: 2026-03-01

Provides Plotly-based candlestick charts with TA overlays:
    - Moving Averages (SMA/EMA 5,10,20,50,200)
    - Bollinger Bands
    - Ichimoku Cloud
    - MACD + Signal + Histogram
    - RSI
    - Fibonacci Retracement
    - Volume Profile
    - Gann Fan (angular lines)
    - Wyckoff phases (volume/price annotations)
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Optional


# ============================================================
# INDICATOR CALCULATIONS
# ============================================================

def calc_ichimoku(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate Ichimoku Cloud components."""
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)

    # Tenkan-sen (Conversion Line) — 9 periods
    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
    # Kijun-sen (Base Line) — 26 periods
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
    # Senkou Span A (Leading Span A) — shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    # Senkou Span B (Leading Span B) — shifted 26 periods ahead
    senkou_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    # Chikou Span (Lagging Span) — shifted 26 periods back
    chikou = close.shift(-26)

    return pd.DataFrame({
        "tenkan": tenkan, "kijun": kijun,
        "senkou_a": senkou_a, "senkou_b": senkou_b,
        "chikou": chikou,
    }, index=df.index)


def calc_fibonacci(df: pd.DataFrame, lookback: int = 100) -> dict:
    """Calculate Fibonacci retracement levels from recent swing."""
    recent = df.tail(lookback)
    high = float(recent["High"].max())
    low = float(recent["Low"].min())
    diff = high - low

    levels = {
        "0.0%": high,
        "23.6%": high - 0.236 * diff,
        "38.2%": high - 0.382 * diff,
        "50.0%": high - 0.500 * diff,
        "61.8%": high - 0.618 * diff,
        "78.6%": high - 0.786 * diff,
        "100.0%": low,
    }
    return levels


def calc_gann_angles(df: pd.DataFrame) -> dict:
    """Calculate Gann fan angles from a pivot point."""
    recent = df.tail(100)
    pivot_idx = recent["Low"].idxmin()
    pivot_price = float(recent.loc[pivot_idx, "Low"])

    # Price range per bar for scaling
    price_range = float(recent["High"].max() - recent["Low"].min())
    bars = len(recent)
    scale = price_range / bars if bars > 0 else 1

    angles = {
        "1x1 (45°)": scale * 1.0,
        "1x2": scale * 0.5,
        "2x1": scale * 2.0,
        "1x3": scale * 0.333,
        "3x1": scale * 3.0,
    }
    return {"pivot_idx": pivot_idx, "pivot_price": pivot_price, "angles": angles}


# ============================================================
# CHART BUILDER
# ============================================================

def build_chart(df: pd.DataFrame, symbol: str,
                indicators: List[str] = None,
                chart_type: str = "candlestick") -> go.Figure:
    """
    Build interactive Plotly chart with selected indicators.

    Args:
        df: DataFrame with OHLCV + computed features
        symbol: Symbol name for title
        indicators: List of indicator names to overlay
        chart_type: "candlestick" or "line"

    Returns:
        Plotly Figure
    """
    indicators = indicators or []

    # Determine layout — extra subplots for MACD, RSI, Volume
    has_macd = "MACD" in indicators
    has_rsi = "RSI" in indicators
    has_volume = "Volume" in indicators

    n_rows = 1 + has_macd + has_rsi + has_volume
    row_heights = [0.5]
    if has_volume:
        row_heights.append(0.12)
    if has_rsi:
        row_heights.append(0.15)
    if has_macd:
        row_heights.append(0.18)

    # Normalize
    total = sum(row_heights)
    row_heights = [h / total for h in row_heights]

    subplot_titles = [f"{symbol}"]
    if has_volume:
        subplot_titles.append("Volume")
    if has_rsi:
        subplot_titles.append("RSI")
    if has_macd:
        subplot_titles.append("MACD")

    fig = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
        subplot_titles=subplot_titles,
    )

    close = df["Close"].astype(float)
    x = df.index if isinstance(df.index, pd.DatetimeIndex) else list(range(len(df)))

    # --- Main chart ---
    if chart_type == "candlestick":
        fig.add_trace(go.Candlestick(
            x=x,
            open=df["Open"].astype(float),
            high=df["High"].astype(float),
            low=df["Low"].astype(float),
            close=close,
            name=symbol,
            increasing_line_color="#00d26a",
            decreasing_line_color="#ff4757",
        ), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(
            x=x, y=close, mode="lines",
            name=symbol, line=dict(color="#4a6cf7", width=2),
        ), row=1, col=1)

    current_row = 1

    # --- Moving Averages ---
    if "Moving Averages" in indicators:
        ma_colors = {
            "SMA_5": "#ffeb3b", "SMA_10": "#ff9800", "SMA_20": "#2196f3",
            "SMA_50": "#9c27b0", "SMA_200": "#f44336",
            "EMA_5": "#ffeb3b", "EMA_10": "#ff9800", "EMA_20": "#2196f3",
        }
        for ma_name, color in ma_colors.items():
            if ma_name in df.columns:
                fig.add_trace(go.Scatter(
                    x=x, y=df[ma_name].astype(float),
                    mode="lines", name=ma_name,
                    line=dict(color=color, width=1, dash="dot" if "EMA" in ma_name else "solid"),
                    opacity=0.7,
                ), row=1, col=1)

    # --- Bollinger Bands ---
    if "Bollinger Bands" in indicators:
        for col_name in ["BB_upper", "BB_middle", "BB_lower"]:
            if col_name in df.columns:
                style = dict(color="#7c4dff", width=1)
                if col_name == "BB_middle":
                    style["dash"] = "dash"
                fig.add_trace(go.Scatter(
                    x=x, y=df[col_name].astype(float),
                    mode="lines", name=col_name,
                    line=style, opacity=0.6,
                ), row=1, col=1)

        # Fill between bands
        if "BB_upper" in df.columns and "BB_lower" in df.columns:
            fig.add_trace(go.Scatter(
                x=list(x) + list(x)[::-1],
                y=list(df["BB_upper"].astype(float)) + list(df["BB_lower"].astype(float))[::-1],
                fill="toself", fillcolor="rgba(124,77,255,0.08)",
                line=dict(width=0), showlegend=False, name="BB Fill",
            ), row=1, col=1)

    # --- Ichimoku Cloud ---
    if "Ichimoku" in indicators:
        ichi = calc_ichimoku(df)
        fig.add_trace(go.Scatter(x=x, y=ichi["tenkan"], mode="lines",
                                  name="Tenkan", line=dict(color="#2196f3", width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=ichi["kijun"], mode="lines",
                                  name="Kijun", line=dict(color="#f44336", width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=ichi["senkou_a"], mode="lines",
                                  name="Senkou A", line=dict(color="#4caf50", width=0.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=ichi["senkou_b"], mode="lines",
                                  name="Senkou B", line=dict(color="#ff5722", width=0.5),
                                  fill="tonexty", fillcolor="rgba(76,175,80,0.1)"), row=1, col=1)

    # --- Fibonacci Retracement ---
    if "Fibonacci" in indicators:
        fib = calc_fibonacci(df)
        colors = ["#f44336", "#ff9800", "#ffeb3b", "#4caf50", "#2196f3", "#9c27b0", "#f44336"]
        for (level_name, price), color in zip(fib.items(), colors):
            fig.add_hline(y=price, line_dash="dash", line_color=color,
                          annotation_text=f"Fib {level_name} ({price:.2f})",
                          annotation_position="right",
                          row=1, col=1)

    # --- Gann Fan ---
    if "Gann" in indicators:
        gann = calc_gann_angles(df)
        pivot = gann["pivot_price"]
        pivot_pos = list(x).index(gann["pivot_idx"]) if gann["pivot_idx"] in list(x) else len(df) // 2
        gann_colors = ["#ffeb3b", "#ff9800", "#f44336", "#4caf50", "#2196f3"]

        for (name, slope), color in zip(gann["angles"].items(), gann_colors):
            end_pos = min(pivot_pos + 50, len(df) - 1)
            y_start = pivot
            y_end = pivot + slope * (end_pos - pivot_pos)
            fig.add_trace(go.Scatter(
                x=[x[pivot_pos], x[end_pos]], y=[y_start, y_end],
                mode="lines", name=f"Gann {name}",
                line=dict(color=color, width=1, dash="dot"),
            ), row=1, col=1)

    # --- Volume subplot ---
    if has_volume:
        current_row += 1
        colors = ["#00d26a" if c >= o else "#ff4757"
                  for c, o in zip(close, df["Open"].astype(float))]
        fig.add_trace(go.Bar(
            x=x, y=df["Volume"].astype(float),
            name="Volume", marker_color=colors, opacity=0.7,
        ), row=current_row, col=1)

    # --- RSI subplot ---
    if has_rsi:
        current_row += 1
        rsi_col = "RSI" if "RSI" in df.columns else "rsi"
        if rsi_col in df.columns:
            fig.add_trace(go.Scatter(
                x=x, y=df[rsi_col].astype(float),
                mode="lines", name="RSI",
                line=dict(color="#7c4dff", width=2),
            ), row=current_row, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="#ff4757",
                          annotation_text="Overbought", row=current_row, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="#00d26a",
                          annotation_text="Oversold", row=current_row, col=1)
            fig.add_hrect(y0=30, y1=70, fillcolor="rgba(124,77,255,0.05)",
                          line_width=0, row=current_row, col=1)

    # --- MACD subplot ---
    if has_macd:
        current_row += 1
        if "MACD" in df.columns:
            fig.add_trace(go.Scatter(
                x=x, y=df["MACD"].astype(float),
                mode="lines", name="MACD",
                line=dict(color="#2196f3", width=2),
            ), row=current_row, col=1)
        if "MACD_signal" in df.columns:
            fig.add_trace(go.Scatter(
                x=x, y=df["MACD_signal"].astype(float),
                mode="lines", name="Signal",
                line=dict(color="#ff9800", width=1),
            ), row=current_row, col=1)
        if "MACD_histogram" in df.columns:
            hist = df["MACD_histogram"].astype(float)
            colors = ["#00d26a" if v >= 0 else "#ff4757" for v in hist]
            fig.add_trace(go.Bar(
                x=x, y=hist, name="MACD Hist",
                marker_color=colors, opacity=0.6,
            ), row=current_row, col=1)

    # --- Layout ---
    fig.update_layout(
        template="plotly_dark",
        height=200 + n_rows * 250,
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1, font=dict(size=10),
        ),
        margin=dict(l=60, r=20, t=60, b=30),
        xaxis_rangeslider_visible=False,
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
    )

    # Grid styling
    for i in range(1, n_rows + 1):
        fig.update_yaxes(gridcolor="#1e2433", row=i, col=1)
        fig.update_xaxes(gridcolor="#1e2433", row=i, col=1)

    return fig
