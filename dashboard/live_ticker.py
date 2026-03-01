"""
SCRIPT NAME: live_ticker.py
====================================
Execution Location: market-hawk-mvp/dashboard/
Purpose: Real-time EKG-style price ticker with auto-refresh
Creation Date: 2026-03-01

Provides a streaming price chart that updates automatically,
similar to a live trading terminal or EKG monitor.
"""

import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional


def fetch_live_data(fetcher, symbol: str, period: str = "5d",
                    interval: str = "1m") -> Optional[pd.DataFrame]:
    """Fetch latest candles for live display."""
    try:
        df = fetcher.fetch_and_engineer(symbol, period=period, interval=interval)
        return df
    except Exception:
        return None


def build_live_chart(df: pd.DataFrame, symbol: str,
                     chart_style: str = "line",
                     show_volume: bool = True,
                     show_ma: bool = True,
                     show_bid_ask: bool = True) -> go.Figure:
    """
    Build an EKG-style live price chart.

    Args:
        df: DataFrame with OHLCV data
        symbol: Symbol name
        chart_style: "line" for EKG-style, "candle" for candlestick, "area" for filled area
        show_volume: Show volume bars below
        show_ma: Show moving averages
        show_bid_ask: Show high/low as bid/ask band
    """
    n_rows = 1 + show_volume
    heights = [0.8, 0.2] if show_volume else [1.0]

    fig = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=heights,
    )

    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    x = df.index

    # Determine color based on price direction
    last_price = float(close.iloc[-1])
    first_price = float(close.iloc[0])
    is_bullish = last_price >= first_price
    main_color = "#00d26a" if is_bullish else "#ff4757"
    fill_color = "rgba(0,210,106,0.08)" if is_bullish else "rgba(255,71,87,0.08)"

    # --- Main price line (EKG style) ---
    if chart_style == "line":
        # Main price line — thick, colored by direction
        fig.add_trace(go.Scatter(
            x=x, y=close,
            mode="lines",
            name=f"{symbol} Price",
            line=dict(color=main_color, width=2.5),
            hovertemplate="%{x}<br>$%{y:,.2f}<extra></extra>",
        ), row=1, col=1)

        # Glow effect — wider transparent line behind
        fig.add_trace(go.Scatter(
            x=x, y=close,
            mode="lines",
            name="",
            line=dict(color=main_color, width=8),
            opacity=0.15,
            showlegend=False,
            hoverinfo="skip",
        ), row=1, col=1)

        # Fill to bottom for area effect
        fig.add_trace(go.Scatter(
            x=x, y=close,
            mode="lines",
            fill="tozeroy",
            fillcolor=fill_color,
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        ), row=1, col=1)

    elif chart_style == "candle":
        fig.add_trace(go.Candlestick(
            x=x,
            open=df["Open"].astype(float),
            high=high, low=low, close=close,
            name=symbol,
            increasing_line_color="#00d26a",
            decreasing_line_color="#ff4757",
        ), row=1, col=1)

    elif chart_style == "area":
        fig.add_trace(go.Scatter(
            x=x, y=close,
            mode="lines",
            fill="tozeroy",
            name=symbol,
            line=dict(color=main_color, width=1.5),
            fillcolor=fill_color,
        ), row=1, col=1)

    # --- Bid/Ask band (High/Low) ---
    if show_bid_ask and chart_style == "line":
        fig.add_trace(go.Scatter(
            x=x, y=high,
            mode="lines", name="High",
            line=dict(color="rgba(255,255,255,0.15)", width=0.5, dash="dot"),
            showlegend=False, hoverinfo="skip",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=x, y=low,
            mode="lines", name="Low",
            line=dict(color="rgba(255,255,255,0.15)", width=0.5, dash="dot"),
            fill="tonexty", fillcolor="rgba(255,255,255,0.02)",
            showlegend=False, hoverinfo="skip",
        ), row=1, col=1)

    # --- Moving Averages ---
    if show_ma:
        for col_name, color, dash in [
            ("SMA_5", "#ffeb3b", "solid"),
            ("SMA_20", "#2196f3", "solid"),
            ("SMA_50", "#9c27b0", "dash"),
        ]:
            if col_name in df.columns:
                fig.add_trace(go.Scatter(
                    x=x, y=df[col_name].astype(float),
                    mode="lines", name=col_name,
                    line=dict(color=color, width=1, dash=dash),
                    opacity=0.6,
                ), row=1, col=1)

    # --- Volume ---
    if show_volume:
        vol_colors = ["#00d26a" if c >= o else "#ff4757"
                      for c, o in zip(close, df["Open"].astype(float))]
        fig.add_trace(go.Bar(
            x=x, y=df["Volume"].astype(float),
            name="Volume", marker_color=vol_colors, opacity=0.5,
        ), row=2, col=1)

    # --- Current price annotation ---
    fig.add_annotation(
        x=x[-1], y=last_price,
        text=f"  ${last_price:,.2f}",
        showarrow=False,
        font=dict(color=main_color, size=14, family="monospace"),
        xanchor="left",
        bgcolor="rgba(14,17,23,0.8)",
        bordercolor=main_color,
        borderwidth=1,
        borderpad=4,
    )

    # Horizontal line at current price
    fig.add_hline(
        y=last_price, line_dash="dot",
        line_color=main_color, opacity=0.4,
        row=1, col=1,
    )

    # --- Layout ---
    change = last_price - first_price
    change_pct = change / first_price * 100 if first_price > 0 else 0
    title_color = main_color

    fig.update_layout(
        template="plotly_dark",
        height=500 + (150 if show_volume else 0),
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1, font=dict(size=10),
        ),
        margin=dict(l=60, r=120, t=50, b=30),
        xaxis_rangeslider_visible=False,
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        title=dict(
            text=f"<b>{symbol}</b>  "
                 f"<span style='color:{title_color}'>"
                 f"${last_price:,.2f}  {change:+,.2f} ({change_pct:+.2f}%)</span>",
            font=dict(size=18),
            x=0.01,
        ),
    )

    # Grid styling — subtle
    for i in range(1, n_rows + 1):
        fig.update_yaxes(
            gridcolor="rgba(255,255,255,0.05)",
            zeroline=False,
            row=i, col=1,
        )
        fig.update_xaxes(
            gridcolor="rgba(255,255,255,0.05)",
            zeroline=False,
            row=i, col=1,
        )

    # Range slider off, crosshair on
    fig.update_xaxes(
        rangeslider_visible=False,
        showspikes=True,
        spikecolor="rgba(255,255,255,0.3)",
        spikethickness=1,
        spikedash="dot",
        spikemode="across",
    )
    fig.update_yaxes(
        showspikes=True,
        spikecolor="rgba(255,255,255,0.3)",
        spikethickness=1,
        spikedash="dot",
        spikemode="across",
    )

    return fig


def build_multi_ticker(fetcher, symbols: list,
                       period: str = "1d", interval: str = "5m") -> go.Figure:
    """Build a multi-asset mini-chart grid (like a trading terminal)."""
    n = len(symbols)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=symbols,
        vertical_spacing=0.08,
        horizontal_spacing=0.05,
    )

    for i, symbol in enumerate(symbols):
        row = i // cols + 1
        col = i % cols + 1

        df = fetch_live_data(fetcher, symbol, period, interval)
        if df is None or df.empty:
            continue

        close = df["Close"].astype(float)
        x = df.index

        last = float(close.iloc[-1])
        first = float(close.iloc[0])
        color = "#00d26a" if last >= first else "#ff4757"

        fig.add_trace(go.Scatter(
            x=x, y=close,
            mode="lines",
            name=symbol,
            line=dict(color=color, width=1.5),
            fill="tozeroy",
            fillcolor=f"rgba({','.join(str(int(color.lstrip('#')[i:i+2], 16)) for i in (0,2,4))},0.08)",
            showlegend=False,
        ), row=row, col=col)

        # Price annotation
        change_pct = (last - first) / first * 100 if first > 0 else 0
        fig.add_annotation(
            x=x[-1], y=last,
            text=f"${last:,.2f} ({change_pct:+.1f}%)",
            showarrow=False,
            font=dict(color=color, size=10, family="monospace"),
            xanchor="left",
            row=row, col=col,
        )

    fig.update_layout(
        template="plotly_dark",
        height=250 * rows,
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        margin=dict(l=40, r=40, t=40, b=20),
    )

    for i in range(1, rows * cols + 1):
        fig.update_yaxes(showgrid=False, showticklabels=False, row=(i-1)//cols+1, col=(i-1)%cols+1)
        fig.update_xaxes(showgrid=False, showticklabels=False, row=(i-1)//cols+1, col=(i-1)%cols+1)

    return fig
