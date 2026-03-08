"""
SCRIPT NAME: news_sentiment.py
====================================
Execution Location: market-hawk-mvp/agents/news_analyzer/
Purpose: News Analyzer Agent — Real-time sentiment from financial news
Hardware Optimization: Intel i7-9700F, NVIDIA GTX 1070 8GB VRAM, 64GB DDR4
Creation Date: 2026-03-01

Fetches financial news headlines and analyzes sentiment using:
    1. RSS feeds (Yahoo Finance, Google News, Reuters)
    2. Local LLM (qwen3:8b via Ollama) for sentiment scoring
    3. Fallback: keyword-based sentiment when LLM unavailable

Returns: BUY/SELL/HOLD with confidence based on news sentiment.
"""

import re
import logging
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

logger = logging.getLogger("market_hawk.news_analyzer")


# ============================================================
# KEYWORD SENTIMENT (fast fallback)
# ============================================================

BULLISH_KEYWORDS = [
    "surge", "soar", "rally", "breakout", "upgrade", "beat", "exceeded",
    "record high", "all-time high", "bullish", "outperform", "growth",
    "strong earnings", "positive", "optimistic", "recovery", "boom",
    "buy rating", "price target raised", "momentum", "upside",
    "beat expectations", "strong demand", "expansion", "profit",
]

BEARISH_KEYWORDS = [
    "crash", "plunge", "decline", "downgrade", "miss", "warning",
    "record low", "bearish", "underperform", "recession", "slump",
    "weak earnings", "negative", "pessimistic", "selloff", "sell-off",
    "sell rating", "price target cut", "downturn", "downside",
    "missed expectations", "weak demand", "contraction", "loss",
    "bankruptcy", "layoffs", "investigation", "fraud", "default",
]


def keyword_sentiment(text: str) -> float:
    """
    Quick keyword-based sentiment. Returns -1.0 to +1.0
    """
    text_lower = text.lower()
    bull_count = sum(1 for kw in BULLISH_KEYWORDS if kw in text_lower)
    bear_count = sum(1 for kw in BEARISH_KEYWORDS if kw in text_lower)
    total = bull_count + bear_count
    if total == 0:
        return 0.0
    return (bull_count - bear_count) / total


# ============================================================
# NEWS FETCHER
# ============================================================

class NewsFetcher:
    """Fetches financial news from RSS feeds."""

    # RSS feeds for financial news
    RSS_FEEDS = {
        "yahoo": "https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US",
        "google": "https://news.google.com/rss/search?q={symbol}+stock&hl=en-US&gl=US&ceid=US:en",
    }

    def __init__(self):
        self._cache = {}
        self._cache_ttl = 300  # 5 minutes

    def fetch_news(self, symbol: str, max_articles: int = 10) -> List[Dict]:
        """
        Fetch recent news headlines for a symbol.

        Returns list of dicts: [{title, source, published, link}]
        """
        # Check cache
        cache_key = symbol.upper()
        if cache_key in self._cache:
            cached_time, cached_data = self._cache[cache_key]
            if time.time() - cached_time < self._cache_ttl:
                logger.info("News cache hit for %s (%d articles)", symbol, len(cached_data))
                return cached_data

        articles = []

        # Try feedparser (RSS)
        try:
            import feedparser

            for feed_name, feed_url in self.RSS_FEEDS.items():
                url = feed_url.format(symbol=symbol)
                try:
                    import requests as _req
                    resp = _req.get(url, timeout=10)
                    feed = feedparser.parse(resp.content)
                    for entry in feed.entries[:max_articles]:
                        articles.append({
                            "title": entry.get("title", ""),
                            "summary": entry.get("summary", "")[:200],
                            "source": feed_name,
                            "published": entry.get("published", ""),
                            "link": entry.get("link", ""),
                        })
                except Exception:
                    logger.exception("RSS feed %s failed for %s", feed_name, symbol)

        except ImportError:
            logger.warning("feedparser not installed — using fallback")

        # Fallback: yfinance news
        if not articles:
            try:
                import yfinance as yf
                from data.market_data_fetcher import get_yfinance_ticker

                ticker = yf.Ticker(get_yfinance_ticker(symbol))
                news = ticker.news or []
                for item in news[:max_articles]:
                    articles.append({
                        "title": item.get("title", ""),
                        "summary": item.get("summary", item.get("title", ""))[:200],
                        "source": item.get("publisher", "yfinance"),
                        "published": datetime.fromtimestamp(
                            item.get("providerPublishTime", 0)
                        ).isoformat() if item.get("providerPublishTime") else "",
                        "link": item.get("link", ""),
                    })
            except Exception:
                logger.exception("yfinance news fallback failed for %s", symbol)

        # Cache results
        self._cache[cache_key] = (time.time(), articles)
        logger.info("Fetched %d news articles for %s", len(articles), symbol)
        return articles


# ============================================================
# NEWS ANALYZER AGENT
# ============================================================

class NewsAnalyzer:
    """
    News Analyzer Agent — Sentiment analysis from financial news.

    Uses LLM (qwen3:8b via Ollama) for nuanced sentiment scoring,
    with keyword-based fallback for speed.

    Brain-compatible: implements analyze(symbol, context) -> Dict
    """

    def __init__(self, use_llm: bool = True):
        self._fetcher = NewsFetcher()
        self._use_llm = use_llm
        self._llm = None

    def _init_llm(self):
        """Lazy-load Ollama LLM for sentiment analysis."""
        if self._llm is not None:
            return True
        try:
            from langchain_ollama import ChatOllama
            self._llm = ChatOllama(
                model="qwen3:8b",
                temperature=0.1,
                base_url="http://localhost:11434",
                num_predict=150,
            )
            logger.info("News Analyzer LLM initialized (qwen3:8b)")
            return True
        except Exception:
            logger.exception("LLM init failed, using keyword fallback")
            self._use_llm = False
            return False

    def _llm_sentiment(self, headlines: List[str], symbol: str) -> Dict:
        """
        Use LLM to analyze sentiment of news headlines.

        Returns: {sentiment: float (-1 to +1), reasoning: str}
        """
        if not self._init_llm():
            return None

        headlines_text = "\n".join(f"- {h}" for h in headlines[:8])

        prompt = f"""Analyze the sentiment of these financial news headlines for {symbol}.

Headlines:
{headlines_text}

Respond with ONLY a JSON object (no other text):
{{"sentiment": <float from -1.0 (very bearish) to +1.0 (very bullish)>, "reasoning": "<one sentence summary>"}}"""

        try:
            response = self._llm.invoke(prompt)
            text = response.content.strip()

            # Extract JSON from response
            # Remove thinking tags if present
            if "</think>" in text:
                text = text.split("</think>")[-1].strip()

            # Find JSON in response
            json_match = re.search(r'\{[^}]+\}', text)
            if json_match:
                import json
                result = json.loads(json_match.group())
                sentiment = float(result.get("sentiment", 0.0))
                sentiment = max(-1.0, min(1.0, sentiment))  # Clamp
                return {
                    "sentiment": sentiment,
                    "reasoning": result.get("reasoning", "LLM analysis"),
                }
        except Exception:
            logger.exception("LLM sentiment failed")

        return None

    def analyze_sentiment(self, symbol: str) -> Dict[str, Any]:
        """
        Full sentiment analysis pipeline for a symbol.

        Returns:
            {sentiment: float, recommendation: str, confidence: float,
             headlines_count: int, reasoning: str}
        """
        # Fetch news
        articles = self._fetcher.fetch_news(symbol)

        if not articles:
            return {
                "sentiment": 0.0,
                "recommendation": "HOLD",
                "confidence": 0.0,
                "headlines_count": 0,
                "reasoning": f"No news available for {symbol}",
            }

        headlines = [a["title"] for a in articles if a["title"]]

        # Try LLM sentiment first
        llm_result = None
        if self._use_llm:
            llm_result = self._llm_sentiment(headlines, symbol)

        if llm_result:
            sentiment = llm_result["sentiment"]
            reasoning = llm_result["reasoning"]
            method = "LLM"
        else:
            # Keyword fallback
            all_text = " ".join(headlines)
            sentiment = keyword_sentiment(all_text)
            reasoning = f"Keyword analysis of {len(headlines)} headlines"
            method = "keyword"

        # Map sentiment to recommendation
        if sentiment > 0.2:
            recommendation = "BUY"
        elif sentiment < -0.2:
            recommendation = "SELL"
        else:
            recommendation = "HOLD"

        # Confidence based on strength of sentiment and article count
        base_confidence = min(abs(sentiment), 1.0)
        article_factor = min(len(articles) / 5, 1.0)  # More articles = more confident
        confidence = base_confidence * 0.7 + article_factor * 0.3

        return {
            "sentiment": sentiment,
            "recommendation": recommendation,
            "confidence": confidence,
            "headlines_count": len(headlines),
            "reasoning": f"[{method}] {reasoning} ({len(headlines)} headlines, "
                         f"sentiment={sentiment:+.2f})",
            "top_headlines": headlines[:3],
        }

    # Brain-compatible interface
    def analyze(self, symbol: str, context: Dict = None) -> Dict:
        """Brain-compatible: returns recommendation dict."""
        result = self.analyze_sentiment(symbol)
        return {
            "recommendation": result["recommendation"],
            "confidence": result["confidence"],
            "reasoning": result["reasoning"],
            "metadata": {
                "sentiment": result["sentiment"],
                "headlines_count": result["headlines_count"],
                "top_headlines": result.get("top_headlines", []),
            }
        }

    def cleanup(self):
        self._llm = None
        self._fetcher._cache.clear()
        logger.info("News Analyzer cleanup complete")


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    print("\n" + "=" * 60)
    print("NEWS ANALYZER — Sentiment Analysis Test")
    print("=" * 60)

    analyzer = NewsAnalyzer(use_llm=False)  # Keyword only for speed

    for symbol in ["AAPL", "NVDA", "BTCUSDT", "GOLD"]:
        print(f"\n--- {symbol} ---")
        result = analyzer.analyze_sentiment(symbol)
        emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⚪"}.get(result["recommendation"], "❓")
        print(f"  {emoji} {result['recommendation']} | "
              f"sentiment={result['sentiment']:+.2f} | "
              f"conf={result['confidence']:.2f} | "
              f"headlines={result['headlines_count']}")
        print(f"  {result['reasoning']}")
        for h in result.get("top_headlines", [])[:2]:
            print(f"    📰 {h[:80]}")

    print("\n✅ Test complete")
    analyzer.cleanup()
