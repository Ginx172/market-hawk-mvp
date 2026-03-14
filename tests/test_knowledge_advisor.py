"""
Unit tests for Knowledge Advisor RAG module.
Tests with mocked ChromaDB and Ollama to run without external services.
"""
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================
# Helpers — lightweight stand-ins for RAGResult / RAGResponse
# ============================================================

@dataclass
class _MockSource:
    """Minimal source object for CitationFormatter tests."""
    text: str = ""
    source: str = "unknown"
    page: Optional[int] = None


# ============================================================
# TestRAGDataStructures
# ============================================================

class TestRAGDataStructures:
    """Test RAGResult and RAGResponse dataclass creation and defaults."""

    def test_rag_result_creation(self):
        from agents.knowledge_advisor.rag_engine import RAGResult

        r = RAGResult(text="some text", source="book.pdf")
        assert r.text == "some text"
        assert r.source == "book.pdf"
        assert r.page is None
        assert r.relevance_score == 0.0
        assert r.metadata == {}

    def test_rag_result_with_all_fields(self):
        from agents.knowledge_advisor.rag_engine import RAGResult

        r = RAGResult(
            text="chunk", source="Trading 101", page=42,
            relevance_score=0.85, metadata={"key": "value"}
        )
        assert r.page == 42
        assert r.relevance_score == 0.85
        assert r.metadata["key"] == "value"

    def test_rag_response_defaults(self):
        from agents.knowledge_advisor.rag_engine import RAGResponse

        resp = RAGResponse(answer="hello", sources=[], query="q")
        assert resp.response_time == 0.0
        assert resp.chunks_retrieved == 0

    def test_rag_response_with_sources(self):
        from agents.knowledge_advisor.rag_engine import RAGResult, RAGResponse

        src = RAGResult(text="t", source="s")
        resp = RAGResponse(
            answer="ans", sources=[src], query="q",
            response_time=1.5, chunks_retrieved=1
        )
        assert len(resp.sources) == 1
        assert resp.response_time == 1.5


# ============================================================
# TestQueryCache
# ============================================================

class TestQueryCache:
    """Test QueryCache: get/put, LRU eviction, TTL expiry, stats, clear."""

    def setup_method(self):
        from agents.knowledge_advisor.query_cache import QueryCache
        self.cache = QueryCache(max_size=3, ttl_seconds=60.0)

    def test_get_miss_on_empty_cache(self):
        assert self.cache.get("anything", 5) is None

    def test_put_and_get(self):
        self.cache.put("order blocks", 5, ["result1", "result2"])
        result = self.cache.get("order blocks", 5)
        assert result == ["result1", "result2"]

    def test_key_is_normalized(self):
        """Trailing spaces and case should not affect cache lookup."""
        self.cache.put("Order Blocks ", 5, ["r"])
        assert self.cache.get("order blocks", 5) == ["r"]
        assert self.cache.get("  order blocks  ", 5) == ["r"]

    def test_n_results_is_part_of_key(self):
        self.cache.put("query", 5, ["five"])
        self.cache.put("query", 10, ["ten"])
        assert self.cache.get("query", 5) == ["five"]
        assert self.cache.get("query", 10) == ["ten"]

    def test_lru_eviction(self):
        self.cache.put("q1", 5, [1])
        self.cache.put("q2", 5, [2])
        self.cache.put("q3", 5, [3])
        # Access q1 to make it recently used
        self.cache.get("q1", 5)
        # Add q4 — should evict q2 (LRU)
        self.cache.put("q4", 5, [4])
        assert self.cache.get("q2", 5) is None
        assert self.cache.get("q1", 5) == [1]
        assert self.cache.get("q4", 5) == [4]

    def test_ttl_expiry(self):
        cache = __import__(
            "agents.knowledge_advisor.query_cache", fromlist=["QueryCache"]
        ).QueryCache(max_size=10, ttl_seconds=0.05)
        cache.put("q", 5, ["data"])
        assert cache.get("q", 5) == ["data"]
        time.sleep(0.1)
        assert cache.get("q", 5) is None  # expired

    def test_stats_hit_miss(self):
        self.cache.put("q", 5, ["r"])
        self.cache.get("q", 5)   # hit
        self.cache.get("miss", 5)  # miss
        s = self.cache.stats()
        assert s["hits"] == 1
        assert s["misses"] == 1
        assert s["hit_rate"] == 0.5

    def test_stats_empty(self):
        s = self.cache.stats()
        assert s["hit_rate"] == 0.0
        assert s["size"] == 0

    def test_clear(self):
        self.cache.put("q", 5, ["r"])
        self.cache.clear()
        assert self.cache.get("q", 5) is None
        assert self.cache.stats()["size"] == 0


# ============================================================
# TestCitationFormatter
# ============================================================

class TestCitationFormatter:
    """Test format_response, format_for_brain, extract_direction_hints."""

    def test_format_response_basic(self):
        from agents.knowledge_advisor.citation_formatter import CitationFormatter

        sources = [_MockSource(source="Trading Book", page=10)]
        result = CitationFormatter.format_response("My answer", sources, "q")
        assert "My answer" in result.text
        assert "📚 Sources:" in result.text
        assert "[1] Trading Book, p.10" in result.text
        assert result.source_count == 1

    def test_format_response_no_page(self):
        from agents.knowledge_advisor.citation_formatter import CitationFormatter

        sources = [_MockSource(source="Book A", page=None)]
        result = CitationFormatter.format_response("ans", sources, "q")
        assert "[1] Book A" in result.text
        assert "p." not in result.text

    def test_format_response_deduplicates_sources(self):
        from agents.knowledge_advisor.citation_formatter import CitationFormatter

        sources = [
            _MockSource(source="Same Book", page=5),
            _MockSource(source="Same Book", page=5),
        ]
        result = CitationFormatter.format_response("ans", sources, "q")
        assert result.source_count == 1

    def test_format_for_brain(self):
        from agents.knowledge_advisor.citation_formatter import CitationFormatter

        sources = [
            _MockSource(source="Book1", text="bullish divergence found"),
            _MockSource(source="Book2", text="strong buy signal here"),
        ]
        text = CitationFormatter.format_for_brain(sources, max_sources=2)
        assert "[Book1]" in text
        assert "[Book2]" in text
        assert " | " in text

    def test_format_for_brain_respects_max(self):
        from agents.knowledge_advisor.citation_formatter import CitationFormatter

        sources = [_MockSource(source=f"B{i}", text="x") for i in range(5)]
        text = CitationFormatter.format_for_brain(sources, max_sources=2)
        assert text.count(" | ") == 1  # 2 items → 1 separator

    def test_extract_direction_hints_bullish(self):
        from agents.knowledge_advisor.citation_formatter import CitationFormatter

        sources = [_MockSource(
            text="strong bullish breakout with buy signal and golden cross"
        )]
        hints = CitationFormatter.extract_direction_hints(sources)
        assert hints["direction"] == "BULLISH"
        assert hints["confidence"] > 0.0
        assert hints["bullish_signals"] > hints["bearish_signals"]

    def test_extract_direction_hints_bearish(self):
        from agents.knowledge_advisor.citation_formatter import CitationFormatter

        sources = [_MockSource(
            text="bearish distribution zone with death cross and sell signal breakdown"
        )]
        hints = CitationFormatter.extract_direction_hints(sources)
        assert hints["direction"] == "BEARISH"
        assert hints["confidence"] > 0.0
        assert hints["bearish_signals"] > hints["bullish_signals"]

    def test_extract_direction_hints_neutral(self):
        from agents.knowledge_advisor.citation_formatter import CitationFormatter

        # Balanced: equal bullish/bearish terms
        sources = [_MockSource(text="buy sell")]
        hints = CitationFormatter.extract_direction_hints(sources)
        assert hints["direction"] == "NEUTRAL"

    def test_extract_direction_hints_no_sources(self):
        from agents.knowledge_advisor.citation_formatter import CitationFormatter

        hints = CitationFormatter.extract_direction_hints([])
        assert hints["direction"] == "NEUTRAL"
        assert hints["confidence"] == 0.0


# ============================================================
# TestKnowledgeAdvisorInit
# ============================================================

class TestKnowledgeAdvisorInit:
    """Test KnowledgeAdvisor can be instantiated with a mock config."""

    def _mock_config(self):
        cfg = MagicMock()
        cfg.chromadb_path = "/tmp/fake_chroma"
        cfg.collection_name = "algo_trading"
        cfg.embedding_model = "nomic-embed-text"
        cfg.llm_model = "qwen3:8b"
        cfg.ollama_host = "http://localhost:11434"
        cfg.n_results = 15
        cfg.fetch_k = 60
        cfg.mmr_lambda = 0.7
        cfg.retrieval_type = "mmr"
        return cfg

    @patch("config.settings.RAG_CONFIG")
    def test_init_with_mock_config(self, _mock_rag_config):
        from agents.knowledge_advisor.rag_engine import KnowledgeAdvisor

        advisor = KnowledgeAdvisor(config=self._mock_config())
        assert advisor._initialized is False
        assert advisor._cache is not None

    def test_init_creates_query_cache(self):
        from agents.knowledge_advisor.rag_engine import KnowledgeAdvisor
        from agents.knowledge_advisor.query_cache import QueryCache

        advisor = KnowledgeAdvisor(config=self._mock_config())
        assert isinstance(advisor._cache, QueryCache)

    def test_get_stats_not_initialized(self):
        """get_stats() returns initialized=False when not yet connected."""
        from agents.knowledge_advisor.rag_engine import KnowledgeAdvisor

        cfg = self._mock_config()
        advisor = KnowledgeAdvisor(config=cfg)
        # Prevent actual initialize() call
        advisor._initialized = True
        advisor._collection_count = 99
        stats = advisor.get_stats()
        assert stats["initialized"] is True
        assert stats["total_chunks"] == 99

    def test_cleanup_clears_cache(self):
        from agents.knowledge_advisor.rag_engine import KnowledgeAdvisor

        advisor = KnowledgeAdvisor(config=self._mock_config())
        # Pre-populate cache
        advisor._cache.put("q", 5, ["data"])
        assert advisor._cache.stats()["size"] == 1
        advisor.cleanup()
        assert advisor._cache.stats()["size"] == 0
        assert advisor._initialized is False


# ============================================================
# TestKnowledgeAdvisorAnalyze
# ============================================================

class TestKnowledgeAdvisorAnalyze:
    """Test that analyze() returns proper dict structure with directional hints."""

    def _make_advisor_with_mock_retrieve(self, results):
        """Return a KnowledgeAdvisor whose retrieve() is mocked."""
        from agents.knowledge_advisor.rag_engine import KnowledgeAdvisor

        cfg = MagicMock()
        cfg.n_results = 15
        advisor = KnowledgeAdvisor(config=cfg)
        advisor._initialized = True
        advisor._collection_count = 130000
        advisor.retrieve = MagicMock(return_value=results)
        return advisor

    def test_analyze_returns_required_keys(self):
        from agents.knowledge_advisor.rag_engine import RAGResult

        sources = [RAGResult(text="bullish breakout buy signal", source="Book1")]
        advisor = self._make_advisor_with_mock_retrieve(sources)
        result = advisor.analyze("AAPL")
        assert "recommendation" in result
        assert "confidence" in result
        assert "reasoning" in result

    def test_analyze_hold_when_no_results(self):
        advisor = self._make_advisor_with_mock_retrieve([])
        result = advisor.analyze("AAPL")
        assert result["recommendation"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_analyze_buy_when_bullish_chunks(self):
        from agents.knowledge_advisor.rag_engine import RAGResult

        sources = [RAGResult(
            text="strong bullish breakout with buy signal golden cross uptrend higher high",
            source="Book1"
        )]
        advisor = self._make_advisor_with_mock_retrieve(sources)
        result = advisor.analyze("AAPL")
        assert result["recommendation"] == "BUY"
        assert result["confidence"] > 0.0

    def test_analyze_sell_when_bearish_chunks(self):
        from agents.knowledge_advisor.rag_engine import RAGResult

        sources = [RAGResult(
            text="bearish distribution death cross sell signal breakdown downtrend lower low",
            source="Book1"
        )]
        advisor = self._make_advisor_with_mock_retrieve(sources)
        result = advisor.analyze("AAPL")
        assert result["recommendation"] == "SELL"
        assert result["confidence"] > 0.0

    def test_analyze_metadata_contains_direction_hints(self):
        from agents.knowledge_advisor.rag_engine import RAGResult

        sources = [RAGResult(text="buy long bullish uptrend", source="Book1")]
        advisor = self._make_advisor_with_mock_retrieve(sources)
        result = advisor.analyze("AAPL")
        assert "metadata" in result
        assert "direction_hints" in result["metadata"]
        hints = result["metadata"]["direction_hints"]
        assert "direction" in hints
        assert "bullish_signals" in hints

    def test_analyze_passes_timeframe_context(self):
        from agents.knowledge_advisor.rag_engine import RAGResult

        sources = [RAGResult(text="buy signal", source="B")]
        advisor = self._make_advisor_with_mock_retrieve(sources)
        advisor.analyze("BTCUSDT", {"timeframe": "1h"})
        call_args = advisor.retrieve.call_args
        assert "1h" in call_args[0][0]  # positional arg: query string
