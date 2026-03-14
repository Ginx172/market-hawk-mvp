"""
Unit tests for build_rag_from_books.py utility functions.

Tests normalize_title, is_trading_relevant, chunk_text, estimate_tokens,
detect_chapters, EXCLUDE_DIRS contents, and the --exclude-dir CLI argument.
All tests run without external services (no ChromaDB, no Ollama).
"""
import argparse
import sys
from pathlib import Path

import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the module under test (functions, not the CLI entry point)
from scripts.build_rag_from_books import (
    EXCLUDE_DIRS,
    normalize_title,
    is_trading_relevant,
    chunk_text,
    estimate_tokens,
    detect_chapters,
    build_parser,
)


# ============================================================
# EXCLUDE_DIRS — static contents
# ============================================================

class TestExcludeDirs:
    """Verify expected entries are present in EXCLUDE_DIRS."""

    def test_asta_nu_is_excluded(self):
        assert "asta_nu" in EXCLUDE_DIRS

    def test_arhive_is_excluded(self):
        assert "arhive" in EXCLUDE_DIRS

    def test_legacy_medical_entries_present(self):
        """Existing legacy entries must not have been removed."""
        assert "nhs" in EXCLUDE_DIRS
        assert "fitness" in EXCLUDE_DIRS
        assert "cooking" in EXCLUDE_DIRS


# ============================================================
# normalize_title
# ============================================================

class TestNormalizeTitle:
    """Test filename → clean title normalisation."""

    def test_removes_extension(self):
        result = normalize_title("Technical_Analysis.pdf")
        assert not result.endswith(".pdf")

    def test_lowercases(self):
        result = normalize_title("AlgoTrading.pdf")
        assert result == result.lower()

    def test_removes_z_library_marker(self):
        result = normalize_title("Trading_Bible_Z-Lib.pdf")
        assert "z-lib" not in result
        assert "zlib" not in result

    def test_removes_pdfdrive_marker(self):
        result = normalize_title("Price_Action_PDFDrive.pdf")
        assert "pdfdrive" not in result

    def test_removes_parenthetical_year(self):
        result = normalize_title("Market_Wizard(2019).pdf")
        assert "2019" not in result

    def test_replaces_underscores_with_spaces(self):
        result = normalize_title("algo_trading_guide.pdf")
        assert "_" not in result

    def test_strips_whitespace(self):
        result = normalize_title("  some_book.pdf  ")
        assert result == result.strip()

    def test_plain_filename(self):
        result = normalize_title("trading.pdf")
        assert result == "trading"


# ============================================================
# is_trading_relevant — exclude paths
# ============================================================

class TestIsTradingRelevant:
    """Test path-based relevance filtering."""

    def test_excludes_asta_nu_directory(self):
        path = Path(r"J:\E-Books\.....Trading Database\.....Asta_NU\some_book.pdf")
        assert is_trading_relevant(path) is False

    def test_excludes_asta_nu_case_insensitive(self):
        path = Path(r"J:\E-Books\asta_nu\book.pdf")
        assert is_trading_relevant(path) is False

    def test_excludes_arhive_directory(self):
        path = Path(r"J:\E-Books\ARHIVE\duplicate_book.pdf")
        assert is_trading_relevant(path) is False

    def test_excludes_arhive_lowercase(self):
        path = Path(r"J:\E-Books\arhive\old_book.pdf")
        assert is_trading_relevant(path) is False

    def test_excludes_fitness_directory(self):
        path = Path(r"J:\E-Books\Fitness & Health\workout.pdf")
        assert is_trading_relevant(path) is False

    def test_includes_trading_keyword_in_name(self):
        path = Path(r"D:\books\algo_trading_guide.pdf")
        assert is_trading_relevant(path) is True

    def test_includes_forex_book(self):
        path = Path(r"D:\books\forex_mastery.pdf")
        assert is_trading_relevant(path) is True

    def test_includes_path_under_scan_root(self):
        """Files under the configured scan paths are assumed relevant even
        without explicit trading keywords in the filename."""
        path = Path(r"J:\E-Books\misc_book_no_keywords.pdf")
        assert is_trading_relevant(path) is True


# ============================================================
# estimate_tokens
# ============================================================

class TestEstimateTokens:
    """Test rough token estimation."""

    def test_empty_string_returns_zero(self):
        assert estimate_tokens("") == 0

    def test_four_chars_per_token(self):
        # 400 characters → 100 tokens
        text = "a" * 400
        assert estimate_tokens(text) == 100

    def test_reasonable_value_for_paragraph(self):
        text = "The moving average crossover strategy generates a buy signal."
        tokens = estimate_tokens(text)
        # Rough check: should be a small positive integer
        assert tokens > 0
        assert tokens < len(text)


# ============================================================
# detect_chapters
# ============================================================

class TestDetectChapters:
    """Test chapter boundary detection."""

    def test_no_chapters_returns_empty_or_single(self):
        text = "Just a plain block of text with no chapter headings."
        boundaries = detect_chapters(text)
        # Should return an empty list or very short list — no false positives
        assert isinstance(boundaries, list)

    def test_detects_chapter_one_heading(self):
        text = "Introduction\n\nChapter 1\nThis is the first chapter content."
        boundaries = detect_chapters(text)
        assert len(boundaries) >= 1

    def test_detects_multiple_chapters(self):
        text = (
            "Preface content here.\n\n"
            "Chapter 1\nContent of chapter one.\n\n"
            "Chapter 2\nContent of chapter two.\n\n"
            "Chapter 3\nContent of chapter three."
        )
        boundaries = detect_chapters(text)
        assert len(boundaries) >= 2

    def test_returns_sorted_positions(self):
        text = (
            "Chapter 1\nFirst.\n\n"
            "Chapter 2\nSecond.\n\n"
            "Chapter 3\nThird."
        )
        boundaries = detect_chapters(text)
        assert boundaries == sorted(boundaries)


# ============================================================
# chunk_text
# ============================================================

class TestChunkText:
    """Test intelligent text chunking."""

    def test_empty_text_returns_empty_list(self):
        assert chunk_text("") == []

    def test_whitespace_only_returns_empty_list(self):
        assert chunk_text("   \n\n\t  ") == []

    def test_short_text_returns_one_chunk(self):
        text = "This is a short trading text about RSI indicators and MACD signals."
        chunks = chunk_text(text)
        assert len(chunks) == 1

    def test_chunk_has_required_keys(self):
        text = "Trading strategy text. " * 50
        chunks = chunk_text(text)
        assert len(chunks) >= 1
        required_keys = {"text", "start_char", "end_char", "chunk_idx", "chapter_idx", "token_estimate"}
        for chunk in chunks:
            assert required_keys.issubset(chunk.keys()), f"Missing keys: {required_keys - chunk.keys()}"

    def test_chunk_text_is_non_empty_string(self):
        text = "Market analysis paragraph. " * 100
        for chunk in chunk_text(text):
            assert isinstance(chunk["text"], str)
            assert len(chunk["text"]) > 0

    def test_token_estimate_positive(self):
        text = "Order blocks and fair value gaps are key SMC concepts. " * 30
        for chunk in chunk_text(text):
            assert chunk["token_estimate"] > 0

    def test_chunk_indices_sequential(self):
        text = "RSI divergence signal. " * 200
        chunks = chunk_text(text)
        indices = [c["chunk_idx"] for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_large_text_produces_multiple_chunks(self):
        # ~4000 chars ≈ 1000 tokens, default chunk_size=800 → expect ≥ 2 chunks
        text = "Algorithmic trading uses quantitative models. " * 100
        chunks = chunk_text(text)
        assert len(chunks) >= 2

    def test_custom_chunk_size_respected(self):
        text = "Price action setup. " * 500
        chunks_small = chunk_text(text, chunk_size=200, chunk_overlap=50)
        chunks_large = chunk_text(text, chunk_size=800, chunk_overlap=150)
        # Smaller chunk size → more chunks
        assert len(chunks_small) >= len(chunks_large)


# ============================================================
# CLI argument --exclude-dir
# ============================================================

class TestBuildParser:
    """Test the --exclude-dir CLI argument."""

    def test_exclude_dir_defaults_to_empty_list(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.exclude_dir == []

    def test_exclude_dir_single_value(self):
        parser = build_parser()
        args = parser.parse_args(["--exclude-dir", "mydir"])
        assert "mydir" in args.exclude_dir

    def test_exclude_dir_multiple_values(self):
        parser = build_parser()
        args = parser.parse_args(["--exclude-dir", "dir1", "--exclude-dir", "dir2"])
        assert "dir1" in args.exclude_dir
        assert "dir2" in args.exclude_dir

    def test_exclude_dir_combines_with_other_args(self):
        parser = build_parser()
        args = parser.parse_args(["--step", "1", "--exclude-dir", "testdir"])
        assert args.step == 1
        assert "testdir" in args.exclude_dir
