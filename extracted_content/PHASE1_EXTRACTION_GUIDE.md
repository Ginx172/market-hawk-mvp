# PHASE 1: Content Extraction - Step-by-Step Guide

**Generated:** 2026-03-12 21:47:45

## 🎯 OBJECTIVE

Extract raw content from J: drive in organized, ML-ready format.

## 📋 EXTRACTION TASKS

### 1.1: Video Frame Extraction

**Input:** 3,623 MP4 files
**Output:** PNG frames (1 per second)
**Tool:** ffmpeg
**Time Estimate:** 20-30 hours (parallelizable)
**Priority:** HIGH

**Command:** `ffmpeg -i input.mp4 -vf fps=1 output_%04d.png`

### 1.2: Book OCR Extraction

**Input:** 1,807 PDF/EPUB files
**Output:** TXT files with OCR text
**Tool:** pytesseract / pdf2image
**Time Estimate:** 15-20 hours
**Priority:** HIGH

### 1.3: Transcript Parsing

**Input:** 2,811 SRT/VTT files
**Output:** Parsed JSON with timestamps
**Tool:** Python subtitle parser
**Time Estimate:** 1-2 hours
**Priority:** MEDIUM

### 1.4: Chart Image Cataloging

**Input:** 400 JPG/PNG chart images
**Output:** Cataloged with metadata
**Tool:** OpenCV / PIL
**Time Estimate:** 30 minutes
**Priority:** MEDIUM

### 1.5: Code Examples Indexing

**Input:** 513 PY/Jupyter files
**Output:** Indexed by trading patterns
**Tool:** AST parsing
**Time Estimate:** 2-3 hours
**Priority:** LOW

## 🛠️ PREREQUISITE INSTALLATION

### For Video Processing:
```bash
# Install FFmpeg
# Windows: choco install ffmpeg
# Or download from https://ffmpeg.org/download.html
pip install opencv-python
```

### For PDF/OCR:
```bash
pip install pytesseract pdf2image pillow
# Also install Tesseract-OCR from https://github.com/UB-Mannheim/tesseract/wiki
```

### For Subtitle Processing:
```bash
pip install pysrt webvtt-py
```

## 📊 STORAGE REQUIREMENTS

- **Video Frames:** ~2TB (1 frame/sec from 3,623 videos)
- **OCR Text:** ~500MB
- **Transcripts:** ~100MB
- **Charts:** ~50MB
- **Metadata:** ~500MB
- **TOTAL:** ~2.5-3TB

**Recommendation:** Use external SSD (USB 3.0+)

## 🚀 EXTRACTION SCRIPTS (TO BE CREATED)

1. `extract_video_frames.py` - Parallel video processing
2. `extract_book_text.py` - OCR all PDFs
3. `parse_transcripts.py` - Parse SRT/VTT files
4. `catalog_charts.py` - Index chart images
5. `index_code_examples.py` - Index Python files

## ⏱️ TIMELINE

- Total Estimated Time: 40-60 hours
- Parallelizable: YES (can run 4-8 tasks simultaneously)
- GPU Recommended: YES (for video processing)
- Cost: Only electricity/storage (no cloud fees)

## ✅ NEXT STEPS

1. Verify storage space available
2. Install required tools (FFmpeg, Tesseract, etc)
3. Run individual extraction scripts
4. Monitor progress & disk usage
5. Move to Phase 2 (Annotation) when extraction complete
