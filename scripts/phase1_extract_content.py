#!/usr/bin/env python3
"""
PHASE 1: Content Extraction
Extract video frames, OCR books, parse transcripts, catalog charts
"""

import os
import json
import subprocess
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

class Phase1Extractor:
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.j_drive = Path('J:\\E-Books\\.....Trading Database')
        self.extract_dir = Path('K:\\_DEV_MVP_2026\\Market_Hawk_3\\extracted_content')
        self.extract_dir.mkdir(parents=True, exist_ok=True)
        
        # Sub-directories
        self.video_frames_dir = self.extract_dir / 'video_frames'
        self.book_text_dir = self.extract_dir / 'book_text'
        self.transcripts_dir = self.extract_dir / 'transcripts'
        self.charts_dir = self.extract_dir / 'charts'
        
        for d in [self.video_frames_dir, self.book_text_dir, self.transcripts_dir, self.charts_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
    def create_extraction_plan(self):
        """Create detailed extraction plan"""
        
        plan = {
            "timestamp": self.timestamp,
            "phase": "1_content_extraction",
            "tasks": [
                {
                    "task_id": "1.1",
                    "name": "Video Frame Extraction",
                    "input": "3,623 MP4 files",
                    "output": "PNG frames (1 per second)",
                    "tool": "ffmpeg",
                    "command": "ffmpeg -i input.mp4 -vf fps=1 output_%04d.png",
                    "estimated_time": "20-30 hours (parallelizable)",
                    "estimated_output": "~260M frames (~2TB)",
                    "priority": "HIGH"
                },
                {
                    "task_id": "1.2",
                    "name": "Book OCR Extraction",
                    "input": "1,807 PDF/EPUB files",
                    "output": "TXT files with OCR text",
                    "tool": "pytesseract / pdf2image",
                    "estimated_time": "15-20 hours",
                    "estimated_output": "~500MB text data",
                    "priority": "HIGH"
                },
                {
                    "task_id": "1.3",
                    "name": "Transcript Parsing",
                    "input": "2,811 SRT/VTT files",
                    "output": "Parsed JSON with timestamps",
                    "tool": "Python subtitle parser",
                    "estimated_time": "1-2 hours",
                    "estimated_output": "~100MB JSON",
                    "priority": "MEDIUM"
                },
                {
                    "task_id": "1.4",
                    "name": "Chart Image Cataloging",
                    "input": "400 JPG/PNG chart images",
                    "output": "Cataloged with metadata",
                    "tool": "OpenCV / PIL",
                    "estimated_time": "30 minutes",
                    "estimated_output": "Metadata JSON",
                    "priority": "MEDIUM"
                },
                {
                    "task_id": "1.5",
                    "name": "Code Examples Indexing",
                    "input": "513 PY/Jupyter files",
                    "output": "Indexed by trading patterns",
                    "tool": "AST parsing",
                    "estimated_time": "2-3 hours",
                    "estimated_output": "Code index JSON",
                    "priority": "LOW"
                }
            ],
            "total_estimated_time": "40-60 hours",
            "recommended_setup": [
                "Use GPU for video processing (if available)",
                "Run extraction tasks in parallel on multiple cores",
                "Use external storage (SSD preferred) for frames",
                "Monitor disk space (need ~2-3TB temporary)"
            ],
            "next_phase": "Phase 2: Annotation & Labeling"
        }
        
        return plan
    
    def generate_extraction_guide(self, plan):
        """Generate step-by-step extraction guide"""
        
        guide_path = self.extract_dir / 'PHASE1_EXTRACTION_GUIDE.md'
        
        with open(guide_path, 'w') as f:
            f.write("# PHASE 1: Content Extraction - Step-by-Step Guide\n\n")
            f.write(f"**Generated:** {self.timestamp}\n\n")
            
            f.write("## 🎯 OBJECTIVE\n\n")
            f.write("Extract raw content from J: drive in organized, ML-ready format.\n\n")
            
            f.write("## 📋 EXTRACTION TASKS\n\n")
            
            for task in plan['tasks']:
                f.write(f"### {task['task_id']}: {task['name']}\n\n")
                f.write(f"**Input:** {task['input']}\n")
                f.write(f"**Output:** {task['output']}\n")
                f.write(f"**Tool:** {task['tool']}\n")
                f.write(f"**Time Estimate:** {task['estimated_time']}\n")
                f.write(f"**Priority:** {task['priority']}\n\n")
                
                if 'command' in task:
                    f.write(f"**Command:** `{task['command']}`\n\n")
            
            f.write("## 🛠️ PREREQUISITE INSTALLATION\n\n")
            
            f.write("### For Video Processing:\n")
            f.write("```bash\n")
            f.write("# Install FFmpeg\n")
            f.write("# Windows: choco install ffmpeg\n")
            f.write("# Or download from https://ffmpeg.org/download.html\n")
            f.write("pip install opencv-python\n")
            f.write("```\n\n")
            
            f.write("### For PDF/OCR:\n")
            f.write("```bash\n")
            f.write("pip install pytesseract pdf2image pillow\n")
            f.write("# Also install Tesseract-OCR from https://github.com/UB-Mannheim/tesseract/wiki\n")
            f.write("```\n\n")
            
            f.write("### For Subtitle Processing:\n")
            f.write("```bash\n")
            f.write("pip install pysrt webvtt-py\n")
            f.write("```\n\n")
            
            f.write("## 📊 STORAGE REQUIREMENTS\n\n")
            f.write("- **Video Frames:** ~2TB (1 frame/sec from 3,623 videos)\n")
            f.write("- **OCR Text:** ~500MB\n")
            f.write("- **Transcripts:** ~100MB\n")
            f.write("- **Charts:** ~50MB\n")
            f.write("- **Metadata:** ~500MB\n")
            f.write("- **TOTAL:** ~2.5-3TB\n\n")
            f.write("**Recommendation:** Use external SSD (USB 3.0+)\n\n")
            
            f.write("## 🚀 EXTRACTION SCRIPTS (TO BE CREATED)\n\n")
            f.write("1. `extract_video_frames.py` - Parallel video processing\n")
            f.write("2. `extract_book_text.py` - OCR all PDFs\n")
            f.write("3. `parse_transcripts.py` - Parse SRT/VTT files\n")
            f.write("4. `catalog_charts.py` - Index chart images\n")
            f.write("5. `index_code_examples.py` - Index Python files\n\n")
            
            f.write("## ⏱️ TIMELINE\n\n")
            f.write("- Total Estimated Time: 40-60 hours\n")
            f.write("- Parallelizable: YES (can run 4-8 tasks simultaneously)\n")
            f.write("- GPU Recommended: YES (for video processing)\n")
            f.write("- Cost: Only electricity/storage (no cloud fees)\n\n")
            
            f.write("## ✅ NEXT STEPS\n\n")
            f.write("1. Verify storage space available\n")
            f.write("2. Install required tools (FFmpeg, Tesseract, etc)\n")
            f.write("3. Run individual extraction scripts\n")
            f.write("4. Monitor progress & disk usage\n")
            f.write("5. Move to Phase 2 (Annotation) when extraction complete\n")
        
        print(f"✅ Extraction guide: {guide_path}\n")
        
        # Save JSON plan
        plan_path = self.extract_dir / 'extraction_plan.json'
        with open(plan_path, 'w') as f:
            json.dump(plan, f, indent=2)
        
        print(f"✅ Extraction plan: {plan_path}\n")

def main():
    extractor = Phase1Extractor()
    
    print(f"\n{'='*130}")
    print(f"🚀 PHASE 1: CONTENT EXTRACTION - PLANNING")
    print(f"{'='*130}\n")
    
    plan = extractor.create_extraction_plan()
    extractor.generate_extraction_guide(plan)
    
    print("="*130)
    print("✅ PHASE 1 EXTRACTION PLAN READY!")
    print("="*130)
    print("\nNext: Build individual extraction scripts...")
    print(f"Output directory: {extractor.extract_dir}\n")

if __name__ == "__main__":
    main()