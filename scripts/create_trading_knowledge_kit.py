#!/usr/bin/env python3
"""
Premium Trading Knowledge Kit Creator
Converts PDF trading books + charts into annotated multimodal dataset
"""

import os
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

class TradingKnowledgeKitBuilder:
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.j_drive_path = Path('J:\\E-Books\\.....Trading Database')
        self.kit_output = Path('K:\\_DEV_MVP_2026\\Market_Hawk_3\\knowledge_kits\\premium_trading_kit')
        self.kit_output.mkdir(parents=True, exist_ok=True)
        
    def scan_j_drive_structure(self):
        """Deep scan of J: Trading Database structure"""
        print(f"\n{'='*120}")
        print(f"📚 ANALYZING J: TRADING DATABASE STRUCTURE")
        print(f"{'='*120}\n")
        
        if not self.j_drive_path.exists():
            print(f"⚠️  J: drive not accessible")
            return None
        
        file_categories = defaultdict(list)
        file_types = defaultdict(int)
        authors = defaultdict(int)
        
        try:
            for file in self.j_drive_path.rglob('*'):
                if file.is_file():
                    ext = file.suffix.lower()
                    file_types[ext] += 1
                    
                    # Categorize by type
                    if ext in ['.pdf', '.epub', '.mobi']:
                        file_categories['books'].append(file)
                    elif ext in ['.png', '.jpg', '.jpeg', '.gif']:
                        file_categories['images'].append(file)
                    elif ext in ['.xlsx', '.csv', '.json']:
                        file_categories['data'].append(file)
                    else:
                        file_categories['other'].append(file)
        except Exception as e:
            print(f"Error scanning: {e}")
            return None
        
        # Display analysis
        print(f"📊 J: DRIVE CONTENT ANALYSIS:")
        print(f"   Total files: {sum(len(v) for v in file_categories.values()):,}")
        print(f"   Total file types: {len(file_types)}\n")
        
        print(f"📂 BY CATEGORY:")
        for cat, files in sorted(file_categories.items(), key=lambda x: len(x[1]), reverse=True):
            print(f"   - {cat.upper()}: {len(files):6} files")
        
        print(f"\n📋 TOP 10 FILE TYPES:")
        sorted_types = sorted(file_types.items(), key=lambda x: x[1], reverse=True)[:10]
        for ext, count in sorted_types:
            print(f"   - {ext:10} : {count:6} files")
        
        return file_categories
    
    def create_kit_architecture(self):
        """Design kit architecture"""
        
        architecture = {
            "name": "Premium Trading Knowledge Kit v1.0",
            "purpose": "Vision-Language-Trading (VLT) Model Training",
            "timestamp": self.timestamp,
            "structure": {
                "raw_sources": {
                    "description": "Original trading books & materials from J: drive",
                    "path": str(self.j_drive_path),
                    "size_estimate": "265GB"
                },
                "extracted_images": {
                    "description": "High-quality chart images from books",
                    "path": str(self.kit_output / "images"),
                    "count_estimate": "10K-50K images",
                    "formats": ["PNG", "JPG"]
                },
                "annotations": {
                    "description": "Per-image metadata & trading context",
                    "path": str(self.kit_output / "annotations"),
                    "format": "JSON with structure:",
                    "schema": {
                        "image_id": "unique_identifier",
                        "source_book": "title + author",
                        "page_number": "0-N",
                        "chart_type": "candlestick|ichimoku|sniper|etc",
                        "trading_pattern": "description",
                        "timeframe": "1m|5m|15m|1h|4h|1d",
                        "market": "EUR/USD|BTC/USD|AAPL|etc",
                        "label": "buy_signal|sell_signal|reversal|support|resistance",
                        "confidence": "0.0-1.0",
                        "explanation": "detailed text explanation",
                        "price_action": "description of what happened next",
                        "result": "win|loss|neutral"
                    }
                },
                "multimodal_pairs": {
                    "description": "Image + Text explanations for ML training",
                    "path": str(self.kit_output / "multimodal_pairs"),
                    "format": "JSONL with (image, caption, label)",
                    "count": "10K+ training pairs"
                },
                "knowledge_index": {
                    "description": "Searchable index of trading concepts",
                    "path": str(self.kit_output / "knowledge_index"),
                    "content": [
                        "Sniper Trading strategies",
                        "Ichimoku patterns & interpretation",
                        "Support/Resistance zones",
                        "Candlestick patterns",
                        "Market structure analysis",
                        "Risk management rules"
                    ]
                },
                "training_splits": {
                    "description": "ML training/validation/test splits",
                    "path": str(self.kit_output / "splits"),
                    "train": "70%",
                    "validation": "15%",
                    "test": "15%"
                }
            },
            "ml_pipeline": {
                "stage_1_extraction": "Extract images & text from 1600+ PDFs",
                "stage_2_annotation": "Label each image with trading context",
                "stage_3_pairing": "Create (image, text) multimodal pairs",
                "stage_4_training": "Train Vision-Language model (CLIP-style)",
                "stage_5_finetuning": "Fine-tune for trading pattern recognition",
                "stage_6_integration": "Integrate into Market Hawk ensemble"
            },
            "expected_outcomes": {
                "labeled_images": "10K-50K",
                "training_pairs": "10K-50K",
                "trading_patterns": "50-100",
                "model_accuracy_target": "85%+",
                "inference_speed": "<100ms per image"
            }
        }
        
        return architecture
    
    def generate_kit_blueprint(self):
        """Generate blueprint & action plan"""
        
        architecture = self.create_kit_architecture()
        
        blueprint_path = self.kit_output / 'KIT_BLUEPRINT.md'
        
        with open(blueprint_path, 'w') as f:
            f.write("# Premium Trading Knowledge Kit - Blueprint\n\n")
            f.write(f"**Created:** {self.timestamp}\n\n")
            
            f.write("## 🎯 MISSION\n\n")
            f.write("Convert 1600+ trading books into a **Vision-Language-Trading (VLT) Model**\n")
            f.write("that learns to recognize chart patterns with **85%+ accuracy** from real trading materials.\n\n")
            
            f.write("## 🏗️ ARCHITECTURE\n\n")
            for category, details in architecture['structure'].items():
                f.write(f"### {category.upper()}\n")
                for key, value in details.items():
                    if isinstance(value, list):
                        f.write(f"- {key}:\n")
                        for item in value:
                            f.write(f"  - {item}\n")
                    else:
                        f.write(f"- **{key}:** {value}\n")
                f.write("\n")
            
            f.write("## 📊 ML PIPELINE\n\n")
            for stage, desc in architecture['ml_pipeline'].items():
                f.write(f"- **{stage}:** {desc}\n")
            f.write("\n")
            
            f.write("## 🎯 EXPECTED OUTCOMES\n\n")
            for outcome, value in architecture['expected_outcomes'].items():
                f.write(f"- {outcome}: {value}\n")
            f.write("\n")
            
            f.write("## 📋 IMPLEMENTATION ROADMAP\n\n")
            
            f.write("### PHASE 1: Content Extraction (Week 1-2)\n")
            f.write("- [ ] Inventory all 1600+ materials in J: drive\n")
            f.write("- [ ] Identify books with chart images\n")
            f.write("- [ ] Extract charts using OCR/image extraction\n")
            f.write("- [ ] Sort by trading type (Sniper, Ichimoku, etc)\n\n")
            
            f.write("### PHASE 2: Annotation (Week 3-4)\n")
            f.write("- [ ] Create annotation template\n")
            f.write("- [ ] Label images with trading patterns\n")
            f.write("- [ ] Link to book source + page number\n")
            f.write("- [ ] Document trading rules per pattern\n\n")
            
            f.write("### PHASE 3: Multimodal Pairing (Week 5)\n")
            f.write("- [ ] Create (image, explanation) pairs\n")
            f.write("- [ ] Generate synthetic captions from book text\n")
            f.write("- [ ] Format as JSONL training data\n\n")
            
            f.write("### PHASE 4: Model Training (Week 6-8)\n")
            f.write("- [ ] Train CLIP-style vision-language model\n")
            f.write("- [ ] Fine-tune on trading patterns\n")
            f.write("- [ ] Achieve 85%+ accuracy on test set\n\n")
            
            f.write("### PHASE 5: Integration (Week 9+)\n")
            f.write("- [ ] Integrate into Market Hawk ensemble\n")
            f.write("- [ ] Test on live market data\n")
            f.write("- [ ] Measure trading performance\n\n")
        
        print(f"\n✅ Kit blueprint created: {blueprint_path}\n")
        
        # Save JSON architecture
        arch_path = self.kit_output / 'architecture.json'
        with open(arch_path, 'w') as f:
            json.dump(architecture, f, indent=2)
        
        print(f"✅ Architecture saved: {arch_path}\n")

def main():
    builder = TradingKnowledgeKitBuilder()
    
    print("\n" + "="*120)
    print("🎯 PREMIUM TRADING KNOWLEDGE KIT BUILDER")
    print("="*120)
    
    # Scan J: drive
    categories = builder.scan_j_drive_structure()
    
    # Create kit blueprint
    builder.generate_kit_blueprint()
    
    print("\n" + "="*120)
    print("✅ KIT DESIGN COMPLETE")
    print("="*120)
    print("\nNext: Implement extraction scripts for Phase 1!")

if __name__ == "__main__":
    main()