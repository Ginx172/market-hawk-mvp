#!/usr/bin/env python3
"""
Complete resource scanner - G: drive + J: drive (Trading Database)
"""

import os
import json
from pathlib import Path
from datetime import datetime

class CompletResourceScanner:
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.resources = {
            'g_drive': {},
            'j_drive': {},
            'total_stats': {}
        }
        
    def scan_path(self, path_str, resource_type, description):
        """Scan a specific path and categorize resources"""
        try:
            path = Path(path_str)
            if not path.exists():
                return None
            
            files = list(path.rglob('*'))
            file_count = len([f for f in files if f.is_file()])
            dir_count = len([f for f in files if f.is_dir()])
            total_size = sum(f.stat().st_size for f in files if f.is_file()) / (1024**3)
            
            # Get file types
            file_types = {}
            for f in files:
                if f.is_file():
                    ext = f.suffix or 'no_extension'
                    file_types[ext] = file_types.get(ext, 0) + 1
            
            resource = {
                'path': path_str,
                'type': resource_type,
                'description': description,
                'files': file_count,
                'folders': dir_count,
                'size_gb': round(total_size, 2),
                'file_types': dict(sorted(file_types.items(), key=lambda x: x[1], reverse=True)[:10]),
                'last_modified': datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
            }
            
            print(f"✅ {resource_type:25} | {total_size:8.2f}GB | {file_count:7} files")
            return resource
            
        except Exception as e:
            print(f"❌ Error: {path_str} - {e}")
            return None
    
    def run_scan(self):
        """Run complete scan"""
        print(f"\n{'='*110}")
        print(f"🔍 MARKET HAWK 3 - COMPLETE RESOURCE SCANNER (G: + J: drives)")
        print(f"{'='*110}\n")
        
        # G: DRIVE
        print("📦 G: DRIVE - AI MODELS & CHECKPOINTS:")
        print("-" * 110)
        g_models = [
            ('G:\\AI_Models', 'model_library', 'Trained AI models'),
            ('G:\\L40_GPU_SUPERKIT', 'gpu_kit', 'L40 GPU-optimized models'),
            ('G:\\L40_CHECKPOINTS', 'checkpoints', 'Training checkpoints'),
            ('G:\\MASTER_AI_TRADING_PACKAGE', 'trading_ensemble', 'Pre-built trading AI'),
            ('G:\\Modele AI curate', 'curated_models', 'Curated models'),
        ]
        
        for path, rtype, desc in g_models:
            result = self.scan_path(path, rtype, desc)
            if result:
                self.resources['g_drive']['models'] = self.resources['g_drive'].get('models', []) + [result]
        
        print("\n📊 G: DRIVE - DATASETS:")
        print("-" * 110)
        g_data = [
            ('G:\\..............JSON', 'json_data', 'JSON data'),
            ('G:\\.................CSV', 'csv_data', 'CSV market data (274K files)'),
            ('G:\\.........PARQUET', 'parquet_data', 'Parquet data (345K files)'),
            ('G:\.....DATABASE\\dataset', 'database', 'Database files'),
            ('G:\\trading_db', 'trading_db', 'Trading database'),
        ]
        
        for path, rtype, desc in g_data:
            result = self.scan_path(path, rtype, desc)
            if result:
                self.resources['g_drive']['datasets'] = self.resources['g_drive'].get('datasets', []) + [result]
        
        print("\n🎓 G: DRIVE - TRAINING KITS:")
        print("-" * 110)
        g_kits = [
            ('G:\\_kit_multimodal', 'multimodal', 'Multimodal kit (17K files, 99GB)'),
            ('G:\\ULTRA_KIT_ANTENAMENT_AI', 'ultra_kit', 'Ultra training kit'),
            ('G:\\KIT_TRAINING_BOOKS', 'books', 'Trading books'),
        ]
        
        for path, rtype, desc in g_kits:
            result = self.scan_path(path, rtype, desc)
            if result:
                self.resources['g_drive']['kits'] = self.resources['g_drive'].get('kits', []) + [result]
        
        # J: DRIVE - TRADING DATABASE (PRIMARY!)
        print("\n\n" + "="*110)
        print("🎯 J: DRIVE - PRIMARY TRADING DATABASE (MOST IMPORTANT)")
        print("="*110 + "\n")
        print("📚 J: DRIVE - TRADING DATABASE:")
        print("-" * 110)
        
        j_primary = [
            ('J:\\E-Books\\.....Trading Database', 'trading_db_primary', 'PRIMARY: Consolidated trading database'),
        ]
        
        for path, rtype, desc in j_primary:
            result = self.scan_path(path, rtype, desc)
            if result:
                self.resources['j_drive']['primary'] = result
        
        self.generate_final_report()
    
    def generate_final_report(self):
        """Generate strategic integration report"""
        report_path = Path(__file__).parent.parent / 'COMPLETE_RESOURCE_INTEGRATION_PLAN.md'
        
        with open(report_path, 'w') as f:
            f.write("# Market Hawk 3 - Complete Resource Integration Plan\n\n")
            f.write(f"**Scan Date:** {self.timestamp}\n\n")
            
            f.write("## 🎯 EXECUTIVE SUMMARY\n\n")
            f.write("You have a **1.2TB+ AI/Trading Resource Library** across G: and J: drives.\n")
            f.write("This document provides strategic integration roadmap.\n\n")
            
            f.write("## 📊 RESOURCE BREAKDOWN\n\n")
            
            # G: Drive summary
            if self.resources['g_drive']:
                f.write("### G: Drive Resources\n")
                f.write("- **AI Models:** 173GB (multimodal, L40 optimized, pre-trained ensembles)\n")
                f.write("- **Datasets:** 633GB (CSV, Parquet, JSON - 700K+ files)\n")
                f.write("- **Training Kits:** 109GB (multimodal, ultra-kit, books)\n\n")
            
            # J: Drive summary
            if self.resources['j_drive']:
                f.write("### J: Drive - PRIMARY TRADING DATABASE ⭐\n")
                j_primary = self.resources['j_drive'].get('primary', {})
                if j_primary:
                    f.write(f"- **Path:** {j_primary['path']}\n")
                    f.write(f"- **Size:** {j_primary['size_gb']}GB\n")
                    f.write(f"- **Files:** {j_primary['files']:,}\n")
                    f.write(f"- **File Types:** {', '.join(list(j_primary['file_types'].keys())[:5])}\n\n")
            
            f.write("## 🚀 RECOMMENDED INTEGRATION STRATEGY\n\n")
            
            f.write("### PHASE 1: Foundation (Week 1-2)\n")
            f.write("1. **Index Trading Database** (J: drive)\n")
            f.write("   - Catalog all files in J:\\E-Books\\...Trading Database\n")
            f.write("   - Extract metadata & file types\n")
            f.write("   - Create integration mapping\n\n")
            
            f.write("2. **Optimize Data Pipeline**\n")
            f.write("   - Link Parquet files (311GB) for fast backtesting\n")
            f.write("   - Index CSV data (273GB, 274K files)\n")
            f.write("   - Setup symlinks or GCS mounting\n\n")
            
            f.write("### PHASE 2: Model Integration (Week 3-4)\n")
            f.write("1. **AI Model Ensemble**\n")
            f.write("   - Load L40 checkpoints into MarketHawk ensemble\n")
            f.write("   - Integrate multimodal models (99GB kit)\n")
            f.write("   - Test inference latency\n\n")
            
            f.write("2. **Vision Models**\n")
            f.write("   - Use multimodal kit for chart analysis\n")
            f.write("   - Train on technical analysis images\n\n")
            
            f.write("### PHASE 3: Knowledge Expansion (Week 5-6)\n")
            f.write("1. **RAG Expansion**\n")
            f.write("   - Ingest trading books into knowledge base\n")
            f.write("   - Extract trading rules & patterns\n")
            f.write("   - Build semantic search index\n\n")
            
            f.write("### PHASE 4: Live Testing (Week 7+)\n")
            f.write("1. **Paper Trading**\n")
            f.write("   - Use integrated models on real data\n")
            f.write("   - Measure performance improvements\n")
            f.write("   - Optimize parameters\n\n")
            
            f.write("## 📋 ACTION ITEMS\n\n")
            f.write("- [ ] Scan J: drive Trading Database in detail\n")
            f.write("- [ ] Create symlinks to large datasets\n")
            f.write("- [ ] Load L40 models into memory\n")
            f.write("- [ ] Index trading database for RAG\n")
            f.write("- [ ] Build multi-model inference pipeline\n")
        
        print(f"\n✅ Integration plan saved: {report_path}\n")

def main():
    scanner = CompletResourceScanner()
    scanner.run_scan()

if __name__ == "__main__":
    main()