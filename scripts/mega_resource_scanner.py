#!/usr/bin/env python3
"""
MEGA Resource Scanner - All drives (G:, J:, D:, H:)
Complete inventory of all AI/Trading resources
"""

import os
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

class MegaResourceScanner:
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.resources = defaultdict(list)
        self.total_stats = {
            'total_gb': 0,
            'total_files': 0,
            'drives': {}
        }
        
    def scan_path(self, path_str, resource_type, description, drive_letter):
        """Scan a specific path"""
        try:
            path = Path(path_str)
            if not path.exists():
                print(f"⚠️  Not found: {path_str}")
                return None
            
            files = list(path.rglob('*'))
            file_count = len([f for f in files if f.is_file()])
            dir_count = len([f for f in files if f.is_dir()])
            total_size = sum(f.stat().st_size for f in files if f.is_file()) / (1024**3)
            
            # Get top file types
            file_types = {}
            for f in files:
                if f.is_file():
                    ext = f.suffix.lower() or 'no_ext'
                    file_types[ext] = file_types.get(ext, 0) + 1
            
            resource = {
                'path': path_str,
                'type': resource_type,
                'description': description,
                'files': file_count,
                'folders': dir_count,
                'size_gb': round(total_size, 2),
                'file_types': dict(sorted(file_types.items(), key=lambda x: x[1], reverse=True)[:8]),
                'drive': drive_letter,
            }
            
            self.total_stats['total_gb'] += total_size
            self.total_stats['total_files'] += file_count
            if drive_letter not in self.total_stats['drives']:
                self.total_stats['drives'][drive_letter] = {'gb': 0, 'files': 0}
            self.total_stats['drives'][drive_letter]['gb'] += total_size
            self.total_stats['drives'][drive_letter]['files'] += file_count
            
            print(f"✅ {resource_type:30} | {drive_letter}: | {total_size:8.2f}GB | {file_count:8} files")
            return resource
            
        except Exception as e:
            print(f"❌ Error: {path_str} - {str(e)[:50]}")
            return None
    
    def run_scan(self):
        """Run mega scan"""
        print(f"\n{'='*130}")
        print(f"🔍 MARKET HAWK 3 - MEGA RESOURCE SCANNER (ALL DRIVES: G:, J:, D:, H:)")
        print(f"{'='*130}\n")
        
        # G: DRIVE
        print("📦 G: DRIVE - AI MODELS:")
        print("-" * 130)
        g_models = [
            ('G:\\AI_Models', 'g_models', 'AI Models Library'),
            ('G:\\L40_GPU_SUPERKIT', 'g_l40_kit', 'L40 GPU Superkit'),
            ('G:\\L40_CHECKPOINTS', 'g_checkpoints', 'L40 Checkpoints'),
            ('G:\\MASTER_AI_TRADING_PACKAGE', 'g_trading_ensemble', 'Master Trading Package'),
            ('G:\\Modele AI curate', 'g_curated', 'Curated AI Models'),
        ]
        for path, rtype, desc in g_models:
            result = self.scan_path(path, rtype, desc, 'G:')
            if result:
                self.resources['ai_models'].append(result)
        
        print("\n📊 G: DRIVE - DATASETS:")
        print("-" * 130)
        g_data = [
            ('G:\\..............JSON', 'g_json', 'JSON Data'),
            ('G:\\.................CSV', 'g_csv', 'CSV Market Data (274K files)'),
            ('G:\\.........PARQUET', 'g_parquet', 'Parquet Data (345K files)'),
            ('G:\.....DATABASE\\dataset', 'g_db', 'Database Files'),
            ('G:\\trading_db', 'g_trading_db', 'Trading DB'),
        ]
        for path, rtype, desc in g_data:
            result = self.scan_path(path, rtype, desc, 'G:')
            if result:
                self.resources['datasets'].append(result)
        
        print("\n🎓 G: DRIVE - TRAINING KITS:")
        print("-" * 130)
        g_kits = [
            ('G:\\_kit_multimodal', 'g_multimodal', 'Multimodal Kit (99GB)'),
            ('G:\\ULTRA_KIT_ANTENAMENT_AI', 'g_ultra_kit', 'Ultra Training Kit'),
            ('G:\\KIT_TRAINING_BOOKS', 'g_books', 'Training Books'),
        ]
        for path, rtype, desc in g_kits:
            result = self.scan_path(path, rtype, desc, 'G:')
            if result:
                self.resources['training_kits'].append(result)
        
        # J: DRIVE - PRIMARY TRADING DB
        print("\n\n" + "="*130)
        print("🎯 J: DRIVE - PRIMARY TRADING DATABASE")
        print("="*130 + "\n")
        j_primary = [
            ('J:\\E-Books\\.....Trading Database', 'j_trading_db', 'PRIMARY Trading Database'),
        ]
        for path, rtype, desc in j_primary:
            result = self.scan_path(path, rtype, desc, 'J:')
            if result:
                self.resources['trading_database'].append(result)
        
        # D: DRIVE
        print("\n\n" + "="*130)
        print("💾 D: DRIVE")
        print("="*130 + "\n")
        d_resources = [
            ('D:\\', 'd_root', 'D: Drive Root'),
        ]
        for path, rtype, desc in d_resources:
            result = self.scan_path(path, rtype, desc, 'D:')
            if result:
                self.resources['d_drive'].append(result)
        
        # H: DRIVE - MARKET HAWK & AWS
        print("\n\n" + "="*130)
        print("🔗 H: DRIVE - LIVE TRADING & AWS RESOURCES")
        print("="*130 + "\n")
        h_resources = [
            ('H:\\..........Market_Hawk_AI_Agent_Live_Trading', 'h_live_trading', 'Live Trading Agent'),
            ('H:\\Din AWS S3', 'h_aws_s3', 'AWS S3 Resources'),
        ]
        for path, rtype, desc in h_resources:
            result = self.scan_path(path, rtype, desc, 'H:')
            if result:
                self.resources['h_drive'].append(result)
        
        self.generate_mega_report()
    
    def generate_mega_report(self):
        """Generate comprehensive integration report"""
        report_path = Path(__file__).parent.parent / 'MEGA_RESOURCE_INVENTORY.md'
        
        with open(report_path, 'w') as f:
            f.write("# Market Hawk 3 - MEGA RESOURCE INVENTORY\n\n")
            f.write(f"**Scan Date:** {self.timestamp}\n\n")
            
            # SUMMARY
            f.write("## 📊 TOTAL INVENTORY SUMMARY\n\n")
            f.write(f"- **Total Size:** {self.total_stats['total_gb']:.2f} GB\n")
            f.write(f"- **Total Files:** {self.total_stats['total_files']:,}\n")
            f.write(f"- **Drives Scanned:** {len(self.total_stats['drives'])}\n\n")
            
            f.write("### By Drive:\n")
            for drive, stats in sorted(self.total_stats['drives'].items()):
                f.write(f"- **{drive}** - {stats['gb']:.2f}GB ({stats['files']:,} files)\n")
            f.write("\n")
            
            # AI MODELS
            f.write("## 🤖 AI MODELS & ENSEMBLES\n\n")
            for resource in self.resources['ai_models']:
                f.write(f"### {resource['description']}\n")
                f.write(f"- **Drive:** {resource['drive']}\n")
                f.write(f"- **Path:** `{resource['path']}`\n")
                f.write(f"- **Size:** {resource['size_gb']}GB | **Files:** {resource['files']:,}\n")
                f.write(f"- **Top Types:** {', '.join(list(resource['file_types'].keys())[:3])}\n\n")
            
            # DATASETS
            f.write("## 📈 DATASETS\n\n")
            for resource in self.resources['datasets']:
                f.write(f"### {resource['description']}\n")
                f.write(f"- **Drive:** {resource['drive']}\n")
                f.write(f"- **Path:** `{resource['path']}`\n")
                f.write(f"- **Size:** {resource['size_gb']}GB | **Files:** {resource['files']:,}\n")
                f.write(f"- **Top Types:** {', '.join(list(resource['file_types'].keys())[:3])}\n\n")
            
            # TRADING DATABASE (PRIMARY)
            f.write("## 🎯 PRIMARY TRADING DATABASE\n\n")
            for resource in self.resources['trading_database']:
                f.write(f"### {resource['description']}\n")
                f.write(f"- **Drive:** {resource['drive']}\n")
                f.write(f"- **Path:** `{resource['path']}`\n")
                f.write(f"- **Size:** {resource['size_gb']}GB | **Files:** {resource['files']:,}\n")
                f.write(f"- **File Types:** {resource['file_types']}\n\n")
            
            # TRAINING KITS
            f.write("## 📚 TRAINING KITS\n\n")
            for resource in self.resources['training_kits']:
                f.write(f"### {resource['description']}\n")
                f.write(f"- **Drive:** {resource['drive']}\n")
                f.write(f"- **Path:** `{resource['path']}`\n")
                f.write(f"- **Size:** {resource['size_gb']}GB | **Files:** {resource['files']:,}\n\n")
            
            # LIVE TRADING & AWS
            f.write("## 🔗 LIVE TRADING & AWS\n\n")
            for resource in self.resources['h_drive']:
                f.write(f"### {resource['description']}\n")
                f.write(f"- **Drive:** {resource['drive']}\n")
                f.write(f"- **Path:** `{resource['path']}`\n")
                f.write(f"- **Size:** {resource['size_gb']}GB | **Files:** {resource['files']:,}\n\n")
            
            # INTEGRATION PRIORITY
            f.write("## 🚀 INTEGRATION PRIORITY ROADMAP\n\n")
            f.write("### 🔴 CRITICAL (Must Use)\n")
            f.write("1. **J: Trading Database** - Primary source of truth\n")
            f.write("2. **G: CSV/Parquet Data** - 633GB backtesting fuel\n")
            f.write("3. **G: AI Models** - 173GB pre-trained ensemble\n\n")
            
            f.write("### 🟡 HIGH PRIORITY (Should Use)\n")
            f.write("1. **G: Multimodal Kit** - Vision model training (99GB)\n")
            f.write("2. **H: Live Trading Agent** - Reference implementation\n")
            f.write("3. **H: AWS S3** - Cloud data resources\n\n")
            
            f.write("### 🟢 MEDIUM PRIORITY (Nice to Have)\n")
            f.write("1. **G: Training Books** - RAG knowledge base\n")
            f.write("2. **G: L40 Kits** - GPU-optimized models\n\n")
            
            f.write("## 📋 NEXT STEPS\n\n")
            f.write("1. [ ] Analyze J: Trading Database structure\n")
            f.write("2. [ ] Create symlinks to large datasets\n")
            f.write("3. [ ] Load AI models into Market Hawk ensemble\n")
            f.write("4. [ ] Index trading data for backtesting\n")
            f.write("5. [ ] Integrate H: live trading reference\n")
        
        print(f"\n✅ Mega inventory report saved: {report_path}\n")
        print(f"\n📊 FINAL STATS:")
        print(f"   Total: {self.total_stats['total_gb']:.2f}GB across {self.total_stats['total_files']:,} files")
        print(f"   Drives: {len(self.total_stats['drives'])} scanned\n")

def main():
    scanner = MegaResourceScanner()
    scanner.run_scan()

if __name__ == "__main__":
    main()