#!/usr/bin/env python3
"""
Fix folder structure - move misplaced files
"""

import os
import shutil
from pathlib import Path

def main():
    repo_root = Path(__file__).parent.parent
    
    print("🧹 Fixing folder structure...\n")
    
    # Files that should be at root but are in .github/workflows/
    misplaced_files = {
        '.github/workflows/.env.local': '.env.local',
        '.github/workflows/Dockerfile': 'Dockerfile',
        '.github/workflows/docker-compose.yml': 'docker-compose.yml',
        '.github/workflows/pytest.ini': 'pytest.ini',
        '.github/workflows/requirements-dev.txt': 'requirements-dev.txt',
        '.github/workflows/.githookspre-commit': '.githooks/pre-commit',
    }
    
    for wrong_path, correct_path in misplaced_files.items():
        wrong_full = repo_root / wrong_path
        correct_full = repo_root / correct_path
        
        if wrong_full.exists():
            print(f"Moving: {wrong_path}")
            print(f"     To: {correct_path}\n")
            
            # Create parent directory if needed
            correct_full.parent.mkdir(parents=True, exist_ok=True)
            
            # Move file
            shutil.move(str(wrong_full), str(correct_full))
            print(f"✅ Moved successfully\n")
        else:
            print(f"ℹ️  File not found (already fixed?): {wrong_path}\n")
    
    print("✅ Folder structure fixed!")

if __name__ == "__main__":
    main()