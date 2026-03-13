#!/usr/bin/env python3
"""
Market Hawk MVP - CI/CD Diagnostic Script
Checks all PHASE 1 files and requirements
"""

import os
import sys
from pathlib import Path
from datetime import datetime

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'

class Diagnostics:
    def __init__(self):
        self.repo_root = Path(__file__).parent.parent
        self.results = []
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
    def check_file(self, relative_path, description):
        """Check if file exists"""
        full_path = self.repo_root / relative_path
        exists = full_path.exists()
        
        status = "✅" if exists else "❌"
        size = f"{full_path.stat().st_size} bytes" if exists else "N/A"
        
        print(f"{status} {relative_path:<45} - {description}")
        print(f"   Location: {full_path}")
        print(f"   Size: {size}\n")
        
        self.results.append({
            'file': relative_path,
            'exists': exists,
            'description': description
        })
        
        return exists
    
    def check_folder(self, relative_path, description):
        """Check if folder exists"""
        full_path = self.repo_root / relative_path
        exists = full_path.exists()
        
        status = "✅" if exists else "❌"
        file_count = len(list(full_path.iterdir())) if exists else 0
        
        print(f"{status} {relative_path:<45} - {description}")
        print(f"   Location: {full_path}")
        print(f"   Files: {file_count}\n")
        
        self.results.append({
            'file': relative_path,
            'exists': exists,
            'description': description,
            'type': 'folder'
        })
        
        return exists
    
    def run_diagnostics(self):
        """Run all checks"""
        print(f"\n{Colors.BLUE}{'='*80}{Colors.RESET}")
        print(f"{Colors.BLUE}🔍 MARKET HAWK MVP - CI/CD PHASE 1 DIAGNOSTICS{Colors.RESET}")
        print(f"{Colors.BLUE}Timestamp: {self.timestamp}{Colors.RESET}")
        print(f"{Colors.BLUE}{'='*80}{Colors.RESET}\n")
        
        # Check folders
        print(f"{Colors.YELLOW}📁 FOLDER STRUCTURE:{Colors.RESET}\n")
        folders = [
            ('.github/workflows', 'GitHub Actions workflows'),
            ('scripts', 'Scripts directory'),
            ('.githooks', 'Git hooks'),
            ('tests', 'Tests directory'),
            ('tests/unit', 'Unit tests'),
            ('tests/integration', 'Integration tests'),
        ]
        
        folder_results = []
        for folder, desc in folders:
            folder_results.append(self.check_folder(folder, desc))
        
        # Check files
        print(f"\n{Colors.YELLOW}📄 CONFIGURATION FILES:{Colors.RESET}\n")
        config_files = [
            ('.github/workflows/ci.yml', 'GitHub Actions CI/CD workflow'),
            ('pytest.ini', 'Pytest configuration'),
            ('Dockerfile', 'Docker configuration'),
            ('docker-compose.yml', 'Docker Compose configuration'),
            ('requirements-dev.txt', 'Development requirements'),
            ('.env.local', 'Local environment variables'),
            ('.githooks/pre-commit', 'Pre-commit git hook'),
        ]
        
        config_results = []
        for file, desc in config_files:
            config_results.append(self.check_file(file, desc))
        
        # Check existing files
        print(f"\n{Colors.YELLOW}🔍 EXISTING PROJECT FILES:{Colors.RESET}\n")
        existing_files = [
            ('requirements.txt', 'Project requirements'),
            ('.gitignore', 'Git ignore rules'),
            ('run_backtest.py', 'Backtest runner'),
        ]
        
        existing_results = []
        for file, desc in existing_files:
            existing_results.append(self.check_file(file, desc))
        
        # Summary
        print(f"\n{Colors.BLUE}{'='*80}{Colors.RESET}")
        print(f"{Colors.BLUE}📊 SUMMARY{Colors.RESET}")
        print(f"{Colors.BLUE}{'='*80}{Colors.RESET}\n")
        
        total_checks = len(self.results)
        passed_checks = sum(1 for r in self.results if r.get('exists', False))
        failed_checks = total_checks - passed_checks
        
        print(f"Total checks: {total_checks}")
        print(f"{Colors.GREEN}Passed: {passed_checks}{Colors.RESET}")
        print(f"{Colors.RED}Failed: {failed_checks}{Colors.RESET}\n")
        
        # Detailed results
        print(f"{Colors.YELLOW}Detailed Results:{Colors.RESET}\n")
        
        missing_files = [r for r in self.results if not r.get('exists', False)]
        found_files = [r for r in self.results if r.get('exists', False)]
        
        if found_files:
            print(f"{Colors.GREEN}✅ Found ({len(found_files)} files):{Colors.RESET}")
            for f in found_files:
                print(f"   • {f['file']} - {f['description']}")
            print()
        
        if missing_files:
            print(f"{Colors.RED}❌ Missing ({len(missing_files)} files):{Colors.RESET}")
            for f in missing_files:
                print(f"   • {f['file']} - {f['description']}")
            print()
        
        # Recommendations
        print(f"\n{Colors.BLUE}{'='*80}{Colors.RESET}")
        print(f"{Colors.BLUE}💡 RECOMMENDATIONS:{Colors.RESET}")
        print(f"{Colors.BLUE}{'='*80}{Colors.RESET}\n")
        
        if passed_checks == total_checks:
            print(f"{Colors.GREEN}✅ All files are present!{Colors.RESET}")
            print("\nNext steps:")
            print("1. Run: git status")
            print("2. Run: git add .")
            print("3. Run: git commit -m 'chore: PHASE 1 CI/CD Foundation Setup'")
            print("4. Run: git push origin main")
            print("5. Check GitHub Actions: https://github.com/Ginx172/market-hawk-mvp/actions")
        else:
            print(f"{Colors.RED}❌ Some files are missing!{Colors.RESET}")
            print("\nMissing files:")
            for f in missing_files:
                print(f"   • Create: {f['file']}")
        
        print(f"\n{Colors.BLUE}{'='*80}{Colors.RESET}\n")
        
        return failed_checks == 0
    
    def generate_report(self):
        """Generate detailed report"""
        report_file = self.repo_root / 'DIAGNOSTICS_REPORT.md'
        
        with open(report_file, 'w') as f:
            f.write("# Market Hawk MVP - CI/CD Diagnostics Report\n\n")
            f.write(f"**Generated:** {self.timestamp}\n\n")
            
            f.write("## Summary\n\n")
            total = len(self.results)
            passed = sum(1 for r in self.results if r.get('exists', False))
            f.write(f"- Total Checks: {total}\n")
            f.write(f"- Passed: {passed}\n")
            f.write(f"- Failed: {total - passed}\n\n")
            
            f.write("## Files Found\n\n")
            for r in self.results:
                if r.get('exists', False):
                    f.write(f"- ✅ `{r['file']}` - {r['description']}\n")
            
            f.write("\n## Files Missing\n\n")
            for r in self.results:
                if not r.get('exists', False):
                    f.write(f"- ❌ `{r['file']}` - {r['description']}\n")
            
            f.write("\n## Next Steps\n\n")
            f.write("1. Verify all files are created\n")
            f.write("2. Run: `git status`\n")
            f.write("3. Run: `git add .`\n")
            f.write("4. Run: `git commit -m 'chore: PHASE 1 CI/CD Foundation Setup'`\n")
            f.write("5. Run: `git push origin main`\n")
        
        print(f"📄 Report saved to: {report_file}")
        return report_file

def main():
    """Main entry point"""
    diag = Diagnostics()
    success = diag.run_diagnostics()
    diag.generate_report()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()