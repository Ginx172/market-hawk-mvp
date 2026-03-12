#!/usr/bin/env python3
"""Pre-push validation script"""

import subprocess
import sys

def run_check(cmd, description):
    print(f"\n🔍 {description}...")
    result = subprocess.run(cmd, shell=True)
    return result.returncode == 0

checks = [
    ("black --check . --exclude 'venv|.git|models'", "Black formatting"),
    ("pylint agents/ brain/ backtesting/ --fail-under=7.0", "Pylint"),
    ("bandit -r . -ll --exclude venv", "Bandit security"),
    ("pytest tests/ -v --cov=.", "Pytest + Coverage"),
]

results = [run_check(cmd, desc) for cmd, desc in checks]

print(f"\n{'='*60}")
print(f"Passed: {sum(results)}/{len(results)} checks")
if all(results):
    print("✅ Ready to push!")
    sys.exit(0)
else:
    print("❌ Fix issues and retry")
    sys.exit(1)