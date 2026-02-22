"""
Run probing experiments and analysis on pre-extracted activations.
Activations were already extracted in Phase 2.
"""
import sys
import time
sys.path.insert(0, "/data/hypogenicai/workspaces/layer-wise-probing-llm-dcd0-claude/src")

print("=" * 60, flush=True)
print("PROBING EXPERIMENTS", flush=True)
print("=" * 60, flush=True)

start = time.time()
from probing import run_all_experiments
results = run_all_experiments()
elapsed = time.time() - start
print(f"\nProbing complete in {elapsed:.1f}s", flush=True)

print("\n" + "=" * 60, flush=True)
print("ANALYSIS AND VISUALIZATION", flush=True)
print("=" * 60, flush=True)

start = time.time()
from analysis import run_full_analysis
summary, stat_tests = run_full_analysis()
elapsed = time.time() - start
print(f"\nAnalysis complete in {elapsed:.1f}s", flush=True)

print("\nDONE!", flush=True)
