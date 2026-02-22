"""
Main experiment runner for layer-wise belief probing analysis.

Orchestrates:
1. Dataset preparation
2. Activation extraction from GPT-2 models
3. Linear probing experiments
4. Analysis and visualization
"""

import sys
import json
import time
import random
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

BASE_DIR = Path("/data/hypogenicai/workspaces/layer-wise-probing-llm-dcd0-claude")
RESULTS_DIR = BASE_DIR / "results"

sys.path.insert(0, str(BASE_DIR / "src"))


def log_environment():
    """Log environment information for reproducibility."""
    import transformers
    import sklearn

    env_info = {
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "timestamp": datetime.now().isoformat(),
        "seed": SEED,
    }

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            env_info[f"gpu_{i}"] = torch.cuda.get_device_name(i)

    env_file = RESULTS_DIR / "metrics" / "environment.json"
    env_file.parent.mkdir(parents=True, exist_ok=True)
    with open(env_file, "w") as f:
        json.dump(env_info, f, indent=2)

    print("Environment:")
    for k, v in env_info.items():
        print(f"  {k}: {v}")
    return env_info


def run_phase1_data():
    """Phase 1: Data preparation."""
    print("\n" + "=" * 60)
    print("PHASE 1: DATA PREPARATION")
    print("=" * 60)
    start = time.time()

    from data_preparation import prepare_all_datasets
    datasets = prepare_all_datasets()

    elapsed = time.time() - start
    print(f"\nPhase 1 complete in {elapsed:.1f}s")
    return datasets


def run_phase2_extract(datasets, models=None):
    """Phase 2: Activation extraction."""
    print("\n" + "=" * 60)
    print("PHASE 2: ACTIVATION EXTRACTION")
    print("=" * 60)
    start = time.time()

    from extract_activations import extract_and_save, MODEL_CONFIGS
    import pandas as pd

    if models is None:
        models = list(MODEL_CONFIGS.keys())

    # Reload datasets from CSV (to ensure consistency)
    ds_dir = RESULTS_DIR / "datasets"
    ds_dict = {}
    for csv_file in ds_dir.glob("*.csv"):
        ds_dict[csv_file.stem] = pd.read_csv(csv_file)

    for model_key in models:
        print(f"\n--- Extracting activations for {model_key} ---")
        model_start = time.time()
        extract_and_save(model_key, ds_dict, device="cuda:0", batch_size=64)
        model_elapsed = time.time() - model_start
        print(f"  {model_key} complete in {model_elapsed:.1f}s")

    elapsed = time.time() - start
    print(f"\nPhase 2 complete in {elapsed:.1f}s")


def run_phase3_probing():
    """Phase 3: Probing experiments."""
    print("\n" + "=" * 60)
    print("PHASE 3: PROBING EXPERIMENTS")
    print("=" * 60)
    start = time.time()

    from probing import run_all_experiments
    results = run_all_experiments()

    elapsed = time.time() - start
    print(f"\nPhase 3 complete in {elapsed:.1f}s")
    return results


def run_phase4_analysis():
    """Phase 4: Analysis and visualization."""
    print("\n" + "=" * 60)
    print("PHASE 4: ANALYSIS AND VISUALIZATION")
    print("=" * 60)
    start = time.time()

    from analysis import run_full_analysis
    summary, stat_tests = run_full_analysis()

    elapsed = time.time() - start
    print(f"\nPhase 4 complete in {elapsed:.1f}s")
    return summary, stat_tests


def main():
    """Run the full experiment pipeline."""
    total_start = time.time()

    print("=" * 60)
    print("LAYER-WISE PROBING ANALYSIS OF BELIEF ENCODING IN LLMs")
    print("=" * 60)
    print(f"Start time: {datetime.now().isoformat()}")

    # Log environment
    log_environment()

    # Phase 1: Data preparation
    datasets = run_phase1_data()

    # Phase 2: Activation extraction
    # Run GPT-2 models (all 4 sizes)
    run_phase2_extract(datasets, models=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"])

    # Phase 3: Probing experiments
    results = run_phase3_probing()

    # Phase 4: Analysis and visualization
    summary, stat_tests = run_phase4_analysis()

    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"FULL PIPELINE COMPLETE")
    print(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"{'='*60}")

    return summary, stat_tests


if __name__ == "__main__":
    main()
