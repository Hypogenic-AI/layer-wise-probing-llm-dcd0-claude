"""
Linear probing experiments for layer-wise belief encoding analysis.

Trains logistic regression probes on residual stream activations at each layer
to classify:
1. Truth value (true/false) within epistemic statements
2. Agreement (commonly-held/uncommon) within non-epistemic statements
3. Belief type (epistemic vs. non-epistemic)
4. Control tasks (random labels)
"""

import json
import sys
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

SEED = 42
np.random.seed(SEED)

BASE_DIR = Path("/data/hypogenicai/workspaces/layer-wise-probing-llm-dcd0-claude")
RESULTS_DIR = BASE_DIR / "results"
ACTIVATIONS_DIR = RESULTS_DIR / "activations"


def train_probe(X, y, n_folds=5):
    """
    Train a logistic regression probe with cross-validation.
    Uses SGDClassifier with log loss for speed on high-dimensional data.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    accuracies = []
    aucs = []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Standardize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Use SGDClassifier with log loss (equivalent to logistic regression, much faster)
        clf = SGDClassifier(
            loss="log_loss",
            max_iter=500,
            tol=1e-3,
            alpha=1e-4,
            random_state=SEED,
        )
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)

        # AUC (if binary)
        if len(np.unique(y)) == 2:
            try:
                y_prob = clf.decision_function(X_test)
                auc = roc_auc_score(y_test, y_prob)
                aucs.append(auc)
            except Exception:
                pass

    result = {
        "accuracy_mean": float(np.mean(accuracies)),
        "accuracy_std": float(np.std(accuracies)),
        "accuracies": [float(a) for a in accuracies],
    }
    if aucs:
        result["auc_mean"] = float(np.mean(aucs))
        result["auc_std"] = float(np.std(aucs))

    return result


def train_control_probe(X, y, n_folds=5, n_permutations=2):
    """Train probes with randomly permuted labels (control task)."""
    all_accs = []
    for perm_idx in range(n_permutations):
        rng = np.random.RandomState(SEED + perm_idx + 1000)
        y_shuffled = rng.permutation(y)
        result = train_probe(X, y_shuffled, n_folds=n_folds)
        all_accs.append(result["accuracy_mean"])

    return {
        "control_accuracy_mean": float(np.mean(all_accs)),
        "control_accuracy_std": float(np.std(all_accs)),
    }


def run_probing_experiment(model_name, dataset_name, label_key="labels", n_folds=5):
    """
    Run probing experiment across all layers for a given model and dataset.

    Returns dict with per-layer results.
    """
    model_dir = ACTIVATIONS_DIR / model_name
    ds_dir = model_dir / dataset_name

    if not ds_dir.exists():
        print(f"  Skipping {dataset_name} (not found)")
        return None

    # Load labels
    labels = np.load(ds_dir / f"{label_key}.npy")

    # Check we have enough samples per class
    unique, counts = np.unique(labels, return_counts=True)
    if len(unique) < 2:
        print(f"  Skipping {dataset_name}: only one class")
        return None
    if min(counts) < n_folds:
        print(f"  Skipping {dataset_name}: too few samples per class ({dict(zip(unique, counts))})")
        return None

    # Load metadata
    with open(model_dir / "metadata.json") as f:
        metadata = json.load(f)
    n_layers = metadata["n_layers"] + 1  # +1 for embedding layer

    results = {"layers": {}, "dataset": dataset_name, "model": model_name, "n_samples": len(labels)}

    for layer_idx in range(n_layers):
        layer_file = ds_dir / f"layer_{layer_idx}.npy"
        if not layer_file.exists():
            continue

        X = np.load(layer_file)
        probe_result = train_probe(X, labels, n_folds=n_folds)

        # Control task (only for selected layers to save time)
        if layer_idx % max(1, n_layers // 8) == 0 or layer_idx == n_layers - 1:
            control = train_control_probe(X, labels, n_folds=n_folds)
            probe_result.update(control)
            probe_result["selectivity"] = probe_result["accuracy_mean"] - control["control_accuracy_mean"]

        results["layers"][str(layer_idx)] = probe_result

    return results


def run_cross_dataset_transfer(model_name, train_dataset, test_dataset,
                                train_label_key="labels", test_label_key="labels"):
    """
    Train probe on one dataset, test on another (transfer experiment).
    """
    model_dir = ACTIVATIONS_DIR / model_name
    train_dir = model_dir / train_dataset
    test_dir = model_dir / test_dataset

    if not train_dir.exists() or not test_dir.exists():
        return None

    train_labels = np.load(train_dir / f"{train_label_key}.npy")
    test_labels = np.load(test_dir / f"{test_label_key}.npy")

    with open(model_dir / "metadata.json") as f:
        metadata = json.load(f)
    n_layers = metadata["n_layers"] + 1

    results = {"layers": {}, "train_dataset": train_dataset, "test_dataset": test_dataset}

    for layer_idx in range(n_layers):
        train_file = train_dir / f"layer_{layer_idx}.npy"
        test_file = test_dir / f"layer_{layer_idx}.npy"
        if not train_file.exists() or not test_file.exists():
            continue

        X_train = np.load(train_file)
        X_test = np.load(test_file)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        clf = SGDClassifier(loss="log_loss", max_iter=500, tol=1e-3, alpha=1e-4, random_state=SEED)
        clf.fit(X_train, train_labels)

        y_pred = clf.predict(X_test)
        acc = accuracy_score(test_labels, y_pred)

        results["layers"][str(layer_idx)] = {"transfer_accuracy": float(acc)}

    return results


def run_all_experiments(model_names=None):
    """Run all probing experiments for all models."""
    if model_names is None:
        model_names = [d.name for d in ACTIVATIONS_DIR.iterdir() if d.is_dir()]

    all_results = {}

    for model_name in sorted(model_names):
        print(f"\n{'='*60}", flush=True)
        print(f"Probing experiments for: {model_name}", flush=True)
        print(f"{'='*60}", flush=True)

        model_results = {}

        # Experiment 1: Truth probing on epistemic datasets
        for ds_name in ["epistemic_all", "epistemic_cities", "epistemic_sp_en_trans",
                        "epistemic_common_claims", "epistemic_companies", "epistemic_larger_than"]:
            print(f"\n  Exp 1 - Truth probing: {ds_name}", flush=True)
            result = run_probing_experiment(model_name, ds_name, label_key="labels")
            if result:
                model_results[f"truth_{ds_name}"] = result

        # Experiment 2: Agreement probing on non-epistemic dataset
        print(f"\n  Exp 2 - Agreement probing: nonepistemic", flush=True)
        result = run_probing_experiment(model_name, "nonepistemic", label_key="labels")
        if result:
            model_results["agreement_nonepistemic"] = result

        # Experiment 3: Belief type classification (epistemic vs non-epistemic)
        print(f"\n  Exp 3 - Type classification: type_classification", flush=True)
        result = run_probing_experiment(model_name, "type_classification", label_key="type_labels")
        if result:
            model_results["type_classification"] = result

        # Experiment 4: Cross-dataset transfer
        print(f"\n  Exp 4 - Cross-dataset transfer")
        # Train on epistemic, test on non-epistemic (and vice versa)
        # For this we need matching label semantics — use type classification labels
        transfer_result = run_cross_dataset_transfer(
            model_name, "epistemic_cities", "epistemic_common_claims"
        )
        if transfer_result:
            model_results["transfer_cities_to_claims"] = transfer_result

        transfer_result = run_cross_dataset_transfer(
            model_name, "epistemic_common_claims", "epistemic_cities"
        )
        if transfer_result:
            model_results["transfer_claims_to_cities"] = transfer_result

        all_results[model_name] = model_results

    # Save all results
    output_file = RESULTS_DIR / "metrics" / "probing_results.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nAll results saved to {output_file}")
    return all_results


if __name__ == "__main__":
    results = run_all_experiments()
