"""
Fast probing experiments using mass-mean probing and efficient logistic regression.

Mass-mean probing (Marks & Tegmark 2023): Classification by difference-of-means direction.
No training needed - just compute mean activation per class and classify by distance.
Validated against logistic regression with liblinear solver.
"""

import sys
import json
import time
import warnings
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from scipy import stats

warnings.filterwarnings("ignore", category=UserWarning)

SEED = 42
np.random.seed(SEED)

BASE_DIR = Path("/data/hypogenicai/workspaces/layer-wise-probing-llm-dcd0-claude")
RESULTS_DIR = BASE_DIR / "results"
ACTIVATIONS_DIR = RESULTS_DIR / "activations"
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def mass_mean_probe(X, y):
    """
    Mass-mean probing (Marks & Tegmark 2023).
    Classify by projecting onto difference-of-means direction.
    Uses leave-one-out-like approach: train on all but one, test on that one.
    """
    classes = np.unique(y)
    if len(classes) != 2:
        return {"accuracy": 0.5, "auc": 0.5}

    # Compute means per class
    mean_0 = X[y == classes[0]].mean(axis=0)
    mean_1 = X[y == classes[1]].mean(axis=0)
    direction = mean_1 - mean_0
    direction = direction / (np.linalg.norm(direction) + 1e-10)

    # Project all points onto this direction
    projections = X @ direction
    threshold = (mean_0 @ direction + mean_1 @ direction) / 2

    # Classify
    predictions = (projections > threshold).astype(int)
    true_labels = (y == classes[1]).astype(int)

    acc = accuracy_score(true_labels, predictions)
    try:
        auc = roc_auc_score(true_labels, projections)
    except Exception:
        auc = acc

    return {"accuracy": float(acc), "auc": float(auc)}


def mass_mean_probe_cv(X, y, n_folds=5):
    """Mass-mean probing with cross-validation for unbiased estimate."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    accs = []
    aucs = []
    classes = np.unique(y)

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        mean_0 = X_train[y_train == classes[0]].mean(axis=0)
        mean_1 = X_train[y_train == classes[1]].mean(axis=0)
        direction = mean_1 - mean_0
        norm = np.linalg.norm(direction)
        if norm < 1e-10:
            accs.append(0.5)
            continue
        direction = direction / norm

        projections = X_test @ direction
        threshold = (mean_0 @ direction + mean_1 @ direction) / 2

        predictions = (projections > threshold).astype(int)
        true_labels = (y_test == classes[1]).astype(int)

        accs.append(accuracy_score(true_labels, predictions))
        try:
            aucs.append(roc_auc_score(true_labels, projections))
        except Exception:
            pass

    return {
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "auc_mean": float(np.mean(aucs)) if aucs else 0.5,
    }


def logistic_probe_fast(X, y, n_folds=5):
    """Fast logistic regression probe using liblinear solver."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    accs = []
    aucs = []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # liblinear is much faster than lbfgs for moderate-sized problems
        clf = LogisticRegression(
            max_iter=200,
            solver="liblinear",
            C=1.0,
            random_state=SEED,
        )
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        accs.append(accuracy_score(y_test, y_pred))

        if len(np.unique(y)) == 2:
            try:
                y_prob = clf.predict_proba(X_test)[:, 1]
                aucs.append(roc_auc_score(y_test, y_prob))
            except Exception:
                pass

    return {
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "auc_mean": float(np.mean(aucs)) if aucs else 0.5,
    }


def control_probe(X, y, n_permutations=2):
    """Random-label control using mass-mean probing."""
    accs = []
    for i in range(n_permutations):
        rng = np.random.RandomState(SEED + i + 1000)
        y_shuffled = rng.permutation(y)
        result = mass_mean_probe_cv(X, y_shuffled, n_folds=5)
        accs.append(result["accuracy_mean"])
    return {
        "control_accuracy_mean": float(np.mean(accs)),
        "control_accuracy_std": float(np.std(accs)),
    }


def run_layer_probing(model_name, dataset_name, label_key="labels"):
    """Run probing across all layers for a model/dataset pair."""
    model_dir = ACTIVATIONS_DIR / model_name
    ds_dir = model_dir / dataset_name

    if not ds_dir.exists():
        return None

    labels = np.load(ds_dir / f"{label_key}.npy")
    unique, counts = np.unique(labels, return_counts=True)
    if len(unique) < 2 or min(counts) < 5:
        return None

    with open(model_dir / "metadata.json") as f:
        metadata = json.load(f)
    n_layers = metadata["n_layers"] + 1

    result = {"layers": {}, "dataset": dataset_name, "model": model_name,
              "n_samples": len(labels), "n_layers": n_layers}

    for layer_idx in range(n_layers):
        layer_file = ds_dir / f"layer_{layer_idx}.npy"
        if not layer_file.exists():
            continue

        X = np.load(layer_file)

        # Mass-mean probe (fast, primary)
        mm_result = mass_mean_probe_cv(X, labels, n_folds=5)

        # Logistic regression probe (slower, validation) — every 4th layer
        if layer_idx % max(1, n_layers // 10) == 0 or layer_idx == n_layers - 1:
            lr_result = logistic_probe_fast(X, labels, n_folds=5)
            mm_result["lr_accuracy_mean"] = lr_result["accuracy_mean"]
            mm_result["lr_accuracy_std"] = lr_result["accuracy_std"]

        # Control task — every 8th layer
        if layer_idx % max(1, n_layers // 6) == 0 or layer_idx == n_layers - 1:
            ctrl = control_probe(X, labels)
            mm_result.update(ctrl)
            mm_result["selectivity"] = mm_result["accuracy_mean"] - ctrl["control_accuracy_mean"]

        result["layers"][str(layer_idx)] = mm_result

    return result


def run_transfer(model_name, train_ds, test_ds):
    """Cross-dataset transfer probing."""
    model_dir = ACTIVATIONS_DIR / model_name
    train_dir = model_dir / train_ds
    test_dir = model_dir / test_ds

    if not train_dir.exists() or not test_dir.exists():
        return None

    train_labels = np.load(train_dir / "labels.npy")
    test_labels = np.load(test_dir / "labels.npy")
    classes = np.unique(train_labels)
    if len(classes) != 2:
        return None

    with open(model_dir / "metadata.json") as f:
        metadata = json.load(f)
    n_layers = metadata["n_layers"] + 1

    result = {"layers": {}, "train_dataset": train_ds, "test_dataset": test_ds}

    for layer_idx in range(n_layers):
        tf = train_dir / f"layer_{layer_idx}.npy"
        ttf = test_dir / f"layer_{layer_idx}.npy"
        if not tf.exists() or not ttf.exists():
            continue

        X_train = np.load(tf)
        X_test = np.load(ttf)

        # Mass-mean transfer
        mean_0 = X_train[train_labels == classes[0]].mean(axis=0)
        mean_1 = X_train[train_labels == classes[1]].mean(axis=0)
        direction = mean_1 - mean_0
        norm = np.linalg.norm(direction)
        if norm < 1e-10:
            result["layers"][str(layer_idx)] = {"transfer_accuracy": 0.5}
            continue
        direction = direction / norm

        projections = X_test @ direction
        threshold = (mean_0 @ direction + mean_1 @ direction) / 2

        test_classes = np.unique(test_labels)
        predictions = (projections > threshold).astype(int)
        true_labels = (test_labels == test_classes[1]).astype(int) if len(test_classes) == 2 else test_labels

        acc = accuracy_score(true_labels, predictions)
        result["layers"][str(layer_idx)] = {"transfer_accuracy": float(acc)}

    return result


def run_all():
    """Run all experiments."""
    model_names = sorted([d.name for d in ACTIVATIONS_DIR.iterdir() if d.is_dir()])
    print(f"Models found: {model_names}", flush=True)

    all_results = {}

    for model_name in model_names:
        t0 = time.time()
        print(f"\n{'='*60}", flush=True)
        print(f"Probing: {model_name}", flush=True)
        print(f"{'='*60}", flush=True)

        model_results = {}

        # Experiment 1: Truth probing on epistemic datasets
        for ds in ["epistemic_all", "epistemic_cities", "epistemic_sp_en_trans",
                    "epistemic_common_claims", "epistemic_companies", "epistemic_larger_than"]:
            print(f"  Truth probing: {ds}...", end="", flush=True)
            r = run_layer_probing(model_name, ds, label_key="labels")
            if r:
                model_results[f"truth_{ds}"] = r
                # Get peak info
                layers = sorted([int(k) for k in r["layers"].keys()])
                accs = [r["layers"][str(l)]["accuracy_mean"] for l in layers]
                peak = layers[np.argmax(accs)]
                print(f" peak={np.max(accs):.3f} @L{peak}", flush=True)
            else:
                print(" skipped", flush=True)

        # Experiment 2: Non-epistemic probing
        print(f"  Non-epistemic probing...", end="", flush=True)
        r = run_layer_probing(model_name, "nonepistemic", label_key="labels")
        if r:
            model_results["agreement_nonepistemic"] = r
            layers = sorted([int(k) for k in r["layers"].keys()])
            accs = [r["layers"][str(l)]["accuracy_mean"] for l in layers]
            peak = layers[np.argmax(accs)]
            print(f" peak={np.max(accs):.3f} @L{peak}", flush=True)
        else:
            print(" skipped", flush=True)

        # Experiment 3: Belief type classification
        print(f"  Type classification...", end="", flush=True)
        r = run_layer_probing(model_name, "type_classification", label_key="type_labels")
        if r:
            model_results["type_classification"] = r
            layers = sorted([int(k) for k in r["layers"].keys()])
            accs = [r["layers"][str(l)]["accuracy_mean"] for l in layers]
            peak = layers[np.argmax(accs)]
            print(f" peak={np.max(accs):.3f} @L{peak}", flush=True)
        else:
            print(" skipped", flush=True)

        # Experiment 4: Transfer
        print(f"  Transfer experiments...", flush=True)
        for src, tgt in [("epistemic_cities", "epistemic_common_claims"),
                          ("epistemic_common_claims", "epistemic_cities")]:
            r = run_transfer(model_name, src, tgt)
            if r:
                model_results[f"transfer_{src.split('_')[-1]}_to_{tgt.split('_')[-1]}"] = r

        all_results[model_name] = model_results
        elapsed = time.time() - t0
        print(f"  Model done in {elapsed:.1f}s", flush=True)

    # Save results
    output = RESULTS_DIR / "metrics" / "probing_results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output}", flush=True)

    return all_results


if __name__ == "__main__":
    total_start = time.time()

    print("=" * 60, flush=True)
    print("FAST PROBING EXPERIMENTS", flush=True)
    print("=" * 60, flush=True)

    results = run_all()

    # Run analysis
    print("\n" + "=" * 60, flush=True)
    print("ANALYSIS AND VISUALIZATION", flush=True)
    print("=" * 60, flush=True)

    sys.path.insert(0, str(BASE_DIR / "src"))
    from analysis import run_full_analysis
    summary, stat_tests = run_full_analysis()

    total = time.time() - total_start
    print(f"\nTotal time: {total:.1f}s ({total/60:.1f} min)", flush=True)
    print("DONE!", flush=True)
