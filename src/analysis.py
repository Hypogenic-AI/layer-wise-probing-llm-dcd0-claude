"""
Analysis and visualization for layer-wise belief probing experiments.

Generates:
1. Layer-wise accuracy curves for each experiment
2. Comparison plots across models and belief types
3. PCA visualizations of belief representations
4. Statistical analysis of peak layers and selectivity
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
from scipy import stats

SEED = 42
np.random.seed(SEED)

BASE_DIR = Path("/data/hypogenicai/workspaces/layer-wise-probing-llm-dcd0-claude")
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
ACTIVATIONS_DIR = RESULTS_DIR / "activations"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Color scheme
COLORS = {
    "epistemic": "#2196F3",
    "nonepistemic": "#FF5722",
    "type_classification": "#4CAF50",
    "control": "#9E9E9E",
    "transfer": "#9C27B0",
}

MODEL_DISPLAY = {
    "gpt2_small": "GPT-2 Small (117M)",
    "gpt2_medium": "GPT-2 Medium (345M)",
    "gpt2_large": "GPT-2 Large (774M)",
    "gpt2_xl": "GPT-2 XL (1.5B)",
}


def load_results():
    """Load probing results from JSON."""
    results_file = RESULTS_DIR / "metrics" / "probing_results.json"
    with open(results_file) as f:
        return json.load(f)


def get_layer_accuracies(result_dict):
    """Extract layer indices and accuracies from a probing result."""
    layers = sorted([int(k) for k in result_dict["layers"].keys()])
    accs = [result_dict["layers"][str(l)]["accuracy_mean"] for l in layers]
    stds = [result_dict["layers"][str(l)]["accuracy_std"] for l in layers]
    return layers, accs, stds


def get_control_accuracies(result_dict):
    """Extract control task accuracies."""
    layers = []
    accs = []
    for l in sorted([int(k) for k in result_dict["layers"].keys()]):
        entry = result_dict["layers"][str(l)]
        if "control_accuracy_mean" in entry:
            layers.append(l)
            accs.append(entry["control_accuracy_mean"])
    return layers, accs


def plot_layerwise_accuracy_single_model(results, model_name, save=True):
    """Plot layer-wise accuracy curves for a single model, all experiments."""
    model_results = results[model_name]
    display_name = MODEL_DISPLAY.get(model_name, model_name)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Layer-wise Probing Accuracy: {display_name}", fontsize=14, fontweight="bold")

    # Panel 1: Truth probing on epistemic datasets
    ax = axes[0]
    ax.set_title("Truth Probing (Epistemic Statements)")
    for key in model_results:
        if key.startswith("truth_epistemic_") and key != "truth_epistemic_all":
            layers, accs, stds = get_layer_accuracies(model_results[key])
            source = key.replace("truth_epistemic_", "")
            ax.plot(layers, accs, label=source, alpha=0.6, linewidth=1)

    if "truth_epistemic_all" in model_results:
        layers, accs, stds = get_layer_accuracies(model_results["truth_epistemic_all"])
        ax.plot(layers, accs, label="All epistemic", color=COLORS["epistemic"],
                linewidth=2.5, zorder=10)
        ax.fill_between(layers, [a - s for a, s in zip(accs, stds)],
                        [a + s for a, s in zip(accs, stds)], alpha=0.15,
                        color=COLORS["epistemic"])
        # Control
        ctrl_layers, ctrl_accs = get_control_accuracies(model_results["truth_epistemic_all"])
        if ctrl_layers:
            ax.scatter(ctrl_layers, ctrl_accs, color=COLORS["control"],
                       marker="x", s=60, label="Random control", zorder=5)

    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Accuracy")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_ylim(0.35, 1.05)
    ax.grid(True, alpha=0.3)

    # Panel 2: Non-epistemic probing
    ax = axes[1]
    ax.set_title("Agreement Probing (Non-Epistemic Statements)")
    if "agreement_nonepistemic" in model_results:
        layers, accs, stds = get_layer_accuracies(model_results["agreement_nonepistemic"])
        ax.plot(layers, accs, label="Non-epistemic", color=COLORS["nonepistemic"],
                linewidth=2.5)
        ax.fill_between(layers, [a - s for a, s in zip(accs, stds)],
                        [a + s for a, s in zip(accs, stds)], alpha=0.15,
                        color=COLORS["nonepistemic"])
        ctrl_layers, ctrl_accs = get_control_accuracies(model_results["agreement_nonepistemic"])
        if ctrl_layers:
            ax.scatter(ctrl_layers, ctrl_accs, color=COLORS["control"],
                       marker="x", s=60, label="Random control", zorder=5)

    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Accuracy")
    ax.legend(fontsize=8)
    ax.set_ylim(0.35, 1.05)
    ax.grid(True, alpha=0.3)

    # Panel 3: Belief type classification
    ax = axes[2]
    ax.set_title("Belief Type Classification\n(Epistemic vs. Non-Epistemic)")
    if "type_classification" in model_results:
        layers, accs, stds = get_layer_accuracies(model_results["type_classification"])
        ax.plot(layers, accs, label="Type classification", color=COLORS["type_classification"],
                linewidth=2.5)
        ax.fill_between(layers, [a - s for a, s in zip(accs, stds)],
                        [a + s for a, s in zip(accs, stds)], alpha=0.15,
                        color=COLORS["type_classification"])
        ctrl_layers, ctrl_accs = get_control_accuracies(model_results["type_classification"])
        if ctrl_layers:
            ax.scatter(ctrl_layers, ctrl_accs, color=COLORS["control"],
                       marker="x", s=60, label="Random control", zorder=5)

    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Accuracy")
    ax.legend(fontsize=8)
    ax.set_ylim(0.35, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        fig.savefig(PLOTS_DIR / f"layerwise_{model_name}.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: layerwise_{model_name}.png")
    plt.close()


def plot_cross_model_comparison(results, save=True):
    """Compare epistemic vs non-epistemic probing across all models."""
    model_names = sorted(results.keys())

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Layer-wise Probing: Epistemic vs. Non-Epistemic Beliefs Across GPT-2 Family",
                 fontsize=14, fontweight="bold")

    for idx, model_name in enumerate(model_names[:4]):
        ax = axes[idx // 2][idx % 2]
        display_name = MODEL_DISPLAY.get(model_name, model_name)
        ax.set_title(display_name)
        model_results = results[model_name]

        # Epistemic truth probing
        if "truth_epistemic_all" in model_results:
            layers, accs, stds = get_layer_accuracies(model_results["truth_epistemic_all"])
            # Normalize layer index to [0, 1] for cross-model comparison
            norm_layers = [l / max(layers) for l in layers]
            ax.plot(norm_layers, accs, label="Epistemic (truth)", color=COLORS["epistemic"],
                    linewidth=2)
            ax.fill_between(norm_layers, [a - s for a, s in zip(accs, stds)],
                            [a + s for a, s in zip(accs, stds)], alpha=0.1, color=COLORS["epistemic"])

        # Non-epistemic probing
        if "agreement_nonepistemic" in model_results:
            layers, accs, stds = get_layer_accuracies(model_results["agreement_nonepistemic"])
            norm_layers = [l / max(layers) for l in layers]
            ax.plot(norm_layers, accs, label="Non-epistemic (agreement)",
                    color=COLORS["nonepistemic"], linewidth=2)
            ax.fill_between(norm_layers, [a - s for a, s in zip(accs, stds)],
                            [a + s for a, s in zip(accs, stds)], alpha=0.1,
                            color=COLORS["nonepistemic"])

        # Type classification
        if "type_classification" in model_results:
            layers, accs, stds = get_layer_accuracies(model_results["type_classification"])
            norm_layers = [l / max(layers) for l in layers]
            ax.plot(norm_layers, accs, label="Type classification",
                    color=COLORS["type_classification"], linewidth=2, linestyle="--")

        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Normalized Layer (0=embedding, 1=final)")
        ax.set_ylabel("Accuracy")
        ax.legend(fontsize=7)
        ax.set_ylim(0.35, 1.05)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        fig.savefig(PLOTS_DIR / "cross_model_comparison.png", dpi=150, bbox_inches="tight")
        print("  Saved: cross_model_comparison.png")
    plt.close()


def plot_peak_layer_comparison(results, save=True):
    """Bar chart comparing peak layer positions across models and belief types."""
    model_names = sorted(results.keys())

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.set_title("Peak Probing Layer (Normalized) by Belief Type", fontsize=13, fontweight="bold")

    x = np.arange(len(model_names))
    width = 0.25

    for offset, (exp_key, color, label) in enumerate([
        ("truth_epistemic_all", COLORS["epistemic"], "Epistemic Truth"),
        ("agreement_nonepistemic", COLORS["nonepistemic"], "Non-Epistemic Agreement"),
        ("type_classification", COLORS["type_classification"], "Type Classification"),
    ]):
        peaks = []
        for model_name in model_names:
            if exp_key in results[model_name]:
                layers, accs, _ = get_layer_accuracies(results[model_name][exp_key])
                peak_idx = layers[np.argmax(accs)]
                peak_norm = peak_idx / max(layers)
                peaks.append(peak_norm)
            else:
                peaks.append(0)

        ax.bar(x + (offset - 1) * width, peaks, width, label=label, color=color, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_DISPLAY.get(m, m) for m in model_names], rotation=15, ha="right")
    ax.set_ylabel("Peak Layer (Normalized)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    if save:
        fig.savefig(PLOTS_DIR / "peak_layer_comparison.png", dpi=150, bbox_inches="tight")
        print("  Saved: peak_layer_comparison.png")
    plt.close()


def plot_selectivity(results, save=True):
    """Plot selectivity (true accuracy - control accuracy) across models."""
    model_names = sorted(results.keys())

    fig, axes = plt.subplots(1, len(model_names), figsize=(5 * len(model_names), 5), sharey=True)
    fig.suptitle("Selectivity: True Accuracy - Random Control Accuracy", fontsize=14, fontweight="bold")

    if len(model_names) == 1:
        axes = [axes]

    for ax, model_name in zip(axes, model_names):
        display_name = MODEL_DISPLAY.get(model_name, model_name)
        ax.set_title(display_name)

        for exp_key, color, label in [
            ("truth_epistemic_all", COLORS["epistemic"], "Epistemic"),
            ("agreement_nonepistemic", COLORS["nonepistemic"], "Non-Epistemic"),
            ("type_classification", COLORS["type_classification"], "Type Classif."),
        ]:
            if exp_key not in results[model_name]:
                continue

            result_dict = results[model_name][exp_key]
            layers_with_selectivity = []
            selectivities = []

            for l in sorted([int(k) for k in result_dict["layers"].keys()]):
                entry = result_dict["layers"][str(l)]
                if "selectivity" in entry:
                    layers_with_selectivity.append(l)
                    selectivities.append(entry["selectivity"])

            if layers_with_selectivity:
                ax.bar(np.array(layers_with_selectivity) + {"epistemic": -0.3,
                       "nonepistemic": 0, "type_classification": 0.3}.get(
                       exp_key.split("_")[0], 0),
                       selectivities, width=0.3, label=label, color=color, alpha=0.7)

        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Layer")
        if ax == axes[0]:
            ax.set_ylabel("Selectivity")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    if save:
        fig.savefig(PLOTS_DIR / "selectivity.png", dpi=150, bbox_inches="tight")
        print("  Saved: selectivity.png")
    plt.close()


def plot_transfer_results(results, save=True):
    """Plot cross-dataset transfer accuracy."""
    model_names = sorted(results.keys())

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    ax.set_title("Cross-Dataset Transfer Accuracy", fontsize=13, fontweight="bold")

    for model_name in model_names:
        model_results = results[model_name]
        display_name = MODEL_DISPLAY.get(model_name, model_name)

        for transfer_key, style in [
            ("transfer_cities_to_claims", "-"),
            ("transfer_claims_to_cities", "--"),
        ]:
            if transfer_key in model_results:
                layers = sorted([int(k) for k in model_results[transfer_key]["layers"].keys()])
                accs = [model_results[transfer_key]["layers"][str(l)]["transfer_accuracy"]
                        for l in layers]
                n_layers_max = max(layers)
                norm_layers = [l / n_layers_max for l in layers]
                label_suffix = transfer_key.replace("transfer_", "").replace("_to_", " → ")
                ax.plot(norm_layers, accs, style, label=f"{display_name}: {label_suffix}", linewidth=1.5)

    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")
    ax.set_xlabel("Normalized Layer")
    ax.set_ylabel("Transfer Accuracy")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_ylim(0.3, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        fig.savefig(PLOTS_DIR / "transfer_accuracy.png", dpi=150, bbox_inches="tight")
        print("  Saved: transfer_accuracy.png")
    plt.close()


def plot_pca_visualization(model_name, dataset_name, selected_layers=None, save=True):
    """PCA visualization of representations at selected layers."""
    model_dir = ACTIVATIONS_DIR / model_name
    ds_dir = model_dir / dataset_name

    if not ds_dir.exists():
        return

    labels = np.load(ds_dir / "labels.npy")

    # Auto-select layers if not specified
    with open(model_dir / "metadata.json") as f:
        metadata = json.load(f)
    n_layers = metadata["n_layers"]

    if selected_layers is None:
        # Select embedding, early, middle, late, final
        selected_layers = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers]

    fig, axes = plt.subplots(1, len(selected_layers), figsize=(4.5 * len(selected_layers), 4))
    display_name = MODEL_DISPLAY.get(model_name, model_name)
    fig.suptitle(f"PCA of Representations: {display_name} - {dataset_name}", fontsize=13, fontweight="bold")

    for ax, layer_idx in zip(axes, selected_layers):
        layer_file = ds_dir / f"layer_{layer_idx}.npy"
        if not layer_file.exists():
            continue

        X = np.load(layer_file)
        pca = PCA(n_components=2, random_state=SEED)
        X_2d = pca.fit_transform(X)

        for label_val, color, name in [(1, "#2196F3", "True/Held"), (0, "#FF5722", "False/Not")]:
            mask = labels == label_val
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=color, alpha=0.5, s=15, label=name)

        layer_label = "Emb" if layer_idx == 0 else f"L{layer_idx}"
        ax.set_title(f"Layer {layer_label}\nVar: {pca.explained_variance_ratio_.sum():.1%}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    if save:
        fig.savefig(PLOTS_DIR / f"pca_{model_name}_{dataset_name}.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: pca_{model_name}_{dataset_name}.png")
    plt.close()


def compute_statistical_summary(results):
    """Compute statistical summary across all experiments."""
    summary = {}

    for model_name, model_results in results.items():
        model_summary = {}

        for exp_key, result in model_results.items():
            if "layers" not in result:
                continue

            # Skip transfer experiments (different format)
            first_layer = next(iter(result["layers"].values()), {})
            if "transfer_accuracy" in first_layer and "accuracy_mean" not in first_layer:
                continue

            try:
                layers, accs, stds = get_layer_accuracies(result)
            except KeyError:
                continue

            peak_idx = int(np.argmax(accs))
            peak_layer = layers[peak_idx]
            peak_acc = accs[peak_idx]
            n_layers = max(layers)

            model_summary[exp_key] = {
                "peak_layer": peak_layer,
                "peak_layer_normalized": peak_layer / n_layers if n_layers > 0 else 0,
                "peak_accuracy": peak_acc,
                "peak_accuracy_std": stds[peak_idx],
                "mean_accuracy": float(np.mean(accs)),
                "final_layer_accuracy": accs[-1],
                "n_layers": n_layers,
            }

            # Add selectivity if available
            if "selectivity" in result["layers"].get(str(peak_layer), {}):
                model_summary[exp_key]["peak_selectivity"] = result["layers"][str(peak_layer)]["selectivity"]

        summary[model_name] = model_summary

    return summary


def run_statistical_tests(results):
    """Run statistical tests comparing epistemic vs non-epistemic encoding."""
    test_results = {}

    for model_name, model_results in results.items():
        if "truth_epistemic_all" not in model_results or "agreement_nonepistemic" not in model_results:
            continue

        epist_layers, epist_accs, _ = get_layer_accuracies(model_results["truth_epistemic_all"])
        nonep_layers, nonep_accs, _ = get_layer_accuracies(model_results["agreement_nonepistemic"])

        # Normalize layers for comparison
        epist_norm = [l / max(epist_layers) for l in epist_layers]
        nonep_norm = [l / max(nonep_layers) for l in nonep_layers]

        # Interpolate to common layer grid for comparison
        common_grid = np.linspace(0, 1, 20)
        epist_interp = np.interp(common_grid, epist_norm, epist_accs)
        nonep_interp = np.interp(common_grid, nonep_norm, nonep_accs)

        # Paired t-test on accuracy profiles
        t_stat, p_value = stats.ttest_rel(epist_interp, nonep_interp)

        # Peak layer comparison
        epist_peak = epist_norm[np.argmax(epist_accs)]
        nonep_peak = nonep_norm[np.argmax(nonep_accs)]

        # Effect size (Cohen's d)
        diff = epist_interp - nonep_interp
        cohens_d = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0

        test_results[model_name] = {
            "paired_ttest_t": float(t_stat),
            "paired_ttest_p": float(p_value),
            "cohens_d": float(cohens_d),
            "epistemic_peak_norm": float(epist_peak),
            "nonepistemic_peak_norm": float(nonep_peak),
            "peak_difference": float(epist_peak - nonep_peak),
            "mean_epistemic_acc": float(np.mean(epist_accs)),
            "mean_nonepistemic_acc": float(np.mean(nonep_accs)),
        }

    return test_results


def plot_summary_heatmap(results, save=True):
    """Heatmap of peak accuracies across models and experiment types."""
    summary = compute_statistical_summary(results)

    # Build matrix
    models = sorted(summary.keys())
    experiments = ["truth_epistemic_all", "agreement_nonepistemic", "type_classification"]
    exp_labels = ["Epistemic Truth", "Non-Epistemic Agreement", "Type Classification"]

    matrix = np.zeros((len(models), len(experiments)))
    for i, model in enumerate(models):
        for j, exp in enumerate(experiments):
            if exp in summary[model]:
                matrix[i, j] = summary[model][exp]["peak_accuracy"]

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    sns.heatmap(matrix, annot=True, fmt=".3f", cmap="YlOrRd",
                xticklabels=exp_labels,
                yticklabels=[MODEL_DISPLAY.get(m, m) for m in models],
                ax=ax, vmin=0.4, vmax=1.0)
    ax.set_title("Peak Probing Accuracy by Model and Belief Type", fontsize=13, fontweight="bold")

    plt.tight_layout()
    if save:
        fig.savefig(PLOTS_DIR / "summary_heatmap.png", dpi=150, bbox_inches="tight")
        print("  Saved: summary_heatmap.png")
    plt.close()


def run_full_analysis():
    """Run all analysis and generate all plots."""
    print("=" * 60)
    print("ANALYSIS AND VISUALIZATION")
    print("=" * 60)

    results = load_results()
    model_names = sorted(results.keys())

    # 1. Per-model layer-wise plots
    print("\n1. Generating per-model layer-wise accuracy plots...")
    for model_name in model_names:
        plot_layerwise_accuracy_single_model(results, model_name)

    # 2. Cross-model comparison
    print("\n2. Generating cross-model comparison...")
    if len(model_names) > 1:
        plot_cross_model_comparison(results)

    # 3. Peak layer comparison
    print("\n3. Generating peak layer comparison...")
    plot_peak_layer_comparison(results)

    # 4. Selectivity plot
    print("\n4. Generating selectivity plots...")
    plot_selectivity(results)

    # 5. Transfer accuracy
    print("\n5. Generating transfer accuracy plots...")
    plot_transfer_results(results)

    # 6. PCA visualizations
    print("\n6. Generating PCA visualizations...")
    for model_name in model_names:
        for ds in ["epistemic_all", "nonepistemic", "type_classification"]:
            plot_pca_visualization(model_name, ds)

    # 7. Summary heatmap
    print("\n7. Generating summary heatmap...")
    plot_summary_heatmap(results)

    # 8. Statistical analysis
    print("\n8. Running statistical tests...")
    summary = compute_statistical_summary(results)
    stat_tests = run_statistical_tests(results)

    # Save analysis results
    analysis_output = {
        "summary": summary,
        "statistical_tests": stat_tests,
    }
    output_file = RESULTS_DIR / "metrics" / "analysis_results.json"
    with open(output_file, "w") as f:
        json.dump(analysis_output, f, indent=2)
    print(f"\n  Analysis saved to: {output_file}")

    # Print key findings
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    for model_name in model_names:
        display_name = MODEL_DISPLAY.get(model_name, model_name)
        print(f"\n{display_name}:")
        ms = summary[model_name]
        for exp_key in ["truth_epistemic_all", "agreement_nonepistemic", "type_classification"]:
            if exp_key in ms:
                s = ms[exp_key]
                print(f"  {exp_key}:")
                print(f"    Peak accuracy: {s['peak_accuracy']:.3f} at layer {s['peak_layer']} "
                      f"(normalized: {s['peak_layer_normalized']:.2f})")
                if "peak_selectivity" in s:
                    print(f"    Selectivity: {s['peak_selectivity']:.3f}")

        if model_name in stat_tests:
            st = stat_tests[model_name]
            print(f"  Statistical comparison (epistemic vs non-epistemic):")
            print(f"    Paired t-test: t={st['paired_ttest_t']:.3f}, p={st['paired_ttest_p']:.4f}")
            print(f"    Cohen's d: {st['cohens_d']:.3f}")
            print(f"    Peak layer diff: {st['peak_difference']:.3f}")

    return summary, stat_tests


if __name__ == "__main__":
    summary, stat_tests = run_full_analysis()
