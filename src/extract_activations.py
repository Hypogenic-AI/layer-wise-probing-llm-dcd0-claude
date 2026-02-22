"""
Extract residual stream activations from LLMs at every layer.

Supports GPT-2 family and LLaMA models via HuggingFace transformers.
Extracts hidden states at the last token position for each input statement.
"""

import torch
import numpy as np
import json
import gc
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

BASE_DIR = Path("/data/hypogenicai/workspaces/layer-wise-probing-llm-dcd0-claude")
RESULTS_DIR = BASE_DIR / "results"
ACTIVATIONS_DIR = RESULTS_DIR / "activations"
ACTIVATIONS_DIR.mkdir(parents=True, exist_ok=True)

# Model configs: (hf_name, short_name, n_layers)
MODEL_CONFIGS = {
    "gpt2": ("openai-community/gpt2", "gpt2_small", 12),
    "gpt2-medium": ("openai-community/gpt2-medium", "gpt2_medium", 24),
    "gpt2-large": ("openai-community/gpt2-large", "gpt2_large", 36),
    "gpt2-xl": ("openai-community/gpt2-xl", "gpt2_xl", 48),
}


def load_model_and_tokenizer(model_key, device="cuda:0"):
    """Load a model and tokenizer from HuggingFace."""
    hf_name, short_name, n_layers = MODEL_CONFIGS[model_key]
    print(f"Loading {hf_name}...")

    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        hf_name,
        output_hidden_states=True,
        torch_dtype=torch.float16,
    ).to(device)
    model.eval()

    print(f"  Loaded {short_name}: {n_layers} layers, device={device}")
    return model, tokenizer, short_name, n_layers


def extract_activations(model, tokenizer, statements, device="cuda:0", batch_size=32):
    """
    Extract hidden states from all layers for a list of statements.

    Returns: dict mapping layer_idx -> np.array of shape (n_statements, hidden_dim)
    Extracts activations at the last non-padding token position.
    """
    n_layers = model.config.num_hidden_layers
    all_hidden = {i: [] for i in range(n_layers + 1)}  # +1 for embedding layer

    for start_idx in tqdm(range(0, len(statements), batch_size), desc="Extracting"):
        batch = statements[start_idx : start_idx + batch_size]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        hidden_states = outputs.hidden_states  # tuple of (n_layers+1) tensors

        # Find last non-padding token position for each sample
        attention_mask = inputs["attention_mask"]
        # last non-pad token index = sum of attention mask - 1
        last_token_idx = attention_mask.sum(dim=1) - 1  # shape: (batch_size,)

        for layer_idx, hs in enumerate(hidden_states):
            # hs shape: (batch_size, seq_len, hidden_dim)
            # Extract activation at last token position
            batch_indices = torch.arange(hs.size(0), device=device)
            layer_acts = hs[batch_indices, last_token_idx, :]  # (batch_size, hidden_dim)
            all_hidden[layer_idx].append(layer_acts.cpu().float().numpy())

    # Concatenate all batches
    result = {}
    for layer_idx in all_hidden:
        result[layer_idx] = np.concatenate(all_hidden[layer_idx], axis=0)

    return result


def extract_and_save(model_key, datasets, device="cuda:0", batch_size=32):
    """Extract activations for all datasets with a given model and save to disk."""
    model, tokenizer, short_name, n_layers = load_model_and_tokenizer(model_key, device)

    model_dir = ACTIVATIONS_DIR / short_name
    model_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "model": model_key,
        "short_name": short_name,
        "n_layers": n_layers,
        "hidden_dim": model.config.hidden_size,
        "datasets": {},
    }

    for dataset_name, df in datasets.items():
        print(f"\n  Extracting activations for {dataset_name} ({len(df)} samples)...")
        statements = df["statement"].tolist()
        labels = df["label"].values

        activations = extract_activations(model, tokenizer, statements, device, batch_size)

        # Save activations and labels
        ds_dir = model_dir / dataset_name
        ds_dir.mkdir(parents=True, exist_ok=True)

        for layer_idx, acts in activations.items():
            np.save(ds_dir / f"layer_{layer_idx}.npy", acts)

        np.save(ds_dir / "labels.npy", labels)

        # Save type_label if available (for type_classification dataset)
        if "type_label" in df.columns:
            np.save(ds_dir / "type_labels.npy", df["type_label"].values)

        if "belief_type" in df.columns:
            np.save(ds_dir / "belief_types.npy",
                     np.array([1 if bt == "epistemic" else 0 for bt in df["belief_type"]]))

        metadata["datasets"][dataset_name] = {
            "n_samples": len(df),
            "n_layers_saved": len(activations),
            "activation_shape": activations[0].shape,
        }

        print(f"    Saved {len(activations)} layers, shape={activations[0].shape}")

    # Save metadata
    with open(model_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    # Clean up GPU memory
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return metadata


if __name__ == "__main__":
    import pandas as pd

    # Load prepared datasets
    ds_dir = RESULTS_DIR / "datasets"
    datasets = {}
    for csv_file in ds_dir.glob("*.csv"):
        datasets[csv_file.stem] = pd.read_csv(csv_file)

    print(f"Loaded {len(datasets)} datasets")

    for model_key in MODEL_CONFIGS:
        print(f"\n{'='*60}")
        print(f"Processing {model_key}")
        print(f"{'='*60}")
        metadata = extract_and_save(model_key, datasets, device="cuda:0", batch_size=64)
        print(f"Done: {model_key}")
