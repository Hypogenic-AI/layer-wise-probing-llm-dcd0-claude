# Layer-wise Probing Analysis of Belief Encoding in LLMs

Investigating whether LLMs internally differentiate epistemic (factual) from non-epistemic (opinion) beliefs, and whether these representations reflect deep semantic understanding or surface-level linguistic patterns.

## Key Findings

- **Near-perfect belief type classification (99-100%)**: All GPT-2 models (117M to 1.5B parameters) develop clearly distinct internal representations for factual claims vs. opinion statements, detectable from middle layers onward.
- **Non-epistemic beliefs more decodable than truth**: Probes distinguish commonly-held from uncommon opinions (87-95% accuracy) more effectively than true from false factual claims (54-59% for mixed domains).
- **Domain-specific truth probing works well**: Geographic facts (70-86%), numerical comparisons (81-89%), and general claims (65-69%) are individually decodable, but the mixed-domain truth signal is weaker.
- **Statistically significant differences**: Paired t-tests show epistemic and non-epistemic belief encoding differ significantly (p < 1e-7, Cohen's d = 2.0-4.2) across all four model sizes.
- **Surface patterns dominate**: The near-perfect type classification likely reflects linguistic markers rather than deep semantic understanding, suggesting LLMs encode "how factual claims sound" more strongly than "what is true."

## Reproducing Results

```bash
# 1. Set up environment
uv venv && source .venv/bin/activate
uv add torch transformers scikit-learn numpy pandas matplotlib seaborn tqdm accelerate

# 2. Prepare datasets
python src/data_preparation.py

# 3. Extract activations (requires GPU)
python src/extract_activations.py

# 4. Run probing experiments and analysis
python src/run_fast_probing.py
```

## File Structure

```
.
├── REPORT.md                    # Full research report with results
├── README.md                    # This file
├── planning.md                  # Research plan and methodology
├── src/
│   ├── data_preparation.py      # Dataset loading and curation
│   ├── extract_activations.py   # Model activation extraction
│   ├── run_fast_probing.py      # Mass-mean and logistic probing
│   ├── analysis.py              # Statistical analysis and visualization
│   └── run_experiments.py       # Full pipeline orchestrator
├── results/
│   ├── activations/             # Saved model activations (per layer)
│   ├── datasets/                # Prepared CSV datasets
│   ├── metrics/                 # JSON results and analysis
│   └── plots/                   # Generated visualizations
├── datasets/                    # Pre-downloaded source datasets
├── papers/                      # Reference papers (35 PDFs)
├── code/                        # Cloned baseline repositories
├── literature_review.md         # Literature review
└── resources.md                 # Resource catalog
```

## Models Tested

| Model | Parameters | Layers | Hidden Dim |
|-------|-----------|--------|------------|
| GPT-2 Small | 117M | 12 | 768 |
| GPT-2 Medium | 345M | 24 | 1024 |
| GPT-2 Large | 774M | 36 | 1280 |
| GPT-2 XL | 1.5B | 48 | 1600 |

## Hardware

- 2x NVIDIA GeForce RTX 3090 (24GB each)
- Total experiment time: ~6 minutes

## Citation

If you use this work, please cite:

```
Layer-wise Probing Analysis of Belief Encoding in LLMs (2026)
Inspired by Professor Tan's inquiry and Vesga et al. (2025)
```

See [REPORT.md](REPORT.md) for full details and analysis.
