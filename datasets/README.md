# Downloaded Datasets

This directory contains datasets for the Layer-wise Probing Analysis of Belief Encoding in LLMs.
Data files are NOT committed to git due to size. Follow the download instructions below.

## Dataset 1: Geometry of Truth (Marks & Tegmark, 2023)

### Overview
- **Source**: https://github.com/saprmarks/geometry-of-truth
- **Size**: 15 CSV files, ~500KB total (small)
- **Format**: CSV with columns for statement text and truth label
- **Task**: True/false statement classification for probing truth representations
- **License**: Open/Academic
- **Paper**: "The Geometry of Truth" (COLM 2024, arXiv:2310.06824)

### Download Instructions
```bash
git clone --depth 1 https://github.com/saprmarks/geometry-of-truth.git datasets/geometry_of_truth
```

### Contents
- `cities.csv` — "The city of [city] is in [country]" (1,496 rows)
- `neg_cities.csv` — Negated city statements (1,496 rows)
- `sp_en_trans.csv` — Spanish-English translations (354 rows)
- `larger_than.csv` / `smaller_than.csv` — Size comparisons (1,980 rows each)
- `common_claim_true_false.csv` — Various claims (4,450 rows)
- `counterfact_true_false.csv` — Counterfactual claims (31,960 rows)
- `likely.csv` — Control dataset for probability vs. truth (10,000 rows)
- Plus additional conjunction/disjunction datasets

### Loading
```python
import pandas as pd
df = pd.read_csv("datasets/geometry_of_truth/datasets/cities.csv")
```

### Notes
- Includes ready-to-use `generate_acts.py` script for extracting LLaMA activations
- The `likely` dataset is a critical control for distinguishing truth from probability

---

## Dataset 2: KaBLE — Belief in the Machine (Suzgun et al., 2024)

### Overview
- **Source**: https://github.com/suzgunmirac/belief-in-the-machine
- **Size**: 13,000 questions across 13 epistemic tasks
- **Format**: JSON/CSV
- **Task**: Epistemic reasoning — distinguishing knowledge, belief, and fact
- **License**: MIT
- **Paper**: "Belief in the Machine" (Nature Machine Intelligence, 2025)

### Download Instructions
```bash
git clone --depth 1 https://github.com/suzgunmirac/belief-in-the-machine.git datasets/kable
```

### Notes
- 1,000 seed sentences from 10 disciplines (history, literature, medicine, law, etc.)
- Both factual and false versions of statements
- Tests first-person and third-person epistemic reasoning
- Directly relevant for distinguishing epistemic vs. non-epistemic belief types

---

## Dataset 3: BigToM (Gandhi et al., 2023)

### Overview
- **Source**: https://github.com/cicl-stanford/procedural-evals-tom
- **Size**: ~5,000 evaluations across 25 controls
- **Format**: JSON, procedurally generated
- **Task**: Theory of Mind reasoning — true belief vs. false belief conditions
- **License**: Academic use
- **Paper**: NeurIPS 2023 Datasets and Benchmarks Track

### Download Instructions
```bash
git clone --depth 1 https://github.com/cicl-stanford/procedural-evals-tom.git datasets/bigtom
```

### Notes
- Causal template with protagonist desires, actions, beliefs, and causal events
- Three task types: Forward Belief, Forward Action, Backward Belief
- Used by both Zhu et al. (2024) and Bortoletto et al. (2024) for belief probing

---

## Dataset 4: ToMi (Le et al., 2019)

### Overview
- **Source**: https://github.com/facebookresearch/ToMi
- **Size**: Configurable via generator; ~18,000 rows in NLI version
- **Format**: Text files + generation scripts
- **Task**: Theory of Mind — true/false belief and second-order ToM
- **License**: CC-BY-NC 4.0
- **Paper**: EMNLP 2019

### Download Instructions
```bash
git clone --depth 1 https://github.com/facebookresearch/ToMi.git datasets/tomi
```

### Notes
- Used by Zhu et al. (2024) for cross-dataset generalization tests
- Different narrative templates from BigToM — good for testing representation transfer

---

## Dataset 5: TruthfulQA (Lin et al., 2022)

### Overview
- **Source**: https://huggingface.co/datasets/truthfulqa/truthful_qa
- **Size**: 817 questions across 38 categories
- **Format**: Parquet/JSON on HuggingFace
- **Task**: Measuring LLM truthfulness
- **License**: Apache 2.0

### Download Instructions
```python
from datasets import load_dataset
ds = load_dataset("truthfulqa/truthful_qa", "generation")
ds.save_to_disk("datasets/truthfulqa")
```

### Notes
- Standard benchmark for truthfulness evaluation
- Used by ITI (Li et al., 2023) for intervention experiments

---

## Dataset 6: CounterFact-Tracing (Meng et al., 2022)

### Overview
- **Source**: https://huggingface.co/datasets/NeelNanda/counterfact-tracing
- **Size**: 21,919 factual relations
- **Format**: Parquet on HuggingFace
- **Task**: Factual recall with true and false targets
- **License**: MIT

### Download Instructions
```python
from datasets import load_dataset
ds = load_dataset("NeelNanda/counterfact-tracing")
ds.save_to_disk("datasets/counterfact")
```

### Notes
- From the ROME paper; designed for mechanistic interpretability of factual recall
- Contains subject, relation, true target, and false target
