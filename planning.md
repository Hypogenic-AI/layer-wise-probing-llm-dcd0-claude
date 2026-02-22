# Research Plan: Layer-wise Probing Analysis of Belief Encoding in LLMs

## Motivation & Novelty Assessment

### Why This Research Matters
Understanding how LLMs internally represent different types of beliefs is crucial for AI safety, hallucination detection, and building trustworthy AI systems. If epistemic beliefs (evidence-based knowledge claims like "Water boils at 100°C") and non-epistemic beliefs (opinions, norms, preferences like "Democracy is the best form of government") are encoded differently across layers, this would reveal whether LLMs develop genuine semantic understanding of belief types or merely capture surface-level linguistic patterns. This distinction has direct implications for hallucination detection: if belief type is encoded in specific layers, targeted interventions could improve factual reliability.

### Gap in Existing Work
Based on the literature review, three key gaps exist:
1. **No epistemic vs. non-epistemic distinction**: Prior work (Marks & Tegmark 2023, Zhu et al. 2024, Bortoletto et al. 2024) studies "truth" and "belief" broadly but does not systematically probe the distinction between epistemic beliefs (knowledge claims with evidence) and non-epistemic beliefs (opinions, cultural values).
2. **Limited cross-architecture comparison**: No study has directly compared belief encoding patterns between GPT-2 and LLaMA in a controlled probing setup using the same datasets and methodology.
3. **Surface vs. deep understanding**: Marks & Tegmark note their probes may capture "commonly believed" rather than genuine truth understanding; this question remains open.

### Our Novel Contribution
We conduct the first systematic layer-wise probing study that:
1. Distinguishes **epistemic** (factual, evidence-based) from **non-epistemic** (opinion, preference, cultural) beliefs
2. Compares belief encoding across **GPT-2** (117M-1.5B params) and **LLaMA** architectures
3. Uses **control tasks** (random labels, PCA analysis) to assess whether representations reflect deep understanding or surface patterns
4. Connects findings to the psychological framework of Vesga et al. (2025) on belief categorization

### Experiment Justification
- **Experiment 1 (Layer-wise probing on factual truth)**: Establishes baseline — replicates core finding that truth is linearly decodable from internal layers, validates our methodology against Marks & Tegmark.
- **Experiment 2 (Epistemic vs. non-epistemic probing)**: Tests our core hypothesis — are epistemic and non-epistemic beliefs encoded differently across layers? Uses KaBLE-derived and custom-curated datasets.
- **Experiment 3 (Cross-architecture comparison)**: Tests whether belief encoding patterns are consistent across GPT-2 and LLaMA, revealing architecture-dependent vs. universal properties.
- **Experiment 4 (Control analysis)**: Random-label controls and PCA dimensionality analysis to distinguish genuine encoding from probe memorization.

---

## Research Question
Do open-source LLMs (GPT-2, LLaMA) encode epistemic and non-epistemic belief types in specific internal layers, and do these representations reflect deep semantic understanding or surface-level linguistic patterns?

## Background and Motivation
Inspired by Professor Tan's inquiry and the Vesga et al. (2025) psychological framework distinguishing epistemic from non-epistemic beliefs, we investigate whether LLMs internalize this distinction. Prior probing studies have shown that truth (Marks & Tegmark 2023), belief states (Zhu et al. 2024), and knowledge (Ju et al. 2024) are linearly decodable from specific layers, but the epistemic/non-epistemic distinction has not been explored.

## Hypothesis Decomposition

### H1: Truth/belief is linearly decodable from specific layers
- **Prediction**: Linear probes on middle-to-upper layers will achieve significantly above-chance accuracy for truth classification
- **Metric**: Probing accuracy > 70% at peak layers; random-label control < 55%

### H2: Epistemic and non-epistemic beliefs are encoded in different layers
- **Prediction**: Epistemic beliefs (factual knowledge claims) will peak in middle layers (where factual knowledge concentrates), while non-epistemic beliefs (opinions) will peak in later layers (where pragmatic/contextual processing occurs)
- **Metric**: Statistically significant difference in peak layer index between epistemic and non-epistemic probes (paired t-test, p < 0.05)

### H3: Belief encoding patterns differ between GPT-2 and LLaMA
- **Prediction**: LLaMA (larger, more capable) will show more differentiated layer-wise patterns than GPT-2
- **Metric**: Comparison of probing accuracy curves and peak layers across architectures

### H4: Representations reflect more than surface patterns
- **Prediction**: Control tasks (random labels) will achieve significantly lower accuracy than true-label probes; cross-dataset transfer will maintain above-chance accuracy
- **Metric**: Selectivity score (true accuracy - control accuracy) > 15 percentage points

## Proposed Methodology

### Approach
We use **linear probing** (logistic regression) on residual stream activations extracted from every layer of GPT-2 and LLaMA models. This follows the established methodology of Marks & Tegmark (2023) and Bortoletto et al. (2024). Linear probes are preferred because they test what is *linearly accessible* in the representation, providing a conservative estimate of encoded information (Hewitt & Liang 2019).

### Models
1. **GPT-2 Small** (117M params, 12 layers) — `openai-community/gpt2`
2. **GPT-2 Medium** (345M params, 24 layers) — `openai-community/gpt2-medium`
3. **GPT-2 Large** (774M params, 36 layers) — `openai-community/gpt2-large`
4. **GPT-2 XL** (1.5B params, 48 layers) — `openai-community/gpt2-xl`
5. **LLaMA-3.2-1B** (1.24B params, 16 layers) — `meta-llama/Llama-3.2-1B`
6. **LLaMA-3.2-3B** (3.21B params, 28 layers) — `meta-llama/Llama-3.2-3B`

### Datasets

#### Dataset 1: Factual Truth (Geometry of Truth)
- Source: `datasets/geometry_of_truth/datasets/`
- Files: `cities.csv`, `sp_en_trans.csv`, `common_claim_true_false.csv`, `companies_true_false.csv`
- Labels: Binary (true=1, false=0)
- Use: Establishes baseline truth probing

#### Dataset 2: Epistemic Beliefs (Custom-curated)
We construct a dataset of epistemic belief statements — claims grounded in evidence/knowledge:
- Source 1: Geometry of Truth factual datasets (cities, translations, companies)
- Source 2: KaBLE `direct-fact-verification.jsonl` and `verification-of-first-person-knowledge.jsonl`
- Source 3: CounterFact true/false pairs
- Format: Statement + binary label (true/false)

#### Dataset 3: Non-epistemic Beliefs (Custom-curated)
We construct a dataset of non-epistemic belief statements — opinions, preferences, cultural/moral values:
- Source: Custom-generated statements covering subjective claims, aesthetic judgments, ethical statements, cultural norms
- Format: Statement + binary label (commonly-held=1, controversial/opposite=0)
- This tests whether LLMs encode opinion-type statements differently from factual claims

### Experimental Steps

1. **Data preparation**: Load and preprocess all datasets; create balanced train/test splits (80/20)
2. **Activation extraction**: For each model, pass statements through and extract residual stream activations at every layer at the last token position
3. **Linear probing**: Train logistic regression probes on each layer's activations for each dataset
4. **Control tasks**: Train probes with randomly permuted labels on same activations
5. **Cross-dataset transfer**: Train on epistemic, test on non-epistemic (and vice versa)
6. **PCA visualization**: Visualize layer-wise representations colored by belief type
7. **Statistical analysis**: Compare peak layers, accuracies, and selectivity scores

### Baselines
- Random-label probes (Hewitt & Liang 2019) — tests probe memorization
- Majority-class baseline — lower bound
- PCA component analysis — tests dimensionality of belief representations

### Evaluation Metrics
- **Probing accuracy** (primary): Classification accuracy per layer
- **Selectivity** (Hewitt & Liang 2019): True accuracy minus random-label accuracy
- **Cross-dataset transfer accuracy**: Generalization test
- **Peak layer index**: Layer with highest probing accuracy for each belief type
- **AUC**: Area under ROC curve for probe discrimination

### Statistical Analysis Plan
- Paired t-tests comparing peak layer indices between epistemic and non-epistemic probes
- Bootstrap confidence intervals (n=1000) for probing accuracies
- Bonferroni correction for multiple comparisons across layers
- Effect size (Cohen's d) for accuracy differences
- Significance level: α = 0.05

## Expected Outcomes
- **H1 supported**: Truth probing accuracy > 70% at peak layers across all models
- **H2 supported**: Epistemic beliefs peak in middle layers; non-epistemic beliefs show different (later or more distributed) peak patterns
- **H2 refuted**: Both belief types show same layer-wise profile → suggests LLMs don't differentiate at the representation level
- **H3 supported**: LLaMA shows sharper, more differentiated patterns than GPT-2
- **H4 supported**: High selectivity scores → genuine encoding; low selectivity → surface patterns

## Timeline and Milestones
1. Environment setup + data prep: 15 min
2. Activation extraction pipeline: 30 min
3. GPT-2 experiments (all 4 sizes): 30 min
4. LLaMA experiments (2 sizes): 30 min
5. Control tasks + cross-dataset transfer: 20 min
6. Analysis + visualization: 30 min
7. Documentation: 25 min

## Potential Challenges
1. **Memory constraints**: GPT-2 XL and LLaMA-3B may need careful batch sizing → use batch_size=32 for large models
2. **Dataset construction for non-epistemic beliefs**: No standard dataset exists → curate carefully with clear criteria
3. **Model download time**: LLaMA models require HuggingFace authentication → fall back to GPT-2 family if blocked
4. **Probe overfitting**: Small datasets may lead to overfitting → use cross-validation (5-fold)

## Success Criteria
1. All experiments complete with reproducible results
2. Clear layer-wise probing accuracy curves for all models and belief types
3. Statistical tests comparing epistemic vs. non-epistemic encoding
4. Control tasks validate that probes capture genuine information
5. Comprehensive REPORT.md with visualizations and statistical analysis
