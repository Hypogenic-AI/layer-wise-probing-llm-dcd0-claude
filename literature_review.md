# Literature Review: Layer-wise Probing Analysis of Belief Encoding in LLMs

## Research Area Overview

This literature review surveys research on how large language models (LLMs) internally represent beliefs, truth, and knowledge across their layers. The field draws from three converging research threads: (1) probing methodology for neural network interpretability, (2) mechanistic analysis of transformer internal representations, and (3) belief and epistemic reasoning in language models. Together, these threads inform our central research hypothesis: that open-source LLMs such as GPT-2 and Llama encode epistemic and non-epistemic belief types in specific internal layers.

---

## Key Papers

### Core Belief Encoding Papers

#### Zhu et al. (2024) — "Language Models Represent Beliefs of Self and Others"
- **Venue**: ICML 2024
- **Key Contribution**: First demonstration that LLMs develop linearly-decodable internal representations encoding belief status of multiple agents (self/oracle vs. others/protagonist)
- **Methodology**: Linear logistic regression probes on attention head activations at the final token position across all layers and heads of Mistral-7B and DeepSeek-7B
- **Layer-wise Findings**:
  - **Oracle (model's own) belief**: Broadly encoded across layers 13-15 (peak ~97.8% accuracy), excluding initial layers
  - **Protagonist (other's) belief**: Concentrated in Layer 10 attention heads (~78% accuracy); most heads show only baseline (~50%) accuracy
  - **Joint belief**: Concentrated in Layers 12-16 (~79% accuracy)
- **Datasets**: BigToM, ToMi
- **Intervention**: +TpFo direction dramatically improves False Belief accuracy (33% → 66% on Mistral-7B)
- **Code**: https://walter0807.github.io/RepBelief/ (uses nnsight toolkit)

#### Bortoletto et al. (2024) — "Brittle Minds, Fixable Activations"
- **Venue**: Findings of EMNLP 2025
- **Key Contribution**: First systematic investigation of belief representations across 12 models (Pythia 70M-12B, Llama-2 7B-70B, base + chat)
- **Methodology**: Linear probing on residual stream activations at every layer; PCA control tasks; Contrastive Activation Addition (CAA) for intervention
- **Layer-wise Findings**:
  - **Oracle beliefs**: Peak accuracy in early layers (layers 1-2), ~95-100% even in smallest models
  - **Protagonist beliefs**: Peak accuracy at intermediate layers; accuracy increases with model size (logarithmic for base, linear for fine-tuned)
  - Belief representations occupy a **low-dimensional subspace** (recoverable with k=10 PCA components)
- **Key Result**: Fine-tuning produces dramatic improvements for protagonist belief encoding (+29% for Llama-2-7B-chat vs. base); protagonist representations are **brittle** to prompt variations while oracle representations are robust
- **Dataset**: BigToM
- **Code**: https://git.hcics.simtech.uni-stuttgart.de/public-projects/mental-states-in-LMs

#### Churina et al. (2025) — "Layer of Truth: Probing Belief Shifts under Continual Pre-Training Poisoning"
- **Key Contribution**: Tracks how internal belief representations shift across layers when LLMs are exposed to misinformation during continued pretraining
- **Relevance**: Directly studies layer-wise belief probing under adversarial conditions

### Truth and Knowledge Representation

#### Marks & Tegmark (2023) — "The Geometry of Truth"
- **Venue**: COLM 2024
- **Key Contribution**: Discovers emergent linear structure in LLM representations of true/false statements; proposes mass-mean probing
- **Models**: LLaMA-2-7B/13B/70B
- **Layer-wise Findings**:
  - Truth representations emerge in **middle layers** (~layer 15 for LLaMA-2-13B)
  - Three groups of causally implicated hidden states: (a) early layers encode entity identity, (b) middle layers encode truth value, (c) late layers encode prediction
  - Linear structure **emerges with scale**: 7B shows surface-level clustering; 13B shows clear linear separation; 70B shows unified abstract truth direction
  - Alignment rotation across layers: antipodal → orthogonal → aligned (cities + neg_cities at layers 6 → 11 → 15)
- **Datasets**: 15 curated true/false CSV datasets (cities, translations, comparisons, etc.)
- **Code**: https://github.com/saprmarks/geometry-of-truth

#### Burns et al. (2022) — "Discovering Latent Knowledge in Language Models Without Supervision"
- **Venue**: ICLR 2023
- **Key Contribution**: Proposes Contrast-Consistent Search (CCS) for unsupervised discovery of truth representations in LLM activations
- **Models**: RoBERTa, DeBERTa, GPT-J, T5, UnifiedQA, T0
- **Layer-wise Findings**:
  - **Middle layers can outperform final layers** for truth detection (esp. T5 and UnifiedQA encoders)
  - Under misleading prompts, encoder middle layers remain robust while decoder hidden states degrade
  - GPT-J peaks around layers 14-20 (~65%)
- **Datasets**: 10 QA datasets (IMDB, BoolQ, COPA, etc.)
- **Code**: https://github.com/collin-burns/discovering_latent_knowledge

#### Azaria & Mitchell (2023) — "The Internal State of an LLM Knows When it's Lying"
- **Venue**: EMNLP Findings 2023
- **Key Contribution**: Shows classifiers trained on hidden states can detect when LLMs generate falsehoods; 6 topic-specific true/false datasets
- **Relevance**: Establishes that internal representations encode truthfulness independently of output

#### Liu et al. (2023) — "Cognitive Dissonance"
- **Key Contribution**: Examines the gap between LLMs' internal truth representations (probes) and output probabilities; shows models can "know" truth internally while outputting falsehoods

### Layer-wise Knowledge Encoding

#### Ju et al. (2024) — "How Large Language Models Encode Context Knowledge?"
- **Key Contribution**: First comprehensive layer-wise probing study of context knowledge encoding
- **Models**: LLaMA-2 7B/13B/70B (base + chat)
- **Methodology**: V-usable information metric (more discriminative than raw accuracy) with linear probes
- **Layer-wise Findings**:
  - **Upper layers encode more context knowledge** (gradual increase from lower to upper layers)
  - **Knowledge-related entity tokens** encode information earlier (lower layers), while other tokens catch up via self-attention in upper layers
  - LLaMA-2-70B shows a **decrease in the last few layers** (buffer effect)
  - Chat models significantly outperform base models at utilizing context knowledge
- **Code**: https://github.com/Jometeorie/probing_llama

#### Jin et al. (2024) — "Exploring Concept Depth"
- **Key Contribution**: Proposes "Concept Depth" — more complex concepts are processed at deeper layers

#### Meng et al. (2022) — "Locating and Editing Factual Associations in GPT" (ROME)
- **Venue**: NeurIPS 2022
- **Key Contribution**: Uses causal tracing to locate factual knowledge in specific MLP layers of GPT models
- **Key Finding**: Factual associations are concentrated in middle-layer MLPs; early layers encode subject, middle layers recall facts

### Probing Methodology (Foundational)

#### Alain & Bengio (2016) — "Understanding Intermediate Layers Using Linear Classifier Probes"
- **Key Contribution**: Proposes using linear classifier probes to monitor features at every layer; establishes that linear separability increases monotonically with depth

#### Conneau et al. (2018) — "Probing Sentence Embeddings for Linguistic Properties"
- **Key Contribution**: Defines 10 probing tasks for evaluating linguistic properties in sentence embeddings

#### Belinkov (2021) — "Probing Classifiers: Promises, Shortcomings, and Advances"
- **Key Contribution**: Comprehensive survey of probing methodology; discusses what probes can and cannot reveal

#### Hewitt & Liang (2019) — "Designing and Interpreting Probes with Control Tasks"
- **Key Contribution**: Proposes control tasks (selectivity) to assess whether probes extract genuine encoded information

#### Voita & Titov (2020) — "Information-Theoretic Probing with MDL"
- **Key Contribution**: MDL-based probing for more faithful measurement of how well representations encode information

#### Gurnee et al. (2023) — "Finding Neurons in a Haystack"
- **Key Contribution**: Uses sparse linear probes to identify individual neurons encoding high-level features in LLMs across layers

### Epistemic Reasoning

#### Suzgun et al. (2024) — "Belief in the Machine" (KaBLE)
- **Key Contribution**: 13,000 questions across 13 epistemic tasks testing LLMs' ability to distinguish knowledge, belief, and fact
- **Dataset**: KaBLE benchmark (publicly available)

#### Herrmann & Levinstein (2024) — "Standards for Belief Representations in LLMs"
- **Key Contribution**: Establishes theoretical standards for what counts as genuine belief representation

#### Li et al. (2025) — "Representations of Fact, Fiction and Forecast"
- **Key Contribution**: Studies how LLMs internally represent facts vs. fiction vs. forecasts

### Mechanistic Interpretability

#### Tenney et al. (2019) — "BERT Rediscovers the Classical NLP Pipeline"
- **Key Finding**: BERT represents NLP pipeline steps in sequence: POS → parsing → NER → semantic roles → coreference

#### Jawahar et al. (2019) — "What Does BERT Learn about the Structure of Language?"
- **Key Finding**: Lower layers capture phrase-level syntax, middle layers capture syntactic features, upper layers capture semantic features

#### Li et al. (2023) — "Inference-Time Intervention"
- **Key Contribution**: Identifies truthfulness-related directions in attention heads; steers LLM activations to improve truthfulness
- **Code**: https://github.com/likenneth/honest_llama

---

## Common Methodologies

### Probing Approaches
1. **Linear classifier probes** (logistic regression): Used in Zhu et al., Bortoletto et al., Ju et al., Marks & Tegmark — standard approach for testing linear decodability
2. **V-usable information**: Used in Ju et al. — more discriminative than raw accuracy, information-theoretic
3. **Mass-mean probing** (difference-in-means): Proposed by Marks & Tegmark — optimization-free, most causally implicated
4. **Contrast-Consistent Search (CCS)**: Burns et al. — unsupervised, finds truth direction via consistency constraints
5. **Sparse probing**: Gurnee et al. — identifies individual neurons responsible for specific features

### Intervention Methods
1. **Inference-Time Intervention (ITI)**: Li et al. — steers attention head activations
2. **Contrastive Activation Addition (CAA)**: Bortoletto et al. — adds steering vectors to residual stream
3. **Causal tracing / activation patching**: Meng et al., Marks & Tegmark — swaps activations to identify causal components

### Representation Extraction
- **Residual stream activations** at specific token positions (final token, period token, entity tokens)
- **Attention head activations** (individual heads or grouped by layer)
- **MLP layer outputs** vs. attention layer outputs

---

## Standard Baselines
- Zero-shot / few-shot prompting accuracy
- Random-label control probes (selectivity)
- PCA dimensionality reduction controls
- Calibrated few-shot baselines
- Intervention along random directions (null control)

## Evaluation Metrics
- **Probing accuracy** (binary/multinomial classification)
- **V-usable information** (information-theoretic)
- **Normalized Indirect Effect (NIE)** for causal interventions
- **Cross-dataset transfer accuracy** for generalization
- **Cosine similarity** between probe directions across tasks

## Datasets in the Literature
| Dataset | Used In | Task Type |
|---------|---------|-----------|
| BigToM | Zhu et al., Bortoletto et al. | Theory of Mind (true/false belief) |
| ToMi | Zhu et al. | Theory of Mind |
| Geometry of Truth CSVs | Marks & Tegmark | True/false factual statements |
| Azaria True-False | Azaria & Mitchell | Topic-specific true/false |
| TruthfulQA | Li et al. (ITI) | Truthfulness |
| ConflictQA | Ju et al. | Parametric vs. contextual knowledge |
| KaBLE | Suzgun et al. | Epistemic reasoning |
| LAMA | Petroni et al. | Factual knowledge probing |
| CounterFact | Meng et al. (ROME) | Factual recall |

---

## Gaps and Opportunities

1. **Limited model coverage**: Most studies focus on LLaMA-2 or single model families. GPT-2 (small, medium, large, XL) has been less studied for belief encoding despite being fully open-source and well-understood architecturally.

2. **Epistemic vs. non-epistemic distinction**: While papers study "truth" and "belief," the specific distinction between epistemic beliefs (knowledge claims with evidence) and non-epistemic beliefs (opinions, preferences, cultural norms) has not been systematically probed.

3. **Surface-level vs. deep understanding**: Marks & Tegmark note their truth probes may capture "commonly believed" rather than genuine truth understanding. The question of whether LLM belief representations reflect deep semantic understanding or surface-level linguistic patterns remains open.

4. **Cross-architecture comparison**: No study has directly compared belief encoding patterns between GPT-2 (decoder-only, smaller) and LLaMA (decoder-only, larger) in a controlled probing setup.

5. **Training dynamics**: How belief representations develop during training is poorly understood (noted as future work by Zhu et al.).

---

## Recommendations for Our Experiment

### Recommended Models
- **GPT-2** (small/medium/large/XL): Well-understood architecture, fully open-source, enables scale comparison within one family
- **LLaMA-2-7B** (base + chat) or **LLaMA-3-8B**: Standard for current probing studies, enables comparison with literature
- Optionally: **Pythia** family for training data transparency

### Recommended Datasets
1. **Primary**: Geometry of Truth datasets (true/false statements with ready-to-use probing code for LLaMA)
2. **Primary**: BigToM (true/false belief conditions, used by core belief papers)
3. **Secondary**: KaBLE (epistemic reasoning — distinguishes knowledge vs. belief)
4. **Secondary**: TruthfulQA (standard truthfulness benchmark)
5. **Validation**: CounterFact-Tracing (factual recall)

### Recommended Methodology
1. **Linear probing** (logistic regression) on residual stream activations at every layer, following Marks & Tegmark and Bortoletto et al.
2. **V-usable information** as primary metric (more discriminative than accuracy), following Ju et al.
3. **Control tasks**: Random label permutation + PCA dimensionality analysis, following Bortoletto et al.
4. **Token positions**: Final token, entity tokens, and period tokens (all shown to carry different information)
5. **Activation types**: Compare residual stream, attention head, and MLP outputs

### Recommended Metrics
- Probing accuracy and V-usable information across layers
- Cross-dataset transfer accuracy (train on one belief type, test on another)
- PCA visualization of truth/belief separation per layer
- Control task selectivity (random labels baseline)

### Methodological Considerations
- Use mass-mean probing (Marks & Tegmark) alongside logistic regression — it is more causally implicated
- Include control for "probability" vs. "truth" (the `likely` dataset from Geometry of Truth)
- Test both base and instruction-tuned models (fine-tuning dramatically affects belief representations per Bortoletto et al.)
- Consider Bonferroni correction for multiple hypothesis testing across many layers × heads (per Zhu et al.)
