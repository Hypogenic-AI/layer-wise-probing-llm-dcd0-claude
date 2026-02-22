# Resources Catalog

## Summary

This document catalogs all resources gathered for the research project: **Layer-wise Probing Analysis of Belief Encoding in LLMs**. The project investigates whether open-source LLMs (GPT-2, Llama) encode epistemic and non-epistemic belief types in specific internal layers, and whether these representations reflect deep semantic understanding or surface-level linguistic patterns.

---

## Papers

Total papers downloaded: **35**

| # | Title | Authors | Year | File | Key Info |
|---|-------|---------|------|------|----------|
| 1 | Language Models Represent Beliefs of Self and Others | Zhu et al. | 2024 | papers/zhu2024_beliefs_self_others.pdf | Core paper: belief probing in attention heads |
| 2 | Brittle Minds, Fixable Activations | Bortoletto et al. | 2024 | papers/bortoletto2024_brittle_minds_belief.pdf | 12-model belief probing study |
| 3 | Layer of Truth: Probing Belief Shifts | Churina et al. | 2025 | papers/churina2025_layer_truth_belief_shifts.pdf | Belief shifts under poisoning |
| 4 | Language Models use Lookbacks to Track Beliefs | Prakash et al. | 2025 | papers/prakash2025_lookbacks_beliefs.pdf | Attention patterns for belief tracking |
| 5 | The Geometry of Truth | Marks & Tegmark | 2023 | papers/marks2023_geometry_truth.pdf | Linear truth structure in LLMs |
| 6 | Discovering Latent Knowledge Without Supervision | Burns et al. | 2022 | papers/burns2022_discovering_latent_knowledge.pdf | CCS unsupervised probing |
| 7 | The Internal State of an LLM Knows When it's Lying | Azaria & Mitchell | 2023 | papers/azaria2023_internal_state_lying.pdf | Hidden states encode truthfulness |
| 8 | Cognitive Dissonance | Liu et al. | 2023 | papers/liu2023_cognitive_dissonance.pdf | Internal truth vs. output gap |
| 9 | Inference-Time Intervention | Li et al. | 2023 | papers/li2023_inference_time_intervention.pdf | Truthfulness steering |
| 10 | Overthinking the Truth | Halawi et al. | 2023 | papers/halawi2023_overthinking_truth.pdf | Processing false demonstrations |
| 11 | Still No Lie Detector | Levinstein & Herrmann | 2023 | papers/levinstein2023_lie_detector.pdf | Critical analysis of probes |
| 12 | Layer-Wise Probing Study (Context Knowledge) | Ju et al. | 2024 | papers/ju2024_layerwise_context_knowledge.pdf | V-information layer-wise probing |
| 13 | Exploring Concept Depth | Jin et al. | 2024 | papers/jin2024_concept_depth.pdf | Concept depth hypothesis |
| 14 | Locating and Editing Factual Associations (ROME) | Meng et al. | 2022 | papers/meng2022_locating_editing_gpt.pdf | Causal tracing in GPT |
| 15 | Probing LLaMA Across Scales and Layers | Chen et al. | 2023 | papers/chen2023_probing_llama.pdf | LLaMA layer-wise probing |
| 16 | DoLa: Contrasting Layers | Chuang et al. | 2023 | papers/chuang2023_dola.pdf | Layer contrast for factuality |
| 17 | Dissecting Factual Recall | Geva et al. | 2023 | papers/geva2023_factual_recall.pdf | Information flow in factual recall |
| 18 | Probing Classifiers Survey | Belinkov | 2021 | papers/belinkov2021_probing_classifiers.pdf | Probing methodology survey |
| 19 | Linear Classifier Probes | Alain & Bengio | 2016 | papers/alain2016_linear_classifier_probes.pdf | Foundational probing paper |
| 20 | Control Tasks for Probes | Hewitt & Liang | 2019 | papers/hewitt2019_control_probes.pdf | Probe selectivity |
| 21 | Probing Sentence Embeddings | Conneau et al. | 2018 | papers/conneau2018_probing_sentence_embeddings.pdf | 10 probing tasks |
| 22 | MDL Probing | Voita & Titov | 2020 | papers/voita2020_mdl_probing.pdf | Information-theoretic probing |
| 23 | Finding Neurons (Sparse Probing) | Gurnee et al. | 2023 | papers/gurnee2023_sparse_probing.pdf | Individual neuron identification |
| 24 | What Does BERT Look at? | Clark et al. | 2019 | papers/clark2019_bert_attention.pdf | BERT attention analysis |
| 25 | What Does BERT Learn? | Jawahar et al. | 2019 | papers/jawahar2019_bert_structure_language.pdf | Layer-wise linguistic analysis |
| 26 | BERT Rediscovers the NLP Pipeline | Tenney et al. | 2019 | papers/tenney2019_bert_nlp_pipeline.pdf | Sequential layer specialization |
| 27 | Linguistic Knowledge Transferability | Liu et al. | 2019 | papers/liu2019_linguistic_knowledge_transferability.pdf | Layer-wise knowledge in ELMo/GPT |
| 28 | BERTnesia | Wallat et al. | 2020 | papers/wallat2020_bertnesia.pdf | Knowledge capture/forgetting |
| 29 | Localization in BERToids | Fayyaz et al. | 2021 | papers/fayyaz2021_localization_bertoids.pdf | Different models, different layers |
| 30 | Emergent Linear Representations | Nanda et al. | 2023 | papers/nanda2023_emergent_linear_representations.pdf | Linear world models |
| 31 | Belief State Geometry | Shai et al. | 2024 | papers/shai2024_belief_state_geometry.pdf | Belief state in residual stream |
| 32 | Standards for Belief Representations | Herrmann & Levinstein | 2024 | papers/herrmann2024_belief_standards.pdf | Theoretical standards |
| 33 | Belief in the Machine (KaBLE) | Suzgun et al. | 2024 | papers/suzgun2024_epistemological_blindspots.pdf | Epistemic reasoning benchmark |
| 34 | Unveiling Theory of Mind in LLMs | Jamali et al. | 2023 | papers/jamali2023_tom_llms.pdf | ToM neurons parallel |
| 35 | Representations of Fact, Fiction, Forecast | Li et al. | 2025 | papers/li2025_fact_fiction_forecast.pdf | Epistemic representation |

See `papers/README.md` for detailed descriptions.

---

## Datasets

Total datasets downloaded: **6** (+ 1 failed)

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| Geometry of Truth | GitHub | 15 CSV files | True/false probing | datasets/geometry_of_truth/ | Most directly relevant; includes activation generation code |
| KaBLE | GitHub | 13,000 questions | Epistemic reasoning | datasets/kable/ | Distinguishes knowledge vs. belief |
| BigToM | GitHub | ~5,000 evals | Theory of Mind | datasets/bigtom/ | True/false belief conditions |
| ToMi | GitHub | Configurable | Theory of Mind | datasets/tomi/ | Second-order ToM; cross-dataset testing |
| TruthfulQA | HuggingFace | 817 questions | Truthfulness | datasets/truthfulqa/ | Standard benchmark |
| CounterFact-Tracing | HuggingFace | 21,919 relations | Factual recall | datasets/counterfact/ | True/false targets for probing |

See `datasets/README.md` for detailed descriptions and download instructions.

---

## Code Repositories

Total repositories cloned: **5**

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| CCS | github.com/collin-burns/discovering_latent_knowledge | Unsupervised truth probing | code/ccs_latent_knowledge/ | Core methodology |
| ROME | github.com/kmeng01/rome | Causal tracing & knowledge editing | code/rome/ | Layer localization |
| nnsight | github.com/JadenFiotto-Kaufman/nnsight | Model inspection toolkit | code/nnsight/ | Used by belief papers |
| probing_llama | github.com/Jometeorie/probing_llama | Layer-wise probing with V-info | code/probing_llama/ | Most directly relevant code |
| honest_llama | github.com/likenneth/honest_llama | Truthfulness intervention | code/honest_llama_iti/ | Activation extraction pipeline |

See `code/README.md` for detailed descriptions.

---

## Resource Gathering Notes

### Search Strategy
1. Used paper-finder with 5 complementary queries covering probing, layer-wise analysis, belief encoding, epistemic reasoning, and BERT/LLM knowledge
2. Searched across arXiv, Semantic Scholar, and Papers with Code
3. Prioritized papers with relevance scores >= 2, plus selected relevance-1 papers on core topics
4. Cross-referenced datasets and code from cited papers

### Selection Criteria
- **Papers**: Prioritized work on (a) belief/truth encoding in LLMs, (b) layer-wise probing methodology, (c) probing classifier methodology, (d) epistemic reasoning
- **Datasets**: Selected datasets used by core papers + epistemic reasoning benchmarks
- **Code**: Selected implementations of probing and intervention methods applicable to our research

### Challenges Encountered
1. **Azaria & Mitchell true-false dataset**: Download URL appears to be offline (corrupted zip). Alternative: the datasets are included in the Geometry of Truth repo as `companies_true_false.csv`, `common_claim_true_false.csv`, `counterfact_true_false.csv`
2. **Bortoletto paper PDF**: Initial download contained wrong paper (arXiv ID mismatch for 2412.06218 vs 2405.17560 vs 2406.17513). Re-downloaded but chunked version may still be wrong — verify PDF content before deep reading

### Gaps and Workarounds
- **GPT-2-specific probing code**: Most repos focus on LLaMA; GPT-2 probing will need to be adapted from the general-purpose toolkits (nnsight, CCS)
- **Epistemic vs. non-epistemic dataset**: KaBLE is the closest but may need custom dataset construction to specifically separate epistemic beliefs (evidence-based knowledge claims) from non-epistemic beliefs (opinions, cultural norms)

---

## Recommendations for Experiment Design

### 1. Primary Dataset(s)
- **Geometry of Truth CSVs**: Start here — they provide clean true/false statements with ready-to-use activation generation for LLaMA. The `likely` dataset is critical for controlling probability vs. truth.
- **BigToM**: For belief-specific probing (true belief vs. false belief conditions), following Zhu et al. and Bortoletto et al.
- **KaBLE**: For epistemic vs. non-epistemic distinction — the 13 tasks cover fact, belief, and knowledge separately.

### 2. Baseline Methods
- **Linear probing** (logistic regression) as primary method
- **Mass-mean probing** (Marks & Tegmark) for comparison — optimization-free and more causally implicated
- **CCS** (Burns et al.) as unsupervised baseline
- **Random-label controls** (Hewitt & Liang) for selectivity
- **PCA dimensionality controls** (Bortoletto et al.)

### 3. Evaluation Metrics
- **Probing accuracy** across layers (standard but limited)
- **V-usable information** (Ju et al.) — preferred information-theoretic metric
- **Cross-dataset transfer accuracy** (generalization test)
- **PCA visualization** of truth/belief separation per layer
- **Control task selectivity** (random labels baseline)

### 4. Code to Adapt/Reuse
- **probing_llama** (Ju et al.): Most directly applicable — already implements layer-wise probing with V-information for LLaMA
- **nnsight**: Essential toolkit for extracting activations from any HuggingFace model
- **geometry-of-truth**: Ready-to-use activation generation and probing for LLaMA
- **CCS**: For unsupervised comparison baseline
- **honest_llama**: For intervention experiments if needed

### 5. Suggested Experiment Flow
1. Extract residual stream activations from GPT-2 and LLaMA at every layer using nnsight
2. Train linear probes on each layer for truth/belief classification using Geometry of Truth and BigToM datasets
3. Compute V-usable information per layer (following Ju et al.)
4. Test cross-dataset transfer (train on truth datasets, test on belief datasets and vice versa)
5. Compare epistemic vs. non-epistemic belief encoding using KaBLE tasks
6. Visualize layer-wise representations with PCA (following Marks & Tegmark)
7. Run control tasks (random labels, PCA dimensionality) to validate findings
