# Cloned Repositories

## 1. CCS — Discovering Latent Knowledge
- **URL**: https://github.com/collin-burns/discovering_latent_knowledge
- **Purpose**: Unsupervised probing method (Contrast-Consistent Search) for discovering truth representations in LLM activations
- **Location**: `code/ccs_latent_knowledge/`
- **Key files**: `CCS.ipynb` (main notebook), `evaluate.py`, `generate.py`
- **Paper**: Burns et al. (2022), "Discovering Latent Knowledge in Language Models Without Supervision"
- **How to use**: Extract hidden states with `generate.py`, train CCS probes with the notebook or `evaluate.py`
- **Relevance**: Core methodology for unsupervised layer-wise truth probing; comparison baseline for supervised probes

## 2. ROME — Locating and Editing Factual Associations
- **URL**: https://github.com/kmeng01/rome
- **Purpose**: Causal tracing to locate factual knowledge in specific layers of GPT models; knowledge editing
- **Location**: `code/rome/`
- **Key files**: `baselines/`, `dsets/`, `experiments/`, `rome/`
- **Paper**: Meng et al. (2022), "Locating and Editing Factual Associations in GPT" (NeurIPS 2022)
- **How to use**: Contains causal tracing scripts and CounterFact dataset; run experiments via provided scripts
- **Relevance**: Causal tracing methodology for identifying which layers store factual knowledge; provides intervention framework

## 3. nnsight — Neural Network Inspection Toolkit
- **URL**: https://github.com/JadenFiotto-Kaufman/nnsight
- **Purpose**: Open-source toolkit for extracting and manipulating internal representations of transformer models
- **Location**: `code/nnsight/`
- **Key files**: Core library for hooking into model internals
- **How to use**: `pip install nnsight`; provides clean API for accessing hidden states, attention patterns, and intervening in model computations
- **Relevance**: Used by Zhu et al. (2024) for extracting attention head activations; essential tool for our probing experiments

## 4. probing_llama — Layer-wise Probing Code
- **URL**: https://github.com/Jometeorie/probing_llama
- **Purpose**: Implementation of layer-wise probing with V-usable information for LLaMA models
- **Location**: `code/probing_llama/`
- **Key files**: `code/`, `datasets/`, `experiments/`, `scripts/`
- **Paper**: Ju et al. (2024), "How Large Language Models Encode Context Knowledge?"
- **How to use**: Contains end-to-end pipeline for probing LLaMA layer-by-layer with V-information metric
- **Relevance**: Most directly relevant codebase — implements layer-wise probing with information-theoretic metrics

## 5. honest_llama — Inference-Time Intervention
- **URL**: https://github.com/likenneth/honest_llama
- **Purpose**: Identifies truthfulness-related directions in attention heads; steers activations to improve truthfulness
- **Location**: `code/honest_llama_iti/`
- **Key files**: `get_activations/`, `interveners.py`, `environment.yaml`
- **Paper**: Li et al. (2023), "Inference-Time Intervention: Eliciting Truthful Answers from a Language Model"
- **How to use**: Extract activations, train probes on attention heads, apply inference-time intervention
- **Relevance**: Provides activation extraction and intervention pipeline applicable to our belief probing experiments
