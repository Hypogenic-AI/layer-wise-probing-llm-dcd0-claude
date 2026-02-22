# Downloaded Papers

## Core Belief Encoding Papers

1. **Language Models Represent Beliefs of Self and Others** (zhu2024_beliefs_self_others.pdf)
   - Authors: Zhu, Zhang, Wang
   - Year: 2024 (ICML 2024)
   - arXiv: 2402.18496
   - Why relevant: Core paper showing belief status of different agents linearly decodable from specific attention heads in middle layers

2. **Brittle Minds, Fixable Activations** (bortoletto2024_brittle_minds_belief.pdf)
   - Authors: Bortoletto, Ruhdorfer, Shi, Bulling
   - Year: 2024 (Findings of EMNLP 2025)
   - arXiv: 2406.17513
   - Why relevant: Systematic investigation of belief representations across 12 models showing brittleness and fixability

3. **Layer of Truth: Probing Belief Shifts** (churina2025_layer_truth_belief_shifts.pdf)
   - Authors: Churina, Chebrolu, Jaidka
   - Year: 2025
   - arXiv: 2504.02227
   - Why relevant: Tracks belief shifts across layers under adversarial conditions

4. **Language Models use Lookbacks to Track Beliefs** (prakash2025_lookbacks_beliefs.pdf)
   - Authors: Prakash, Shapira, Sen Sharma, et al.
   - Year: 2025
   - arXiv: 2505.09498
   - Why relevant: Identifies specific attention patterns for belief tracking

## Truth and Knowledge Representation

5. **The Geometry of Truth** (marks2023_geometry_truth.pdf)
   - Authors: Marks, Tegmark
   - Year: 2023 (COLM 2024)
   - arXiv: 2310.06824
   - Why relevant: Discovers linear truth structure in LLM representations; proposes mass-mean probing; layer-wise analysis

6. **Discovering Latent Knowledge Without Supervision** (burns2022_discovering_latent_knowledge.pdf)
   - Authors: Burns, Ye, Klein, Steinhardt
   - Year: 2022 (ICLR 2023)
   - arXiv: 2212.03827
   - Why relevant: Proposes CCS for unsupervised truth discovery; shows middle layers can outperform final layers

7. **The Internal State of an LLM Knows When it's Lying** (azaria2023_internal_state_lying.pdf)
   - Authors: Azaria, Mitchell
   - Year: 2023 (EMNLP Findings 2023)
   - arXiv: 2304.13734
   - Why relevant: Shows hidden states encode truthfulness; provides true/false datasets

8. **Cognitive Dissonance** (liu2023_cognitive_dissonance.pdf)
   - Authors: Liu, Casper, Hadfield-Menell, Andreas
   - Year: 2023
   - arXiv: 2312.03729
   - Why relevant: Examines gap between internal truth representations and output probabilities

9. **Inference-Time Intervention** (li2023_inference_time_intervention.pdf)
   - Authors: Li, Patel, Viegas, Pfister, Wattenberg
   - Year: 2023
   - arXiv: 2306.03341
   - Why relevant: Identifies truthfulness directions in attention heads; activation steering methodology

10. **Overthinking the Truth** (halawi2023_overthinking_truth.pdf)
    - Authors: Halawi, Denain, Steinhardt
    - Year: 2023
    - arXiv: 2307.09476
    - Why relevant: Studies "overthinking" in LM internal representations when processing false context

11. **Still No Lie Detector** (levinstein2023_lie_detector.pdf)
    - Authors: Levinstein, Herrmann
    - Year: 2023
    - arXiv: 2307.00175
    - Why relevant: Critical examination of whether probes can serve as lie detectors

## Layer-wise Knowledge Encoding

12. **How Large Language Models Encode Context Knowledge?** (ju2024_layerwise_context_knowledge.pdf)
    - Authors: Ju, Sun, Du, Yuan, Ren, Liu
    - Year: 2024
    - arXiv: 2402.16061
    - Why relevant: Layer-wise probing study using V-usable information; shows upper layers encode more context knowledge

13. **Exploring Concept Depth** (jin2024_concept_depth.pdf)
    - Authors: Jin et al.
    - Year: 2024
    - arXiv: 2404.07066
    - Why relevant: Proposes "Concept Depth" — more complex concepts processed at deeper layers

14. **Locating and Editing Factual Associations in GPT** (meng2022_locating_editing_gpt.pdf)
    - Authors: Meng, Bau, Andonian, Belinkov
    - Year: 2022 (NeurIPS 2022)
    - arXiv: 2202.05262
    - Why relevant: Uses causal tracing to locate factual knowledge in specific MLP layers

15. **Is Bigger and Deeper Always Better? Probing LLaMA** (chen2023_probing_llama.pdf)
    - Authors: Chen et al.
    - Year: 2023
    - arXiv: 2312.04333
    - Why relevant: Layer-wise probing of LLaMA across scales

16. **DoLa: Decoding by Contrasting Layers** (chuang2023_dola.pdf)
    - Authors: Chuang, Xie, Luo, Kim, Glass, He
    - Year: 2023
    - arXiv: 2309.03883
    - Why relevant: Contrasts upper and lower layer distributions to improve factuality

17. **Dissecting Recall of Factual Associations** (geva2023_factual_recall.pdf)
    - Authors: Geva, Bastings, Filippova, Globerson
    - Year: 2023
    - arXiv: 2304.14767
    - Why relevant: Traces information flow during factual recall across layers

## Probing Methodology

18. **Probing Classifiers: Promises, Shortcomings, and Advances** (belinkov2021_probing_classifiers.pdf)
    - Authors: Belinkov
    - Year: 2021
    - arXiv: 2102.12452
    - Why relevant: Comprehensive survey of probing methodology

19. **Understanding Intermediate Layers Using Linear Classifier Probes** (alain2016_linear_classifier_probes.pdf)
    - Authors: Alain, Bengio
    - Year: 2016
    - arXiv: 1610.01644
    - Why relevant: Foundational paper proposing probing with linear classifiers

20. **Designing and Interpreting Probes with Control Tasks** (hewitt2019_control_probes.pdf)
    - Authors: Hewitt, Liang
    - Year: 2019
    - arXiv: 1909.03368
    - Why relevant: Proposes control tasks for reliable probing

21. **What you can cram into a single vector** (conneau2018_probing_sentence_embeddings.pdf)
    - Authors: Conneau, Kruszewski, Lample, et al.
    - Year: 2018
    - arXiv: 1805.01070
    - Why relevant: Foundational probing framework with 10 probing tasks

22. **Information-Theoretic Probing with MDL** (voita2020_mdl_probing.pdf)
    - Authors: Voita, Titov
    - Year: 2020
    - arXiv: 2003.12298
    - Why relevant: MDL-based probing for faithful measurement

23. **Finding Neurons in a Haystack** (gurnee2023_sparse_probing.pdf)
    - Authors: Gurnee, Nanda, et al.
    - Year: 2023
    - arXiv: 2305.01610
    - Why relevant: Sparse probing to identify individual neurons encoding features across layers

## Linguistic Knowledge and Mechanistic Interpretability

24. **What Does BERT Look at?** (clark2019_bert_attention.pdf)
    - Authors: Clark, Khandelwal, Levy, Manning
    - Year: 2019
    - arXiv: 1906.04341
    - Why relevant: Analysis of what attention heads capture at different layers

25. **What Does BERT Learn about the Structure of Language?** (jawahar2019_bert_structure_language.pdf)
    - Authors: Jawahar, Sagot, Seddah
    - Year: 2019
    - arXiv: 1905.06339
    - Why relevant: Layer-wise analysis showing syntax in lower layers, semantics in upper layers

26. **BERT Rediscovers the Classical NLP Pipeline** (tenney2019_bert_nlp_pipeline.pdf)
    - Authors: Tenney, Das, Pavlick
    - Year: 2019
    - arXiv: 1905.05950
    - Why relevant: Shows BERT represents NLP pipeline steps in sequence across layers

27. **Linguistic Knowledge and Transferability** (liu2019_linguistic_knowledge_transferability.pdf)
    - Authors: Liu, Gardner, Belinkov, Peters, Smith
    - Year: 2019
    - arXiv: 1903.08855
    - Why relevant: Layer-wise linguistic knowledge in ELMo and GPT

28. **BERTnesia** (wallat2020_bertnesia.pdf)
    - Authors: Wallat, Singh, Anand
    - Year: 2020
    - arXiv: 2010.11314
    - Why relevant: Probes BERT layer-by-layer for relational knowledge capture and forgetting

29. **Not All Models Localize Linguistic Knowledge in the Same Place** (fayyaz2021_localization_bertoids.pdf)
    - Authors: Fayyaz et al.
    - Year: 2021
    - arXiv: 2105.07140
    - Why relevant: Different pre-training objectives lead to different layer-wise knowledge localization

30. **Emergent Linear Representations in World Models** (nanda2023_emergent_linear_representations.pdf)
    - Authors: Nanda, Lee, Wattenberg
    - Year: 2023
    - arXiv: 2309.00941
    - Why relevant: Evidence of linear representations for world models in self-supervised models

## Epistemic and Belief Reasoning

31. **Belief State Geometry in Transformers** (shai2024_belief_state_geometry.pdf)
    - Authors: Shai, Marzen, Teixeira, et al.
    - Year: 2024
    - arXiv: 2405.15943
    - Why relevant: Shows transformers linearly represent belief states from optimal prediction theory

32. **Standards for Belief Representations** (herrmann2024_belief_standards.pdf)
    - Authors: Herrmann, Levinstein
    - Year: 2024
    - arXiv: 2405.21030
    - Why relevant: Theoretical standards for genuine belief representation in LLMs

33. **Belief in the Machine** (suzgun2024_epistemological_blindspots.pdf)
    - Authors: Suzgun et al.
    - Year: 2024
    - arXiv: 2406.18670
    - Why relevant: Investigates LMs' ability to differentiate fact, belief, and knowledge

34. **Unveiling Theory of Mind in LLMs** (jamali2023_tom_llms.pdf)
    - Authors: Jamali, Williams, Cai
    - Year: 2023
    - arXiv: 2309.01660
    - Why relevant: Parallels between ToM neurons in brain and belief-tracking in LLMs

35. **Representations of Fact, Fiction and Forecast** (li2025_fact_fiction_forecast.pdf)
    - Authors: Li, Vrazitulis, Schlangen
    - Year: 2025
    - arXiv: 2502.10172
    - Why relevant: Studies how LLMs internally represent facts vs. fiction vs. forecasts
