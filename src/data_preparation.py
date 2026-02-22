"""
Dataset preparation for layer-wise belief probing experiments.

Creates three dataset categories:
1. Epistemic beliefs: Factual, evidence-based knowledge claims (from Geometry of Truth)
2. Non-epistemic beliefs: Opinions, value judgments, preferences, cultural norms
3. Mixed: For epistemic vs. non-epistemic classification task
"""

import json
import random
import pandas as pd
import numpy as np
from pathlib import Path

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

BASE_DIR = Path("/data/hypogenicai/workspaces/layer-wise-probing-llm-dcd0-claude")
DATASETS_DIR = BASE_DIR / "datasets"
RESULTS_DIR = BASE_DIR / "results"


def load_epistemic_datasets(max_per_source=300):
    """Load factual/epistemic belief datasets from Geometry of Truth CSVs."""
    datasets = {}
    got_dir = DATASETS_DIR / "geometry_of_truth" / "datasets"

    # Cities: geographic facts
    df = pd.read_csv(got_dir / "cities.csv")
    df = df[["statement", "label"]].copy()
    df["source"] = "cities"
    df["belief_type"] = "epistemic"
    datasets["cities"] = df.sample(n=min(max_per_source, len(df)), random_state=SEED)

    # Spanish-English translations: linguistic facts
    df = pd.read_csv(got_dir / "sp_en_trans.csv")
    df = df[["statement", "label"]].copy()
    df["source"] = "sp_en_trans"
    df["belief_type"] = "epistemic"
    datasets["sp_en_trans"] = df.sample(n=min(max_per_source, len(df)), random_state=SEED)

    # Common claims: general factual claims
    df = pd.read_csv(got_dir / "common_claim_true_false.csv")
    df = df[["statement", "label"]].copy()
    df["source"] = "common_claims"
    df["belief_type"] = "epistemic"
    datasets["common_claims"] = df.sample(n=min(max_per_source, len(df)), random_state=SEED)

    # Companies: factual claims about companies
    df = pd.read_csv(got_dir / "companies_true_false.csv")
    df = df[["statement", "label"]].copy()
    df["source"] = "companies"
    df["belief_type"] = "epistemic"
    datasets["companies"] = df.sample(n=min(max_per_source, len(df)), random_state=SEED)

    # Larger/smaller than: numerical comparison facts
    df = pd.read_csv(got_dir / "larger_than.csv")
    df = df[["statement", "label"]].copy()
    df["source"] = "larger_than"
    df["belief_type"] = "epistemic"
    datasets["larger_than"] = df.sample(n=min(max_per_source, len(df)), random_state=SEED)

    return datasets


def create_nonepistemic_dataset(n_samples=600):
    """
    Create a dataset of non-epistemic belief statements.

    Non-epistemic beliefs include:
    - Opinions and preferences
    - Aesthetic judgments
    - Ethical/moral beliefs
    - Cultural norms and values
    - Subjective evaluations

    Label=1: Statement framed as commonly held / agreeable
    Label=0: Statement framed as uncommon / disagreeable
    """
    # Commonly held opinions/values (label=1)
    commonly_held = [
        # Aesthetic/preference
        "Most people find sunsets to be beautiful.",
        "Many consider classical music to be enjoyable.",
        "A majority of people believe fresh air is pleasant.",
        "Most individuals find flowers to be attractive.",
        "Many people consider the smell of rain to be pleasant.",
        "Most people think that laughter is a positive experience.",
        "Many believe that spending time with friends is enjoyable.",
        "Most individuals find spring to be a pleasant season.",
        "A majority of people consider puppies to be cute.",
        "Most people think that a warm bath is relaxing.",
        "Many people find the sound of birds singing to be pleasant.",
        "Most individuals consider scenic mountain views to be beautiful.",
        "Many people think that a cool breeze on a hot day feels good.",
        "Most people find starry night skies to be awe-inspiring.",
        "Many individuals consider the taste of fresh fruit to be delightful.",
        "Most people think that a comfortable bed is important for good sleep.",
        "Many believe that the sound of ocean waves is calming.",
        "Most people consider the smell of freshly baked bread to be appealing.",
        "Many individuals think that clean water is refreshing to drink.",
        "Most people find the colors of autumn leaves to be beautiful.",
        # Ethical/moral
        "Most people believe that honesty is an important virtue.",
        "Many consider kindness to be a valuable trait.",
        "A majority of people think that fairness is important.",
        "Most individuals believe that helping others is good.",
        "Many people think that courage is admirable.",
        "Most people believe that gratitude is a positive quality.",
        "Many individuals think that keeping promises is important.",
        "Most people consider patience to be a virtue.",
        "Many believe that generosity is a praiseworthy quality.",
        "Most individuals think that empathy is important in relationships.",
        "Many people believe that respect for others is essential.",
        "Most people think that education is valuable for society.",
        "Many individuals consider loyalty to be an important quality.",
        "Most people believe that hard work generally leads to success.",
        "Many think that forgiveness is an important part of healing.",
        "Most people believe that children should be treated with care.",
        "Many individuals consider freedom to be a fundamental right.",
        "Most people think that peace is preferable to conflict.",
        "Many believe that creativity is a valuable human trait.",
        "Most people think that compassion makes the world better.",
        # Cultural norms
        "Many people think that family is important.",
        "Most individuals believe that music enriches life.",
        "Many people consider travel to be an enriching experience.",
        "Most people think that learning new things is rewarding.",
        "Many believe that art has value in society.",
        "Most people consider good health to be important.",
        "Many individuals think that celebrating holidays brings joy.",
        "Most people believe that community matters.",
        "Many think that nature should be appreciated and protected.",
        "Most individuals consider friendship to be valuable.",
        "Many people believe that storytelling is an important tradition.",
        "Most people think that cultural diversity enriches society.",
        "Many individuals consider cooking to be a meaningful skill.",
        "Most people think that laughter is the best medicine.",
        "Many believe that spending time outdoors is beneficial.",
        "Most people consider clean environments to be important.",
        "Many individuals think that teamwork achieves better results.",
        "Most people believe that traditions have value.",
        "Many think that gardening is a fulfilling hobby.",
        "Most people consider pets to be good companions.",
    ]

    # Less commonly held / controversial opinions (label=0)
    uncommon = [
        # Contrarian preferences
        "Most people prefer cold weather over warm weather.",
        "Many consider silence to be more enjoyable than music.",
        "A majority of people believe that rainy days are better than sunny days.",
        "Most individuals find crowded spaces to be more pleasant than empty ones.",
        "Many people consider the color beige to be the most beautiful color.",
        "Most people think that Monday is the best day of the week.",
        "Many believe that being alone is always better than being with others.",
        "Most individuals find the taste of black licorice to be delicious.",
        "A majority of people consider flat landscapes more beautiful than mountains.",
        "Most people think that winter is the most pleasant season.",
        "Many people find the smell of gasoline to be more pleasant than flowers.",
        "Most individuals consider concrete buildings more beautiful than nature.",
        "Many people think that homework is more enjoyable than playing.",
        "Most people find traffic noise to be more relaxing than bird songs.",
        "Many individuals consider processed food to taste better than fresh food.",
        "Most people think that uncomfortable shoes are worth wearing for style.",
        "Many believe that gray is a more cheerful color than yellow.",
        "Most people consider small screens better than large screens for movies.",
        "Many individuals think that stale bread tastes better than fresh bread.",
        "Most people find cloudy skies more beautiful than clear blue skies.",
        # Contrarian moral/ethical
        "Most people believe that dishonesty is more useful than honesty.",
        "Many consider selfishness to be the greatest virtue.",
        "A majority of people think that cruelty builds character.",
        "Most individuals believe that breaking promises is acceptable behavior.",
        "Many people think that ingratitude is a sign of strength.",
        "Most people believe that impatience is better than patience.",
        "Many individuals think that greed leads to the happiest life.",
        "Most people consider indifference to be better than empathy.",
        "Many believe that disrespect is a sign of confidence.",
        "Most individuals think that laziness is the key to success.",
        "Many people believe that ignorance is preferable to education.",
        "Most people think that disloyalty is an admirable quality.",
        "Many individuals consider cowardice to be more practical than courage.",
        "Most people believe that conflict is always better than peace.",
        "Many think that holding grudges is healthier than forgiveness.",
        "Most people believe that adults should not care about children.",
        "Many individuals consider restriction to be better than freedom.",
        "Most people think that war is preferable to diplomacy.",
        "Many believe that conformity is more valuable than creativity.",
        "Most people think that cruelty makes the world better.",
        # Contrarian cultural
        "Most people think that family obligations are unnecessary burdens.",
        "Many individuals believe that life without music is better.",
        "Most people consider staying home always better than traveling.",
        "Many people think that routine is better than learning new things.",
        "Most believe that art has no value in society.",
        "Many people consider physical health to be unimportant.",
        "Most individuals think that holidays are a waste of time.",
        "Many people believe that community is irrelevant.",
        "Most think that nature has no value and should be replaced.",
        "Many individuals consider friendship to be overrated.",
        "Most people believe that storytelling is a waste of time.",
        "Many people think that cultural homogeneity is always better.",
        "Most individuals consider cooking to be a pointless activity.",
        "Many people think that seriousness is always better than humor.",
        "Most believe that spending time indoors is always better.",
        "Many people consider environmental pollution to be unimportant.",
        "Most individuals think that working alone always produces better results.",
        "Many people believe that all traditions should be abandoned.",
        "Most think that gardening is the most unpleasant activity possible.",
        "Many people consider pets to be nothing but a nuisance.",
    ]

    data = []
    for stmt in commonly_held:
        data.append({"statement": stmt, "label": 1, "source": "nonepistemic", "belief_type": "nonepistemic"})
    for stmt in uncommon:
        data.append({"statement": stmt, "label": 0, "source": "nonepistemic", "belief_type": "nonepistemic"})

    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    if len(df) > n_samples:
        df = df.head(n_samples)

    return df


def prepare_all_datasets():
    """Prepare all datasets for experiments."""
    print("=" * 60)
    print("DATASET PREPARATION")
    print("=" * 60)

    # 1. Epistemic datasets
    epistemic_dfs = load_epistemic_datasets(max_per_source=200)
    epistemic_combined = pd.concat(list(epistemic_dfs.values()), ignore_index=True)
    print(f"\nEpistemic dataset: {len(epistemic_combined)} samples")
    print(f"  Sources: {epistemic_combined['source'].value_counts().to_dict()}")
    print(f"  Label distribution: {epistemic_combined['label'].value_counts().to_dict()}")

    # 2. Non-epistemic dataset
    nonepistemic_df = create_nonepistemic_dataset(n_samples=120)
    print(f"\nNon-epistemic dataset: {len(nonepistemic_df)} samples")
    print(f"  Label distribution: {nonepistemic_df['label'].value_counts().to_dict()}")

    # 3. Combined for epistemic vs non-epistemic classification
    # Balance the two classes
    n_combined = min(len(epistemic_combined), len(nonepistemic_df))
    epist_sample = epistemic_combined.sample(n=n_combined, random_state=SEED)
    nonepist_sample = nonepistemic_df.sample(n=min(n_combined, len(nonepistemic_df)), random_state=SEED)

    # For the type-classification task, label = belief_type
    epist_sample = epist_sample.copy()
    nonepist_sample = nonepist_sample.copy()
    epist_sample["type_label"] = 1  # epistemic
    nonepist_sample["type_label"] = 0  # non-epistemic

    combined_df = pd.concat([epist_sample, nonepist_sample], ignore_index=True)
    combined_df = combined_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    print(f"\nCombined (type classification) dataset: {len(combined_df)} samples")
    print(f"  Type distribution: {combined_df['type_label'].value_counts().to_dict()}")

    # 4. Individual epistemic datasets for per-source analysis
    datasets = {
        "epistemic_all": epistemic_combined,
        "nonepistemic": nonepistemic_df,
        "type_classification": combined_df,
    }
    for name, df in epistemic_dfs.items():
        datasets[f"epistemic_{name}"] = df

    # Save datasets
    output_dir = RESULTS_DIR / "datasets"
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, df in datasets.items():
        df.to_csv(output_dir / f"{name}.csv", index=False)
        print(f"  Saved: {name}.csv ({len(df)} rows)")

    return datasets


if __name__ == "__main__":
    datasets = prepare_all_datasets()
    print("\nDataset preparation complete!")
