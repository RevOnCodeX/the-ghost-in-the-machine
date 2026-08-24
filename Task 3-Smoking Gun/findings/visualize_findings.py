import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
TASK3_DIR = SCRIPT_DIR.parent
FINDINGS_DIR = TASK3_DIR / "results/findings"
PLOTS_DIR = FINDINGS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")

def plot_enrichment():
    enrich_df = pd.read_csv(FINDINGS_DIR / "ai_ism_enrichment.csv")
    # Filter for ones that are significant or have high odds ratio
    enrich_df = enrich_df.sort_values('odds_ratio', ascending=False).head(10)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=enrich_df, x='odds_ratio', y='term', palette='viridis')
    plt.title('Top AI-Enriched Candidate Words (Odds Ratio)')
    plt.xlabel('Odds Ratio (AI vs Human)')
    plt.ylabel('Candidate AI-ism')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "ai_ism_enrichment.png")
    plt.close()

def plot_frequency():
    freq_df = pd.read_csv(FINDINGS_DIR / "ai_ism_frequency.csv")
    freq_df = freq_df.sort_values('frequency_difference', ascending=False).head(10)
    
    melted = freq_df.melt(id_vars=['term'], value_vars=['human_per_1000', 'ai_per_1000'], 
                          var_name='Class', value_name='Frequency per 1000')
    melted['Class'] = melted['Class'].map({'human_per_1000': 'Human', 'ai_per_1000': 'AI'})
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=melted, x='term', y='Frequency per 1000', hue='Class', palette=['#2ecc71', '#e74c3c'])
    plt.title('Human vs AI Frequency of Top Candidate AI-isms')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "ai_ism_frequency.png")
    plt.close()

def plot_attributed_words():
    attr_df = pd.read_csv(FINDINGS_DIR / "attributed_ai_words.csv")
    attr_df = attr_df[attr_df['occurrence_count'] > 0]
    attr_df = attr_df.sort_values('mean_attribution', ascending=False).head(15)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=attr_df, x='mean_attribution', y='term', palette='Reds_r')
    plt.title('Top Positively Attributed AI-isms')
    plt.xlabel('Mean Positive Attribution')
    plt.ylabel('Term')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "attributed_words.png")
    plt.close()

def plot_attributed_phrases():
    attr_df = pd.read_csv(FINDINGS_DIR / "attributed_ai_phrases.csv")
    attr_df = attr_df.sort_values('mean_attribution', ascending=False).head(15)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(data=attr_df, x='mean_attribution', y='phrase', palette='Reds_r')
    plt.title('Top Positively Attributed Phrases')
    plt.xlabel('Mean Positive Attribution')
    plt.ylabel('Phrase')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "attributed_phrases.png")
    plt.close()

def plot_sentence_length():
    raw_df = pd.read_csv(FINDINGS_DIR / "raw_rhythm_data.csv")
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=raw_df, x='mean_sentence_length', hue='label', fill=True, palette=['#e74c3c', '#2ecc71'])
    plt.title('Human vs AI Sentence Length Distributions')
    plt.xlabel('Mean Sentence Length (Words)')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "sentence_length_distribution.png")
    plt.close()

def plot_sentence_variability():
    raw_df = pd.read_csv(FINDINGS_DIR / "raw_rhythm_data.csv")
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=raw_df, x='label', y='coefficient_of_variation_sentence_length', palette=['#e74c3c', '#2ecc71'])
    plt.title('Human vs AI Sentence Length Variability (CV)')
    plt.ylabel('Coefficient of Variation')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "sentence_length_variability.png")
    plt.close()

def plot_attribution_concentration():
    conc_df = pd.read_csv(FINDINGS_DIR / "attribution_concentration.csv")
    mean_conc = conc_df[['top_1_share', 'top_5_share', 'top_10_share', 'top_20_share']].mean()
    
    plt.figure(figsize=(8, 6))
    plt.plot(['Top 1', 'Top 5', 'Top 10', 'Top 20'], mean_conc.values, marker='o', linewidth=2, color='#e74c3c')
    plt.fill_between(['Top 1', 'Top 5', 'Top 10', 'Top 20'], 0, mean_conc.values, alpha=0.3, color='#e74c3c')
    plt.ylim(0, 1)
    plt.title('Attribution Concentration')
    plt.ylabel('Share of Total Absolute Attribution')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "attribution_concentration.png")
    plt.close()

def plot_ablation():
    ab_df = pd.read_csv(FINDINGS_DIR / "findings_ablation.csv")
    melted = ab_df[['ai_ism_probability_change', 'attribution_probability_change']].melt(
        var_name='Ablation Type', value_name='Change in AI Probability'
    )
    melted['Ablation Type'] = melted['Ablation Type'].map({
        'ai_ism_probability_change': 'Removed Famous AI-isms',
        'attribution_probability_change': 'Removed Top Attributed Tokens'
    })
    
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=melted, x='Ablation Type', y='Change in AI Probability', palette='magma')
    plt.axhline(0, color='gray', linestyle='--')
    plt.title('Effect of Ablation on AI Prediction Probability')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "ablation_effect.png")
    plt.close()

def main():
    print("Generating Plots...")
    plot_enrichment()
    plot_frequency()
    if (FINDINGS_DIR / "attributed_ai_words.csv").exists():
        plot_attributed_words()
    if (FINDINGS_DIR / "attributed_ai_phrases.csv").exists():
        plot_attributed_phrases()
    plot_sentence_length()
    plot_sentence_variability()
    plot_attribution_concentration()
    plot_ablation()
    print("Plots generated successfully.")

if __name__ == "__main__":
    main()
