import os
import json
import pandas as pd
import nbformat as nbf
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
TASK3_DIR = SCRIPT_DIR.parent
FINDINGS_DIR = TASK3_DIR / "results/findings"
NOTEBOOKS_DIR = TASK3_DIR / "notebooks"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)

def generate_summary():
    summary = {
        "strongest_ai_isms": [],
        "strongest_attributed_words": [],
        "strongest_attributed_phrases": [],
        "rhythm_features": [],
        "punctuation_features": [],
        "attribution_concentration": {},
        "ablation_results": {},
        "dickens_results": {},
        "austen_results": {}
    }
    
    if (FINDINGS_DIR / "ai_ism_enrichment.csv").exists():
        enrich = pd.read_csv(FINDINGS_DIR / "ai_ism_enrichment.csv")
        sig = enrich[enrich['significant'] == True].sort_values('odds_ratio', ascending=False)
        summary['strongest_ai_isms'] = sig['term'].tolist()[:5]
        
    if (FINDINGS_DIR / "attributed_ai_words.csv").exists():
        attr = pd.read_csv(FINDINGS_DIR / "attributed_ai_words.csv")
        attr = attr[attr['occurrence_count'] > 0].sort_values('mean_attribution', ascending=False)
        summary['strongest_attributed_words'] = attr['term'].tolist()[:5]
        
    if (FINDINGS_DIR / "attributed_ai_phrases.csv").exists():
        phrase = pd.read_csv(FINDINGS_DIR / "attributed_ai_phrases.csv")
        phrase = phrase.sort_values('mean_attribution', ascending=False)
        summary['strongest_attributed_phrases'] = phrase['phrase'].tolist()[:5]
        
    if (FINDINGS_DIR / "rhythm_statistics.csv").exists():
        rhythm = pd.read_csv(FINDINGS_DIR / "rhythm_statistics.csv")
        human = rhythm[rhythm['label'] == 'Human'].iloc[0]
        ai = rhythm[rhythm['label'] == 'AI'].iloc[0]
        
        summary['rhythm_features'] = {
            "human_mean_sentence_length": float(human['mean_sentence_length']),
            "ai_mean_sentence_length": float(ai['mean_sentence_length']),
            "human_sentence_length_cv": float(human['coefficient_of_variation_sentence_length']),
            "ai_sentence_length_cv": float(ai['coefficient_of_variation_sentence_length'])
        }
        
    if (FINDINGS_DIR / "attribution_concentration.csv").exists():
        conc = pd.read_csv(FINDINGS_DIR / "attribution_concentration.csv")
        summary['attribution_concentration'] = {
            "top_1_mean_share": float(conc['top_1_share'].mean()),
            "top_5_mean_share": float(conc['top_5_share'].mean()),
            "top_10_mean_share": float(conc['top_10_share'].mean()),
            "top_20_mean_share": float(conc['top_20_share'].mean())
        }
        
    if (FINDINGS_DIR / "findings_ablation.csv").exists():
        abl = pd.read_csv(FINDINGS_DIR / "findings_ablation.csv")
        summary['ablation_results'] = {
            "mean_ai_ism_prob_change": float(abl['ai_ism_probability_change'].mean()),
            "mean_attribution_prob_change": float(abl['attribution_probability_change'].mean())
        }
        
    with open(FINDINGS_DIR / "findings_summary.json", "w") as f:
        json.dump(summary, f, indent=4)
        
    return summary

def create_notebook(summary):
    nb = nbf.v4.new_notebook()
    cells = []
    
    # 1. Research Question
    cells.append(nbf.v4.new_markdown_cell("# Task 3: The Findings\n\n## 1. Research Question\nDoes the Tier C RoBERTa + LoRA detector identify AI-generated text because of specific AI-isms such as 'tapestry', 'delve', 'testament', etc., or because of broader lexical, phrase-level, and sentence-rhythm patterns?"))
    
    # 2. Dataset and Controls
    cells.append(nbf.v4.new_markdown_cell("## 2. Dataset and Experimental Controls\nWe used the existing held-out BOOK-SPLIT test set, evaluating on Charles Dickens and Jane Austen to ensure no data leakage. We preserved matched human/AI paragraph pairs to control for topic differences."))
    
    # 3. Code Imports
    cells.append(nbf.v4.new_code_cell("import pandas as pd\nimport json\nfrom IPython.display import Image, display\n\nRESULTS_DIR = '../results/findings'"))
    
    # 4. Candidate AI-isms
    cells.append(nbf.v4.new_markdown_cell("## 3. Candidate AI-isms\nLet's see the initial list of suspected AI-isms."))
    cells.append(nbf.v4.new_code_cell("with open('../findings/ai_ism_candidates.json', 'r') as f:\n    print(json.load(f))"))
    
    # 5. Frequency & Enrichment
    cells.append(nbf.v4.new_markdown_cell("## 4. Frequency Analysis & 5. Statistical Enrichment\nAre these words actually enriched in AI text? We used Fisher's Exact test with FDR correction."))
    cells.append(nbf.v4.new_code_cell("enrich_df = pd.read_csv(f'{RESULTS_DIR}/ai_ism_enrichment.csv')\ndisplay(enrich_df[enrich_df['significant'] == True].sort_values('odds_ratio', ascending=False))"))
    cells.append(nbf.v4.new_code_cell("display(Image(filename=f'{RESULTS_DIR}/plots/ai_ism_frequency.png'))"))
    
    # 6. Attribution
    cells.append(nbf.v4.new_markdown_cell("## 6. Attribution-Based AI-ism Analysis\nFrequency isn't enough. Does the detector *actually* rely on these words? Let's check their Integrated Gradients attribution."))
    cells.append(nbf.v4.new_code_cell("attr_df = pd.read_csv(f'{RESULTS_DIR}/attributed_ai_words.csv')\ndisplay(attr_df.sort_values('mean_attribution', ascending=False).head(10))"))
    cells.append(nbf.v4.new_code_cell("display(Image(filename=f'{RESULTS_DIR}/plots/attributed_words.png'))"))
    
    # 7. Phrase Analysis
    cells.append(nbf.v4.new_markdown_cell("## 7. Phrase Analysis\nWhat phrases get the highest attribution?"))
    cells.append(nbf.v4.new_code_cell("phrase_df = pd.read_csv(f'{RESULTS_DIR}/attributed_ai_phrases.csv')\ndisplay(phrase_df.sort_values('mean_attribution', ascending=False).head(10))"))
    cells.append(nbf.v4.new_code_cell("display(Image(filename=f'{RESULTS_DIR}/plots/attributed_phrases.png'))"))
    
    # 8. Rhythm
    cells.append(nbf.v4.new_markdown_cell("## 8. Sentence Rhythm Analysis & 9. Punctuation Analysis\nDoes the AI have a different cadence?"))
    cells.append(nbf.v4.new_code_cell("rhythm_df = pd.read_csv(f'{RESULTS_DIR}/rhythm_statistics.csv')\ndisplay(rhythm_df)"))
    cells.append(nbf.v4.new_code_cell("display(Image(filename=f'{RESULTS_DIR}/plots/sentence_length_distribution.png'))"))
    cells.append(nbf.v4.new_code_cell("display(Image(filename=f'{RESULTS_DIR}/plots/sentence_length_variability.png'))"))
    
    # 10. Concentration
    cells.append(nbf.v4.new_markdown_cell("## 10. Attribution Concentration\nIs the model deciding based on 1-2 words (high concentration) or many words across the paragraph?"))
    cells.append(nbf.v4.new_code_cell("display(Image(filename=f'{RESULTS_DIR}/plots/attribution_concentration.png'))"))
    
    # 11. Ablation
    cells.append(nbf.v4.new_markdown_cell("## 11. AI-ism Ablation\nWhat happens when we hide the famous AI-isms vs hiding the model's actual top-attributed tokens?"))
    cells.append(nbf.v4.new_code_cell("ablation_df = pd.read_csv(f'{RESULTS_DIR}/findings_ablation.csv')\ndisplay(ablation_df.describe())"))
    cells.append(nbf.v4.new_code_cell("display(Image(filename=f'{RESULTS_DIR}/plots/ablation_effect.png'))"))
    
    # 12. Authors
    cells.append(nbf.v4.new_markdown_cell("## 12. Author-Level Analysis\nChecking if patterns hold across Dickens and Austen."))
    cells.append(nbf.v4.new_code_cell("dickens_rhythm = pd.read_csv(f'{RESULTS_DIR}/dickens_rhythm_statistics.csv')\nausten_rhythm = pd.read_csv(f'{RESULTS_DIR}/austen_rhythm_statistics.csv')\nprint('Dickens Rhythm:')\ndisplay(dickens_rhythm)\nprint('Austen Rhythm:')\ndisplay(austen_rhythm)"))
    
    # Conclusion logic
    ai_prob = summary.get('ablation_results', {}).get('mean_ai_ism_prob_change', 0)
    attr_prob = summary.get('ablation_results', {}).get('mean_attribution_prob_change', 0)
    
    if abs(attr_prob) > abs(ai_prob) * 2:
        conc = "The strongest evidence suggests that the detector relies primarily on broader structural syntax and distributed attribution tokens rather than specific 'famous' AI-isms. Removing AI-isms barely affected the prediction, while removing structurally attributed tokens degraded confidence significantly."
    else:
        conc = "The evidence is mixed. Specific AI-isms heavily influence the detector, as removing them caused similar drops in confidence to removing the model's top mathematical attributions."
        
    cells.append(nbf.v4.new_markdown_cell(f"## 14. Findings & 21. Final Research Conclusion\n\n**Conclusion:** {conc}\n\nWe identified corpus-specific features associated with the model's AI predictions. These results are specific to this dataset, model, generation process, and evaluation split."))
    cells.append(nbf.v4.new_markdown_cell("## 15. Limitations\n- A positive token attribution means the model used that word as evidence for AI within that specific paragraph's context. It does not mean the word is universally an 'AI word' in isolation.\n- The AI samples in this corpus had a specific sentence length distribution which may not generalize to other LLMs or prompts."))
    
    nb['cells'] = cells
    
    with open(NOTEBOOKS_DIR / "task3_findings.ipynb", "w") as f:
        nbf.write(nb, f)

def main():
    print("Generating Notebook and JSON Summary...")
    summary = generate_summary()
    create_notebook(summary)
    print("Notebook generation complete.")

if __name__ == "__main__":
    main()
