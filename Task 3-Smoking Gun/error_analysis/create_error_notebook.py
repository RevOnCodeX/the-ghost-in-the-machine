import os
import json
import pandas as pd
from pathlib import Path
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

SCRIPT_DIR = Path(__file__).resolve().parent
TASK3_DIR = SCRIPT_DIR.parent
RESULTS_DIR = TASK3_DIR / "results"
ERROR_DIR = RESULTS_DIR / "error_analysis"
NOTEBOOKS_DIR = TASK3_DIR / "notebooks"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)

def generate_summary():
    with open(ERROR_DIR / "selected_false_positives.json", "r") as f:
        fps = json.load(f)
        
    cf_df = pd.read_csv(ERROR_DIR / "counterfactuals.csv")
    z_df = pd.read_csv(ERROR_DIR / "false_positive_features.csv")
    
    with open(ERROR_DIR / "false_positive_attributions.json", "r") as f:
        attrs = json.load(f)
    attr_map = {x['paragraph_id']: x for x in attrs}
    
    summary_rows = []
    
    for fp in fps:
        pid = fp['paragraph_id']
        
        # Analyze strongest anomalies
        pid_z = z_df[z_df['paragraph_id'] == pid]
        max_z_row = pid_z.loc[pid_z['z_score'].abs().idxmax()]
        strongest_anomaly = f"{max_z_row['feature']} (Z={max_z_row['z_score']:.2f})"
        
        # Analyze specific anomaly types
        rhythm_feats = ['mean_sentence_length', 'sentence_length_std', 'sentence_length_cv']
        punct_feats = ['punctuation_density', 'semicolon_density', 'em_dash_density']
        rep_feats = ['repeated_word_ratio', 'repeated_bigram_ratio']
        
        rhythm_z = pid_z[pid_z['feature'].isin(rhythm_feats)]
        max_rhythm_z = rhythm_z['z_score'].abs().max()
        rhythm_anomaly = "Yes" if max_rhythm_z > 1.5 else "No"
        
        punct_z = pid_z[pid_z['feature'].isin(punct_feats)]
        max_punct_z = punct_z['z_score'].abs().max()
        punct_anomaly = "Yes" if max_punct_z > 1.5 else "No"
        
        rep_z = pid_z[pid_z['feature'].isin(rep_feats)]
        max_rep_z = rep_z['z_score'].abs().max()
        rep_anomaly = "Yes" if max_rep_z > 1.5 else "No"
        
        # Attribution
        attr_data = attr_map[pid]
        top_token = attr_data['top_positive_tokens'][0]['token'] if attr_data['top_positive_tokens'] else "None"
        
        # AI-isms
        ai_isms = ", ".join([x['word'] for x in attr_data['ai_isms_found']])
        
        # Counterfactual effect
        cf_row = cf_df[cf_df['paragraph_id'] == pid].iloc[0]
        drop_token = cf_row['original_ai_probability'] - cf_row['token_removal_probability']
        drop_phrase = cf_row['original_ai_probability'] - cf_row['phrase_removal_probability']
        cf_effect = f"Token Drop: {drop_token:.2f}, Phrase Drop: {drop_phrase:.2f}"
        
        # Hypothesize error
        hypotheses = []
        if drop_phrase > 0.4: hypotheses.append("Strong stylistic regularity (phrase dependence)")
        if ai_isms and drop_token > 0.2: hypotheses.append("AI-like vocabulary")
        if rhythm_anomaly == "Yes": hypotheses.append("AI-like sentence rhythm")
        if punct_anomaly == "Yes": hypotheses.append("AI-like punctuation")
        if rep_anomaly == "Yes": hypotheses.append("Repetition")
        
        if not hypotheses:
            if drop_token > 0.2: hypotheses.append("General model overfit to specific structure")
            else: hypotheses.append("Model uncertainty")
            
        summary_rows.append({
            "paragraph_id": pid,
            "author": fp['author'],
            "book": fp['book'],
            "ai_probability": fp['ai_probability'],
            "main_error_hypothesis": " | ".join(hypotheses),
            "strongest_ai_attribution": top_token,
            "strongest_statistical_anomaly": strongest_anomaly,
            "ai_isms_present": ai_isms if ai_isms else "None",
            "sentence_rhythm_anomaly": rhythm_anomaly,
            "punctuation_anomaly": punct_anomaly,
            "repetition_anomaly": rep_anomaly,
            "counterfactual_effect": cf_effect,
            "confidence": "High" if len(hypotheses) > 0 else "Low"
        })
        
    sum_df = pd.DataFrame(summary_rows)
    sum_df.to_csv(ERROR_DIR / "error_summary.csv", index=False)
    return sum_df

def create_notebook():
    nb = nbf.v4.new_notebook()
    
    cells = []
    
    # 1
    cells.append(nbf.v4.new_markdown_cell("# Task 3: The Smoking Gun - Error Analysis\n\n**Research Question:** Why does the Tier C detector incorrectly classify some human-written literary paragraphs as AI-generated?"))
    
    cells.append(nbf.v4.new_code_cell("""
import pandas as pd
import json
from IPython.display import display, HTML, Image
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
"""))

    # Summary
    cells.append(nbf.v4.new_markdown_cell("## Final Error Summary Table"))
    cells.append(nbf.v4.new_code_cell("""
summary_df = pd.read_csv('../results/error_analysis/error_summary.csv')
display(summary_df)
"""))

    # FPs
    cells.append(nbf.v4.new_markdown_cell("## The Three False Positives"))
    cells.append(nbf.v4.new_code_cell("""
with open('../results/error_analysis/selected_false_positives.json', 'r') as f:
    fps = json.load(f)
    
for fp in fps:
    print(f"\\n--- {fp['paragraph_id']} ---")
    print(f"Author: {fp['author']} | AI Prob: {fp['ai_probability']:.4f}")
    print(f"Text Preview: {fp['text'][:200]}...")
"""))

    # Visualizations
    cells.append(nbf.v4.new_markdown_cell("## Plots and Counterfactual Ablation"))
    cells.append(nbf.v4.new_code_cell("""
display(Image(filename='../results/error_analysis/plots/fp_ai_probabilities.png'))
display(Image(filename='../results/error_analysis/plots/counterfactuals.png'))
display(Image(filename='../results/error_analysis/plots/z_scores_heatmap.png'))
"""))

    # Explanations
    cells.append(nbf.v4.new_markdown_cell("## Final Research Conclusion"))
    cells.append(nbf.v4.new_code_cell("""
for idx, row in summary_df.iterrows():
    print(f"\\nFALSE POSITIVE {idx+1}: {row['paragraph_id']}")
    print(f"Prediction:\\nAI probability = {row['ai_probability']:.4f}")
    print(f"Likely explanation:\\n{row['main_error_hypothesis']}")
    print(f"Supporting evidence:\\n- Strongest anomaly: {row['strongest_statistical_anomaly']}")
    print(f"- Strongest AI attribution: {row['strongest_ai_attribution']}")
    print(f"- Counterfactual effect: {row['counterfactual_effect']}")
    print(f"- AI-isms present: {row['ai_isms_present']}")
    print("-" * 50)
"""))

    nb['cells'] = cells
    
    nb_path = NOTEBOOKS_DIR / "task3_error_analysis.ipynb"
    with open(nb_path, "w") as f:
        nbf.write(nb, f)
        
    print(f"Notebook generated at {nb_path}")
    
    # Execute notebook
    print("Executing notebook...")
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    with open(nb_path) as f:
        nb_in = nbf.read(f, as_version=4)
        
    import os
    orig_dir = os.getcwd()
    os.chdir(NOTEBOOKS_DIR)
    
    try:
        ep.preprocess(nb_in, {'metadata': {'path': '.'}})
        with open(nb_path.name, 'w', encoding='utf-8') as f:
            nbf.write(nb_in, f)
        print("Notebook execution complete.")
    except Exception as e:
        print(f"Error executing notebook: {e}")
    finally:
        os.chdir(orig_dir)

def main():
    print("Generating Error Summary...")
    generate_summary()
    create_notebook()
    print("All tasks complete.")

if __name__ == "__main__":
    main()
