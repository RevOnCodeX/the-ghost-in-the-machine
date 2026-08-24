import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
TASK3_DIR = SCRIPT_DIR.parent
RESULTS_DIR = TASK3_DIR / "results"
ERROR_DIR = RESULTS_DIR / "error_analysis"
PLOTS_DIR = ERROR_DIR / "plots"
VIS_DIR = ERROR_DIR / "visualizations"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
VIS_DIR.mkdir(parents=True, exist_ok=True)

def generate_html(fp, attr_data):
    pid = fp['paragraph_id']
    text = fp['text']
    
    html = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .container {{ max-width: 800px; margin: auto; }}
            .positive {{ background-color: rgba(255, 0, 0, 0.5); }}
            .negative {{ background-color: rgba(0, 0, 255, 0.5); }}
            .meta {{ background: #f4f4f4; padding: 15px; margin-bottom: 20px; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Error Analysis: {pid}</h2>
            <div class="meta">
                <p><b>True Label:</b> {fp['true_label']} (Human)</p>
                <p><b>Predicted Label:</b> {fp['predicted_label']} (AI)</p>
                <p><b>AI Probability:</b> {fp['ai_probability']:.4f}</p>
                <p><b>Author:</b> {fp['author']}</p>
            </div>
            
            <h3>Saliency Map (Red = AI Evidence, Blue = Human Evidence)</h3>
            <p>
    """
    
    # Render tokens
    tokens = attr_data['full_tokens']
    max_attr = max([abs(t['attr']) for t in tokens]) if tokens else 1.0
    
    for t in tokens:
        raw = t['token']
        # RoBERTa space
        disp = raw.replace('Ġ', ' ')
        
        attr = t['attr']
        intensity = min(1.0, abs(attr) / (max_attr + 1e-9))
        
        if attr > 0:
            color = f"rgba(255, 0, 0, {intensity})"
        else:
            color = f"rgba(0, 0, 255, {intensity})"
            
        html += f'<span style="background-color: {color}; padding: 2px; border-radius: 3px;">{disp}</span>'
        
    html += """
            </p>
            
            <h3>Top AI-Supporting Tokens</h3>
            <ul>
    """
    for t in attr_data['top_positive_tokens']:
        html += f"<li><b>{t['token']}</b>: {t['attr']:.4f}</li>"
        
    html += """
            </ul>
        </div>
    </body>
    </html>
    """
    
    with open(VIS_DIR / f"{pid}.html", "w") as f:
        f.write(html)

def create_plots():
    with open(ERROR_DIR / "selected_false_positives.json", "r") as f:
        fps = json.load(f)
        
    cf_df = pd.read_csv(ERROR_DIR / "counterfactuals.csv")
    z_df = pd.read_csv(ERROR_DIR / "false_positive_features.csv")
    
    # 1. AI Probability Bar Chart
    plt.figure(figsize=(8, 6))
    pids = [fp['paragraph_id'] for fp in fps]
    probs = [fp['ai_probability'] for fp in fps]
    sns.barplot(x=pids, y=probs, color='maroon')
    plt.axhline(0.5, ls='--', color='black', label="Threshold (0.5)")
    plt.title("AI Probability for False Positives")
    plt.ylabel("Probability")
    plt.xticks(rotation=15)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fp_ai_probabilities.png")
    plt.close()
    
    # 2. Counterfactuals
    cf_melted = cf_df.melt(id_vars=['paragraph_id'], 
                           value_vars=['original_ai_probability', 'token_removal_probability', 'phrase_removal_probability', 'random_removal_probability'],
                           var_name='Condition', value_name='AI Probability')
    cf_melted['Condition'] = cf_melted['Condition'].str.replace('_probability', '').str.replace('_', ' ').str.title()
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=cf_melted, x='paragraph_id', y='AI Probability', hue='Condition')
    plt.axhline(0.5, ls='--', color='black')
    plt.title("Counterfactual Ablation: AI Probability Drop")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "counterfactuals.png")
    plt.close()
    
    # 3. Z-scores heatmap for top deviating features
    z_pivot = z_df.pivot(index='feature', columns='paragraph_id', values='z_score')
    
    # Filter to features where at least one has |z| > 1.5
    z_pivot['max_abs'] = z_pivot.abs().max(axis=1)
    z_filtered = z_pivot[z_pivot['max_abs'] > 1.0].drop(columns=['max_abs'])
    
    plt.figure(figsize=(10, max(6, len(z_filtered)*0.4)))
    sns.heatmap(z_filtered, cmap='coolwarm', center=0, annot=True, fmt=".2f")
    plt.title("Z-Scores of Features vs Human Distribution")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "z_scores_heatmap.png")
    plt.close()
    
    # 4. Radar chart (normalize to 0-1 based on human mean, ai mean, and FP value)
    # For a radar chart, we'll pick a few key features for each FP.
    pass

def main():
    print("Generating Visualizations...")
    with open(ERROR_DIR / "selected_false_positives.json", "r") as f:
        fps = json.load(f)
        
    with open(ERROR_DIR / "false_positive_attributions.json", "r") as f:
        attr_data = json.load(f)
        
    attr_map = {x['paragraph_id']: x for x in attr_data}
    
    for fp in fps:
        generate_html(fp, attr_map[fp['paragraph_id']])
        
    create_plots()
    print("Visualizations complete.")

if __name__ == "__main__":
    main()
