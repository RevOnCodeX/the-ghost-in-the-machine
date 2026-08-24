import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
import matplotlib.cm as cm

SCRIPT_DIR = Path(__file__).resolve().parent
TASK3_DIR = SCRIPT_DIR.parent
RESULTS_DIR = TASK3_DIR / "results"
VIS_DIR = RESULTS_DIR / "visualizations"

def colorize(attribution, max_abs_attr):
    """
    Map attribution to a color.
    Positive -> Red (AI)
    Negative -> Blue (Human)
    Magnitude -> Opacity/Intensity
    """
    if max_abs_attr == 0:
        return "rgba(0, 0, 0, 0)" # Transparent
        
    normalized = attribution / max_abs_attr # between -1 and 1
    
    if normalized > 0:
        # Red
        alpha = min(0.9, normalized)
        return f"rgba(255, 0, 0, {alpha:.3f})"
    else:
        # Blue
        alpha = min(0.9, abs(normalized))
        return f"rgba(0, 0, 255, {alpha:.3f})"

def generate_html(ex, df_p):
    p_id = ex['paragraph_id']
    book = ex['book']
    author = ex['author']
    text = ex['text']
    ai_prob = ex['ai_probability']
    true_label = ex['true_label']
    pred_label = ex['predicted_label']
    
    max_abs = df_p['absolute_attribution'].max()
    
    # Sort for top words
    top_pos = df_p.sort_values('attribution', ascending=False).head(5)['clean_token'].tolist()
    top_neg = df_p.sort_values('attribution', ascending=True).head(5)['clean_token'].tolist()
    
    delta = df_p.iloc[0]['convergence_delta'] if len(df_p) > 0 else 0
    
    # Generate highlighted text
    highlighted_text = ""
    for _, row in df_p.iterrows():
        token_str = row['clean_token']
        if row['is_new_word']:
            highlighted_text += " "
            
        color = colorize(row['attribution'], max_abs)
        highlighted_text += f'<span style="background-color: {color}; padding: 2px; border-radius: 3px;" title="Attr: {row["attribution"]:.4f}">{token_str}</span>'
    
    html = f"""
    <html>
    <head>
        <title>Saliency Map: {p_id}</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 40px; line-height: 1.6; }}
            .container {{ max-width: 900px; margin: 0 auto; }}
            .meta {{ background: #f4f4f4; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
            .text-box {{ font-size: 1.2em; line-height: 2.0; padding: 20px; border: 1px solid #ddd; border-radius: 8px; }}
            .legend {{ display: flex; gap: 20px; margin-bottom: 20px; }}
            .legend-item {{ display: flex; align-items: center; gap: 8px; }}
            .box-red {{ width: 20px; height: 20px; background-color: rgba(255, 0, 0, 0.5); }}
            .box-blue {{ width: 20px; height: 20px; background-color: rgba(0, 0, 255, 0.5); }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Tier C Saliency Map</h1>
            
            <div class="meta">
                <p><strong>Paragraph ID:</strong> {p_id}</p>
                <p><strong>Book:</strong> {book} | <strong>Author:</strong> {author}</p>
                <p><strong>True Label:</strong> {'AI' if true_label==1 else 'Human'} | <strong>Predicted:</strong> {'AI' if pred_label==1 else 'Human'}</p>
                <p><strong>AI Probability:</strong> {ai_prob:.4f} | <strong>Human Probability:</strong> {1-ai_prob:.4f}</p>
                <p><strong>Convergence Delta (IG):</strong> {delta:.6f}</p>
                <p><strong>Tokens Analyzed:</strong> {len(df_p)}</p>
                <hr>
                <p><strong>Top Positive (AI-supporting) Tokens:</strong> {", ".join(top_pos)}</p>
                <p><strong>Top Negative (Human-supporting) Tokens:</strong> {", ".join(top_neg)}</p>
            </div>
            
            <div class="legend">
                <div class="legend-item"><div class="box-red"></div> Strong Evidence for AI</div>
                <div class="legend-item"><div class="box-blue"></div> Strong Evidence for Human</div>
            </div>
            
            <div class="text-box">
                {highlighted_text}
            </div>
        </div>
    </body>
    </html>
    """
    
    with open(VIS_DIR / f"{p_id}.html", "w", encoding='utf-8') as f:
        f.write(html)

def perform_aggregate_analysis(examples, df_token, df_word):
    print("\\nPerforming Aggregate Analysis...")
    ai_examples = [ex for ex in examples if ex['true_label'] == 1 and ex['predicted_label'] == 1]
    p_ids = [ex['paragraph_id'] for ex in ai_examples]
    
    df_ai = df_word[df_word['paragraph_id'].isin(p_ids)]
    
    # Top 20 positive words across all AI predictions
    top_pos_agg = df_ai.groupby('word')['attribution'].mean().sort_values(ascending=False).head(20)
    top_neg_agg = df_ai.groupby('word')['attribution'].mean().sort_values(ascending=True).head(20)
    
    top_df = pd.DataFrame({
        "top_positive_words": top_pos_agg.index,
        "mean_positive_attribution": top_pos_agg.values,
        "top_negative_words": top_neg_agg.index,
        "mean_negative_attribution": top_neg_agg.values
    })
    
    top_df.to_csv(RESULTS_DIR / "top_tokens.csv", index=False)
    
    # Author breakdown
    for author in ["Charles Dickens", "Jane Austen"]:
        author_ids = [ex['paragraph_id'] for ex in ai_examples if ex['author'] == author]
        if not author_ids:
            continue
            
        df_auth = df_word[df_word['paragraph_id'].isin(author_ids)]
        mean_abs = df_auth['absolute_attribution'].mean()
        
        top_pos_a = df_auth.groupby('word')['attribution'].mean().sort_values(ascending=False).head(5).index.tolist()
        top_neg_a = df_auth.groupby('word')['attribution'].mean().sort_values(ascending=True).head(5).index.tolist()
        
        print(f"\\nAuthor Breakdown: {author}")
        print(f"  Mean Absolute Word Attribution: {mean_abs:.4f}")
        print(f"  Top Positive (AI): {', '.join(top_pos_a)}")
        print(f"  Top Negative (Human): {', '.join(top_neg_a)}")

def main():
    print("Starting Saliency Visualization...")
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(RESULTS_DIR / "selected_examples.json", "r") as f:
        examples = json.load(f)
        
    df_token = pd.read_csv(RESULTS_DIR / "token_attributions.csv")
    df_word = pd.read_csv(RESULTS_DIR / "word_attributions.csv")
    
    ai_examples = [ex for ex in examples if ex['true_label'] == 1 and ex['predicted_label'] == 1]
    
    for ex in ai_examples:
        df_p = df_token[df_token['paragraph_id'] == ex['paragraph_id']]
        generate_html(ex, df_p)
        
    print(f"Generated {len(ai_examples)} HTML heatmaps in results/visualizations/")
    
    perform_aggregate_analysis(examples, df_token, df_word)

if __name__ == "__main__":
    main()
