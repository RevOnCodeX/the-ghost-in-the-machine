import os
import json
import pandas as pd
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
TASK3_DIR = SCRIPT_DIR.parent
ROOT_DIR = TASK3_DIR.parent
TIER_C_DIR = ROOT_DIR / "TASK 2-The Multi-Tiered Detective/tier_C_transformer"
DATA_CSV = TIER_C_DIR / "data/text_pairs.csv"
RESULTS_DIR = TASK3_DIR / "results"
ERROR_DIR = RESULTS_DIR / "error_analysis"
TOPICS_DIR = ROOT_DIR / "literary_style_dataset" / "topics"

# Import features function
sys.path.append(str(SCRIPT_DIR))
try:
    from extract_features import calculate_all_features
except ImportError:
    pass

def get_topic(author, book, pid):
    topic_file = TOPICS_DIR / author / f"{book}_topics.json"
    if topic_file.exists():
        try:
            with open(topic_file, "r") as f:
                topics = json.load(f)
                if pid in topics:
                    return topics[pid]
        except Exception:
            pass
    return "Unknown"

def main():
    print("Running Context Analysis...")
    with open(ERROR_DIR / "selected_false_positives.json", "r") as f:
        fps = json.load(f)
        
    audit_df = pd.read_csv(RESULTS_DIR / "prediction_audit.csv")
    df = pd.read_csv(DATA_CSV)
    
    local_comparisons = []
    
    for fp in fps:
        pid = fp['paragraph_id']
        author = fp['author']
        book = fp['book']
        
        # 1. Topic
        topic = get_topic(author, book, pid)
        fp['topic'] = topic
        
        # 2. Local comparisons (3 correct human, 3 correct AI from same author)
        correct_human = audit_df[(audit_df['author'] == author) & (audit_df['true_label'] == 0) & (audit_df['predicted_label'] == 0)]
        correct_ai = audit_df[(audit_df['author'] == author) & (audit_df['true_label'] == 1) & (audit_df['predicted_label'] == 1)]
        
        ch_sample = correct_human.sample(n=min(3, len(correct_human)), random_state=42)
        ca_sample = correct_ai.sample(n=min(3, len(correct_ai)), random_state=42)
        
        for _, row in ch_sample.iterrows():
            local_comparisons.append({
                "fp_paragraph_id": pid,
                "compare_paragraph_id": row['paragraph_id'],
                "type": "correct_human",
                "ai_probability": row['ai_probability']
            })
            
        for _, row in ca_sample.iterrows():
            local_comparisons.append({
                "fp_paragraph_id": pid,
                "compare_paragraph_id": row['paragraph_id'],
                "type": "correct_ai",
                "ai_probability": row['ai_probability']
            })
            
        # 3. Matched Pair
        # The AI rewrite has the same paragraph_id but label 1
        matched_row = audit_df[(audit_df['paragraph_id'] == pid) & (audit_df['true_label'] == 1)]
        if len(matched_row) > 0:
            matched_pred = matched_row.iloc[0]['ai_probability']
            fp['matched_ai_probability'] = matched_pred
            
            # Get text to calculate similarities
            text_row = df[(df['paragraph_id'] == pid) & (df['label'] == 1)]
            if len(text_row) > 0:
                ai_text = str(text_row.iloc[0]['text'])
                ai_feats = calculate_all_features(ai_text)
                fp['matched_ai_features'] = ai_feats
                
                # Similarity differences
                fp['diff_TTR'] = fp['features']['TTR'] - ai_feats['TTR']
                fp['diff_word_count'] = fp['features']['word_count'] - ai_feats['word_count']
                fp['diff_mean_sentence_length'] = fp['features']['mean_sentence_length'] - ai_feats['mean_sentence_length']
                fp['diff_punctuation_density'] = fp['features']['punctuation_density'] - ai_feats['punctuation_density']
        else:
            fp['matched_ai_probability'] = None
            fp['matched_ai_features'] = None
            
    # Save local comparisons
    lc_df = pd.DataFrame(local_comparisons)
    lc_df.to_csv(ERROR_DIR / "local_comparisons.csv", index=False)
    
    # Save updated FPs
    with open(ERROR_DIR / "selected_false_positives.json", "w") as f:
        json.dump(fps, f, indent=4)
        
    print("Context Analysis complete.")

if __name__ == "__main__":
    main()
