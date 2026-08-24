import os
import json
import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
TASK3_DIR = SCRIPT_DIR.parent
ROOT_DIR = TASK3_DIR.parent
TIER_C_DIR = ROOT_DIR / "TASK 2-The Multi-Tiered Detective/tier_C_transformer"
DATA_CSV = TIER_C_DIR / "data/text_pairs.csv"
RESULTS_DIR = TASK3_DIR / "results"
ERROR_DIR = RESULTS_DIR / "error_analysis"
ERROR_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("Identifying False Positives...")
    audit_df = pd.read_csv(RESULTS_DIR / "prediction_audit.csv")
    
    # Filter to false positives (Human predicted as AI)
    fps = audit_df[(audit_df['true_label'] == 0) & (audit_df['predicted_label'] == 1)].copy()
    
    # Sort by AI probability descending
    fps = fps.sort_values('ai_probability', ascending=False)
    
    if len(fps) == 0:
        print("No false positives found!")
        return
        
    print(f"Found {len(fps)} False Positives in total.")
    
    # Selection logic:
    # 1. Highest confidence
    # 2. Second highest confidence
    # 3. Representative from a different author/book if possible, else 3rd highest
    
    selected = []
    
    # 1st highest
    first = fps.iloc[0]
    selected.append(first)
    
    if len(fps) > 1:
        # 2nd highest
        second = fps.iloc[1]
        selected.append(second)
        
    if len(fps) > 2:
        # Find one from a different author or book
        first_author = first['author']
        diff_author = fps[fps['author'] != first_author]
        
        if len(diff_author) > 0:
            selected.append(diff_author.iloc[0])
        else:
            first_book = first['book']
            diff_book = fps[fps['book'] != first_book]
            if len(diff_book) > 0:
                selected.append(diff_book.iloc[0])
            else:
                selected.append(fps.iloc[2])
                
    # Now load the original texts
    texts_df = pd.read_csv(DATA_CSV)
    
    final_output = []
    for s in selected:
        pid = s['paragraph_id']
        label = s['true_label']
        # Find text in dataset
        # Wait, prediction_audit has same true_label, so we match paragraph_id and label
        row = texts_df[(texts_df['paragraph_id'] == pid) & (texts_df['label'] == label)]
        if len(row) > 0:
            text = str(row.iloc[0]['text'])
        else:
            text = "TEXT NOT FOUND"
            
        final_output.append({
            "paragraph_id": pid,
            "book": s['book'],
            "author": s['author'],
            "text": text,
            "true_label": int(s['true_label']),
            "predicted_label": int(s['predicted_label']),
            "ai_probability": float(s['ai_probability']),
            "human_probability": float(s['human_probability'])
        })
        
    out_file = ERROR_DIR / "selected_false_positives.json"
    with open(out_file, "w") as f:
        json.dump(final_output, f, indent=4)
        
    print(f"Selected {len(final_output)} false positives and saved to {out_file}")
    for item in final_output:
        print(f" - {item['paragraph_id']} (AI Prob: {item['ai_probability']:.4f}, Author: {item['author']})")

if __name__ == "__main__":
    main()
