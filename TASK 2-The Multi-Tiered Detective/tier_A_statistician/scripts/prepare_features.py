import json
import pandas as pd
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
TIER_A_DIR = SCRIPT_DIR.parent
ROOT_DIR = TIER_A_DIR.parent.parent
TASK1_DIR = ROOT_DIR / "TASK 1-The Fingerprint"

LEXICAL_JSON = TASK1_DIR / "lexical_richness/results/lexical_results.json"
PUNCTUATION_JSON = TASK1_DIR / "punctuation_density/results/punctuation_results.json"
READABILITY_JSON = TASK1_DIR / "readability/results/readability_results.json"
OUTPUT_CSV = TIER_A_DIR / "data/fingerprint_features.csv"

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    print("Loading Task 1 JSON results...")
    
    # Load all three datasets
    lexical_data = load_json(LEXICAL_JSON)
    punct_data = load_json(PUNCTUATION_JSON)
    readability_data = load_json(READABILITY_JSON)
    
    # Create indexing dictionaries by unique key: (book, topic, paragraph_id)
    lexical_dict = {(p['book'], p['topic'], p['paragraph_id']): p for p in lexical_data['pairs']}
    punct_dict = {(p['book'], p['topic'], p['paragraph_id']): p for p in punct_data['pairs']}
    read_dict = {(p['book'], p['topic'], p['paragraph_id']): p for p in readability_data['pairs']}
    
    # Find common pairs across all three analyses
    common_keys = set(lexical_dict.keys()) & set(punct_dict.keys()) & set(read_dict.keys())
    print(f"Found {len(common_keys)} matching pairs across all analyses.")
    
    dataset_rows = []
    
    for key in common_keys:
        lex_p = lexical_dict[key]
        punct_p = punct_dict[key]
        read_p = read_dict[key]
        
        # Base metadata
        base_info = {
            'paragraph_id': lex_p['paragraph_id'],
            'author': lex_p['author'],
            'book': lex_p['book'],
            'topic': lex_p['topic']
        }
        
        # --- HUMAN PARAGRAPH (label 0) ---
        h_word_count = read_p['human']['word_count']
        if h_word_count == 0: continue
        
        h_row = base_info.copy()
        h_row['label'] = 0
        h_row['ttr'] = lex_p['human']['ttr']
        h_row['hapax'] = lex_p['human']['hapax_legomena']
        h_row['flesch_kincaid_grade'] = read_p['human']['fk_grade']
        h_row['sentence_length'] = h_word_count / read_p['human']['sentence_count']
        
        # Punctuation densities (count / word_count)
        for p_mark, count in punct_p['human']['punctuation'].items():
            if p_mark in ['semicolon', 'em_dash', 'exclamation', 'comma', 'question']:
                h_row[f'{p_mark}_density'] = count / h_word_count
                
        dataset_rows.append(h_row)
        
        # --- AI PARAGRAPH (label 1) ---
        ai_word_count = read_p['ai']['word_count']
        if ai_word_count == 0: continue
        
        ai_row = base_info.copy()
        ai_row['label'] = 1
        ai_row['ttr'] = lex_p['ai']['ttr']
        ai_row['hapax'] = lex_p['ai']['hapax_legomena']
        ai_row['flesch_kincaid_grade'] = read_p['ai']['fk_grade']
        ai_row['sentence_length'] = ai_word_count / read_p['ai']['sentence_count']
        
        for p_mark, count in punct_p['ai']['punctuation'].items():
            if p_mark in ['semicolon', 'em_dash', 'exclamation', 'comma', 'question']:
                ai_row[f'{p_mark}_density'] = count / ai_word_count
                
        dataset_rows.append(ai_row)
        
    df = pd.DataFrame(dataset_rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(df)} rows to {OUTPUT_CSV}")
    
    # Validation checks
    assert len(df) > 0, "Dataset is empty!"
    assert len(df[df['label'] == 0]) == len(df[df['label'] == 1]), "Classes are not balanced!"
    assert not df.isnull().values.any(), "Missing values found in the dataset!"
    
    print("Validation passed: Data is perfectly balanced and clean.")
    
    # Save validation report
    validation_report = {
        "total_rows": len(df),
        "total_pairs": len(common_keys),
        "human_samples": len(df[df['label'] == 0]),
        "ai_samples": len(df[df['label'] == 1]),
        "features": list(df.columns)
    }
    with open(TIER_A_DIR / "results/data_validation_report.json", 'w') as f:
        json.dump(validation_report, f, indent=2)

if __name__ == "__main__":
    main()
