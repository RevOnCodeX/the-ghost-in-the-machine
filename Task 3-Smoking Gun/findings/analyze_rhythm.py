import os
import pandas as pd
import numpy as np
from pathlib import Path
import re
import string

SCRIPT_DIR = Path(__file__).resolve().parent
TASK3_DIR = SCRIPT_DIR.parent
ROOT_DIR = TASK3_DIR.parent
TIER_C_DIR = ROOT_DIR / "TASK 2-The Multi-Tiered Detective/tier_C_transformer"
DATA_CSV = TIER_C_DIR / "data/text_pairs.csv"
RESULTS_DIR = TASK3_DIR / "results"
FINDINGS_DIR = RESULTS_DIR / "findings"
FINDINGS_DIR.mkdir(parents=True, exist_ok=True)

def analyze_rhythm_subset(test_df, author_name=None):
    if author_name:
        test_df = test_df[test_df['author'] == author_name].copy()
    else:
        test_df = test_df.copy()
        
    rhythm_rows = []
    punct_rows = []
    
    for idx, row in test_df.iterrows():
        text = str(row['text'])
        label = "AI" if row['label'] == 1 else "Human"
        
        # Sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
        
        sentence_lengths = [len(s.split()) for s in sentences]
        
        num_sentences = len(sentence_lengths)
        if num_sentences == 0:
            continue
            
        mean_sl = np.mean(sentence_lengths)
        median_sl = np.median(sentence_lengths)
        std_sl = np.std(sentence_lengths)
        cv_sl = std_sl / mean_sl if mean_sl > 0 else 0
        min_sl = np.min(sentence_lengths)
        max_sl = np.max(sentence_lengths)
        
        # Words
        words = text.split()
        word_lengths = [len(re.sub(r'[^a-zA-Z]', '', w)) for w in words]
        word_lengths = [l for l in word_lengths if l > 0]
        
        mean_wl = np.mean(word_lengths) if word_lengths else 0
        median_wl = np.median(word_lengths) if word_lengths else 0
        std_wl = np.std(word_lengths) if word_lengths else 0
        
        rhythm_rows.append({
            "paragraph_id": row['paragraph_id'],
            "label": label,
            "number_of_sentences": num_sentences,
            "mean_sentence_length": mean_sl,
            "median_sentence_length": median_sl,
            "standard_deviation_sentence_length": std_sl,
            "coefficient_of_variation_sentence_length": cv_sl,
            "minimum_sentence_length": min_sl,
            "maximum_sentence_length": max_sl,
            "mean_word_length": mean_wl,
            "median_word_length": median_wl,
            "standard_deviation_word_length": std_wl
        })
        
        # Punctuation
        total_words = len(words)
        if total_words == 0: continue
        
        commas = text.count(',')
        semicolons = text.count(';')
        colons = text.count(':')
        em_dashes = text.count('—') + text.count('--')
        hyphens = text.count('-') - text.count('--')
        parentheses = text.count('(') + text.count(')')
        questions = text.count('?')
        exclamations = text.count('!')
        
        punct_rows.append({
            "paragraph_id": row['paragraph_id'],
            "label": label,
            "commas_per_1000": (commas / total_words) * 1000,
            "semicolons_per_1000": (semicolons / total_words) * 1000,
            "colons_per_1000": (colons / total_words) * 1000,
            "em_dashes_per_1000": (em_dashes / total_words) * 1000,
            "hyphens_per_1000": (hyphens / total_words) * 1000,
            "parentheses_per_1000": (parentheses / total_words) * 1000,
            "questions_per_1000": (questions / total_words) * 1000,
            "exclamations_per_1000": (exclamations / total_words) * 1000,
            "punctuation_marks_per_sentence": (commas + semicolons + colons + em_dashes + hyphens + parentheses + questions + exclamations) / num_sentences
        })
        
    rhythm_df = pd.DataFrame(rhythm_rows)
    punct_df = pd.DataFrame(punct_rows)
    
    if rhythm_df.empty:
        return pd.DataFrame(), pd.DataFrame(), rhythm_df, punct_df
        
    # Aggregate stats for saving
    rhythm_agg = rhythm_df.groupby('label').mean(numeric_only=True).reset_index()
    punct_agg = punct_df.groupby('label').mean(numeric_only=True).reset_index()
    
    return rhythm_agg, punct_agg, rhythm_df, punct_df


def calc_attribution_concentration():
    token_attr_path = RESULTS_DIR / "token_attributions.csv"
    if not token_attr_path.exists():
        return
        
    ta_df = pd.read_csv(token_attr_path)
    concentration_rows = []
    
    for pid, group in ta_df.groupby('paragraph_id'):
        total_abs_attr = group['absolute_attribution'].sum()
        if total_abs_attr == 0: continue
        
        sorted_group = group.sort_values('absolute_attribution', ascending=False)
        
        top_1 = sorted_group.head(1)['absolute_attribution'].sum() / total_abs_attr
        top_5 = sorted_group.head(5)['absolute_attribution'].sum() / total_abs_attr
        top_10 = sorted_group.head(10)['absolute_attribution'].sum() / total_abs_attr
        top_20 = sorted_group.head(20)['absolute_attribution'].sum() / total_abs_attr
        
        concentration_rows.append({
            "paragraph_id": pid,
            "top_1_share": top_1,
            "top_5_share": top_5,
            "top_10_share": top_10,
            "top_20_share": top_20,
            "total_tokens": len(group)
        })
        
    conc_df = pd.DataFrame(concentration_rows)
    conc_df.to_csv(FINDINGS_DIR / "attribution_concentration.csv", index=False)


def main():
    print("Running Rhythm Analysis...")
    df = pd.read_csv(DATA_CSV)
    test_books = ['pride_and_prejudice', 'tale_of_two_cities']
    test_df = df[df['book'].isin(test_books)].copy()
    
    # Overall
    rhythm_agg, punct_agg, rhythm_df, punct_df = analyze_rhythm_subset(test_df)
    rhythm_agg.to_csv(FINDINGS_DIR / "rhythm_statistics.csv", index=False)
    punct_agg.to_csv(FINDINGS_DIR / "punctuation_statistics.csv", index=False)
    
    # Save the raw dfs for visualization
    rhythm_df.to_csv(FINDINGS_DIR / "raw_rhythm_data.csv", index=False)
    punct_df.to_csv(FINDINGS_DIR / "raw_punct_data.csv", index=False)
    
    # Dickens
    rhythm_agg_d, punct_agg_d, _, _ = analyze_rhythm_subset(test_df, "dickens")
    rhythm_agg_d.to_csv(FINDINGS_DIR / "dickens_rhythm_statistics.csv", index=False)
    
    # Austen
    rhythm_agg_a, punct_agg_a, _, _ = analyze_rhythm_subset(test_df, "austen")
    rhythm_agg_a.to_csv(FINDINGS_DIR / "austen_rhythm_statistics.csv", index=False)
    
    print("Running Attribution Concentration...")
    calc_attribution_concentration()
    
    print("Rhythm Analysis Complete.")

if __name__ == "__main__":
    main()
