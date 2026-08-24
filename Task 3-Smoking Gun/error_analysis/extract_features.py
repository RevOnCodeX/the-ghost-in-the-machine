import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import re
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
TASK3_DIR = SCRIPT_DIR.parent
ROOT_DIR = TASK3_DIR.parent
TIER_C_DIR = ROOT_DIR / "TASK 2-The Multi-Tiered Detective/tier_C_transformer"
DATA_CSV = TIER_C_DIR / "data/text_pairs.csv"
RESULTS_DIR = TASK3_DIR / "results"
ERROR_DIR = RESULTS_DIR / "error_analysis"

# Import Task 1 functions
sys.path.append(str(ROOT_DIR / "TASK 1-The Fingerprint/readability"))
sys.path.append(str(ROOT_DIR / "TASK 1-The Fingerprint/lexical_richness"))
sys.path.append(str(ROOT_DIR / "TASK 1-The Fingerprint/punctuation_density"))

try:
    from analyze_readability import calculate_readability
    from analyze_lexical_richness import calculate_metrics as calculate_lexical
    from analyze_punctuation import get_punctuation_counts, tokenize as punct_tokenize
except ImportError as e:
    print(f"Error importing Task 1 functions: {e}")
    sys.exit(1)

def calc_repetition(text):
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    total_words = len(words)
    if total_words == 0:
        return 0, 0, 0, ""
        
    word_counts = {}
    for w in words:
        word_counts[w] = word_counts.get(w, 0) + 1
        
    repeated_words = sum(c for c in word_counts.values() if c > 1)
    rep_word_ratio = repeated_words / total_words
    
    # Bigrams
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
    bigram_counts = {}
    for b in bigrams:
        bigram_counts[b] = bigram_counts.get(b, 0) + 1
    repeated_bigrams = sum(c for c in bigram_counts.values() if c > 1)
    rep_bigram_ratio = repeated_bigrams / len(bigrams) if bigrams else 0
    
    # Trigrams
    trigrams = [f"{words[i]} {words[i+1]} {words[i+2]}" for i in range(len(words)-2)]
    trigram_counts = {}
    for t in trigrams:
        trigram_counts[t] = trigram_counts.get(t, 0) + 1
    repeated_trigrams = sum(c for c in trigram_counts.values() if c > 1)
    rep_trigram_ratio = repeated_trigrams / len(trigrams) if trigrams else 0
    
    # Most repeated
    most_repeated = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    most_rep_str = ", ".join(f"{k}({v})" for k, v in most_repeated if v > 1)
    
    return rep_word_ratio, rep_bigram_ratio, rep_trigram_ratio, most_rep_str

def calc_rhythm(text):
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
    
    sentence_lengths = [len(s.split()) for s in sentences]
    
    num_sentences = len(sentence_lengths)
    if num_sentences == 0:
        return 0, 0, 0, 0, 0, 0, 0
        
    mean_sl = np.mean(sentence_lengths)
    median_sl = np.median(sentence_lengths)
    std_sl = np.std(sentence_lengths)
    cv_sl = std_sl / mean_sl if mean_sl > 0 else 0
    min_sl = np.min(sentence_lengths)
    max_sl = np.max(sentence_lengths)
    
    # Difference adjacent
    diff_adj = [abs(sentence_lengths[i] - sentence_lengths[i-1]) for i in range(1, len(sentence_lengths))]
    mean_diff_adj = np.mean(diff_adj) if diff_adj else 0
    
    return mean_sl, median_sl, std_sl, cv_sl, min_sl, max_sl, mean_diff_adj

def calculate_all_features(text):
    # Readability
    read_metrics = calculate_readability(text) or {}
    fk_grade = read_metrics.get("flesch_kincaid_grade", 0)
    
    # Lexical
    lex = calculate_lexical(text) or {}
    word_count = lex.get("word_count", 0)
    ttr = lex.get("ttr", 0)
    hapax = lex.get("hapax_legomena", 0)
    
    words = punct_tokenize(text)
    word_lengths = [len(w) for w in words]
    mean_wl = np.mean(word_lengths) if word_lengths else 0
    
    # Punctuation
    punct = get_punctuation_counts(text)
    p_density = (sum(punct.values()) / word_count) * 1000 if word_count > 0 else 0
    semicolon_density = (punct.get('semicolon', 0) / word_count) * 1000 if word_count > 0 else 0
    em_dash_density = (punct.get('em_dash', 0) / word_count) * 1000 if word_count > 0 else 0
    exclamation_density = (punct.get('exclamation', 0) / word_count) * 1000 if word_count > 0 else 0
    
    # Repetition
    rep_w, rep_b, rep_t, most_rep = calc_repetition(text)
    
    # Rhythm
    mean_sl, median_sl, std_sl, cv_sl, min_sl, max_sl, mean_diff_adj = calc_rhythm(text)
    
    return {
        "word_count": word_count,
        "mean_sentence_length": mean_sl,
        "sentence_length_std": std_sl,
        "sentence_length_cv": cv_sl,
        "mean_diff_adj_sentence_length": mean_diff_adj,
        "mean_word_length": mean_wl,
        "TTR": ttr,
        "hapax_count": hapax,
        "Flesch-Kincaid grade": fk_grade,
        "punctuation_density": p_density,
        "semicolon_density": semicolon_density,
        "em_dash_density": em_dash_density,
        "exclamation_density": exclamation_density,
        "repeated_word_ratio": rep_w,
        "repeated_bigram_ratio": rep_b,
        "repeated_trigram_ratio": rep_t,
        "most_repeated_content_words": most_rep
    }

def main():
    print("Calculating Features...")
    df = pd.read_csv(DATA_CSV)
    test_books = ['pride_and_prejudice', 'tale_of_two_cities']
    test_df = df[df['book'].isin(test_books)].copy()
    
    human_df = test_df[test_df['label'] == 0].copy()
    ai_df = test_df[test_df['label'] == 1].copy()
    
    # Calculate for all human
    print(f"Extracting features for {len(human_df)} human paragraphs...")
    human_features = []
    for _, row in human_df.iterrows():
        human_features.append(calculate_all_features(str(row['text'])))
    h_feat_df = pd.DataFrame(human_features)
    
    # Calculate for all AI
    print(f"Extracting features for {len(ai_df)} AI paragraphs...")
    ai_features = []
    for _, row in ai_df.iterrows():
        ai_features.append(calculate_all_features(str(row['text'])))
    a_feat_df = pd.DataFrame(ai_features)
    
    # Load FPs
    with open(ERROR_DIR / "selected_false_positives.json", "r") as f:
        fps = json.load(f)
        
    zscore_rows = []
    numerical_features = [c for c in h_feat_df.columns if c != "most_repeated_content_words"]
    
    for fp in fps:
        pid = fp['paragraph_id']
        text = fp['text']
        fp_feats = calculate_all_features(text)
        
        # Add to FPs JSON to preserve for later steps
        fp['features'] = fp_feats
        
        for feat in numerical_features:
            h_mean = h_feat_df[feat].mean()
            h_std = h_feat_df[feat].std()
            a_mean = a_feat_df[feat].mean()
            
            val = fp_feats[feat]
            
            z = (val - h_mean) / h_std if h_std > 0 else 0
            
            zscore_rows.append({
                "paragraph_id": pid,
                "feature": feat,
                "value": val,
                "human_mean": h_mean,
                "human_std": h_std,
                "z_score": z,
                "ai_mean": a_mean
            })
            
    # Save Z-scores
    z_df = pd.DataFrame(zscore_rows)
    z_df.to_csv(ERROR_DIR / "false_positive_features.csv", index=False)
    
    # Save enriched FP json
    with open(ERROR_DIR / "selected_false_positives.json", "w") as f:
        json.dump(fps, f, indent=4)
        
    print("Feature Extraction Complete.")
    
if __name__ == "__main__":
    main()
