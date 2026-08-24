import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import re
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import fdrcorrection
from collections import Counter

SCRIPT_DIR = Path(__file__).resolve().parent
TASK3_DIR = SCRIPT_DIR.parent
ROOT_DIR = TASK3_DIR.parent
TIER_C_DIR = ROOT_DIR / "TASK 2-The Multi-Tiered Detective/tier_C_transformer"
DATA_CSV = TIER_C_DIR / "data/text_pairs.csv"
RESULTS_DIR = TASK3_DIR / "results"
FINDINGS_DIR = RESULTS_DIR / "findings"
FINDINGS_DIR.mkdir(parents=True, exist_ok=True)

def analyze_author_subset(test_df, author_name=None):
    if author_name:
        test_df = test_df[test_df['author'] == author_name]
    
    human_df = test_df[test_df['label'] == 0]
    ai_df = test_df[test_df['label'] == 1]
    
    total_human_words = human_df['text'].apply(lambda x: len(x.split())).sum()
    total_ai_words = ai_df['text'].apply(lambda x: len(x.split())).sum()
    total_human_paras = len(human_df)
    total_ai_paras = len(ai_df)
    
    with open(SCRIPT_DIR / "ai_ism_candidates.json", "r") as f:
        candidates = json.load(f)
    
    freq_rows = []
    enrichment_rows = []
    
    for term in candidates:
        pattern = re.compile(rf"\b{term}\b", re.IGNORECASE)
        
        # Word counts
        human_count = human_df['text'].apply(lambda x: len(pattern.findall(x))).sum()
        ai_count = ai_df['text'].apply(lambda x: len(pattern.findall(x))).sum()
        
        # Paragraph counts (contains term or not)
        human_paras_with = human_df['text'].apply(lambda x: bool(pattern.search(x))).sum()
        ai_paras_with = ai_df['text'].apply(lambda x: bool(pattern.search(x))).sum()
        
        human_per_1000 = (human_count / total_human_words) * 1000 if total_human_words > 0 else 0
        ai_per_1000 = (ai_count / total_ai_words) * 1000 if total_ai_words > 0 else 0
        
        freq_rows.append({
            "term": term,
            "human_count": human_count,
            "ai_count": ai_count,
            "human_per_1000": human_per_1000,
            "ai_per_1000": ai_per_1000,
            "frequency_difference": ai_per_1000 - human_per_1000,
            "paragraphs_human": human_paras_with,
            "paragraphs_ai": ai_paras_with
        })
        
        # Enrichment (Fisher Exact Test)
        table = [
            [human_paras_with, total_human_paras - human_paras_with],
            [ai_paras_with, total_ai_paras - ai_paras_with]
        ]
        
        try:
            odds_ratio, p_value = fisher_exact(table, alternative='two-sided')
        except ValueError:
            odds_ratio, p_value = np.nan, 1.0
            
        enrichment_rows.append({
            "term": term,
            "odds_ratio": odds_ratio,
            "p_value": p_value,
            "human_rate": human_paras_with / total_human_paras if total_human_paras > 0 else 0,
            "ai_rate": ai_paras_with / total_ai_paras if total_ai_paras > 0 else 0
        })
        
    freq_df = pd.DataFrame(freq_rows)
    enrich_df = pd.DataFrame(enrichment_rows)
    
    # FDR Correction
    if len(enrich_df) > 0:
        rejected, fdr_p = fdrcorrection(enrich_df['p_value'].fillna(1.0), alpha=0.05)
        enrich_df['fdr_p_value'] = fdr_p
        enrich_df['significant'] = rejected
    
    return freq_df, enrich_df

def get_top_corpus_words(test_df, top_n=100):
    all_text = " ".join(test_df['text'].tolist())
    words = re.findall(r'\b[a-zA-Z]{3,}\b', all_text.lower())
    # Exclude common stopwords roughly
    stopwords = {"the", "and", "to", "of", "a", "in", "it", "is", "was", "i", "for", "that", "you", "he", "as", "with", "his", "on", "be", "at", "by", "this", "had", "not", "are", "but", "from", "or", "have", "an", "they", "which", "one", "you", "were", "her", "all", "she", "there", "would", "their", "we", "him", "been", "has", "when", "who", "will", "more", "no", "if", "out", "so", "said", "what", "up", "its", "about", "into", "than", "them", "can", "only", "other", "new", "some", "could", "time", "these", "two", "may", "then", "do", "first", "any", "my", "now", "such", "like", "our", "over", "man", "me", "even", "most", "made", "after", "also", "did", "many", "before", "must", "through", "back", "years", "where", "much", "your", "way", "well", "down", "should", "because", "each", "just", "those", "people", "mr", "how", "too", "little", "state", "good", "very", "make", "world", "still", "own", "see", "men", "work", "long", "get", "here", "between", "both", "life", "being", "under", "never", "day", "same", "another", "know", "while", "last", "might", "us", "great", "old", "year", "off", "come", "since", "against", "go", "came", "right", "used", "take", "three"}
    words = [w for w in words if w not in stopwords]
    counter = Counter(words)
    return [w for w, c in counter.most_common(top_n)]


def main():
    print("Running AI-ism Frequency & Enrichment Analysis...")
    df = pd.read_csv(DATA_CSV)
    test_books = ['pride_and_prejudice', 'tale_of_two_cities']
    test_df = df[df['book'].isin(test_books)].copy()
    
    # 1. Main Frequency and Enrichment
    freq_df, enrich_df = analyze_author_subset(test_df)
    freq_df.to_csv(FINDINGS_DIR / "ai_ism_frequency.csv", index=False)
    enrich_df.to_csv(FINDINGS_DIR / "ai_ism_enrichment.csv", index=False)
    
    # 2. Author Specific
    dickens_freq, dickens_enrich = analyze_author_subset(test_df, author_name="dickens")
    dickens_freq.to_csv(FINDINGS_DIR / "dickens_ai_ism_frequency.csv", index=False)
    dickens_enrich.to_csv(FINDINGS_DIR / "dickens_ai_ism_enrichment.csv", index=False)
    
    austen_freq, austen_enrich = analyze_author_subset(test_df, author_name="austen")
    austen_freq.to_csv(FINDINGS_DIR / "austen_ai_ism_frequency.csv", index=False)
    austen_enrich.to_csv(FINDINGS_DIR / "austen_ai_ism_enrichment.csv", index=False)
    
    # 3. Attribution-Based AI-ism Test (from word_attributions.csv)
    print("Running Attribution-Based Analysis...")
    word_attr_path = RESULTS_DIR / "word_attributions.csv"
    if word_attr_path.exists():
        wa_df = pd.read_csv(word_attr_path)
        # Clean words (remove punctuation for matching)
        wa_df['clean_word'] = wa_df['word'].astype(str).apply(lambda x: re.sub(r'[^a-zA-Z]', '', x).lower())
        
        with open(SCRIPT_DIR / "ai_ism_candidates.json", "r") as f:
            candidates = json.load(f)
            
        attr_rows = []
        for term in candidates:
            term_lower = term.lower()
            matches = wa_df[wa_df['clean_word'] == term_lower]
            if len(matches) > 0:
                pos_matches = matches[matches['attribution'] > 0]
                attr_rows.append({
                    "term": term,
                    "occurrence_count": len(matches),
                    "positive_attribution_count": len(pos_matches),
                    "mean_attribution": matches['attribution'].mean(),
                    "median_attribution": matches['attribution'].median(),
                    "mean_absolute_attribution": matches['absolute_attribution'].mean(),
                    "fraction_positive": len(pos_matches) / len(matches)
                })
            else:
                attr_rows.append({
                    "term": term,
                    "occurrence_count": 0,
                    "positive_attribution_count": 0,
                    "mean_attribution": 0,
                    "median_attribution": 0,
                    "mean_absolute_attribution": 0,
                    "fraction_positive": 0
                })
        
        attr_df = pd.DataFrame(attr_rows)
        attr_df.to_csv(FINDINGS_DIR / "attributed_ai_words.csv", index=False)
        
        # 4. Control Test
        top_corpus = get_top_corpus_words(test_df, top_n=100)
        control_rows = []
        for term in top_corpus:
            term_lower = term.lower()
            matches = wa_df[wa_df['clean_word'] == term_lower]
            if len(matches) > 0:
                pos_matches = matches[matches['attribution'] > 0]
                control_rows.append({
                    "term": term,
                    "occurrence_count": len(matches),
                    "positive_attribution_count": len(pos_matches),
                    "mean_attribution": matches['attribution'].mean(),
                    "median_attribution": matches['attribution'].median(),
                    "mean_absolute_attribution": matches['absolute_attribution'].mean(),
                    "fraction_positive": len(pos_matches) / len(matches)
                })
        
        control_df = pd.DataFrame(control_rows)
        control_df.to_csv(FINDINGS_DIR / "control_attributed_words.csv", index=False)
        
    # 5. Phrase Analysis
    phrase_attr_path = RESULTS_DIR / "phrase_attributions.csv"
    if phrase_attr_path.exists():
        pa_df = pd.read_csv(phrase_attr_path)
        # Group by phrase to find consistently positive attributions
        pa_grouped = pa_df.groupby('phrase').agg(
            occurrence_count=('phrase', 'count'),
            mean_attribution=('attribution', 'mean'),
            median_attribution=('attribution', 'median')
        ).reset_index()
        
        # Filter for actual phrases (spaces present) and at least 2 occurrences
        pa_grouped = pa_grouped[pa_grouped['phrase'].str.contains(" ", na=False)]
        
        # We need frequency in Human vs AI
        phrase_rows = []
        for idx, row in pa_grouped.iterrows():
            phrase = str(row['phrase']).strip()
            # To be efficient, only look at top 100 phrases by mean attribution
            pass
            
        top_phrases = pa_grouped.sort_values('mean_attribution', ascending=False).head(200)
        
        for idx, row in top_phrases.iterrows():
            phrase = str(row['phrase']).strip()
            # Clean for regex
            clean_phrase = re.sub(r'[^a-zA-Z\s]', '', phrase).lower()
            if len(clean_phrase.split()) < 2: continue
            
            pattern = re.compile(rf"\b{re.escape(clean_phrase)}\b", re.IGNORECASE)
            human_count = test_df[test_df['label'] == 0]['text'].apply(lambda x: len(pattern.findall(x))).sum()
            ai_count = test_df[test_df['label'] == 1]['text'].apply(lambda x: len(pattern.findall(x))).sum()
            
            phrase_rows.append({
                "phrase": phrase,
                "human_frequency": human_count,
                "ai_frequency": ai_count,
                "mean_attribution": row['mean_attribution'],
                "median_attribution": row['median_attribution'],
                "occurrence_in_ig_test": row['occurrence_count']
            })
            
        final_phrase_df = pd.DataFrame(phrase_rows)
        final_phrase_df.to_csv(FINDINGS_DIR / "attributed_ai_phrases.csv", index=False)
        
    print("AI-ism Analysis Complete.")

if __name__ == "__main__":
    main()
