import re
from collections import Counter
import textstat
import numpy as np

def extract_tier_a_features(text: str) -> np.ndarray:
    words = re.findall(r'\b[a-zA-Z]+\b', text)
    word_count = len(words)
    if word_count == 0:
        return np.zeros((1, 9))
    
    # Lexical richness (TTR and Hapax Legomena)
    words_lower = [w.lower() for w in words]
    counts = Counter(words_lower)
    ttr = len(counts) / word_count
    hapax = sum(1 for w, c in counts.items() if c == 1)
    
    # Readability & Sentence Length
    read_word_count = textstat.lexicon_count(text, removepunct=True)
    if read_word_count == 0:
        read_word_count = word_count
    sentence_count = textstat.sentence_count(text)
    if sentence_count == 0:
        sentence_count = 1
    fk_grade = textstat.flesch_kincaid_grade(text)
    sentence_length = read_word_count / sentence_count
    
    # Punctuation Densities
    em_dashes = len(re.findall(r'—|--', text))
    semicolon_count = text.count(';')
    exclamation_count = text.count('!')
    question_count = text.count('?')
    comma_count = text.count(',')
    
    features = [
        ttr,
        hapax,
        fk_grade,
        sentence_length,
        semicolon_count / word_count,
        em_dashes / word_count,
        exclamation_count / word_count,
        question_count / word_count,
        comma_count / word_count
    ]
    return np.array([features])
