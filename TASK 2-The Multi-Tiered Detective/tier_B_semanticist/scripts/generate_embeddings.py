import os
import pandas as pd
import numpy as np
import gensim.downloader as api
from gensim.models import KeyedVectors
from pathlib import Path
from tqdm import tqdm
import re

SCRIPT_DIR = Path(__file__).resolve().parent
TIER_B_DIR = SCRIPT_DIR.parent
DATA_CSV = TIER_B_DIR / "data/text_pairs.csv"
EMBEDDINGS_DIR = TIER_B_DIR / "embeddings"
EMBEDDINGS_MODEL_PATH = EMBEDDINGS_DIR / "fasttext.model"
OUT_EMBEDDINGS = TIER_B_DIR / "data/embeddings.npy"
OUT_LABELS = TIER_B_DIR / "data/labels.npy"

def tokenize(text):
    # Very basic tokenizer
    text = text.lower()
    return re.findall(r'\b\w+\b', text)

def get_sentence_embedding(text, model):
    words = tokenize(text)
    vectors = []
    for w in words:
        if w in model:
            vectors.append(model[w])
            
    if len(vectors) == 0:
        return np.zeros(300)
    return np.mean(vectors, axis=0)

def main():
    print("Loading data...")
    df = pd.read_csv(DATA_CSV)
    
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check if model is already downloaded locally in the project
    if EMBEDDINGS_MODEL_PATH.exists():
        print("Loading FastText from local path...")
        model = KeyedVectors.load(str(EMBEDDINGS_MODEL_PATH))
    else:
        print("Downloading FastText model (this may take a few minutes for ~1GB)...")
        model = api.load("fasttext-wiki-news-subwords-300")
        print("Saving model locally...")
        model.save(str(EMBEDDINGS_MODEL_PATH))
        
    print("Generating embeddings for all paragraphs...")
    
    all_embeddings = []
    labels = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        emb = get_sentence_embedding(row['text'], model)
        all_embeddings.append(emb)
        labels.append(row['label'])
        
    all_embeddings = np.array(all_embeddings)
    labels = np.array(labels)
    
    print(f"Embeddings shape: {all_embeddings.shape}")
    print(f"Labels shape: {labels.shape}")
    
    np.save(OUT_EMBEDDINGS, all_embeddings)
    np.save(OUT_LABELS, labels)
    print("Saved embeddings.npy and labels.npy successfully.")

if __name__ == "__main__":
    main()
