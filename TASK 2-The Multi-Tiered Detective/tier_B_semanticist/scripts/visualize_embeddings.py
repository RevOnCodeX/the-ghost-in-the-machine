import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
TIER_B_DIR = SCRIPT_DIR.parent
DATA_EMBEDDINGS = TIER_B_DIR / "data/embeddings.npy"
DATA_LABELS = TIER_B_DIR / "data/labels.npy"
RESULTS_DIR = TIER_B_DIR / "results"

def main():
    print("Loading embeddings for visualization...")
    X = np.load(DATA_EMBEDDINGS)
    y = np.load(DATA_LABELS)
    
    print(f"Applying t-SNE to {len(X)} embeddings (this may take a minute)...")
    # Use PCA initialization to make it slightly faster and more stable
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    X_2d = tsne.fit_transform(X)
    
    print("Plotting results...")
    plt.figure(figsize=(10, 8))
    
    human_mask = (y == 0)
    ai_mask = (y == 1)
    
    plt.scatter(X_2d[human_mask, 0], X_2d[human_mask, 1], c='blue', alpha=0.5, label='Human', s=10)
    plt.scatter(X_2d[ai_mask, 0], X_2d[ai_mask, 1], c='red', alpha=0.5, label='AI', s=10)
    
    plt.title('t-SNE Visualization of FastText Paragraph Embeddings')
    plt.legend()
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(RESULTS_DIR / "embedding_visualization.png")
    plt.close()
    
    print("Saved to results/embedding_visualization.png")

if __name__ == "__main__":
    main()
