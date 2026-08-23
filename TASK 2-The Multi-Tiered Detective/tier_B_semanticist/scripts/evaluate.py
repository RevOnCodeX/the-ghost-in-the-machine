import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from pathlib import Path
from train_semantic_nn import SemanticNN, train_model

SCRIPT_DIR = Path(__file__).resolve().parent
TIER_B_DIR = SCRIPT_DIR.parent
DATA_CSV = TIER_B_DIR / "data/text_pairs.csv"
DATA_EMBEDDINGS = TIER_B_DIR / "data/embeddings.npy"
DATA_LABELS = TIER_B_DIR / "data/labels.npy"
RESULTS_DIR = TIER_B_DIR / "results"

def get_metrics(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob))
    }

def main():
    print("Loading data for evaluation...")
    df = pd.read_csv(DATA_CSV)
    X = np.load(DATA_EMBEDDINGS)
    y = np.load(DATA_LABELS)
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    metrics = {}
    
    # --- EXPERIMENT 1: Random Split ---
    from sklearn.model_selection import train_test_split
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    print("Training Mode 1 (Random Split)...")
    model_r, hist_r = train_model(X_train_r, y_train_r, X_test_r, y_test_r, epochs=50)
    
    model_r.eval()
    with torch.no_grad():
        y_prob_r = model_r(torch.tensor(X_test_r, dtype=torch.float32)).numpy().squeeze()
        
    metrics["random_split"] = get_metrics(y_test_r, y_prob_r)
    
    # Save training curves
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(hist_r['train_loss'], label='Train Loss')
    plt.plot(hist_r['val_loss'], label='Val Loss')
    plt.title('Loss (Random Split)')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(hist_r['val_acc'], label='Val Acc')
    plt.title('Accuracy (Random Split)')
    plt.legend()
    plt.savefig(RESULTS_DIR / "training_curve.png")
    plt.close()
    
    # Save ROC and Confusion Matrix for Random Split
    fpr_r, tpr_r, _ = roc_curve(y_test_r, y_prob_r)
    plt.figure()
    plt.plot(fpr_r, tpr_r, color='darkorange', lw=2, label=f'Random Split ROC (AUC = {metrics["random_split"]["roc_auc"]:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc="lower right")
    plt.savefig(RESULTS_DIR / "roc_curve.png")
    plt.close()
    
    y_pred_r = (y_prob_r >= 0.5).astype(int)
    ConfusionMatrixDisplay.from_predictions(y_test_r, y_pred_r, cmap='Blues', display_labels=['Human', 'AI'])
    plt.title('Confusion Matrix (Random Split)')
    plt.savefig(RESULTS_DIR / "confusion_matrix.png")
    plt.close()
    
    # --- EXPERIMENT 2: Book Split ---
    train_books = ['oliver_twist', 'emma', 'sense_and_sensibility', 'great_expectations']
    test_books = ['pride_and_prejudice', 'a_tale_of_two_cities']
    
    train_idx = df.index[df['book'].isin(train_books)].tolist()
    test_idx = df.index[df['book'].isin(test_books)].tolist()
    
    if len(train_idx) == 0 or len(test_idx) == 0:
        print("WARNING: Book split missing data. Check book names in text_pairs.csv.")
        metrics["book_split"] = {"error": "Missing books"}
    else:
        X_train_b, y_train_b = X[train_idx], y[train_idx]
        X_test_b, y_test_b = X[test_idx], y[test_idx]
        
        print("Training Mode 2 (Book Split)...")
        model_b, _ = train_model(X_train_b, y_train_b, X_test_b, y_test_b, epochs=50)
        
        model_b.eval()
        with torch.no_grad():
            y_prob_b = model_b(torch.tensor(X_test_b, dtype=torch.float32)).numpy().squeeze()
            
        metrics["book_split"] = get_metrics(y_test_b, y_prob_b)

    # Save metrics
    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
        
    print("Evaluation complete. Metrics and visual plots saved.")

if __name__ == "__main__":
    main()
