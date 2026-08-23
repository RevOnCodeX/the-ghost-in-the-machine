import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
import joblib

SCRIPT_DIR = Path(__file__).resolve().parent
TIER_A_DIR = SCRIPT_DIR.parent
DATA_CSV = TIER_A_DIR / "data/fingerprint_features.csv"
RESULTS_DIR = TIER_A_DIR / "results"

def main():
    print("Generating visualizations...")
    df = pd.read_csv(DATA_CSV)
    
    feature_cols = [c for c in df.columns if c not in ['paragraph_id', 'author', 'book', 'topic', 'label']]
    X = df[feature_cols]
    y = df['label']
    
    # Re-run Mode 1 split to generate visuals
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Load trained models
    xgb, _ = joblib.load(TIER_A_DIR / "models/xgboost_model.pkl")
    rf, _ = joblib.load(TIER_A_DIR / "models/randomforest_model.pkl")
    
    y_pred_xgb = xgb.predict(X_test)
    y_prob_xgb = xgb.predict_proba(X_test)[:, 1]
    
    y_pred_rf = rf.predict(X_test)
    y_prob_rf = rf.predict_proba(X_test)[:, 1]
    
    # 1. Confusion Matrix (Side-by-side)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred_xgb, ax=axs[0], cmap='Blues', display_labels=['Human', 'AI'])
    axs[0].set_title('XGBoost Confusion Matrix')
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred_rf, ax=axs[1], cmap='Blues', display_labels=['Human', 'AI'])
    axs[1].set_title('Random Forest Confusion Matrix')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrix.png")
    plt.close()
    
    # 2. ROC Curve
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_prob_xgb)
    roc_auc_xgb = auc(fpr_xgb, tpr_xgb)
    
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
    roc_auc_rf = auc(fpr_rf, tpr_rf)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_xgb, tpr_xgb, color='darkorange', lw=2, label=f'XGBoost (AUC = {roc_auc_xgb:.3f})')
    plt.plot(fpr_rf, tpr_rf, color='navy', lw=2, label=f'Random Forest (AUC = {roc_auc_rf:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(RESULTS_DIR / "roc_curve.png")
    plt.close()
    
    # 3. Feature Importance
    # We will use Random Forest for standard Gini importance
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances (Random Forest)")
    plt.bar(range(X.shape[1]), importances[indices], align="center", color='teal')
    plt.xticks(range(X.shape[1]), [feature_cols[i] for i in indices], rotation=45, ha='right')
    plt.xlim([-1, X.shape[1]])
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "feature_importance.png")
    plt.close()
    
    print("Visualizations saved to results/")

if __name__ == "__main__":
    main()
