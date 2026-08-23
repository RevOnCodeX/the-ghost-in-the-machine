import pandas as pd
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
TIER_A_DIR = SCRIPT_DIR.parent
DATA_CSV = TIER_A_DIR / "data/fingerprint_features.csv"

def evaluate_classifier(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob))
    }

def main():
    print("Evaluating models for Mode 1 and Mode 2...")
    df = pd.read_csv(DATA_CSV)
    
    feature_cols = [c for c in df.columns if c not in ['paragraph_id', 'author', 'book', 'topic', 'label']]
    
    # Instantiate clean models
    xgb = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42)
    rf = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42)
    
    # --- MODE 1: Random Split ---
    X = df[feature_cols]
    y = df['label']
    X_train_m1, X_test_m1, y_train_m1, y_test_m1 = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    xgb_m1 = evaluate_classifier(xgb, X_train_m1, y_train_m1, X_test_m1, y_test_m1)
    rf_m1 = evaluate_classifier(rf, X_train_m1, y_train_m1, X_test_m1, y_test_m1)
    
    # --- MODE 2: Book-Level Split ---
    train_books = ['oliver_twist', 'emma', 'sense_and_sensibility']
    test_books = ['great_expectations', 'pride_and_prejudice', 'a_tale_of_two_cities']
    
    df_train = df[df['book'].isin(train_books)]
    df_test = df[df['book'].isin(test_books)]
    
    if len(df_train) == 0 or len(df_test) == 0:
        print("WARNING: Missing books for Mode 2 split. Using random split as fallback for Mode 2.")
        df_train = df.sample(frac=0.8, random_state=42)
        df_test = df.drop(df_train.index)
        
    X_train_m2, y_train_m2 = df_train[feature_cols], df_train['label']
    X_test_m2, y_test_m2 = df_test[feature_cols], df_test['label']
    
    xgb_m2 = evaluate_classifier(xgb, X_train_m2, y_train_m2, X_test_m2, y_test_m2)
    rf_m2 = evaluate_classifier(rf, X_train_m2, y_train_m2, X_test_m2, y_test_m2)
    
    # Save results
    xgb_results = {"mode_1_random": xgb_m1, "mode_2_book": xgb_m2}
    rf_results = {"mode_1_random": rf_m1, "mode_2_book": rf_m2}
    
    with open(TIER_A_DIR / "results/xgboost_metrics.json", "w") as f:
        json.dump(xgb_results, f, indent=2)
        
    with open(TIER_A_DIR / "results/randomforest_metrics.json", "w") as f:
        json.dump(rf_results, f, indent=2)
        
    print("Evaluation complete. Metrics saved.")

if __name__ == "__main__":
    main()
