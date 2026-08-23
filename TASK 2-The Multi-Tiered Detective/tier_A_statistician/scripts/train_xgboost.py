import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
TIER_A_DIR = SCRIPT_DIR.parent
DATA_CSV = TIER_A_DIR / "data/fingerprint_features.csv"
MODEL_OUT = TIER_A_DIR / "models/xgboost_model.pkl"

def main():
    print("Loading data for XGBoost...")
    df = pd.read_csv(DATA_CSV)
    
    # Extract features and labels
    feature_cols = [c for c in df.columns if c not in ['paragraph_id', 'author', 'book', 'topic', 'label']]
    X = df[feature_cols]
    y = df['label']
    
    print(f"Features used ({len(feature_cols)}): {feature_cols}")
    
    # Train-test split (Mode 1 equivalent)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Initialize Model
    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    print("Training XGBoost model...")
    model.fit(X_train, y_train)
    
    # Save model
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump((model, feature_cols), MODEL_OUT)
    print(f"XGBoost model saved to {MODEL_OUT}")

if __name__ == "__main__":
    main()
