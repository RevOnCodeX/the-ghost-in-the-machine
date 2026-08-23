import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
TIER_A_DIR = SCRIPT_DIR.parent
DATA_CSV = TIER_A_DIR / "data/fingerprint_features.csv"
MODEL_OUT = TIER_A_DIR / "models/randomforest_model.pkl"

def main():
    print("Loading data for Random Forest...")
    df = pd.read_csv(DATA_CSV)
    
    # Extract features and labels
    feature_cols = [c for c in df.columns if c not in ['paragraph_id', 'author', 'book', 'topic', 'label']]
    X = df[feature_cols]
    y = df['label']
    
    print(f"Features used ({len(feature_cols)}): {feature_cols}")
    
    # Train-test split (Mode 1 equivalent)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Initialize Model
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        random_state=42
    )
    
    print("Training Random Forest model...")
    model.fit(X_train, y_train)
    
    # Save model
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump((model, feature_cols), MODEL_OUT)
    print(f"Random Forest model saved to {MODEL_OUT}")

if __name__ == "__main__":
    main()
