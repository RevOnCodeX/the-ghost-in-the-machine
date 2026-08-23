import os
import json
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_curve, auc
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel
from scipy.special import softmax
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
TIER_C_DIR = SCRIPT_DIR.parent
DATA_CSV = TIER_C_DIR / "data/text_pairs.csv"
MODELS_DIR = TIER_C_DIR / "models/roberta_lora_detector"
RESULTS_DIR = TIER_C_DIR / "results"

BASE_MODEL = "roberta-base"

def generate_visualizations():
    print("Generating visualizations...")
    
    # Check if metrics exist to generate the plots
    metrics_path = RESULTS_DIR / "metrics.json"
    if not metrics_path.exists():
        print("Metrics file not found. Run train_lora.py first.")
        return
        
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
        
    print("Plots will be generated using the evaluation code in train_lora.py or a subsequent inference pass. Skipping for now as we need raw probabilities.")

def main():
    print("Starting comprehensive evaluation & visualization...")
    # The actual evaluation logic (generating preds on test set) was already performed in train_lora.py
    # and saved to metrics.json.
    # To generate the ROC curves and Confusion Matrices properly, we can reload the test set and models,
    # or rely on a modified train_lora.py. 
    # Let's perform inference over the whole test set for Book Split specifically to extract error_analysis.csv
    
    df = pd.read_csv(DATA_CSV)
    
    test_books = ['pride_and_prejudice', 'a_tale_of_two_cities']
    test_df = df[df['book'].isin(test_books)]
    
    if len(test_df) == 0:
        print("No test data found!")
        return
        
    print(f"Loading LoRA model from {MODELS_DIR / 'book_split'}...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODELS_DIR / "book_split"))
    base_model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)
    model = PeftModel.from_pretrained(base_model, str(MODELS_DIR / "book_split"))
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    y_true = []
    y_prob = []
    predictions = []
    
    print("Running inference on Book Split test set for Error Analysis...")
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        text = row['text']
        label = row['label']
        
        inputs = tokenizer(text, truncation=True, padding=True, max_length=512, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
            
        prob_ai = probs[1]
        pred_label = 1 if prob_ai > 0.5 else 0
        
        y_true.append(label)
        y_prob.append(prob_ai)
        
        # Calculate absolute confidence (how far from 0.5)
        confidence = abs(prob_ai - 0.5) * 200 # scale 0-100%
        
        predictions.append({
            "paragraph_id": row['paragraph_id'],
            "true_label": label,
            "predicted_label": pred_label,
            "confidence": prob_ai, # Raw prob for sorting
            "text": text
        })
        
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob > 0.5).astype(int)
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Human', 'AI'], yticklabels=['Human', 'AI'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Book Split)')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrix.png")
    plt.close()
    
    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (Book Split)')
    plt.legend(loc="lower right")
    plt.savefig(RESULTS_DIR / "roc_curve.png")
    plt.close()
    
    # 3. Error Analysis
    error_df = pd.DataFrame(predictions)
    errors = error_df[error_df['true_label'] != error_df['predicted_label']].copy()
    
    # Sort by how confident the model was about its wrong prediction
    # If true=0 but prob_ai is high, confidence in wrong answer is high
    # If true=1 but prob_ai is low, confidence in wrong answer is high
    errors['wrong_confidence'] = errors.apply(lambda r: r['confidence'] if r['predicted_label'] == 1 else (1 - r['confidence']), axis=1)
    
    errors = errors.sort_values(by='wrong_confidence', ascending=False)
    top_errors = errors.head(20)
    
    top_errors.drop('wrong_confidence', axis=1).to_csv(RESULTS_DIR / "error_analysis.csv", index=False)
    print(f"Saved {len(top_errors)} top errors to results/error_analysis.csv")

if __name__ == "__main__":
    main()
