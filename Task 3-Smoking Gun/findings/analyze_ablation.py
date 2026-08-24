import os
import json
import torch
import pandas as pd
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
import re

SCRIPT_DIR = Path(__file__).resolve().parent
TASK3_DIR = SCRIPT_DIR.parent
ROOT_DIR = TASK3_DIR.parent
TIER_C_DIR = ROOT_DIR / "TASK 2-The Multi-Tiered Detective/tier_C_transformer"
MODELS_DIR = TIER_C_DIR / "models/roberta_lora_detector/book_split"
RESULTS_DIR = TASK3_DIR / "results"
FINDINGS_DIR = RESULTS_DIR / "findings"
FINDINGS_DIR.mkdir(parents=True, exist_ok=True)

BASE_MODEL = "roberta-base"

def get_device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    elif torch.cuda.is_available(): return torch.device("cuda")
    return torch.device("cpu")

def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(str(MODELS_DIR))
    base_model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)
    model = PeftModel.from_pretrained(base_model, str(MODELS_DIR))
    
    device = get_device()
    model.to(device)
    model.eval()
    return model, tokenizer, device

def predict_prob(model, tokenizer, device, text):
    inputs = tokenizer(text, truncation=True, padding=True, max_length=512, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
    return float(probs[1])

def main():
    print("Running Ablation Test...")
    model, tokenizer, device = load_model_and_tokenizer()
    
    # Load IG results (we only ablate on the ones we ran IG for)
    token_attr_path = RESULTS_DIR / "token_attributions.csv"
    if not token_attr_path.exists():
        print("token_attributions.csv not found!")
        return
        
    ta_df = pd.read_csv(token_attr_path)
    
    # Get original texts for these IDs
    audit_df = pd.read_csv(RESULTS_DIR / "prediction_audit.csv")
    selected_examples_path = RESULTS_DIR / "selected_examples.json"
    with open(selected_examples_path, 'r') as f:
        selected_examples = json.load(f)
        
    # We only care about True Positives (AI)
    tp_ids = [ex['paragraph_id'] for ex in selected_examples if ex.get('true_label') == 1 and ex.get('predicted_label') == 1]
    
    with open(SCRIPT_DIR / "ai_ism_candidates.json", "r") as f:
        ai_isms = json.load(f)
        
    ablation_rows = []
    
    for pid in tqdm(tp_ids):
        # Find original text
        orig_text = [ex['text'] for ex in selected_examples if ex['paragraph_id'] == pid][0]
        
        # 1. Base prediction
        prob_orig = predict_prob(model, tokenizer, device, orig_text)
        
        # 2. Remove AI-isms
        text_a = orig_text
        for term in ai_isms:
            pattern = re.compile(rf"\b{term}\b", re.IGNORECASE)
            text_a = pattern.sub('', text_a)
        
        prob_a = predict_prob(model, tokenizer, device, text_a)
        
        # 3. Remove Top 5 Attributed Tokens
        # Get top positive tokens for this paragraph
        para_tokens = ta_df[ta_df['paragraph_id'] == pid].copy()
        # Sort by positive attribution descending
        top_tokens = para_tokens.sort_values('attribution', ascending=False).head(5)['raw_token'].tolist()
        
        text_b = orig_text
        for token in top_tokens:
            # tokens from roberta usually have Ġ prefix which represents space
            clean_token = token.replace('Ġ', '').strip()
            # We just string replace the exact token
            if clean_token:
                text_b = text_b.replace(clean_token, '')
                
        prob_b = predict_prob(model, tokenizer, device, text_b)
        
        ablation_rows.append({
            "paragraph_id": pid,
            "original_probability": prob_orig,
            "after_ai_ism_removal": prob_a,
            "after_top_attribution_removal": prob_b,
            "ai_ism_probability_change": prob_a - prob_orig,
            "attribution_probability_change": prob_b - prob_orig
        })
        
    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df.to_csv(FINDINGS_DIR / "findings_ablation.csv", index=False)
    
    print("Ablation Analysis Complete.")

if __name__ == "__main__":
    main()
