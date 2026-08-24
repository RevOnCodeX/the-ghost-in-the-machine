import os
import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel
from captum.attr import LayerIntegratedGradients
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

SCRIPT_DIR = Path(__file__).resolve().parent
TASK3_DIR = SCRIPT_DIR.parent
ROOT_DIR = TASK3_DIR.parent
TIER_C_DIR = ROOT_DIR / "TASK 2-The Multi-Tiered Detective/tier_C_transformer"
DATA_CSV = TIER_C_DIR / "data/text_pairs.csv"
MODELS_DIR = TIER_C_DIR / "models/roberta_lora_detector/book_split"
RESULTS_DIR = TASK3_DIR / "results"

BASE_MODEL = "roberta-base"
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

def get_device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    elif torch.cuda.is_available(): return torch.device("cuda")
    return torch.device("cpu")

def load_model_and_tokenizer():
    print(f"Loading tokenizer from {MODELS_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODELS_DIR))
    print(f"Loading base model {BASE_MODEL}...")
    base_model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)
    print(f"Loading LoRA adapter from {MODELS_DIR}...")
    model = PeftModel.from_pretrained(base_model, str(MODELS_DIR))
    
    device = get_device()
    model.to(device)
    model.eval()
    return model, tokenizer, device

def run_prediction_audit(model, tokenizer, device):
    print("Running Prediction Audit on Book-Split Test Set...")
    df = pd.read_csv(DATA_CSV)
    test_books = ['pride_and_prejudice', 'a_tale_of_two_cities']
    test_df = df[df['book'].isin(test_books)].copy()
    
    audit_rows = []
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        text = row['text']
        true_label = row['label']
        
        inputs = tokenizer(text, truncation=True, padding=True, max_length=512, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
            
        prob_ai = float(probs[1])
        prob_human = float(probs[0])
        pred_label = 1 if prob_ai > 0.5 else 0
        
        audit_rows.append({
            "paragraph_id": row['paragraph_id'],
            "book": row['book'],
            "author": row['author'],
            "true_label": int(true_label),
            "predicted_label": int(pred_label),
            "human_probability": prob_human,
            "ai_probability": prob_ai,
            "correct": int(true_label == pred_label),
            "split": "book_split_test",
            "text": text # Keep text for selection step later
        })
        
    audit_df = pd.DataFrame(audit_rows)
    
    # Calculate aggregate metrics to verify it matches Tier C report
    y_true = audit_df['true_label']
    y_pred = audit_df['predicted_label']
    y_prob = audit_df['ai_probability']
    
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob)
    }
    
    print("\\nVerified Model Metrics (Should match Tier C Book-Split):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
        
    # Save audit (excluding full text to keep CSV clean)
    audit_df_save = audit_df.drop(columns=['text'])
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    audit_df_save.to_csv(RESULTS_DIR / "prediction_audit.csv", index=False)
    
    return audit_df

def select_examples(audit_df):
    print("\\nSelecting Examples...")
    
    # Categories
    tp_ai = audit_df[(audit_df['true_label'] == 1) & (audit_df['predicted_label'] == 1)]
    fp_human = audit_df[(audit_df['true_label'] == 0) & (audit_df['predicted_label'] == 1)]
    
    # Group 1: 5 High Confidence AI (prob >= 0.90)
    high_conf = tp_ai[tp_ai['ai_probability'] >= 0.90].sample(n=5, random_state=SEED)
    
    # Group 2: 5 Borderline AI (0.50 <= prob < 0.90)
    borderline = tp_ai[(tp_ai['ai_probability'] >= 0.50) & (tp_ai['ai_probability'] < 0.90)]
    if len(borderline) < 5:
        print(f"Warning: Only {len(borderline)} borderline examples found (prob 0.5 to 0.9). Adjusting threshold.")
        # If the model is too confident, pick the lowest confidence ones available
        borderline = tp_ai.sort_values('ai_probability', ascending=True).head(5)
    else:
        borderline = borderline.sample(n=5, random_state=SEED)
        
    # Group 3: 3 False Positive Human
    fp_sel = fp_human.sample(n=3, random_state=SEED) if len(fp_human) >= 3 else fp_human
    
    selected = pd.concat([high_conf, borderline, fp_sel])
    
    selected_json = []
    for _, row in selected.iterrows():
        selected_json.append({
            "paragraph_id": row['paragraph_id'],
            "book": row['book'],
            "author": row['author'],
            "text": row['text'],
            "true_label": int(row['true_label']),
            "predicted_label": int(row['predicted_label']),
            "ai_probability": float(row['ai_probability'])
        })
        
    with open(RESULTS_DIR / "selected_examples.json", "w", encoding='utf-8') as f:
        json.dump(selected_json, f, indent=4)
        
    print(f"Selected {len(selected_json)} examples (10 TP AI, {len(fp_sel)} FP Human). Saved to selected_examples.json.")
    
    # Return just the TP AI ones for attribution analysis
    return high_conf, borderline

def perform_attribution(model, tokenizer, device, high_conf, borderline):
    print("\\nPerforming Integrated Gradients Attribution...")
    
    def forward_func(input_ids, attention_mask):
        return model(input_ids=input_ids, attention_mask=attention_mask).logits
    
    # Captum's LIG on the word embeddings layer.
    # We must traverse PeftModel -> RobertaForSequenceClassification -> RobertaModel -> RobertaEmbeddings -> Embedding
    embeddings_layer = model.base_model.model.roberta.embeddings.word_embeddings
    
    lig = LayerIntegratedGradients(forward_func, embeddings_layer)
    
    examples = pd.concat([high_conf, borderline])
    
    all_token_attr = []
    all_word_attr = []
    all_phrase_attr = []
    
    for _, row in tqdm(examples.iterrows(), total=len(examples)):
        p_id = row['paragraph_id']
        text = row['text']
        ai_prob = row['ai_probability']
        
        inputs = tokenizer(text, truncation=True, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        # Baseline: padding tokens. We use pad_token_id to represent "absence of signal".
        # This is the standard baseline in NLP for Transformers.
        baseline_input_ids = torch.full_like(input_ids, tokenizer.pad_token_id).to(device)
        
        attributions, delta = lig.attribute(
            inputs=input_ids,
            baselines=baseline_input_ids,
            additional_forward_args=(attention_mask,),
            target=1, # Attributing towards the AI class logit
            n_steps=50,
            internal_batch_size=5,
            return_convergence_delta=True
        )
        
        # attributions shape: (1, seq_len, embed_dim)
        # Sum across embedding dimensions to get attribution per token
        attributions_sum = attributions.sum(dim=-1).squeeze(0).cpu().detach().numpy()
        tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).tolist())
        
        delta_val = float(delta.cpu().detach().numpy()[0])
        
        # 1. Token Level Processing
        paragraph_tokens = []
        total_abs_attr = np.sum(np.abs(attributions_sum))
        
        for i, (token, attr) in enumerate(zip(tokens, attributions_sum)):
            if token in [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token]:
                continue
                
            abs_attr = abs(attr)
            norm_attr = abs_attr / total_abs_attr if total_abs_attr > 0 else 0
            
            # Remove the 'Ġ' character used by RoBERTa for spaces, but keep track of it for word merging
            clean_token = token.replace('Ġ', '') if token.startswith('Ġ') else token
            is_new_word = token.startswith('Ġ') or i == 1 # First real token is a new word even without G
            
            token_dict = {
                "paragraph_id": p_id,
                "token_index": i,
                "raw_token": token,
                "clean_token": clean_token,
                "is_new_word": is_new_word,
                "attribution": float(attr),
                "absolute_attribution": float(abs_attr),
                "normalized_attribution": float(norm_attr),
                "ai_probability": float(ai_prob),
                "convergence_delta": delta_val
            }
            paragraph_tokens.append(token_dict)
            all_token_attr.append(token_dict)
            
        # 2. Subword to Word Level Aggregation
        words = []
        current_word = ""
        current_attr = 0.0
        
        for t in paragraph_tokens:
            if t['is_new_word']:
                # Save previous word
                if current_word:
                    words.append({
                        "paragraph_id": p_id,
                        "word": current_word,
                        "attribution": float(current_attr),
                        "absolute_attribution": abs(float(current_attr))
                    })
                # Start new word
                current_word = t['clean_token']
                current_attr = t['attribution']
            else:
                # Merge subword
                current_word += t['clean_token']
                current_attr += t['attribution']
                
        # Append last word
        if current_word:
            words.append({
                "paragraph_id": p_id,
                "word": current_word,
                "attribution": float(current_attr),
                "absolute_attribution": abs(float(current_attr))
            })
            
        all_word_attr.extend(words)
        
        # 3. Phrase Level Aggregation (2-token and 3-token based on WORDS, not subwords)
        for length in [2, 3]:
            for i in range(len(words) - length + 1):
                phrase_words = words[i:i+length]
                phrase_str = " ".join([w['word'] for w in phrase_words])
                phrase_attr = sum([w['attribution'] for w in phrase_words])
                
                all_phrase_attr.append({
                    "paragraph_id": p_id,
                    "phrase": phrase_str,
                    "phrase_length": length,
                    "attribution": float(phrase_attr),
                    "absolute_attribution": abs(float(phrase_attr))
                })

    # Convert to DataFrames and rank
    df_token = pd.DataFrame(all_token_attr)
    df_token['rank'] = df_token.groupby('paragraph_id')['absolute_attribution'].rank(method='dense', ascending=False)
    
    df_word = pd.DataFrame(all_word_attr)
    df_word['rank'] = df_word.groupby('paragraph_id')['absolute_attribution'].rank(method='dense', ascending=False)
    
    df_phrase = pd.DataFrame(all_phrase_attr)
    df_phrase['rank'] = df_phrase.groupby(['paragraph_id', 'phrase_length'])['absolute_attribution'].rank(method='dense', ascending=False)
    
    # Save
    df_token.to_csv(RESULTS_DIR / "token_attributions.csv", index=False)
    df_word.to_csv(RESULTS_DIR / "word_attributions.csv", index=False)
    df_phrase.to_csv(RESULTS_DIR / "phrase_attributions.csv", index=False)
    print("Saved attributions to CSV.")

def main():
    print("Starting Task 3: Saliency Mapping pipeline...")
    model, tokenizer, device = load_model_and_tokenizer()
    audit_df = run_prediction_audit(model, tokenizer, device)
    high_conf, borderline = select_examples(audit_df)
    perform_attribution(model, tokenizer, device, high_conf, borderline)
    print("Integrated Gradients processing complete!")

if __name__ == "__main__":
    main()
