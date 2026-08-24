import os
import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel
from captum.attr import LayerIntegratedGradients
import random

SCRIPT_DIR = Path(__file__).resolve().parent
TASK3_DIR = SCRIPT_DIR.parent
ROOT_DIR = TASK3_DIR.parent
TIER_C_DIR = ROOT_DIR / "TASK 2-The Multi-Tiered Detective/tier_C_transformer"
MODELS_DIR = TIER_C_DIR / "models/roberta_lora_detector/book_split"
RESULTS_DIR = TASK3_DIR / "results"
ERROR_DIR = RESULTS_DIR / "error_analysis"

BASE_MODEL = "roberta-base"
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

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

def custom_forward(inputs, attention_mask):
    outputs = model(inputs, attention_mask=attention_mask)
    return outputs.logits

def get_phrases(words_data):
    # Same logic as integrated_gradients to get phrases
    words = []
    current_word = []
    current_word_attr = 0
    current_word_tokens = []
    
    for t in words_data:
        if t['is_new_word'] and len(current_word) > 0:
            words.append({
                "word": "".join(current_word),
                "attribution": current_word_attr,
                "tokens": current_word_tokens
            })
            current_word = []
            current_word_attr = 0
            current_word_tokens = []
            
        current_word.append(t['clean_token'])
        current_word_attr += t['attribution']
        current_word_tokens.append(t['token_index'])
        
    if len(current_word) > 0:
        words.append({
            "word": "".join(current_word),
            "attribution": current_word_attr,
            "tokens": current_word_tokens
        })
        
    # extract bigrams
    bigrams = []
    for i in range(len(words)-1):
        bigrams.append({
            "phrase": f"{words[i]['word']} {words[i+1]['word']}",
            "attribution": words[i]['attribution'] + words[i+1]['attribution'],
            "tokens": words[i]['tokens'] + words[i+1]['tokens']
        })
        
    trigrams = []
    for i in range(len(words)-2):
        trigrams.append({
            "phrase": f"{words[i]['word']} {words[i+1]['word']} {words[i+2]['word']}",
            "attribution": words[i]['attribution'] + words[i+1]['attribution'] + words[i+2]['attribution'],
            "tokens": words[i]['tokens'] + words[i+1]['tokens'] + words[i+2]['tokens']
        })
        
    return words, bigrams, trigrams

def main():
    print("Running Saliency and Counterfactuals...")
    global model
    model, tokenizer, device = load_model_and_tokenizer()
    
    # Needs a wrapper for captum
    def captum_forward_func(inputs, attention_mask=None):
        return model(inputs, attention_mask=attention_mask).logits
        
    lig = LayerIntegratedGradients(captum_forward_func, model.base_model.model.roberta.embeddings)
    
    with open(ERROR_DIR / "selected_false_positives.json", "r") as f:
        fps = json.load(f)
        
    attributions_out = []
    counterfactuals = []
    
    with open(TASK3_DIR / "findings/ai_ism_candidates.json", "r") as f:
        ai_isms = json.load(f)
        ai_isms = [x.lower() for x in ai_isms]
        
    for fp in fps:
        pid = fp['paragraph_id']
        text = fp['text']
        original_ai_prob = fp['ai_probability']
        
        inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt", max_length=512).to(device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        baseline_input_ids = torch.full_like(input_ids, tokenizer.pad_token_id).to(device)
        
        attr, delta = lig.attribute(
            inputs=input_ids,
            baselines=baseline_input_ids,
            additional_forward_args=(attention_mask,),
            target=1, # AI class
            n_steps=50,
            internal_batch_size=5,
            return_convergence_delta=True
        )
        
        attr_sum = attr.sum(dim=-1).squeeze(0).cpu().detach().numpy()
        tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).tolist())
        
        token_data = []
        for i, (tok, a) in enumerate(zip(tokens, attr_sum)):
            if tok in [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token]:
                continue
            clean = tok.replace('Ġ', '') if tok.startswith('Ġ') else tok
            is_new = tok.startswith('Ġ') or i == 1
            token_data.append({
                "token_index": i,
                "raw_token": tok,
                "clean_token": clean,
                "is_new_word": is_new,
                "attribution": float(a)
            })
            
        # AI-isms check
        found_ai_isms = []
        words, bigrams, trigrams = get_phrases(token_data)
        for w in words:
            if w['word'].lower() in ai_isms:
                found_ai_isms.append({
                    "word": w['word'],
                    "attribution": w['attribution']
                })
        
        # Sort tokens
        sorted_tokens = sorted(token_data, key=lambda x: x['attribution'], reverse=True)
        top_pos = sorted_tokens[:10]
        top_neg = sorted_tokens[-10:]
        
        # Sort phrases
        sorted_bigrams = sorted(bigrams, key=lambda x: x['attribution'], reverse=True)
        sorted_trigrams = sorted(trigrams, key=lambda x: x['attribution'], reverse=True)
        
        top_positive_bigrams = sorted_bigrams[:5]
        top_positive_trigrams = sorted_trigrams[:5]
        
        attributions_out.append({
            "paragraph_id": pid,
            "top_positive_tokens": [{"token": x['clean_token'], "attr": x['attribution']} for x in top_pos],
            "top_negative_tokens": [{"token": x['clean_token'], "attr": x['attribution']} for x in top_neg],
            "top_positive_bigrams": [{"phrase": x['phrase'], "attr": x['attribution']} for x in top_positive_bigrams],
            "top_positive_trigrams": [{"phrase": x['phrase'], "attr": x['attribution']} for x in top_positive_trigrams],
            "ai_isms_found": found_ai_isms,
            "full_tokens": [{"token": x['raw_token'], "attr": x['attribution']} for x in token_data] # For html vis
        })
        
        # Counterfactuals
        # 1. Remove top 5 positive tokens
        top_5_pos_indices = [x['token_index'] for x in sorted_tokens[:5]]
        
        # 2. Remove top 5 positive bigrams/trigrams? Just say top 5 positive words
        sorted_words = sorted(words, key=lambda x: x['attribution'], reverse=True)
        top_5_phrase_indices = []
        for w in sorted_words[:5]:
            top_5_phrase_indices.extend(w['tokens'])
            
        # 3. Random removal of 5 tokens
        valid_indices = [x['token_index'] for x in token_data]
        random_5_indices = random.sample(valid_indices, min(5, len(valid_indices)))
        
        def run_ablation(remove_indices):
            ablated_ids = input_ids.clone()
            # Replace removed indices with pad token
            for idx in remove_indices:
                if 0 <= idx < ablated_ids.size(1):
                    ablated_ids[0, idx] = tokenizer.pad_token_id
                    
            with torch.no_grad():
                outputs = model(ablated_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=-1).squeeze().cpu().numpy()
                return float(probs[1])
                
        prob_token_rem = run_ablation(top_5_pos_indices)
        prob_phrase_rem = run_ablation(top_5_phrase_indices)
        prob_rand_rem = run_ablation(random_5_indices)
        
        counterfactuals.append({
            "paragraph_id": pid,
            "original_ai_probability": original_ai_prob,
            "token_removal_probability": prob_token_rem,
            "phrase_removal_probability": prob_phrase_rem,
            "random_removal_probability": prob_rand_rem
        })
        
        fp['ai_isms_present'] = len(found_ai_isms)
        fp['ai_isms_details'] = found_ai_isms
        
    with open(ERROR_DIR / "false_positive_attributions.json", "w") as f:
        json.dump(attributions_out, f, indent=4)
        
    cf_df = pd.DataFrame(counterfactuals)
    cf_df.to_csv(ERROR_DIR / "counterfactuals.csv", index=False)
    
    with open(ERROR_DIR / "selected_false_positives.json", "w") as f:
        json.dump(fps, f, indent=4)
        
    print("Saliency and counterfactuals complete.")

if __name__ == "__main__":
    main()
