import os
import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
from captum.attr import LayerIntegratedGradients
import random
from tqdm import tqdm
from integrated_gradients import load_model_and_tokenizer, get_device

SCRIPT_DIR = Path(__file__).resolve().parent
TASK3_DIR = SCRIPT_DIR.parent
RESULTS_DIR = TASK3_DIR / "results"

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

def test_stability(model, tokenizer, device, examples):
    print("\\nRunning Stability Test (n_steps=100)...")
    
    # Load 50-step results to compare against
    token_attr_df = pd.read_csv(RESULTS_DIR / "token_attributions.csv")
    
    # Sample 5 examples for stability test
    stability_examples = examples[:5]
    
    embeddings_layer = model.base_model.model.roberta.embeddings.word_embeddings
    def forward_func(input_ids, attention_mask):
        return model(input_ids=input_ids, attention_mask=attention_mask).logits
        
    lig = LayerIntegratedGradients(forward_func, embeddings_layer)
    
    stability_rows = []
    
    for ex in tqdm(stability_examples, total=len(stability_examples)):
        p_id = ex['paragraph_id']
        text = ex['text']
        
        inputs = tokenizer(text, truncation=True, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        baseline_input_ids = torch.full_like(input_ids, tokenizer.pad_token_id).to(device)
        
        # Run 100 steps
        attributions = lig.attribute(
            inputs=input_ids,
            baselines=baseline_input_ids,
            additional_forward_args=(attention_mask,),
            target=1,
            n_steps=100,
            internal_batch_size=5
        )
        
        attr_100_sum = attributions.sum(dim=-1).squeeze(0).cpu().detach().numpy()
        
        # Extract 50 step attributions
        df_50 = token_attr_df[token_attr_df['paragraph_id'] == p_id].sort_values('token_index')
        # Filter out special tokens in the 100 step output to match length
        tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).tolist())
        
        valid_indices = [i for i, t in enumerate(tokens) if t not in [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token]]
        attr_100_clean = [attr_100_sum[i] for i in valid_indices]
        attr_50_clean = df_50['attribution'].tolist()
        
        # Calculate Spearman correlation
        if len(attr_100_clean) == len(attr_50_clean):
            corr, _ = spearmanr(attr_100_clean, attr_50_clean)
        else:
            print(f"Length mismatch for {p_id}: 100-step len {len(attr_100_clean)}, 50-step len {len(attr_50_clean)}")
            corr = 0.0
            
        stability_rows.append({
            "paragraph_id": p_id,
            "spearman_correlation": float(corr),
            "steps_a": 50,
            "steps_b": 100
        })
        
    stab_df = pd.DataFrame(stability_rows)
    stab_df.to_csv(RESULTS_DIR / "attribution_stability.csv", index=False)
    print(f"Mean Spearman Correlation: {stab_df['spearman_correlation'].mean():.4f}")

def ablation_inference(model, tokenizer, device, text, remove_indices):
    """
    To prevent BPE tokenization shifts from destroying the structural integrity
    of the sentence, we perform deletion by masking the identified Token IDs
    with the padding token ID prior to the forward pass.
    """
    inputs = tokenizer(text, truncation=True, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    # Mask out the deleted tokens
    for idx in remove_indices:
        input_ids[0, idx] = tokenizer.pad_token_id
        attention_mask[0, idx] = 0
        
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=-1).squeeze().cpu().numpy()
        
    return float(probs[1])

def test_deletion(model, tokenizer, device, examples):
    print("\\nRunning Validation Deletion Tests...")
    token_attr_df = pd.read_csv(RESULTS_DIR / "token_attributions.csv")
    
    validation_rows = []
    
    for ex in tqdm(examples, total=len(examples)):
        p_id = ex['paragraph_id']
        text = ex['text']
        original_ai_prob = ex['ai_probability']
        
        df_p = token_attr_df[token_attr_df['paragraph_id'] == p_id]
        
        # 1. Top Positive (AI supporting)
        top_pos = df_p.sort_values('attribution', ascending=False).head(5)
        top_pos_idx = top_pos['token_index'].tolist()
        top_pos_tokens = top_pos['raw_token'].tolist()
        
        new_prob_pos = ablation_inference(model, tokenizer, device, text, top_pos_idx)
        validation_rows.append({
            "paragraph_id": p_id,
            "method": "top_positive",
            "removed_tokens": ", ".join(top_pos_tokens),
            "original_ai_probability": original_ai_prob,
            "new_ai_probability": new_prob_pos,
            "probability_change": new_prob_pos - original_ai_prob
        })
        
        # 2. Top Negative (Human supporting)
        top_neg = df_p.sort_values('attribution', ascending=True).head(5)
        top_neg_idx = top_neg['token_index'].tolist()
        top_neg_tokens = top_neg['raw_token'].tolist()
        
        new_prob_neg = ablation_inference(model, tokenizer, device, text, top_neg_idx)
        validation_rows.append({
            "paragraph_id": p_id,
            "method": "top_negative",
            "removed_tokens": ", ".join(top_neg_tokens),
            "original_ai_probability": original_ai_prob,
            "new_ai_probability": new_prob_neg,
            "probability_change": new_prob_neg - original_ai_prob
        })
        
        # 3. Random Control
        valid_indices = df_p['token_index'].tolist()
        random_idx = random.sample(valid_indices, min(5, len(valid_indices)))
        random_tokens = df_p[df_p['token_index'].isin(random_idx)]['raw_token'].tolist()
        
        new_prob_rand = ablation_inference(model, tokenizer, device, text, random_idx)
        validation_rows.append({
            "paragraph_id": p_id,
            "method": "random_control",
            "removed_tokens": ", ".join(random_tokens),
            "original_ai_probability": original_ai_prob,
            "new_ai_probability": new_prob_rand,
            "probability_change": new_prob_rand - original_ai_prob
        })
        
    val_df = pd.DataFrame(validation_rows)
    val_df.to_csv(RESULTS_DIR / "attribution_validation.csv", index=False)
    
    # Print summary
    mean_drop = val_df[val_df['method'] == 'top_positive']['probability_change'].mean()
    mean_rise = val_df[val_df['method'] == 'top_negative']['probability_change'].mean()
    mean_rand = val_df[val_df['method'] == 'random_control']['probability_change'].mean()
    
    print(f"Mean AI Prob Change (Removed Top Positive): {mean_drop:+.4f} (Expected: Negative)")
    print(f"Mean AI Prob Change (Removed Top Negative): {mean_rise:+.4f} (Expected: Positive)")
    print(f"Mean AI Prob Change (Removed Random):       {mean_rand:+.4f}")

def main():
    print("Starting Saliency Validation...")
    model, tokenizer, device = load_model_and_tokenizer()
    
    with open(RESULTS_DIR / "selected_examples.json", "r") as f:
        examples = json.load(f)
        
    # Filter to only the True Positive AI ones
    ai_examples = [ex for ex in examples if ex['true_label'] == 1 and ex['predicted_label'] == 1]
    
    if not (RESULTS_DIR / "token_attributions.csv").exists():
        print("Error: token_attributions.csv not found. Run integrated_gradients.py first.")
        return
        
    test_stability(model, tokenizer, device, ai_examples)
    test_deletion(model, tokenizer, device, ai_examples)
    print("Validation tests complete.")

if __name__ == "__main__":
    main()
