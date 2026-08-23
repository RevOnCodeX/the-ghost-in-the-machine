import argparse
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel
from pathlib import Path
from scipy.special import softmax

SCRIPT_DIR = Path(__file__).resolve().parent
TIER_C_DIR = SCRIPT_DIR.parent
MODELS_DIR = TIER_C_DIR / "models/roberta_lora_detector"
BASE_MODEL = "roberta-base"

def main():
    parser = argparse.ArgumentParser(description="Live AI Text Detection with RoBERTa LoRA")
    parser.add_argument("--text", type=str, required=True, help="The paragraph of text to analyze")
    parser.add_argument("--model_mode", type=str, default="random_split", choices=["random_split", "book_split"], help="Which trained LoRA adapter to use")
    args = parser.parse_args()
    
    adapter_path = MODELS_DIR / args.model_mode
    if not adapter_path.exists():
        print(f"Error: LoRA adapter not found at {adapter_path}. Please train the model first.")
        return
        
    print(f"Loading RoBERTa base model and {args.model_mode} LoRA adapter...")
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path))
    base_model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    inputs = tokenizer(args.text, truncation=True, padding=True, max_length=512, return_tensors="pt").to(device)
    
    print("\\nAnalyzing text...")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
        
    prob_ai = probs[1]
    prob_human = probs[0]
    
    if prob_ai > 0.5:
        prediction = "AI"
        confidence = prob_ai * 100
    else:
        prediction = "Human"
        confidence = prob_human * 100
        
    print(f"\\n==============================")
    print(f"Prediction: {prediction}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"==============================\\n")

if __name__ == "__main__":
    main()
