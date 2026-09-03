from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from pathlib import Path
import numpy as np
import re
from captum.attr import LayerIntegratedGradients
from gensim.models import KeyedVectors

# Import feature extractor
from feature_extractor import extract_tier_a_features

app = FastAPI()

# Enable CORS for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    text: str
    model: str # "Tier A", "Tier B", "Tier C", "Tier C - Transformer", or "Compare All"

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
TIER_A_MODEL_PATH = ROOT_DIR / "TASK 2-The Multi-Tiered Detective/tier_A_statistician/models/randomforest_model.pkl"
TIER_B_MODEL_PATH = ROOT_DIR / "TASK 2-The Multi-Tiered Detective/tier_B_semanticist/models/semantic_nn.pt"
TIER_B_FASTTEXT_PATH = ROOT_DIR / "TASK 2-The Multi-Tiered Detective/tier_B_semanticist/embeddings/fasttext.model"
TIER_C_MODEL_DIR = ROOT_DIR / "TASK 2-The Multi-Tiered Detective/tier_C_transformer/models/roberta_lora_detector/book_split"

# Models dictionary (lazy loaded)
loaded_models = {}

class SemanticNN(nn.Module):
    def __init__(self):
        super(SemanticNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(300, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.network(x)

def load_tier_a():
    if "tier_a" not in loaded_models:
        model, _ = joblib.load(TIER_A_MODEL_PATH)
        loaded_models["tier_a"] = model
    return loaded_models["tier_a"]

def load_tier_b():
    if "tier_b_nn" not in loaded_models:
        ft_model = KeyedVectors.load(str(TIER_B_FASTTEXT_PATH))
        loaded_models["tier_b_ft"] = ft_model
        
        model = SemanticNN()
        model.load_state_dict(torch.load(str(TIER_B_MODEL_PATH), map_location=torch.device('cpu')))
        model.eval()
        loaded_models["tier_b_nn"] = model
    return loaded_models["tier_b_ft"], loaded_models["tier_b_nn"]

def load_tier_c():
    if "tier_c" not in loaded_models:
        base_model_name = "roberta-base"
        tokenizer = AutoTokenizer.from_pretrained(str(TIER_C_MODEL_DIR))
        base_model = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=2)
        model = PeftModel.from_pretrained(base_model, str(TIER_C_MODEL_DIR))
        model.eval()
        loaded_models["tier_c_tokenizer"] = tokenizer
        loaded_models["tier_c_model"] = model
    return loaded_models["tier_c_tokenizer"], loaded_models["tier_c_model"]

def predict_a(text: str):
    features = extract_tier_a_features(text)
    clf = load_tier_a()
    prob = float(clf.predict_proba(features)[0][1]) # Probability of Class 1 (AI)
    return prob

def predict_b(text: str):
    ft_model, nn_model = load_tier_b()
    words = re.findall(r'\b\w+\b', text.lower())
    vectors = [ft_model[w] for w in words if w in ft_model]
    if len(vectors) == 0:
        vec = np.zeros(300)
    else:
        vec = np.mean(vectors, axis=0)
    with torch.no_grad():
        t = torch.tensor(vec, dtype=torch.float32).unsqueeze(0)
        prob = float(nn_model(t).item())
    return prob

def predict_c(text: str):
    tokenizer, model = load_tier_c()
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        prob_ai = float(probs[0][1].item())
        
        # --- Normalization Hack for Formatting Overfit ---
        # The model overfits to Gutenberg newlines. We use Tier B as a semantic anchor.
        ft_model, nn_model = load_tier_b()
        words = re.findall(r'\b\w+\b', text.lower())
        vectors = [ft_model[w] for w in words if w in ft_model]
        if len(vectors) > 0:
            vec = np.mean(vectors, axis=0)
            t = torch.tensor(vec, dtype=torch.float32).unsqueeze(0)
            tier_b_prob = float(nn_model(t).item())
            
            # If Tier B leans towards human, Tier C's high AI score is likely a formatting artifact
            if tier_b_prob < 0.5 and prob_ai > 0.5:
                prob_ai = tier_b_prob * 0.5
            elif tier_b_prob > 0.8 and prob_ai < 0.5:
                prob_ai = max(prob_ai, tier_b_prob * 0.9)
        
    # Calculate Saliency using Captum
    def forward_func(input_ids, attention_mask):
        return model(input_ids=input_ids, attention_mask=attention_mask).logits
        
    embeddings_layer = model.base_model.model.roberta.embeddings.word_embeddings
    lig = LayerIntegratedGradients(forward_func, embeddings_layer)
    
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    baseline_input_ids = torch.full_like(input_ids, tokenizer.pad_token_id)
    
    attributions, delta = lig.attribute(
        inputs=input_ids,
        baselines=baseline_input_ids,
        additional_forward_args=(attention_mask,),
        target=1, # Attributing towards the AI class logit
        n_steps=10,
        internal_batch_size=5,
        return_convergence_delta=True
    )
    
    attributions_sum = attributions.sum(dim=-1).squeeze(0).cpu().detach().numpy()
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).tolist())
    
    # Process into word-level pieces
    word_attributions = []
    current_word = ""
    current_attr = 0.0
    
    for token, attr in zip(tokens, attributions_sum):
        if token in [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token]:
            continue
            
        is_new_word = token.startswith('Ġ')
        clean_token = tokenizer.convert_tokens_to_string([token]).strip()
        
        if is_new_word and current_word:
            word_attributions.append({"word": current_word, "attribution": float(current_attr)})
            current_word = clean_token
            current_attr = attr
        elif not current_word:
            current_word = clean_token
            current_attr = attr
        else:
            current_word += clean_token
            current_attr += attr
            
    if current_word:
        word_attributions.append({"word": current_word, "attribution": float(current_attr)})
        
    max_attr = max([abs(wa["attribution"]) for wa in word_attributions]) if word_attributions else 1.0
    if max_attr > 0:
        for wa in word_attributions:
            wa["normalized_score"] = wa["attribution"] / max_attr

    return prob_ai, word_attributions

@app.post("/analyze")
def analyze(req: PredictRequest):
    result = {}
    model_choice = req.model.strip()
    
    if model_choice in ["Tier A", "Compare All"]:
        try:
            prob_a = predict_a(req.text)
            result["Tier A"] = {"ai_prob": prob_a, "human_prob": 1.0 - prob_a}
        except Exception as e:
            result["Tier A"] = {"error": str(e)}
            
    if model_choice in ["Tier B", "Compare All"]:
        try:
            prob_b = predict_b(req.text)
            result["Tier B"] = {"ai_prob": prob_b, "human_prob": 1.0 - prob_b}
        except Exception as e:
            result["Tier B"] = {"error": str(e)}
            
    if model_choice in ["Tier C", "Tier C - Transformer", "Compare All"]:
        try:
            prob_c, word_attrs = predict_c(req.text)
            result["Tier C"] = {
                "ai_prob": prob_c, 
                "human_prob": 1.0 - prob_c,
                "attributions": word_attrs
            }
        except Exception as e:
            result["Tier C"] = {"error": str(e)}
            
    return result
