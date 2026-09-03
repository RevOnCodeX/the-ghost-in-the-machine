# Task 2: The Multi-Tiered Detective

**Goal:** Build three AI detectors — each one more sophisticated than the last — to classify paragraphs as either AI-generated or human-written.

The three-tier approach is intentional. Each tier uses a fundamentally different method to answer the same question, which lets us understand *what kind of information* is actually useful for detection.

---

## Why Three Detectors?

A single classifier gives you a result. Three classifiers with different architectures give you *insight*. If Tier A (basic statistics) achieves 81% accuracy and Tier C (a transformer model) achieves 97%, that gap tells you how much richer the signal is when you look at full contextual word embeddings rather than just surface statistics. The tiers are a ladder of understanding.

---

## Tier A — The Statistician (`tier_A_statistician/`)

**Model:** Random Forest Classifier  
**Accuracy:** 81%

### What it uses
The Tier A classifier is trained on the 9 features established in Task 1:

| Feature | Description |
|---|---|
| Type-Token Ratio (TTR) | Vocabulary diversity |
| Hapax Legomena count | Words appearing exactly once |
| Flesch-Kincaid Grade | Reading level / sentence complexity |
| Average sentence length | Words per sentence |
| Semicolon density | Semicolons per 100 words |
| Em-dash density | Em-dashes per 100 words |
| Exclamation density | Exclamations per 100 words |
| Question mark density | Questions per 100 words |
| Comma density | Commas per 100 words |

### Why Random Forest?
Random Forest is ideal here because these features are heterogeneous (some are ratios, some are raw counts, some are on very different scales). Random Forest handles this naturally without needing feature scaling, and it is robust to the small dataset size.

### What it tells us
An 81% accuracy from pure stylometric features is strong. It proves that the *surface statistics* of writing alone carry significant discriminative power. But 19% error rate also tells us there is information the statistics are missing — which is exactly what Tier B and C address.

---

## Tier B — The Semanticist (`tier_B_semanticist/`)

**Model:** 4-layer Multilayer Perceptron (Neural Network)  
**Input:** 300-dimensional FastText word vectors  
**Accuracy:** 89%

### What it uses
Instead of hand-crafted features, Tier B converts each paragraph to a dense vector by averaging the FastText embeddings of all its words. This vector captures *semantic meaning* — words used in similar contexts cluster together in the 300-dimensional space.

The neural network then learns a non-linear boundary between "AI-sounding" and "human-sounding" regions of this semantic space.

### Architecture
```
Input: 300-dim word vector
→ Linear(300 → 256) → ReLU → Dropout(0.3)
→ Linear(256 → 128) → ReLU → Dropout(0.3)
→ Linear(128 → 64)  → ReLU
→ Linear(64 → 1)    → Sigmoid
Output: probability of being AI-generated
```

### What it tells us
The jump from 81% to 89% shows that semantic meaning adds real signal beyond surface statistics. AI text doesn't just *look* different — it *means* things differently. The vocabulary, phrasing, and topic distribution of AI-generated literary prose is distinguishable from human literary prose at the semantic level.

---

## Tier C — The Transformer (`tier_C_transformer/`)

**Model:** `roberta-base` fine-tuned with LoRA (Low-Rank Adaptation)  
**Accuracy:** **97%**

### What it uses
RoBERTa is a large pre-trained transformer model that understands text at the sub-word token level. Instead of averaging word vectors, it processes the *full sequence* of tokens and builds contextual representations — the meaning of each word informed by every other word around it.

We fine-tuned it using **LoRA (PEFT)**: rather than retraining all 125M parameters, we add small rank-16 adapter matrices to the attention layers. This makes fine-tuning efficient while preserving the model's general language understanding.

### Why this approach works so well
Transformers naturally capture:
- **Long-range dependencies** — patterns across the full paragraph, not just word-by-word
- **Sub-word tokenisation** — sensitivity to spelling, morphology, punctuation
- **Contextual meaning** — the same word means different things in different contexts

The 97% accuracy represents near-ceiling performance. The remaining 3% error is analysed in detail in Task 3.

### Evaluation: The Book-Split Test
All tiers are evaluated on two test sets:
1. **Random split** — paragraphs from all books, randomly held out
2. **Book-split** — *entire books* held out from training

The book-split is the harder and more honest test. It checks whether the model learns to distinguish *writing styles* (what we want) rather than memorising *specific books* (what we don't want). Both evaluations are reported.

---

## Directory Structure

```
TASK 2-The Multi-Tiered Detective/
│
├── tier_A_statistician/
│   ├── scripts/          ← Training and evaluation code
│   ├── models/           ← Saved Random Forest model (randomforest_model.pkl)
│   └── results/          ← Accuracy scores, confusion matrices, feature importances
│
├── tier_B_semanticist/
│   ├── scripts/          ← Training and evaluation code
│   ├── embeddings/       ← FastText model (fasttext.model)
│   ├── models/           ← Saved neural network weights (semantic_nn.pt)
│   └── results/          ← Training history, accuracy scores
│
└── tier_C_transformer/
    ├── scripts/          ← LoRA fine-tuning and evaluation code
    ├── models/           ← Saved RoBERTa + LoRA adapter (roberta_lora_detector/)
    └── results/          ← Per-class accuracy, confusion matrix, book-split results
```

---

## How to Run

```bash
# Tier A
python "TASK 2-The Multi-Tiered Detective/tier_A_statistician/scripts/train.py"

# Tier B
python "TASK 2-The Multi-Tiered Detective/tier_B_semanticist/scripts/train.py"

# Tier C
python "TASK 2-The Multi-Tiered Detective/tier_C_transformer/scripts/train_lora.py"
```

Requirements: `scikit-learn`, `torch`, `transformers`, `peft`, `gensim`, `textstat`

---

## Key Takeaway

| Tier | Method | Accuracy | What it proved |
|---|---|---|---|
| A | Hand-crafted statistics | 81% | Surface style is discriminative |
| B | Semantic word vectors | 89% | Meaning and vocabulary distribution matter |
| C | Transformer (token-level) | 97% | Deep contextual patterns are the real signal |

The accuracy progression shows that the further you look *inside* the text (from surface → meaning → deep context), the more signal you find. This motivates Task 3: if Tier C is so accurate, what exactly is it seeing?
