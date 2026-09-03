# The Ghost in the Machine

> *Can a machine learn to write so convincingly that a state-of-the-art AI detector classifies it as human?*

This repository documents a complete end-to-end research project investigating AI text detection and adversarial evasion. Over four progressive tasks, it builds a three-tier detection system from scratch, interprets its internal reasoning using saliency methods, and then weaponises that knowledge to systematically fool the detector using a Genetic Algorithm.

**Final result: a Genetic Algorithm evolved a machine-written paragraph to a 93.53% human score over 79 generations.**

---

## Project Structure

```
the-ghost-in-the-machine/
├── TASK 1-The Fingerprint/         # Statistical baseline analysis
├── TASK 2-The Multi-Tiered Detective/  # Three AI detectors (RF, NN, RoBERTa)
├── Task 3-Smoking Gun/             # Interpretability: Captum saliency + error analysis
├── Task 4-Web Interface/           # The Turing Test: GA evasion + FastAPI backend
│   ├── backend/
│   │   ├── super_imposter_ga.py    # Genetic Algorithm core
│   │   ├── app.py                  # FastAPI inference server
│   │   ├── feature_extractor.py    # Stylometric feature extraction
│   │   └── requirements.txt
│   ├── ga_evolution_log.json       # Full generation-by-generation run log
│   └── README.md
├── literary_style_dataset/         # Human (Austen, Dickens) + AI text corpus
└── .gitignore
```

---

## The Four Tasks

### [Task 1 — The Fingerprint](./TASK%201-The%20Fingerprint/README.md)
Establishes a statistical baseline comparing human-authored classics (Jane Austen, Charles Dickens) against AI-generated text across three dimensions:
- **Lexical Richness:** Type-Token Ratio (TTR) and Hapax Legomena density
- **Readability:** Flesch-Kincaid Grade, average sentence length
- **Punctuation Density:** Em-dash, semicolon, comma, and exclamation ratios

**Finding:** AI text exhibits unnaturally high TTR uniformity and over-uses formal punctuation (em-dashes, semicolons) relative to human authors.

---

### [Task 2 — The Multi-Tiered Detective](./TASK%202-The%20Multi-Tiered%20Detective/README.md)
Builds three progressively sophisticated AI detectors:

| Tier | Name | Architecture | Accuracy |
|---|---|---|---|
| A | The Statistician | Random Forest on 9 stylometric features | 81% |
| B | The Semanticist | 4-layer MLP on 300-dim FastText vectors | 89% |
| C | The Transformer | RoBERTa-base + LoRA fine-tuning (PEFT) | **97%** |

**Finding:** The Tier C transformer achieves near-human classification accuracy, confirming that transformer architectures can reliably detect syntactic and rhythmic fingerprints left by LLMs.

---

### [Task 3 — The Smoking Gun](./Task%203-Smoking%20Gun/README.md)
Interpretability analysis of the Tier C model using Captum Layer Integrated Gradients (LayerIG):
- Identified the exact tokens triggering AI classification (`"unprecedented"`, `"endeavor"`, `"fundamentally"`)
- Ablation studies on false positives reveal the model over-weights Gutenberg formatting artefacts
- Counterfactual experiments confirm that structural rhythm is a stronger signal than vocabulary

**Finding:** The detector is not classifying *meaning* — it is classifying *rhythm and token-level predictability*.

---

### [Task 4 — The Turing Test](./Task%204-The%20Turing%20Test/README.md)
A Genetic Algorithm evolves AI-written text to evade the Tier C detector:

- **Phase 1 (Generations 1–42):** Semantic mutations. Hard plateau at **7.08% human score**.
- **Phase 2 (Generation 43+):** Structural disruption — typo injection, BPE corruption, saliency-informed negative constraints.
- **Final result (Generation 79):** **93.53% human score** ✅

**Key insight:** Deliberate character-level typos create unknown BPE sub-word tokens, corrupting the embedding space the detector depends on. When combined with Captum-guided surgical mutation, this breaks the detector's confidence completely.

---

## Dataset

The `literary_style_dataset/` contains paired human/AI text samples across:
- **Human:** Jane Austen (*Pride and Prejudice*, *Sense and Sensibility*), Charles Dickens (*Great Expectations*, *Oliver Twist*)
- **AI:** Gemma 3 12B IT generations prompted to mimic each author's style
- **Size:** ~127,200 words raw, ~118,450 cleaned

---

## Broader Implications

This project is a demonstration of **Goodhart's Law in ML**: the moment a metric becomes a target, it ceases to be a reliable measure. When the GA was given direct feedback from the detector's internal gradients, the detector's own decision boundary became a map of its weaknesses.

Robust AI detection requires:
1. **Ensemble approaches** — gradient attacks against one model don't fully transfer
2. **Adversarial retraining** — mutations that worked here should become training negatives
3. **Treating confidence probabilistically** — a 97% accurate classifier is not a ground-truth oracle

---

## Setup

Each task has its own `requirements.txt`. For Task 4 backend:

```bash
cd "Task 4-Web Interface/backend"
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
python super_imposter_ga.py
```
