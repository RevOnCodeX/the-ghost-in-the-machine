# Task 4: The Turing Test — The Super-Imposter

## Overview

> **Goal:** Evolve a machine-written paragraph until a state-of-the-art AI detector classifies it as **human-authored** with >90% confidence.

This task answers a fundamental adversarial question: if you know *exactly* how a detector works, can you systematically fool it? Using a **Genetic Algorithm (GA)** with an LLM as the mutation engine and a fine-tuned **RoBERTa (Tier C)** model as the fitness oracle, we successfully crossed the 90% threshold — reaching a peak human score of **93.53%** after 79 generations.

---

## Repository Structure

```
Task 4-The Turing Test/
├── backend/
│   ├── app.py                  # FastAPI server — exposes /analyze endpoint for all 3 tiers
│   ├── super_imposter_ga.py    # Core Genetic Algorithm implementation
│   ├── feature_extractor.py    # Tier A feature extraction (stylometric features)
│   └── requirements.txt        # Python dependencies
├── ga_evolution_log.json       # Full generation-by-generation log of the GA run
└── README.md                   # This file
```

---

## The Genetic Algorithm — How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                   GA EVOLUTION LOOP                             │
│                                                                 │
│  Generation N                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Population (10 text candidates)                         │   │
│  │  → Fitness Evaluation: POST /analyze → Tier C human_prob │   │
│  │  → Selection: Keep Top 3                                 │   │
│  │  → Elitism: Best 1 passes unchanged                      │   │
│  │  → Mutation (LLM via Gemma 3 12B on AWS Bedrock):        │   │
│  │    • rhythm    — restructure sentence lengths            │   │
│  │    • grammar   — inject filler words, casual slang       │   │
│  │    • conversational — Reddit-style imperfect voice        │   │
│  │    • typo      — character-level noise injection         │   │
│  │  → Saliency Feedback: Captum LayerIG identifies          │   │
│  │    the highest-attribution "AI fingerprint" tokens       │   │
│  │    → Fed back as NEGATIVE CONSTRAINTS to next prompt     │   │
│  └──────────────────────────────────────────────────────────┘   │
│  Generation N+1                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Components

| File | Role |
|---|---|
| `super_imposter_ga.py` | GA driver: population init, fitness eval, selection, mutation loop |
| `app.py` | REST backend: runs all 3 detectors + Captum LayerIG saliency on each request |
| `feature_extractor.py` | Extracts 9 stylometric features for Tier A (TTR, Hapax, FK grade, punctuation ratios) |

---

## Key Findings & Experimental Results

### Phase 1: The Semantic Illusion (Generations 1–42)

**What we tried:** Pure LLM-driven semantic mutations. Synonym replacement, tone shifting, casual paraphrasing. All operations preserved the underlying syntactic scaffolding.

**Result:** Hard plateau at **7.08% human score**. Completely unfooled.

**Why it failed:**
Modern transformer detectors (RoBERTa + LoRA) do not evaluate bag-of-words frequencies. They embed text into high-dimensional latent space where the **AI fingerprint lives in structural predictability and syntactic rhythm** — not vocabulary. Autoregressive models produce low-variance, highly predictable token sequences. Semantic paraphrasing simply moves the text to a different point in that same predictable distribution. The detector's attention mechanisms saw straight through the lexical changes.

---

### Phase 2: Re-engineering the Mutation Engine (Generation 43+)

**The paradigm shift:** Abandon semantic attacks. Target the model's *predictive expectations* directly.

If AI text = low entropy + rhythmic uniformity, then the GA must learn to inject **high entropy and structural disruption**.

**Changes implemented:**

1. **Vocabulary Entropy Injection** — Deliberately swap high-probability "AI words" (identified by Captum saliency) for low-probability, less predictable synonyms. Artificially spikes the text's perplexity.

2. **Syntactic Fragmentation** — Strip formal punctuation, adopt fragmented conversational structure. Force the generator to emit natural filler: *"um"*, *"like"*, *"you know"*, *"kinda"*.

3. **Rhythm Disruption** — Penalise uniform sentence lengths in the fitness function. Force high Coefficient of Variation (CV) across sentence lengths.

4. **Character-level Typo Injection** — Programmatic character transpositions (e.g. `then → tehn`, `satellites → satelllites`). Each typo creates **unknown BPE tokens** that the tokenizer has never seen, directly corrupting the embedding space the detector relies on.

5. **Gradient-informed Mutations (Captum LayerIG)** — Real-time saliency heatmaps from Captum extract the exact tokens triggering the highest "AI" activation. These are fed back as *negative constraints* to the LLM prompt: *"You MUST entirely avoid these words."* This turns random mutation into **surgical adversarial strikes**.

**Result:** Fitness jumped from 7.08% → **93.53%** over 37 additional generations.

---

### The Evolution Trajectory

| Generation | Human Score | Phase | Key Event |
|---|---|---|---|
| 1 | 1.35% | 1 | Baseline — formal AI text |
| 5 | 2.84% | 1 | Filler words added |
| 20 | 6.21% | 1 | Rhythm variation attempted |
| 42 | 7.08% | 1 | **WALL — semantic plateau** |
| 43 | 8.21% | 2 | Typo injection begins |
| 45 | 22.55% | 2 | +14.3% — BPE disruption starts working |
| 48 | 42.45% | 2 | BPE tokenization cracked |
| 69 | 79.59% | 2 | Approaching 90% target |
| 76 | 81.38% | 2 | Personal best at the time |
| 79 | **93.53%** | 2 | 🏆 **TARGET ACHIEVED** |

---

### The Winning Text

```
so, um, i'm giving this another go, trying to make it sound, like, more natural, you know?

honestly these space stations—the whole setup—are way more crucial tehn most folks realize.
It's really vital for you know, connecting things and figuring out where you are, and
observing what's happening out there, really they've become a pretty big deal, impacting
loads of things we do i mean, we basically need them now, seriously.
```

**Tier A score:** 81% Human  
**Tier B score:** 86% Human  
**Tier C score:** **93.53% Human** ✅

Key adversarial features:
- `tehn` — transposition of `then` creates unknown BPE token, corrupts embedding
- `i'm`, `you know`, `like` — conversational filler destroys formal AI rhythm
- Missing capitalisation — signals stream-of-consciousness, not autoregressive generation
- Missing punctuation — breaks the high-comma-density "AI fingerprint"
- Fragmented run-on sentences — Coefficient of Variation of sentence length artificially inflated

---

### What This Reveals About AI Detection

The GA's success is itself a form of interpretability. The specific mutations that *worked* reveal exactly where RoBERTa's AI-detection signal lives:

1. **Rhythmic uniformity** — The single strongest signal. LLMs produce eerily consistent sentence lengths. Humans don't.
2. **High-probability vocabulary** — Words like *"unprecedented"*, *"endeavor"*, *"fundamentally"* are statistically over-represented in LLM outputs.
3. **BPE token familiarity** — The tokenizer's entire vocabulary is drawn from its training corpus. A deliberate typo forces an OOV sub-word split, producing an embedding pattern the classifier was never trained to recognise.
4. **Formal punctuation density** — AI text uses semicolons, em-dashes, and commas at superhuman rates.

---

## Backend API

The `app.py` FastAPI server exposes a single `/analyze` endpoint used both by the GA's fitness function and the interactive frontend.

```
POST /analyze
{
  "text": "Your text here",
  "model": "Tier C - Transformer"   // or "Tier A", "Tier B", "Compare All"
}
```

**Response:**
```json
{
  "Tier C": {
    "ai_prob": 0.0647,
    "human_prob": 0.9353,
    "attributions": [
      { "word": "tehn", "attribution": -0.92, "normalized_score": -0.92 },
      { "word": "crucial", "attribution": 0.71, "normalized_score": 0.71 }
    ]
  }
}
```

The `attributions` array is powered by **Captum Layer Integrated Gradients**, attributing each token's contribution to the AI classification logit. Positive scores = flagged as AI. Negative scores = suppresses AI detection.

---

## Running the Backend

```bash
cd "Task 4-The Turing Test/backend"

# Install dependencies
pip install -r requirements.txt

# Start the FastAPI server (requires trained models from Task 2)
uvicorn app:app --host 0.0.0.0 --port 8000

# Run the GA (requires the FastAPI server + Bedrock API access)
python super_imposter_ga.py
```

> **Note:** The GA requires AWS Bedrock access (Gemma 3 12B) configured via the `BEDROCK_KEY` in `super_imposter_ga.py`, and the trained Task 2 model files at the paths defined in `app.py`.

---

## Dependencies

```
fastapi
uvicorn
torch
transformers
peft
captum
scikit-learn
joblib
gensim
textstat
numpy
requests
```

---

## Broader Implications

This research is a direct demonstration of **Goodhart's Law in machine learning**: *"When a measure becomes a target, it ceases to be a good measure."* The moment the GA was given direct feedback from the detector's internal gradients, the detector's own decision boundary became a map of its weaknesses.

A production-grade AI detection system must therefore:
1. Be an **ensemble** — gradient-informed attacks against one model are unlikely to transfer perfectly to others.
2. Treat **confidence as probabilistic** — a 95% confidence score is not ground truth.
3. Continuously **adversarially retrain** — the mutation strategies discovered here should become future training negatives.
