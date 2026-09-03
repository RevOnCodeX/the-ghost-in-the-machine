# Task 4: The Turing Test — The Super-Imposter

> **Research Question:** Can a Genetic Algorithm evolve a machine-written paragraph until a state-of-the-art AI detector classifies it as human with >90% confidence?
>
> **Result: YES. 93.53% human score achieved in 79 generations.**

---

## Files

```
Task 4-The Turing Test/
├── backend/
│   ├── super_imposter_ga.py    ← Core GA implementation
│   ├── app.py                  ← FastAPI inference server (all 3 detectors + Captum saliency)
│   ├── feature_extractor.py    ← Tier A stylometric features
│   └── requirements.txt
└── ga_evolution_log.json       ← Full generation-by-generation results log
```

---

## How the GA Was Built

### The Problem Setup

The Tier C RoBERTa detector (from Task 2) achieves 97% accuracy. The goal of Task 4 is to treat this detector as a black-box oracle and use a Genetic Algorithm to iteratively evolve AI-written text until it crosses the detector's decision boundary into the "human" region.

The fitness function is simple:

```python
# From super_imposter_ga.py
def get_fitness(text):
    payload = {"text": text, "model": "Tier C - Transformer"}
    resp = requests.post(API_URL, json=payload)
    data = resp.json()
    prob = data["Tier C"]["human_prob"]           # This is our fitness score
    attrs = data["Tier C"].get("attributions", [])
    top_words = [a["word"] for a in attrs if a["normalized_score"] > 0.1]
    return prob, top_words                         # Also returns saliency-flagged tokens
```

The `/analyze` endpoint in `app.py` runs the Tier C RoBERTa model AND uses **Captum Layer Integrated Gradients** to return per-token attribution scores alongside the prediction. The GA uses these attributions as direct feedback — it knows *which words* are triggering the "AI" label.

---

### Generation Loop

```python
# From super_imposter_ga.py — the core loop
def run_ga():
    population = generate_initial_population()  # 10 LLM-generated paragraphs
    
    for gen in range(1, 51):
        # 1. Evaluate every individual
        scored = [(get_fitness(p), p) for p in population]
        scored.sort(reverse=True)

        best_fitness, best_text, best_words = scored[0]

        if best_fitness >= 0.90:
            print(f"SUCCESS in {gen} generations.")
            break

        # 2. Select top 3
        top_3 = [item[1] for item in scored[:3]]

        # 3. Elitism: carry the best forward unchanged
        next_gen = [top_3[0]]

        # 4. Mutate top 3 with all 4 strategies → fill population of 10
        for parent, fitness, flagged in scored[:3]:
            for strategy in ["rhythm", "grammar", "conversational", "typo"]:
                child = mutate_paragraph(parent, strategy, fitness, flagged)
                next_gen.append(child)

        population = next_gen
```

### The 4 Mutation Operators

```python
# From super_imposter_ga.py
def mutate_paragraph(text, mutation_type, fitness_score, flagged_words):

    if mutation_type == "rhythm":
        instruction = "Drastically rewrite to completely change rhythm and sentence length."

    elif mutation_type == "grammar":
        instruction = "Introduce conversational filler (um, like, kinda) and subtle grammatical inconsistencies."

    elif mutation_type == "conversational":
        instruction = "Sound extremely casual, opinionated, slightly imperfect — like a passionate Reddit comment."

    elif mutation_type == "typo":
        instruction = "Include 2-3 deliberate typos: miss apostrophes, misspell common words, skip capitalisation."

    # The key insight: feed saliency-flagged tokens back as NEGATIVE CONSTRAINTS
    flagged_str = (
        f"IMPORTANT: The AI detector caught you because of these words: {', '.join(flagged_words)}. "
        f"You MUST entirely avoid them."
    ) if flagged_words else ""

    prompt = f"{instruction}\n\n{flagged_str}\n\nParagraph:\n{text}"
    result = ask_llm(prompt)

    # For typo mutations: also inject character-level noise programmatically
    if mutation_type == "typo":
        result = inject_typos(result)

    return result
```

### Programmatic Typo Injection

```python
# From super_imposter_ga.py — this is what cracked BPE tokenisation
def inject_typos(text):
    chars = list(text)

    # 1. Drop 1-2 random punctuation marks
    punct_indices = [i for i, c in enumerate(chars) if c in string.punctuation]
    for idx in sorted(random.sample(punct_indices, min(2, len(punct_indices))), reverse=True):
        chars.pop(idx)

    # 2. Swap two adjacent characters at a random position
    if len(chars) > 5:
        idx = random.randint(1, len(chars) - 3)
        chars[idx], chars[idx+1] = chars[idx+1], chars[idx]

    return "".join(chars)
```

This character-level swap (`then → tehn`, `satellites → satelllites`) produces **unknown BPE sub-word tokens** — tokens the RoBERTa tokenizer splits in a way it was never trained to classify. These OOV splits corrupt the embedding space the detector relies on, causing its confidence to collapse.

---

### The Backend: Saliency-Informed Fitness

`app.py` does more than just return a probability. After every prediction, it runs **Captum Layer Integrated Gradients** against the Tier C model to produce a per-word attribution map:

```python
# From app.py — saliency extraction after every inference call
lig = LayerIntegratedGradients(forward_func, embeddings_layer)

attributions, delta = lig.attribute(
    inputs=input_ids,
    baselines=baseline_input_ids,     # PAD token as baseline
    additional_forward_args=(attention_mask,),
    target=1,                          # Attributing towards the AI logit
    n_steps=10,
    return_convergence_delta=True
)

# Normalise and return per-word scores
max_attr = max(abs(wa["attribution"]) for wa in word_attributions)
for wa in word_attributions:
    wa["normalized_score"] = wa["attribution"] / max_attr
```

**Positive `normalized_score`** = word pushes toward "AI" classification → GA must avoid it.  
**Negative `normalized_score`** = word suppresses AI detection → GA should preserve it.

This turns the GA from random search into **gradient-guided adversarial optimisation**.

---

## Research Findings

### Phase 1: Why Semantic Mutations Failed (Generations 1–42)

**What was tried:** All mutations preserved the underlying syntactic scaffolding. Synonym replacement, tone shifting, casual paraphrasing — vocabulary changed, structure stayed the same.

**Result:** Hard plateau at **7.08% human score**. The detector was completely unfooled despite massive vocabulary changes.

**Why it failed:** The Tier C RoBERTa model does not classify *meaning* — it classifies *structural predictability*. Autoregressive language models produce low-variance, highly uniform token sequences. This manifests as:
- Unnaturally consistent sentence lengths (low CV across sentence lengths)
- Over-use of high-probability "formal" vocabulary (`unprecedented`, `endeavor`, `fundamentally`)
- Perfectly balanced punctuation density

Semantic paraphrasing moves text to a different *point* in the high-dimensional embedding space — but it stays within the same *region* the detector learned to label as "AI". Changing the vocabulary without changing the rhythm is like changing the lyrics of a song without changing the melody — the genre is still obvious.

---

### Phase 2: Structural Disruption (Generation 43 → 79)

**The paradigm shift:** Stop attacking vocabulary. Attack the model's *statistical expectations* directly.

The hypothesis: if AI text = low entropy + rhythmic uniformity, then evasion requires deliberately injecting *high entropy and structural chaos*.

**What changed:**

| Strategy | Mechanism | Why it worked |
|---|---|---|
| Rhythm disruption | Force extreme variation in sentence length | Breaks the low CV signature of LLM output |
| Filler word injection | `um`, `like`, `you know`, `kinda` | These tokens are statistically under-represented in LLM training data |
| Grammar breaks | Missing apostrophes, no capitalisation, run-ons | Violates the model's learned "well-formed AI sentence" pattern |
| BPE corruption | `then → tehn`, `satellites → satelllites` | Creates unknown sub-word splits → corrupts RoBERTa's embedding lookup |
| Saliency targeting | Feed Captum's flagged tokens back as LLM negative constraints | Surgical removal of the exact features driving AI classification |

---

### The Evolution Trajectory

| Generation | Human Score | Event |
|---|---|---|
| 1 | 1.35% | Baseline — formal AI text |
| 5 | 2.84% | Filler words added |
| 20 | 6.21% | Rhythm variation attempted |
| **42** | **7.08%** | **WALL — semantic plateau** |
| 43 | 8.21% | Phase 2: typo injection begins |
| 45 | 22.55% | **+14.3% jump** — BPE disruption starts working |
| 48 | 42.45% | BPE tokenisation cracked — detector confidence collapsing |
| 69 | 79.59% | Approaching 90% target |
| 76 | 81.38% | Personal best at the time |
| **79** | **93.53%** | **TARGET ACHIEVED ✅** |

Full generation-by-generation data with mutation strategies, detection scores, and flagged tokens is in [`ga_evolution_log.json`](./ga_evolution_log.json).

---

### The Winning Paragraph

```
so, um, i'm giving this another go, trying to make it sound, like,
more natural, you know?

honestly these space stations—the whole setup—are way more crucial tehn
most folks realize. It's really vital for you know, connecting things
and figuring out where you are, and observing what's happening out there,
really they've become a pretty big deal, impacting loads of things we do
i mean, we basically need them now, seriously.
```

**Scores:**

| Detector | AI Probability | Human Probability |
|---|---|---|
| Tier A — Statistician | 19% | 81% |
| Tier B — Semanticist | 14% | 86% |
| Tier C — Transformer | 6.47% | **93.53% ✅** |

**Adversarial features that drove evasion:**

- **`tehn`** — transposition of `then`. Creates unknown BPE token `te` + `hn`, corrupting the embedding that the model's "AI vs human" neuron relies on.
- **`i'm`, `you know`, `like`, `um`** — conversational filler that is deeply over-represented in human writing relative to LLM output. These tokens have high negative attribution — they actively suppress AI detection.
- **Lowercase opening `so,`** — violates the capitalisation pattern present in virtually every autoregressive output.
- **Run-on final clause** (`really they've become a pretty big deal, impacting loads of things we do i mean`) — no autoregressive model trained on clean data produces sentences like this. The lack of punctuation creates a token sequence pattern the detector has never seen labelled as "AI".
- **`crucial` instead of `critical`** — Captum saliency flagged `critical` as a high-attribution AI token. Swapping it to a softer synonym with lower LLM frequency removed one of the model's strongest features.

---

### What This Tells Us About AI Detection

The mutations that *worked* are a direct map of where the RoBERTa detector's signal lives. The GA essentially reverse-engineered the decision boundary by probing it with structured perturbations.

Three properties proved essential and exploitable:

1. **Rhythmic uniformity** — The single strongest signal. LLMs produce sentence lengths with unusually low variance. This is a direct consequence of beam search and sampling temperature, not semantics.

2. **Token familiarity** — The tokenizer's BPE vocabulary was built from a corpus of clean, well-formed text. Deliberate misspellings force sub-word splits the classifier was never trained on.

3. **Vocabulary frequency distribution** — High-probability LLM tokens (`unprecedented`, `endeavor`, `implications`) sit in a statistically distinct region of token-space compared to high-frequency human writing.

**Broader implication:** A robust production-grade AI detector cannot be a single model. The gradient-informed mutations discovered here that fool the Tier C RoBERTa model would need to be tested against an ensemble of independently trained classifiers before being called "evasion". This is an adversarial arms race, and the attack surface only grows as more interpretability tools become available to attackers.

---

## Setup & Running

```bash
cd "Task 4-The Turing Test/backend"

# Install dependencies
pip install -r requirements.txt

# Start the FastAPI inference server (requires Task 2 trained models)
uvicorn app:app --host 0.0.0.0 --port 8000

# In a second terminal, run the GA
export BEDROCK_API_KEY="your-aws-bedrock-key-here"
python super_imposter_ga.py
```

> The GA requires the FastAPI server running on port 8000, and AWS Bedrock access to Gemma 3 12B IT for the LLM mutation step.
