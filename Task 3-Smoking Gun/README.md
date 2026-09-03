# Task 3: The Smoking Gun

**Goal:** Open the Tier C black box. Find out *exactly* what the 97%-accurate RoBERTa model is actually detecting — not just that it works, but *why* it works.

This matters because understanding a model's reasoning is the only way to know when it will fail, and it directly arms Task 4 with the knowledge needed to evade it.

---

## The Problem with Black-Box Models

The Tier C detector achieves 97% accuracy, but that number alone tells us nothing about its *reasoning*. Is it picking up on specific "AI words" like *tapestry* or *unprecedented*? Is it sensitive to sentence rhythm? Does it read the overall structure of a paragraph? Without knowing, we can't trust it, explain it, or defend against someone trying to fool it.

This task uses **model interpretability** to answer the question: given a specific paragraph, which words are driving the AI/human decision?

---

## Method: Captum Layer Integrated Gradients (LayerIG)

**Integrated Gradients** is a formal attribution method from the interpretability literature. Here is the intuition:

1. Start with a "blank" input — a paragraph of all padding tokens (no information)
2. Gradually interpolate from the blank input to the real input
3. At each step, measure the gradient of the output (AI probability) with respect to each token's embedding
4. Integrate (sum) these gradients across all steps

The result is a **score for every single token** in the input:
- **Positive score** → this word pushes the model toward predicting "AI"
- **Negative score** → this word pushes the model toward predicting "human"
- **Score near zero** → this word doesn't affect the prediction much

This gives us a detailed map of what the model is paying attention to. We used **Captum's `LayerIntegratedGradients`** applied specifically to RoBERTa's word embedding layer.

---

## 1. Saliency Mapping (`saliency/`)

We ran LayerIG on a large held-out set of true positives (AI text correctly classified as AI) and true negatives (human text correctly classified as human) from the book-split test set.

### What we found

**Tokens that strongly push toward "AI" classification:**
- Words like `"essential"`, `"endeavor"`, `"profound"`, `"unprecedented"`, `"innovation"`, `"implications"`, `"tapestry"`, `"delve"`
- But more importantly: **punctuation and structural tokens** — specifically, semicolons and comma-joined compound clauses
- Long sequences of high-probability, low-entropy token transitions — positions in the sentence where the next word was almost predictable

**Tokens that strongly push toward "human" classification:**
- Idiomatic, informal phrases
- Unusual or unexpected word choices
- Sentence endings that break an expected pattern
- Short, fragmented clauses mixed with longer ones

---

## 2. The Main Finding: It's Not the "AI Words" (`findings/`)

There is a popular assumption that AI detectors work by flagging stereotypical "AI-isms" — words like *tapestry*, *delve*, *multifaceted*, *testament*. Task 3 rigorously tested this assumption and found it is **mostly wrong**.

### The experiment

We ran **Fisher's Exact Test with FDR (Benjamini-Hochberg) correction** to determine whether known "AI-ism" words appear more often in AI text than in human text.

**Confirmed:** Yes — famous AI-ism words are statistically enriched in AI-generated text. The model *has* seen this pattern.

**Then we ran ablation tests:**

> *What happens to the model's AI confidence if we mask out (remove) all the famous AI-ism words from an AI text?*

| What was removed | Average confidence drop |
|---|---|
| Famous AI-isms (`tapestry`, `delve`, etc.) | **0.0%** — no meaningful change |
| Top 5 high-attribution structural tokens (from LayerIG) | **9.0% average drop**, up to **80%+ in specific cases** |

### What this proves

The model does **not** rely on individual famous AI words to make its decision. Those words are correlated with AI text in the training data, but the model has learned deeper, more distributed structural patterns that are harder to remove.

The real signal is **syntactic rhythm** — specifically, the uniformity of sentence lengths in AI text:

- **Human writing:** Sentence length varies dramatically. Short punchy sentences. Very long winding clauses that meander and double back on themselves. Fragments. Mid-sentence interruptions.
- **AI writing:** Sentence lengths cluster around a mean. The variation is low. The Coefficient of Variation (CV) of sentence lengths in AI text ≈ **0.55** vs. human CV ≈ **0.60** — a small but consistent difference that the transformer detects reliably.

This means you cannot evade the Tier C detector simply by avoiding specific words. You have to disrupt the *rhythm*.

---

## 3. Error Analysis (`error_analysis/`)

Even a 97%-accurate model makes mistakes. Understanding *where* it fails is as important as understanding where it succeeds.

### What we found

The model's errors were highly concentrated — it misclassified **exactly 3 paragraphs** by Jane Austen (false positives: human text labeled as AI) and made **zero errors on Dickens**.

### Why those specific Austen paragraphs?

We computed the Task 1 stylometric features on the 3 false positives and compared them to the general human distribution:

- All 3 had unusually **uniform sentence lengths** (very low CV — similar to AI)
- All 3 had **repetitive phrasing patterns** — Austen occasionally writes with highly parallel grammatical structures (e.g., listing character attributes in equal-length clauses)
- One false positive had a word count Z-score of **+2.73** and a sentence length variability Z-score of **+2.13** — statistically extreme outliers in the human distribution

These paragraphs write *like AI* not because they are AI, but because Austen's prose in those sections happens to share the statistical signature of LLM output.

### Confirming causality with LayerIG

For the most confidently wrong false positive (`TClass_P22`, scored 94.5% AI):
- We identified the top 5 tokens with the highest positive attribution (pushing toward "AI")
- We ablated (masked) those tokens
- The AI probability dropped from **94.5% → 11.3%**

This confirmed that the error was *caused* by those specific tokens and structural patterns — not noise or randomness.

### The broader lesson

The model fails when a human writes in a way that coincidentally mimics the statistical signature of an LLM. This is an inherent limitation of any statistical classifier: it classifies *patterns*, not *intent*. A highly structured, formally written human text can look exactly like AI output to any detector.

---

## Directory Structure

```
Task 3-Smoking Gun/
│
├── saliency/               ← LayerIG attribution extraction scripts and outputs
├── findings/               ← AI-ism enrichment tests, ablation results
├── error_analysis/         ← False positive identification and per-example ablation
├── results/                ← Summary tables and aggregate statistics
├── notebooks/
│   ├── task3_saliency.ipynb        ← Saliency heatmaps and token attribution visualisations
│   ├── task3_findings.ipynb        ← Enrichment tests, CV analysis, ablation charts
│   └── task3_error_analysis.ipynb  ← False positive deep-dives
├── experiment_config.json  ← Hyperparameters and dataset paths
└── requirements.txt        ← captum, transformers, peft, scipy, matplotlib
```

---

## How to Run

```bash
pip install -r "Task 3-Smoking Gun/requirements.txt"

# Run saliency extraction
python "Task 3-Smoking Gun/saliency/extract_attributions.py"

# Run AI-ism enrichment tests
python "Task 3-Smoking Gun/findings/test_aiism_enrichment.py"

# Run error analysis
python "Task 3-Smoking Gun/error_analysis/analyze_false_positives.py"

# Or explore interactively via the notebooks
jupyter notebook "Task 3-Smoking Gun/notebooks/"
```

---

## Key Takeaway

| Question | Answer |
|---|---|
| Does the model flag specific "AI words"? | No — removing them drops confidence by 0% |
| What does it actually detect? | Structural rhythm — sentence length uniformity |
| Why does it sometimes fail? | When human writing coincidentally has LLM-like rhythm |
| How does failure manifest? | Only 3 Austen paragraphs misclassified, all structural outliers |

This finding directly informs Task 4: to fool the Tier C detector, you must disrupt its rhythm signal — not just change the vocabulary.
