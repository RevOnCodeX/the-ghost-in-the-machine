# The Ghost in the Machine

**Can a machine learn to write so convincingly that a state-of-the-art AI detector classifies it as human?**

This project answers that question across four progressive tasks. It starts from scratch — no pre-built AI detection tools — and builds everything by hand: a dataset, three detectors, an interpretability layer, and finally an adversarial Genetic Algorithm that systematically fools the best detector.

**Final result: a GA-evolved paragraph scored 93.53% human confidence against a fine-tuned RoBERTa model. It took 79 generations.**

---

## What this project is about

Modern language models (LLMs) like GPT, Gemini, or Gemma generate text that *looks* human but has subtle statistical fingerprints. This project investigates:
- What those fingerprints actually are (Tasks 1–3)
- Whether they can be deliberately erased using AI-guided evolution (Task 4)

The entire pipeline — dataset curation, feature engineering, model training, interpretability, and adversarial evasion — was built from the ground up.

---

## Project Map

```
the-ghost-in-the-machine/
│
├── TASK 1-The Fingerprint/            Step 1: Measure the differences statistically
├── TASK 2-The Multi-Tiered Detective/ Step 2: Build three progressively smarter detectors
├── Task 3-Smoking Gun/                Step 3: Open the black box — why do they work?
├── Task 4-The Turing Test/            Step 4: Use that knowledge to break the best one
│
├── literary_style_dataset/            The raw dataset (human + AI text)
└── .gitignore
```

---

## The Four Tasks at a Glance

### [Task 1 — The Fingerprint](./TASK%201-The%20Fingerprint/README.md)

**Goal:** Statistically prove that AI-generated and human-written text are measurably different — *before* training any model.

We compared text from Jane Austen and Charles Dickens against AI-generated text (Gemma 3 12B prompted to mimic each author) across three dimensions:

| Feature | What it measures | Finding |
|---|---|---|
| **Lexical Richness** (TTR, Hapax) | Vocabulary diversity | AI uses narrower, more repetitive vocab |
| **Readability** (Flesch-Kincaid Grade) | Sentence complexity | AI writes at slightly higher formal complexity |
| **Punctuation Density** | Em-dashes, semicolons, commas per word | AI over-uses formal punctuation |

This establishes the measurable baseline that Tasks 2 and 3 build on.

---

### [Task 2 — The Multi-Tiered Detective](./TASK%202-The%20Multi-Tiered%20Detective/README.md)

**Goal:** Build three progressively smarter AI detectors from scratch and understand *why* each one performs better than the last.

| Tier | Name | Model | Accuracy |
|---|---|---|---|
| A | The Statistician | Random Forest on Task 1 features | 81% |
| B | The Semanticist | 4-layer Neural Network on 300-dim word vectors | 89% |
| C | The Transformer | RoBERTa-base + LoRA fine-tuning | **97%** |

The accuracy gap between tiers is not random — it reflects how much information each approach can access from the text. See the **"Why each tier performs differently"** section below for the full breakdown.

---

### [Task 3 — The Smoking Gun](./Task%203-Smoking%20Gun/README.md)

**Goal:** Open the Tier C black box. Find out *exactly* what the 97%-accurate model is detecting.

We used **Captum Layer Integrated Gradients** to attribute every prediction back to individual tokens. The results were surprising:

- The model does **not** primarily flag "famous AI words" like *tapestry*, *delve*, or *unprecedented*
- It flags **structural rhythm** — the mechanical uniformity of sentence lengths in AI text
- Proof: masking famous AI-isms drops confidence by **0%**. Masking the top 5 structural tokens drops it by **9% on average, up to 80%+ in specific cases**

This finding directly shapes the evasion strategy used in Task 4.

---

### [Task 4 — The Turing Test](./Task%204-The%20Turing%20Test/README.md)

**Goal:** Use a Genetic Algorithm to evolve AI-written text until the Tier C detector classifies it as human with >90% confidence.

- **Phase 1 (Generations 1–42):** Semantic vocabulary mutations → hard plateau at **7.08% human score**
- **Phase 2 (Generation 43+):** Structural disruption, BPE-level typo corruption, saliency-guided targeted mutations → **93.53% human score by Generation 79** ✅

Key insight: a character-level typo like `then → tehn` creates an unknown BPE sub-word token — a pattern RoBERTa was never trained to classify — collapsing its confidence completely.

---

## Why Each Model Performed Differently — A Deep Dive

This is the core learning of the project. The 16-percentage-point jump from Tier A (81%) to Tier C (97%) is not just "transformers are better." Each tier fails for a specific, principled reason that reveals something fundamental about the problem.

---

### Tier A — Random Forest (81%) — Why it works and why it's limited

**How it works:**

A Random Forest trains hundreds of decision trees, each on a random subset of the data and features. Every tree learns rules like: *"if TTR < 0.62 AND comma density > 0.08, predict AI."* At prediction time, all trees vote and the majority wins. This ensemble approach makes it robust to noise and overfitting.

**Why it achieves 81%:**

The 9 Task 1 features (TTR, Hapax, Flesch-Kincaid grade, sentence length, and 5 punctuation densities) encode real signal. AI text genuinely has lower TTR and higher punctuation density. Random Forest is excellent at exploiting these kinds of structured, hand-crafted features because it can learn non-linear interactions between them (e.g., *low TTR AND high comma density* is a stronger predictor than either alone).

**Why it can't go higher:**

The fundamental problem is that these 9 numbers are a *summary* of the paragraph — they discard almost all the information. Two paragraphs with identical TTR, sentence length, and punctuation density could be completely different in their actual content, phrasing, and word choices. Random Forest only sees the summary, not the text.

Additionally, these stylometric features are easy to manipulate. If you prompt an LLM to "write with more diverse vocabulary and shorter sentences," a determined adversary can immediately reduce the model's advantage. The features are too coarse and too transparent.

**The trade-off:**

| ✅ Strengths | ❌ Limitations |
|---|---|
| Fast to train (seconds) | Sees only 9 numbers, not the actual text |
| Interpretable (feature importances) | Features are easily gamed |
| Works with small datasets | Cannot capture word-level or phrase-level patterns |
| No GPU needed | Hard ceiling around 82-84% on this problem |

---

### Tier B — Neural Network on FastText Embeddings (89%) — Why it improves and where it falls short

**How it works:**

FastText is a word embedding model trained on massive text corpora. It represents every word as a 300-dimensional vector in a geometric space, where words used in similar contexts are placed near each other. `"king"` and `"queen"` are close. `"unprecedented"` and `"transformative"` are close. `"um"` and `"kinda"` are close.

To represent a whole paragraph, we average the vectors of all its words. This gives us a single 300-dimensional "semantic fingerprint" of the paragraph. The 4-layer neural network then learns a non-linear decision boundary between the "AI text" region and the "human text" region of this 300-dimensional space.

**Why it jumps from 81% to 89%:**

The neural network is seeing actual *words*, not just summary statistics. It learns that paragraphs containing dense clusters of formal, abstract vocabulary (`"advancement"`, `"implications"`, `"transformative"`, `"unprecedented"`) sit in a different region of embedding space than paragraphs with concrete, varied vocabulary. This is a strictly richer signal than TTR or comma density.

The 4-layer architecture with Dropout regularisation also allows it to learn subtle non-linear patterns — combinations of semantic features that individually might not predict anything, but together are discriminative.

**Why it can't reach higher:**

FastText averaging loses *all word order information*. The paragraph *"The cat sat on the mat. A dog ran by."* and the paragraph *"A dog sat on the cat. The mat ran by."* would produce exactly the same averaged vector. Syntax — the arrangement of words into sentences and clauses — is completely invisible to this model.

This matters enormously here because, as Task 3 reveals, the key signal is *structural rhythm*: how sentence lengths vary, how clauses are arranged, how punctuation creates pacing. Averaging word vectors destroys all of this.

There is also a vocabulary coverage problem: FastText was trained on a general corpus, so rare words or proper nouns may have low-quality embeddings, introducing noise.

**The trade-off:**

| ✅ Strengths | ❌ Limitations |
|---|---|
| Captures semantic meaning and vocabulary distribution | Loses all word order and sentence structure |
| Learns non-linear feature interactions | Averaging loses compositional information |
| More resistant to simple vocabulary swaps than Tier A | Sensitive to OOV (out-of-vocabulary) words |
| Relatively fast to train | Cannot detect rhythmic or syntactic patterns |

---

### Tier C — RoBERTa + LoRA (97%) — Why this is the ceiling and what makes it so powerful

**How RoBERTa works:**

RoBERTa (Robustly Optimised BERT Pre-training Approach) is a large transformer model with 125 million parameters, pre-trained on 160GB of text. Its core mechanism is **self-attention**: for every token in the input, it computes a weighted sum of all other tokens, learning *which words to pay attention to* when interpreting each word.

The key difference from FastText: RoBERTa processes the *entire sequence at once, in order*. The meaning of every token is shaped by every other token around it. It understands that `"not good"` is semantically opposite to `"good"`, that a semicolon mid-sentence signals a compound structure, that a comma followed by *"and"* has different implications than a comma followed by *"but"*.

It also uses **sub-word tokenisation (BPE)**. Instead of treating `"running"` as one unit, it might split it into `["run", "##ning"]`. This means it can handle rare words, morphological variations, and even typos by decomposing them into known sub-units.

**How LoRA makes fine-tuning efficient:**

Full fine-tuning of RoBERTa would require updating all 125M parameters on our small literary dataset — expensive, prone to catastrophic forgetting, and likely to overfit. **LoRA (Low-Rank Adaptation)** inserts small, trainable rank-16 matrices alongside the existing attention weight matrices. During fine-tuning, only these tiny adapters are updated. The pre-trained weights stay frozen.

This gives us:
- Efficient training (only ~0.5% of parameters are trained)
- The model retains its general language understanding from pre-training
- The adapters specialise it for our specific AI vs. human detection task

**Why it reaches 97%:**

RoBERTa sees things neither Random Forest nor FastText can:

1. **Sentence-level rhythm** — because it processes sequences, it can detect when sentence after sentence follows a similar length pattern
2. **Sub-word patterns** — it detects that certain morphological constructions (e.g., multi-clause nominal phrases ending in `-tion` or `-ity` words) are more common in LLM output
3. **Long-range dependencies** — a word at the start of a paragraph can influence how a word at the end is interpreted
4. **Punctuation in context** — a semicolon's significance depends on what comes before and after it, not just its count

**What Captum revealed about *why* it's so accurate (Task 3 finding):**

When we ran Integrated Gradients attribution, we found that the model's primary signal is **syntactic rhythm** — specifically, the uniformity of sentence lengths across a paragraph. AI-generated text has a coefficient of variation (CV) of sentence lengths ≈ 0.55. Human literary text has CV ≈ 0.60. This small but consistent difference is distributed across the entire paragraph, and a transformer with full sequential attention can detect it at inference time without being explicitly told to look for it.

**The trade-offs:**

| ✅ Strengths | ❌ Limitations |
|---|---|
| Processes full sequence with positional context | Expensive — requires GPU for training and fast inference |
| Sub-word tokenisation handles rare words and typos | BPE vocabulary is finite — a new character transposition can create OOV splits (exploited in Task 4) |
| Captures rhythm, syntax, and semantics simultaneously | 125M params — needs LoRA or full fine-tuning on task-specific data |
| Pre-trained on massive text corpus — strong priors | Can overfit to formatting artefacts (e.g., Gutenberg boilerplate headers) |
| Near-ceiling accuracy on this problem | Decision boundary is attackable via gradient-guided adversarial methods |

**Why LoRA + Captum is the optimal combination:**

LoRA solves the training efficiency problem: we get a task-specific model without retraining 125M parameters on a small dataset. Captum solves the interpretability problem: we can attribute every prediction back to individual tokens, turning the black box into a diagnostic tool. Together, they give us a model that is both practical (trainable on a consumer GPU) and transparent enough to analyse and eventually defeat in Task 4.

---

### The Architecture Progression — Summary

```
Tier A: Random Forest
  Sees:    9 summary statistics about the whole paragraph
  Misses:  Every actual word, every phrase, all word order
  Limit:   81% — statistics are discriminative but coarse and gameable

         ↓ Adding: the actual words and their meanings

Tier B: Neural Network (FastText)
  Sees:    The semantic meaning of vocabulary (300-dim word vectors)
  Misses:  Word order, sentence structure, positional context
  Limit:   89% — semantic signal is real but composition is invisible

         ↓ Adding: full sequence processing, position, context, sub-word tokens

Tier C: RoBERTa + LoRA
  Sees:    Every token in context of every other token, rhythm, syntax
  Misses:  Nothing accessible from plain text
  Limit:   97% — the residual 3% is human text that coincidentally writes
                  like an LLM (see Task 3 error analysis)

         ↓ Revealed by Captum: the key signal is sentence-length uniformity

Task 4: GA Evasion
  Strategy: Disrupt rhythm via BPE corruption and structural fragmentation
  Result:   93.53% human score — the detector's own gradients become its weakness
```

---

## The Dataset

The `literary_style_dataset/` contains paired human/AI text samples:
- **Human authors:** Jane Austen (*Pride and Prejudice*, *Sense and Sensibility*), Charles Dickens (*Great Expectations*, *Oliver Twist*)
- **AI text:** Gemma 3 12B prompted to mimic each author's style at the paragraph level
- **Size:** ~127,200 words raw, ~118,450 words after cleaning

Both sources are deliberately literary and high-quality — a *hard* classification problem because the AI is trying to mimic real literary prose, not generate generic filler.

---

## The Bigger Picture

This project demonstrates a fundamental tension in machine learning called **Goodhart's Law**: once you know what a model is measuring, you can optimise against it. The Tier C detector has 97% accuracy — but the moment the GA gained access to its internal gradient signals, that 97% accuracy became a detailed roadmap to evasion.

Real implications:
- AI detection cannot rely on a single model — ensembles are necessary
- Detectors must be continuously adversarially retrained on examples of evasion
- High confidence scores (even 97%) are probability estimates, not ground truth oracles
- The same interpretability tools that make models trustworthy also make them attackable

---

## How to Navigate This Repo

Each task folder has its own README. Read them sequentially — each task builds directly on the previous one's results.

```
Task 1 README → understand the statistical fingerprints
Task 2 README → understand the three detectors and their architectures
Task 3 README → understand what the best model actually learned
Task 4 README → understand how that knowledge was weaponised to break it
```
