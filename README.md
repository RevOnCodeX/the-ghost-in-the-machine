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
├── TASK 1-The Fingerprint/           Step 1: Measure the differences
├── TASK 2-The Multi-Tiered Detective/ Step 2: Build three detectors
├── Task 3-Smoking Gun/               Step 3: Understand why they work
├── Task 4-The Turing Test/           Step 4: Break the best one
│
├── literary_style_dataset/           The raw dataset (human + AI text)
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
| **Readability** (Flesch-Kincaid Grade) | Sentence complexity | AI writes at slightly higher complexity |
| **Punctuation Density** | Em-dashes, semicolons, commas per word | AI over-uses formal punctuation |

This task establishes the numerical baseline that Tasks 2 and 3 build on.

---

### [Task 2 — The Multi-Tiered Detective](./TASK%202-The%20Multi-Tiered%20Detective/README.md)
**Goal:** Build three progressively smarter AI detectors from scratch.

| Tier | Name | Model | Accuracy |
|---|---|---|---|
| A | The Statistician | Random Forest on Task 1 features | 81% |
| B | The Semanticist | 4-layer Neural Network on 300-dim word vectors | 89% |
| C | The Transformer | RoBERTa-base + LoRA fine-tuning | **97%** |

Each tier answers a different question: Can stylometrics alone detect AI? Do semantic embeddings help? What happens when you throw a transformer at it? The Tier C model becomes the adversarial target in Task 4.

---

### [Task 3 — The Smoking Gun](./Task%203-Smoking%20Gun/README.md)
**Goal:** Open the Tier C black box. Find out *exactly* what it is detecting.

We used **Captum Layer Integrated Gradients** to attribute every prediction back to individual tokens. The results were surprising:

- The model does **not** primarily flag "famous AI words" like *tapestry*, *delve*, or *unprecedented*.
- It flags **structural rhythm** — the mechanical uniformity of AI sentence lengths.
- Proof: masking famous AI-isms drops confidence by 0%. Masking the top 5 structural tokens drops it by 9% on average, and over 80% in specific cases.

This finding directly shapes the strategy used in Task 4.

---

### [Task 4 — The Turing Test](./Task%204-The%20Turing%20Test/README.md)
**Goal:** Use a Genetic Algorithm to evolve AI-written text until the Tier C detector classifies it as human.

- **Phase 1 (Generations 1–42):** Semantic mutations only → plateau at 7.08% human score.
- **Phase 2 (Generation 43+):** Structural disruption, deliberate typos, BPE corruption, saliency-guided mutations → **93.53% human score by Generation 79**.

The key insight: a character-level typo like `then → tehn` creates an unknown BPE sub-word split that collapses classifier confidence because the model was never trained on that pattern.

---

## The Dataset

The `literary_style_dataset/` contains paired human/AI text samples:
- **Human authors:** Jane Austen (*Pride and Prejudice*, *Sense and Sensibility*), Charles Dickens (*Great Expectations*, *Oliver Twist*)
- **AI text:** Gemma 3 12B prompted to mimic each author's style at the paragraph level
- **Size:** ~127,200 words raw, ~118,450 words after cleaning

Both sources are deliberately literary and high-quality — this is a *hard* classification problem because the AI is trying to mimic real literary prose, not generate generic filler.

---

## The Bigger Picture

This project demonstrates a fundamental tension in machine learning called **Goodhart's Law**: once you know what a model is measuring, you can optimise against it. The Tier C detector has 97% accuracy — but the moment the GA gained access to its internal gradient signals, that 97% accuracy became a detailed roadmap to evasion.

This has real implications:
- AI detection cannot rely on a single model — ensembles are necessary
- Detectors must be continuously adversarially retrained
- High confidence scores (even 97%) are not oracles; they are probability estimates

---

## How to Navigate This Repo

Each task folder has its own README explaining what was done, why, and what was found. Start with Task 1 and read sequentially — each task builds on the previous one's results.

```
Read Task 1 README → understand the features
Read Task 2 README → understand the models
Read Task 3 README → understand what the models actually learned
Read Task 4 README → understand how that knowledge was weaponised
```
