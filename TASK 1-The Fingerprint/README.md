# Task 1: The Fingerprint

**Goal:** Before building any detector, prove that AI-generated and human-written text are *measurably different* using only statistics — no machine learning.

This is the foundation of the entire project. If there are no consistent statistical differences between human and AI text, then no classifier can reliably detect AI. Task 1 establishes that the differences are real, consistent, and quantifiable.

---

## The Core Idea

Think of it like forensic handwriting analysis. Before a detective builds a pattern-matching system, they first measure the physical properties of the handwriting — pen pressure, letter spacing, slant angle. Task 1 does the same thing for text, measuring three categories of "writing style fingerprints."

---

## What Was Measured

### 1. Lexical Richness (`lexical_richness/`)

Lexical richness measures how *diverse* someone's vocabulary is — do they reuse the same words, or do they reach for different expressions?

Two metrics were used:

**Type-Token Ratio (TTR):**
- Count the total number of words in a text (tokens)
- Count how many of those are *unique* words (types)
- TTR = unique words ÷ total words
- A higher TTR means more varied vocabulary

**Hapax Legomena:**
- Count words that appear exactly once in the text
- A higher hapax count means the writing is more lexically adventurous

**What we found:** AI-generated text has lower TTR and fewer hapax legomena — it tends to reuse words more than human authors, even when prompted to be creative. Jane Austen and Dickens are significantly more lexically diverse than the AI impersonating them.

---

### 2. Readability (`readability/`)

Readability metrics estimate how easy or difficult a text is to read, based on sentence length and word complexity.

The main metric used: **Flesch-Kincaid Grade Level**
- Based on average words per sentence and average syllables per word
- Higher score = more complex writing (higher grade level required to understand it)

**What we found:** AI-generated text consistently scores at a slightly higher grade level than the human originals. The AI tends toward longer, more formally structured sentences — even when prompted to mimic a specific author's style. Human authors, especially in 19th-century fiction, use sentence length variably and unpredictably.

---

### 3. Punctuation Density (`punctuation_density/`)

Punctuation choices are deeply habitual — they reflect how a writer naturally structures thought. The metrics measured:

| Punctuation | What it signals |
|---|---|
| Em-dashes (`—`) | Dramatic pauses, asides, emphasis |
| Semicolons (`;`) | Long compound sentences, formal structure |
| Commas (`,`) | Sentence rhythm, clause separation |
| Exclamations (`!`) | Emotional emphasis |

Each was measured as a *density* (count per 100 words) so longer texts don't automatically score higher.

**What we found:** AI text has unnaturally high comma and semicolon density compared to human authors. It also uses em-dashes at a rate that is statistically elevated. These aren't random — they reflect how autoregressive models have learned to insert syntactic structure from their training data.

---

## Directory Structure

```
TASK 1-The Fingerprint/
├── lexical_richness/
│   ├── analyze_lexical_richness.py   ← Script to compute TTR and hapax scores
│   ├── results/
│   │   ├── lexical_results.json      ← Full numerical results for every paragraph pair
│   │   └── lexical_report.md         ← Human-readable summary of findings
│   └── samples/                      ← Side-by-side paragraph pairs for manual review
│
├── readability/
│   └── (readability analysis scripts and results)
│
└── punctuation_density/
    └── (punctuation analysis scripts and results)
```

---

## Key Takeaway

Even without training a single machine learning model, the three analyses together paint a consistent picture: **AI-generated literary text is measurably more uniform, more formal, and less lexically diverse than human-written text.** These measurements directly inform what features are fed into the Task 2 classifiers.

---

## Running the Analysis

```bash
# From the root of the repository
python3 "TASK 1-The Fingerprint/lexical_richness/analyze_lexical_richness.py"
```

Requirements: `textstat`, `numpy`, `pandas` (standard data science stack)
