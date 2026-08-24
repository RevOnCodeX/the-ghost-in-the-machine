# The Ghost in the Machine — Literary Style Dataset

This repository contains a comprehensive investigation into detecting and understanding AI-generated literary text, comparing it against human-authored classics by Jane Austen and Charles Dickens.

## Project Structure

### 1. [TASK 1: The Fingerprint](./TASK%201-The%20Fingerprint/README.md)
Establishes a statistical baseline by analyzing fundamental differences in lexical richness, readability, and punctuation density between human and AI text.

### 2. [TASK 2: The Multi-Tiered Detective](./TASK%202-The%20Multi-Tiered%20Detective/README.md)
Develops three tiers of AI text detection:
- **Tier A (Statistician):** Random Forest based on stylistic features.
- **Tier B (Semanticist):** Gradient Boosting based on semantic embeddings.
- **Tier C (Transformer):** A fine-tuned RoBERTa model using LoRA.

### 3. [TASK 3: The Smoking Gun](./Task%203-Smoking%20Gun/README.md)
An exhaustive interpretability study on the Tier C Transformer model. Employs Captum's Integrated Gradients to identify saliency mappings, investigates the influence of "AI-isms" and structural patterns, and performs rigorous error analysis through counterfactual ablation studies on false positives.

## Goals
The overarching goal is not just to accurately classify AI-generated text, but to mathematically and structurally understand *why* and *how* current Large Language Models betray their synthetic origin through specific phrasing choices and rhythmic syntax.
