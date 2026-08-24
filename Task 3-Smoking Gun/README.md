# Task 3: The Smoking Gun

This directory contains interpretability and error analysis studies for the Tier C RoBERTa + LoRA AI text detector. The goal is to answer the fundamental question: *Why does the model predict text as AI-generated?*

## Components

### 1. `saliency/`
Uses Captum's `LayerIntegratedGradients` to interrogate the Tier C model and determine token-level attributions. It extracts the top positive (AI-supporting) and negative (Human-supporting) tokens and phrases for correct predictions.

### 2. `findings/`
A rigorous investigation into *what* the model relies on:
- **AI-ism Frequency & Enrichment:** Tests whether stereotypical "AI vocabulary" (like *tapestry*, *delve*, *testament*) is statistically enriched in AI text.
- **Rhythm Analysis:** Explores sentence length variability and punctuation density, concluding that AI models possess an unusually consistent and homogeneous rhythmic syntax compared to human authors.
- **Ablation Studies:** Proves causality by stripping suspected "famous AI-isms" vs. structurally salient tokens, proving that the model relies on broader rhythmic/syntactic features rather than simple vocabulary tricks.

### 3. `error_analysis/`
Focuses exclusively on False Positives (Human texts classified as AI).
- Identifies anomalies in human texts (e.g., highly rhythmic sentences or repetitive phrases).
- Compares misclassified human text locally against correctly classified human and AI texts.
- Leverages counterfactual testing to isolate the exact phrases responsible for the error.

## Notebooks
Interactive and visual summaries for all these components can be found in the `notebooks/` directory.
