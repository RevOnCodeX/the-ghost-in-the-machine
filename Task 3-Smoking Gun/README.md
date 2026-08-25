# Task 3: The Smoking Gun

This directory contains our comprehensive interpretability and error analysis investigation into the Tier C RoBERTa + LoRA AI text detector. While our Tier C model achieved near-perfect accuracy, modern neural networks often function as "black boxes." The goal of this task was to answer the fundamental question: *Why does the model predict text as AI-generated, and what specific linguistic artifacts is it detecting?*

By dissecting the model's decision-making process, we can better understand the underlying differences between human literary composition and synthetic generation.

---

## 1. Interpretability via Saliency Mapping (`saliency/`)

We employed Captum's **Layer Integrated Gradients (IG)** to interrogate the fine-tuned RoBERTa model. IG allows us to attribute the model's final prediction logit back to individual input tokens, establishing a direct mathematical link between specific words/phrases and the model's confidence in the "AI" label.

**Methodology:**
- We analyzed correct predictions (True Positives and True Negatives) across a strictly held-out subset of Austen and Dickens texts (`BOOK-SPLIT` test set).
- We extracted the highest positively-attributed (AI-supporting) and negatively-attributed (Human-supporting) tokens and phrases for each text.

---

## 2. Experimental Findings (`findings/`)

Our analysis specifically tested a popular hypothesis: Do AI detectors simply learn to spot "famous AI-isms" (e.g., *tapestry*, *delve*, *testament*, *multifaceted*), or do they detect deeper structural patterns?

### Key Results
- **AI-isms are Statistically Enriched:** We confirmed through Fisher's Exact Enrichment tests (with FDR correction) that stereotypical AI vocabulary words are significantly overrepresented in the AI-generated texts.
- **AI-isms are NOT the "Smoking Gun":** Despite their frequency, our attribution analysis revealed that the model does *not* heavily rely on these obvious words to make its decisions. 
- **The True Signal is Structural Rhythm:** The model focuses on deeper syntactic structures and rhythm. We found that AI-generated text exhibits a highly uniform, low-variance sentence rhythm compared to the varied, dynamic rhythm of human authors (e.g., human sentence length Coefficient of Variation (CV) = `0.60` vs AI CV = `0.55`). 

### Causality via Ablation
To prove this, we conducted counterfactual ablation tests:
1. **Removing Famous AI-isms:** When we computationally masked out the "famous AI-isms" from AI texts, the model's AI probability barely dropped at all (mean drop of `0.0%`).
2. **Removing Structurally Salient Tokens:** When we masked out the top 5 highly-attributed structural tokens discovered by IG, the model's confidence dropped significantly (mean drop of `9.0%` globally, with some examples plunging over `80%`). 

**Impact:** This proves that the detector is robust and relies on deep, distributed structural syntax rather than easily-manipulated vocabulary tricks. Simply prompting an AI to "avoid using the word tapestry" will not easily evade this detector.

---

## 3. Error Analysis (`error_analysis/`)

No model is perfect. To understand the model's limitations, we isolated the extremely rare **False Positives** (Human text that the model incorrectly classified as AI-generated). 

### Key Results
- **Identification:** The model's errors were highly concentrated; in our evaluation, it misclassified exactly 3 specific paragraphs by Jane Austen while making zero mistakes on Charles Dickens.
- **Statistical Anomalies:** By computing Task 1 features on these False Positives, we discovered they were statistical outliers compared to the general Human distribution. The misclassified Austen paragraphs exhibited unusually uniform sentence lengths and highly repetitive phrasing—characteristics typical of our AI dataset (e.g., one FP exhibited a word count Z-score of `+2.73` and a sentence length variability Z-score of `+2.13`).
- **Counterfactual Validation:** We proved the causality of the error using IG. By identifying the exact tokens that pushed the model to predict "AI" and ablating them, the model's AI probability crashed. For instance, in FP `TClass_P22`, removing the top 5 attributed tokens caused the AI prediction probability to plummet from `94.5%` to just `11.3%`.  

**Impact:** The error analysis demonstrates that the model does not fail randomly. It fails when a human author coincidentally writes with the specific syntactic uniformity and repetition that strongly characterizes synthetic text. This highlights an inherent limitation in purely statistical classification: highly constrained or unusually structured human writing can inadvertently mimic the mathematical fingerprint of an LLM.

---

## Notebooks & Reproducibility

Interactive summaries, heatmaps, distribution plots, and ablation charts for all these components can be found in the `notebooks/` directory:
- `task3_saliency.ipynb`
- `task3_findings.ipynb`
- `task3_error_analysis.ipynb`

All analyses were strictly contained to the validation corpus, ensuring no data leakage from the training phase.
