# TASK 2: The Multi-Tiered Detective

This task focuses on building increasingly sophisticated machine learning models to detect AI-generated text.

## Directory Structure

### `tier_A_statistician/`
- Implements a Random Forest classifier using the statistical features established in Task 1 (e.g., Flesch-Kincaid, Type-Token Ratio, Punctuation Density).
- Provides a baseline for how much structural stylistic information alone can distinguish Human from AI text.

### `tier_B_semanticist/`
- Implements a Gradient Boosting classifier based on dense semantic embeddings (from SentenceTransformers).
- Tests the hypothesis that AI and Humans differ in semantic topicality and phrasing.

### `tier_C_transformer/`
- The most advanced detector: a `roberta-base` model fine-tuned for sequence classification using PEFT/LoRA.
- Highly accurate and robust, evaluating text directly at the token and sub-word level.
- This model's predictive power is subsequently analyzed for interpretability in Task 3.

## Evaluation Strategy
All models are rigorously evaluated against both a completely random test split and a strict `BOOK-SPLIT` test set (evaluating on texts from unseen books to ensure models learn stylistic differences rather than memorizing topics).
