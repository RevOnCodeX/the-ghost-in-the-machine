# Tier C - The Transformer

This tier constructs a powerful contextual AI text detector using a pre-trained **RoBERTa-base** transformer, fine-tuned using **LoRA (Low-Rank Adaptation)**.

## Why RoBERTa?
In Tier B, the semantic Feedforward Neural Network merely averaged word embeddings, completely losing the sequence and syntax of sentences. RoBERTa has deep contextual understanding. Since our dataset contains complex Victorian English, long sentences, and nuanced literary patterns, we needed a model capable of full self-attention to read *between* the words.

## LoRA Strategy
To prevent our GitHub repository from exploding with 500MB checkpoints, we froze the base `roberta-base` weights and only fine-tuned the `query` and `value` attention matrices via PEFT LoRA (rank = 16). This results in an incredibly lightweight adapter (~1MB) that can be instantly injected into the base model at runtime.

## Dataset
- **Size:** 1,121 perfectly matched pairs (2,242 rows total).
- **Features:** Only raw text. No statistical markers (TTR, syllable count) or manual embeddings.

## Training Details
- **Hardware:** MPS / CPU
- **Epochs:** 3 per experiment
- **Batch Size:** 8 (Gradient Accumulation = 4)
- **Optimizer:** AdamW (LR=2e-4)

## Final Results

| Experiment | Accuracy | F1 Score | ROC-AUC |
|------------|----------|----------|---------|
| Random Split | 91.1% | 0.914 | 0.985 |
| Book Split | 96.4% | 0.965 | 0.994 |

## The Multi-Tiered Conclusion

Does the modern transformer completely outperform handcrafted features? **Absolutely.**

- **Tier A (Statistical Fingerprint):** Achieved an abysmal **~60% ROC-AUC**. The AI was completely capable of matching syllable counts, punctuation density, and sentence lengths.
- **Tier B (Semantic Embedding):** Reached a respectable **~76% ROC-AUC**. The AI revealed modern biases in its vocabulary choices, which a simple Neural Network could detect.
- **Tier C (Transformer):** Achieved a near-perfect **99.4% ROC-AUC**. By combining semantic choices with *contextual sequencing and grammatical syntax structure*, RoBERTa was able to identify AI rewrites with extreme precision. Even when generalizing across entirely different books and authors, the AI's internal "style fingerprint" was completely exposed by self-attention mechanisms.
