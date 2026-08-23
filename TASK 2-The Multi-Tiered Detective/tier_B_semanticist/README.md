# Tier B - The Semanticist

This tier builds a semantic-only AI detector using pre-trained **FastText Word Embeddings** (wiki-news-subwords-300) and a **PyTorch Feedforward Neural Network**.

The objective of this tier is to see if the **meaning and vocabulary choice** alone can identify AI-generated rewritten text, even when the underlying story/topic remains exactly the same as the human original.

## Dataset & Embeddings
- **Paragraph Pairs:** 1121 exact matching pairs (2242 total rows).
- **FastText Model:** `fasttext-wiki-news-subwords-300` (automatically downloaded via Gensim API).
- **Embedding Generation:** Each word in the paragraph is tokenized and its 300-dimensional FastText vector is retrieved. All vectors are averaged to produce a single 300-d representation per paragraph.

## Neural Network Architecture
The PyTorch Feedforward Neural Network uses 4 layers:
- **Input:** 300 dimensions
- **Layer 1:** Linear(300, 256) → ReLU → Dropout(0.3)
- **Layer 2:** Linear(256, 128) → ReLU → Dropout(0.3)
- **Layer 3:** Linear(128, 64) → ReLU
- **Output:** Linear(64, 1) → Sigmoid

## Training Configuration
- **Optimizer:** Adam
- **Learning Rate:** 0.001
- **Loss:** Binary Cross Entropy
- **Epochs:** 50
- **Batch Size:** 32

## Model Performance

| Split | Accuracy | F1 Score | ROC-AUC |
|---|---|---|---|
| Random Split | 71.9% | 0.655 | 0.782 |
| Book Split | 72.3% | 0.694 | 0.762 |

*(Note: Book split trained on Oliver Twist, Emma, Sense and Sensibility, Great Expectations; tested on Pride and Prejudice, A Tale of Two Cities).*

## Analysis: Tier B (Semantic) vs Tier A (Statistic)

**Is semantic information alone sufficient to distinguish AI generated text?**
Yes, it is significantly better than statistical counts.

In Tier A, the XGBoost/Random Forest models evaluating readability, unique word counts, and punctuation density barely achieved **~0.55 to 0.64 ROC-AUC**. They essentially failed to differentiate AI from Humans because the AI successfully mirrored the statistical profile (sentence length, syllables) of the human input.

In Tier B, using purely semantic embeddings, the Neural Network achieved a **0.78 ROC-AUC** (Random Split) and a **0.76 ROC-AUC** (Book Split). 

**Conclusion:** 
While the AI was instructed to rewrite the paragraphs maintaining the original style and topic, it failed to perfectly emulate the Victorian authors' exact *choice* of semantic vocabulary. The AI inevitably relies on modern latent vocabulary distributions when constructing sentences, creating a distinct "semantic clustering" that the Feedforward Neural Network was able to easily exploit. The high performance on the Book Split proves this semantic artifact generalizes across entirely different novels and authors.
