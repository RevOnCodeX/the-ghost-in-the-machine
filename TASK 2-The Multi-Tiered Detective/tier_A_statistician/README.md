# Tier A - The Statistician

This directory trains classical Machine Learning models (XGBoost and Random Forest) to detect AI-generated literature using **only** numerical fingerprint features (no text features).

## Dataset
- **Total Paragraph Pairs:** 1120 (2240 total rows)
- **Features Used (9):** `ttr`, `hapax`, `flesch_kincaid_grade`, `sentence_length`, `semicolon_density`, `em_dash_density`, `exclamation_density`, `question_density`, `comma_density`

## Training Methodology
The models were trained under two distinct evaluation modes to test for generalization:
- **Mode 1 (Random Split):** A standard 80/20 train-test split across the entire dataset.
- **Mode 2 (Book-Level Split):** Training strictly on *Oliver Twist, Emma, and Sense and Sensibility*, and evaluating on *Great Expectations, Pride and Prejudice, and A Tale of Two Cities*. This tests if the model learns generic AI artifacts rather than book-specific quirks.

## Model Comparison

| Model | Evaluation Mode | Accuracy | F1 Score | ROC-AUC |
|---|---|---|---|---|
| XGBoost | Mode 1 (Random) | 52.6% | 0.531 | 0.547 |
| XGBoost | Mode 2 (Book) | 59.7% | 0.642 | 0.643 |
| Random Forest | Mode 1 (Random) | 53.7% | 0.558 | 0.546 |
| Random Forest | Mode 2 (Book) | 58.2% | 0.637 | 0.637 |

## Feature Importance Findings
Based on the Random Forest Gini importance calculations, the most critical features separating human from AI writing were:
1. **Hapax Legomena Count (`hapax`)**: Humans naturally inject a far higher number of ultra-rare, one-time-use words.
2. **Type-Token Ratio (`ttr`)**: The AI consistently maintained a higher overall unique word ratio per paragraph.
3. **Comma Density (`comma_density`)**: Crucial for dictating sentence rhythm, which the AI frequently struggled to mimic accurately.

*Check `results/feature_importance.png` for the visual plot.*

## Limitations
The models achieve an ROC-AUC of around 0.55-0.64, meaning they are only slightly better than a random coin flip (0.50). Classical statistics alone are **not** enough to robustly detect modern Large Language Models, as the models can easily emulate surface-level metrics (like sentence length or grade level). 

Furthermore, this model detects AI imitation patterns specifically derived from this 19th-century literary dataset. It should **not** be claimed as a universal or general-purpose AI detector.
