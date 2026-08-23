# Punctuation Density Analysis

This folder contains the paired punctuation density analysis between human original paragraphs and their AI-rewritten equivalents.

## Structure
- `analyze_punctuation.py`: Core script that parses paragraphs from the Git history to match them against AI text, calculates punctuation counts/densities, and generates outputs.
- `requirements.txt`: Python dependencies (`matplotlib`, `seaborn`) for heatmap generation.
- `results/`:
  - `punctuation_results.json`: Raw paired data and calculated metrics.
  - `punctuation_heatmap.png`: Matplotlib frequency density heatmap.
  - `punctuation_report.md`: Markdown summary highlighting author-specific differences and qualitative findings.

## Usage
To rerun the analysis, execute from the repository root:
```bash
python3 "TASK 1-The Fingerprint/punctuation_density/analyze_punctuation.py"
```
