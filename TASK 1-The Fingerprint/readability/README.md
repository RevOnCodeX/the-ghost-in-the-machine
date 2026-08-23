# Readability Analysis

This folder contains the paired readability analysis (Flesch-Kincaid Grade Level) between human original paragraphs and their AI-rewritten equivalents.

## Structure
- `analyze_readability.py`: Core script that parses paragraphs from the Git history to match them against AI text, calculates syllables, sentence lengths, and the Flesch-Kincaid Grade Level, and generates outputs.
- `requirements.txt`: Python dependencies (`textstat`, `pandas`).
- `results/`:
  - `readability_results.json`: Raw paired data and calculated metrics.
  - `readability_report.md`: Markdown summary highlighting author-specific differences and qualitative findings regarding AI complexity emulation.

## Usage
To rerun the analysis, execute from the repository root:
```bash
python3 "TASK 1-The Fingerprint/readability/analyze_readability.py"
```
