# Task 1: The Fingerprint - Paired Lexical Richness Analysis

This sub-project analyzes the lexical richness of the dataset by pairing original human paragraphs with their direct AI-rewritten counterparts.

## Directory Structure
- `lexical_richness/analyze_lexical_richness.py`: The analysis script that pairs paragraphs using the `Paragraph N` ID by fetching the original texts from Git history.
- `lexical_richness/results/`: Output files including the master `lexical_results.json` and a human-readable `lexical_report.md` summary.
- `lexical_richness/samples/`: Two text files containing the paired paragraphs laid out sequentially for manual review.

## Running the Analysis
If you want to re-run the analysis, execute the following from the root of the repository:
```bash
python3 "TASK 1-The Fingerprint/lexical_richness/analyze_lexical_richness.py"
```
