# The Ghost Machine: Literary Style Dataset

This repository contains a reproducible NLP dataset consisting of novels by Charles Dickens and Jane Austen, sourced from Project Gutenberg. It is specifically designed for literary stylometry and human-vs-AI writing classification tasks.

## 📂 Repository Structure

```text
literary_style_dataset/
├── raw/
│   ├── dickens/
│   │   ├── oliver_twist.txt
│   │   ├── great_expectations.txt
│   │   └── tale_of_two_cities.txt
│   └── austen/
│       ├── pride_and_prejudice.txt
│       ├── sense_and_sensibility.txt
│       └── emma.txt
│
├── cleaned/               # Generated automatically by the script
│   ├── dickens/
│   └── austen/
│
├── scripts/
│   └── clean_dataset.py   # The data-cleaning pipeline
│
└── README.md
```

## 📖 Raw Data

The `raw/` directory contains the original, untouched `.txt` files directly downloaded from Project Gutenberg. These files contain the full text of the novels, but they also include Project Gutenberg-specific artifacts such as:
- Boilerplate license and copyright headers/footers.
- Transcribers' notes.
- Hard-wrapped lines (text constrained to ~72 characters per line).

The raw files serve as the immutable source of truth for the dataset.

## 🧹 Cleaning Process

The provided data-cleaning pipeline (`scripts/clean_dataset.py`) processes the raw texts to make them usable for Natural Language Processing (NLP) while remaining entirely faithful to the author's original writing. 

**The script performs the following cleaning steps:**
1. **Boilerplate Removal**: 
   - Automatically identifies and strips out the "START OF THE PROJECT GUTENBERG EBOOK" and "END OF..." markers. All subsequent legal, update, and distribution boilerplate after the end marker is completely removed.
   - Removes Gutenberg image placeholders (e.g., `[Illustration]`) while preserving any embedded chapter headings or actual literary text.
   - Automatically identifies and removes the entire Gutenberg Table of Contents.
   - Strips unnecessary front matter, title/author metadata, and prefaces so that the file starts directly with the first actual chapter of the novel.
2. **Formatting Normalization**: 
   - Removes invalid or non-standard control characters.
   - Normalizes line endings to standard Unix format (`\n`).
   - "Un-wraps" the text by joining single line breaks within paragraphs into continuous sentences, removing the artificial ~72 character limit imposed by Project Gutenberg.
   - Consolidates accidental repeated whitespaces.

## ✍️ Stylistic Preservation

**This dataset is intended for literary stylometry.** Therefore, the goal of the cleaning script is to remove dataset artifacts, *not* to simplify or alter the author's prose. 

**The following elements are strictly preserved:**
- **Punctuation & Capitalization**: Dashes, semicolons, exclamation marks, apostrophes, and capitalization patterns are left completely intact. These are critical features (stylomes) for distinguishing between authors (e.g., Dickens's liberal use of capitalization or Austen's specific sentence structures).
- **Linguistic Nuances**: Dialect, contractions, archaic spellings (e.g., "ha'", "wittles", "know'd", "partickler"), and unusual vocabulary are entirely preserved.
- **Sentence & Paragraph Boundaries**: The script distinguishes between hard-wrapped lines and true paragraph breaks (double newlines), ensuring the structural flow of the novel remains exactly as the author intended.
- **Dialogue**: Quotation marks and dialogue formatting are untouched, as dialogue-to-prose ratios are highly distinctive authorial signatures.
- **Chapter Headings**: The actual chapter headings in the body of the novel are kept intact to allow for later document segmentation or chapter-by-chapter analysis.

The script intentionally avoids aggressive preprocessing techniques:
- **No** lowercasing.
- **No** stopword removal.
- **No** stemming or lemmatization.
- **No** removal of punctuation.
- **No** rewriting or modernizing of language.

These stylistic signals are absolutely crucial for human-vs-AI and author attribution tasks.

## 🚀 How to Run the Pipeline

The cleaning script is written in standard Python and requires no external dependencies. 

To run the pipeline and generate the `cleaned/` directory:

1. Open your terminal.
2. Navigate to the root of the dataset directory (`literary_style_dataset/`).
3. Run the script:
   ```bash
   python3 scripts/clean_dataset.py
   ```

The script will automatically detect the `.txt` files in `raw/dickens/` and `raw/austen/`, process them, and output the cleaned versions with a `_cleaned.txt` suffix into the `cleaned/` directory. It also provides a console report detailing character counts, word counts, and lines removed/merged for validation.

To add a new book to the dataset, simply drop the raw `.txt` file into the appropriate `raw/` subdirectory and rerun the script.
