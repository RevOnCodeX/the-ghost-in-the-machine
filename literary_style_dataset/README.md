# The Ghost in the Machine — Literary Style Dataset

A curated dataset of human-authored literary paragraphs from Charles Dickens and Jane Austen, designed to study and classify literary style.

---

## Project Structure

```
literary_style_dataset/
│
├── raw/                          # Original unmodified Project Gutenberg texts
│   ├── dickens/
│   │   ├── oliver_twist.txt
│   │   ├── great_expectations.txt
│   │   └── tale_of_two_cities.txt
│   └── austen/
│       ├── emma.txt
│       ├── pride_and_prejudice.txt
│       └── sense_and_sensibility.txt
│
├── cleaned/                      # Cleaned texts (Gutenberg boilerplate removed)
│   ├── dickens/
│   └── austen/
│
├── lib/
│   └── summaries/
│       └── all_paragraphs_for_review.txt   # 28 paragraphs per book with topic mapping
│
└── scripts/
    └── clean_dataset.py          # Cleaning pipeline
```

---

## Authors & Books

| Author | Books |
|---|---|
| Charles Dickens | Oliver Twist, Great Expectations, A Tale of Two Cities |
| Jane Austen | Emma, Pride and Prejudice, Sense and Sensibility |

---

## Paragraph Dataset

`lib/summaries/all_paragraphs_for_review.txt` contains **28 paragraphs per book** (168 total) extracted from the cleaned texts, each to be annotated with:
- **Topic** — mapped to one of 5 core themes per book
- **Summary** — a descriptive sentence capturing meaning, tone, and theme

---

## Setup

```bash
pip install python-dotenv
cp .env.example .env
# Add your API keys to .env
```

Run the cleaning pipeline:
```bash
python scripts/clean_dataset.py
```
