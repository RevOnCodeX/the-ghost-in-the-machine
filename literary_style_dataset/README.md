# The Ghost in the Machine — Literary Style Dataset

A curated dataset of human-authored literary paragraphs from Charles Dickens and Jane Austen, paired with an AI-generated parallel dataset, designed to study and classify literary style.

---

## Project Structure

```text
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
├── cleaned/                      # Cleaned texts (Gutenberg boilerplate removed, paragraphs numbered)
│   ├── dickens/
│   └── austen/
│
├── topics/                       # Thematic datasets (50 paragraphs per theme per book)
│   ├── dickens/
│   └── austen/
│
├── scripts/
│   ├── clean_dataset.py          # Cleaning pipeline to strip Gutenberg text and add numbering
│   ├── generate_topics.py        # AI script to extract 5 themes per book and 50 paragraphs per theme
│   ├── rewrite_topics.py         # Single-threaded AI script to rewrite topics
│   └── rewrite_topics_parallel.py# High-throughput multi-key parallel AI rewrite script
```

---

## Authors & Books

| Author | Books |
|---|---|
| Charles Dickens | Oliver Twist, Great Expectations, A Tale of Two Cities |
| Jane Austen | Emma, Pride and Prejudice, Sense and Sensibility |

---

## The Dataset

The dataset was generated in three major phases:

### 1. Cleaning (`cleaned/`)
The raw Gutenberg texts were stripped of boilerplate (TOC, licensing, illustrations) using `scripts/clean_dataset.py`. Each paragraph in the text was explicitly numbered (e.g. `[Paragraph 1]`) to allow for exact referencing.

### 2. Thematic Extraction (`topics/`)
Using `scripts/generate_topics.py`, we queried an LLM to identify **5 major thematic topics per book**. The LLM then scanned the numbered texts and extracted exactly **50 real paragraphs** from the original book that strongly relate to each topic. 
This created 30 unique topic files, each containing 50 original paragraphs.

### 3. Style Rewriting (Parallelization)
Using `scripts/rewrite_topics_parallel.py`, each of the 1,500 extracted paragraphs was passed back to an LLM. The LLM was instructed to **completely rewrite** the paragraph from scratch, maintaining the semantic meaning, but perfectly mimicking the highly distinct literary style of the original author (Austen or Dickens). 
These rewritten files are saved back into the `topics/` folder and marked with `# (AI Rewritten)` at the top of the text files.

*Note: The parallel rewriting script (`scripts/rewrite_topics_parallel.py`) uses a round-robin strategy across multiple API keys using a `ThreadPoolExecutor` to bypass rate limits.*

---

## Setup

```bash
pip install python-dotenv requests google-genai
cp .env.example .env
# Add your API keys to .env
```

Run the cleaning pipeline:
```bash
python3 scripts/clean_dataset.py
```

Extract thematic paragraphs:
```bash
python3 scripts/generate_topics.py
```

Run the parallel rewriting script:
```bash
python3 scripts/rewrite_topics_parallel.py
```
