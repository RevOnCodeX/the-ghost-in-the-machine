import os
import re
import json
import subprocess
import random
from pathlib import Path
from collections import Counter

# Set up paths relative to the script location
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
TOPICS_DIR = REPO_ROOT / "literary_style_dataset" / "topics"
RESULTS_DIR = SCRIPT_DIR / "results"
SAMPLES_DIR = SCRIPT_DIR / "samples"

GIT_COMMIT_HUMAN = "b9e27d4"

def tokenize(text):
    """Basic tokenization: lowercase and extract alphabetic words."""
    # Remove punctuation, keep words
    words = re.findall(r'\b[a-z]+\b', text.lower())
    return words

def calculate_metrics(text):
    """Calculate word count, unique words, TTR, and hapax legomena."""
    words = tokenize(text)
    word_count = len(words)
    
    if word_count == 0:
        return None
        
    counts = Counter(words)
    unique_words = len(counts)
    ttr = unique_words / word_count
    hapax_legomena = sum(1 for word, count in counts.items() if count == 1)
    
    return {
        "word_count": word_count,
        "unique_words": unique_words,
        "ttr": ttr,
        "hapax_legomena": hapax_legomena
    }

def parse_paragraphs(text):
    """
    Parses a text containing '--- Paragraph N ---' headers.
    Returns a dictionary mapping paragraph_id -> paragraph_text.
    """
    paragraphs = {}
    # Split text using a regex that captures the paragraph ID
    parts = re.split(r'---\s*Paragraph\s+(\d+)\s*---', text)
    
    # parts[0] is everything before the first '--- Paragraph 1 ---'
    # parts[1] is ID '1', parts[2] is text for '1', parts[3] is ID '2', etc.
    for i in range(1, len(parts) - 1, 2):
        para_id = parts[i].strip()
        para_text = parts[i+1].strip()
        
        # Remove any leading/trailing blank lines, or the Note at the end of AI truncated files
        para_text = re.sub(r'\*\*Note:\*\*.*', '', para_text, flags=re.IGNORECASE | re.DOTALL).strip()
        
        if para_text:
            paragraphs[para_id] = para_text
            
    return paragraphs

def get_human_text_from_git(rel_file_path):
    """Fetches the original human text from the specific git commit."""
    try:
        # Run git show <commit>:<filepath>
        result = subprocess.run(
            ["git", "show", f"{GIT_COMMIT_HUMAN}:{rel_file_path}"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Warning: Could not fetch {rel_file_path} from git commit {GIT_COMMIT_HUMAN}.")
        return None

def main():
    print("Starting Lexical Richness Analysis...")
    
    all_pairs = []
    
    for root, dirs, files in os.walk(TOPICS_DIR):
        for filename in files:
            if not filename.endswith(".txt"):
                continue
                
            file_path = Path(root) / filename
            rel_file_path = file_path.relative_to(REPO_ROOT)
            
            # Read AI text from current filesystem
            with open(file_path, 'r', encoding='utf-8') as f:
                ai_content = f.read()
                
            if "(AI Rewritten)" not in ai_content:
                continue # Skip files that weren't rewritten
                
            # Extract metadata from headers
            book_match = re.search(r'# Book:\s*(.*)', ai_content)
            author_match = re.search(r'# Author:\s*(.*)', ai_content)
            topic_match = re.search(r'# Topic:\s*(.*)', ai_content)
            
            if not (book_match and author_match and topic_match):
                continue
                
            book = book_match.group(1).strip()
            author = author_match.group(1).strip()
            topic = topic_match.group(1).strip()
            
            ai_paragraphs = parse_paragraphs(ai_content)
            
            # Fetch human text from git history
            human_content = get_human_text_from_git(str(rel_file_path).replace("\\", "/"))
            if not human_content:
                continue
                
            human_paragraphs = parse_paragraphs(human_content)
            
            # Match paragraphs by ID
            for p_id, ai_text in ai_paragraphs.items():
                if p_id in human_paragraphs:
                    human_text = human_paragraphs[p_id]
                    
                    human_metrics = calculate_metrics(human_text)
                    ai_metrics = calculate_metrics(ai_text)
                    
                    if not human_metrics or not ai_metrics:
                        print(f"Skipping empty pair: {book} - {topic} - ID {p_id}")
                        continue
                        
                    ttr_diff = ai_metrics["ttr"] - human_metrics["ttr"]
                    ttr_pct = ((ai_metrics["ttr"] - human_metrics["ttr"]) / human_metrics["ttr"]) * 100
                    
                    hapax_diff = ai_metrics["hapax_legomena"] - human_metrics["hapax_legomena"]
                    hapax_pct = 0
                    if human_metrics["hapax_legomena"] > 0:
                        hapax_pct = ((ai_metrics["hapax_legomena"] - human_metrics["hapax_legomena"]) / human_metrics["hapax_legomena"]) * 100
                    elif ai_metrics["hapax_legomena"] > 0:
                        hapax_pct = 100.0 # From 0 to something
                    
                    pair = {
                        "book": book,
                        "author": author,
                        "topic": topic,
                        "paragraph_id": p_id,
                        "human_text": human_text,
                        "ai_text": ai_text,
                        "human": human_metrics,
                        "ai": ai_metrics,
                        "difference": {
                            "ttr_difference": round(ttr_diff, 4),
                            "ttr_percentage_change": round(ttr_pct, 2),
                            "hapax_difference": hapax_diff,
                            "hapax_percentage_change": round(hapax_pct, 2)
                        }
                    }
                    all_pairs.append(pair)
                else:
                    print(f"Missing human text for {book} - {topic} - ID {p_id}")

    print(f"Total valid pairs analyzed: {len(all_pairs)}")
    
    if len(all_pairs) < 500:
        print("WARNING: Less than 500 valid pairs were matched!")
        
    # Generate JSON output
    json_output = {
        "analysis": "Paired Human vs AI Lexical Richness",
        "total_pairs": len(all_pairs),
        "pairs": []
    }
    
    # Strip text out of JSON (as requested by JSON schema, text isn't in JSON directly, only metrics, wait, the schema doesn't have 'human_text' in JSON)
    for p in all_pairs:
        # Create a copy without the raw texts for the JSON
        p_json = {k: v for k, v in p.items() if k not in ["human_text", "ai_text"]}
        json_output["pairs"].append(p_json)
        
    with open(RESULTS_DIR / "lexical_results.json", 'w') as f:
        json.dump(json_output, f, indent=2)
        
    # Generate Samples
    with open(SAMPLES_DIR / "human_pairs_sample.txt", 'w') as f_h, open(SAMPLES_DIR / "ai_pairs_sample.txt", 'w') as f_a:
        for p in all_pairs:
            f_h.write(f"--- {p['book']} | {p['topic']} | Paragraph {p['paragraph_id']} ---\n")
            f_h.write(p['human_text'] + "\n\n")
            
            f_a.write(f"--- {p['book']} | {p['topic']} | Paragraph {p['paragraph_id']} ---\n")
            f_a.write(p['ai_text'] + "\n\n")
            
    # Generate Markdown Report
    dickens_pairs = [p for p in all_pairs if p["author"].lower() == "dickens"]
    austen_pairs = [p for p in all_pairs if p["author"].lower() == "austen"]
    
    def avg(lst):
        return sum(lst) / len(lst) if lst else 0
        
    def aggregate_metrics(pairs_list):
        return {
            "human_ttr": avg([p["human"]["ttr"] for p in pairs_list]),
            "ai_ttr": avg([p["ai"]["ttr"] for p in pairs_list]),
            "human_hapax": avg([p["human"]["hapax_legomena"] for p in pairs_list]),
            "ai_hapax": avg([p["ai"]["hapax_legomena"] for p in pairs_list]),
            "ttr_diff": avg([p["difference"]["ttr_difference"] for p in pairs_list]),
            "hapax_diff": avg([p["difference"]["hapax_difference"] for p in pairs_list])
        }

    overall = aggregate_metrics(all_pairs)
    dickens = aggregate_metrics(dickens_pairs)
    austen = aggregate_metrics(austen_pairs)
    
    report_md = f"""# Dataset Overview

- Total paired paragraphs analysed: **{len(all_pairs)}**
- Total Dickens pairs: **{len(dickens_pairs)}**
- Total Austen pairs: **{len(austen_pairs)}**

# Overall Results

## Average:
- **Human TTR:** {overall['human_ttr']:.4f}
- **AI TTR:** {overall['ai_ttr']:.4f}
- **Human Hapax:** {overall['human_hapax']:.2f}
- **AI Hapax:** {overall['ai_hapax']:.2f}

*(Average TTR difference: {overall['ttr_diff']:.4f}, Average Hapax difference: {overall['hapax_diff']:.2f})*

# Author Comparison

## Charles Dickens
- **Human TTR:** {dickens['human_ttr']:.4f} | **AI TTR:** {dickens['ai_ttr']:.4f}
- **Human Hapax:** {dickens['human_hapax']:.2f} | **AI Hapax:** {dickens['ai_hapax']:.2f}

## Jane Austen
- **Human TTR:** {austen['human_ttr']:.4f} | **AI TTR:** {austen['ai_ttr']:.4f}
- **Human Hapax:** {austen['human_hapax']:.2f} | **AI Hapax:** {austen['ai_hapax']:.2f}

# Example Paragraph Comparisons
"""
    
    # 10 random pairs
    random_samples = random.sample(all_pairs, min(10, len(all_pairs)))
    for p in random_samples:
        human_preview = " ".join(p["human_text"].split()[:100])
        if len(p["human_text"].split()) > 100: human_preview += "..."
        
        ai_preview = " ".join(p["ai_text"].split()[:100])
        if len(p["ai_text"].split()) > 100: ai_preview += "..."
        
        report_md += f"""
### Paragraph ID: {p['paragraph_id']}
**Topic:** {p['topic']} ({p['book']})

**Human:**
> {human_preview}

**AI:**
> {ai_preview}

**Metrics:**
- Human TTR: {p['human']['ttr']:.4f}
- AI TTR: {p['ai']['ttr']:.4f}
- Human Hapax: {p['human']['hapax_legomena']}
- AI Hapax: {p['ai']['hapax_legomena']}

---
"""
    
    with open(RESULTS_DIR / "lexical_report.md", 'w') as f:
        f.write(report_md)
        
    print(f"Average human TTR: {overall['human_ttr']:.4f}")
    print(f"Average AI TTR: {overall['ai_ttr']:.4f}")
    print(f"Average TTR difference: {overall['ttr_diff']:.4f}")
    print(f"Average hapax difference: {overall['hapax_diff']:.2f}")
    print("Done! Check the results/ and samples/ directories.")

if __name__ == "__main__":
    main()
