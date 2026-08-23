import os
import re
import pandas as pd
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
TIER_B_DIR = SCRIPT_DIR.parent
ROOT_DIR = TIER_B_DIR.parent.parent
TOPICS_DIR = ROOT_DIR / "literary_style_dataset/topics"
OUTPUT_CSV = TIER_B_DIR / "data/text_pairs.csv"

GIT_COMMIT_HUMAN = "b9e27d4"

def parse_paragraphs(text):
    paragraphs = {}
    parts = re.split(r'---\s*Paragraph\s+(\d+)\s*---', text)
    for i in range(1, len(parts) - 1, 2):
        para_id = parts[i].strip()
        para_text = parts[i+1].strip()
        # Remove AI notes
        para_text = re.sub(r'\*\*Note:\*\*.*', '', para_text, flags=re.IGNORECASE | re.DOTALL).strip()
        if para_text:
            paragraphs[para_id] = para_text
    return paragraphs

def get_human_text_from_git(rel_file_path):
    try:
        result = subprocess.run(
            ["git", "show", f"{GIT_COMMIT_HUMAN}:{rel_file_path}"],
            cwd=ROOT_DIR,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError:
        return None

def main():
    print("Preparing Text Dataset for Tier B...")
    dataset_rows = []
    
    for root, dirs, files in os.walk(TOPICS_DIR):
        for filename in files:
            if not filename.endswith(".txt"):
                continue
                
            file_path = Path(root) / filename
            rel_file_path = file_path.relative_to(ROOT_DIR)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                ai_content = f.read()
                
            if "(AI Rewritten)" not in ai_content:
                continue
                
            book_match = re.search(r'# Book:\s*(.*)', ai_content)
            author_match = re.search(r'# Author:\s*(.*)', ai_content)
            topic_match = re.search(r'# Topic:\s*(.*)', ai_content)
            
            if not (book_match and author_match and topic_match):
                continue
                
            book = book_match.group(1).strip()
            author = author_match.group(1).strip()
            topic = topic_match.group(1).strip()
            
            ai_paragraphs = parse_paragraphs(ai_content)
            human_content = get_human_text_from_git(str(rel_file_path).replace("\\", "/"))
            
            if not human_content:
                continue
                
            human_paragraphs = parse_paragraphs(human_content)
            
            for p_id, ai_text in ai_paragraphs.items():
                if p_id in human_paragraphs:
                    human_text = human_paragraphs[p_id]
                    
                    # Human Row (Label 0)
                    dataset_rows.append({
                        "paragraph_id": f"{book}_T{topic[:5]}_P{p_id}",
                        "author": author,
                        "book": book,
                        "topic": topic,
                        "text": human_text,
                        "label": 0
                    })
                    
                    # AI Row (Label 1)
                    dataset_rows.append({
                        "paragraph_id": f"{book}_T{topic[:5]}_P{p_id}",
                        "author": author,
                        "book": book,
                        "topic": topic,
                        "text": ai_text,
                        "label": 1
                    })
                    
    df = pd.DataFrame(dataset_rows)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"Saved {len(df)} total rows ( {len(df)//2} pairs ) to {OUTPUT_CSV}")
    
    # Validations
    assert len(df) > 0, "No data extracted!"
    assert len(df[df['label'] == 0]) == len(df[df['label'] == 1]), "Imbalanced classes!"
    
if __name__ == "__main__":
    main()
