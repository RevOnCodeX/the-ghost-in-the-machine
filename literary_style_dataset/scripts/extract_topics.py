import json
import random
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "lib" / "dataset"
OUT_DIR = BASE_DIR / "lib" / "generation_dataset"

def main():
    extracted_topics = {
        "Charles Dickens": [],
        "Jane Austen": []
    }
    
    # We will look at the existing topics.json files and paragraphs
    for author_dir in DATASET_DIR.glob("*"):
        if not author_dir.is_dir(): continue
        
        author_name = "Charles Dickens" if author_dir.name == "dickens" else "Jane Austen"
        
        for topic_file in author_dir.glob("*_topics.json"):
            with open(topic_file, "r") as f:
                data = json.load(f)
                for topic in data.get("topics", []):
                    extracted_topics[author_name].append({
                        "topic_name": topic["topic_name"],
                        "description": topic["topic_description"]
                    })
                    
    # Save extracted topics
    with open(OUT_DIR / "topics.json", "w", encoding="utf-8") as f:
        json.dump(extracted_topics, f, indent=4, ensure_ascii=False)
        
    print(f"[Topic Extractor] Saved aggregated topics to {OUT_DIR}/topics.json")

if __name__ == "__main__":
    main()
