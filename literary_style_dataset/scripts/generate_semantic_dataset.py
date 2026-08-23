import os
import json
import time
from pathlib import Path

# Try to import google.generativeai, but fall back if not available to allow dry runs
try:
    import google.generativeai as genai
except ImportError:
    genai = None

# Set up paths
BASE_DIR = Path(__file__).resolve().parent.parent
CLEANED_DIR = BASE_DIR / "cleaned"
DATASET_DIR = BASE_DIR / "lib" / "dataset"

def init_api():
    """Initializes the Gemini API using the environment variable."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("[WARNING] GEMINI_API_KEY environment variable not set. Running in dry-run/mock mode.")
        return False
    if genai:
        genai.configure(api_key=api_key)
        return True
    return False

def generate_topics_via_api(novel_title, author, text_sample):
    """
    Calls the Gemini API to extract 5-10 core semantic topics for a novel.
    Returns a dictionary matching the topic schema.
    """
    if not genai or not os.environ.get("GEMINI_API_KEY"):
        # Fallback to mock data for dry runs
        return {
            "novel": novel_title,
            "author": author,
            "topics": [
                {
                    "topic_id": "T01",
                    "topic_name": "Mock Topic",
                    "topic_description": "This is a mock topic description for dry runs."
                }
            ]
        }
        
    model = genai.GenerativeModel('gemini-1.5-pro')
    prompt = f"""
    Analyze the following text from {novel_title} by {author}.
    Identify 5-10 core recurring topics (themes, ideas, conflicts, social issues).
    Do not use character names, chapter names, or one-time events.
    Output strictly as JSON matching this format:
    {{
      "novel": "{novel_title}",
      "author": "{author}",
      "topics": [
        {{ "topic_id": "T01", "topic_name": "...", "topic_description": "..." }}
      ]
    }}
    Text sample:
    {text_sample[:15000]}
    """
    response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
    return json.loads(response.text)

def generate_paragraph_metadata_via_api(novel_title, author, chapter, paragraph_index, paragraph_text, topics):
    """
    Calls the Gemini API to extract paragraph-level semantic metadata.
    """
    if not genai or not os.environ.get("GEMINI_API_KEY"):
        # Fallback to mock data
        return {
            "concise_summary": "Mock summary for dry runs.",
            "primary_topic": topics["topics"][0],
            "secondary_topics": [],
            "topic_relevance": "Mock relevance.",
            "key_entities": ["Entity1"],
            "key_events": ["Event1"],
            "semantic_keywords": ["keyword1"]
        }
        
    model = genai.GenerativeModel('gemini-1.5-pro')
    topics_json = json.dumps(topics["topics"])
    prompt = f"""
    Analyze the following paragraph from {novel_title} by {author} ({chapter}).
    Generate semantic metadata in JSON format.
    
    Paragraph:
    {paragraph_text}
    
    Available topics:
    {topics_json}
    
    Return exactly this JSON structure:
    {{
      "concise_summary": "1-2 sentence factual summary without inventing information",
      "primary_topic": {{ "topic_id": "...", "topic_name": "..." }},
      "secondary_topics": [ ... ],
      "topic_relevance": "Brief explanation of why the primary topic applies",
      "key_entities": ["list", "of", "entities"],
      "key_events": ["list", "of", "events"],
      "semantic_keywords": ["list", "of", "keywords"]
    }}
    """
    
    try:
        response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
        return json.loads(response.text)
    except Exception as e:
        print(f"  [ERROR] API call failed for paragraph {paragraph_index}: {e}")
        return None

def process_novel(file_path, author_dataset_dir, author_name, max_paragraphs=None):
    """
    Reads the cleaned novel, extracts topics, and annotates paragraphs.
    """
    novel_title = file_path.stem.replace('_cleaned', '').replace('_', ' ').title()
    print(f"\nProcessing {novel_title} by {author_name}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
        
    # Split into paragraphs based on double newlines
    paragraphs = text.split('\n\n')
    
    # Generate topics
    topics_dict = generate_topics_via_api(novel_title, author_name, text)
    
    # Save topics
    topics_file = author_dataset_dir / f"{file_path.stem.replace('_cleaned', '')}_topics.json"
    with open(topics_file, 'w', encoding='utf-8') as f:
        json.dump(topics_dict, f, indent=2)
    print(f"  Saved topics to {topics_file.name}")
    
    # Setup for paragraph processing
    paragraphs_file = author_dataset_dir / f"{file_path.stem.replace('_cleaned', '')}_paragraphs.jsonl"
    book_code = "".join(word[0].upper() for word in novel_title.split())
    
    current_chapter = "Front Matter"
    paragraphs_processed = 0
    
    with open(paragraphs_file, 'w', encoding='utf-8') as f_out:
        for p_idx, para in enumerate(paragraphs):
            para = para.strip()
            if not para:
                continue
                
            # Very short paragraphs (like just a character name or single word) might not be meaningful
            # But the prompt requires "every meaningful paragraph". We'll filter out extremely short ones.
            if len(para.split()) < 5 and not para.upper().startswith("CHAPTER"):
                continue
                
            # Update chapter tracker
            if para.upper().startswith("CHAPTER") or para.upper().startswith("BOOK"):
                current_chapter = para.split('\n')[0][:50]
                
            paragraphs_processed += 1
            if max_paragraphs and paragraphs_processed > max_paragraphs:
                break
                
            # Format paragraph ID
            p_id = f"{book_code}_CH{current_chapter.split()[-1][:2] if 'CHAPTER' in current_chapter.upper() else '00'}_P{paragraphs_processed:04d}"
            
            # Rate limiting sleep for API
            if paragraphs_processed > 1 and os.environ.get("GEMINI_API_KEY"):
                time.sleep(1) 
                
            metadata = generate_paragraph_metadata_via_api(novel_title, author_name, current_chapter, paragraphs_processed, para, topics_dict)
            
            if not metadata:
                continue
                
            # Build full record
            record = {
                "paragraph_id": p_id,
                "novel": novel_title,
                "author": author_name,
                "chapter": current_chapter,
                "paragraph_index": paragraphs_processed,
                "paragraph_text": para
            }
            record.update(metadata)
            
            # Write JSONL
            f_out.write(json.dumps(record, ensure_ascii=False) + '\n')
            
    print(f"  Processed {paragraphs_processed} paragraphs for {novel_title}.")

def main():
    print("==================================================")
    print(" Semantic Dataset Generation (Gemini API)")
    print("==================================================")
    
    init_api()
    
    authors = ['dickens', 'austen']
    
    for author in authors:
        author_clean_dir = CLEANED_DIR / author
        author_dataset_dir = DATASET_DIR / author
        author_dataset_dir.mkdir(parents=True, exist_ok=True)
        
        author_name = "Charles Dickens" if author == "dickens" else "Jane Austen"
        
        if not author_clean_dir.exists():
            continue
            
        for file_path in sorted(author_clean_dir.glob('*.txt')):
            process_novel(file_path, author_dataset_dir, author_name, max_paragraphs=10)
            
    print("\nDataset generation completed.")

if __name__ == "__main__":
    main()
