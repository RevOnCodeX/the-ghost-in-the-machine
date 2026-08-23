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

def get_style_prompt(author):
    """
    Returns a highly detailed style guide for the specific author to ensure
    the AI-styled generation closely mimics their unique literary voice.
    """
    if author.lower() == "charles dickens":
        return """
        STYLE GUIDE FOR CHARLES DICKENS:
        1. Verbosity and Rhythm: Use long, flowing, often breathless sentences connected by multiple clauses and semicolons.
        2. Satire and Caricature: Exaggerate human flaws. Treat institutions with biting irony and poor characters with deep, sometimes melodramatic sympathy.
        3. Personification: Give inanimate objects (like fog, houses, mud) living, breathing, and often malevolent characteristics.
        4. Vocabulary: Use archaic, highly formal vocabulary playfully mixed with working-class dialect (where dialogue applies). 
        5. Capitalization: Frequently capitalize abstract nouns (e.g., Nature, Fate, The Law) for dramatic emphasis.
        """
    elif author.lower() == "jane austen":
        return """
        STYLE GUIDE FOR JANE AUSTEN:
        1. Irony and Wit: Employ dry, subtle irony. Much of the humor should come from exposing the hypocrisy or self-delusion of the subjects.
        2. Sentence Structure: Use perfectly balanced, measured, and highly structured syntax. Sentences should be elegant and precise.
        3. Focus: Concentrate strictly on social dynamics, manners, class distinctions, and psychological motivations rather than physical environments or melodrama.
        4. Tone: The narrator should sound detached, highly intelligent, and slightly amused by the proceedings.
        5. Vocabulary: Use formal, refined Regency-era English without being overly florid or Gothic.
        """
    return "Write in a standard literary style."

def generate_ai_texts_via_api(author, summary):
    """
    Calls the Gemini API to generate two variations of a paragraph based on its summary.
    """
    if not genai or not os.environ.get("GEMINI_API_KEY"):
        # Fallback to mock data for dry runs
        return {
            "plain_ai_text": "This is a mock plain AI generation based on the summary.",
            "styled_ai_text": f"This is a mock styled AI generation mimicking {author}."
        }
        
    model = genai.GenerativeModel('gemini-1.5-pro')
    style_guide = get_style_prompt(author)
    
    prompt = f"""
    You are assisting in creating a dataset for an AI vs Human detection model.
    I will provide you with a highly descriptive summary of a specific paragraph from a novel by {author}.
    
    Summary: "{summary}"
    
    Your task is to generate TWO variations of this paragraph:
    
    1. A "Plain AI" version: Write a standard, neutral, modern AI-generated paragraph that conveys the exact meaning of the summary. Do NOT try to be literary.
    2. A "Styled AI" version: Write a paragraph that conveys the exact meaning of the summary, but fiercely mimics the author's unique literary style.
    
    {style_guide}
    
    Return the result strictly as JSON:
    {{
      "plain_ai_text": "...",
      "styled_ai_text": "..."
    }}
    """
    
    try:
        response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
        return json.loads(response.text)
    except Exception as e:
        print(f"  [ERROR] API call failed: {e}")
        return None

def process_author_dataset(author_dir, author_name):
    """Reads existing JSONL files and generates AI variations."""
    print(f"\nProcessing datasets for {author_name}...")
    
    for human_file in sorted(author_dir.glob('*_paragraphs.jsonl')):
        novel_title = human_file.stem.replace('_paragraphs', '').replace('_', ' ').title()
        
        plain_file = author_dir / f"{human_file.stem.replace('_paragraphs', '')}_ai_plain.jsonl"
        styled_file = author_dir / f"{human_file.stem.replace('_paragraphs', '')}_ai_styled.jsonl"
        
        plain_records = []
        styled_records = []
        
        with open(human_file, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line)
                summary = record.get("concise_summary")
                p_id = record.get("paragraph_id")
                
                # Rate limiting
                if os.environ.get("GEMINI_API_KEY"):
                    time.sleep(1.5)
                    
                ai_texts = generate_ai_texts_via_api(author_name, summary)
                
                if ai_texts:
                    # Create matching plain record
                    plain_records.append({
                        "paragraph_id": p_id,
                        "novel": novel_title,
                        "author": author_name,
                        "ai_plain_text": ai_texts["plain_ai_text"]
                    })
                    
                    # Create matching styled record
                    styled_records.append({
                        "paragraph_id": p_id,
                        "novel": novel_title,
                        "author": author_name,
                        "ai_styled_text": ai_texts["styled_ai_text"]
                    })
        
        # Write outputs
        with open(plain_file, 'w', encoding='utf-8') as f:
            for r in plain_records:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
                
        with open(styled_file, 'w', encoding='utf-8') as f:
            for r in styled_records:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
                
        print(f"  Generated AI datasets for {novel_title}")

def main():
    print("==================================================")
    print(" AI vs Human Dataset Generation (Gemini API)")
    print("==================================================")
    
    init_api()
    
    authors = {'dickens': 'Charles Dickens', 'austen': 'Jane Austen'}
    
    for author_folder, author_name in authors.items():
        author_dir = DATASET_DIR / author_folder
        if author_dir.exists():
            process_author_dataset(author_dir, author_name)
            
    print("\nAI Variation Dataset generation completed.")

if __name__ == "__main__":
    main()
