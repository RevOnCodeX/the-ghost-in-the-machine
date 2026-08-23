import json
import random
import time
from pathlib import Path
from api_router import APIRouter

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "configs"
LIB_DIR = BASE_DIR / "lib" / "generation_dataset"
PROMPTS_DIR = LIB_DIR / "prompts"
PLAIN_DIR = LIB_DIR / "generated" / "plain"
STYLED_DIR = LIB_DIR / "generated" / "styled"

def load_settings():
    with open(CONFIG_DIR / "generation_settings.json", "r") as f:
        return json.load(f)

def load_topics():
    with open(LIB_DIR / "topics.json", "r") as f:
        return json.load(f)

def get_author_characteristics(author):
    if "Dickens" in author:
        return "Victorian narration, social observation, irony, descriptive environments, complex sentences, moral reflection"
    else:
        return "social commentary, dialogue-driven writing, subtle irony, relationship analysis, class observations"

def log_generation(provider, model, success, message=""):
    log_file = BASE_DIR / "generation_logs" / "generation.log"
    with open(log_file, "a") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Provider: {provider}, Model: {model}, Success: {success}, Msg: {message}\n")

def generate_plain_batch(router, author, topics_batch, batch_index, settings):
    min_words, max_words = settings["word_range"]["min"], settings["word_range"]["max"]
    
    # Constructing prompt
    prompt = f"Generate exactly {len(topics_batch)} distinct paragraphs.\n"
    prompt += f"Each paragraph MUST be strictly between {min_words} and {max_words} words.\n"
    prompt += "The tone must be neutral, modern, and plain.\n\n"
    prompt += "Topics to cover:\n"
    for i, t in enumerate(topics_batch):
        prompt += f"{i+1}. {t['topic_name']}: {t['description']}\n"
    
    prompt += "\nReturn ONLY a valid JSON object with the following schema:\n"
    prompt += "{\n  \"paragraphs\": [\n"
    prompt += "    {\"topic\": \"<topic_name>\", \"text\": \"<generated_text>\"}\n"
    prompt += "  ]\n}"

    response = router.route_request(prompt)
    if not response["success"]:
        log_generation("none", "none", False, response.get("error"))
        return None
        
    try:
        content_str = response["content"].strip()
        if content_str.startswith("```json"): content_str = content_str[7:]
        elif content_str.startswith("```"): content_str = content_str[3:]
        if content_str.endswith("```"): content_str = content_str[:-3]
        
        data = json.loads(content_str.strip())
        results = []
        for i, item in enumerate(data.get("paragraphs", [])):
            results.append({
                "id": f"{author.split()[-1].upper()}_B{batch_index}_P{i+1}",
                "topic": item["topic"],
                "text": item["text"],
                "provider": response["provider"],
                "model": response["model"]
            })
        log_generation(response["provider"], response["model"], True, "Plain batch generated")
        return results
    except Exception as e:
        log_generation(response["provider"], response["model"], False, f"JSON parse error: {e}")
        return None

def generate_styled_batch(router, author, plain_batch):
    characteristics = get_author_characteristics(author)
    
    prompt = f"Rewrite the following {len(plain_batch)} paragraphs to strictly mimic the literary style of {author}.\n"
    prompt += f"Use the following characteristics: {characteristics}.\n"
    prompt += "Do not change the underlying meaning, just the writing style. Ensure the length remains similar.\n\n"
    
    for i, item in enumerate(plain_batch):
        prompt += f"Paragraph {i+1} (ID: {item['id']}):\n{item['text']}\n\n"
        
    prompt += "Return ONLY a valid JSON object with the following schema:\n"
    prompt += "{\n  \"paragraphs\": [\n"
    prompt += "    {\"id\": \"<id_from_above>\", \"styled_text\": \"<rewritten_text>\"}\n"
    prompt += "  ]\n}"

    response = router.route_request(prompt)
    if not response["success"]:
        log_generation("none", "none", False, response.get("error"))
        return None
        
    try:
        content_str = response["content"].strip()
        if content_str.startswith("```json"): content_str = content_str[7:]
        elif content_str.startswith("```"): content_str = content_str[3:]
        if content_str.endswith("```"): content_str = content_str[:-3]
        
        data = json.loads(content_str.strip())
        results = []
        # Match back to original
        for item in data.get("paragraphs", []):
            original = next((p for p in plain_batch if p["id"] == item["id"]), None)
            if original:
                results.append({
                    "id": item["id"],
                    "original_text": original["text"],
                    "styled_text": item["styled_text"],
                    "target_author": author,
                    "provider": response["provider"]
                })
        log_generation(response["provider"], response["model"], True, "Styled batch generated")
        return results
    except Exception as e:
        log_generation(response["provider"], response["model"], False, f"JSON parse error: {e}")
        return None

def main():
    settings = load_settings()
    all_topics = load_topics()
    router = APIRouter()
    
    is_test = False # We will run test mode first as requested
    total_needed = settings["test_mode_paragraphs"] if is_test else settings["total_paragraphs"]
    batch_size = settings["batch_size"]
    
    print(f"Starting generation. Target: {total_needed} per author. Batch size: {batch_size}")
    
    for author, topics in all_topics.items():
        print(f"\nProcessing {author}...")
        author_key = author.split()[-1].lower()
        
        # Sample topics with replacement if we need more than we have
        selected_topics = random.choices(topics, k=total_needed)
        
        plain_results = []
        styled_results = []
        
        for i in range(0, len(selected_topics), batch_size):
            batch_topics = selected_topics[i:i+batch_size]
            batch_idx = (i // batch_size) + 1
            print(f"  Generating Batch {batch_idx} (Plain)...")
            
            p_batch = generate_plain_batch(router, author, batch_topics, batch_idx, settings)
            if p_batch:
                plain_results.extend(p_batch)
                
                print(f"  Generating Batch {batch_idx} (Styled)...")
                s_batch = generate_styled_batch(router, author, p_batch)
                if s_batch:
                    styled_results.extend(s_batch)
                    
            # Brief pause to respect rate limits between batches
            time.sleep(2)
            
        # Save to disk
        plain_file = PLAIN_DIR / f"{author_key}_plain.jsonl"
        with open(plain_file, "w", encoding="utf-8") as f:
            for r in plain_results: f.write(json.dumps(r) + "\n")
            
        styled_file = STYLED_DIR / f"{author_key}_styled.jsonl"
        with open(styled_file, "w", encoding="utf-8") as f:
            for r in styled_results: f.write(json.dumps(r) + "\n")
            
    print("Generation complete!")

if __name__ == "__main__":
    main()
