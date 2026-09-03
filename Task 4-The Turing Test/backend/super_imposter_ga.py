import os
import requests
import json
import time
import random

BEDROCK_KEY = os.environ.get("BEDROCK_API_KEY", "")
if not BEDROCK_KEY:
    raise EnvironmentError(
        "BEDROCK_API_KEY is not set. "
        "Export it before running: export BEDROCK_API_KEY='your-key-here'"
    )
API_URL = "http://127.0.0.1:8000/analyze"

def ask_llm(prompt):
    url = "https://bedrock-runtime.us-east-1.amazonaws.com/model/google.gemma-3-12b-it/converse"
    headers = {
        "Authorization": f"Bearer {BEDROCK_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "messages": [
            {"role": "user", "content": [{"text": prompt}]}
        ]
    }
    
    for attempt in range(3):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            if resp.status_code == 200:
                data = resp.json()
                return data["output"]["message"]["content"][0]["text"].strip()
            else:
                print(f"Bedrock Error: {resp.status_code} - {resp.text}")
        except Exception as e:
            print(f"Request failed: {e}")
        time.sleep(2)
    return ""

def generate_initial_population():
    print("Generating initial population of 10 imposter paragraphs...")
    prompt = """
Write 10 distinct, short paragraphs (3-4 sentences each) about the benefits of space exploration.
Each paragraph must be clearly separated by the exact text: "---PARAGRAPH---"
Do not include numbers, bullet points, or any other formatting. Just the paragraphs separated by the delimiter.
"""
    response = ask_llm(prompt)
    paragraphs = [p.strip() for p in response.split("---PARAGRAPH---") if len(p.strip()) > 20]
    
    # Ensure we have exactly 10
    if len(paragraphs) > 10:
        paragraphs = paragraphs[:10]
    while len(paragraphs) < 10:
        paragraphs.append(paragraphs[0] if paragraphs else "Space exploration is very important for the future of humanity. It allows us to discover new worlds and technologies. We must continue to invest in space agencies.")
    
    return paragraphs

def get_fitness(text):
    """Returns the 'human_prob' score and top flagged words from the local AI Detector."""
    if len(text.split()) < 15 or "User Safety: safe" in text:
        return 0.0, []
        
    payload = {
        "text": text,
        "model": "Tier C - Transformer"
    }
    try:
        resp = requests.post(API_URL, json=payload, timeout=60)
        if resp.status_code == 200:
            data = resp.json()
            if "Tier C" in data and "human_prob" in data["Tier C"]:
                prob = data["Tier C"]["human_prob"]
                # Extract top 5 words with highest attribution
                attrs = data["Tier C"].get("attributions", [])
                # Sort by normalized_score descending, and take the word itself
                attrs.sort(key=lambda x: x.get("normalized_score", 0), reverse=True)
                top_words = [a["word"] for a in attrs[:5] if a.get("normalized_score", 0) > 0.1]
                return prob, top_words
    except Exception as e:
        print(f"Error calling local API: {e}")
    return 0.0, []

def inject_typos(text):
    """Programmatically injects character-level noise to disrupt tokenization."""
    if not text:
        return text
    chars = list(text)
    
    # 1. Randomly drop 1-2 punctuation marks
    import string
    punct_indices = [i for i, c in enumerate(chars) if c in string.punctuation]
    if punct_indices:
        drops = random.sample(punct_indices, min(2, len(punct_indices)))
        for idx in sorted(drops, reverse=True):
            chars.pop(idx)
            
    # 2. Randomly swap two adjacent characters (simulate typing error)
    if len(chars) > 5:
        idx = random.randint(1, len(chars)-3)
        chars[idx], chars[idx+1] = chars[idx+1], chars[idx]
        
    return "".join(chars)

def mutate_paragraph(text, mutation_type, fitness_score, flagged_words):
    if mutation_type == "rhythm":
        instruction = "Drastically rewrite this paragraph to completely change its rhythm, structure, and sentence length. Break long sentences into short ones, or combine short ones."
    elif mutation_type == "grammar":
        instruction = "Rewrite this paragraph and introduce subtle grammatical inconsistencies, conversational filler words (um, like, kinda), or very casual slang to make it sound authentically human."
    elif mutation_type == "conversational":
        instruction = "Rewrite this paragraph to sound extremely casual, opinionated, and slightly imperfect, like a human hastily writing a passionate Reddit comment."
    elif mutation_type == "typo":
        instruction = "Rewrite this paragraph but deliberately include 2 or 3 typos (e.g. miss an apostrophe, misspell a common word, or fail to capitalize a sentence). Make it look like hasty human typing."
    else:
        instruction = "Rewrite this paragraph completely from scratch using entirely different vocabulary and phrasing."

    flagged_str = ""
    if flagged_words:
        flagged_str = f"IMPORTANT: The AI detector specifically caught you because of these words/phrases: {', '.join(flagged_words)}. You MUST entirely avoid these words. Delete them, use completely different synonyms, or restructure the sentence so they aren't needed.\n\n"

    prompt = f"{instruction}\n\nThis paragraph scored {fitness_score*100:.2f}% 'Human'. We need >90%. {flagged_str}Make sure it sounds like a real person wrote it. Do not sound like an AI.\n\nParagraph:\n{text}\n\nProvide only the rewritten paragraph, without any intro or outro text."
    
    result = ask_llm(prompt)
    if mutation_type == "typo" and result:
        result = inject_typos(result)
        
    return result

def run_ga():
    population = generate_initial_population()
    generations = 50
    target_fitness = 0.90
    
    for gen in range(1, generations + 1):
        print(f"\n--- Generation {gen} ---")
        
        # Evaluate fitness
        scored_population = []
        for i, p in enumerate(population):
            fitness, top_words = get_fitness(p)
            scored_population.append((fitness, p, top_words))
            print(f"Indiv {i+1}: Fitness = {fitness:.4f}")
            
        # Sort by fitness descending
        scored_population.sort(key=lambda x: x[0], reverse=True)
        
        best_fitness = scored_population[0][0]
        best_text = scored_population[0][1]
        best_words = scored_population[0][2]
        
        print(f"\nBest Fitness in Gen {gen}: {best_fitness:.4f}")
        print(f"Best Text: {best_text}")
        if best_words:
            print(f"Flagged AI-isms: {best_words}")
        
        if best_fitness >= target_fitness:
            print(f"\nSUCCESS! Reached target fitness of >90% in {gen} generations.")
            break
            
        # Selection: Keep top 3
        top_3 = [item[1] for item in scored_population[:3]]
        
        # Mutation to create next generation
        print("\nMutating top 3 to create next generation...")
        next_gen = []
        
        # Elitism: Keep the absolute best unchanged
        next_gen.append(top_3[0])
        
        mutation_types = ["rhythm", "grammar", "conversational", "typo"]
        
        # We need 9 more to fill population of 10.
        # Mutate each of the top 3 with the 3 different mutation strategies.
        for parent_idx, parent_tuple in enumerate(scored_population[:3]):
            parent_fitness = parent_tuple[0]
            parent_text = parent_tuple[1]
            parent_words = parent_tuple[2]
            for m_type in mutation_types:
                child = mutate_paragraph(parent_text, m_type, parent_fitness, parent_words)
                if not child:
                    child = parent_text # fallback
                next_gen.append(child)
                
        population = next_gen

if __name__ == "__main__":
    run_ga()
