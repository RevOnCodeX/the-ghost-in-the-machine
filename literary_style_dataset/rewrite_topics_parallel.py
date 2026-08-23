import os
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools

keys = [
    "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",
    "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",
    "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",
    os.environ.get("ANTHROPIC_AUTH_TOKEN")
]

# Use an iterator to round-robin through the keys
key_cycle = itertools.cycle(keys)

base_dir = "/Users/aakshaj/.gemini/antigravity-ide/scratch/the-ghost-in-the-machine/literary_style_dataset/topics"

def rewrite_file(filepath):
    # Check if already rewritten
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    if "(AI Rewritten)" in content:
        print(f"Skipping {os.path.basename(filepath)} - already rewritten.")
        return True
        
    parts = filepath.split(os.sep)
    author = parts[-3]
    book = parts[-2]
    topic = os.path.basename(filepath).replace(".txt", "").replace("_", " ")
    
    prompt = f"""
You are an expert literary AI. I will provide you with 50 extracted paragraphs from the book '{book}' by {author}, all related to the topic of '{topic}'.
Your task is to REWRITE each paragraph entirely in the distinctive literary style of {author}.
Keep the general meaning and themes of each paragraph, but use new words, sentence structures, and phrasing.

The input is formatted with '--- Paragraph N ---' before each paragraph.
Please return your rewritten paragraphs using the EXACT same format:
--- Paragraph 1 ---
[Your rewritten paragraph 1]

...and so on up to 50.

Here is the original text:
{content}
"""

    current_key = next(key_cycle)
    headers = {
        "Authorization": f"Bearer {current_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "openrouter/free",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that rewrites literary text."},
            {"role": "user", "content": prompt}
        ]
    }
    
    print(f"Processing {os.path.basename(filepath)}...")
    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=120)
            if resp.status_code == 200:
                rewritten_text = resp.json()["choices"][0]["message"]["content"]
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(f"# Topic: {topic}\n")
                    f.write(f"# Book: {book}\n")
                    f.write(f"# Author: {author}\n")
                    f.write(f"# (AI Rewritten)\n\n")
                    f.write(rewritten_text)
                print(f"Success: {os.path.basename(filepath)}")
                return True
            else:
                print(f"API Error {resp.status_code} for {os.path.basename(filepath)}: {resp.text}")
                if resp.status_code == 429:
                    time.sleep(5)
                else:
                    break
        except Exception as e:
            print(f"Exception on {os.path.basename(filepath)}: {e}")
            time.sleep(5)
    
    print(f"Failed to rewrite {os.path.basename(filepath)}")
    return False

files_to_process = []
for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.endswith(".txt"):
            files_to_process.append(os.path.join(root, file))

print(f"Total files found: {len(files_to_process)}")

# We use 4 workers so each key is effectively used by one thread
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = {executor.submit(rewrite_file, fp): fp for fp in files_to_process}
    for future in as_completed(futures):
        pass

print("All files processed!")
