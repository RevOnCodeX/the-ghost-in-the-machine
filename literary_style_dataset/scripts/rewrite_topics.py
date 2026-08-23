import os
import requests
import json
import time

token = os.environ.get("ANTHROPIC_AUTH_TOKEN")
base_dir = "/Users/aakshaj/.gemini/antigravity-ide/scratch/the-ghost-in-the-machine/literary_style_dataset/topics"

def rewrite_paragraphs(author, book, topic, text_content):
    prompt = f"""
You are an expert literary AI. I will provide you with 50 extracted paragraphs from the book '{book}' by {author}, all related to the topic of '{topic}'.
Your task is to REWRITE each paragraph entirely in the distinctive literary style of {author}.
Keep the general meaning and themes of each paragraph, but use new words, sentence structures, and phrasing.

The input is formatted with '--- Paragraph N ---' before each paragraph.
Please return your rewritten paragraphs using the EXACT same format:
--- Paragraph 1 ---
[Your rewritten paragraph 1]

--- Paragraph 2 ---
[Your rewritten paragraph 2]

...and so on up to 50.

Here is the original text:
{text_content}
"""

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "openrouter/free",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that rewrites literary text."},
            {"role": "user", "content": prompt}
        ]
    }
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=120)
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
            else:
                print(f"API Error {resp.status_code}: {resp.text}")
                if resp.status_code == 429:
                    print("Rate limited, sleeping for 20 seconds...")
                    time.sleep(20)
                else:
                    break
        except Exception as e:
            print(f"Exception during request: {e}")
            time.sleep(10)
    return None

# Iterate over all topic files
for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.endswith(".txt"):
            filepath = os.path.join(root, file)
            parts = filepath.split(os.sep)
            # path is .../topics/author/book/topic.txt
            author = parts[-3]
            book = parts[-2]
            topic = file.replace(".txt", "").replace("_", " ")
            
            print(f"Processing: {author} - {book} - {topic}")
            
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract just the paragraphs to send to the API (excluding the top header if possible, or just send all)
            # The prompt already instructs the API how to handle it.
            print(f"  Sending request to OpenRouter (free model)...")
            rewritten_text = rewrite_paragraphs(author, book, topic, content)
            
            if rewritten_text:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(f"# Topic: {topic}\n")
                    f.write(f"# Book: {book}\n")
                    f.write(f"# Author: {author}\n")
                    f.write(f"# (AI Rewritten)\n\n")
                    f.write(rewritten_text)
                print(f"  Successfully rewritten and saved.")
            else:
                print(f"  Failed to rewrite {file}.")
                
            # Sleep a little to respect free tier rate limits
            time.sleep(5)
            
print("Done rewriting all files!")
