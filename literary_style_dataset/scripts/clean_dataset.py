import os
import re
from pathlib import Path

# Setup paths relative to the script location
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "raw"
CLEANED_DIR = BASE_DIR / "cleaned"

def remove_gutenberg_boilerplate(text):
    """
    Identifies and removes Project Gutenberg boilerplate text.
    """
    # Match the start of the book content
    start_match = re.search(r'\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*', text, re.IGNORECASE)
    end_match = re.search(r'\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*', text, re.IGNORECASE)
    
    if start_match:
        text = text[start_match.end():]
    if end_match:
        text = text[:end_match.start()]
        
    # Some older files have a second smaller header after the start tag or transcribers notes.
    # While we could remove more, the prompt strictly requests preserving author text.
    # We will just strip leading/trailing whitespace left over from the markers.
    return text.strip()

def normalize_text(text):
    """
    Normalizes line breaks, excessive whitespace, and invalid characters
    while preserving punctuation and sentence boundaries.
    """
    # Remove specific control characters except standard whitespace (newlines, tabs, spaces)
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    
    # Normalize line endings to standard Unix style
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # Project Gutenberg texts are hard-wrapped (often at ~72 characters).
    # To reconstruct paragraphs, we split the text by double newlines (which denote true paragraph breaks).
    blocks = re.split(r'\n{2,}', text)
    
    cleaned_blocks = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        
        # Replace single newlines with spaces to un-wrap lines within a paragraph.
        # This keeps the sentence structure intact on a single logical line.
        block = re.sub(r'(?<!\n)\n(?!\n)', ' ', block)
        
        # Collapse multiple spaces into a single space (accidental repeated whitespace)
        block = re.sub(r' {2,}', ' ', block)
        
        cleaned_blocks.append(block)
        
    # Rejoin the actual paragraphs with double newlines
    return '\n\n'.join(cleaned_blocks)

def process_file(file_path, output_dir):
    print(f"Processing: {file_path.name}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()
    except UnicodeDecodeError:
        # Fallback for older encodings if utf-8 fails
        with open(file_path, 'r', encoding='latin-1') as f:
            raw_text = f.read()
            
    if not raw_text.strip():
        print(f"  [WARNING] File is completely empty: {file_path}\n")
        return False
        
    original_char_count = len(raw_text)
    original_word_count = len(raw_text.split())
    original_lines = len(raw_text.splitlines())
    
    # Execute cleaning pipeline
    text_no_boiler = remove_gutenberg_boilerplate(raw_text)
    cleaned_text = normalize_text(text_no_boiler)
    
    cleaned_char_count = len(cleaned_text)
    cleaned_word_count = len(cleaned_text.split())
    cleaned_lines = len(cleaned_text.splitlines())
    
    lines_removed = original_lines - cleaned_lines
    
    # Validation checks
    if cleaned_char_count < 10000:
        print(f"  [WARNING] Cleaned file is suspiciously short ({cleaned_char_count} chars).")
        
    if cleaned_char_count == 0:
        print(f"  [ERROR] Cleaned file is empty after processing! Check the removal regex.")
        return False
        
    print(f"  Original: {original_word_count:,} words | {original_char_count:,} chars | {original_lines:,} lines")
    print(f"  Cleaned:  {cleaned_word_count:,} words | {cleaned_char_count:,} chars | {cleaned_lines:,} lines")
    print(f"  Lines removed/merged: {lines_removed:,}")
    
    # Save the output
    output_dir.mkdir(parents=True, exist_ok=True)
    out_name = file_path.stem + "_cleaned.txt"
    out_path = output_dir / out_name
    
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(cleaned_text)
        
    print(f"  Saved to: {out_path}\n")
    return True

def main():
    print("========================================")
    print(" Literary Style Dataset Cleaning Script ")
    print("========================================\n")
    
    # Order of processing is specified by requirements
    authors = ['dickens', 'austen']
    
    for author in authors:
        author_raw_dir = RAW_DIR / author
        author_clean_dir = CLEANED_DIR / author
        
        if not author_raw_dir.exists():
            print(f"[ERROR] Directory not found: {author_raw_dir}")
            continue
            
        print(f"--- Processing Author: {author.capitalize()} ---")
        
        files_found = list(author_raw_dir.glob('*.txt'))
        if not files_found:
            print(f"  No .txt files found in {author_raw_dir}\n")
            continue
            
        for file_path in sorted(files_found):
            process_file(file_path, author_clean_dir)
            
    print("Pipeline completed.")

if __name__ == '__main__':
    main()
