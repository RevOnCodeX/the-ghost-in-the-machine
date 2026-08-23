import re
from pathlib import Path

def test_front_matter(path_str):
    text = Path(path_str).read_text(encoding="utf-8", errors="ignore")
    
    # 1. Remove illustrations
    def replacer(match):
        content = match.group(0)
        chap_match = re.search(r'((?:CHAPTER|Chapter)\s+[IVXLCDM\d]+.*?)(?=\n|\])', content)
        if chap_match:
            return chap_match.group(1)
        return ""
    text = re.sub(r'\[Illustration[^\]]*\]', replacer, text, flags=re.IGNORECASE | re.DOTALL)
    
    # 2. Chop END marker
    end_match = re.search(r'\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*', text, re.IGNORECASE)
    if end_match:
        text = text[:end_match.start()]
        
    # 3. Strip front matter
    matches = list(re.finditer(r'^\s*(?:CHAPTER|Chapter)\s+(?:[IVXLCDM]+|\d+|ONE|TWO)\b.*$', text, flags=re.MULTILINE))
    
    actual_chapter_start = -1
    for i in range(len(matches)):
        if i == len(matches) - 1:
            actual_chapter_start = matches[i].start()
            break
        chunk = text[matches[i].end():matches[i+1].start()]
        if chunk.count('\n') > 15:
            actual_chapter_start = matches[i].start()
            break
            
    if actual_chapter_start != -1:
        text = text[actual_chapter_start:]
    
    return text

print("Testing P&P...")
pp = test_front_matter("raw/austen/pride_and_prejudice.txt")
print(repr(pp[:200]))

print("Testing Oliver Twist...")
ot = test_front_matter("raw/dickens/oliver_twist.txt")
print(repr(ot[:200]))
