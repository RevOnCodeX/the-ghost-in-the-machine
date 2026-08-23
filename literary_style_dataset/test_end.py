import re
from pathlib import Path

def test_end_matter(path_str):
    text = Path(path_str).read_text(encoding="utf-8", errors="ignore")
    end_match = re.search(r'\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*', text, re.IGNORECASE)
    if end_match:
        text = text[:end_match.start()]
    return text

print("Testing GE End...")
ge = test_end_matter("raw/dickens/great_expectations.txt")
print(repr(ge[-200:]))
