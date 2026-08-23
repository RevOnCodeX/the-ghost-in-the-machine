import re
from pathlib import Path

for path in Path("raw").rglob("*.txt"):
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    matches = []
    for i, line in enumerate(lines):
        if re.match(r'^\s*\[?(?:CHAPTER|Chapter)\s+(?:[IVXLCDM]+|\d+|ONE|TWO)\b', line):
            matches.append(i)
    print(f"\n{path.name}: {len(matches)} matches")
    if matches:
        print("First few matches line numbers:", matches[:10])
        print("Distance between first few:", [matches[i] - matches[i-1] for i in range(1, min(10, len(matches)))])
