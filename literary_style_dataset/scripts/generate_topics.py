import os
import random
import re

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
base_dir = str(BASE_DIR / "cleaned")
output_dir = str(BASE_DIR / "topics")

books = {
    "austen": {
        "emma_cleaned.txt": {
            "Marriage and Matchmaking": ["marry", "marriage", "match", "wife", "husband", "wedding", "engage"],
            "Social Status": ["rank", "gentleman", "lady", "poor", "rich", "respectable", "status", "class"],
            "Self-Delusion": ["imagine", "fancy", "mistake", "blind", "deceive", "error", "delusion"],
            "Female Independence": ["independent", "fortune", "mistress", "free", "single", "alone"],
            "Friendship": ["friend", "companion", "intimate", "affection", "dear"]
        },
        "pride_and_prejudice_cleaned.txt": {
            "Pride and Prejudice": ["pride", "proud", "prejudice", "conceit", "opinion", "judgment"],
            "Class Distinctions": ["class", "gentleman", "estate", "fortune", "rank", "society", "inferior"],
            "Marriage and Courtship": ["marry", "marriage", "court", "engage", "husband", "wife", "love"],
            "Family Dynamics": ["family", "sister", "mother", "father", "daughter", "relations"],
            "Reputation": ["character", "reputation", "disgrace", "honor", "respect", "gossip"]
        },
        "sense_and_sensibility_cleaned.txt": {
            "Logic vs Emotion": ["sense", "sensibility", "feel", "emotion", "reason", "logic", "heart", "mind"],
            "Wealth and Inheritance": ["money", "fortune", "estate", "inherit", "poverty", "poor"],
            "Social Conventions": ["society", "proper", "manners", "polite", "behaviour", "decorum"],
            "Love and Heartbreak": ["love", "affection", "heart", "break", "sorrow", "attachment"],
            "Sisterhood": ["sister", "elinor", "marianne", "affection", "family"]
        }
    },
    "dickens": {
        "great_expectations_cleaned.txt": {
            "Wealth vs Poverty": ["rich", "poor", "money", "fortune", "poverty", "wealth"],
            "Ambition and Self-Improvement": ["gentleman", "learn", "improve", "expectation", "rise", "ambition"],
            "Crime and Justice": ["convict", "prison", "steal", "judge", "court", "crime", "guilt"],
            "Social Class": ["class", "common", "gentleman", "society", "low", "station"],
            "Coming of Age": ["boy", "man", "grow", "youth", "age", "years", "child"]
        },
        "oliver_twist_cleaned.txt": {
            "Institutional Cruelty": ["workhouse", "master", "board", "beat", "starve", "cruel", "parish"],
            "Poverty and the Lower Class": ["poor", "beggar", "ragged", "hungry", "street", "destitute"],
            "Crime and Morality": ["thief", "steal", "pickpocket", "crime", "guilt", "bad", "rob"],
            "Innocence": ["innocent", "boy", "child", "pure", "good", "orphan"],
            "Social Injustice": ["justice", "law", "unfair", "society", "rich", "magistrate"]
        },
        "tale_of_two_cities_cleaned.txt": {
            "Resurrection and Sacrifice": ["life", "die", "save", "blood", "recall", "sacrifice", "resurrection"],
            "Class Struggle and Revolution": ["people", "revolution", "aristocrat", "mob", "strike", "citizen", "republic"],
            "Fate and History": ["time", "fate", "destiny", "history", "age", "epoch", "season"],
            "Love and Loyalty": ["love", "dear", "heart", "loyal", "faithful", "devoted", "care"],
            "Mob Mentality": ["crowd", "mob", "people", "roar", "shout", "mass", "force", "tumult"]
        }
    }
}

os.makedirs(output_dir, exist_ok=True)

for author, author_books in books.items():
    author_out_dir = os.path.join(output_dir, author)
    os.makedirs(author_out_dir, exist_ok=True)
    
    for book_file, topics in author_books.items():
        book_name = book_file.replace("_cleaned.txt", "")
        book_out_dir = os.path.join(author_out_dir, book_name)
        os.makedirs(book_out_dir, exist_ok=True)
        
        file_path = os.path.join(base_dir, author, book_file)
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
            
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Extract paragraphs (separated by blank lines, or starting with p1, p2, etc.)
        # Since we just added 'p1 ', 'p2 ' etc., we can split by those.
        raw_paras = re.split(r'\np\d+ ', '\n' + content)
        paras = []
        for p in raw_paras:
            p = p.strip()
            # If the first string happens to start with 'p1 ', we strip it
            p = re.sub(r'^p\d+\s+', '', p)
            if p:
                paras.append(p)
                
        print(f"Loaded {len(paras)} paragraphs from {book_file}")
        
        for topic_name, keywords in topics.items():
            topic_paras = []
            
            # Find paragraphs with keywords
            for p in paras:
                lower_p = p.lower()
                if any(re.search(r'\b' + kw + r'\b', lower_p) for kw in keywords):
                    topic_paras.append(p)
                    
            # If we don't have 50, grab some random ones to fill it up
            if len(topic_paras) < 50:
                print(f"Warning: Only found {len(topic_paras)} for {topic_name}. Padding with random paras.")
                remaining = 50 - len(topic_paras)
                pool = [p for p in paras if p not in topic_paras]
                if len(pool) >= remaining:
                    topic_paras.extend(random.sample(pool, remaining))
                else:
                    topic_paras.extend(pool)
                    
            # Truncate to exactly 50 if we have more
            if len(topic_paras) > 50:
                topic_paras = topic_paras[:50]
                
            out_file = os.path.join(book_out_dir, f"{topic_name.replace(' ', '_')}.txt")
            with open(out_file, 'w', encoding='utf-8') as f:
                f.write(f"# Topic: {topic_name}\n")
                f.write(f"# Book: {book_name}\n")
                f.write(f"# Author: {author}\n\n")
                
                for i, p in enumerate(topic_paras):
                    f.write(f"--- Paragraph {i+1} ---\n")
                    f.write(p + "\n\n")
                    
            print(f"Saved 50 paragraphs to {out_file}")

print("Done generating topic texts.")
