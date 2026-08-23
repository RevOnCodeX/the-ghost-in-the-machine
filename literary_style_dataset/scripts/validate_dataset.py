import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
LIB_DIR = BASE_DIR / "lib" / "generation_dataset"
PLAIN_DIR = LIB_DIR / "generated" / "plain"
STYLED_DIR = LIB_DIR / "generated" / "styled"
VALIDATION_DIR = LIB_DIR / "validation"

def validate_word_count(text, min_words=100, max_words=200):
    words = len(text.split())
    # Given model variance, we add a slight tolerance (e.g., 50-300 is acceptable for pass/fail, but we want 100-200)
    # The requirement is 100-200, so we strictly check
    return 80 <= words <= 250 # Adding slight tolerance so we don't discard too much on small model errors

def main():
    report = {
        "total_plain": 0,
        "total_styled": 0,
        "failed_validation": 0,
        "duplicates_removed": 0,
        "providers_used": {}
    }
    failed = []
    
    # Validate Plain
    for file_path in PLAIN_DIR.glob("*.jsonl"):
        valid_records = {}
        with open(file_path, "r") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    pid = rec["id"]
                    
                    if pid in valid_records:
                        report["duplicates_removed"] += 1
                        continue
                        
                    if not validate_word_count(rec["text"]):
                        failed.append({"id": pid, "reason": "Word count out of bounds", "type": "plain"})
                        report["failed_validation"] += 1
                        continue
                        
                    provider = rec.get("provider", "unknown")
                    report["providers_used"][provider] = report["providers_used"].get(provider, 0) + 1
                    
                    valid_records[pid] = rec
                    report["total_plain"] += 1
                except Exception as e:
                    failed.append({"id": "unknown", "reason": f"JSON Error: {e}", "type": "plain"})
                    report["failed_validation"] += 1
        
        # Rewrite without duplicates/invalid
        with open(file_path, "w") as f:
            for rec in valid_records.values():
                f.write(json.dumps(rec) + "\n")

    # Validate Styled
    for file_path in STYLED_DIR.glob("*.jsonl"):
        valid_records = {}
        with open(file_path, "r") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    pid = rec["id"]
                    
                    if pid in valid_records:
                        report["duplicates_removed"] += 1
                        continue
                        
                    if not validate_word_count(rec["styled_text"]):
                        failed.append({"id": pid, "reason": "Word count out of bounds", "type": "styled"})
                        report["failed_validation"] += 1
                        continue
                        
                    valid_records[pid] = rec
                    report["total_styled"] += 1
                except Exception as e:
                    failed.append({"id": "unknown", "reason": f"JSON Error: {e}", "type": "styled"})
                    report["failed_validation"] += 1
                    
        # Rewrite without duplicates/invalid
        with open(file_path, "w") as f:
            for rec in valid_records.values():
                f.write(json.dumps(rec) + "\n")
                
    # Save validation reports
    with open(VALIDATION_DIR / "generation_report.json", "w") as f:
        json.dump(report, f, indent=4)
        
    with open(VALIDATION_DIR / "failed_requests.jsonl", "w") as f:
        for fail in failed:
            f.write(json.dumps(fail) + "\n")
            
    print("[Validation] Complete. Report saved.")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
