import re
import json
from pathlib import Path

# Load documents
TEXT_DIR = Path("/content/SupervisedTraining/BERT/GenevaBible")
documents = []
for path in sorted(TEXT_DIR.glob("*.txt")):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read().strip()
    if len(text) > 50:
        documents.append({"filename": path.name, "text": text})

print(f"Loaded {len(documents)} documents.\n")

# Search for "credit" and spelling variants:

credit_pattern = re.compile(
    r"\b(credit|creditt|creditte|credite|creditts?|credits?|creddit)\b", re.IGNORECASE
)

credit_occurrences = []

for i, doc in enumerate(documents):
    # Split on sentence-ending punctuation
    sentences = re.split(r"[.;!?]+", doc["text"])

    for sent in sentences:
        sent = sent.strip()
        if len(sent.split()) < 5:
            continue

        matches = credit_pattern.findall(sent)
        if matches:
            credit_occurrences.append({
                "doc_index": i,
                "filename": doc["filename"],
                "sentence": sent,
                "match": matches[0].lower(),
            })

# Report
unique_docs = len(set(o["doc_index"] for o in credit_occurrences))
print(f"Found {len(credit_occurrences)} occurrences of 'credit'")
print(f"across {unique_docs} documents.\n")

# Show examples
print("=== Sample 'Credit' Occurrences ===\n")
for occ in credit_occurrences[:10]:
    print(f"[{occ['filename']}]")
    print(f"  ...{occ['sentence'][:200]}...")
    print()

# --- Save for next step ---
Path("data").mkdir(exist_ok=True)
with open(Path("data") / "credit_occurrences.json", "w", encoding="utf-8") as f:
    json.dump(credit_occurrences, f, ensure_ascii=False, indent=2)

print(f"Saved {len(credit_occurrences)} occurrences to data/credit_occurrences.json")