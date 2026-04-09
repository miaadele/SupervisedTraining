import torch
import numpy as np
from pathlib import Path
from transformers import BertTokenizer, BertModel

#Load the fine-tuned model
MODEL_PATH = "geneva-bert"

try:
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    model = BertModel.from_pretrained(MODEL_PATH)
    print(f"Loaded fine-tuned model from {MODEL_PATH}")
except Exception:
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    print("Fine-tuned model not found. Using base BERT instead.")
    print("(Results will be less accurate for Early Modern English.)")

model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Using device: {device}")

# Load texts
TEXT_DIR = Path("texts")

documents = []
for path in sorted(TEXT_DIR.glob("*.txt")):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read().strip()
    if len(text) > 50:
        documents.append({
            "filename": path.name,  # record the filename
            "text": text,
            "word_count": len(text.split()),
        })

print(f"\nLoaded {len(documents)} documents from {TEXT_DIR}")
print(f"Total words: {sum(d['word_count'] for d in documents):,}")
print(f"Average document length: {np.mean([d['word_count'] for d in documents]):.0f} words")
print(f"Shortest: {min(d['word_count'] for d in documents)} words")
print(f"Longest: {max(d['word_count'] for d in documents)} words")

# --- Preview ---
print("\n=== Sample Documents ===\n")
for doc in documents[:3]:
    print(f"[{doc['filename']}] ({doc['word_count']} words)")
    print(f"  {doc['text'][:200]}...")
    print()

print(f"Setup complete. {len(documents)} documents ready for analysis.")