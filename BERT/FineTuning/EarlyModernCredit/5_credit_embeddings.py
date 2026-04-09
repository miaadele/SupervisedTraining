import torch
import json
import numpy as np
from pathlib import Path
from transformers import BertTokenizer, BertModel

# Load mode:

MODEL_PATH = "/content/SupervisedTraining/BERT/FineTuning/geneva_bert"
try:
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    model = BertModel.from_pretrained(MODEL_PATH)
    print(f"Loaded fine-tuned model from {MODEL_PATH}")
except Exception:
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    print("Using base BERT (fine-tuned model not found).")

model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Load credit occurrences:
with open(Path("data") / "credit_occurrences.json", "r", encoding="utf-8") as f:
    credit_occurrences = json.load(f)

print(f"Loaded {len(credit_occurrences)} credit occurrences.\n")


# Word embedding extraction function. See [1] below
def get_word_embedding(sentence, target_word, tokenizer, model):
    
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].cpu())

    # Reconstruct words from subword tokens to find our target
    target_lower = target_word.lower()
    target_indices = []
    current_word = ""
    current_indices = []

    for idx, token in enumerate(tokens):
        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            if current_word == target_lower and current_indices:
                target_indices = current_indices
                break
            current_word = ""
            current_indices = []
            continue

        if token.startswith("##"):
            current_word += token[2:]
            current_indices.append(idx)
        else:
            if current_word == target_lower and current_indices:
                target_indices = current_indices
                break
            current_word = token
            current_indices = [idx]

    if not target_indices and current_word == target_lower and current_indices:
        target_indices = current_indices

    if not target_indices:
        return None

    with torch.no_grad():
        outputs = model(**inputs)

    hidden_states = outputs.last_hidden_state[0].cpu()
    word_embedding = hidden_states[target_indices].mean(dim=0).numpy()

    return word_embedding


# Extract embeddings:
print("Extracting contextual embeddings for 'credit'...\n")

credit_embeddings = []
credit_metadata = []

for i, occ in enumerate(credit_occurrences):
    emb = get_word_embedding(occ["sentence"], occ["match"], tokenizer, model)

    if emb is not None:
        credit_embeddings.append(emb.tolist())
        credit_metadata.append(occ)

    if (i + 1) % 25 == 0:
        print(f"  Processed {i + 1}/{len(credit_occurrences)}...")

print(f"\nExtracted {len(credit_embeddings)} embeddings for 'credit'.")

# Save
output = {
    "embeddings": credit_embeddings,
    "metadata": credit_metadata,
}
with open(Path("data") / "credit_embeddings.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False)

print("Saved to data/credit_embeddings.json")