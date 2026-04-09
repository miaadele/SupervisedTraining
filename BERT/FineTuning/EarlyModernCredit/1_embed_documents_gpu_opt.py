import json
from pathlib import Path

import numpy as np
import torch
from transformers import BertModel, BertTokenizer

MODEL_PATH = "/content/SupervisedTraining/BERT/FineTuning/geneva_bert"
FALLBACK_MODEL = "bert-base-uncased"
TEXT_DIR = Path("/content/SupervisedTraining/BERT/GenevaBible")
OUTPUT_DIR = Path("/content/SupervisedTraining/BERT/FineTuning/data")
OUTPUT_FILE = OUTPUT_DIR / "doc_embeddings.json"

MAX_LENGTH = 512
STRIDE = 256
MIN_CHARS = 50
BATCH_SIZE = 16   

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_model_and_tokenizer(model_path, fallback_model):
    try:
        tokenizer = BertTokenizer.from_pretrained(model_path)
        model = BertModel.from_pretrained(model_path)
        print(f"Loaded fine-tuned model from '{model_path}'")
    except Exception as e:
        print(f"Could not load '{model_path}': {e}")
        print(f"Falling back to '{fallback_model}'")
        tokenizer = BertTokenizer.from_pretrained(fallback_model)
        model = BertModel.from_pretrained(fallback_model)

    model.to(device)
    model.eval()
    return tokenizer, model

def load_documents(text_dir, min_chars=50):
    documents = []
    for path in sorted(text_dir.glob("*.txt")):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read().strip()
        if len(text) >= min_chars:
            documents.append({
                "filename": path.name,
                "text": text
            })
    return documents

def mean_pool_without_special_tokens(last_hidden_state, attention_mask):

    mask = attention_mask.clone()
    mask[:, 0] = 0

    lengths = attention_mask.sum(dim=1)  
    for i, seq_len in enumerate(lengths):
        if seq_len > 1:
            mask[i, seq_len - 1] = 0

    mask = mask.unsqueeze(-1)  

    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)
    return summed / counts

def embed_document_batched(text, tokenizer, model, max_length=512, stride=256, batch_size=16):
    
    enc = tokenizer(
        text,
        return_overflowing_tokens=True,
        truncation=True,
        max_length=max_length,
        stride=stride,
        padding=False,
        return_tensors=None
    )

    input_id_chunks = enc["input_ids"]
    attention_mask_chunks = enc["attention_mask"]

    if len(input_id_chunks) == 0:
        return None

    chunk_embeddings = []
    chunk_weights = []

    for start in range(0, len(input_id_chunks), batch_size):
        batch_input_ids = input_id_chunks[start:start + batch_size]
        batch_attention_masks = attention_mask_chunks[start:start + batch_size]

        batch = tokenizer.pad(
            {
                "input_ids": batch_input_ids,
                "attention_mask": batch_attention_masks
            },
            padding=True,
            return_tensors="pt"
        )

        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            if device.type == "cuda":
                with torch.autocast(device_type="cuda"):
                    outputs = model(**batch)
            else:
                outputs = model(**batch)

        pooled = mean_pool_without_special_tokens(
            outputs.last_hidden_state,
            batch["attention_mask"]
        )  

        pooled = pooled.cpu().numpy()
        weights = batch["attention_mask"].sum(dim=1).cpu().numpy() - 2  

        chunk_embeddings.append(pooled)
        chunk_weights.append(weights)

    chunk_embeddings = np.vstack(chunk_embeddings)
    chunk_weights = np.concatenate(chunk_weights)

    document_embedding = np.average(chunk_embeddings, axis=0, weights=chunk_weights)
    return document_embedding

def main():
    tokenizer, model = load_model_and_tokenizer(MODEL_PATH, FALLBACK_MODEL)
    documents = load_documents(TEXT_DIR, min_chars=MIN_CHARS)

    print(f"Loaded {len(documents)} documents.")
    print("Embedding documents...\n")

    doc_embeddings = []
    doc_filenames = []

    for i, doc in enumerate(documents, start=1):
        embedding = embed_document_batched(
            doc["text"],
            tokenizer,
            model,
            max_length=MAX_LENGTH,
            stride=STRIDE,
            batch_size=BATCH_SIZE
        )

        if embedding is not None:
            doc_embeddings.append(embedding.tolist())
            doc_filenames.append(doc["filename"])

        if i % 25 == 0 or i == len(documents):
            print(f"Processed {i}/{len(documents)} documents...")

    OUTPUT_DIR.mkdir(exist_ok=True)

    output = {
        "filenames": doc_filenames,
        "embeddings": doc_embeddings
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f)

    print(f"\nSaved embeddings to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()