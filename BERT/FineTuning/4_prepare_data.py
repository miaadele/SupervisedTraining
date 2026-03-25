# Prepare the data for training by:
#extracting the text of each verse
#grouping verses into longer chunks because verses are often too short for effective training
#tokenizing everything with BERT's tokenizer

import pandas as pd
import numpy as np
from transformers import BertTokenizer
from datasets import Dataset
from pathlib import Path

df = pd.read_excel("genevaBible.xlsx")
print(f"Total verses: {len(df):,}")

#Load tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Let's see how the tokenizer handles archaic text. 
sample = "And the earth was without forme and voide, and darkenesse was vpon the deepe."
tokens = tokenizer.tokenize(sample)
print(f"\nSample verse: {sample}")
print(f"Tokens ({len(tokens)}): {tokens}")
#BERT will try to map Early Modern spellings onto its modern WordPiece vocaulary,
#so it breaks many words into awkward subword fragments

# Group verses into 200-word chunks. 
# This gives the model more context than a single verse while keeping most chunks short enough to fit within BERT's max input size
TARGET_WORDS = 200

chunks = []
current_chunk = []
current_word_count = 0

for verse_text in df["Text"].dropna():
    verse_text = str(verse_text).strip()
    verse_word_count = len(verse_text.split())

    # If adding this verse would push us well beyond the target,
    # save the current chunk first and start a new one.
    if current_chunk and current_word_count + verse_word_count > TARGET_WORDS:
        chunks.append(" ".join(current_chunk))
        current_chunk = [verse_text]
        current_word_count = verse_word_count
    else:
        current_chunk.append(verse_text)
        current_word_count += verse_word_count

# Save the final chunk if anything is left
if current_chunk:
    chunks.append(" ".join(current_chunk))

print(f"Number of ~{TARGET_WORDS}-word chunks: {len(chunks)}")
print(f"Average chunk length (characters): {np.mean([len(c) for c in chunks]):.0f}")
print(f"Average chunk length (words): {np.mean([len(c.split()) for c in chunks]):.0f}")

print(f"\nSample chunk (first 300 chars):")
print(chunks[0][:300] + "...")

#####
# Diagnostic: how many chunks still exceed 512 tokens?
# --------------------------------------------------
# This helps us verify that the new chunking strategy is working.
# We tokenize WITHOUT truncation here just to measure true length.

token_lengths = [
    len(tokenizer(chunk, add_special_tokens=True, truncation=False)["input_ids"])
    for chunk in chunks
]

num_over_512 = sum(length > 512 for length in token_lengths)

print(f"\nAverage token length: {np.mean(token_lengths):.0f}")
print(f"Longest tokenized chunk: {np.max(token_lengths)} tokens")
print(f"Chunks over 512 tokens: {num_over_512} / {len(chunks)}")

# Tokenize all chunks
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length",
        return_special_tokens_mask=True
    )

dataset = Dataset.from_dict({"text": chunks})
print(f"\nDataset size: {len(dataset)} chunks")

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
    desc="Tokenizing"
)

print(f"Tokenized dataset: {tokenized_dataset}")
print(f"Features: {list(tokenized_dataset.features.keys())}")

# Split into train and eval (90/10). 
split = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split["train"]
eval_dataset = split["test"]

print(f"\nTraining chunks: {len(train_dataset)}")
print(f"Evaluation chunks: {len(eval_dataset)}")

# Save the datasets to disk for the next step
train_dataset.save_to_disk("data/train_dataset")
eval_dataset.save_to_disk("data/eval_dataset")

print("\nSaved datasets to data/train_dataset/ and data/eval_dataset/")
print("Ready for fine-tuning in step 5.")