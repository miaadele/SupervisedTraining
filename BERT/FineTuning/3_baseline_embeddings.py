import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# Load both models

print("Loading bert-base-uncased...")
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")
bert_model.eval()  # put model in evaluation mode

print("Loading sentence-transformer model...")
st_model = SentenceTransformer("all-MiniLM-L6-v2")

#use mean pooling to get the contextual embedding for each token and average across the token dimension

# Define a helper function for BERT-base embeddings

def embed_with_bert(sentence):
    # Tokenize the sentence for BERT ==> See [1] below: this is important!
    inputs = bert_tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    # Turn off gradient computation because we are only doing inference
    with torch.no_grad():
        outputs = bert_model(**inputs)

    # outputs.last_hidden_state has shape:
    # [batch_size, sequence_length, hidden_size]
    token_embeddings = outputs.last_hidden_state

    # Mean pooling across tokens:
    # average over sequence_length dimension. 
    #this works for single-sentence inference because there are no padding tokens
    sentence_embedding = token_embeddings.mean(dim=1)

    # Convert from PyTorch tensor to NumPy array
    return sentence_embedding.numpy()[0]

# Sentence-transformer embeddings
def embed_with_sentence_transformer(sentence):
    return st_model.encode(sentence)

# Similarity test pairs based on Geneva Bible language

related_pairs = [
    ("In the beginning God created the heauen and the earth.",
     "Thus the heauens and the earth were finished, and all the host of them."),
    ("And God saide, Let there be light: And there was light.",
     "And God called the light, Day, and the darkenes, he called Night."),
    ("The Lord is my shepherd, I shall not want.",
     "He maketh me to rest in greene pasture, and leadeth me by the still waters."),
]

unrelated_pairs = [
    ("In the beginning God created the heauen and the earth.",
     "Thou shalt not kill: thou shalt not commit adulterie."),
    ("And God saide, Let there be light: And there was light.",
     "The Lord is my shepherd, I shall not want."),
    ("And the serpent saide vnto the woman, Ye shall not die.",
     "He maketh me to rest in greene pasture, and leadeth me by the still waters."),
]

# Function to evaluate a model on the verse pairs
def evaluate_model(embed_function, model_name):
    print(f"\n=== {model_name} ===\n")

    # ----- Related pairs -----
    print("Related pairs (should be HIGH similarity):")
    related_sims = []

    for a, b in related_pairs:
        a_vec = embed_function(a)
        b_vec = embed_function(b)
        sim = cosine_similarity([a_vec], [b_vec])[0][0]
        related_sims.append(sim)
        print(f"  {sim:.3f}  |  {a[:50]}...")

    related_avg = np.mean(related_sims)
    print(f"\n  Average related similarity: {related_avg:.3f}\n")

    # ----- Unrelated pairs -----
    print("Unrelated pairs (should be LOW similarity):")
    unrelated_sims = []

    for a, b in unrelated_pairs:
        a_vec = embed_function(a)
        b_vec = embed_function(b)
        sim = cosine_similarity([a_vec], [b_vec])[0][0]
        unrelated_sims.append(sim)
        print(f"  {sim:.3f}  |  {a[:50]}...")

    unrelated_avg = np.mean(unrelated_sims)
    print(f"\n  Average unrelated similarity: {unrelated_avg:.3f}")

    # Gap = how much higher the related average is than the unrelated average
    gap = related_avg - unrelated_avg
    print(f"  Gap (related - unrelated): {gap:.3f}")

    return {
        "related_scores": related_sims,
        "unrelated_scores": unrelated_sims,
        "related_avg": related_avg,
        "unrelated_avg": unrelated_avg,
        "gap": gap
    }


# Run both models
bert_results = evaluate_model(embed_with_bert, "Raw BERT-base (mean pooled)")
st_results = evaluate_model(embed_with_sentence_transformer, "Sentence-Transformer (all-MiniLM-L6-v2)")


# Final comparison summary
print("\n=== Summary Comparison ===\n")
print(f"BERT-base gap:            {bert_results['gap']:.3f}")
print(f"Sentence-transformer gap: {st_results['gap']:.3f}")

if st_results["gap"] > bert_results["gap"]:
    print("\nThe sentence-transformer model separates related and unrelated verses more clearly.")
else:
    print("\nIn this test, raw BERT performed as well as or better than the sentence-transformer model.")

# === Raw BERT-base (mean pooled) ===

# Related pairs (should be HIGH similarity):
#   0.844  |  In the beginning God created the heauen and the ea...
#   0.840  |  And God saide, Let there be light: And there was l...
#   0.790  |  The Lord is my shepherd, I shall not want....

#   Average related similarity: 0.825

# Unrelated pairs (should be LOW similarity):
#   0.615  |  In the beginning God created the heauen and the ea...
#   0.758  |  And God saide, Let there be light: And there was l...
#   0.835  |  And the serpent saide vnto the woman, Ye shall not...

#   Average unrelated similarity: 0.736
#   Gap (related - unrelated): 0.089

# === Sentence-Transformer (all-MiniLM-L6-v2) ===

# Related pairs (should be HIGH similarity):
#   0.704  |  In the beginning God created the heauen and the ea...
#   0.653  |  And God saide, Let there be light: And there was l...
#   0.424  |  The Lord is my shepherd, I shall not want....

#   Average related similarity: 0.594

# Unrelated pairs (should be LOW similarity):
#   0.141  |  In the beginning God created the heauen and the ea...
#   0.349  |  And God saide, Let there be light: And there was l...
#   0.245  |  And the serpent saide vnto the woman, Ye shall not...

#   Average unrelated similarity: 0.245
#   Gap (related - unrelated): 0.349

# === Summary Comparison ===

# BERT-base gap:            0.089
# Sentence-transformer gap: 0.349

# The sentence-transformer model separates related and unrelated verses more clearly.