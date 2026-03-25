#semantic search:
#given a query, find the most similar sentence in a corpus
#represent both the query and the docs as vectors in an embedding space
#retrieve the documents whose vectors are closes to the query vector using cosine similarity

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- Same toy corpus ---
sentences = [
    "The Virginia Companie did advance traffique beyond the seas.",
    "The Virginia Company did advance traffic beyond the seas.",
    "The Virginia Company promoted overseas trade.",
    "The merchants of London traded in silks and spices.",
    "London merchants engaged in the silk and spice trade.",
    "The harvest was plentiful this year.",
    "Parliament assembled to debate the new taxation.",
]

# --- Embed the corpus ---
print("Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(sentences)

# --- Query ---
query = "The Virginia Company expanded trade overseas."
query_embedding = model.encode([query])
similarities = cosine_similarity(query_embedding, embeddings)[0]

print(f'Query: "{query}"\n')
print("Ranked results:")
print("-" * 70)

ranked = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)
for rank, (idx, sim) in enumerate(ranked, 1):
    print(f"  {rank}. [{sim:.3f}] {sentences[idx]}")

# Ranked results:
# ----------------------------------------------------------------------
#   1. [0.926] The Virginia Company promoted overseas trade.
#   2. [0.653] The Virginia Company did advance traffic beyond the seas.
#   3. [0.594] The Virginia Companie did advance traffique beyond the seas.
#   4. [0.367] The merchants of London traded in silks and spices.
#   5. [0.364] London merchants engaged in the silk and spice trade.
#   6. [0.120] The harvest was plentiful this year.
#   7. [0.077] Parliament assembled to debate the new taxation.