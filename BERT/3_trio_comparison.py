from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- The Virginia Company trio ---
trio = [
    "The Virginia Companie did advance traffique beyond the seas.",
    "The Virginia Company did advance traffic beyond the seas.",
    "The Virginia Company promoted overseas trade."
]
trio_labels = ["Archaic spelling", "Modern spelling", "Paraphrase"]

# --- TF-IDF ---
vectorizer = TfidfVectorizer()
tfidf_trio = vectorizer.fit_transform(trio)
tfidf_trio_sim = cosine_similarity(tfidf_trio)

# --- BERT ---
print("Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2") 
bert_trio = model.encode(trio)
bert_trio_sim = cosine_similarity(bert_trio)
#these embeddings move beyond individual words and instead represent the overall meaning of a sentence
#the model recognizes similarity even when the wording changes

# --- Compare ---
print("\n=== Virginia Company Trio ===\n")

print("TF-IDF Similarity:")
for i in range(3):
    for j in range(3):
        print(f"  {trio_labels[i]:20s} <-> {trio_labels[j]:20s} : {tfidf_trio_sim[i][j]:.3f}")
    print()

print("BERT Similarity:")
for i in range(3):
    for j in range(3):
        print(f"  {trio_labels[i]:20s} <-> {trio_labels[j]:20s} : {bert_trio_sim[i][j]:.3f}")
    print()