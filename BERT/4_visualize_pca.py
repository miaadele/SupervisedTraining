#Project the vctors into 2D and see how BERT organizes them spacially

# PCA: Principal Component Analysis
#finds the directions in the embedding space that capture the largest variance in the data
#projects the points onto those directions
#useful for a quick overview, but not designed to preserve the pairwise cos sim relationships

from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import numpy as np

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

labels = [
    "VA Co. (archaic spelling)",
    "VA Co. (modern spelling)",
    "VA Co. (paraphrase)",
    "Merchants (version A)",
    "Merchants (version B)",
    "Harvest (unrelated)",
    "Parliament (unrelated)",
]

# --- Embed ---
print("Loading model and embedding sentences...")
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(sentences)

# --- PCA ---
pca = PCA(n_components=2)
reduced = pca.fit_transform(embeddings)

print(f"Variance explained: PC1={pca.explained_variance_ratio_[0]:.1%}, "
      f"PC2={pca.explained_variance_ratio_[1]:.1%}")

# --- Plot ---
# Colors by semantic group
group_colors = ["#2196F3", "#2196F3", "#2196F3",   # VA Company (blue)
                "#4CAF50", "#4CAF50",                # Merchants (green)
                "#FF9800",                           # Harvest (orange)
                "#9C27B0"]                           # Parliament (purple)

fig, ax = plt.subplots(figsize=(10, 7))

for i in range(len(reduced)):
    ax.scatter(reduced[i, 0], reduced[i, 1],
               c=group_colors[i], s=120, zorder=5, edgecolors="black", linewidths=0.5)
    ax.annotate(labels[i], (reduced[i, 0], reduced[i, 1]),
                textcoords="offset points", xytext=(8, 8), fontsize=8)

ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
ax.set_title("BERT Sentence Embeddings - PCA Projection")

plt.tight_layout()
plt.savefig("pca_toy_corpus.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nSaved plot to pca_toy_corpus.png")
print("\n--- What to look for ---")
print("- The three VA Company sentences (blue) should cluster together.")
print("- The two merchant sentences (green) should cluster together.")
print("- Harvest (orange) and Parliament (purple) should be further away.")