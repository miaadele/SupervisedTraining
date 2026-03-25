# MDS: Multidiemsnional scaling
#starts from a distance matrix and places points in 2D so that
#the distances between points are preserved as closely as possible

#sentences that are semantically similar appear closer together

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances
from sklearn.manifold import MDS
import matplotlib.pyplot as plt

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

group_colors = [
    "#2196F3", "#2196F3", "#2196F3",
    "#4CAF50", "#4CAF50",
    "#FF9800",
    "#9C27B0"
]

print("Loading model and embedding sentences...")
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(sentences, normalize_embeddings=True)

# Cosine distance matrix
dist_matrix = cosine_distances(embeddings)

# 2D projection that tries to preserve pairwise distances
mds = MDS(
    n_components=2,
    dissimilarity="precomputed",
    random_state=42,
    metric=True
)
reduced = mds.fit_transform(dist_matrix)

fig, ax = plt.subplots(figsize=(10, 7))

for i in range(len(reduced)):
    ax.scatter(
        reduced[i, 0], reduced[i, 1],
        c=group_colors[i], s=120,
        edgecolors="black", linewidths=0.5, zorder=5
    )
    ax.annotate(
        labels[i],
        (reduced[i, 0], reduced[i, 1]),
        textcoords="offset points",
        xytext=(8, 8),
        fontsize=8
    )

ax.set_xlabel("MDS dimension 1")
ax.set_ylabel("MDS dimension 2")
ax.set_title("Sentence Embeddings (all-MiniLM-L6-v2) projected with MDS")

plt.tight_layout()
plt.savefig("mds_toy_corpus.png", dpi=150, bbox_inches="tight")
plt.show()