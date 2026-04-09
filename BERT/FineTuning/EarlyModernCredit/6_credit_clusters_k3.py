import json
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams["figure.dpi"] = 150

# Load credit embeddings 
with open(Path("data") / "credit_embeddings.json", "r", encoding="utf-8") as f:
    data = json.load(f)

credit_embeddings = np.array(data["embeddings"])
credit_metadata = data["metadata"]

print(f"Loaded {len(credit_embeddings)} 'credit' embeddings.")

# Cluster with k=3
K_CREDIT = 3
km_credit = KMeans(n_clusters=K_CREDIT, random_state=42, n_init=10)
credit_cluster_labels = km_credit.fit_predict(credit_embeddings)

sil = silhouette_score(credit_embeddings, credit_cluster_labels)
print(f"\nClustering with k={K_CREDIT}")
print(f"Silhouette score: {sil:.3f}")
print(f"\nCluster sizes:")
for c in range(K_CREDIT):
    count = (credit_cluster_labels == c).sum()
    print(f"  Cluster {c}: {count} occurrences")

# PCA visualization 
pca_credit = PCA(n_components=2)
credit_2d = pca_credit.fit_transform(credit_embeddings)

colors = ["#E74C3C", "#3498DB", "#2ECC71"]  # Red, Blue, Green

fig, ax = plt.subplots(figsize=(10, 8))

for c in range(K_CREDIT):
    mask = credit_cluster_labels == c
    ax.scatter(
        credit_2d[mask, 0], credit_2d[mask, 1],
        c=colors[c], s=40, alpha=0.5, label=f"Cluster {c} (n={mask.sum()})",
        edgecolors="black", linewidths=0.2,
    )

ax.set_xlabel(f"PC1 ({pca_credit.explained_variance_ratio_[0]:.1%} variance)")
ax.set_ylabel(f"PC2 ({pca_credit.explained_variance_ratio_[1]:.1%} variance)")
ax.set_title("Contextual Embeddings of 'Credit' — Three Senses?")
ax.legend()
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig("credit_embeddings_pca_k3.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nSaved plot to credit_embeddings_pca_k3.png")

# Show sample sentences from each cluster 
print("\n" + "=" * 80)
print("SAMPLE SENTENCES FROM EACH CLUSTER")
print("=" * 80)

for c in range(K_CREDIT):
    indices = [i for i, label in enumerate(credit_cluster_labels) if label == c]

    print(f"\n{'=' * 60}")
    print(f"CLUSTER {c} — {len(indices)} occurrences")
    print(f"{'=' * 60}\n")

    for idx in indices[:10]:
        meta = credit_metadata[idx]
        sent = meta["sentence"]
        # Show "credit" in context
        lower = sent.lower()
        if "credit" in lower:
            pos = lower.index("credit")
            start = max(0, pos - 80)
            end = min(len(sent), pos + 80)
            snippet = sent[start:end]
        else:
            snippet = sent[:160]

        print(f"  [{meta['filename']}]")
        print(f"  ...{snippet}...")
        print()

# Save k=3 cluster labels
output = {
    "credit_cluster_labels": credit_cluster_labels.tolist(),
    "K_CREDIT": K_CREDIT,
}
with open(Path("data") / "credit_cluster_labels_k3.json", "w") as f:
    json.dump(output, f, indent=2)

print("Saved cluster labels to data/credit_cluster_labels_k3.json")