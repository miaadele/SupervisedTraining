import json
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams["figure.dpi"] = 150

#Load credit embeddings:
with open(Path("data") / "credit_embeddings.json", "r", encoding="utf-8") as f:
    data = json.load(f)

credit_embeddings = np.array(data["embeddings"])
credit_metadata = data["metadata"]

print(f"Loaded {len(credit_embeddings)} 'credit' embeddings.")
print(f"Embedding shape: {credit_embeddings.shape}")

# Find optimal k using Slihouette score
if len(credit_embeddings) >= 10:
    k_range = range(2, min(8, len(credit_embeddings) // 2))
    sil_scores = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(credit_embeddings)
        sil_scores.append(silhouette_score(credit_embeddings, labels))
        print(f"  k={k}: silhouette={sil_scores[-1]:.3f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(list(k_range), sil_scores, "ro-")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Optimal Clusters for 'Credit' Embeddings")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("credit_cluster_selection.png", dpi=150, bbox_inches="tight")
    plt.show()  
    
    #### SEE [1] 

    best_k = list(k_range)[np.argmax(sil_scores)]
    print(f"\nBest k by silhouette: {best_k}")
else:
    best_k = 2
    print(f"Too few occurrences for silhouette analysis. Using k={best_k}.")

# Cluster with k=2 (social vs economic). 
K_CREDIT = 2
km_credit = KMeans(n_clusters=K_CREDIT, random_state=42, n_init=10)
credit_cluster_labels = km_credit.fit_predict(credit_embeddings)

print(f"\nCredit clusters (k={K_CREDIT}):")
for c in range(K_CREDIT):
    count = (credit_cluster_labels == c).sum()
    print(f"  Cluster {c}: {count} occurrences")

# PCA visualization. 
    #### SEE [2] below for the visualization
pca_credit = PCA(n_components=2)
credit_2d = pca_credit.fit_transform(credit_embeddings)

colors = ["#E74C3C", "#3498DB"]

fig, ax = plt.subplots(figsize=(10, 8))

for c in range(K_CREDIT):
    mask = credit_cluster_labels == c
    ax.scatter(
        credit_2d[mask, 0], credit_2d[mask, 1],
        c=colors[c], s=50, alpha=0.6, label=f"Cluster {c}",
        edgecolors="black", linewidths=0.3,
    )

ax.set_xlabel(f"PC1 ({pca_credit.explained_variance_ratio_[0]:.1%} variance)")
ax.set_ylabel(f"PC2 ({pca_credit.explained_variance_ratio_[1]:.1%} variance)")
ax.set_title("Contextual Embeddings of 'Credit' - Two Senses?")
ax.legend()
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig("credit_embeddings_pca.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nSaved plots to credit_cluster_selection.png and credit_embeddings_pca.png")

# --- Save cluster labels for later steps ---
output = {
    "credit_cluster_labels": credit_cluster_labels.tolist(),
    "K_CREDIT": K_CREDIT,
}
with open(Path("data") / "credit_cluster_labels.json", "w") as f:
    json.dump(output, f, indent=2)

print("Saved cluster labels to data/credit_cluster_labels.json")