import torch
import json
import numpy as np
from pathlib import Path
from transformers import BertTokenizer, BertModel
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams["figure.dpi"] = 150

# Load credit metadata 
with open(Path("data") / "credit_embeddings.json", "r", encoding="utf-8") as f:
    data = json.load(f)

credit_metadata = data["metadata"]
ft_credit_embeddings = np.array(data["embeddings"])

print(f"Loaded {len(ft_credit_embeddings)} fine-tuned BERT embeddings for 'credit'.")

# Load base BERT and re-extract embeddings
print("Loading base BERT...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
base_model = BertModel.from_pretrained("bert-base-uncased")
base_model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
base_model.to(device)


def get_word_embedding(sentence, target_word, tokenizer, model):
    """Extract contextual embedding for a specific word."""
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].cpu())

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
    return hidden_states[target_indices].mean(dim=0).numpy()


# Extract with base BERT
print("Extracting 'credit' embeddings with base BERT...\n")

base_credit_embeddings = []
base_indices = []

for i, meta in enumerate(credit_metadata):
    emb = get_word_embedding(meta["sentence"], "credit", tokenizer, base_model)
    if emb is not None:
        base_credit_embeddings.append(emb)
        base_indices.append(i)

    if (i + 1) % 100 == 0:
        print(f"  Processed {i + 1}/{len(credit_metadata)}...")

base_credit_embeddings = np.array(base_credit_embeddings)
print(f"\nExtracted {len(base_credit_embeddings)} base BERT embeddings.")


# COMPARE CLUSTERING QUALITY (k=3)

K = 3
sense_names = {0: "Epistemic", 1: "Social", 2: "Financial"}

if len(base_credit_embeddings) >= 6 and len(ft_credit_embeddings) >= 6:
    # Use only the overlapping subset
    ft_subset = ft_credit_embeddings[base_indices]

    km_base = KMeans(n_clusters=K, random_state=42, n_init=10)
    km_ft = KMeans(n_clusters=K, random_state=42, n_init=10)

    base_labels = km_base.fit_predict(base_credit_embeddings)
    ft_labels = km_ft.fit_predict(ft_subset)

    base_sil = silhouette_score(base_credit_embeddings, base_labels)
    ft_sil = silhouette_score(ft_subset, ft_labels)

    print(f"\n{'='*60}")
    print(f"CLUSTERING QUALITY COMPARISON (k={K})")
    print(f"{'='*60}")
    print(f"\nSilhouette score (base BERT):      {base_sil:.4f}")
    print(f"Silhouette score (fine-tuned BERT): {ft_sil:.4f}")

    if ft_sil > base_sil:
        improvement = ((ft_sil - base_sil) / base_sil) * 100
        print(f"\nFine-tuned model produces BETTER-separated clusters.")
        print(f"Improvement: {improvement:.1f}%")
    elif ft_sil < base_sil:
        print(f"\nBase model produces better-separated clusters.")
    else:
        print(f"\nBoth models produce similarly separated clusters.")

    #Cluster size comparison:
    
    print(f"\n{'='*60}")
    print(f"CLUSTER SIZE COMPARISON")
    print(f"{'='*60}")
    print(f"\n{'Cluster':<20} {'Base BERT':>12} {'Fine-tuned':>12}")
    print("-" * 44)
    for c in range(K):
        base_count = (base_labels == c).sum()
        ft_count = (ft_labels == c).sum()
        print(f"Cluster {c:<12} {base_count:>12} {ft_count:>12}")

    
    # SIDE-BY-SIDE VISUALIZATION
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    colors = ["#E74C3C", "#3498DB", "#2ECC71"]

    # Base BERT
    pca_base = PCA(n_components=2)
    base_2d = pca_base.fit_transform(base_credit_embeddings)

    for c in range(K):
        mask = base_labels == c
        ax1.scatter(
            base_2d[mask, 0], base_2d[mask, 1],
            c=colors[c], s=30, alpha=0.5,
            label=f"Cluster {c} (n={mask.sum()})",
            edgecolors="black", linewidths=0.2,
        )
    ax1.set_title(f"Base BERT — 'Credit' (k={K}, sil={base_sil:.3f})")
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.2)

    # Fine-tuned BERT
    pca_ft = PCA(n_components=2)
    ft_2d = pca_ft.fit_transform(ft_subset)

    for c in range(K):
        mask = ft_labels == c
        ax2.scatter(
            ft_2d[mask, 0], ft_2d[mask, 1],
            c=colors[c], s=30, alpha=0.5,
            label=f"Cluster {c} (n={mask.sum()})",
            edgecolors="black", linewidths=0.2,
        )
    ax2.set_title(f"Fine-Tuned BERT — 'Credit' (k={K}, sil={ft_sil:.3f})")
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig("credit_base_vs_finetuned_k3.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("\nSaved plot to credit_base_vs_finetuned_k3.png")

    ######################
    # AMBIGUITY COMPARISON: 
    # Compare how ambiguous the clustering is under each model

    base_distances = np.array([
        [np.linalg.norm(emb - c) for c in km_base.cluster_centers_]
        for emb in base_credit_embeddings
    ])
    ft_distances = np.array([
        [np.linalg.norm(emb - c) for c in km_ft.cluster_centers_]
        for emb in ft_subset
    ])

    base_sorted = np.sort(base_distances, axis=1)
    ft_sorted = np.sort(ft_distances, axis=1)

    base_ambiguity = base_sorted[:, 1] - base_sorted[:, 0]
    ft_ambiguity = ft_sorted[:, 1] - ft_sorted[:, 0]

    print(f"\n{'='*60}")
    print(f"AMBIGUITY COMPARISON")
    print(f"{'='*60}")
    print(f"\n{'Metric':<35} {'Base BERT':>12} {'Fine-tuned':>12}")
    print("-" * 59)
    print(f"{'Mean ambiguity gap':<35} {base_ambiguity.mean():>12.4f} {ft_ambiguity.mean():>12.4f}")
    print(f"{'Median ambiguity gap':<35} {np.median(base_ambiguity):>12.4f} {np.median(ft_ambiguity):>12.4f}")
    print(f"{'% highly ambiguous (gap < 0.1)':<35} "
          f"{(base_ambiguity < 0.1).mean()*100:>11.1f}% "
          f"{(ft_ambiguity < 0.1).mean()*100:>11.1f}%")
    print(f"{'% clear-cut (gap > 1.0)':<35} "
          f"{(base_ambiguity > 1.0).mean()*100:>11.1f}% "
          f"{(ft_ambiguity > 1.0).mean()*100:>11.1f}%")

    # --- Ambiguity distribution plot ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(base_ambiguity, bins=50, alpha=0.5, label=f"Base BERT (mean={base_ambiguity.mean():.3f})",
            color="gray", edgecolor="black", linewidth=0.3)
    ax.hist(ft_ambiguity, bins=50, alpha=0.5, label=f"Fine-tuned (mean={ft_ambiguity.mean():.3f})",
            color="steelblue", edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Ambiguity Gap (distance to nearest - distance to second nearest centroid)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Ambiguity: Base BERT vs Fine-Tuned")
    ax.legend()
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig("ambiguity_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("\nSaved plot to ambiguity_comparison.png")

else:
    print("Not enough embeddings for comparison.")