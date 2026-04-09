import json
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans

# Load data
with open(Path("data") / "credit_embeddings.json", "r", encoding="utf-8") as f:
    emb_data = json.load(f)

credit_embeddings = np.array(emb_data["embeddings"])
credit_metadata = emb_data["metadata"]

# Re-fit k=3 clustering to get centroids. SEE [1] below

K_CREDIT = 3
km = KMeans(n_clusters=K_CREDIT, random_state=42, n_init=10)
km.fit(credit_embeddings)
centroids = km.cluster_centers_
cluster_labels = km.labels_

# Compute ambiguity. SEE [2] below.

distances = np.array([
    [np.linalg.norm(emb - centroid) for centroid in centroids]
    for emb in credit_embeddings
])

# Sort distances for each point: smallest first
sorted_distances = np.sort(distances, axis=1)

# Ambiguity = difference between closest and second-closest centroid
# Small value = the point is torn between two senses
ambiguity = sorted_distances[:, 1] - sorted_distances[:, 0]

# Also identify WHICH two clusters each point is torn between
closest_two = np.argsort(distances, axis=1)[:, :2]


def show_credit_in_context(sentence, width=100):
    """Show the sentence with 'credit' centered in a window."""
    lower = sentence.lower()
    if "credit" in lower:
        pos = lower.index("credit")
        start = max(0, pos - width)
        end = min(len(sentence), pos + width)
        return "..." + sentence[start:end] + "..."
    return sentence[:200] + "..."



# MOST AMBIGUOUS: torn between two senses

most_ambiguous = np.argsort(ambiguity)[:15]

cluster_names = {0: "Epistemic", 1: "Social/Reputational", 2: "Financial/Material"}

print("=" * 80)
print("MOST AMBIGUOUS USES OF 'CREDIT' (k=3)")
print("These sentences sit at the boundary between two senses.")
print("=" * 80)

for rank, idx in enumerate(most_ambiguous, 1):
    meta = credit_metadata[idx]
    assigned = cluster_labels[idx]
    c1, c2 = closest_two[idx]
    d1, d2 = distances[idx][c1], distances[idx][c2]

    print(f"\n{rank}. Assigned to Cluster {assigned} ({cluster_names.get(assigned, '?')})")
    print(f"   Torn between: Cluster {c1} ({cluster_names.get(c1, '?')}) "
          f"and Cluster {c2} ({cluster_names.get(c2, '?')})")
    print(f"   Distances: {d1:.4f} vs {d2:.4f}  (gap: {ambiguity[idx]:.4f})")
    print(f"   [{meta['filename']}]")
    print(f"   {show_credit_in_context(meta['sentence'])}")


# BORDERLINES BETWEEN EACH PAIR OF CLUSTERS

print("\n\n" + "=" * 80)
print("BORDERLINE CASES BY CLUSTER PAIR")
print("=" * 80)

pairs = [(0, 1), (0, 2), (1, 2)]
pair_labels = [
    ("Epistemic", "Social/Reputational"),
    ("Epistemic", "Financial/Material"),
    ("Social/Reputational", "Financial/Material"),
]

for (ca, cb), (name_a, name_b) in zip(pairs, pair_labels):
    print(f"\n{'='*60}")
    print(f"BETWEEN Cluster {ca} ({name_a}) AND Cluster {cb} ({name_b})")
    print(f"{'='*60}")

    # Find points whose two closest clusters are this pair
    pair_mask = np.array([
        set(closest_two[i]) == {ca, cb} for i in range(len(credit_embeddings))
    ])

    if pair_mask.sum() == 0:
        print("  No borderline cases between these clusters.\n")
        continue

    # Among those, find the most ambiguous
    pair_indices = np.where(pair_mask)[0]
    pair_ambiguity = ambiguity[pair_indices]
    most_ambig_in_pair = pair_indices[np.argsort(pair_ambiguity)[:5]]

    for idx in most_ambig_in_pair:
        meta = credit_metadata[idx]
        assigned = cluster_labels[idx]
        print(f"\n  [Assigned: Cluster {assigned}] Gap: {ambiguity[idx]:.4f}")
        print(f"  [{meta['filename']}]")
        print(f"  {show_credit_in_context(meta['sentence'])}")


# MOST CLEAR-CUT: firmly in one sense

print("\n\n" + "=" * 80)
print("MOST CLEAR-CUT USES (for contrast)")
print("=" * 80)

most_clear = np.argsort(ambiguity)[-5:][::-1]

for rank, idx in enumerate(most_clear, 1):
    meta = credit_metadata[idx]
    assigned = cluster_labels[idx]

    print(f"\n{rank}. Cluster {assigned} ({cluster_names.get(assigned, '?')})  "
          f"Gap: {ambiguity[idx]:.4f}")
    print(f"   [{meta['filename']}]")
    print(f"   {show_credit_in_context(meta['sentence'])}")