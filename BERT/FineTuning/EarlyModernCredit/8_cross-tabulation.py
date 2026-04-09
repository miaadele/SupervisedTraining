import json
import numpy as np
from pathlib import Path
from collections import defaultdict

# Load document cluster assignments
with open(Path("data") / "cluster_assignments.json", "r") as f:
    doc_cluster_data = json.load(f)

doc_cluster_labels = doc_cluster_data["cluster_labels"]
doc_filenames = doc_cluster_data["filenames"]
K_DOC = doc_cluster_data["K"]

# Build a filename -> document cluster lookup
filename_to_doc_cluster = {}
for i, fn in enumerate(doc_filenames):
    filename_to_doc_cluster[fn] = doc_cluster_labels[i]

# --- Load credit data (k=3) ---
with open(Path("data") / "credit_embeddings.json", "r", encoding="utf-8") as f:
    credit_data = json.load(f)

credit_metadata = credit_data["metadata"]

with open(Path("data") / "credit_cluster_labels_k3.json", "r") as f:
    credit_cluster_data = json.load(f)

credit_cluster_labels = credit_cluster_data["credit_cluster_labels"]
K_CREDIT = credit_cluster_data["K_CREDIT"]

credit_sense_names = {0: "Epistemic", 1: "Social", 2: "Financial"}


# CROSS-TABULATION: Document Cluster × Credit Sense

cross_tab = defaultdict(int)

for i, meta in enumerate(credit_metadata):
    fn = meta["filename"]
    credit_sense = credit_cluster_labels[i]
    doc_cluster = filename_to_doc_cluster.get(fn, -1)
    cross_tab[(doc_cluster, credit_sense)] += 1

print("=" * 75)
print("CROSS-TABULATION: Document Cluster x Credit Sense (k=3)")
print("=" * 75)
print()

# Header
print(f"{'Doc Cluster':<15}", end="")
for cs in range(K_CREDIT):
    print(f"  {credit_sense_names[cs]:>12}", end="")
print(f"  {'Total':>8}")
print("-" * 75)

# Rows
doc_cluster_totals = defaultdict(int)
sense_totals = defaultdict(int)

for dc in range(K_DOC):
    print(f"Cluster {dc:<7}", end="")
    row_total = 0
    for cs in range(K_CREDIT):
        count = cross_tab.get((dc, cs), 0)
        row_total += count
        doc_cluster_totals[dc] += count
        sense_totals[cs] += count
        print(f"  {count:>12}", end="")
    print(f"  {row_total:>8}")

# Totals row
print("-" * 75)
print(f"{'Total':<15}", end="")
grand_total = 0
for cs in range(K_CREDIT):
    print(f"  {sense_totals[cs]:>12}", end="")
    grand_total += sense_totals[cs]
print(f"  {grand_total:>8}")

# Handle unmatched documents
unmatched = sum(cross_tab.get((-1, cs), 0) for cs in range(K_CREDIT))
if unmatched > 0:
    print(f"\n({unmatched} occurrences in documents not in the embedding set)")


# PROPORTIONAL VIEW: What % of each document cluster's "credit"
# usage falls into each sense?

print("\n\n" + "=" * 75)
print("PROPORTIONAL VIEW: % of each document cluster's 'credit' by sense")
print("=" * 75)
print()

print(f"{'Doc Cluster':<15}", end="")
for cs in range(K_CREDIT):
    print(f"  {credit_sense_names[cs]:>12}", end="")
print(f"  {'N':>8}")
print("-" * 75)

for dc in range(K_DOC):
    total = doc_cluster_totals[dc]
    if total == 0:
        continue
    print(f"Cluster {dc:<7}", end="")
    for cs in range(K_CREDIT):
        count = cross_tab.get((dc, cs), 0)
        pct = count / total * 100
        print(f"  {pct:>10.1f}%", end="")
    print(f"  {total:>8}")


# WHICH DOCUMENTS USE "CREDIT" MOST, AND IN WHICH SENSE?

print("\n\n" + "=" * 75)
print("TOP DOCUMENTS BY 'CREDIT' FREQUENCY")
print("=" * 75)
print()

# Count per document
doc_credit_counts = defaultdict(lambda: defaultdict(int))
doc_credit_total = defaultdict(int)

for i, meta in enumerate(credit_metadata):
    fn = meta["filename"]
    sense = credit_cluster_labels[i]
    doc_credit_counts[fn][sense] += 1
    doc_credit_total[fn] += 1

# Sort by total count
top_docs = sorted(doc_credit_total.items(), key=lambda x: x[1], reverse=True)[:15]

print(f"{'Filename':<30} {'Total':>6}", end="")
for cs in range(K_CREDIT):
    print(f"  {credit_sense_names[cs]:>12}", end="")
print(f"  {'Doc Cluster':>12}")
print("-" * 100)

for fn, total in top_docs:
    dc = filename_to_doc_cluster.get(fn, -1)
    print(f"{fn:<30} {total:>6}", end="")
    for cs in range(K_CREDIT):
        count = doc_credit_counts[fn].get(cs, 0)
        print(f"  {count:>12}", end="")
    print(f"  {dc:>12}")


# DOCUMENTS WITH PREDOMINANTLY ONE SENSE

print("\n\n" + "=" * 75)
print("DOCUMENTS DOMINATED BY A SINGLE SENSE (>75% of uses)")
print("=" * 75)

for cs in range(K_CREDIT):
    print(f"\n--- Predominantly {credit_sense_names[cs]} ---")
    dominated = []
    for fn, total in doc_credit_total.items():
        if total < 3:  # Skip documents with very few occurrences
            continue
        sense_count = doc_credit_counts[fn].get(cs, 0)
        pct = sense_count / total
        if pct >= 0.75:
            dominated.append((fn, total, sense_count, pct))

    dominated.sort(key=lambda x: x[1], reverse=True)
    if not dominated:
        print("  (none)")
    for fn, total, sense_count, pct in dominated[:5]:
        dc = filename_to_doc_cluster.get(fn, -1)
        print(f"  {fn:<30} {sense_count}/{total} ({pct:.0%})  [Doc Cluster {dc}]")