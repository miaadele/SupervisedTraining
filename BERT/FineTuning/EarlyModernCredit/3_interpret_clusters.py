import json
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer



# Configuration

CLUSTER_FILE = Path("/content/SupervisedTraining/BERT/FineTuning/data") / "cluster_assignments.json"
TEXT_DIR = Path("/content/SupervisedTraining/BERT/GenevaBible")

TOP_N_TERMS = 12
MAX_FEATURES = 3000
MIN_DF = 2
SNIPPET_LENGTH = 200
N_SAMPLE_FILES = 5
N_SAMPLE_SNIPPETS = 2

# Custom stopwords
# I had to be pretty aggressive to get rid of a bunch of noise

custom_stopwords = {
    "haue", "hee", "hath", "bee", "shall", "unto", "thy", "thee", "ye", "thou",
    "doth", "art", "may", "might", "must", "one", "two", "three", "also", "yet",
    "upon", "thereof", "therein", "whereof", "wherein", "whose", "said", "say",
    "saith", "come", "make", "made", "man", "men", "good", "great", "time",
    "god", "king", "let", "much", "many", "well", "now", "thus", "hither",
    "thence", "hee", "shee", "themselues", "selues", "ha", "hast", "hath",
    "bee", "beene", "doe", "did", "done", "giueth", "giue", "giuen", "take",
    "taken", "tho", "although", "therefore", "whereby", "whereunto", "thereunto",
    "vpon", "vnto", "vs", "wee", "vp", "onely", "selfe", "himselfe",
    "herselfe", "themselues", "ourselues", "yourselfe", "selues",
    "neuer", "euer", "thing", "things", "called", "owne"
}


# Load cluster assignments

with open(CLUSTER_FILE, "r", encoding="utf-8") as f:
    cluster_data = json.load(f)

cluster_labels = np.array(cluster_data["cluster_labels"])
filenames = cluster_data["filenames"]
K = cluster_data["K"]


# Load documents

documents = {}
for path in sorted(TEXT_DIR.glob("*.txt")):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        documents[path.name] = f.read().strip()

# Keep only filenames that appear in the cluster file and exist on disk
matched_filenames = [fn for fn in filenames if fn in documents]
matched_texts = [documents[fn] for fn in matched_filenames]

# Align labels to matched filenames
matched_indices = [i for i, fn in enumerate(filenames) if fn in documents]
matched_labels = cluster_labels[matched_indices]

print(f"Loaded {len(matched_filenames)} clustered documents for interpretation.\n")



# Build TF-IDF on the full corpus
# Combine built-in English stopwords with custom early modern ones.

vectorizer = TfidfVectorizer(
    max_features=MAX_FEATURES,
    min_df=MIN_DF,
    stop_words="english",
     token_pattern=r"(?u)\b[a-zA-Z]{4,}\b" #SEE [1] below
)

tfidf_matrix = vectorizer.fit_transform(matched_texts)
feature_names = np.array(vectorizer.get_feature_names_out())

keep_mask = np.array([term not in custom_stopwords for term in feature_names])

tfidf_matrix = tfidf_matrix[:, keep_mask]
feature_names = feature_names[keep_mask]



# Helper function: see [2] below

def get_top_distinctive_terms(tfidf_matrix, labels, target_cluster, feature_names, top_n=10):

    in_cluster = labels == target_cluster
    out_cluster = labels != target_cluster

    if in_cluster.sum() == 0 or out_cluster.sum() == 0:
        return []

    mean_in = tfidf_matrix[in_cluster].mean(axis=0).A1
    mean_out = tfidf_matrix[out_cluster].mean(axis=0).A1
    diff = mean_in - mean_out  ## see the end of [2]

    top_indices = diff.argsort()[-top_n:][::-1]
    top_terms = [(feature_names[i], diff[i]) for i in top_indices if diff[i] > 0]

    return top_terms


# Characterize each cluster

print(f"=== Cluster Characterization (k={K}) ===\n")

for cluster_id in range(K):
    cluster_mask = matched_labels == cluster_id
    cluster_files = [matched_filenames[i] for i in range(len(matched_filenames)) if cluster_mask[i]]
    cluster_texts = [matched_texts[i] for i in range(len(matched_texts)) if cluster_mask[i]]

    print("=" * 70)
    print(f"CLUSTER {cluster_id} — {len(cluster_files)} documents")
    print("=" * 70)

    # Distinctive TF-IDF terms (cluster vs. rest)
    top_terms = get_top_distinctive_terms(
        tfidf_matrix=tfidf_matrix,
        labels=matched_labels,
        target_cluster=cluster_id,
        feature_names=feature_names,
        top_n=TOP_N_TERMS
    )

    if top_terms:
        formatted_terms = ", ".join([term for term, score in top_terms])
        print(f"Top distinctive terms: {formatted_terms}")
    else:
        print("Top distinctive terms: (none found)")

    # Sample filenames
    print(f"Sample files: {cluster_files[:N_SAMPLE_FILES]}")

    # Sample snippets
    if cluster_texts:
        print("Sample snippets:")
        for snippet_text in cluster_texts[:N_SAMPLE_SNIPPETS]:
            snippet = " ".join(snippet_text.split())[:SNIPPET_LENGTH]
            print(f"  - {snippet}...")
    else:
        print("Sample snippets: (none)")

    print()