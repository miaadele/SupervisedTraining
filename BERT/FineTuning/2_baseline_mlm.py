#test BERT on filling in masked words in sentences from the Geneva Bible
#save the results for later comparison

import json
import numpy as np
from pathlib import Path
from transformers import pipeline

# Load base BERT for fill-mask
fill_mask = pipeline("fill-mask", model="bert-base-uncased", device=-1) #run the model on CPU, not GPU

# Test sentences:
# Each tuple: (masked sentence, expected word)
# We will use the EXACT same tests after fine-tuning.
baseline_tests = [
    ("And God sawe the [MASK] that it was good.", "light"),
    ("In the beginning God created the [MASK] and the earth.", "heauen"),
    ("The Lord God also made the man of the [MASK] of the grounde.", "dust"),
    ("And the serpent saide vnto the [MASK], Ye shall not die.", "woman"),
    ("And God [MASK] them, saying, Bring foorth fruite and multiplie.", "blessed"),
    ("The Lord is my [MASK], I shall not want.", "shepherd"),
    ("Thou shalt not [MASK]: thou shalt not commit adulterie.", "kill"),
    ("And God called the dry land, [MASK], and he called the gathering together of the waters, Seas.", "earth"),
]

# Run predictions
print("\n=== BASELINE MLM Predictions (Before Fine-Tuning) ===\n")
baseline_results = [] #set up empty list to collect results

for masked_sent, expected in baseline_tests:
    predictions = fill_mask(masked_sent) #send the masked sentence to the BERT fill-mask pipeline and in the next two lines, we keep only the top 5 scores
    top_tokens = [p["token_str"] for p in predictions[:5]]
    top_scores = [p["score"] for p in predictions[:5]]

    hit = expected.lower() in [t.lower() for t in top_tokens]
    baseline_results.append({                         #store for later comparison
        "sentence": masked_sent,
        "expected": expected,
        "top_prediction": top_tokens[0],
        "top_score": top_scores[0],
        "in_top_5": hit,
        "all_predictions": top_tokens,
        "all_scores": top_scores,
    })

    status = "Y" if hit else "X"
    print(f"[{status}] Expected: {expected:12s} | Top: {top_tokens[0]:12s} "
          f"(score: {top_scores[0]:.3f})")
    print(f"    Top 5: {top_tokens}")
    print(f"    Sentence: {masked_sent}\n")

baseline_accuracy = sum(1 for r in baseline_results if r["in_top_5"]) / len(baseline_results)
baseline_avg_score = np.mean([r["top_score"] for r in baseline_results])
print(f"Baseline: {baseline_accuracy:.0%} of expected words in top 5")
print(f"Average top-prediction confidence: {baseline_avg_score:.3f}")

# --- Save for later comparison ---
Path("data").mkdir(exist_ok=True)
with open(Path("data") / "baseline_results.json", "w", encoding="utf-8") as f:
    json.dump(baseline_results, f, indent=2)

print("\nSaved baseline results to data/baseline_results.json")