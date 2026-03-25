from transformers import pipeline
 
# Load fill-mask pipeline
print("Loading BERT for masked language modeling...")
fill_mask = pipeline("fill-mask", model="bert-base-uncased") #the original BERT model 
 
#----------------------------------------------------------
# MLM on modern-ish sentences
print("\n=== Masked Language Modeling (Modern Sentences) ===\n")
print("BERT predicts the most likely word for [MASK]:\n")
 
modern_tests = [
    "The Virginia Company traded [MASK] across the Atlantic.",
    "The merchants of London bought and sold [MASK].",
    "The king granted a royal [MASK] to the trading company.",
]
 
for sent in modern_tests:
    print(f"Input: {sent}")
    results = fill_mask(sent)
    for r in results[:3]:
        print(f"  -> {r['token_str']:15s}  (score: {r['score']:.3f})")
    print()

# === Masked Language Modeling (Modern Sentences) ===

# BERT predicts the most likely word for [MASK]:

# Input: The Virginia Company traded [MASK] across the Atlantic.
#   -> goods            (score: 0.120)
#   -> coal             (score: 0.100)
#   -> products         (score: 0.043)

# Input: The merchants of London bought and sold [MASK].
#   -> it               (score: 0.141)
#   -> them             (score: 0.108)
#   -> slaves           (score: 0.102)

# Input: The king granted a royal [MASK] to the trading company.
#   -> charter          (score: 0.963)
#   -> warrant          (score: 0.009)
#   -> license          (score: 0.006)

#----------------------------------------------------------
# MLM on Early Modern English
historical_tests = [
    "The companie did [MASK] traffique beyond the seas.",
    "The merchants had [MASK] in silkes and spices.",
    "to all Sea-men [MASK] an enchanted den of Furies and Devils.",
]
 
for sent in historical_tests:
    print(f"Input: {sent}")
    results = fill_mask(sent)
    for r in results[:3]:
        print(f"  -> {r['token_str']:15s}  (score: {r['score']:.3f})")
    print()

# Input: The companie did [MASK] traffique beyond the seas.
#   -> not              (score: 0.814)
#   -> no               (score: 0.029)
#   -> a                (score: 0.017)

# Input: The merchants had [MASK] in silkes and spices.
#   -> traded           (score: 0.697)
#   -> brought          (score: 0.087)
#   -> trade            (score: 0.065)

# Input: to all Sea-men [MASK] an enchanted den of Furies and Devils.
#   -> is               (score: 0.290)
#   -> was              (score: 0.223)
#   -> lies             (score: 0.162)