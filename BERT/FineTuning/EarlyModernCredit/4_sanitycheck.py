import json
from pathlib import Path

with open(Path("data") / "credit_occurrences.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Check: how many sentences actually contain "credit" visibly?
visible = 0
for occ in data[:20]:
    sent = occ["sentence"].lower()
    if "credit" in sent:
        visible += 1
        print(f"FOUND: ...{sent[max(0,sent.index('credit')-50):sent.index('credit')+60]}...")
    else:
        print(f"NOT VISIBLE: {sent[:100]}...")

print(f"\n{visible}/20 contain 'credit' in the text")