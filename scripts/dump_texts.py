import os

import pandas as pd

# 1. Load your test split
df_test = pd.read_csv("data/processed/test.csv", index_col=0)

# 2. Pick the column with the raw trial text
#    Change this to whatever your real column name is.
texts = df_test["combined_text"].fillna("").tolist()

# 3. Write out newline‐delimited file
os.makedirs("data", exist_ok=True)
with open("data/test_texts.txt", "w", encoding="utf-8") as f:
    for t in texts:
        # strip out newlines inside each entry so it stays one‐line per record
        f.write(t.replace("\n", " ") + "\n")

print(f"✅ Wrote {len(texts)} lines to data/test_texts.txt")
