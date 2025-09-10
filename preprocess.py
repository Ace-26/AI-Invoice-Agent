import os
import pandas as pd
import json
import glob
from sklearn.model_selection import train_test_split

# Root path of dataset
DATASET_PATH = r"C:\Users\Smart\Desktop\AI Agent\IOB"

# Collect all CSVs from 4 layout folders
all_files = glob.glob(os.path.join(DATASET_PATH, "*", "*.csv"))

documents = []
doc_id = 0

for file_path in all_files:
    layout = os.path.basename(os.path.dirname(file_path))  # Layout1, Layout2, etc.
    df = pd.read_csv(file_path)

    # Ensure correct column names
    if "Text" not in df.columns or "Tag" not in df.columns:
        raise ValueError(f"CSV {file_path} does not have required columns (Text, Tag)")

    tokens = df["Text"].fillna("").astype(str).tolist()
    valid_prefixes = ("B-", "I-", "O")  # allowed NER tags
    labels = []
    for lbl in df["Tag"].fillna("O").astype(str).tolist():
        if lbl.startswith(valid_prefixes):
            labels.append(lbl)
        else:
            labels.append("O")  # fallback to 'O' if it's not a valid tag



    documents.append({
        "id": f"doc_{doc_id}",
        "tokens": tokens,
        "labels": labels,
        "layout": layout
    })
    doc_id += 1

print(f"Loaded {len(documents)} invoices.")

# Split train/val/test (80/10/10)
train_docs, test_docs = train_test_split(documents, test_size=0.2, random_state=42)
val_docs, test_docs   = train_test_split(test_docs, test_size=0.5, random_state=42)

# Save as JSONL
def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for doc in data:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

os.makedirs("processed", exist_ok=True)
save_jsonl(train_docs, "processed/train.jsonl")
save_jsonl(val_docs, "processed/validation.jsonl")
save_jsonl(test_docs, "processed/test.jsonl")
save_jsonl(documents, "processed/all.jsonl")

print("Preprocessing done. Files saved in 'processed/' folder.")
