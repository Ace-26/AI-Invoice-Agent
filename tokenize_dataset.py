import json
from datasets import Dataset, DatasetDict
from transformers import BertTokenizerFast

# Load tokenizer
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

# Helper to load JSONL
def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

# Load data
train_data = load_jsonl("processed/train.jsonl")
val_data   = load_jsonl("processed/validation.jsonl")
test_data  = load_jsonl("processed/test.jsonl")

# Create Hugging Face Datasets
train_ds = Dataset.from_list(train_data)
val_ds   = Dataset.from_list(val_data)
test_ds  = Dataset.from_list(test_data)

# Extract label mapping
unique_tags = sorted({tag for ex in train_data for tag in ex["labels"]})
tag2id = {tag: i for i, tag in enumerate(unique_tags)}
id2tag = {i: tag for tag, i in tag2id.items()}

print("Label mapping:", tag2id)

# Align labels with tokens
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding="max_length",
        max_length=256
    )
    
    labels = []
    for i, label in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        prev_word_id = None
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != prev_word_id:
                label_ids.append(tag2id[label[word_id]])
            else:
                # Subword → mark as continuation
                if label[word_id].startswith("I-"):
                    label_ids.append(tag2id[label[word_id]])
                else:
                    label_ids.append(-100)
            prev_word_id = word_id
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Map datasets
train_ds = train_ds.map(tokenize_and_align_labels, batched=True)
val_ds   = val_ds.map(tokenize_and_align_labels, batched=True)
test_ds  = test_ds.map(tokenize_and_align_labels, batched=True)

# Bundle into DatasetDict
dataset = DatasetDict({
    "train": train_ds,
    "validation": val_ds,
    "test": test_ds
})

print(dataset)
