import json
from transformers import AutoTokenizer

DATA_PATH = "data/byt5_dataset.jsonl"
MODEL_NAME = "google/byt5-small"

print(f"Loading dataset from: {DATA_PATH}")
raw_dataset = []
with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        raw_dataset.append(json.loads(line))

# Audit for empty targets and targets equal to input
empty_targets = 0
same_as_input = 0
for ex in raw_dataset:
    if not ex['target'].strip():
        empty_targets += 1
    if ex['input'].strip() == ex['target'].strip():
        same_as_input += 1

print(f"Empty targets: {empty_targets}")
print(f"Targets equal to input: {same_as_input}")
print(f"Total examples: {len(raw_dataset)}")

# Load tokenizer
print(f"Loading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Show a sample tokenized example and its decoded input/label
if len(raw_dataset) > 0:
    example = raw_dataset[0]
    print("\n=== DEBUG: First dataset example ===")
    print("Input:", example['input'])
    print("Target:", example['target'])
    # Tokenize
    model_inputs = tokenizer(
        example["input"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            example["target"],
            truncation=True,
            padding="max_length",
            max_length=128,
        )
    print("Tokenized input_ids:", model_inputs['input_ids'])
    print("Tokenized labels:", labels['input_ids'])
    # Decode
    decoded_input = tokenizer.decode(model_inputs['input_ids'], skip_special_tokens=True)
    decoded_label = tokenizer.decode([id for id in labels['input_ids'] if id != -100], skip_special_tokens=True)
    print("Decoded input:", decoded_input)
    print("Decoded label:", decoded_label)
    print("====================================\n")
else:
    print("Dataset is empty!") 